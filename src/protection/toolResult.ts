import { ToolMessage } from '@langchain/core/messages';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
  ProviderTextProtectionErrorCode,
} from './providerText';
import type { ToolExecuteResult } from '@/types/tools';
import {
  ProviderTextAttempt,
  readProviderTextResult,
  ProviderTextProtectionError,
  validateProviderTextProtection,
} from './providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { cloneToolMessageWithContent } from '@/utils/toolContent';
import { StreamLimitExceededError } from '@/llm/streamLimits';

export const TOOL_RESULT_PROTECTION_VERSION = 1;

export type ToolResultProtectionResult = ProviderTextProtectionResult;

/**
 * Host inspection of the text returned by selected tools. Default off.
 *
 * The SDK calls `inspect` on a selected tool's text before anything else reads
 * it: hooks, completion events, the output reference registry, checkpoints and
 * the model all receive the inspected text. A failed inspection ends the run
 * with a {@link ToolResultProtectionError}.
 *
 * Trust boundary: the tool's text is untrusted data; code running in the host
 * process is trusted. Tools, hooks, event handlers, callbacks and checkpoint
 * savers are not defended against, text they author is not inspected, and
 * history the host supplies is the host's to have inspected. Tracing callbacks
 * attached to a direct tool observe its raw return; redact those with the
 * tracing configuration.
 *
 * Selected tools must return plain text. A structured result, an artifact or a
 * `Command` fails as `unsupported`. Selected tools are not offered to
 * programmatic tool calling, are not started eagerly, and cannot be client
 * delegated. Unselected tools are untouched.
 */
export interface ToolResultProtection {
  readonly version: 1;
  readonly toolNames: readonly string[];
  readonly timeoutMs: number;
  readonly maxAttemptBytes: number;
  /** Shared across all inspections using this policy, including child graphs. */
  readonly maxBufferedBytes: number;
  readonly classify: (content: string) => 'prose' | 'unsupported';
  readonly inspect: (input: {
    readonly version: 1;
    readonly content: string;
    readonly toolName: string;
    readonly toolCallId: string;
    readonly target: {
      readonly source: 'tool_argument';
      readonly field: 'output';
      readonly provenance: 'tool';
      readonly outcome: ToolResultOutcome;
    };
    readonly signal: AbortSignal;
  }) => ToolResultProtectionResult | Promise<ToolResultProtectionResult>;
}

export class ToolResultProtectionError extends ProviderTextProtectionError {
  constructor(code: ProviderTextProtectionErrorCode) {
    super(code);
    this.name = 'ToolResultProtectionError';
    this.message = `Tool result protection: ${this.code}`;
  }
}

export type ToolResultOutcome = 'success' | 'error';

export interface ToolResultCall {
  readonly name: string;
  readonly id: string;
}

/** A validated policy. One per policy object, so every graph of a run shares its byte budget. */
export interface ToolResultGate {
  readonly policy: ToolResultProtection;
  readonly toolNames: ReadonlySet<string>;
  readonly budget: ProviderTextProtection;
}

const gates = new WeakMap<ToolResultProtection, ToolResultGate>();

/** Budget accounting only: tool text is never released through the provider-text path. */
const rejectProviderRelease = (): ProviderTextProtectionResult => ({
  version: 1,
  ok: false,
  error: { code: 'incompatible' },
});

/** Run-level safety aborts keep their identity; nothing else carries text out of an inspection. */
function toToolResultError(error: unknown): Error {
  if (
    error instanceof ToolResultProtectionError ||
    error instanceof StreamLimitExceededError ||
    error instanceof PreparedSubagentError
  ) {
    return error;
  }
  return new ToolResultProtectionError(
    error instanceof ProviderTextProtectionError ? error.code : 'unavailable'
  );
}

function hasValidToolNames(toolNames: readonly string[]): boolean {
  return (
    Array.isArray(toolNames) &&
    toolNames.length > 0 &&
    toolNames.every((name) => typeof name === 'string' && name.trim() !== '')
  );
}

export function createToolResultGate(
  policy: ToolResultProtection
): ToolResultGate {
  const existing = gates.get(policy);
  if (existing != null) {
    return existing;
  }
  if (typeof policy.inspect !== 'function') {
    throw new ToolResultProtectionError('unavailable');
  }
  if (!hasValidToolNames(policy.toolNames)) {
    throw new ToolResultProtectionError('incompatible');
  }
  const budget: ProviderTextProtection = {
    version: policy.version,
    timeoutMs: policy.timeoutMs,
    maxAttemptBytes: policy.maxAttemptBytes,
    maxBufferedBytes: policy.maxBufferedBytes,
    classify: policy.classify,
    inspect: rejectProviderRelease,
  };
  try {
    validateProviderTextProtection(budget);
  } catch (error) {
    throw toToolResultError(error);
  }
  const gate: ToolResultGate = {
    policy,
    toolNames: new Set(policy.toolNames),
    budget,
  };
  gates.set(policy, gate);
  return gate;
}

/** Returns the inspected text, or throws; the raw text and inspector errors never leave. */
export async function inspectToolText(
  gate: ToolResultGate,
  call: ToolResultCall,
  content: string,
  outcome: ToolResultOutcome,
  signal?: AbortSignal
): Promise<string> {
  if (content === '') {
    return content;
  }
  let attempt: ProviderTextAttempt | undefined;
  try {
    attempt = new ProviderTextAttempt(gate.budget, signal);
    const inspection = attempt;
    inspection.append(content);
    if (gate.policy.classify(content) !== 'prose') {
      throw new ToolResultProtectionError('unsupported');
    }
    const result = await inspection.wait(
      (async (): Promise<ToolResultProtectionResult> =>
        gate.policy.inspect({
          version: 1,
          content,
          toolName: call.name,
          toolCallId: call.id,
          target: {
            source: 'tool_argument',
            field: 'output',
            provenance: 'tool',
            outcome,
          },
          signal: inspection.signal,
        }))()
    );
    inspection.check();
    const inspected = readProviderTextResult(result);
    inspection.retain(inspected.length * 2);
    return inspected;
  } catch (error) {
    throw toToolResultError(
      attempt?.signal.aborted === true ? attempt.signal.reason : error
    );
  } finally {
    attempt?.finish();
  }
}

/** Inspects what a selected in-process tool returned. */
export async function inspectToolOutput(
  gate: ToolResultGate,
  call: ToolResultCall,
  output: unknown,
  signal?: AbortSignal
): Promise<ToolMessage | string> {
  if (typeof output === 'string') {
    return inspectToolText(gate, call, output, 'success', signal);
  }
  if (
    !ToolMessage.isInstance(output) ||
    output.artifact != null ||
    typeof output.content !== 'string'
  ) {
    throw new ToolResultProtectionError('unsupported');
  }
  const content = await inspectToolText(
    gate,
    call,
    output.content,
    output.status === 'error' ? 'error' : 'success',
    signal
  );
  return content === output.content
    ? output
    : cloneToolMessageWithContent(output, content);
}

/** Inspects what the host returned for a selected tool call. */
export async function inspectToolExecuteResult(
  gate: ToolResultGate,
  call: ToolResultCall,
  result: ToolExecuteResult,
  signal?: AbortSignal
): Promise<ToolExecuteResult> {
  if (result.artifact != null) {
    throw new ToolResultProtectionError('unsupported');
  }
  if (result.status === 'error') {
    if (result.errorMessage == null) {
      return { ...result, content: '' };
    }
    if (typeof result.errorMessage !== 'string') {
      throw new ToolResultProtectionError('unsupported');
    }
    return {
      ...result,
      content: '',
      errorMessage: await inspectToolText(
        gate,
        call,
        result.errorMessage,
        'error',
        signal
      ),
    };
  }
  if (typeof result.content !== 'string') {
    throw new ToolResultProtectionError('unsupported');
  }
  return {
    ...result,
    content: await inspectToolText(
      gate,
      call,
      result.content,
      'success',
      signal
    ),
  };
}

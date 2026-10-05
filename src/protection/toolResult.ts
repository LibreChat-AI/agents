import { types } from 'node:util';
import { ToolMessage } from '@langchain/core/messages';
import { GraphInterrupt, ParentCommand } from '@langchain/langgraph';
import {
  StructuredTool,
  DynamicStructuredTool,
  DynamicTool,
  Tool,
} from '@langchain/core/tools';
import type { CallbackManagerForToolRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  ProviderTextProtection,
  ProviderTextProtectionResult,
  ProviderTextProtectionErrorCode,
} from './providerText';
import type { GenericTool, ToolExecuteResult, ToolCallRequest } from '@/types';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  validateProviderTextProtection,
} from './providerText';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';

export const TOOL_RESULT_PROTECTION_VERSION = 1;
export type ToolResultProtectionResult = ProviderTextProtectionResult;

/** Host selects trusted synchronous plain-text tools. Structured/file/background outputs stay gated. */
export interface ToolResultProtection {
  readonly version: 1;
  readonly toolNames: readonly string[];
  readonly timeoutMs: number;
  readonly maxAttemptBytes: number;
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
      readonly outcome: 'success' | 'error';
    };
    readonly signal: AbortSignal;
  }) => ToolResultProtectionResult | Promise<ToolResultProtectionResult>;
}

export class ToolResultProtectionError extends ProviderTextProtectionError {
  constructor(code: ProviderTextProtectionErrorCode) {
    super(code);
    this.name = 'ToolResultProtectionError';
  }
}

const adapters = new WeakMap<ToolResultProtection, ProviderTextProtection>();
const approved = new WeakMap<
  object,
  {
    policy: ToolResultProtection;
    name: string;
    id: string;
    text: string;
    status: string;
    referenceContent?: string;
  }
>();
const errors = new Set<ProviderTextProtectionErrorCode>([
  'blocked',
  'unavailable',
  'timeout',
  'overflow',
  'cancelled',
  'unsupported',
  'incompatible',
]);

export function validateToolResultProtection(
  policy: ToolResultProtection
): void {
  const version: number = policy.version;
  if (version !== 1) throw new ToolResultProtectionError('incompatible');
  if (
    typeof policy.inspect !== 'function' ||
    typeof policy.classify !== 'function'
  )
    throw new ToolResultProtectionError('unavailable');
  if (
    !Array.isArray(policy.toolNames) ||
    policy.toolNames.length === 0 ||
    policy.toolNames.length > 128 ||
    policy.toolNames.some(
      (name) =>
        typeof name !== 'string' || name.trim() !== name || name.length === 0
    )
  )
    throw new ToolResultProtectionError('incompatible');
  budget(policy);
}

function budget(policy: ToolResultProtection): ProviderTextProtection {
  let adapter = adapters.get(policy);
  if (adapter == null) {
    adapter = {
      version: 1,
      timeoutMs: policy.timeoutMs,
      maxAttemptBytes: policy.maxAttemptBytes,
      maxBufferedBytes: policy.maxBufferedBytes,
      classify: (): 'prose' => 'prose',
      inspect: ({ content }): ProviderTextProtectionResult => ({
        version: 1,
        ok: true,
        value: { content, replacements: 0, categories: [] },
      }),
    };
    validateProviderTextProtection(adapter);
    adapters.set(policy, adapter);
  }
  return adapter;
}

export function requiresToolResultProtection(
  policy: ToolResultProtection | undefined,
  name: string
): policy is ToolResultProtection {
  return policy != null && policy.toolNames.includes(name);
}

function plainObject(value: object, allowed: readonly string[]): void {
  if (types.isProxy(value)) throw new ToolResultProtectionError('unsupported');
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!allowed.includes(key) || !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
  }
}

function decision(result: ToolResultProtectionResult | undefined): string {
  if (result == null || typeof result !== 'object' || types.isProxy(result))
    throw new ToolResultProtectionError('incompatible');
  const version: number = result.version;
  if (version !== 1 || typeof result.ok !== 'boolean')
    throw new ToolResultProtectionError('incompatible');
  if (!result.ok)
    throw new ToolResultProtectionError(
      errors.has(result.error.code) ? result.error.code : 'incompatible'
    );
  if (
    typeof result.value.content !== 'string' ||
    !Number.isSafeInteger(result.value.replacements) ||
    result.value.replacements < 0 ||
    !Array.isArray(result.value.categories) ||
    result.value.categories.length > 32 ||
    result.value.categories.some(
      (entry) =>
        typeof entry.category !== 'string' ||
        !Number.isSafeInteger(entry.count) ||
        entry.count <= 0
    )
  )
    throw new ToolResultProtectionError('incompatible');
  return result.value.content;
}

async function release(
  policy: ToolResultProtection,
  name: string,
  id: string,
  text: unknown,
  status: 'success' | 'error',
  attempt: ProviderTextAttempt
): Promise<string> {
  attempt.check();
  if (typeof text !== 'string')
    throw new ToolResultProtectionError('unsupported');
  attempt.append(text);
  try {
    if (policy.classify(text) !== 'prose')
      throw new ToolResultProtectionError('unsupported');
    attempt.check();
    const result = await attempt.wait(
      Promise.resolve(
        policy.inspect({
          version: 1,
          content: text,
          toolName: name,
          toolCallId: id,
          target: {
            source: 'tool_argument',
            field: 'output',
            provenance: 'tool',
            outcome: status,
          },
          signal: attempt.signal,
        })
      )
    );
    attempt.check();
    const canonical = decision(result);
    attempt.check();
    attempt.retain(canonical.length * 2);
    return canonical;
  } catch (error) {
    attempt.check();
    if (error instanceof ProviderTextProtectionError)
      throw new ToolResultProtectionError(error.code);
    throw new ToolResultProtectionError('unavailable');
  }
}

export async function protectToolText(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  text: unknown,
  status: 'success' | 'error',
  signal?: AbortSignal
): Promise<unknown> {
  if (!requiresToolResultProtection(policy, name)) return text;
  validateToolResultProtection(policy);
  const attempt = new ProviderTextAttempt(budget(policy), signal);
  try {
    return await release(policy, name, id, text, status, attempt);
  } finally {
    attempt.finish();
  }
}

export function isReleasedToolError(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  error: Error
): boolean {
  const entry = approved.get(error);
  return (
    entry != null &&
    entry.policy === policy &&
    entry.name === name &&
    entry.id === id &&
    entry.status === 'error' &&
    entry.text === error.message
  );
}

/** Internal provenance only after an awaited release; never exposed from the root SDK entry. */
export function markReleasedToolMessage(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  message: ToolMessage,
  referenceContent?: string
): ToolMessage {
  if (
    requiresToolResultProtection(policy, name) &&
    typeof message.content === 'string'
  ) {
    approved.set(message, {
      policy,
      name,
      id,
      text: message.content,
      status: message.status ?? 'success',
      referenceContent:
        referenceContent ?? approved.get(message)?.referenceContent,
    });
  }
  return message;
}

export function hasReleasedToolReference(
  policy: ToolResultProtection,
  name: string,
  id: string,
  message: ToolMessage,
  referenceContent: string | undefined
): boolean {
  const entry = approved.get(message);
  return (
    entry != null &&
    entry.policy === policy &&
    entry.name === name &&
    entry.id === id &&
    entry.text === message.content &&
    entry.status === message.status &&
    entry.referenceContent === referenceContent
  );
}

function validateReplayReferenceMetadata(message: ToolMessage): void {
  const metadata = message.additional_kwargs;
  const key = metadata._refKey;
  const scope = metadata._refScope;
  const unresolved = metadata._unresolvedRefs;
  if (
    (key != null &&
      (typeof key !== 'string' || !/^tool\d+turn\d+$/.test(key))) ||
    (scope != null && typeof scope !== 'string') ||
    (unresolved != null &&
      (!Array.isArray(unresolved) ||
        unresolved.length > 128 ||
        unresolved.some(
          (ref) => typeof ref !== 'string' || !/^tool\d+turn\d+$/.test(ref)
        )))
  )
    throw new ToolResultProtectionError('unsupported');
}

export async function protectToolMessage(
  policy: ToolResultProtection | undefined,
  name: string,
  id: string,
  message: ToolMessage,
  signal?: AbortSignal,
  ownedReplayMetadata = false
): Promise<ToolMessage> {
  if (!requiresToolResultProtection(policy, name)) return message;
  if (types.isProxy(message))
    throw new ToolResultProtectionError('unsupported');
  for (const key of [
    'content',
    'artifact',
    'name',
    'id',
    'tool_call_id',
    'status',
    'additional_kwargs',
    'response_metadata',
  ]) {
    const descriptor = Object.getOwnPropertyDescriptor(message, key);
    if (descriptor != null && !('value' in descriptor))
      throw new ToolResultProtectionError('unsupported');
  }
  const status: string | undefined = message.status;
  if (
    types.isProxy(message) ||
    message.artifact != null ||
    message.tool_call_id !== id ||
    (message.name != null && message.name !== name) ||
    (status != null && status !== 'success' && status !== 'error')
  )
    throw new ToolResultProtectionError('unsupported');
  plainObject(message.additional_kwargs, [
    '_refKey',
    '_refScope',
    '_unresolvedRefs',
  ]);
  if (ownedReplayMetadata) validateReplayReferenceMetadata(message);
  if (
    approved.get(message) == null &&
    !ownedReplayMetadata &&
    (Object.keys(message.additional_kwargs).length > 0 ||
      (message.id != null && message.id !== id))
  )
    throw new ToolResultProtectionError('unsupported');
  if (Object.keys(message.response_metadata).length > 0)
    throw new ToolResultProtectionError('unsupported');
  const existing = approved.get(message);
  if (
    existing?.policy === policy &&
    existing.name === name &&
    existing.id === id &&
    existing.text === message.content &&
    existing.status === (message.status ?? 'success')
  )
    return message;
  const content = await protectToolText(
    policy,
    name,
    id,
    message.content,
    message.status === 'error' ? 'error' : 'success',
    signal
  );
  const safe = new ToolMessage({
    name: message.name ?? name,
    id: message.id,
    tool_call_id: id,
    status: message.status ?? 'success',
    content: content as string,
    additional_kwargs: message.additional_kwargs,
  });
  approved.set(safe, {
    policy,
    name,
    id,
    text: safe.content as string,
    status: safe.status ?? 'success',
  });
  return safe;
}

export async function protectToolExecuteResult(
  policy: ToolResultProtection | undefined,
  request: Pick<ToolCallRequest, 'id' | 'name'>,
  result: ToolExecuteResult,
  signal?: AbortSignal
): Promise<ToolExecuteResult> {
  if (!requiresToolResultProtection(policy, request.name)) return result;
  plainObject(result, [
    'toolCallId',
    'received_at',
    'content',
    'status',
    'errorMessage',
    'artifact',
    'injectedMessages',
  ]);
  const status: string = result.status;
  if (
    result.toolCallId !== request.id ||
    (status !== 'success' && status !== 'error') ||
    result.artifact != null ||
    (result.received_at != null &&
      (typeof result.received_at !== 'number' ||
        !Number.isFinite(result.received_at))) ||
    (result.injectedMessages?.length ?? 0) > 0 ||
    (result.status === 'success' && result.errorMessage != null)
  )
    throw new ToolResultProtectionError('unsupported');
  const text =
    result.status === 'error'
      ? (result.errorMessage ?? result.content)
      : result.content;
  const cached = approved.get(result);
  if (
    cached?.policy === policy &&
    cached.name === request.name &&
    cached.id === request.id &&
    cached.text === text &&
    cached.status === result.status
  )
    return result;
  const canonical = (await protectToolText(
    policy,
    request.name,
    request.id,
    text,
    result.status,
    signal
  )) as string;
  const safe: ToolExecuteResult = {
    toolCallId: request.id,
    status: result.status,
    content: result.status === 'error' ? '' : canonical,
    ...(result.status === 'error' ? { errorMessage: canonical } : {}),
    ...(result.received_at == null ? {} : { received_at: result.received_at }),
  };
  approved.set(safe, {
    policy,
    name: request.name,
    id: request.id,
    text: canonical,
    status: safe.status,
  });
  return safe;
}

/** Keep child/control APIs usable, but no raw body observations escape before release. */
function toolBodyCallbacks(
  manager: CallbackManagerForToolRun | undefined
): CallbackManagerForToolRun | undefined {
  if (manager == null) return manager;
  return Object.create(Object.getPrototypeOf(manager), {
    ...Object.getOwnPropertyDescriptors(manager),
    handlers: { value: [], enumerable: true },
    inheritableHandlers: { value: [], enumerable: true },
  }) as CallbackManagerForToolRun;
}

/** Intercepts StructuredTool before native tool-end callbacks, never mutating a shared tool. */
export function withToolResultBoundary(
  tool: GenericTool,
  policy: ToolResultProtection | undefined,
  id: string,
  config: RunnableConfig
): GenericTool {
  if (!requiresToolResultProtection(policy, tool.name)) return tool;
  if (
    [
      'subagent',
      'run_tools_with_code',
      'run_tools_with_bash',
      'read_file',
      'skill',
    ].includes(tool.name)
  )
    throw new ToolResultProtectionError('unsupported');
  if (!(tool instanceof StructuredTool))
    throw new ToolResultProtectionError('unsupported');
  if (
    tool.invoke !== StructuredTool.prototype.invoke ||
    ![
      StructuredTool.prototype.call,
      DynamicStructuredTool.prototype.call,
      DynamicTool.prototype.call,
      Tool.prototype.call,
    ].includes(tool.call)
  )
    throw new ToolResultProtectionError('unsupported');
  const protectedTool = Object.create(
    Object.getPrototypeOf(tool),
    Object.getOwnPropertyDescriptors(tool)
  ) as StructuredTool;
  protectedTool.responseFormat = 'content';
  const invoke: (
    input: unknown,
    manager: CallbackManagerForToolRun | undefined,
    config: RunnableConfig
  ) => Promise<unknown> = Reflect.get(tool, '_call');
  Object.defineProperty(protectedTool, '_call', {
    value: async (
      input: unknown,
      manager: CallbackManagerForToolRun | undefined,
      effective?: RunnableConfig
    ): Promise<ToolMessage> => {
      const attempt = new ProviderTextAttempt(
        budget(policy),
        effective?.signal ?? config.signal
      );
      try {
        let raw: unknown;
        let status: 'success' | 'error' = 'success';
        try {
          raw = await attempt.wait(
            Promise.resolve(
              invoke.call(tool, input, toolBodyCallbacks(manager), {
                ...effective,
                callbacks: [],
              })
            )
          );
        } catch (error) {
          attempt.check();
          if (
            error instanceof ProviderTextProtectionError ||
            error instanceof PreparedSubagentError ||
            error instanceof StreamLimitExceededError ||
            error instanceof GraphInterrupt
          )
            throw error;
          if (error instanceof ParentCommand)
            throw new ToolResultProtectionError('unsupported');
          if (!(error instanceof Error))
            throw new ToolResultProtectionError('unsupported');
          raw = error.message;
          status = 'error';
        }
        attempt.check();
        if (raw instanceof ToolMessage)
          return await protectToolMessage(
            policy,
            tool.name,
            id,
            raw,
            attempt.signal
          );
        if (
          status === 'success' &&
          tool.responseFormat === 'content_and_artifact'
        ) {
          if (!Array.isArray(raw) || raw.length !== 2 || raw[1] != null)
            throw new ToolResultProtectionError('unsupported');
          raw = raw[0];
        }
        const content = await release(
          policy,
          tool.name,
          id,
          raw,
          status,
          attempt
        );
        if (status === 'error') {
          const error = new Error(content);
          approved.set(error, {
            policy,
            name: tool.name,
            id,
            text: content,
            status,
          });
          throw error;
        }
        const message = new ToolMessage({
          name: tool.name,
          tool_call_id: id,
          content,
          status,
        });
        approved.set(message, {
          policy,
          name: tool.name,
          id,
          text: content,
          status,
        });
        return message;
      } catch (error) {
        attempt.check();
        if (
          error instanceof ProviderTextProtectionError ||
          error instanceof PreparedSubagentError ||
          error instanceof StreamLimitExceededError ||
          error instanceof GraphInterrupt ||
          (error instanceof Error &&
            isReleasedToolError(policy, tool.name, id, error))
        )
          throw error;
        throw new ToolResultProtectionError('unavailable');
      } finally {
        attempt.finish();
      }
    },
  });
  return protectedTool;
}

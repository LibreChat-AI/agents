import type { AIMessageChunk, BaseMessage } from '@langchain/core/messages';
import type { ProviderName } from '@/types';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

/**
 * Providers whose adapters deliver tool calls as complete, atomic objects
 * rather than streamed argument deltas. Google/Vertex (GenAI) emit each
 * `functionCall` whole and seal it on arrival, so a `MAX_TOKENS` finish does
 * NOT imply the arguments were cut off — the truncation guard must skip them.
 */
const ATOMIC_TOOL_CALL_ARG_PROVIDERS = new Set<ProviderName>([
  Providers.GOOGLE,
  Providers.VERTEXAI,
]);

/**
 * Normalized truncation reasons across providers. Providers disagree on the
 * key and the value: Anthropic/Bedrock use `max_tokens`, OpenAI uses `length`,
 * Google uses `MAX_TOKENS`.
 */
export type TruncationStopReason = 'max_tokens' | 'length';

const MAX_TOKEN_VALUES = new Set([
  'max_tokens',
  'max_token',
  'maxtokens',
  'max_output_tokens',
]);
const LENGTH_VALUES = new Set(['length']);

function normalizeStopValue(value: unknown): TruncationStopReason | null {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (MAX_TOKEN_VALUES.has(normalized)) {
    return 'max_tokens';
  }
  if (LENGTH_VALUES.has(normalized)) {
    return 'length';
  }
  return null;
}

/** Normalizes native invoke and stream stop reasons without hiding conflicting fields. */
export function getProviderStopReasons(
  message:
    | Pick<BaseMessage, 'response_metadata' | 'additional_kwargs'>
    | undefined
    | null
): string[] {
  const meta = message?.response_metadata as
    | Record<string, unknown>
    | undefined;
  const additionalKwargs = message?.additional_kwargs as
    | Record<string, unknown>
    | undefined;
  if (meta == null && additionalKwargs == null) {
    return [];
  }

  const messageStop = meta?.messageStop as { stopReason?: unknown } | undefined;
  const incompleteDetails = meta?.incomplete_details as
    | { reason?: unknown }
    | undefined;
  const candidates: unknown[] = [
    meta?.stopReason,
    meta?.stop_reason,
    meta?.finish_reason,
    meta?.finishReason,
    messageStop?.stopReason,
    incompleteDetails?.reason,
    additionalKwargs?.stop_reason,
  ];

  const reasons: string[] = [];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const reason = candidate.trim().toLowerCase();
    if (reason !== '') reasons.push(reason);
  }
  return reasons;
}

/**
 * Reads the truncation stop reason off a message, covering every provider
 * shape we stream:
 * - `response_metadata.stopReason` (Bedrock Converse, non-streaming)
 * - `response_metadata.messageStop.stopReason` (Bedrock Converse streaming)
 * - `response_metadata.stop_reason` (Anthropic, non-streaming)
 * - `additional_kwargs.stop_reason` (Anthropic streaming `message_delta`)
 * - `response_metadata.finish_reason` (OpenAI chat-completions / compatible)
 * - `response_metadata.incomplete_details.reason` (OpenAI Responses API)
 * - `response_metadata.finishReason` (Google)
 *
 * Returns the normalized reason when the model stopped because it hit the
 * output token ceiling, otherwise `null`.
 */
export function getTruncationStopReason(
  message:
    | Pick<BaseMessage, 'response_metadata' | 'additional_kwargs'>
    | undefined
    | null
): TruncationStopReason | null {
  for (const candidate of getProviderStopReasons(message)) {
    const normalized = normalizeStopValue(candidate);
    if (normalized != null) {
      return normalized;
    }
  }
  return null;
}

function hasOpenToolCall(message: AIMessageChunk | BaseMessage): boolean {
  const m = message as AIMessageChunk & {
    tool_calls?: unknown[];
    tool_call_chunks?: unknown[];
    invalid_tool_calls?: unknown[];
  };
  return (
    (m.tool_calls?.length ?? 0) > 0 ||
    (m.tool_call_chunks?.length ?? 0) > 0 ||
    (m.invalid_tool_calls?.length ?? 0) > 0
  );
}

/**
 * Error raised when a model turn is cut off by the output token limit while it
 * was still emitting a tool call. The arguments are necessarily incomplete, so
 * executing or re-prompting the call would loop on a malformed request. Failing
 * fast with this surfaces an actionable message to the caller instead.
 */
export class OutputTruncationError extends Error {
  readonly stopReason: TruncationStopReason;
  readonly toolCallNames: string[];

  constructor(stopReason: TruncationStopReason, toolCallNames: string[]) {
    const named =
      toolCallNames.length > 0
        ? ` (tool call: ${toolCallNames.join(', ')})`
        : '';
    super(
      'The model response was truncated at the maximum output token limit before ' +
        `the tool call arguments were complete${named}. Increase the model's max ` +
        'output tokens, or have the model produce smaller arguments — for large ' +
        'files, write a lean main file and move bulky content into separate files.'
    );
    this.name = 'OutputTruncationError';
    this.stopReason = stopReason;
    this.toolCallNames = toolCallNames;
  }
}

function collectToolCallNames(message: AIMessageChunk | BaseMessage): string[] {
  const m = message as AIMessageChunk & {
    tool_calls?: Array<{ name?: string }>;
    tool_call_chunks?: Array<{ name?: string }>;
    invalid_tool_calls?: Array<{ name?: string }>;
  };
  const names = [
    ...(m.tool_calls ?? []),
    ...(m.tool_call_chunks ?? []),
    ...(m.invalid_tool_calls ?? []),
  ]
    .map((tc) => tc.name)
    .filter((name): name is string => typeof name === 'string' && name !== '');
  return [...new Set(names)];
}

/**
 * Throws {@link OutputTruncationError} when `message` was truncated by the
 * output token limit AND still carries a tool call. For providers that stream
 * tool arguments incrementally (Anthropic, Bedrock, OpenAI), a tool call
 * emitted under truncation has incomplete arguments — the tool-use block is the
 * last thing the model streams — so letting it through makes the agent loop on a
 * malformed call. No-ops for normal completions, truncated plain-text turns, and
 * providers that deliver complete tool calls atomically (Google/Vertex), where
 * `MAX_TOKENS` does not imply the arguments were cut off.
 */
export function assertNotTruncatedToolCall(
  message: AIMessageChunk | BaseMessage | undefined | null,
  provider?: ProviderName
): void {
  if (message == null) {
    return;
  }
  if (
    provider != null &&
    (ATOMIC_TOOL_CALL_ARG_PROVIDERS.has(provider) ||
      getProviderFamily(provider) === 'google')
  ) {
    return;
  }
  const stopReason = getTruncationStopReason(message);
  if (stopReason == null || !hasOpenToolCall(message)) {
    return;
  }
  throw new OutputTruncationError(stopReason, collectToolCallNames(message));
}

import { isProxy } from 'node:util/types';
import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type { AIMessageChunk } from '@langchain/core/messages';
import type {
  ThinkingContentText,
  MessageContentComplex,
  ReasoningContentText,
} from '@/types/stream';
import type { ContextPruningConfig, FadingTier } from '@/types/graph';
import type { FadingCaps, FadingSignals } from './fading';
import type { TokenCounter } from '@/types/run';
import type { ProviderName } from '@/types';
import {
  HARD_MAX_TOOL_CALL_INPUT_CHARS,
  HARD_MAX_TOOL_RESULT_CHARS,
  MIN_JSON_VALUE_CHARS,
  sliceWithoutSplittingSurrogates,
  calculateMaxToolCallInputChars,
  calculateMaxToolResultChars,
} from '@/utils/truncation';
import {
  cloneToolMessageWithContent,
  compactToolContent,
  isComputerCallOutputMessage,
  serializeStructuredValueBounded,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  MASKED_RESULT_MIN_CHARS,
  fadingRungForExchangeChars,
  isFadingTier,
  resolveFadingCaps,
  resolveFadingTier,
  seedFadingTier,
} from './fading';
import {
  dropIncompleteToolStreamContent,
  hasNonEmptyTextContent,
} from './core';
import { resolveContextPruningSettings } from './contextPruningSettings';
import { hasUnsafeStructuredSerialization } from '@/utils/tokens';
import { ContentTypes, Providers, Constants } from '@/common';
import { getProviderFamily } from '@/llm/providerRegistry';
import { applyContextPruning } from './contextPruning';
import { toLangChainContent } from './langchain';
import { cloneMessage } from './cache';

function sumTokenCounts(
  tokenMap: Record<string, number | undefined>,
  count: number
): number {
  let total = 0;
  for (let i = 0; i < count; i++) {
    total += tokenMap[i] ?? 0;
  }
  return total;
}

function appendItems<T>(target: T[], source: readonly T[]): void {
  for (const item of source) {
    target.push(item);
  }
}

/** Prepends in stable order without turning a large array into call arguments. */
function prependItems<T>(target: T[], prefix: readonly T[]): void {
  const prefixLength = prefix.length;
  if (prefixLength === 0) {
    return;
  }
  const originalLength = target.length;
  target.length = prefixLength + originalLength;
  for (let index = originalLength - 1; index >= 0; index--) {
    target[prefixLength + index] = target[index];
  }
  for (let index = 0; index < prefixLength; index++) {
    target[index] = prefix[index];
  }
}

/** Default fraction of the token budget reserved as headroom (5 %). */
export const DEFAULT_RESERVE_RATIO = 0.05;

/** Provider framing reserved for the assistant reply label. */
export const REPLY_PRIMER_TOKENS = 3;

/** Hard cap for the originalToolContent store (~2 MB estimated from char length). */
export const ORIGINAL_CONTENT_MAX_CHARS = 2_000_000;

/**
 * Evicts oldest entries from `map` (in Map-iteration / insertion order) until
 * the cumulative char length of remaining values fits within
 * `ORIGINAL_CONTENT_MAX_CHARS`.  Used by the recency-window carry-over merge
 * path in Graph.ts to bound long-running session memory: the pruner enforces
 * the cap inside its own `originalToolContent` map, but a key-wise union with
 * recency carry-over bypasses that cap unless re-applied here.
 */
export function enforceOriginalContentCap(map: Map<number, string>): void {
  let total = 0;
  for (const v of map.values()) {
    total += v.length;
  }
  while (total > ORIGINAL_CONTENT_MAX_CHARS && map.size > 0) {
    const oldest = map.keys().next();
    if (oldest.done === true) {
      break;
    }
    const removed = map.get(oldest.value);
    if (removed != null) {
      total -= removed.length;
    }
    map.delete(oldest.value);
  }
}

/** Minimum cumulative calibration ratio — provider can't count fewer tokens
 *  than our raw estimate (within reason). Prevents divide-by-zero edge cases. */
export const CALIBRATION_RATIO_MIN = 0.5;

/** Maximum cumulative calibration ratio — sanity cap for the running ratio. */
export const CALIBRATION_RATIO_MAX = 5;

/** Keeps provider/local token calibration within the shared safe range. */
export function clampCalibrationRatio(ratio: number): number {
  return Math.max(
    CALIBRATION_RATIO_MIN,
    Math.min(CALIBRATION_RATIO_MAX, ratio)
  );
}

export type PruneMessagesFactoryParams = {
  provider?: ProviderName;
  maxTokens: number;
  /** Per-tool-result character cap applied while reconciling cached counts. */
  maxToolResultChars?: number;
  startIndex: number;
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
  thinkingEnabled?: boolean;
  /** Context pruning configuration for position-based tool result degradation. */
  contextPruningConfig?: ContextPruningConfig;
  /**
   * When true, context pressure fading (pre-flight tool result truncation)
   * is skipped.  Summarization replaces pruning as the primary context
   * management strategy — the summarizer needs full un-truncated tool results
   * to produce an accurate summary.  Hard pruning still runs as a fallback
   * when summarization is skipped or capped.
   */
  summarizationEnabled?: boolean;
  /**
   * Returns the current instruction-token overhead (system message + tool schemas + summary).
   * Called on each prune invocation so the budget reflects dynamic changes
   * (e.g. summary added between turns).  When messages don't include a leading
   * SystemMessage, these tokens are subtracted from the available budget so
   * the pruner correctly reserves space for the system prompt that will be
   * prepended later by `buildSystemRunnable`.
   */
  getInstructionTokens?: () => number;
  /**
   * Fraction of the effective token budget to reserve as headroom (0–1).
   * When set, pruning triggers at `effectiveMax * (1 - reserveRatio)` instead of
   * filling the context window to 100%.  Defaults to 5 % (0.05) when omitted.
   */
  reserveRatio?: number;
  /**
   * Initial calibration ratio from a previous run's persisted contextMeta.
   * Seeds the running EMA so new messages are scaled immediately instead
   * of waiting for the first provider response.  Ignored when <= 0.
   */
  calibrationRatio?: number;
  /**
   * Context-fading tier persisted from a previous run's contextMeta. Seeds the
   * latched cap ladder so historical tool results keep the same truncated
   * bytes across runs. Invalid values start fresh; valid values clamp to the
   * current context window without losing their latched provenance.
   */
  fadingTier?: FadingTier | null;
  /** Optional diagnostic log callback wired by the graph for observability. */
  log?: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    data?: Record<string, unknown>
  ) => void;
};
export type PruneMessagesParams = {
  /**
   * Immutable graph history corresponding index-for-index with `messages`.
   * When supplied, provider projections always derive from this source rather
   * than from an earlier, already-truncated projection.
   */
  canonicalMessages?: BaseMessage[];
  /**
   * The caller guarantees that an existing canonical prefix cannot have been
   * rewritten since the previous call. Graph reducers provide this guarantee
   * by invalidating and recreating the pruner on replacements/removals.
   */
  canonicalPrefixStable?: boolean;
  messages: BaseMessage[];
  usageMetadata?: Partial<UsageMetadata>;
  startType?: ReturnType<BaseMessage['getType']>;
  /**
   * Fallback usage from the most recent LLM call only (not accumulated).
   * Calibration prefers the provider's raw `usageMetadata.input_tokens`
   * when available, because cache detail fields may use non-window
   * accounting units.
   */
  lastCallUsage?: {
    totalTokens: number;
    inputTokens?: number;
  };
  /**
   * Whether the token data is fresh (from a just-completed LLM call).
   * When false, provider calibration is skipped to avoid applying
   * stale ratios.
   */
  totalTokensFresh?: boolean;
};

function getToolCallIds(message: BaseMessage): Set<string> {
  const messageRole = (message as BaseMessage & { role?: unknown }).role;
  if (message.getType() !== 'ai' && messageRole !== 'assistant') {
    return new Set<string>();
  }

  const ids = new Set<string>();
  const aiMessage = message as AIMessage;
  const toolCalls = aiMessage.tool_calls;
  if (Array.isArray(toolCalls) && !isProxy(toolCalls)) {
    for (const toolCall of toolCalls) {
      if (typeof toolCall !== 'object') {
        continue;
      }
      const id = getStringProperty(toolCall, 'id');
      if (id != null && id.length > 0) {
        ids.add(id);
      }
    }
  }

  const rawToolCalls = readPropertyWithoutAccessors(
    aiMessage.additional_kwargs,
    'tool_calls'
  );
  if (
    !rawToolCalls.accessor &&
    Array.isArray(rawToolCalls.value) &&
    !isProxy(rawToolCalls.value)
  ) {
    for (const toolCall of rawToolCalls.value) {
      if (toolCall == null || typeof toolCall !== 'object') {
        continue;
      }
      const id = getStringProperty(toolCall, 'id');
      if (id != null && id.length > 0) {
        ids.add(id);
      }
    }
  }

  if (Array.isArray(aiMessage.content) && !isProxy(aiMessage.content)) {
    for (const part of aiMessage.content) {
      if (typeof part !== 'object') {
        continue;
      }
      const type = getStringProperty(part, 'type');
      const id = getStringProperty(part, 'id');
      if (
        (type === 'tool_use' || type === 'tool_call') &&
        id != null &&
        id.length > 0
      ) {
        ids.add(id);
      }
    }
  }

  const toolOutputSource = getResponsesToolOutputSource(message);
  if (toolOutputSource != null) {
    for (const item of toolOutputSource.items) {
      if (item == null || typeof item !== 'object') {
        continue;
      }
      const type = getStringProperty(item, 'type');
      const callId = getStringProperty(item, 'call_id');
      if (
        (type === 'function_call' ||
          type === 'custom_tool_call' ||
          type === 'computer_call') &&
        callId != null &&
        callId !== ''
      ) {
        ids.add(callId);
      }
    }
  }

  return ids;
}

type ResponsesToolOutputSource = {
  source: 'response_metadata' | 'tool_outputs';
  items: unknown[];
};

function getResponsesToolOutputSource(
  message: BaseMessage
): ResponsesToolOutputSource | undefined {
  const responseOutput = readPropertyWithoutAccessors(
    message.response_metadata,
    'output'
  );
  if (
    !responseOutput.accessor &&
    Array.isArray(responseOutput.value) &&
    !isProxy(responseOutput.value) &&
    responseOutput.value.length > 0
  ) {
    return {
      source: 'response_metadata',
      items: responseOutput.value,
    };
  }
  const fallbackOutput = readPropertyWithoutAccessors(
    message.additional_kwargs,
    'tool_outputs'
  );
  if (
    !fallbackOutput.accessor &&
    Array.isArray(fallbackOutput.value) &&
    !isProxy(fallbackOutput.value)
  ) {
    return {
      source: 'tool_outputs',
      items: fallbackOutput.value,
    };
  }
  return undefined;
}

function getToolResultId(message: BaseMessage): string | null {
  if (message.getType() !== 'tool') {
    return null;
  }
  const toolMessage = message as ToolMessage & {
    tool_call_id?: unknown;
    toolCallId?: unknown;
  };
  if (
    typeof toolMessage.tool_call_id === 'string' &&
    toolMessage.tool_call_id.length > 0
  ) {
    return toolMessage.tool_call_id;
  }
  if (
    typeof toolMessage.toolCallId === 'string' &&
    toolMessage.toolCallId.length > 0
  ) {
    return toolMessage.toolCallId;
  }
  return null;
}

function resolveTokenCountForMessage({
  message,
  messageIndexMap,
  tokenCounter,
  indexTokenCountMap,
}: {
  message: BaseMessage;
  messageIndexMap: Map<BaseMessage, number>;
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
}): number {
  const originalIndex = messageIndexMap.get(message) ?? -1;
  if (originalIndex > -1 && indexTokenCountMap[originalIndex] != null) {
    return indexTokenCountMap[originalIndex] as number;
  }
  return tokenCounter(message);
}

export function repairOrphanedToolMessages({
  context,
  allMessages,
  tokenCounter,
  indexTokenCountMap,
}: {
  context: BaseMessage[];
  allMessages: BaseMessage[];
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
}): {
  context: BaseMessage[];
  reclaimedTokens: number;
  droppedOrphanCount: number;
  /** Messages removed from context during orphan repair.  These should be
   *  appended to `messagesToRefine` so that summarization can still see them
   *  (e.g. a ToolMessage whose parent AI was pruned). */
  droppedMessages: BaseMessage[];
} {
  const messageIndexMap = new Map<BaseMessage, number>();
  for (let i = 0; i < allMessages.length; i++) {
    messageIndexMap.set(allMessages[i], i);
  }

  const validToolCallIds = new Set<string>();
  const presentToolResultIds = new Set<string>();
  for (const message of context) {
    for (const id of getToolCallIds(message)) {
      validToolCallIds.add(id);
    }
    const resultId = getToolResultId(message);
    if (resultId != null) {
      presentToolResultIds.add(resultId);
    }
  }

  let reclaimedTokens = 0;
  let droppedOrphanCount = 0;
  const repairedContext: BaseMessage[] = [];
  const droppedMessages: BaseMessage[] = [];

  for (const message of context) {
    if (message.getType() === 'tool') {
      const toolResultId = getToolResultId(message);
      if (toolResultId == null || !validToolCallIds.has(toolResultId)) {
        droppedOrphanCount += 1;
        reclaimedTokens += resolveTokenCountForMessage({
          message,
          tokenCounter,
          messageIndexMap,
          indexTokenCountMap,
        });
        droppedMessages.push(message);
        continue;
      }
      repairedContext.push(message);
      continue;
    }

    const messageRole = (message as BaseMessage & { role?: unknown }).role;
    if (message.getType() === 'ai' || messageRole === 'assistant') {
      const toolCallIds = getToolCallIds(message);
      if (toolCallIds.size > 0) {
        let hasOrphanToolCalls = false;
        for (const id of toolCallIds) {
          if (!presentToolResultIds.has(id)) {
            hasOrphanToolCalls = true;
            break;
          }
        }
        if (hasOrphanToolCalls) {
          const originalTokens = resolveTokenCountForMessage({
            message,
            messageIndexMap,
            tokenCounter,
            indexTokenCountMap,
          });
          const stripped = stripOrphanToolUseBlocks(
            message,
            presentToolResultIds
          );
          if (stripped != null) {
            const strippedTokens = tokenCounter(stripped);
            reclaimedTokens += originalTokens - strippedTokens;
            repairedContext.push(stripped);
          } else {
            droppedOrphanCount += 1;
            reclaimedTokens += originalTokens;
            droppedMessages.push(message);
          }
          continue;
        }
      }
    }

    repairedContext.push(message);
  }

  return {
    context: repairedContext,
    reclaimedTokens,
    droppedOrphanCount,
    droppedMessages,
  };
}

/**
 * Strips tool_use content blocks and tool_calls entries from an AI message
 * when their corresponding ToolMessages are not in the context.
 * Returns null if the message has no content left after stripping.
 */
function stripOrphanToolUseBlocks(
  message: BaseMessage,
  presentToolResultIds: Set<string>
): BaseMessage | null {
  const aiMessage = message as AIMessage;
  const keptToolCalls = (aiMessage.tool_calls ?? []).filter(
    (tc) => typeof tc.id === 'string' && presentToolResultIds.has(tc.id)
  );

  let keptContent: MessageContentComplex[] | string;
  if (Array.isArray(aiMessage.content)) {
    const filtered = (aiMessage.content as MessageContentComplex[]).filter(
      (block) => {
        if (typeof block !== 'object') {
          return true;
        }
        const record = block as { type?: unknown; id?: unknown };
        if (
          (record.type === 'tool_use' || record.type === 'tool_call') &&
          typeof record.id === 'string'
        ) {
          return presentToolResultIds.has(record.id);
        }
        return true;
      }
    );

    keptContent = filtered;
  } else {
    keptContent = aiMessage.content;
  }

  const toolOutputSource = getResponsesToolOutputSource(message);
  const keptToolOutputs =
    toolOutputSource?.items.filter((item) => {
      if (item == null || typeof item !== 'object') {
        return true;
      }
      const record = item as {
        type?: unknown;
        call_id?: unknown;
      };
      if (
        record.type !== 'function_call' &&
        record.type !== 'custom_tool_call' &&
        record.type !== 'computer_call'
      ) {
        return true;
      }
      return (
        typeof record.call_id === 'string' &&
        presentToolResultIds.has(record.call_id)
      );
    }) ?? [];

  if (
    keptToolCalls.length === 0 &&
    Array.isArray(keptContent) &&
    keptContent.length === 0 &&
    keptToolOutputs.length === 0
  ) {
    return null;
  }

  let responseMetadata = message.response_metadata;
  let additionalKwargs = message.additional_kwargs;
  if (toolOutputSource?.source === 'response_metadata') {
    responseMetadata = cloneWithProjectedProperties(responseMetadata, {
      output: keptToolOutputs,
    });
  } else if (toolOutputSource?.source === 'tool_outputs') {
    additionalKwargs = cloneWithProjectedProperties(additionalKwargs, {
      tool_outputs: keptToolOutputs,
    });
  }

  return cloneWithProjectedProperties(message, {
    content: toLangChainContent(keptContent),
    tool_calls: keptToolCalls.length > 0 ? keptToolCalls : undefined,
    response_metadata: responseMetadata,
    additional_kwargs: additionalKwargs,
  });
}

/**
 * Lightweight structural cleanup: strips orphan tool_use blocks from AI messages
 * and drops orphan ToolMessages whose AI counterpart is missing.
 *
 * Unlike `repairOrphanedToolMessages`, this does NOT track tokens — it is
 * intended as a final safety net in Graph.ts right before model invocation
 * to prevent Anthropic/Bedrock structural validation errors.
 *
 * Uses duck-typing instead of `getType()` because messages at this stage
 * may be plain objects (from LangGraph state serialization) rather than
 * proper BaseMessage class instances.
 *
 * Includes a fast-path: if every tool_call has a matching tool_result and
 * vice-versa, the original array is returned immediately with zero allocation.
 */
export function sanitizeOrphanToolBlocks(
  messages: BaseMessage[],
  onMessageCloned?: (source: BaseMessage, clone: BaseMessage) => void
): BaseMessage[] {
  const allToolCallIds = new Set<string>();
  const allToolResultIds = new Set<string>();

  for (const msg of messages) {
    const msgAny = msg as unknown as Record<string, unknown>;
    if (typeof (msg as { getType?: unknown }).getType === 'function') {
      for (const id of getToolCallIds(msg)) {
        allToolCallIds.add(id);
      }
    }
    const toolCalls = msgAny.tool_calls as Array<{ id?: string }> | undefined;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        if (
          typeof tc.id === 'string' &&
          tc.id.length > 0 &&
          !tc.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        ) {
          allToolCallIds.add(tc.id);
        }
      }
    }
    if (Array.isArray(msgAny.content)) {
      for (const block of msgAny.content as Array<Record<string, unknown>>) {
        if (
          typeof block === 'object' &&
          (block.type === 'tool_use' || block.type === 'tool_call') &&
          typeof block.id === 'string' &&
          !block.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        ) {
          allToolCallIds.add(block.id);
        }
      }
    }
    const toolCallId = msgAny.tool_call_id as string | undefined;
    if (typeof toolCallId === 'string' && toolCallId.length > 0) {
      allToolResultIds.add(toolCallId);
    }
  }

  let hasOrphans = false;
  for (const id of allToolCallIds) {
    if (!allToolResultIds.has(id)) {
      hasOrphans = true;
      break;
    }
  }
  if (!hasOrphans) {
    for (const id of allToolResultIds) {
      if (!allToolCallIds.has(id)) {
        hasOrphans = true;
        break;
      }
    }
  }
  if (!hasOrphans) {
    return messages;
  }

  const result: BaseMessage[] = [];
  const strippedAiIndices = new Set<number>();

  for (const msg of messages) {
    const msgAny = msg as unknown as Record<string, unknown>;
    const msgType =
      typeof (msg as { getType?: unknown }).getType === 'function'
        ? msg.getType()
        : ((msgAny.role as string | undefined) ??
          (msgAny._type as string | undefined));

    const toolCallId = msgAny.tool_call_id as string | undefined;
    if (
      (msgType === 'tool' || msg instanceof ToolMessage) &&
      typeof toolCallId === 'string' &&
      !allToolCallIds.has(toolCallId)
    ) {
      continue;
    }

    const toolCalls = msgAny.tool_calls as Array<{ id?: string }> | undefined;
    if (
      (msgType === 'ai' ||
        msgType === 'assistant' ||
        msg instanceof AIMessage) &&
      Array.isArray(toolCalls) &&
      toolCalls.length > 0
    ) {
      const hasOrphanCalls = toolCalls.some(
        (tc) => typeof tc.id === 'string' && !allToolResultIds.has(tc.id)
      );
      if (hasOrphanCalls) {
        if (msg instanceof AIMessage) {
          const stripped = stripOrphanToolUseBlocks(msg, allToolResultIds);
          if (stripped != null) {
            strippedAiIndices.add(result.length);
            onMessageCloned?.(msg, stripped);
            result.push(stripped);
          }
          continue;
        }
        const keptToolCalls = toolCalls.filter(
          (tc) => typeof tc.id === 'string' && allToolResultIds.has(tc.id)
        );
        const keptContent = Array.isArray(msgAny.content)
          ? (msgAny.content as Array<Record<string, unknown>>).filter(
            (block) => {
              if (typeof block !== 'object') return true;
              if (
                (block.type === 'tool_use' || block.type === 'tool_call') &&
                  typeof block.id === 'string'
              ) {
                return allToolResultIds.has(block.id);
              }
              return true;
            }
          )
          : msgAny.content;
        if (
          keptToolCalls.length === 0 &&
          Array.isArray(keptContent) &&
          keptContent.length === 0
        ) {
          continue;
        }
        strippedAiIndices.add(result.length);
        const patched = Object.create(
          Object.getPrototypeOf(msg),
          Object.getOwnPropertyDescriptors(msg)
        );
        patched.tool_calls = keptToolCalls.length > 0 ? keptToolCalls : [];
        patched.content = keptContent;
        onMessageCloned?.(msg, patched as BaseMessage);
        result.push(patched as BaseMessage);
        continue;
      }
    }

    result.push(msg);
  }

  // Bedrock/Anthropic require the conversation to end with a user message;
  // a stripped AI message (tool_use removed) represents a dead-end exchange.
  while (result.length > 0 && strippedAiIndices.has(result.length - 1)) {
    result.pop();
  }

  return result;
}

/**
 * Truncates an oversized tool_use `input` field using head+tail, preserving
 * it as a valid JSON object. Head gets ~70%, tail gets ~30% so the model
 * sees both the beginning (what was called) and end (closing structure/values).
 * Falls back to head-only when the budget is too small for a meaningful tail.
 */
function isIndexInContext(
  arrayA: unknown[],
  arrayB: unknown[],
  targetIndex: number
): boolean {
  const startingIndexInA = arrayA.length - arrayB.length;
  return targetIndex >= startingIndexInA;
}

function addThinkingBlock(
  message: AIMessage,
  thinkingBlock: ThinkingContentText | ReasoningContentText
): AIMessage {
  const content: MessageContentComplex[] = Array.isArray(message.content)
    ? (message.content as MessageContentComplex[])
    : [
      {
        type: ContentTypes.TEXT,
        text: message.content,
      },
    ];
  /** Edge case, the message already has the thinking block */
  if (content[0]?.type === thinkingBlock.type) {
    return message;
  }
  content.unshift(thinkingBlock);
  return new AIMessage({
    ...message,
    content: toLangChainContent(content),
  });
}

/**
 * Calculates the total tokens from a single usage object
 *
 * @param usage The usage metadata object containing token information
 * @returns An object containing the total input and output tokens
 */
export function calculateTotalTokens(
  usage: Partial<UsageMetadata>
): UsageMetadata {
  const baseInputTokens = Number(usage.input_tokens) || 0;
  const cacheCreation = Number(usage.input_token_details?.cache_creation) || 0;
  const cacheRead = Number(usage.input_token_details?.cache_read) || 0;
  const totalOutputTokens = Number(usage.output_tokens) || 0;
  const cacheSum = cacheCreation + cacheRead;
  // Anthropic: input_tokens excludes cache, cache_read can be much larger than input_tokens.
  // OpenAI: input_tokens includes cache, cache_read is always <= input_tokens.
  const cacheIsAdditive = cacheSum > 0 && cacheSum > baseInputTokens;
  const totalInputTokens = cacheIsAdditive
    ? baseInputTokens + cacheSum
    : baseInputTokens;

  return {
    input_tokens: totalInputTokens,
    output_tokens: totalOutputTokens,
    total_tokens: totalInputTokens + totalOutputTokens,
  };
}

export type PruningResult = {
  context: BaseMessage[];
  remainingContextTokens: number;
  messagesToRefine: BaseMessage[];
  thinkingStartIndex?: number;
};

/**
 * Locates a reasoning block in assistant content. Reasoning blocks carry
 * provider-specific `type` tags: Anthropic emits `thinking`, while Bedrock and
 * OpenAI-compatible reasoning providers (DeepSeek-R1, DashScope/Qwen-thinking)
 * emit `reasoning_content`. DeepSeek/Qwen route through the `THINKING` default
 * even though their blocks are `reasoning_content` and aren't normalized
 * upstream, so for the `THINKING` case we also accept `reasoning_content` — this
 * is what fixes issue #191.
 *
 * The broadening is intentionally one-directional. A Bedrock run
 * (`REASONING_CONTENT`) must NOT match an Anthropic `thinking` block: the
 * Bedrock input converter rejects `thinking` blocks outright
 * (`src/llm/bedrock/utils/message_inputs.ts`), so reattaching one to a
 * surviving message would make the request fail before it is sent.
 */
function findReasoningBlock(
  content: MessageContentComplex[],
  reasoningType: ContentTypes
): ThinkingContentText | ReasoningContentText | undefined {
  return content.find(
    (part) =>
      part.type === reasoningType ||
      (reasoningType === ContentTypes.THINKING &&
        part.type === ContentTypes.REASONING_CONTENT)
  ) as ThinkingContentText | ReasoningContentText | undefined;
}

/**
 * Processes an array of messages and returns a context of messages that fit within a specified token limit.
 * It iterates over the messages from newest to oldest, adding them to the context until the token limit is reached.
 *
 * @param options Configuration options for processing messages
 * @returns Object containing the message context, remaining tokens, messages not included, and summary index
 */
export function getMessagesWithinTokenLimit({
  messages: _messages,
  maxContextTokens,
  indexTokenCountMap,
  startType: _startType,
  thinkingEnabled,
  tokenCounter,
  thinkingStartIndex: _thinkingStartIndex = -1,
  reasoningType = ContentTypes.THINKING,
  instructionTokens: _instructionTokens = 0,
}: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  startType?: string | string[];
  thinkingEnabled?: boolean;
  tokenCounter: TokenCounter;
  thinkingStartIndex?: number;
  reasoningType?: ContentTypes.THINKING | ContentTypes.REASONING_CONTENT;
  /**
   * Token overhead for instructions (system message + tool schemas + summary)
   * that are NOT included in `messages`.  When messages[0] is already a
   * SystemMessage the budget is deducted from its indexTokenCountMap entry
   * as before; otherwise this value is subtracted from the available budget.
   */
  instructionTokens?: number;
}): PruningResult {
  // Every reply is primed with <|start|>assistant<|message|>, so we
  // start with 3 tokens for the label after all messages have been counted.
  let currentTokenCount = REPLY_PRIMER_TOKENS;
  const instructions =
    _messages[0]?.getType() === 'system' ? _messages[0] : undefined;
  const instructionsTokenCount =
    instructions != null ? (indexTokenCountMap[0] ?? 0) : _instructionTokens;
  const initialContextTokens = maxContextTokens - instructionsTokenCount;
  let remainingContextTokens = initialContextTokens;
  let startType = _startType;
  const originalLength = _messages.length;
  const messages = [..._messages];
  /**
   * IMPORTANT: this context array gets reversed at the end, since the latest messages get pushed first.
   *
   * This may be confusing to read, but it is done to ensure the context is in the correct order for the model.
   * */
  let context: Array<BaseMessage | undefined> = [];

  let thinkingStartIndex = _thinkingStartIndex;
  let thinkingEndIndex = -1;
  let thinkingBlock: ThinkingContentText | ReasoningContentText | undefined;
  const endIndex = instructions != null ? 1 : 0;
  const prunedMemory: BaseMessage[] = [];

  if (_thinkingStartIndex > -1) {
    const thinkingMessageContent = messages[_thinkingStartIndex]?.content;
    if (Array.isArray(thinkingMessageContent)) {
      thinkingBlock = findReasoningBlock(thinkingMessageContent, reasoningType);
    }
  }

  if (currentTokenCount < remainingContextTokens) {
    let currentIndex = messages.length;
    while (
      messages.length > 0 &&
      currentTokenCount < remainingContextTokens &&
      currentIndex > endIndex
    ) {
      currentIndex--;
      if (messages.length === 1 && instructions) {
        break;
      }
      const poppedMessage = messages.pop();
      if (!poppedMessage) continue;
      const messageType = poppedMessage.getType();
      if (
        thinkingEnabled === true &&
        thinkingEndIndex === -1 &&
        currentIndex === originalLength - 1 &&
        (messageType === 'ai' || messageType === 'tool')
      ) {
        thinkingEndIndex = currentIndex;
      }
      if (
        thinkingEndIndex > -1 &&
        !thinkingBlock &&
        thinkingStartIndex < 0 &&
        messageType === 'ai' &&
        Array.isArray(poppedMessage.content)
      ) {
        thinkingBlock = findReasoningBlock(
          poppedMessage.content,
          reasoningType
        );
        thinkingStartIndex = thinkingBlock != null ? currentIndex : -1;
      }
      /**
       * Exited the trailing assistant/tool sequence without finding a
       * thinking block. Anthropic does not require Claude to emit a
       * thinking block before every tool call, so the absence of one is
       * a valid sequence — clear thinkingEndIndex so the pruner does not
       * treat it as malformed.
       */
      if (
        thinkingEndIndex > -1 &&
        thinkingStartIndex < 0 &&
        !thinkingBlock &&
        messageType !== 'ai' &&
        messageType !== 'tool'
      ) {
        thinkingEndIndex = -1;
      }

      const tokenCount = indexTokenCountMap[currentIndex] ?? 0;

      if (
        prunedMemory.length === 0 &&
        currentTokenCount + tokenCount <= remainingContextTokens
      ) {
        context.push(poppedMessage);
        currentTokenCount += tokenCount;
      } else {
        prunedMemory.push(poppedMessage);
        if (thinkingEndIndex > -1 && thinkingStartIndex < 0) {
          continue;
        }
        break;
      }
    }

    if (context[context.length - 1]?.getType() === 'tool') {
      startType = ['ai', 'human'];
    }

    if (startType != null && startType.length > 0 && context.length > 0) {
      let requiredTypeIndex = -1;

      let totalTokens = 0;
      for (let i = context.length - 1; i >= 0; i--) {
        const currentType = context[i]?.getType() ?? '';
        if (
          Array.isArray(startType)
            ? startType.includes(currentType)
            : currentType === startType
        ) {
          requiredTypeIndex = i + 1;
          break;
        }
        const originalIndex = originalLength - 1 - i;
        totalTokens += indexTokenCountMap[originalIndex] ?? 0;
      }

      if (requiredTypeIndex > 0) {
        currentTokenCount -= totalTokens;
        context = context.slice(0, requiredTypeIndex);
      }
    }
  }

  if (instructions && originalLength > 0) {
    context.push(_messages[0] as BaseMessage);
    messages.shift();
  }

  // The backward iteration pushed messages in reverse chronological order
  // (newest first).  Restore correct chronological order before prepending
  // the remaining (older) messages so that messagesToRefine is always
  // ordered oldest → newest.  Without this, callers that rely on
  // messagesToRefine order (e.g. the summarization node extracting the
  // latest turn) would see tool_use/tool_result pairs in the wrong order.
  prunedMemory.reverse();

  if (messages.length > 0) {
    prependItems(prunedMemory, messages);
  }

  remainingContextTokens -= currentTokenCount;
  const result: PruningResult = {
    remainingContextTokens,
    context: [] as BaseMessage[],
    messagesToRefine: prunedMemory,
  };

  if (thinkingStartIndex > -1) {
    result.thinkingStartIndex = thinkingStartIndex;
  }

  if (
    prunedMemory.length === 0 ||
    thinkingEndIndex < 0 ||
    (thinkingStartIndex > -1 &&
      isIndexInContext(_messages, context, thinkingStartIndex))
  ) {
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  /**
   * A trailing reasoning sequence was detected but its block could not be
   * located in the surviving context. Rather than throw — which permanently
   * bricks the conversation, re-firing on every retry of the same thread (see
   * issue #191) — return the partially-pruned context and let the provider
   * surface a real, recoverable error if the payload is genuinely malformed.
   * Strict providers (Anthropic) reject it cleanly; lenient ones (DeepSeek,
   * Qwen) proceed. The pruner cannot know which applies, so it must not be the
   * one to make the failure fatal.
   */
  if ((thinkingEndIndex > -1 && thinkingStartIndex < 0) || !thinkingBlock) {
    /**
     * No block was located, so any `thinkingStartIndex` set above came from a
     * stale carried-over index pointing at a block-less message. Drop it:
     * `createPruneMessages` persists the returned index as
     * `runThinkingStartIndex`, and a stale value would suppress the trailing
     * scan (`thinkingStartIndex < 0`) on later turns, causing a real reasoning
     * block to be missed and never reattached.
     */
    delete result.thinkingStartIndex;
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  let assistantIndex = -1;
  for (let i = 0; i < context.length; i++) {
    const currentMessage = context[i];
    const type = currentMessage?.getType();
    if (type === 'ai') {
      assistantIndex = i;
    }
    if (assistantIndex > -1 && (type === 'human' || type === 'system')) {
      break;
    }
  }

  if (assistantIndex === -1) {
    // No AI messages survived pruning — skip thinking block reattachment.
    // The caller handles empty/insufficient context via overflow recovery.
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  thinkingStartIndex = originalLength - 1 - assistantIndex;
  const thinkingTokenCount = tokenCounter(
    new AIMessage({ content: toLangChainContent([thinkingBlock]) })
  );
  const newRemainingCount = remainingContextTokens - thinkingTokenCount;
  const newMessage = addThinkingBlock(
    context[assistantIndex] as AIMessage,
    thinkingBlock
  );
  context[assistantIndex] = newMessage;
  if (newRemainingCount > 0) {
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  const thinkingMessage: AIMessage = context[assistantIndex] as AIMessage;
  const newThinkingMessageTokenCount =
    (indexTokenCountMap[thinkingStartIndex] ?? 0) + thinkingTokenCount;
  remainingContextTokens = initialContextTokens - newThinkingMessageTokenCount;
  currentTokenCount = REPLY_PRIMER_TOKENS;
  let newContext: BaseMessage[] = [];
  const secondRoundMessages = [..._messages];
  let currentIndex = secondRoundMessages.length;
  while (
    secondRoundMessages.length > 0 &&
    currentTokenCount < remainingContextTokens &&
    currentIndex > thinkingStartIndex
  ) {
    currentIndex--;
    const poppedMessage = secondRoundMessages.pop();
    if (!poppedMessage) continue;
    const tokenCount = indexTokenCountMap[currentIndex] ?? 0;
    if (currentTokenCount + tokenCount <= remainingContextTokens) {
      newContext.push(poppedMessage);
      currentTokenCount += tokenCount;
    } else {
      messages.push(poppedMessage);
      break;
    }
  }

  const firstMessage = newContext[newContext.length - 1];
  const firstMessageType = newContext[newContext.length - 1].getType();
  if (firstMessageType === 'tool') {
    startType = ['ai', 'human'];
  }

  if (startType != null && startType.length > 0 && newContext.length > 0) {
    let requiredTypeIndex = -1;

    for (let i = newContext.length - 1; i >= 0; i--) {
      const currentType = newContext[i]?.getType() ?? '';
      if (
        Array.isArray(startType)
          ? startType.includes(currentType)
          : currentType === startType
      ) {
        requiredTypeIndex = i + 1;
        break;
      }
    }

    if (requiredTypeIndex > 0) {
      newContext = newContext.slice(0, requiredTypeIndex);
    }
  }

  if (firstMessageType === 'ai') {
    const newMessage = addThinkingBlock(
      firstMessage as AIMessage,
      thinkingBlock
    );
    newContext[newContext.length - 1] = newMessage;
  } else {
    newContext.push(thinkingMessage);
  }

  if (instructions && originalLength > 0) {
    newContext.push(_messages[0] as BaseMessage);
    secondRoundMessages.shift();
  }

  result.context = newContext.reverse();
  return result;
}

export function checkValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && value > 0;
}

type FadingApplyParams = {
  canonicalMessages?: BaseMessage[];
  messages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
  caps: Pick<FadingCaps, 'resultChars' | 'consumedChars' | 'inputChars'>;
  /** Whether consumed results shrink to `caps.consumedChars`. */
  masked: boolean;
  /** First index to visit for fresh results and tool-call inputs. */
  fromIndex?: number;
  /** First consumed index to visit for masking. */
  maskedFromIndex?: number;
  /** Original (pre-masking) content keyed by message index, captured for the summarizer. */
  originalContentStore?: Map<number, string>;
  /** Called after storing a newly captured entry. */
  onContentStored?: (index: number, content: string) => void;
};

export type FadingApplyResult = {
  /** Fresh tool results rewritten. */
  truncated: number;
  /** Tool-call inputs rewritten. */
  inputs: number;
  /** Consumed tool results rewritten. */
  masked: number;
  /** Index of the newest AI message with text; tool results before it are consumed. */
  consumedBoundary: number;
};

/**
 * Index of the newest AI message with substantive text. Every ToolMessage
 * before it has been answered by the model ("consumed"); results after it are
 * still fresh. Scans backward, so the cost is bounded by the last turn.
 */
function findConsumedBoundary(messages: BaseMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.getType() === 'ai' && hasNonEmptyTextContent(message.content)) {
      return i;
    }
  }
  return 0;
}

/**
 * Applies a fading tier's caps in one forward pass. Consumed results (before
 * the boundary) shrink to `consumedChars`, fresh results to `resultChars` and
 * historical tool-call inputs to `inputChars`. Messages already within their
 * cap keep object identity and token count, so at an unchanged tier the pass
 * only touches what arrived since the watermarks. Truncation is a pure
 * function of (content, cap), which is what keeps the bytes of a historical
 * result identical from call to call.
 */
export function applyFadingCaps(params: FadingApplyParams): FadingApplyResult {
  const { messages, indexTokenCountMap, tokenCounter, caps, masked } = params;
  const fromIndex = params.fromIndex ?? 0;
  const maskedFromIndex = params.maskedFromIndex ?? 0;
  const consumedBoundary = masked ? findConsumedBoundary(messages) : 0;
  const start = masked ? Math.min(fromIndex, maskedFromIndex) : fromIndex;
  let truncated = 0;
  let maskedCount = 0;

  for (let i = start; i < messages.length; i++) {
    const message = messages[i];
    if (message.getType() !== 'tool' || isComputerCallOutputMessage(message)) {
      continue;
    }
    const consumed = masked && i < consumedBoundary;
    if (consumed ? i < maskedFromIndex : i < fromIndex) {
      continue;
    }
    const maxChars = consumed ? caps.consumedChars : caps.resultChars;
    if (!Number.isFinite(maxChars)) {
      continue;
    }
    const canonicalContent =
      params.canonicalMessages?.[i]?.content ?? message.content;
    const compacted = compactToolContent(canonicalContent, maxChars);
    if (!compacted.changed) {
      continue;
    }
    if (
      consumed &&
      params.originalContentStore &&
      !params.originalContentStore.has(i)
    ) {
      const original = serializeToolContentBounded(
        canonicalContent,
        ORIGINAL_CONTENT_MAX_CHARS
      );
      params.originalContentStore.set(i, original);
      params.onContentStored?.(i, original);
    }
    const cloned = cloneToolMessageWithContent(
      message as ToolMessage,
      compacted.content
    );
    messages[i] = cloned;
    indexTokenCountMap[i] = tokenCounter(cloned);
    if (consumed) {
      maskedCount++;
    } else {
      truncated++;
    }
  }

  const inputs = Number.isFinite(caps.inputChars)
    ? applyToolCallInputCaps({
      messages,
      canonicalMessages: params.canonicalMessages,
      maxInputChars: caps.inputChars,
      indexTokenCountMap,
      tokenCounter,
      fromIndex,
    })
    : 0;

  return { truncated, inputs, masked: maskedCount, consumedBoundary };
}

/**
 * Observation masking: replaces consumed ToolMessage content with tight
 * head+tail truncations that serve as informative placeholders. Fresh results
 * and tool-call inputs are left alone.
 *
 * @returns The number of tool messages that were masked.
 */
export function maskConsumedToolResults(params: {
  messages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
  /** Character cap applied to every consumed result (never below
   *  MASKED_RESULT_MIN_CHARS, which is also the default). */
  maxChars?: number;
  /** @deprecated Aggregate raw-token budget distributed by recency. Prefer
   *  `maxChars` for byte-stable masking across otherwise identical calls. */
  availableRawBudget?: number;
  /** When provided, original (pre-masking) content is stored here keyed by
   *  message index — only for entries that actually get truncated. */
  originalContentStore?: Map<number, string>;
  /** Called after storing a newly captured entry. */
  onContentStored?: (index: number, content: string) => void;
}): number {
  if (
    params.maxChars == null &&
    params.availableRawBudget != null &&
    params.availableRawBudget > 0
  ) {
    const consumedBoundary = findConsumedBoundary(params.messages);
    const consumedIndices: number[] = [];
    for (let i = 0; i < consumedBoundary; i++) {
      if (params.messages[i].getType() === 'tool') {
        consumedIndices.push(i);
      }
    }
    const count = consumedIndices.length;
    const totalBudgetChars = params.availableRawBudget * 4;
    let masked = 0;
    for (let c = 0; c < count; c++) {
      const i = consumedIndices[c];
      const message = params.messages[i];
      if (isComputerCallOutputMessage(message)) {
        continue;
      }
      const position = count > 1 ? c / (count - 1) : 1;
      const weight = 0.2 + 0.8 * position;
      const totalWeight = count > 1 ? 0.6 * count : 1;
      const maxChars = Math.max(
        MASKED_RESULT_MIN_CHARS,
        Math.floor((weight / totalWeight) * totalBudgetChars)
      );
      const compacted = compactToolContent(message.content, maxChars);
      if (!compacted.changed) {
        continue;
      }
      if (
        params.originalContentStore != null &&
        !params.originalContentStore.has(i)
      ) {
        const original = serializeToolContentBounded(
          message.content,
          ORIGINAL_CONTENT_MAX_CHARS
        );
        params.originalContentStore.set(i, original);
        params.onContentStored?.(i, original);
      }
      const cloned = cloneToolMessageWithContent(
        message as ToolMessage,
        compacted.content
      );
      params.messages[i] = cloned;
      params.indexTokenCountMap[i] = params.tokenCounter(cloned);
      masked++;
    }
    return masked;
  }
  return applyFadingCaps({
    messages: params.messages,
    indexTokenCountMap: params.indexTokenCountMap,
    tokenCounter: params.tokenCounter,
    caps: {
      resultChars: Number.POSITIVE_INFINITY,
      consumedChars: Math.max(
        MASKED_RESULT_MIN_CHARS,
        Math.floor(params.maxChars ?? MASKED_RESULT_MIN_CHARS)
      ),
      inputChars: Number.POSITIVE_INFINITY,
    },
    masked: true,
    originalContentStore: params.originalContentStore,
    onContentStored: params.onContentStored,
  }).masked;
}

/**
 * Pre-flight truncation: truncates oversized ToolMessage content before the
 * main backward-iteration pruning runs, applying one cap derived from
 * `maxContextTokens` to every tool result.
 *
 * @returns The number of tool messages that were truncated.
 */
export function preFlightTruncateToolResults(params: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
}): number {
  return applyFadingCaps({
    messages: params.messages,
    indexTokenCountMap: params.indexTokenCountMap,
    tokenCounter: params.tokenCounter,
    caps: {
      resultChars: calculateMaxToolResultChars(params.maxContextTokens),
      consumedChars: Number.POSITIVE_INFINITY,
      inputChars: Number.POSITIVE_INFINITY,
    },
    masked: false,
  }).truncated;
}

/**
 * Pre-flight truncation: truncates oversized `tool_use` input fields in AI messages.
 *
 * Tool call inputs (arguments) can be very large — e.g., code evaluation payloads from
 * MCP tools like chrome-devtools. Since these tool calls have already been executed,
 * the model only needs a summary of what was called, not the full arguments. Truncating
 * them before pruning can prevent entire messages from being dropped.
 *
 * Uses 15% of the context window (in estimated characters, ~4 chars/token) as the
 * per-input cap, capped at 200K chars.
 *
 * @returns The number of AI messages that had tool_use inputs truncated.
 */
const ACCESSOR_INPUT_PLACEHOLDER = '[Property accessor omitted]';

type ToolInputProjection = {
  value: unknown;
  changed: boolean;
};

type PropertyRead = {
  found: boolean;
  own: boolean;
  accessor: boolean;
  value?: unknown;
};

function normalizeToolInputLimit(maxChars: number): number {
  if (!Number.isFinite(maxChars)) {
    return HARD_MAX_TOOL_CALL_INPUT_CHARS;
  }
  return Math.max(MIN_JSON_VALUE_CHARS, Math.floor(maxChars));
}

function readPropertyWithoutAccessors(
  value: object,
  key: PropertyKey
): PropertyRead {
  let current: object | null = value;
  for (let depth = 0; current != null && depth < 100; depth++) {
    if (isProxy(current)) {
      return { found: true, own: current === value, accessor: true };
    }
    let descriptor: PropertyDescriptor | undefined;
    try {
      descriptor = Object.getOwnPropertyDescriptor(current, key);
    } catch {
      return { found: true, own: current === value, accessor: true };
    }
    if (descriptor != null) {
      if (!('value' in descriptor)) {
        return { found: true, own: current === value, accessor: true };
      }
      return {
        found: true,
        own: current === value,
        accessor: false,
        value: descriptor.value,
      };
    }
    try {
      current = Object.getPrototypeOf(current) as object | null;
    } catch {
      return { found: true, own: false, accessor: true };
    }
  }
  return current == null
    ? { found: false, own: false, accessor: false }
    : { found: true, own: false, accessor: true };
}

function cloneWithProjectedProperties<T extends object>(
  value: T,
  changes: Readonly<Record<string, unknown>>
): T {
  const descriptors = Object.getOwnPropertyDescriptors(value) as Record<
    string,
    PropertyDescriptor | undefined
  >;
  for (const [key, projectedValue] of Object.entries(changes)) {
    const descriptor = descriptors[key];
    descriptors[key] = {
      configurable: descriptor?.configurable ?? true,
      enumerable: descriptor?.enumerable ?? true,
      value: projectedValue,
      writable: descriptor?.writable ?? true,
    };
  }
  return Object.create(
    Object.getPrototypeOf(value),
    descriptors as PropertyDescriptorMap
  ) as T;
}

function cloneAIMessageWithProjectedStreamContent(
  message: AIMessage | AIMessageChunk,
  changes: Readonly<Record<string, unknown>>,
  streamContent: AIMessage['content']
): AIMessage | AIMessageChunk {
  const descriptors = Object.getOwnPropertyDescriptors(message) as Record<
    string,
    PropertyDescriptor | undefined
  >;
  for (const [key, projectedValue] of Object.entries(changes)) {
    const descriptor = descriptors[key];
    descriptors[key] = {
      configurable: descriptor?.configurable ?? true,
      enumerable: descriptor?.enumerable ?? true,
      value: projectedValue,
      writable: descriptor?.writable ?? true,
    };
  }
  const lcKwargs = descriptors.lc_kwargs;
  if (
    lcKwargs != null &&
    'value' in lcKwargs &&
    typeof lcKwargs.value === 'object' &&
    lcKwargs.value != null
  ) {
    descriptors.lc_kwargs = {
      ...lcKwargs,
      value: {
        ...(lcKwargs.value as Record<string, unknown>),
        ...(message.response_metadata.output_version === 'v1'
          ? { content: undefined, contentBlocks: streamContent }
          : { content: streamContent }),
      },
    };
  }
  return Object.create(
    Object.getPrototypeOf(message),
    descriptors as PropertyDescriptorMap
  ) as AIMessage | AIMessageChunk;
}

/** Marker that led the preview in the legacy `{_truncated, _originalChars}` envelope. */
const LEGACY_TOOL_INPUT_TRUNCATION_MARKER = '… [truncated]\n';

/**
 * Leads every shortened tool-call input. Fading only shortens calls already in
 * history, which ran with their full input; a model that reads a shortened copy
 * as a call that was cut off re-issues it, and a side-effecting tool (sending an
 * email) then runs again. The note says what actually happened.
 */
export const TOOL_INPUT_ELISION_NOTE =
  'Completed call; input shortened to save context.';

function toolInputElision(
  originalChars: number,
  inputPrefix: string
): { _note: string; _originalChars: number; _inputPrefix: string } {
  return {
    _note: TOOL_INPUT_ELISION_NOTE,
    _originalChars: originalChars,
    _inputPrefix: inputPrefix,
  };
}

function createBoundedTruncationValue(
  preview: string,
  originalChars: number,
  maxChars: number
): unknown {
  const normalizedMaxChars = normalizeToolInputLimit(maxChars);
  const canonicalPrefix = preview.startsWith(
    LEGACY_TOOL_INPUT_TRUNCATION_MARKER
  )
    ? preview.slice(LEGACY_TOOL_INPUT_TRUNCATION_MARKER.length)
    : preview;
  const emptyEnvelope = toolInputElision(originalChars, '');
  if (JSON.stringify(emptyEnvelope).length > normalizedMaxChars) {
    /**
     * Even the empty envelope overflows the cap, so no preview survives —
     * but the result must still be a JSON OBJECT, never `null`. This value
     * replaces a `tool_use.input` / tool-call `args` on messages that are
     * mutated IN PLACE into graph state (`preFlightTruncateToolCallInputs`),
     * and Anthropic rejects a replayed non-object input with a 400
     * (`tool_use.input: Input should be an object`). Observed live: a tight
     * summarization budget shrank the cap below the envelope, nulled a
     * retained calculator call's input and args, and the next model call
     * failed on replay.
     */
    return {};
  }

  let low = 0;
  let high = Math.min(canonicalPrefix.length, normalizedMaxChars);
  while (low < high) {
    const next = Math.ceil((low + high) / 2);
    const candidate = toolInputElision(
      originalChars,
      sliceWithoutSplittingSurrogates(canonicalPrefix, 0, next)
    );
    if (JSON.stringify(candidate).length <= normalizedMaxChars) {
      low = next;
    } else {
      high = next - 1;
    }
  }
  // The prefix stays a pure canonical prefix, so another, slightly smaller cap
  // can be derived from it without nesting the envelope.
  return toolInputElision(
    originalChars,
    sliceWithoutSplittingSurrogates(canonicalPrefix, 0, low)
  );
}

const ELISION_ENVELOPE_KEYS = ['_note', '_originalChars', '_inputPrefix'];
const LEGACY_ENVELOPE_KEYS = ['_truncated', '_originalChars'];

function hasExactKeys(keys: string[], expected: string[]): boolean {
  return (
    keys.length === expected.length &&
    expected.every((key) => keys.includes(key))
  );
}

/** Reads a shortened-input envelope, including the legacy `{_truncated,
 *  _originalChars}` shape persisted before the elision note existed. */
function readBoundedTruncationValue(
  input: unknown
): { preview: string; originalChars: number; legacy: boolean } | undefined {
  if (input == null || typeof input !== 'object' || isProxy(input)) {
    return undefined;
  }
  let previewKey: '_inputPrefix' | '_truncated';
  try {
    const prototype = Object.getPrototypeOf(input);
    const keys = Object.keys(input);
    if (prototype !== Object.prototype && prototype !== null) {
      return undefined;
    }
    if (hasExactKeys(keys, ELISION_ENVELOPE_KEYS)) {
      const note = readPropertyWithoutAccessors(input, '_note');
      if (
        !note.own ||
        note.accessor ||
        note.value !== TOOL_INPUT_ELISION_NOTE
      ) {
        return undefined;
      }
      previewKey = '_inputPrefix';
    } else if (hasExactKeys(keys, LEGACY_ENVELOPE_KEYS)) {
      previewKey = '_truncated';
    } else {
      return undefined;
    }
  } catch {
    return undefined;
  }
  const preview = readPropertyWithoutAccessors(input, previewKey);
  const originalChars = readPropertyWithoutAccessors(input, '_originalChars');
  return preview.own &&
    !preview.accessor &&
    typeof preview.value === 'string' &&
    originalChars.own &&
    !originalChars.accessor &&
    typeof originalChars.value === 'number' &&
    Number.isFinite(originalChars.value) &&
    originalChars.value >= 0
    ? {
      preview: preview.value,
      originalChars: originalChars.value,
      legacy: previewKey === '_truncated',
    }
    : undefined;
}

function projectToolInputWithinLimit(
  input: unknown,
  maxChars: number
): ToolInputProjection {
  const normalizedMaxChars = normalizeToolInputLimit(maxChars);
  const priorTruncation = readBoundedTruncationValue(input);
  if (priorTruncation != null) {
    const serializedLength = serializeStructuredValueBounded(
      input,
      normalizedMaxChars
    );
    /** A legacy envelope converts even when it fits: its `_truncated` wording
     *  is what led models to re-issue completed calls. */
    if (!serializedLength.truncated && !priorTruncation.legacy) {
      return { value: input, changed: false };
    }
    return {
      value: createBoundedTruncationValue(
        priorTruncation.preview,
        priorTruncation.originalChars,
        normalizedMaxChars
      ),
      changed: true,
    };
  }
  const serialized = serializeStructuredValueBounded(
    input,
    normalizedMaxChars,
    normalizedMaxChars
  );
  if (serialized.truncated) {
    return {
      value: createBoundedTruncationValue(
        serialized.prefix,
        serialized.originalChars,
        normalizedMaxChars
      ),
      changed: true,
    };
  }
  if (!hasUnsafeStructuredSerialization(input)) {
    return { value: input, changed: false };
  }

  try {
    return { value: JSON.parse(serialized.content), changed: true };
  } catch {
    return {
      value: createBoundedTruncationValue(
        serialized.content,
        serialized.originalChars,
        normalizedMaxChars
      ),
      changed: true,
    };
  }
}

/**
 * Serializes one structured tool-call input as valid, bounded JSON without
 * invoking user-defined accessors or `toJSON`.
 */
export function serializeToolCallInput(
  input: unknown,
  maxChars = HARD_MAX_TOOL_CALL_INPUT_CHARS
): string {
  const normalizedMaxChars = normalizeToolInputLimit(maxChars);
  const projected = projectToolInputWithinLimit(input, normalizedMaxChars);
  const serialized = serializeStructuredValueBounded(
    projected.value,
    normalizedMaxChars,
    normalizedMaxChars
  );
  if (!serialized.truncated) {
    return serialized.content === 'undefined' ? 'null' : serialized.content;
  }
  const fallback = createBoundedTruncationValue(
    serialized.prefix,
    serialized.originalChars,
    normalizedMaxChars
  );
  return JSON.stringify(fallback);
}

function projectKnownInputProperty(
  value: object,
  key: string,
  maxChars: number,
  changes: Record<string, unknown>
): void {
  const property = readPropertyWithoutAccessors(value, key);
  if (!property.found) {
    return;
  }
  const projected = property.accessor
    ? { value: ACCESSOR_INPUT_PLACEHOLDER, changed: true }
    : projectToolInputWithinLimit(property.value, maxChars);
  if (projected.changed || !property.own) {
    changes[key] = projected.value;
  }
}

function getStringProperty(value: object, key: string): string | undefined {
  const property = readPropertyWithoutAccessors(value, key);
  return !property.accessor && typeof property.value === 'string'
    ? property.value
    : undefined;
}

function projectInlineToolInput(
  block: MessageContentComplex,
  maxChars: number
): MessageContentComplex {
  const type = getStringProperty(block, 'type');
  if (type !== 'tool_use' && type !== 'tool_call') {
    return block;
  }

  const changes: Record<string, unknown> = {};
  projectKnownInputProperty(block, 'input', maxChars, changes);
  projectKnownInputProperty(block, 'args', maxChars, changes);

  const nestedProperty = readPropertyWithoutAccessors(block, 'tool_call');
  if (
    nestedProperty.found &&
    !nestedProperty.accessor &&
    nestedProperty.value != null &&
    typeof nestedProperty.value === 'object'
  ) {
    const nestedChanges: Record<string, unknown> = {};
    projectKnownInputProperty(
      nestedProperty.value,
      'args',
      maxChars,
      nestedChanges
    );
    if (Object.keys(nestedChanges).length > 0) {
      changes.tool_call = isProxy(nestedProperty.value)
        ? { args: ACCESSOR_INPUT_PLACEHOLDER }
        : cloneWithProjectedProperties(nestedProperty.value, nestedChanges);
    } else if (!nestedProperty.own) {
      changes.tool_call = nestedProperty.value;
    }
  } else if (nestedProperty.accessor) {
    changes.tool_call = { args: ACCESSOR_INPUT_PLACEHOLDER };
  }

  return Object.keys(changes).length > 0
    ? cloneWithProjectedProperties(block, changes)
    : block;
}

function projectSerializedArguments(
  value: unknown,
  maxChars: number
): { value: string; changed: boolean } {
  const normalizedMaxChars = normalizeToolInputLimit(maxChars);
  if (
    typeof value === 'string' &&
    value.includes('"_originalChars"') &&
    (value.includes('"_inputPrefix"') || value.includes('"_truncated"'))
  ) {
    try {
      const priorTruncation = readBoundedTruncationValue(JSON.parse(value));
      if (priorTruncation != null) {
        if (!priorTruncation.legacy && value.length <= normalizedMaxChars) {
          return { value, changed: false };
        }
        return {
          value: JSON.stringify(
            createBoundedTruncationValue(
              priorTruncation.preview,
              priorTruncation.originalChars,
              normalizedMaxChars
            )
          ),
          changed: true,
        };
      }
    } catch {
      // Fall through to the accessor-safe serializer for malformed JSON.
    }
  }
  if (typeof value === 'string' && value.length <= normalizedMaxChars) {
    return { value, changed: false };
  }
  return {
    value: serializeToolCallInput(value, normalizedMaxChars),
    changed: true,
  };
}

const TRUNCATED_STRING_INPUT_PATTERN =
  /\n… \[(?:truncated|shortened; call completed): (\d+) chars\]$/u;
const LEGACY_TRUNCATED_STRING_INPUT_PATTERN = /\n… \[truncated: \d+ chars\]$/u;

function projectStringInputWithinLimit(
  value: string,
  maxChars: number
): { value: string; changed: boolean } {
  const normalizedMaxChars = normalizeToolInputLimit(maxChars);
  if (
    value.length <= normalizedMaxChars &&
    !LEGACY_TRUNCATED_STRING_INPUT_PATTERN.test(value)
  ) {
    return { value, changed: false };
  }
  const match = TRUNCATED_STRING_INPUT_PATTERN.exec(value);
  const originalChars = match == null ? value.length : Number(match[1]);
  const prefix = match == null ? value : value.slice(0, match.index);
  const marker = `\n… [shortened; call completed: ${originalChars} chars]`;
  return {
    value:
      marker.length >= normalizedMaxChars
        ? sliceWithoutSplittingSurrogates(prefix, 0, normalizedMaxChars)
        : sliceWithoutSplittingSurrogates(
          prefix,
          0,
          normalizedMaxChars - marker.length
        ) + marker,
    changed: true,
  };
}

function selectProjectedSerializedArguments(
  property: PropertyRead,
  canonical: string | undefined,
  maxChars: number
): { value: string; changed: boolean } {
  if (canonical != null) {
    return { value: canonical, changed: property.value !== canonical };
  }
  if (property.accessor) {
    return {
      value: serializeToolCallInput(ACCESSOR_INPUT_PLACEHOLDER, maxChars),
      changed: true,
    };
  }
  return projectSerializedArguments(property.value, maxChars);
}

function projectRawOpenAIToolCalls(
  rawToolCalls: unknown,
  maxChars: number,
  canonicalArguments: ReadonlyMap<string, string>
): { value: unknown; changed: boolean } {
  if (!Array.isArray(rawToolCalls)) {
    return { value: rawToolCalls, changed: false };
  }

  const state = { changed: false };
  const projected = rawToolCalls.map((rawToolCall) => {
    if (rawToolCall == null || typeof rawToolCall !== 'object') {
      return rawToolCall;
    }
    const functionProperty = readPropertyWithoutAccessors(
      rawToolCall,
      'function'
    );
    if (
      !functionProperty.found ||
      functionProperty.accessor ||
      functionProperty.value == null ||
      typeof functionProperty.value !== 'object'
    ) {
      return rawToolCall;
    }

    const argsProperty = readPropertyWithoutAccessors(
      functionProperty.value,
      'arguments'
    );
    if (!argsProperty.found) {
      return rawToolCall;
    }
    const callId = getStringProperty(rawToolCall, 'id');
    const canonical =
      callId != null ? canonicalArguments.get(callId) : undefined;
    const projectedArgs = selectProjectedSerializedArguments(
      argsProperty,
      canonical,
      maxChars
    );
    if (!projectedArgs.changed && functionProperty.own && argsProperty.own) {
      return rawToolCall;
    }

    state.changed = true;
    const projectedFunction = cloneWithProjectedProperties(
      functionProperty.value,
      { arguments: projectedArgs.value }
    );
    return cloneWithProjectedProperties(rawToolCall, {
      function: projectedFunction,
    });
  });
  return {
    value: state.changed ? projected : rawToolCalls,
    changed: state.changed,
  };
}

function projectLegacyFunctionCall(
  property: PropertyRead,
  maxChars: number
): { value: unknown; changed: boolean } {
  if (!property.found) {
    return { value: property.value, changed: false };
  }
  if (property.own && !property.accessor && property.value === undefined) {
    return { value: undefined, changed: false };
  }
  if (
    property.accessor ||
    !property.own ||
    property.value == null ||
    typeof property.value !== 'object'
  ) {
    return { value: undefined, changed: true };
  }

  const nameProperty = readPropertyWithoutAccessors(property.value, 'name');
  const argsProperty = readPropertyWithoutAccessors(
    property.value,
    'arguments'
  );
  if (
    nameProperty.accessor ||
    !nameProperty.own ||
    typeof nameProperty.value !== 'string' ||
    nameProperty.value === '' ||
    nameProperty.value.length > normalizeToolInputLimit(maxChars) ||
    !argsProperty.found ||
    argsProperty.accessor ||
    !argsProperty.own
  ) {
    return { value: undefined, changed: true };
  }
  try {
    const prototype = Object.getPrototypeOf(property.value);
    const enumerableKeys = Object.keys(property.value);
    if (
      (prototype === Object.prototype || prototype === null) &&
      enumerableKeys.length === 2 &&
      enumerableKeys.includes('name') &&
      enumerableKeys.includes('arguments') &&
      typeof argsProperty.value === 'string' &&
      argsProperty.value.length <= normalizeToolInputLimit(maxChars) &&
      !hasUnsafeStructuredSerialization(property.value)
    ) {
      return { value: property.value, changed: false };
    }
  } catch {
    return { value: undefined, changed: true };
  }
  const projectedArgs = selectProjectedSerializedArguments(
    argsProperty,
    undefined,
    maxChars
  );
  return {
    value: {
      name: nameProperty.value,
      arguments: projectedArgs.value,
    },
    changed: true,
  };
}

function projectResponsesOutput(
  output: unknown,
  maxChars: number,
  canonicalArguments: ReadonlyMap<string, string>,
  canonicalInputs: ReadonlyMap<string, unknown>
): { value: unknown; changed: boolean } {
  if (!Array.isArray(output)) {
    return { value: output, changed: false };
  }

  const state = { changed: false };
  const projected = output.map((item) => {
    if (item == null || typeof item !== 'object') {
      return item;
    }
    const type = getStringProperty(item, 'type');
    if (
      type !== 'function_call' &&
      type !== 'custom_tool_call' &&
      type !== 'computer_call'
    ) {
      return item;
    }
    let inputKey = 'action';
    if (type === 'function_call') {
      inputKey = 'arguments';
    } else if (type === 'custom_tool_call') {
      inputKey = 'input';
    }
    const inputProperty = readPropertyWithoutAccessors(item, inputKey);
    if (!inputProperty.found) {
      return item;
    }
    const callId =
      getStringProperty(item, 'call_id') ?? getStringProperty(item, 'id');
    let projectedInput: ToolInputProjection;
    if (type === 'function_call') {
      const canonical =
        callId != null ? canonicalArguments.get(callId) : undefined;
      projectedInput = selectProjectedSerializedArguments(
        inputProperty,
        canonical,
        maxChars
      );
    } else {
      const canonicalArgs =
        callId != null ? canonicalInputs.get(callId) : undefined;
      const canonicalProperty =
        canonicalArgs != null && typeof canonicalArgs === 'object'
          ? readPropertyWithoutAccessors(canonicalArgs, inputKey)
          : undefined;
      let source = inputProperty.accessor
        ? ACCESSOR_INPUT_PLACEHOLDER
        : inputProperty.value;
      if (canonicalProperty != null && canonicalProperty.found) {
        source = canonicalProperty.accessor
          ? ACCESSOR_INPUT_PLACEHOLDER
          : canonicalProperty.value;
      }
      if (type === 'custom_tool_call') {
        /** The string projection keeps fitting input as-is, except a legacy
         *  `[truncated: N chars]` marker, which it rewrites. */
        const value =
          typeof source === 'string'
            ? projectStringInputWithinLimit(source, maxChars).value
            : serializeToolCallInput(source, maxChars);
        projectedInput = {
          value,
          changed: !inputProperty.own || inputProperty.value !== value,
        };
      } else {
        const projected = projectToolInputWithinLimit(source, maxChars);
        projectedInput = {
          value: projected.value,
          changed:
            projected.changed ||
            !inputProperty.own ||
            inputProperty.value !== projected.value,
        };
      }
    }
    if (!projectedInput.changed && inputProperty.own) {
      return item;
    }
    state.changed = true;
    return cloneWithProjectedProperties(item, {
      [inputKey]: projectedInput.value,
    });
  });
  return {
    value: state.changed ? projected : output,
    changed: state.changed,
  };
}

function projectToolCallInputsInternal(
  messages: BaseMessage[],
  maxInputChars: number,
  dropIncompleteStreamContent: boolean,
  fromIndex = 0,
  retainRawOpenAIToolCalls = true
): BaseMessage[] {
  const normalizedMaxInputChars = normalizeToolInputLimit(maxInputChars);
  let projectedMessages: BaseMessage[] | undefined;
  for (let i = fromIndex; i < messages.length; i++) {
    const message = messages[i];
    const messageRole = (message as BaseMessage & { role?: unknown }).role;
    if (message.getType() !== 'ai' && messageRole !== 'assistant') {
      continue;
    }

    const aiMessage = message as AIMessage | AIMessageChunk;
    const streamContent = dropIncompleteStreamContent
      ? dropIncompleteToolStreamContent(aiMessage.content)
      : aiMessage.content;
    let projectedContent = streamContent;
    let contentChanged = streamContent !== aiMessage.content;
    if (Array.isArray(streamContent)) {
      const originalContent = streamContent as MessageContentComplex[];
      const mappedContent = originalContent.map((block) =>
        projectInlineToolInput(block, normalizedMaxInputChars)
      );
      contentChanged =
        contentChanged ||
        mappedContent.some(
          (block, blockIndex) => block !== originalContent[blockIndex]
        );
      projectedContent = toLangChainContent(mappedContent);
    }

    const originalToolCalls = aiMessage.tool_calls ?? [];
    const projectedToolCalls = originalToolCalls.map((toolCall) => {
      const argsProperty = readPropertyWithoutAccessors(toolCall, 'args');
      const projected = argsProperty.accessor
        ? { value: ACCESSOR_INPUT_PLACEHOLDER, changed: true }
        : projectToolInputWithinLimit(
          argsProperty.value,
          normalizedMaxInputChars
        );
      if (argsProperty.own && !projected.changed) {
        return toolCall;
      }
      return cloneWithProjectedProperties(toolCall, {
        args: projected.value as Record<string, unknown>,
      });
    });
    const toolCallsChanged = projectedToolCalls.some(
      (toolCall, toolCallIndex) => toolCall !== originalToolCalls[toolCallIndex]
    );

    const rawAdditionalToolCallsProperty = readPropertyWithoutAccessors(
      aiMessage.additional_kwargs,
      'tool_calls'
    );
    const rawAdditionalToolCalls = rawAdditionalToolCallsProperty.accessor
      ? undefined
      : rawAdditionalToolCallsProperty.value;
    const rawResponsesOutputProperty = readPropertyWithoutAccessors(
      aiMessage.response_metadata,
      'output'
    );
    const rawResponsesOutput = rawResponsesOutputProperty.accessor
      ? undefined
      : rawResponsesOutputProperty.value;
    const canonicalArguments = new Map<string, string>();
    const canonicalInputs = new Map<string, unknown>();
    if (
      Array.isArray(rawAdditionalToolCalls) ||
      Array.isArray(rawResponsesOutput)
    ) {
      for (const toolCall of projectedToolCalls) {
        const id = getStringProperty(toolCall, 'id');
        if (id == null) {
          continue;
        }
        const args = readPropertyWithoutAccessors(toolCall, 'args');
        const canonicalInput = args.accessor
          ? ACCESSOR_INPUT_PLACEHOLDER
          : args.value;
        canonicalInputs.set(id, canonicalInput);
        canonicalArguments.set(
          id,
          serializeToolCallInput(canonicalInput, normalizedMaxInputChars)
        );
      }
    }

    const additionalKwargsChanges: Record<string, unknown> = {};
    if (!retainRawOpenAIToolCalls && rawAdditionalToolCallsProperty.found) {
      additionalKwargsChanges.tool_calls = undefined;
    } else if (rawAdditionalToolCallsProperty.accessor) {
      additionalKwargsChanges.tool_calls = undefined;
    } else {
      const rawToolCalls = projectRawOpenAIToolCalls(
        rawAdditionalToolCalls,
        normalizedMaxInputChars,
        canonicalArguments
      );
      if (rawToolCalls.changed) {
        additionalKwargsChanges.tool_calls = rawToolCalls.value;
      }
    }
    const legacyFunctionCall = projectLegacyFunctionCall(
      readPropertyWithoutAccessors(
        aiMessage.additional_kwargs,
        'function_call'
      ),
      normalizedMaxInputChars
    );
    if (legacyFunctionCall.changed) {
      additionalKwargsChanges.function_call = legacyFunctionCall.value;
    }
    const additionalKwargsChanged =
      Object.keys(additionalKwargsChanges).length > 0;
    let projectedAdditionalKwargs = aiMessage.additional_kwargs;
    if (additionalKwargsChanged) {
      projectedAdditionalKwargs = isProxy(aiMessage.additional_kwargs)
        ? { ...additionalKwargsChanges }
        : cloneWithProjectedProperties(
          aiMessage.additional_kwargs,
          additionalKwargsChanges
        );
    }

    let projectedResponseMetadata = aiMessage.response_metadata;
    let responseMetadataChanged = rawResponsesOutputProperty.accessor;
    const responsesOutput = projectResponsesOutput(
      rawResponsesOutput,
      normalizedMaxInputChars,
      canonicalArguments,
      canonicalInputs
    );
    if (responsesOutput.changed || rawResponsesOutputProperty.accessor) {
      const output = rawResponsesOutputProperty.accessor
        ? undefined
        : responsesOutput.value;
      projectedResponseMetadata = isProxy(aiMessage.response_metadata)
        ? { output }
        : cloneWithProjectedProperties(aiMessage.response_metadata, {
          output,
        });
      responseMetadataChanged = true;
    }

    if (
      !contentChanged &&
      !toolCallsChanged &&
      !additionalKwargsChanged &&
      !responseMetadataChanged
    ) {
      continue;
    }

    const changes = {
      content: projectedContent,
      tool_calls: projectedToolCalls.length > 0 ? projectedToolCalls : [],
      additional_kwargs: projectedAdditionalKwargs,
      response_metadata: projectedResponseMetadata,
    };
    const projectedMessage =
      streamContent === aiMessage.content
        ? cloneWithProjectedProperties(aiMessage, changes)
        : cloneAIMessageWithProjectedStreamContent(
          aiMessage,
          changes,
          streamContent
        );
    projectedMessages ??= [...messages];
    projectedMessages[i] = projectedMessage;
  }
  return projectedMessages ?? messages;
}

/** Projects all historical tool-call input representations to bounded values. */
export function projectToolCallInputs(
  messages: BaseMessage[],
  maxInputChars: number,
  fromIndex = 0
): BaseMessage[] {
  return projectToolCallInputsInternal(
    messages,
    maxInputChars,
    false,
    fromIndex
  );
}

/**
 * Derives provider-safe tool history in one pass by dropping incomplete stream
 * content and bounding every provider-consumed tool-call input representation.
 */
export function projectToolMessagesForProvider(
  messages: BaseMessage[],
  maxInputChars: number,
  provider?: ProviderName
): BaseMessage[] {
  const providerFamily =
    provider == null ? undefined : getProviderFamily(provider);
  const retainRawOpenAIToolCalls =
    provider == null ||
    provider === Providers.OPENAI ||
    providerFamily === 'openai';
  return projectToolCallInputsInternal(
    messages,
    maxInputChars,
    true,
    0,
    retainRawOpenAIToolCalls
  );
}

function applyToolCallInputCaps(params: {
  canonicalMessages?: BaseMessage[];
  messages: BaseMessage[];
  maxInputChars: number;
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
  fromIndex?: number;
}): number {
  const { messages, maxInputChars, indexTokenCountMap, tokenCounter } = params;
  const fromIndex = params.fromIndex ?? 0;
  const sourceMessages = params.canonicalMessages ?? messages;
  const projected = projectToolCallInputs(
    sourceMessages,
    maxInputChars,
    fromIndex
  );
  if (projected === sourceMessages) {
    return 0;
  }

  let truncatedCount = 0;
  for (let i = fromIndex; i < messages.length; i++) {
    if (projected[i] === sourceMessages[i]) {
      continue;
    }
    if (projected[i] === messages[i]) {
      continue;
    }
    const current = messages[i] as AIMessage;
    const canonical = sourceMessages[i] as AIMessage;
    const capped = projected[i] as AIMessage;
    const changes: Record<string, unknown> = {};
    if (capped.content !== canonical.content) {
      changes.content = capped.content;
    }
    if (capped.tool_calls !== canonical.tool_calls) {
      changes.tool_calls = capped.tool_calls;
    }
    const additionalKwargsChanges: Record<string, unknown> = {};
    for (const key of ['tool_calls', 'function_call']) {
      if (capped.additional_kwargs[key] !== canonical.additional_kwargs[key]) {
        additionalKwargsChanges[key] = capped.additional_kwargs[key];
      }
    }
    if (Object.keys(additionalKwargsChanges).length > 0) {
      changes.additional_kwargs = cloneWithProjectedProperties(
        current.additional_kwargs,
        additionalKwargsChanges
      );
    }
    if (
      capped.response_metadata.output !== canonical.response_metadata.output
    ) {
      changes.response_metadata = cloneWithProjectedProperties(
        current.response_metadata,
        { output: capped.response_metadata.output }
      );
    }
    const merged = cloneWithProjectedProperties(current, changes);
    messages[i] = merged;
    indexTokenCountMap[i] = tokenCounter(merged);
    truncatedCount++;
  }
  return truncatedCount;
}

/**
 * Whether a message opens a user turn: a human message or a role-based `user`
 * chat message. SDK-injected context (hook output after a tool batch, steers) is
 * stamped as a `HumanMessage` but belongs to the turn it follows.
 */
function startsUserTurn(message: BaseMessage): boolean {
  const type = message.getType();
  if (type === 'generic') {
    return (message as { role?: unknown }).role === 'user';
  }
  return type === 'human' && message.additional_kwargs.injected !== true;
}

export function preFlightTruncateToolCallInputs(params: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
}): number {
  return applyToolCallInputCaps({
    messages: params.messages,
    maxInputChars: calculateMaxToolCallInputChars(params.maxContextTokens),
    indexTokenCountMap: params.indexTokenCountMap,
    tokenCounter: params.tokenCounter,
  });
}

type ThinkingBlocks = {
  thinking_blocks?: Array<{
    type: 'thinking';
    thinking: string;
    signature: string;
  }>;
};

export function createPruneMessages(factoryParams: PruneMessagesFactoryParams) {
  const providerFamily =
    factoryParams.provider == null
      ? undefined
      : getProviderFamily(factoryParams.provider);
  const usesBedrockThinking =
    factoryParams.provider === Providers.BEDROCK ||
    providerFamily === 'bedrock';
  const usesOpenAIThinking =
    factoryParams.provider === Providers.OPENAI || providerFamily === 'openai';
  const indexTokenCountMap = { ...factoryParams.indexTokenCountMap };
  let lastTurnStartIndex = factoryParams.startIndex;
  let lastCutOffIndex = 0;
  let totalTokens = 0;
  for (const key in indexTokenCountMap) {
    totalTokens += indexTokenCountMap[key] ?? 0;
  }
  const reconciledToolMessages = new WeakSet<BaseMessage>();
  let runThinkingStartIndex = -1;
  /** Cumulative raw tiktoken tokens we've sent to the provider (messages only,
   *  excludes instruction overhead and new outputs not yet seen by provider). */
  let cumulativeRawSent = 0;
  /** Cumulative provider-reported message tokens (providerInput - instructionOverhead). */
  let cumulativeProviderReported = 0;
  /** Stable calibration ratio = cumulativeProviderReported / cumulativeRawSent.
   *  Converges monotonically as data accumulates. Falls back to seeded value. */
  let calibrationRatio =
    factoryParams.calibrationRatio != null && factoryParams.calibrationRatio > 0
      ? factoryParams.calibrationRatio
      : 1;
  /** Best observed instruction overhead from a near-zero variance turn.
   *  Self-seeds from provider observations within the run. */
  let bestInstructionOverhead: number | undefined;
  const reconciledLegacyAiMessages = new WeakSet<BaseMessage>();
  const canonicalizedToolCallMessages = new WeakSet<BaseMessage>();
  let bestVarianceAbs = Infinity;
  /** Local estimate at the time bestInstructionOverhead was observed.
   *  Used to invalidate the cached overhead when instructions change
   *  mid-run (e.g. tool discovery adds tools to the bound set). */
  let bestInstructionEstimate: number | undefined;
  /** Original (pre-masking) tool result content keyed by message index.
   *  Allows the summarizer to see full tool outputs even after masking
   *  has truncated them in the live message array. Cleared when the
   *  pruner is recreated after summarization. */
  const originalToolContent = new Map<number, string>();
  let originalToolContentSize = 0;
  /** Recovers canonical sources for direct callers that reuse this mutating
   * projection without passing graph-owned history explicitly. */
  const canonicalByProjection = new WeakMap<BaseMessage, BaseMessage>();
  /** Latched fading tier; caps derive from it alone so bytes stay stable. */
  let fadingTier = seedFadingTier(
    factoryParams.maxTokens,
    factoryParams.fadingTier
  );
  let restoredTierPending = isFadingTier(factoryParams.fadingTier);
  /** Widest exchange of the current turn (since the last human message);
   *  updated only from the appended suffix. */
  let maxToolExchangeWidth = 1;
  let toolExchangeWidthThrough = 0;
  let toolExchangeWidthSources: BaseMessage[] = [];
  /** Fresh results and inputs below this index already carry the tier's caps. */
  let fadedThrough = 0;
  /** Consumed results below this index already carry the tier's masked cap. */
  let maskedThrough = 0;
  const contextPruningSettings = resolveContextPruningSettings(
    factoryParams.contextPruningConfig
  );

  return function pruneMessages(params: PruneMessagesParams): {
    context: BaseMessage[];
    indexTokenCountMap: Record<string, number | undefined>;
    messagesToRefine?: BaseMessage[];
    prePruneContextTokens?: number;
    remainingContextTokens?: number;
    contextPressure?: number;
    originalToolContent?: Map<number, string>;
    newOriginalToolContent?: Map<number, string>;
    calibrationRatio?: number;
    /** Latched fading tier after this call; hosts persist it beside calibrationRatio. */
    fadingTier: FadingTier;
    resolvedInstructionOverhead?: number;
    /** Usable budget this call: maxTokens minus output reserve */
    contextBudget?: number;
    /** Calibrated instruction overhead actually applied this call */
    effectiveInstructionTokens?: number;
  } {
    const suppliedCanonicalMessages = params.canonicalMessages;
    const derivesCanonicalHistory = suppliedCanonicalMessages == null;
    const canonicalMessages =
      suppliedCanonicalMessages ??
      params.messages.map((message) => {
        const recovered = canonicalByProjection.get(message);
        if (recovered != null) {
          return recovered;
        }
        return Array.isArray(message.content)
          ? cloneMessage(message, [...message.content])
          : message;
      });
    const priorWidthLength = toolExchangeWidthSources.length;
    const widthPrefixChanged = derivesCanonicalHistory
      ? canonicalMessages.length < priorWidthLength ||
        toolExchangeWidthSources.some(
          (source, index) => canonicalMessages[index] !== source
        )
      : canonicalMessages.length < priorWidthLength ||
        (params.canonicalPrefixStable === true
          ? priorWidthLength > 0 &&
            canonicalMessages[priorWidthLength - 1] !==
              toolExchangeWidthSources[priorWidthLength - 1]
          : toolExchangeWidthSources.some(
            (source, index) => canonicalMessages[index] !== source
          ));
    if (widthPrefixChanged) {
      maxToolExchangeWidth = 1;
      toolExchangeWidthThrough = 0;
      toolExchangeWidthSources = [];
      fadedThrough = 0;
      maskedThrough = 0;
      originalToolContent.clear();
      originalToolContentSize = 0;
    }
    for (let i = toolExchangeWidthThrough; i < canonicalMessages.length; i++) {
      const message = canonicalMessages[i];
      /** Earlier turns come back from storage with each turn's steps merged into
       *  one assistant message, so their call counts are an artifact of that
       *  reconstruction, not fan-out; only the current turn's are real. The tier
       *  itself stays latched, so a narrower width never loosens it. */
      maxToolExchangeWidth = startsUserTurn(message)
        ? 1
        : Math.max(maxToolExchangeWidth, getToolCallIds(message).size);
      toolExchangeWidthSources[i] = message;
    }
    toolExchangeWidthThrough = canonicalMessages.length;
    let newOriginalToolContent: Map<number, string> | undefined;
    if (params.messages.length === 0) {
      /** Post-compaction calls still invoke the model — report the same
       *  reserve-adjusted budget fields as the populated paths */
      const emptyInstructionTokens =
        factoryParams.getInstructionTokens?.() ?? 0;
      const emptyReserveRatio =
        factoryParams.reserveRatio ?? DEFAULT_RESERVE_RATIO;
      const emptyBudget =
        factoryParams.maxTokens -
        (emptyReserveRatio > 0 && emptyReserveRatio < 1
          ? Math.round(factoryParams.maxTokens * emptyReserveRatio)
          : 0);
      const emptyReplyPrimerTokens = Math.round(
        REPLY_PRIMER_TOKENS * calibrationRatio
      );
      return {
        context: [],
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneContextTokens: 0,
        remainingContextTokens: Math.max(
          0,
          emptyBudget - emptyInstructionTokens - emptyReplyPrimerTokens
        ),
        calibrationRatio,
        fadingTier,
        resolvedInstructionOverhead: bestInstructionOverhead,
        contextBudget: emptyBudget,
        effectiveInstructionTokens: emptyInstructionTokens,
      };
    }

    if (usesOpenAIThinking && factoryParams.thinkingEnabled === true) {
      for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
        const m = params.messages[i];
        if (
          m.getType() === 'ai' &&
          typeof m.additional_kwargs.reasoning_content === 'string' &&
          Array.isArray(
            (
              m.additional_kwargs.provider_specific_fields as
                | ThinkingBlocks
                | undefined
            )?.thinking_blocks
          ) &&
          (m as AIMessage).tool_calls &&
          ((m as AIMessage).tool_calls?.length ?? 0) > 0
        ) {
          const message = m as AIMessage;
          const thinkingBlocks = (
            message.additional_kwargs.provider_specific_fields as ThinkingBlocks
          ).thinking_blocks;
          const signature =
            thinkingBlocks?.[thinkingBlocks.length - 1].signature;
          const thinkingBlock: ThinkingContentText = {
            signature,
            type: ContentTypes.THINKING,
            thinking: message.additional_kwargs.reasoning_content as string,
          };

          params.messages[i] = new AIMessage({
            ...message,
            content: toLangChainContent([thinkingBlock]),
            additional_kwargs: {
              ...message.additional_kwargs,
              reasoning_content: undefined,
            },
          });
        }
      }
    }

    let currentUsage: UsageMetadata | undefined;
    if (
      params.usageMetadata &&
      (checkValidNumber(params.usageMetadata.input_tokens) ||
        (checkValidNumber(params.usageMetadata.input_token_details) &&
          (checkValidNumber(
            params.usageMetadata.input_token_details.cache_creation
          ) ||
            checkValidNumber(
              params.usageMetadata.input_token_details.cache_read
            )))) &&
      checkValidNumber(params.usageMetadata.output_tokens)
    ) {
      currentUsage = calculateTotalTokens(params.usageMetadata);
    }

    const newOutputs = new Set<number>();

    // Host token maps predate LangChain formatting and may omit tool output or
    // legacy function-call arguments. Reconcile those provider-bound shapes
    // once before making any budget decision.
    const applyReconciledCount = (
      index: number,
      cachedCount: number | undefined,
      reconciledCount: number
    ): void => {
      if (cachedCount === undefined) {
        indexTokenCountMap[index] = reconciledCount;
        totalTokens += reconciledCount;
        if (index >= lastTurnStartIndex) {
          newOutputs.add(index);
        }
        return;
      }
      // Preserve a larger host estimate: reconciliation is a safety floor,
      // not permission to reduce an upstream count that may include provider
      // serialization overhead unavailable to the local counter.
      if (reconciledCount <= cachedCount) {
        return;
      }
      indexTokenCountMap[index] = reconciledCount;
      totalTokens += reconciledCount - cachedCount;
      if (index >= lastTurnStartIndex) {
        newOutputs.add(index);
      }
    };
    for (let i = 0; i < params.messages.length; i++) {
      let message = params.messages[i];
      let cachedCount = indexTokenCountMap[i];
      const messageType = message.getType();
      const messageRole = (message as BaseMessage & { role?: unknown }).role;
      const isAssistant = messageType === 'ai' || messageRole === 'assistant';
      if (isAssistant && !canonicalizedToolCallMessages.has(message)) {
        const [canonicalized] = projectToolCallInputs(
          [message],
          HARD_MAX_TOOL_CALL_INPUT_CHARS
        );
        if (canonicalized !== message) {
          message = canonicalized;
          params.messages[i] = message;
          const canonicalizedCount = factoryParams.tokenCounter(message);
          indexTokenCountMap[i] = canonicalizedCount;
          totalTokens += canonicalizedCount - (cachedCount ?? 0);
          cachedCount = canonicalizedCount;
          if (i >= lastTurnStartIndex) {
            newOutputs.add(i);
          }
        }
        canonicalizedToolCallMessages.add(message);
      }
      const legacyFunctionCall = isAssistant
        ? readPropertyWithoutAccessors(
          message.additional_kwargs,
          'function_call'
        )
        : undefined;
      if (
        legacyFunctionCall?.found === true &&
        !reconciledLegacyAiMessages.has(message)
      ) {
        const [projected] = projectToolCallInputs(
          [message],
          calculateMaxToolCallInputChars(factoryParams.maxTokens)
        );
        if (projected !== message) {
          message = projected;
          params.messages[i] = message;
          const projectedCount = factoryParams.tokenCounter(message);
          indexTokenCountMap[i] = projectedCount;
          totalTokens += projectedCount - (cachedCount ?? 0);
          cachedCount = projectedCount;
          if (i >= lastTurnStartIndex) {
            newOutputs.add(i);
          }
        }
        reconciledLegacyAiMessages.add(message);
        if (cachedCount !== undefined || i < lastTurnStartIndex) {
          applyReconciledCount(
            i,
            cachedCount,
            factoryParams.tokenCounter(message)
          );
        }
        continue;
      }
      if (messageType !== 'tool' || reconciledToolMessages.has(message)) {
        continue;
      }
      if (!isComputerCallOutputMessage(message)) {
        const normalized = compactToolContent(
          message.content,
          factoryParams.maxToolResultChars ?? HARD_MAX_TOOL_RESULT_CHARS
        );
        if (normalized.changed) {
          message = cloneToolMessageWithContent(
            message as ToolMessage,
            normalized.content
          );
          params.messages[i] = message;
        }
      }
      const reconciledCount = factoryParams.tokenCounter(message);
      reconciledToolMessages.add(message);
      applyReconciledCount(i, cachedCount, reconciledCount);
    }

    let outputTokensAssigned = false;
    for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
      const message = params.messages[i];
      if (indexTokenCountMap[i] !== undefined) {
        continue;
      }

      // Assign output_tokens to the first uncounted AI message — this is the
      // model's response.  Previous code blindly targeted lastTurnStartIndex
      // which could hit a pre-counted HumanMessage or miss the AI entirely.
      if (!outputTokensAssigned && currentUsage && message.getType() === 'ai') {
        indexTokenCountMap[i] = currentUsage.output_tokens;
        newOutputs.add(i);
        outputTokensAssigned = true;
      } else {
        // Always store raw tiktoken count — the map stays in raw space.
        // Budget decisions multiply by calibrationRatio on the fly.
        indexTokenCountMap[i] = factoryParams.tokenCounter(message);
        if (currentUsage) {
          newOutputs.add(i);
        }
      }
      totalTokens += indexTokenCountMap[i] ?? 0;
    }

    // Cumulative calibration: accumulate raw tiktoken tokens and provider-
    // reported tokens across turns.  The ratio of the two running totals
    // converges monotonically to the true provider multiplier — no EMA,
    // no per-turn oscillation, no map mutation.
    if (currentUsage && params.totalTokensFresh !== false) {
      const instructionOverhead = factoryParams.getInstructionTokens?.() ?? 0;
      const rawProviderInputTokens = Number(params.usageMetadata?.input_tokens);
      const providerInputTokens = checkValidNumber(rawProviderInputTokens)
        ? rawProviderInputTokens
        : (params.lastCallUsage?.inputTokens ?? currentUsage.input_tokens);

      // Sum raw tiktoken counts for messages the provider saw (excludes
      // new outputs from this turn — the provider hasn't seen them yet).
      let rawSentThisTurn = 0;
      const firstIsSystem =
        params.messages.length > 0 && params.messages[0].getType() === 'system';
      if (firstIsSystem) {
        rawSentThisTurn += indexTokenCountMap[0] ?? 0;
      }
      for (let i = lastCutOffIndex; i < params.messages.length; i++) {
        if ((i === 0 && firstIsSystem) || newOutputs.has(i)) {
          continue;
        }
        rawSentThisTurn += indexTokenCountMap[i] ?? 0;
      }

      const providerMessageTokens = Math.max(
        0,
        providerInputTokens - instructionOverhead
      );
      const minimumComparableInputTokens =
        instructionOverhead + rawSentThisTurn * CALIBRATION_RATIO_MIN;
      let calibrationSkipReason: string | undefined;
      if (rawSentThisTurn <= 0) {
        calibrationSkipReason = 'no_sent_messages';
      } else if (providerMessageTokens <= 0) {
        calibrationSkipReason = 'input_below_instruction_overhead';
      } else if (providerInputTokens < minimumComparableInputTokens) {
        calibrationSkipReason = 'input_below_calibration_floor';
      }
      // No upper-bound rejection: maxTokens is an application budget, not the
      // provider's real context window.  Usage above the budget is a genuine
      // measurement — and the one that must drive pruning/summarization.

      if (calibrationSkipReason == null) {
        cumulativeRawSent += rawSentThisTurn;
        cumulativeProviderReported += providerMessageTokens;
        const newRatio = cumulativeProviderReported / cumulativeRawSent;
        calibrationRatio = clampCalibrationRatio(newRatio);

        const calibratedOurTotal =
          instructionOverhead + rawSentThisTurn * calibrationRatio;
        const overallRatio =
          calibratedOurTotal > 0 ? providerInputTokens / calibratedOurTotal : 0;
        const variancePct = Math.round((overallRatio - 1) * 100);

        const absVariance = Math.abs(overallRatio - 1);
        if (absVariance < bestVarianceAbs) {
          bestVarianceAbs = absVariance;
          bestInstructionOverhead = Math.max(
            0,
            Math.round(providerInputTokens - rawSentThisTurn * calibrationRatio)
          );
          bestInstructionEstimate = factoryParams.getInstructionTokens?.() ?? 0;
        }

        factoryParams.log?.('debug', 'Calibration observed', {
          providerInputTokens,
          calibratedEstimate: Math.round(calibratedOurTotal),
          variance: `${variancePct > 0 ? '+' : ''}${variancePct}%`,
          calibrationRatio: Math.round(calibrationRatio * 100) / 100,
          instructionOverhead,
          cumulativeRawSent,
          cumulativeProviderReported,
        });
      } else {
        factoryParams.log?.('debug', 'Calibration skipped', {
          reason: calibrationSkipReason,
          providerInputTokens,
          minimumComparableInputTokens: Math.round(
            minimumComparableInputTokens
          ),
          rawSentThisTurn,
          instructionOverhead,
        });
      }
    }

    // Computed BEFORE pre-flight truncation so the effective budget can drive
    // truncation thresholds — without this, thresholds based on maxTokens are
    // too generous and leave individual messages larger than the actual budget.
    const estimatedInstructionTokens =
      factoryParams.getInstructionTokens?.() ?? 0;
    const estimateStable =
      bestInstructionEstimate != null &&
      bestInstructionEstimate > 0 &&
      Math.abs(estimatedInstructionTokens - bestInstructionEstimate) /
        bestInstructionEstimate <
        0.1;
    const currentInstructionTokens =
      bestInstructionOverhead != null &&
      bestInstructionOverhead <= estimatedInstructionTokens &&
      estimateStable
        ? bestInstructionOverhead
        : estimatedInstructionTokens;

    const reserveRatio = factoryParams.reserveRatio ?? DEFAULT_RESERVE_RATIO;
    const reserveTokens =
      reserveRatio > 0 && reserveRatio < 1
        ? Math.round(factoryParams.maxTokens * reserveRatio)
        : 0;
    const pruningBudget = factoryParams.maxTokens - reserveTokens;

    const effectiveMaxTokens = Math.max(
      0,
      pruningBudget - currentInstructionTokens
    );

    let calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);

    // When instructions alone consume the entire budget, no message can
    // fit regardless of truncation.  Short-circuit: yield all messages for
    // summarization and return an empty context so the Graph can route to
    // the summarize node immediately instead of falling through to the
    // emergency path that would reach the same outcome more expensively.
    if (
      effectiveMaxTokens === 0 &&
      factoryParams.summarizationEnabled === true &&
      params.messages.length > 0
    ) {
      factoryParams.log?.(
        'warn',
        'Instructions consume entire budget — yielding all messages for summarization',
        {
          instructionTokens: currentInstructionTokens,
          pruningBudget,
          messageCount: params.messages.length,
        }
      );

      lastTurnStartIndex = params.messages.length;
      return {
        context: [],
        indexTokenCountMap,
        messagesToRefine: [...params.messages],
        prePruneContextTokens: calibratedTotalTokens,
        remainingContextTokens: 0,
        contextPressure:
          pruningBudget > 0 ? calibratedTotalTokens / pruningBudget : 0,
        calibrationRatio,
        fadingTier,
        resolvedInstructionOverhead: bestInstructionOverhead,
        contextBudget: pruningBudget,
        effectiveInstructionTokens: currentInstructionTokens,
      };
    }

    // ---------------------------------------------------------------------------
    // Progressive context fading — inspired by Claude Code's staged compaction.
    // Every cap comes from the latched fading tier (see ./fading.ts): the fit
    // rung keeps a single tool result within the effective budget, pressure
    // bands deepen the rung when summarization is off, and masking (80%+)
    // shrinks consumed results the model has already answered. The tier only
    // ever deepens, so a historical tool result maps to the same bytes on
    // every call and provider prompt-cache prefixes survive from turn to turn;
    // only escalation rewrites them.
    // ---------------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Observation masking (80%+ pressure, both paths):
    // Replace consumed ToolMessage content with tight head+tail placeholders.
    // AI messages stay intact so the model can read its own prior reasoning
    // and won't repeat work.  Unconsumed results (latest tool outputs the
    // model hasn't acted on yet) stay full.
    //
    // When summarization is enabled, snapshot messages first so the
    // summarizer can see the full originals when compaction fires.
    // -----------------------------------------------------------------------
    const storeOriginalToolContent = (index: number, content: string): void => {
      originalToolContentSize += content.length;
      if (newOriginalToolContent == null) {
        newOriginalToolContent = new Map();
      }
      newOriginalToolContent.set(index, content);
      while (
        originalToolContentSize > ORIGINAL_CONTENT_MAX_CHARS &&
        originalToolContent.size > 0
      ) {
        const oldest = originalToolContent.keys().next();
        if (oldest.done === true) {
          break;
        }
        const removed = originalToolContent.get(oldest.value);
        if (removed != null) {
          originalToolContentSize -= removed.length;
        }
        originalToolContent.delete(oldest.value);
      }
    };

    let restoredFading: FadingApplyResult = {
      truncated: 0,
      inputs: 0,
      masked: 0,
      consumedBoundary: 0,
    };
    if (restoredTierPending) {
      restoredFading = applyFadingCaps({
        messages: params.messages,
        canonicalMessages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        caps: resolveFadingCaps(fadingTier, factoryParams.maxToolResultChars),
        masked: fadingTier.masked,
        originalContentStore:
          factoryParams.summarizationEnabled === true
            ? originalToolContent
            : undefined,
        onContentStored:
          factoryParams.summarizationEnabled === true
            ? storeOriginalToolContent
            : undefined,
      });
      fadedThrough = params.messages.length;
      maskedThrough = restoredFading.consumedBoundary;
      restoredTierPending = false;
    }

    totalTokens = sumTokenCounts(indexTokenCountMap, params.messages.length);
    calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);
    const contextPressure =
      pruningBudget > 0 ? calibratedTotalTokens / pruningBudget : 0;

    factoryParams.log?.('debug', 'Budget', {
      maxTokens: factoryParams.maxTokens,
      pruningBudget,
      effectiveMax: effectiveMaxTokens,
      instructionTokens: currentInstructionTokens,
      messageCount: params.messages.length,
      calibratedTotalTokens,
      calibrationRatio: Math.round(calibrationRatio * 100) / 100,
    });

    /** Advances the tier from the signals and applies its caps; escalation rescans everything. */
    const fade = (signals: FadingSignals): FadingApplyResult => {
      const nextTier = resolveFadingTier(
        fadingTier,
        factoryParams.maxTokens,
        signals,
        factoryParams.maxToolResultChars
      );
      if (nextTier !== fadingTier) {
        fadingTier = nextTier;
        fadedThrough = 0;
        maskedThrough = 0;
      }
      const result = applyFadingCaps({
        messages: params.messages,
        canonicalMessages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        caps: resolveFadingCaps(fadingTier, factoryParams.maxToolResultChars),
        masked: fadingTier.masked,
        fromIndex: fadedThrough,
        maskedFromIndex: maskedThrough,
        originalContentStore:
          factoryParams.summarizationEnabled === true
            ? originalToolContent
            : undefined,
        onContentStored:
          factoryParams.summarizationEnabled === true
            ? storeOriginalToolContent
            : undefined,
      });
      fadedThrough = params.messages.length;
      maskedThrough = result.consumedBoundary;
      return result;
    };

    const fadingSignals: FadingSignals = {
      contextPressure,
      effectiveRawTokens:
        calibrationRatio > 0
          ? Math.floor(effectiveMaxTokens / calibrationRatio)
          : effectiveMaxTokens,
      summarizationEnabled: factoryParams.summarizationEnabled === true,
      toolExchangeWidth: maxToolExchangeWidth,
    };
    const faded = fade(fadingSignals);
    const observationsMasked = restoredFading.masked + faded.masked;
    const preFlightResultCount = restoredFading.truncated + faded.truncated;
    const preFlightInputCount = restoredFading.inputs + faded.inputs;
    if (observationsMasked > 0) {
      cumulativeRawSent = 0;
      cumulativeProviderReported = 0;
    }

    if (
      factoryParams.contextPruningConfig?.enabled === true &&
      factoryParams.summarizationEnabled !== true
    ) {
      applyContextPruning({
        messages: params.messages,
        canonicalMessages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        resolvedSettings: contextPruningSettings,
      });
    }
    if (derivesCanonicalHistory) {
      for (let i = 0; i < params.messages.length; i++) {
        canonicalByProjection.set(params.messages[i], canonicalMessages[i]);
      }
    }

    const preTruncationTotalTokens = totalTokens;
    totalTokens = sumTokenCounts(indexTokenCountMap, params.messages.length);
    calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);

    const anyAdjustment =
      observationsMasked > 0 ||
      preFlightResultCount > 0 ||
      preFlightInputCount > 0 ||
      totalTokens !== preTruncationTotalTokens;

    if (anyAdjustment) {
      factoryParams.log?.('debug', 'Context adjusted', {
        contextPressure: Math.round(contextPressure * 100),
        observationsMasked,
        toolOutputsTruncated: preFlightResultCount,
        toolInputsTruncated: preFlightInputCount,
        tokensBefore: preTruncationTotalTokens,
        tokensAfter: totalTokens,
        tokensSaved: preTruncationTotalTokens - totalTokens,
      });
    }

    lastTurnStartIndex = params.messages.length;
    const calibratedReplyPrimerTokens = Math.round(
      REPLY_PRIMER_TOKENS * calibrationRatio
    );
    if (
      lastCutOffIndex === 0 &&
      calibratedTotalTokens +
        calibratedReplyPrimerTokens +
        currentInstructionTokens <=
        pruningBudget
    ) {
      return {
        context: params.messages,
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneContextTokens: calibratedTotalTokens,
        remainingContextTokens:
          pruningBudget -
          calibratedTotalTokens -
          calibratedReplyPrimerTokens -
          currentInstructionTokens,
        contextPressure,
        originalToolContent:
          originalToolContent.size > 0 ? originalToolContent : undefined,
        newOriginalToolContent,
        calibrationRatio,
        fadingTier,
        resolvedInstructionOverhead: bestInstructionOverhead,
        contextBudget: pruningBudget,
        effectiveInstructionTokens: currentInstructionTokens,
      };
    }

    const rawSpaceBudget =
      calibrationRatio > 0
        ? Math.round(pruningBudget / calibrationRatio)
        : pruningBudget;

    const rawSpaceInstructionTokens =
      calibrationRatio > 0
        ? Math.round(currentInstructionTokens / calibrationRatio)
        : currentInstructionTokens;

    const {
      context: initialContext,
      thinkingStartIndex,
      messagesToRefine,
      remainingContextTokens: initialRemainingContextTokens,
    } = getMessagesWithinTokenLimit({
      maxContextTokens: rawSpaceBudget,
      messages: params.messages,
      indexTokenCountMap,
      startType: params.startType,
      thinkingEnabled: factoryParams.thinkingEnabled,
      tokenCounter: factoryParams.tokenCounter,
      instructionTokens: rawSpaceInstructionTokens,
      reasoningType: usesBedrockThinking
        ? ContentTypes.REASONING_CONTENT
        : ContentTypes.THINKING,
      thinkingStartIndex:
        factoryParams.thinkingEnabled === true
          ? runThinkingStartIndex
          : undefined,
    });

    const {
      context: repairedContext,
      reclaimedTokens: initialReclaimedTokens,
      droppedMessages,
    } = repairOrphanedToolMessages({
      context: initialContext,
      allMessages: params.messages,
      tokenCounter: factoryParams.tokenCounter,
      indexTokenCountMap,
    });

    const contextBreakdown = repairedContext.map((msg) => {
      const type = msg.getType();
      const name = type === 'tool' ? (msg.name ?? 'unknown') : '';
      return name !== '' ? `${type}(${name})` : type;
    });
    factoryParams.log?.('debug', 'Pruning complete', {
      contextLength: repairedContext.length,
      contextTypes: contextBreakdown.join(', '),
      messagesToRefineCount: messagesToRefine.length,
      droppedOrphans: droppedMessages.length,
      remainingTokens: initialRemainingContextTokens,
    });

    let context = repairedContext;
    let reclaimedTokens = initialReclaimedTokens;

    // Orphan repair may drop ToolMessages whose parent AI was pruned.
    // Append them to messagesToRefine so summarization can still see the
    // tool results (otherwise the summary says "in progress" for a tool
    // call that already completed, causing the model to repeat it).
    if (droppedMessages.length > 0) {
      appendItems(messagesToRefine, droppedMessages);
    }

    // ---------------------------------------------------------------
    // Emergency truncation: if pruning produced an empty context but
    // messages exist, derive a deeper, temporary tier from a per-message
    // share of the effective budget (~4 chars/token, floor 200 chars),
    // apply it to a clone and retry.  The latched tier is left alone: this
    // share depends on the message count, so latching it would pin every
    // future result to one transient event.  The clone keeps graph state
    // intact for later turns where more budget may be available.
    // ---------------------------------------------------------------
    if (
      context.length === 0 &&
      params.messages.length > 0 &&
      effectiveMaxTokens > 0
    ) {
      const perMessageTokenBudget = Math.floor(
        effectiveMaxTokens / Math.max(1, params.messages.length)
      );
      const emergencyMaxChars = Math.max(200, perMessageTokenBudget * 4);
      const emergencyTier = resolveFadingTier(
        fadingTier,
        factoryParams.maxTokens,
        {
          ...fadingSignals,
          minRung: fadingRungForExchangeChars(
            factoryParams.maxTokens,
            emergencyMaxChars,
            factoryParams.maxToolResultChars
          ),
        },
        factoryParams.maxToolResultChars
      );
      const emergencyCaps = resolveFadingCaps(
        emergencyTier,
        factoryParams.maxToolResultChars
      );

      factoryParams.log?.(
        'warn',
        'Empty context, entering emergency truncation',
        {
          messageCount: params.messages.length,
          effectiveMax: effectiveMaxTokens,
          emergencyMaxChars,
          budgetTokens: emergencyCaps.budgetTokens,
        }
      );

      const emergencyMessages = [...params.messages];
      const preEmergencyTokenCounts: Record<string, number | undefined> = {};
      for (let i = 0; i < params.messages.length; i++) {
        preEmergencyTokenCounts[i] = indexTokenCountMap[i];
      }

      try {
        const emergency = applyFadingCaps({
          messages: emergencyMessages,
          canonicalMessages,
          indexTokenCountMap,
          tokenCounter: factoryParams.tokenCounter,
          caps: emergencyCaps,
          masked: emergencyTier.masked,
        });

        factoryParams.log?.('info', 'Emergency truncation complete');
        factoryParams.log?.('debug', 'Emergency truncation details', {
          truncatedCount:
            emergency.truncated + emergency.inputs + emergency.masked,
          budgetTokens: emergencyCaps.budgetTokens,
        });

        const retryResult = getMessagesWithinTokenLimit({
          maxContextTokens: pruningBudget,
          messages: emergencyMessages,
          indexTokenCountMap,
          startType: params.startType,
          thinkingEnabled: factoryParams.thinkingEnabled,
          tokenCounter: factoryParams.tokenCounter,
          instructionTokens: currentInstructionTokens,
          reasoningType: usesBedrockThinking
            ? ContentTypes.REASONING_CONTENT
            : ContentTypes.THINKING,
          thinkingStartIndex:
            factoryParams.thinkingEnabled === true
              ? runThinkingStartIndex
              : undefined,
        });

        const repaired = repairOrphanedToolMessages({
          context: retryResult.context,
          allMessages: emergencyMessages,
          tokenCounter: factoryParams.tokenCounter,
          indexTokenCountMap,
        });

        context = repaired.context;
        reclaimedTokens = repaired.reclaimedTokens;
        /** The retry supersedes the failed pass: messages now in context
         *  must not also be handed to the summarizer. */
        messagesToRefine.length = 0;
        appendItems(messagesToRefine, retryResult.messagesToRefine);
        if (repaired.droppedMessages.length > 0) {
          appendItems(messagesToRefine, repaired.droppedMessages);
        }

        factoryParams.log?.('debug', 'Emergency truncation retry result', {
          contextLength: context.length,
          messagesToRefineCount: messagesToRefine.length,
          remainingTokens: retryResult.remainingContextTokens,
        });
      } finally {
        // Restore the closure's indexTokenCountMap to pre-emergency values so the
        // next turn counts old messages at their original (un-truncated) size.
        for (const [key, value] of Object.entries(preEmergencyTokenCounts)) {
          indexTokenCountMap[key] = value;
        }
      }
    }

    /** Scale raw-space remaining back to calibrated/provider units so it is
     *  directly comparable with pruningBudget and prePruneContextTokens */
    const rawRemaining = Math.max(
      0,
      initialRemainingContextTokens + reclaimedTokens
    );
    const remainingContextTokens = Math.max(
      0,
      Math.min(
        pruningBudget,
        calibrationRatio > 0
          ? Math.round(rawRemaining * calibrationRatio)
          : rawRemaining
      )
    );

    runThinkingStartIndex = thinkingStartIndex ?? -1;
    /** The index is the first value of `context`, index relative to `params.messages` */
    lastCutOffIndex = Math.max(
      params.messages.length -
        (context.length - (context[0]?.getType() === 'system' ? 1 : 0)),
      0
    );

    return {
      context,
      indexTokenCountMap,
      messagesToRefine,
      prePruneContextTokens: calibratedTotalTokens,
      remainingContextTokens,
      contextPressure,
      originalToolContent:
        originalToolContent.size > 0 ? originalToolContent : undefined,
      newOriginalToolContent,
      calibrationRatio,
      fadingTier,
      resolvedInstructionOverhead: bestInstructionOverhead,
      contextBudget: pruningBudget,
      effectiveInstructionTokens: currentInstructionTokens,
    };
  };
}

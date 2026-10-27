import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderToolCallIndex } from './toolResultTypes';
import type { ProviderName } from '@/types/llm';
import {
  appendProviderMessageToolCalls,
  appendProviderToolCallDescriptor,
  consumeProviderToolResultPair,
  getBoundedProviderPairingArray,
  getProviderMessageRole,
  getProviderToolCallPartDescriptor,
  getProviderToolMessageResultDescriptor,
  getProviderToolResultPartDescriptor,
} from './toolResultTypes';
import { getProviderSourceMessageIds } from './provenance';

export const DEFAULT_RETAIN_RECENT_TURNS = 2;
export const DEFAULT_INTRA_TURN_RETAIN_RATIO = 0.16;

/**
 * Configuration for splitting a message list into a head (to be summarized)
 * and a tail (to be preserved verbatim).
 */
export interface RecencyWindowOptions {
  /** Serving provider used to interpret generic `ChatMessage` roles. */
  provider?: ProviderName;
  /**
   * Maximum number of recent user-led turns to keep in the tail. A "turn"
   * begins at a user-authored HumanMessage and includes every following
   * AIMessage and tool result up to the next user-authored HumanMessage.
   * Provider-native HumanMessages containing only tool results remain in the
   * current turn, so the boundary cannot split them from their calls.
   *
   * The most recent turn is preserved unless `intraTurnTokens` enables the
   * pairing-balanced fallback for a tool-heavy history. A lone oversized user
   * message is never eligible for that fallback.
   *
   * Defaults to `2`.  A value of `0` disables the recency window (head =
   * everything, tail = empty), restoring the pre-recency-window behavior.
   */
  turns?: number;
  /**
   * Optional cap on tail size in tokens.  When set, additional turns
   * beyond the most recent one are added to the tail only while the
   * cumulative token count stays at or below this cap.  Turns are added
   * whole — never partially — so a turn that would exceed the cap is
   * left in the head.
   *
   * The most recent turn is always preserved even if it exceeds the cap.
   */
  tokens?: number;
  /** Token-counter used to evaluate the optional `tokens` cap. */
  tokenCounter?: (m: BaseMessage) => number;
  /**
   * Minimum token budget to retain when the turn window would otherwise make
   * the whole history indivisible. When set with `tokenCounter`, older closed
   * tool-call/result units may be summarized from within the earliest retained
   * turn. A lone user payload and open tool units remain indivisible.
   */
  intraTurnTokens?: number;
}

export interface RecencySplit {
  /** Older messages eligible for summarization.  Empty when nothing to summarize. */
  head: BaseMessage[];
  /** Recent messages preserved verbatim, beginning at a pairing-balanced boundary. */
  tail: BaseMessage[];
  /** Number of user-led turns retained in the tail (0 if no HumanMessage exists). */
  tailTurnCount: number;
  /** Index in the original `messages` array where the tail begins. */
  tailStartIndex: number;
  /** True when the pairing-balanced fallback selected the boundary. */
  usedIntraTurnFallback: boolean;
}

export function resolveIntraTurnRetainTokens({
  tokens,
  maxContextTokens,
}: {
  tokens?: number;
  maxContextTokens?: number;
}): number | undefined {
  if (tokens != null) {
    return Number.isFinite(tokens) && tokens > 0
      ? Math.floor(tokens)
      : undefined;
  }
  if (
    maxContextTokens == null ||
    !Number.isFinite(maxContextTokens) ||
    maxContextTokens <= 0
  ) {
    return undefined;
  }
  return Math.max(
    1,
    Math.floor(maxContextTokens * DEFAULT_INTRA_TURN_RETAIN_RATIO)
  );
}

interface MessagePairingResult {
  completedToolCalls: number;
  trustedHumanToolResult: boolean;
  userMessage: boolean;
}

function inspectMessagePairing(
  message: BaseMessage,
  calls: ProviderToolCallIndex,
  provider?: ProviderName
): MessagePairingResult {
  const isHuman = getProviderMessageRole(message, provider) === 'user';
  const candidateCalls = isHuman ? new Map(calls) : calls;
  if (!isHuman) {
    appendProviderMessageToolCalls(message, candidateCalls, provider);
  }
  const content = getBoundedProviderPairingArray(message.content);
  let completedToolCalls = 0;
  let resultPartCount = 0;
  let everyPartIsTrustedHumanResult = isHuman && content != null;
  if (content != null) {
    let previousPart: unknown;
    for (const part of content) {
      const callDescriptor = getProviderToolCallPartDescriptor(part);
      if (callDescriptor != null && !isHuman) {
        appendProviderToolCallDescriptor(candidateCalls, callDescriptor);
      }
      const resultDescriptor = getProviderToolResultPartDescriptor(part);
      if (resultDescriptor == null) {
        everyPartIsTrustedHumanResult = false;
      } else {
        resultPartCount += 1;
        const canPairHumanResult =
          !isHuman || resultDescriptor.allowHumanMessagePairing === true;
        if (
          canPairHumanResult &&
          consumeProviderToolResultPair(
            resultDescriptor,
            candidateCalls,
            previousPart
          )
        ) {
          completedToolCalls += 1;
        } else {
          everyPartIsTrustedHumanResult = false;
        }
      }
      previousPart = part;
    }
  }

  const toolMessageResult = getProviderToolMessageResultDescriptor(
    message,
    provider
  );
  if (
    toolMessageResult != null &&
    consumeProviderToolResultPair(toolMessageResult, candidateCalls)
  ) {
    completedToolCalls += 1;
  }

  const trustedHumanToolResult =
    everyPartIsTrustedHumanResult &&
    resultPartCount > 0 &&
    resultPartCount === content?.length;
  if (trustedHumanToolResult) {
    calls.clear();
    for (const [callId, entry] of candidateCalls) {
      calls.set(callId, entry);
    }
  }
  return {
    completedToolCalls:
      isHuman && !trustedHumanToolResult ? 0 : completedToolCalls,
    trustedHumanToolResult,
    userMessage: isHuman,
  };
}

function findTurnStarts(
  messages: BaseMessage[],
  provider?: ProviderName
): number[] {
  const turnStarts: number[] = [];
  const calls: ProviderToolCallIndex = new Map();
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i] as BaseMessage;
    const pairing = inspectMessagePairing(message, calls, provider);
    if (!pairing.userMessage || pairing.trustedHumanToolResult) {
      continue;
    }
    turnStarts.push(i);
    calls.clear();
  }
  return turnStarts;
}

function getCoverageSourceIds(message: BaseMessage): string[] {
  const sourceIds = getProviderSourceMessageIds(message);
  if (sourceIds.length > 0) {
    return sourceIds;
  }
  const messageId = message.id?.trim();
  return messageId != null && messageId !== '' ? [messageId] : [];
}

function findIntraTurnBoundary(
  messages: BaseMessage[],
  countMessageAt: (index: number) => number,
  retainTokens: number,
  earliestRetainedTurnEndIndex: number,
  provider?: ProviderName
): number | undefined {
  let remainingTokens = 0;
  for (let i = 0; i < messages.length; i++) {
    remainingTokens += countMessageAt(i);
  }
  if (remainingTokens <= retainTokens) {
    return undefined;
  }

  const sourceIdsByIndex = new Array<string[]>(messages.length);
  const lastIndexBySourceId = new Map<string, number>();
  for (let i = 0; i < messages.length; i++) {
    const sourceIds = getCoverageSourceIds(messages[i] as BaseMessage);
    sourceIdsByIndex[i] = sourceIds;
    for (const sourceId of sourceIds) {
      lastIndexBySourceId.set(sourceId, i);
    }
  }

  const pendingToolCalls: ProviderToolCallIndex = new Map();
  const straddlingSourceIds = new Set<string>();
  let completedToolCalls = 0;
  let boundary: number | undefined;

  const boundarySearchEnd = Math.min(
    messages.length - 1,
    earliestRetainedTurnEndIndex - 1
  );
  for (let i = 0; i <= boundarySearchEnd; i++) {
    const message = messages[i] as BaseMessage;
    remainingTokens -= countMessageAt(i);

    const pairing = inspectMessagePairing(message, pendingToolCalls, provider);
    completedToolCalls += pairing.completedToolCalls;
    if (pairing.userMessage && !pairing.trustedHumanToolResult) {
      pendingToolCalls.clear();
    }

    for (const sourceId of sourceIdsByIndex[i] as string[]) {
      if (lastIndexBySourceId.get(sourceId) === i) {
        straddlingSourceIds.delete(sourceId);
      } else {
        straddlingSourceIds.add(sourceId);
      }
    }

    if (remainingTokens < retainTokens) {
      break;
    }
    if (
      completedToolCalls > 0 &&
      pendingToolCalls.size === 0 &&
      straddlingSourceIds.size === 0 &&
      (!pairing.userMessage || pairing.trustedHumanToolResult)
    ) {
      boundary = i + 1;
    }
  }

  return boundary;
}

/**
 * Splits `messages` into a head (older, to summarize) and a tail (recent,
 * to preserve verbatim), preferring user-message boundaries. The most recent
 * user-led turn is normally included in the tail; additional older turns are
 * added subject to `turns` and `tokens` caps.
 *
 * When that policy exposes no compactable head, `intraTurnTokens` may select
 * a boundary after older closed tool-call/result units. This keeps runaway
 * first-turn tool loops compactable without splitting parallel calls from
 * their results. A user payload without a completed tool unit stays intact.
 *
 * When `messages` contains no HumanMessage (degenerate state — e.g. system
 * + assistant messages from a programmatic preamble), everything is
 * placed in the head and the tail is empty.  The summarize node treats
 * an empty tail as "nothing recent to preserve" and falls through to its
 * existing logic.
 */
export function splitAtRecencyBoundary(
  messages: BaseMessage[],
  options: RecencyWindowOptions = {}
): RecencySplit {
  const turnsCap = options.turns ?? DEFAULT_RETAIN_RECENT_TURNS;

  if (messages.length === 0 || turnsCap <= 0) {
    return {
      head: messages,
      tail: [],
      tailTurnCount: 0,
      tailStartIndex: messages.length,
      usedIntraTurnFallback: false,
    };
  }

  const turnStarts = findTurnStarts(messages, options.provider);

  if (turnStarts.length === 0) {
    return {
      head: messages,
      tail: [],
      tailTurnCount: 0,
      tailStartIndex: messages.length,
      usedIntraTurnFallback: false,
    };
  }

  const lastTurnStart = turnStarts[turnStarts.length - 1] as number;
  let tailStartIndex = lastTurnStart;
  let tailTurnCount = 1;

  const tokensCap = options.tokens;
  const tokenCounter = options.tokenCounter;
  const trackTokens =
    tokensCap != null && Number.isFinite(tokensCap) && tokenCounter != null;
  const messageTokens: Array<number | undefined> = new Array(messages.length);
  const countMessageAt = (index: number): number => {
    const cached = messageTokens[index];
    if (cached != null) {
      return cached;
    }
    const count = tokenCounter?.(messages[index] as BaseMessage) ?? 0;
    messageTokens[index] = count;
    return count;
  };

  /**
   * Token-counting strategy: each candidate turn `t` spans the half-open
   * range `[turnStarts[t], turnStarts[t + 1])` (or `[turnStarts[t], messages.length)`
   * for the most recent turn).  Successive iterations of the outer loop
   * walk older turns one at a time and never revisit messages from a
   * later turn — so each message contributes to `tokenCounter` at most
   * once across the entire selection, making the boundary search
   * `O(messages_in_visited_turns)` and bounded by `O(messages.length)`
   * even before the `turnsCap` short-circuit applies.  The inner upper
   * bound uses `turnStarts[t + 1]` (a value derived from immutable
   * `turnStarts`) rather than the mutated `tailStartIndex` to make the
   * disjoint-range invariant self-evident.
   */
  let tailTokens = 0;
  if (trackTokens) {
    for (let i = lastTurnStart; i < messages.length; i++) {
      tailTokens += countMessageAt(i);
    }
  }

  for (let t = turnStarts.length - 2; t >= 0; t--) {
    if (tailTurnCount >= turnsCap) {
      break;
    }
    const turnStart = turnStarts[t] as number;
    const turnEnd = turnStarts[t + 1] as number;

    if (trackTokens) {
      let turnTokens = 0;
      for (let i = turnStart; i < turnEnd; i++) {
        turnTokens += countMessageAt(i);
      }
      if (tailTokens + turnTokens > (tokensCap as number)) {
        break;
      }
      tailTokens += turnTokens;
    }

    tailStartIndex = turnStart;
    tailTurnCount += 1;
  }

  if (tailStartIndex === turnStarts[0]) {
    const intraTurnTokens = options.intraTurnTokens;
    if (
      intraTurnTokens != null &&
      Number.isFinite(intraTurnTokens) &&
      intraTurnTokens > 0 &&
      tokenCounter != null
    ) {
      const intraTurnBoundary = findIntraTurnBoundary(
        messages,
        countMessageAt,
        intraTurnTokens,
        turnStarts[1] ?? messages.length,
        options.provider
      );
      if (intraTurnBoundary != null) {
        return {
          head: messages.slice(0, intraTurnBoundary),
          tail: messages.slice(intraTurnBoundary),
          tailTurnCount,
          tailStartIndex: intraTurnBoundary,
          usedIntraTurnFallback: true,
        };
      }
    }
  }

  return {
    head: messages.slice(0, tailStartIndex),
    tail: messages.slice(tailStartIndex),
    tailTurnCount,
    tailStartIndex,
    usedIntraTurnFallback: false,
  };
}

/** Safe suffix starts, including closed intra-turn tool exchanges. */
export function getPairingBalancedSuffixStarts(
  messages: BaseMessage[],
  provider?: ProviderName
): number[] {
  const lastIndexBySourceId = new Map<string, number>();
  const sourceIds = messages.map((message, index) => {
    const ids = getCoverageSourceIds(message);
    for (const id of ids) lastIndexBySourceId.set(id, index);
    return ids;
  });
  const calls: ProviderToolCallIndex = new Map();
  const straddling = new Set<string>();
  const starts = [0];
  let newestExchangeStart = 0;
  for (let i = 0; i < messages.length; i++) {
    const beforeIsBalanced = calls.size === 0 && straddling.size === 0;
    const pairing = inspectMessagePairing(messages[i], calls, provider);
    if (beforeIsBalanced && Number(calls.size) > 0) newestExchangeStart = i;
    if (pairing.userMessage && !pairing.trustedHumanToolResult) {
      newestExchangeStart = i;
      if (i > 0 && beforeIsBalanced && starts[starts.length - 1] !== i) {
        starts.push(i);
      }
    }
    for (const id of sourceIds[i]) {
      if (lastIndexBySourceId.get(id) === i) straddling.delete(id);
      else straddling.add(id);
    }
    if (
      i + 1 < messages.length &&
      pairing.completedToolCalls > 0 &&
      calls.size === 0 &&
      straddling.size === 0
    ) {
      starts.push(i + 1);
    }
  }
  while (starts[starts.length - 1] > newestExchangeStart) starts.pop();
  return starts;
}

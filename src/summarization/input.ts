import { ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderName, TokenCounter } from '@/types';
import {
  cloneToolMessageWithContent,
  compactToolContent,
} from '@/utils/toolContent';

export const DEFAULT_SUMMARY_OUTPUT_TOKENS = 4_096;
export const DEFAULT_RETAIN_RECENT_TOKENS = 16_000;

export class SummarizationInputTooLargeError extends Error {
  constructor() {
    super('Summarization context exceeds the available token budget');
    this.name = 'SummarizationInputTooLargeError';
  }
}

/** Compresses tool output, then omits only the tail retained verbatim after compaction. */
export function boundSummarizationInput({
  messages,
  tokenCounter,
  budget,
  requiredHeadLength = messages.length,
  shrink = false,
}: {
  messages: BaseMessage[];
  tokenCounter: TokenCounter;
  budget: number;
  provider: ProviderName;
  requiredHeadLength?: number;
  shrink?: boolean;
}): BaseMessage[] {
  const counts = new WeakMap<BaseMessage, number>();
  const count = (message: BaseMessage): number => {
    const cached = counts.get(message);
    if (cached != null) return cached;
    const tokens = tokenCounter(message);
    counts.set(message, tokens);
    return tokens;
  };
  if (!Number.isSafeInteger(requiredHeadLength) || requiredHeadLength < 0 || requiredHeadLength > messages.length) {
    throw new SummarizationInputTooLargeError();
  }
  let total = 0;
  for (const message of messages) total += count(message);
  if (!shrink && total <= budget) return messages;
  if (budget <= 0) throw new SummarizationInputTooLargeError();

  const minimal = messages.map((message) => {
    if (!(message instanceof ToolMessage)) return message;
    const result = compactToolContent(message.content, 0);
    if (!result.changed) return message;
    const candidate = cloneToolMessageWithContent(message, result.content);
    return count(candidate) < count(message) ? candidate : message;
  });
  let minimumTokens = 0;
  let requiredTokens = 0;
  let toolCount = 0;
  let requiredToolCount = 0;
  for (let i = 0; i < messages.length; i++) {
    const tokens = count(minimal[i]);
    minimumTokens += tokens;
    if (messages[i] instanceof ToolMessage) toolCount++;
    if (i < requiredHeadLength) {
      requiredTokens += tokens;
      if (messages[i] instanceof ToolMessage) requiredToolCount++;
    }
  }
  const includesTail = minimumTokens <= budget;
  if (!includesTail && requiredTokens > budget) throw new SummarizationInputTooLargeError();
  const end = includesTail ? messages.length : requiredHeadLength;
  const additionalTokens = budget - (includesTail ? minimumTokens : requiredTokens);
  const maxChars = Math.max(0, Math.floor((additionalTokens * 2) / Math.max(1, includesTail ? toolCount : requiredToolCount)));
  for (const scale of [1, 0.5, 0.25, 0.125, 0]) {
    let tokens = 0;
    const candidate: BaseMessage[] = [];
    for (let i = 0; i < end; i++) {
      const original = messages[i];
      let message = original;
      if (original instanceof ToolMessage) {
        message = minimal[i];
        if (scale > 0) {
          const result = compactToolContent(original.content, Math.floor(maxChars * scale));
          if (!result.changed) message = original;
          else {
            const compacted = cloneToolMessageWithContent(original, result.content);
            message = count(compacted) < count(original) ? compacted : original;
          }
        }
      }
      tokens += count(message);
      candidate.push(message);
    }
    if (tokens <= budget) return candidate;
  }
  throw new SummarizationInputTooLargeError();
}

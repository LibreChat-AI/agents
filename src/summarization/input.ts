import { ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderName, TokenCounter } from '@/types';
import {
  cloneToolMessageWithContent,
  compactToolContent,
} from '@/utils/toolContent';
import { getPairingBalancedSuffixStarts } from '@/messages/recency';

export const DEFAULT_SUMMARY_OUTPUT_TOKENS = 4_096;
export const DEFAULT_RETAIN_RECENT_TOKENS = 16_000;

export class SummarizationInputTooLargeError extends Error {
  constructor() {
    super('Summarization context exceeds the available token budget');
    this.name = 'SummarizationInputTooLargeError';
  }
}

/** Bounds a provider window without ever removing its newest exchange. */
export function boundSummarizationInput({
  messages,
  tokenCounter,
  budget,
  provider,
  shrink = false,
}: {
  messages: BaseMessage[];
  tokenCounter: TokenCounter;
  budget: number;
  provider: ProviderName;
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
  const minimumSuffix = new Array<number>(messages.length + 1);
  const toolCounts = new Array<number>(messages.length + 1);
  minimumSuffix[messages.length] = 0;
  toolCounts[messages.length] = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    minimumSuffix[i] = minimumSuffix[i + 1] + count(minimal[i]);
    toolCounts[i] = toolCounts[i + 1] + Number(messages[i] instanceof ToolMessage);
  }
  const starts = getPairingBalancedSuffixStarts(messages, provider);
  const start = starts.find((index) => minimumSuffix[index] <= budget);
  if (start == null) throw new SummarizationInputTooLargeError();

  const additionalTokens = budget - minimumSuffix[start];
  const maxChars = Math.max(0, Math.floor((additionalTokens * 2) / Math.max(1, toolCounts[start])));
  for (const scale of [1, 0.5, 0.25, 0.125, 0]) {
    let tokens = 0;
    const candidate: BaseMessage[] = [];
    for (let i = start; i < messages.length; i++) {
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

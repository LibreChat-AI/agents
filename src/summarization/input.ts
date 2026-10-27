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
  const count = (message: BaseMessage): number => tokenCounter(message);
  let total = 0;
  for (const message of messages) total += count(message);
  if (!shrink && total <= budget) return messages;
  if (budget <= 0) throw new SummarizationInputTooLargeError();

  let toolCount = 0;
  for (const message of messages) {
    if (message instanceof ToolMessage) toolCount++;
  }
  const maxToolChars = Math.max(
    0,
    Math.floor((budget * 2) / Math.max(1, toolCount))
  );
  const compacted =
    toolCount === 0
      ? messages
      : messages.map((message) => {
        if (!(message instanceof ToolMessage)) return message;
        const result = compactToolContent(message.content, maxToolChars);
        return result.changed
          ? cloneToolMessageWithContent(message, result.content)
          : message;
      });
  const suffixTokens = new Array<number>(compacted.length + 1);
  suffixTokens[compacted.length] = 0;
  for (let i = compacted.length - 1; i >= 0; i--) {
    suffixTokens[i] = suffixTokens[i + 1] + count(compacted[i]);
  }
  const starts = getPairingBalancedSuffixStarts(compacted, provider);
  for (const start of starts) {
    if (suffixTokens[start] <= budget) {
      return start === 0 ? compacted : compacted.slice(start);
    }
  }
  throw new SummarizationInputTooLargeError();
}

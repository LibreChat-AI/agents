import { AIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { ContentTypes, Providers } from '@/common';

const REASONING_REJECTION = /signature|thinking|reasoning|redacted/i;

/**
 * Whether a Bedrock error is the service refusing replayed reasoning (a
 * validation error naming the signature, thinking or reasoning blocks) rather
 * than any other bad request.
 */
export function isReasoningRejection(error: unknown): boolean {
  const failure = error as
    | { name?: string; message?: string; $metadata?: { httpStatusCode?: number } }
    | null
    | undefined;
  if (failure == null || typeof failure !== 'object') {
    return false;
  }
  const rejected =
    failure.name === 'ValidationException' ||
    failure.$metadata?.httpStatusCode === 400;
  return rejected && REASONING_REJECTION.test(failure.message ?? '');
}

/** Signed reasoning block types per provider; only the model that signed them accepts them. */
const SIGNED_REASONING_TYPES: Partial<Record<string, ReadonlySet<string>>> = {
  [Providers.ANTHROPIC]: new Set([ContentTypes.THINKING, 'redacted_thinking']),
  [Providers.BEDROCK]: new Set([ContentTypes.REASONING_CONTENT]),
};

/**
 * The messages without the signed reasoning blocks of `provider`. The same
 * array when there is none, so a caller can tell a retry would send the same
 * request.
 */
export function stripSignedReasoning(
  messages: BaseMessage[],
  provider: string
): BaseMessage[] {
  const types = SIGNED_REASONING_TYPES[provider];
  if (types == null) {
    return messages;
  }
  let changed = false;
  const stripped = messages.flatMap((message) => {
    if (message.getType() !== 'ai' || !Array.isArray(message.content)) {
      return [message];
    }
    const aiMessage = message as AIMessage;
    const content = (aiMessage.content as Array<{ type?: string }>).filter(
      (part) => !types.has(part.type ?? '')
    );
    if (content.length === aiMessage.content.length) {
      return [message];
    }
    changed = true;
    if (
      content.length === 0 &&
      (aiMessage.tool_calls?.length ?? 0) === 0 &&
      (aiMessage.invalid_tool_calls?.length ?? 0) === 0
    ) {
      return [];
    }
    return [
      new AIMessage({
        ...aiMessage.lc_kwargs,
        content: content as AIMessage['content'],
      }),
    ];
  });
  return changed ? stripped : messages;
}

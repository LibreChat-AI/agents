import { AIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { ContentTypes } from '@/common';

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

/**
 * The messages without any signed reasoning block. The same array when there is
 * none, so a caller can tell a retry would send the same request.
 */
export function stripBedrockReasoning(messages: BaseMessage[]): BaseMessage[] {
  const state = { changed: false };
  const stripped = messages.flatMap((message) => {
    if (message.getType() !== 'ai' || !Array.isArray(message.content)) {
      return [message];
    }
    const aiMessage = message as AIMessage;
    const content = (aiMessage.content as Array<{ type?: string }>).filter(
      (part) => part.type !== ContentTypes.REASONING_CONTENT
    );
    if (content.length === aiMessage.content.length) {
      return [message];
    }
    state.changed = true;
    if (
      content.length === 0 &&
      (aiMessage.tool_calls?.length ?? 0) === 0 &&
      (aiMessage.invalid_tool_calls?.length ?? 0) === 0
    ) {
      return [];
    }
    return [
      new AIMessage({
        id: aiMessage.id,
        name: aiMessage.name,
        content: content as AIMessage['content'],
        additional_kwargs: aiMessage.additional_kwargs,
        response_metadata: aiMessage.response_metadata,
        tool_calls: aiMessage.tool_calls,
        invalid_tool_calls: aiMessage.invalid_tool_calls,
        usage_metadata: aiMessage.usage_metadata,
      }),
    ];
  });
  return state.changed ? stripped : messages;
}

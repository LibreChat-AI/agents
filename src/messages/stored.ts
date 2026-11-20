import {
  isAIMessage,
  isAIMessageChunk,
  isToolMessage,
  isChatMessage,
  isBaseMessageChunk,
} from '@langchain/core/messages';
import type { BaseMessage, StoredMessage } from '@langchain/core/messages';

/**
 * Native stored-message data without LangChain's JSON-oriented object escaping.
 * Values retain their native types and references: callers must snapshot/encode
 * with their own bounded codec before retaining or transmitting the result.
 */
export function toStoredMessage(message: BaseMessage): StoredMessage {
  if (Object.keys(message.lc_secrets ?? {}).length > 0) {
    throw new Error('SDK native message secrets require an explicit codec');
  }
  const data = {
    ...message.lc_kwargs,
    content: message.content,
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
    id: message.id,
    name: message.name,
    role: isChatMessage(message) ? message.role : undefined,
    tool_call_id: undefined,
    ...(isAIMessage(message)
      ? {
        tool_calls: message.tool_calls,
        invalid_tool_calls: message.invalid_tool_calls,
        usage_metadata: message.usage_metadata,
      }
      : {}),
    ...(isBaseMessageChunk(message) && isAIMessageChunk(message)
      ? { tool_call_chunks: message.tool_call_chunks }
      : {}),
    ...(isToolMessage(message)
      ? {
        tool_call_id: message.tool_call_id,
        artifact: message.artifact,
        status: message.status,
      }
      : {}),
  };
  // LangChain's StoredMessageData types content as string, but its codec supports native blocks.
  return { type: message.getType(), data } as StoredMessage;
}

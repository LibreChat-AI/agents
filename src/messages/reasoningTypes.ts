import { createHash } from 'node:crypto';
import { AIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';

const REASONING_CONTENT_BLOCK_TYPES: ReadonlySet<string> = new Set([
  'think',
  'thinking',
  'thinking_delta',
  'redacted_thinking',
  'reasoning',
  'reasoning-delta',
  'reasoning_content',
]);

/** Identifies every persisted or streamed reasoning content-block shape. */
export function isReasoningContentBlock(block: { type?: unknown }): boolean {
  return (
    typeof block.type === 'string' &&
    REASONING_CONTENT_BLOCK_TYPES.has(block.type)
  );
}

export function isAnthropicThinkingContentBlock(block: {
  type?: unknown;
}): boolean {
  return block.type === 'thinking';
}

interface SignedReasoningCandidate {
  type?: unknown;
  signature?: unknown;
  reasoningText?: { signature?: unknown } | null;
  redactedContent?: unknown;
}

/**
 * Blocks the provider signs over the conversation before them: Anthropic
 * `thinking` / `redacted_thinking`, signed `reasoning`, and Bedrock
 * `reasoning_content` carrying a signature or redacted payload.
 */
export function isSignedReasoningBlock(block: SignedReasoningCandidate): boolean {
  switch (block.type) {
  case 'thinking':
  case 'redacted_thinking':
    return true;
  case 'reasoning':
    return typeof block.signature === 'string';
  case 'reasoning_content':
    return (
      typeof block.reasoningText?.signature === 'string' ||
        block.redactedContent != null
    );
  default:
    return false;
  }
}

/**
 * Messages without the signed reasoning of their AI turns. A signature binds
 * the block to the prefix it was produced under, so once a summary replaces
 * that prefix the block can no longer be sent. A turn left with no content and
 * no tool calls is dropped.
 */
export function stripSignedReasoning(messages: BaseMessage[]): BaseMessage[] {
  return messages.flatMap((message) => {
    if (message.getType() !== 'ai' || !Array.isArray(message.content)) {
      return [message];
    }
    const aiMessage = message as AIMessage;
    const content = (message.content as SignedReasoningCandidate[]).filter(
      (part) => !isSignedReasoningBlock(part)
    );
    if (content.length === message.content.length) {
      return [message];
    }
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
}

/** What a request sent, kept to tell whether the next one rewrote any of it. */
export interface SentPrefix {
  digests?: string[];
  /** Digests of turns whose signed reasoning was taken off for good. */
  stale: Set<string>;
}

function digestMessage(message: BaseMessage): string {
  const { tool_calls: toolCalls, tool_call_id: toolCallId } = message as {
    tool_calls?: unknown;
    tool_call_id?: unknown;
  };
  return createHash('sha1')
    .update(
      JSON.stringify([message.getType(), message.content, toolCalls, toolCallId])
    )
    .digest('hex');
}

function hasSignedReasoning(message: BaseMessage): boolean {
  return (
    message.getType() === 'ai' &&
    Array.isArray(message.content) &&
    (message.content as SignedReasoningCandidate[]).some(isSignedReasoningBlock)
  );
}

/**
 * Takes the signed reasoning off the turns whose prefix this request changed.
 * A block is bound to every message before it as it was sent with that turn's
 * request, so when pruning, truncation or masking rewrote an earlier message
 * after the block was produced, the provider no longer accepts it. The turns
 * stay, tool calls included; a turn once stripped stays stripped, so later
 * requests keep one byte-identical prefix. The first request has nothing to
 * compare with and keeps what the history carries. `dropped` lists the turns
 * stripped for the first time by this request.
 */
export function dropReasoningOfRewrittenPrefix(
  messages: BaseMessage[],
  sent: SentPrefix
): { messages: BaseMessage[]; dropped: BaseMessage[] } {
  const digests = messages.map(digestMessage);
  const previous = sent.digests;
  let rewrittenAt = Infinity;
  if (previous != null) {
    rewrittenAt = Math.min(previous.length, digests.length);
    for (let i = 0; i < rewrittenAt; i++) {
      if (previous[i] !== digests[i]) {
        rewrittenAt = i;
        break;
      }
    }
  }
  sent.digests = digests;
  const dropped: BaseMessage[] = [];
  let changed = false;
  const kept = messages.map((message, index) => {
    if (!hasSignedReasoning(message)) {
      return message;
    }
    const alreadyStale = sent.stale.has(digests[index]);
    if (!alreadyStale && index > rewrittenAt) {
      sent.stale.add(digests[index]);
      dropped.push(message);
    } else if (!alreadyStale) {
      return message;
    }
    const [stripped] = stripSignedReasoning([message]);
    changed = changed || stripped != null;
    return stripped ?? message;
  });
  return { messages: changed ? kept : messages, dropped };
}

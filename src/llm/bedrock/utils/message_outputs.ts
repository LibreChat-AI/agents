/**
 * Utility functions for converting Bedrock Converse responses to LangChain messages.
 * Ported from @langchain/aws common.js
 */
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type {
  BedrockContentBlock,
  BedrockMessage,
  ConverseResponse,
  ContentBlockDeltaEvent,
  ConverseStreamMetadataEvent,
  ContentBlockStartEvent,
  ReasoningContentBlock,
  ReasoningContentBlockDelta,
  MessageContentReasoningBlock,
  MessageContentReasoningBlockReasoningTextPartial,
  MessageContentReasoningBlockRedacted,
} from '../types';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { toLangChainContent } from '@/messages/langchain';

/**
 * Convert a Bedrock reasoning block delta to a LangChain partial reasoning block.
 */
export function bedrockReasoningDeltaToLangchainPartialReasoningBlock(
  reasoningContent: ReasoningContentBlockDelta
):
  | MessageContentReasoningBlockReasoningTextPartial
  | MessageContentReasoningBlockRedacted {
  const { text, redactedContent, signature } =
    reasoningContent as ReasoningContentBlockDelta & {
      text?: string;
      redactedContent?: Uint8Array;
      signature?: string;
    };

  if (typeof text === 'string') {
    return {
      type: 'reasoning_content',
      reasoningText: { text },
    };
  }
  if (signature != null) {
    return {
      type: 'reasoning_content',
      reasoningText: { signature },
    };
  }
  if (redactedContent != null) {
    return {
      type: 'reasoning_content',
      redactedContent: Buffer.from(redactedContent).toString('base64'),
    };
  }
  throw new Error('Invalid reasoning content');
}

/**
 * Convert a Bedrock reasoning block to a LangChain reasoning block.
 */
export function bedrockReasoningBlockToLangchainReasoningBlock(
  reasoningContent: ReasoningContentBlock
): MessageContentReasoningBlock {
  const { reasoningText, redactedContent } =
    reasoningContent as ReasoningContentBlock & {
      reasoningText?: { text?: string; signature?: string };
      redactedContent?: Uint8Array;
    };

  if (reasoningText != null) {
    return {
      type: 'reasoning_content',
      reasoningText: reasoningText,
    };
  }
  if (redactedContent != null) {
    return {
      type: 'reasoning_content',
      redactedContent: Buffer.from(redactedContent).toString('base64'),
    };
  }
  throw new Error('Invalid reasoning content');
}

type BedrockResponseContentBlock =
  | {
      type: 'cache_point';
      cachePoint: NonNullable<BedrockContentBlock['cachePoint']>;
    }
  | {
      type: 'citations_content';
      citationsContent: NonNullable<BedrockContentBlock['citationsContent']>;
    }
  | {
      type: 'document';
      document: NonNullable<BedrockContentBlock['document']>;
    }
  | {
      type: 'guard_content';
      guardContent: NonNullable<BedrockContentBlock['guardContent']>;
    }
  | { type: 'image'; image: NonNullable<BedrockContentBlock['image']> }
  | MessageContentReasoningBlock
  | { type: 'text'; text: string }
  | {
      type: 'tool_result';
      toolResult: NonNullable<BedrockContentBlock['toolResult']>;
    }
  | { type: 'video'; video: NonNullable<BedrockContentBlock['video']> }
  | { type: 'non_standard'; value: Record<string, unknown> };

const BEDROCK_RESPONSE_PROVIDER = 'bedrock-converse' as const;

/**
 * Convert a Bedrock Converse message to a LangChain message.
 */
export function convertConverseMessageToLangChainMessage(
  message: BedrockMessage,
  responseMetadata: Omit<ConverseResponse, 'output'>
): AIMessage {
  if (message.content == null) {
    throw new Error('No message content found in response.');
  }
  if (message.role !== 'assistant') {
    throw new Error(
      `Unsupported message role received in ChatBedrockConverse response: ${message.role}`
    );
  }

  let requestId: string | undefined;
  if (
    '$metadata' in responseMetadata &&
    responseMetadata.$metadata != null &&
    typeof responseMetadata.$metadata === 'object' &&
    'requestId' in responseMetadata.$metadata &&
    typeof responseMetadata.$metadata.requestId === 'string'
  ) {
    requestId = responseMetadata.$metadata.requestId;
  }

  let tokenUsage: UsageMetadata | undefined;
  if (responseMetadata.usage != null) {
    const usage = responseMetadata.usage;
    const cacheReadInputTokens = usage.cacheReadInputTokens ?? 0;
    const cacheWriteInputTokens = usage.cacheWriteInputTokens ?? 0;
    const input_tokens =
      (usage.inputTokens ?? 0) + cacheReadInputTokens + cacheWriteInputTokens;
    const output_tokens = usage.outputTokens ?? 0;
    const inputTokenDetails = {
      ...(usage.cacheReadInputTokens !== undefined && {
        cache_read: usage.cacheReadInputTokens,
      }),
      ...(usage.cacheWriteInputTokens !== undefined && {
        cache_creation: usage.cacheWriteInputTokens,
      }),
    };
    tokenUsage = {
      input_tokens,
      output_tokens,
      total_tokens: usage.totalTokens ?? input_tokens + output_tokens,
      input_token_details:
        Object.keys(inputTokenDetails).length > 0 ? inputTokenDetails : undefined,
    };
  }

  const normalizedResponseMetadata = {
    ...responseMetadata,
    model_provider: BEDROCK_RESPONSE_PROVIDER,
  };
  if (
    message.content.length === 1 &&
    'text' in message.content[0] &&
    typeof message.content[0].text === 'string'
  ) {
    return new AIMessage({
      content: message.content[0].text,
      response_metadata: normalizedResponseMetadata,
      usage_metadata: tokenUsage,
      id: requestId,
    });
  }

  const toolCalls: Array<{
    id?: string;
    name: string;
    args: Record<string, unknown>;
    type: 'tool_call';
  }> = [];
  const content: BedrockResponseContentBlock[] = [];

  message.content.forEach((block) => {
    if ('cachePoint' in block && block.cachePoint != null) {
      content.push({ type: 'cache_point', cachePoint: block.cachePoint });
    } else if ('citationsContent' in block && block.citationsContent != null) {
      content.push({
        type: 'citations_content',
        citationsContent: block.citationsContent,
      });
    } else if ('document' in block && block.document != null) {
      content.push({ type: 'document', document: block.document });
    } else if ('guardContent' in block && block.guardContent != null) {
      content.push({ type: 'guard_content', guardContent: block.guardContent });
    } else if ('image' in block && block.image != null) {
      content.push({ type: 'image', image: block.image });
    } else if ('reasoningContent' in block && block.reasoningContent != null) {
      content.push(
        bedrockReasoningBlockToLangchainReasoningBlock(block.reasoningContent)
      );
    } else if ('text' in block && typeof block.text === 'string') {
      content.push({ type: 'text', text: block.text });
    } else if ('toolResult' in block && block.toolResult != null) {
      content.push({ type: 'tool_result', toolResult: block.toolResult });
    } else if (
      'toolUse' in block &&
      block.toolUse != null &&
      block.toolUse.name != null &&
      block.toolUse.name !== '' &&
      block.toolUse.input != null &&
      typeof block.toolUse.input === 'object' &&
      !Array.isArray(block.toolUse.input)
    ) {
      toolCalls.push({
        id: block.toolUse.toolUseId,
        name: block.toolUse.name,
        args: block.toolUse.input,
        type: 'tool_call',
      });
    } else if ('video' in block && block.video != null) {
      content.push({ type: 'video', video: block.video });
    } else {
      content.push({ type: 'non_standard', value: { ...block } });
    }
  });

  return new AIMessage({
    content: content.length ? content : '',
    tool_calls: toolCalls.length ? toolCalls : undefined,
    response_metadata: normalizedResponseMetadata,
    usage_metadata: tokenUsage,
    id: requestId,
  });
}

/**
 * Handle a content block delta event from Bedrock Converse stream.
 */
export function handleConverseStreamContentBlockDelta(
  contentBlockDelta: ContentBlockDeltaEvent
): ChatGenerationChunk {
  if (contentBlockDelta.delta == null) {
    throw new Error('No delta found in content block.');
  }

  if (typeof contentBlockDelta.delta.text === 'string') {
    return new ChatGenerationChunk({
      text: contentBlockDelta.delta.text,
      message: new AIMessageChunk({
        content: contentBlockDelta.delta.text,
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
        },
      }),
    });
  } else if (contentBlockDelta.delta.toolUse != null) {
    const index = contentBlockDelta.contentBlockIndex;
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            args: contentBlockDelta.delta.toolUse.input as string,
            index,
            type: 'tool_call_chunk',
          },
        ],
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
          [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
            BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
        },
      }),
    });
  } else if (contentBlockDelta.delta.reasoningContent != null) {
    const reasoningBlock =
      bedrockReasoningDeltaToLangchainPartialReasoningBlock(
        contentBlockDelta.delta.reasoningContent
      );
    let reasoningText = '';
    if ('reasoningText' in reasoningBlock) {
      reasoningText = reasoningBlock.reasoningText.text ?? '';
    } else if ('redactedContent' in reasoningBlock) {
      reasoningText = reasoningBlock.redactedContent;
    }
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: toLangChainContent([reasoningBlock]),
        additional_kwargs: {
          // Set reasoning_content for stream handler to detect reasoning mode
          reasoning_content: reasoningText,
        },
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
        },
      }),
    });
  } else {
    throw new Error(
      `Unsupported content block type(s): ${JSON.stringify(contentBlockDelta.delta, null, 2)}`
    );
  }
}

/**
 * Handle a content block start event from Bedrock Converse stream.
 */
export function handleConverseStreamContentBlockStart(
  contentBlockStart: ContentBlockStartEvent
): ChatGenerationChunk | null {
  const index = contentBlockStart.contentBlockIndex;

  if (contentBlockStart.start?.toolUse != null) {
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            name: contentBlockStart.start.toolUse.name,
            id: contentBlockStart.start.toolUse.toolUseId,
            index,
            type: 'tool_call_chunk',
          },
        ],
        response_metadata: {
          contentBlockIndex: index,
          [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
            BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
        },
      }),
    });
  }

  // Return null for non-tool content block starts (text blocks don't need special handling)
  return null;
}

/**
 * Build the chunk emitted when a Converse `contentBlockStop` event closes a
 * toolUse block. The Converse protocol guarantees a block's input is complete
 * at `contentBlockStop`, so this chunk carries an explicit streamed tool-call
 * seal for that block index. The empty `args` delta merges as a no-op into the
 * accumulated tool call; id/name are omitted so the chunk matches the existing
 * entry purely by index.
 */
export function createConverseToolUseStopChunk(
  contentBlockIndex: number
): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          args: '',
          index: contentBlockIndex,
          type: 'tool_call_chunk',
        },
      ],
      response_metadata: {
        [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
          BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: {
          kind: 'single',
          index: contentBlockIndex,
        },
      },
    }),
  });
}

/**
 * Handle a metadata event from Bedrock Converse stream.
 */
export function handleConverseStreamMetadata(
  metadata: ConverseStreamMetadataEvent,
  extra: { streamUsage: boolean }
): ChatGenerationChunk {
  const usage = metadata.usage as
    | (NonNullable<ConverseStreamMetadataEvent['usage']> & {
        cacheReadInputTokens?: number;
        cacheWriteInputTokens?: number;
      })
    | undefined;
  const inputTokens = usage?.inputTokens ?? 0;
  const outputTokens = usage?.outputTokens ?? 0;
  const cacheRead = usage?.cacheReadInputTokens;
  const cacheWrite = usage?.cacheWriteInputTokens;

  const usage_metadata: Record<string, unknown> = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: usage?.totalTokens ?? inputTokens + outputTokens,
  };

  if (cacheRead != null || cacheWrite != null) {
    usage_metadata.input_token_details = {
      cache_read: cacheRead ?? 0,
      cache_creation: cacheWrite ?? 0,
    };
  }

  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      usage_metadata: extra.streamUsage
        ? (usage_metadata as UsageMetadata)
        : undefined,
      response_metadata: {
        // Use the same key as returned from the Converse API
        metadata,
      },
    }),
  });
}

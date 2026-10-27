/** This util file contains functions for converting Anthropic messages to LangChain messages. */
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ChatGeneration } from '@langchain/core/outputs';
import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicMessageResponse } from '../types';
import type { MessageContentComplex } from '@/types';
import { toLangChainContent } from '@/messages/langchain';
import { extractToolCalls } from './output_parsers';

export interface AnthropicUsageData {
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

export function getAnthropicUsageMetadata(
  usage: AnthropicUsageData | null | undefined
): UsageMetadata | undefined {
  if (usage == null) {
    return undefined;
  }

  const cacheCreationInputTokens = usage.cache_creation_input_tokens ?? 0;
  const cacheReadInputTokens = usage.cache_read_input_tokens ?? 0;
  // Anthropic reports uncached input separately from cache creation/read tokens.
  const inputTokens =
    (usage.input_tokens ?? 0) + cacheCreationInputTokens + cacheReadInputTokens;
  const outputTokens = usage.output_tokens ?? 0;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    input_token_details: {
      cache_creation: cacheCreationInputTokens,
      cache_read: cacheReadInputTokens,
    },
  };
}

function _isAnthropicCompactionBlock(
  block: unknown
): block is Anthropic.Beta.BetaCompactionBlockParam {
  return (
    typeof block === 'object' &&
    block !== null &&
    'type' in block &&
    block.type === 'compaction'
  );
}

export function _makeMessageChunkFromAnthropicEvent(
  data: Anthropic.Beta.Messages.BetaRawMessageStreamEvent,
  fields: {
    streamUsage: boolean;
    coerceContentToString: boolean;
  }
): {
  chunk: AIMessageChunk;
} | null {
  const responseMetadata = { model_provider: 'anthropic' };
  if (data.type === 'message_start') {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { content, usage, ...additionalKwargs } = data.message;
    const {
      input_tokens: _inputTokens,
      output_tokens: _outputTokens,
      ...rest
    } = usage;
    const usageMetadata = getAnthropicUsageMetadata(usage);
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? '' : [],
        additional_kwargs: additionalKwargs,
        usage_metadata: fields.streamUsage ? usageMetadata : undefined,
        response_metadata: {
          ...responseMetadata,
          usage: {
            ...rest,
          },
        },
        id: data.message.id,
      }),
    };
  } else if (data.type === 'message_delta') {
    const messageDeltaResponseMetadata = { ...responseMetadata };
    if ('context_management' in data.delta) {
      Object.assign(messageDeltaResponseMetadata, {
        context_management: data.delta.context_management,
      });
    }
    const deltaUsage: AnthropicUsageData = data.usage;
    const hasInputUsage =
      deltaUsage.input_tokens != null ||
      deltaUsage.cache_creation_input_tokens != null ||
      deltaUsage.cache_read_input_tokens != null;
    const cumulativeUsageMetadata = hasInputUsage
      ? getAnthropicUsageMetadata(deltaUsage)
      : undefined;
    const usageMetadata: UsageMetadata = cumulativeUsageMetadata ?? {
      input_tokens: 0,
      output_tokens: data.usage.output_tokens,
      total_tokens: data.usage.output_tokens,
    };
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? '' : [],
        response_metadata: messageDeltaResponseMetadata,
        additional_kwargs: { ...data.delta },
        usage_metadata: fields.streamUsage ? usageMetadata : undefined,
      }),
    };
  } else if (
    data.type === 'content_block_start' &&
    [
      'tool_use',
      'document',
      'server_tool_use',
      'web_search_tool_result',
    ].includes(data.content_block.type)
  ) {
    const contentBlock = data.content_block;
    let toolCallChunks: ToolCallChunk[];
    if (contentBlock.type === 'tool_use') {
      toolCallChunks = [
        {
          id: contentBlock.id,
          index: data.index,
          name: contentBlock.name,
          args: '',
        },
      ];
    } else if (contentBlock.type === 'server_tool_use') {
      // Handle anthropic built-in server tool use (like web search)
      toolCallChunks = [
        {
          id: contentBlock.id,
          index: data.index,
          name: contentBlock.name,
          args: '',
        },
      ];
    } else {
      toolCallChunks = [];
    }
    const content = [
      {
        index: data.index,
        ...data.content_block,
        input:
          contentBlock.type === 'server_tool_use' ||
          contentBlock.type === 'tool_use'
            ? ''
            : undefined,
      },
    ];
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? '' : content,
        response_metadata: responseMetadata,
        additional_kwargs: {},
        tool_call_chunks: toolCallChunks,
      }),
    };
  } else if (
    data.type === 'content_block_delta' &&
    [
      'text_delta',
      'citations_delta',
      'thinking_delta',
      'signature_delta',
    ].includes(data.delta.type)
  ) {
    if (fields.coerceContentToString && 'text' in data.delta) {
      return {
        chunk: new AIMessageChunk({
          content: data.delta.text,
        }),
      };
    } else {
      const contentBlock: Record<string, unknown> = { ...data.delta };
      if ('citation' in contentBlock) {
        contentBlock.citations = [contentBlock.citation];
        delete contentBlock.citation;
      }
      if (
        contentBlock.type === 'thinking_delta' ||
        contentBlock.type === 'signature_delta'
      ) {
        return {
          chunk: new AIMessageChunk({
            content: [{ index: data.index, ...contentBlock, type: 'thinking' }],
            response_metadata: responseMetadata,
          }),
        };
      }

      return {
        chunk: new AIMessageChunk({
          content: [{ index: data.index, ...contentBlock, type: 'text' }],
          response_metadata: responseMetadata,
        }),
      };
    }
  } else if (
    data.type === 'content_block_delta' &&
    data.delta.type === 'input_json_delta'
  ) {
    const content = [
      {
        // No `type`: core's streaming aggregation merges this partial input into the
        // sibling tool_use/server_tool_use block at the same index, keeping its type.
        // A typed delta block won't merge under core >= 1.1.46 ("keep different block
        // types separate"), which would orphan the input and empty the tool_use input.
        index: data.index,
        input: data.delta.partial_json,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any,
    ];
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? '' : content,
        response_metadata: responseMetadata,
        additional_kwargs: {},
        tool_call_chunks: [
          {
            index: data.index,
            args: data.delta.partial_json,
          },
        ],
      }),
    };
  } else if (
    data.type === 'content_block_start' &&
    data.content_block.type === 'text'
  ) {
    const content = data.content_block.text;
    const contentBlock = [
      {
        index: data.index,
        ...data.content_block,
      },
    ];
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? content : contentBlock,
        response_metadata: responseMetadata,
        additional_kwargs: {},
      }),
    };
  } else if (
    data.type === 'content_block_start' &&
    data.content_block.type === 'redacted_thinking'
  ) {
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString
          ? ''
          : [{ index: data.index, ...data.content_block }],
        response_metadata: responseMetadata,
      }),
    };
  } else if (
    data.type === 'content_block_start' &&
    data.content_block.type === 'thinking'
  ) {
    const content = data.content_block.thinking;
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString
          ? content
          : [{ index: data.index, ...data.content_block }],
        response_metadata: responseMetadata,
      }),
    };
  } else if (
    data.type === 'content_block_start' &&
    _isAnthropicCompactionBlock(data.content_block)
  ) {
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString
          ? ''
          : [{ index: data.index, ...data.content_block }],
        response_metadata: responseMetadata,
      }),
    };
  } else if (
    data.type === 'content_block_delta' &&
    data.delta.type === 'compaction_delta'
  ) {
    const content = [
      {
        index: data.index,
        ...data.delta,
        type: 'compaction',
      },
    ];
    return {
      chunk: new AIMessageChunk({
        content: fields.coerceContentToString ? '' : content,
        response_metadata: responseMetadata,
      }),
    };
  }
  return null;
}

export function anthropicResponseToChatMessages(
  messages: AnthropicMessageResponse[],
  additionalKwargs: Record<string, unknown>
): ChatGeneration[] {
  const responseMetadata = {
    ...additionalKwargs,
    model_provider: 'anthropic',
  };
  const usage = additionalKwargs.usage as AnthropicUsageData | null | undefined;
  const usageMetadata = getAnthropicUsageMetadata(usage);
  if (messages.length === 1 && messages[0].type === 'text') {
    return [
      {
        text: messages[0].text,
        message: new AIMessage({
          content: messages[0].text,
          additional_kwargs: additionalKwargs,
          usage_metadata: usageMetadata,
          response_metadata: responseMetadata,
          id: additionalKwargs.id as string,
        }),
      },
    ];
  } else {
    const toolCalls = extractToolCalls(messages);
    const generations: ChatGeneration[] = [
      {
        text: '',
        message: new AIMessage({
          content: toLangChainContent(messages as MessageContentComplex[]),
          additional_kwargs: additionalKwargs,
          tool_calls: toolCalls,
          usage_metadata: usageMetadata,
          response_metadata: responseMetadata,
          id: additionalKwargs.id as string,
        }),
      },
    ];
    return generations;
  }
}

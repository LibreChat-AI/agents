/* eslint-disable @typescript-eslint/ban-ts-comment */
/* eslint-disable @typescript-eslint/explicit-function-return-type */
import { type OpenAI as OpenAIClient } from 'openai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  makeInvalidToolCall,
  parseToolCall,
} from '@langchain/core/output_parsers/openai_tools';
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  ChatMessage,
  ToolMessage,
  isAIMessage,
  type UsageMetadata,
  type BaseMessageFields,
  type MessageContentComplex,
  type InvalidToolCall,
  type MessageContentImageUrl,
  StandardContentBlockConverter,
  parseBase64DataUrl,
  parseMimeType,
  convertToProviderContentBlock,
  isDataContentBlock,
} from '@langchain/core/messages';
import type {
  ChatCompletionContentPartText,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartInputAudio,
  ChatCompletionContentPart,
} from 'openai/resources/chat/completions';
import type {
  OpenAICallOptions,
  OpenAIChatInput,
  ChatOpenAIReasoningSummary,
} from '@langchain/openai';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import {
  getBoundedCacheControlledTextToolContent,
  getComputerCallOutputScreenshot,
  isComputerCallOutputMessage,
  serializeStructuredValueBounded,
} from '@/utils/toolContent';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  HARD_MAX_TOOL_RESULT_CHARS,
  calculateMaxToolCallInputChars,
} from '@/utils/truncation';
import {
  projectToolCallInputs,
  serializeToolCallInput,
} from '@/messages/prune';
import { toLangChainContent } from '@/messages/langchain';
import { isAnthropicThinkingContentBlock } from '@/messages/reasoningTypes';

export type { OpenAICallOptions, OpenAIChatInput };

// Utility types to get hidden OpenAI response types
type ExtractAsyncIterableType<T> = T extends AsyncIterable<infer U> ? U : never;
type ExcludeController<T> = T extends { controller: unknown } ? never : T;
type ExcludeNonController<T> = T extends { controller: unknown } ? T : never;

type ResponsesCreate = OpenAIClient.Responses['create'];
type ResponsesParse = OpenAIClient.Responses['parse'];

type ResponsesInputItem = OpenAIClient.Responses.ResponseInputItem;

type ResponsesCreateInvoke = ExcludeController<
  Awaited<ReturnType<ResponsesCreate>>
>;

type ResponsesParseInvoke = ExcludeController<
  Awaited<ReturnType<ResponsesParse>>
>;

type ResponsesCreateStream = ExcludeNonController<
  Awaited<ReturnType<ResponsesCreate>>
>;

export type ResponseReturnStreamEvents =
  ExtractAsyncIterableType<ResponsesCreateStream>;

// TODO import from SDK when available
type OpenAIRoleEnum =
  | 'system'
  | 'developer'
  | 'assistant'
  | 'user'
  | 'function'
  | 'tool';

type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;

const MAX_PROVIDER_TOOL_CALL_INPUT_CHARS = calculateMaxToolCallInputChars();

function convertLangChainToolCallToBoundedOpenAI(toolCall: ToolCall): {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
} {
  if (toolCall.id == null) {
    throw new Error('All OpenAI tool calls must have an "id" field.');
  }
  return {
    id: toolCall.id,
    type: 'function',
    function: {
      name: toolCall.name,
      arguments: serializeToolCallInput(
        toolCall.args,
        MAX_PROVIDER_TOOL_CALL_INPUT_CHARS
      ),
    },
  };
}

function extractGenericMessageCustomRole(message: ChatMessage) {
  if (
    message.role !== 'system' &&
    message.role !== 'developer' &&
    message.role !== 'assistant' &&
    message.role !== 'user' &&
    message.role !== 'function' &&
    message.role !== 'tool'
  ) {
    console.warn(`Unknown message role: ${message.role}`);
  }

  return message.role as OpenAIRoleEnum;
}

export function messageToOpenAIRole(message: BaseMessage): OpenAIRoleEnum {
  const type = message._getType();
  switch (type) {
  case 'system':
    return 'system';
  case 'ai':
    return 'assistant';
  case 'human':
    return 'user';
  case 'function':
    return 'function';
  case 'tool':
    return 'tool';
  case 'generic': {
    if (!ChatMessage.isInstance(message))
      throw new Error('Invalid generic chat message');
    return extractGenericMessageCustomRole(message);
  }
  default:
    throw new Error(`Unknown message type: ${type}`);
  }
}

const completionsApiContentBlockConverter: StandardContentBlockConverter<{
  text: ChatCompletionContentPartText;
  image: ChatCompletionContentPartImage;
  audio: ChatCompletionContentPartInputAudio;
  file: ChatCompletionContentPart.File;
}> = {
  providerName: 'ChatOpenAI',

  fromStandardTextBlock(block): ChatCompletionContentPartText {
    return { type: 'text', text: block.text };
  },

  fromStandardImageBlock(block): ChatCompletionContentPartImage {
    if (block.source_type === 'url') {
      return {
        type: 'image_url',
        image_url: {
          url: block.url,
          ...(block.metadata?.detail
            ? { detail: block.metadata.detail as 'auto' | 'low' | 'high' }
            : {}),
        },
      };
    }

    if (block.source_type === 'base64') {
      const url = `data:${block.mime_type ?? ''};base64,${block.data}`;
      return {
        type: 'image_url',
        image_url: {
          url,
          ...(block.metadata?.detail
            ? { detail: block.metadata.detail as 'auto' | 'low' | 'high' }
            : {}),
        },
      };
    }

    throw new Error(
      `Image content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },

  fromStandardAudioBlock(block): ChatCompletionContentPartInputAudio {
    if (block.source_type === 'url') {
      const data = parseBase64DataUrl({ dataUrl: block.url });
      if (!data) {
        throw new Error(
          `URL audio blocks with source_type ${block.source_type} must be formatted as a data URL for ChatOpenAI`
        );
      }

      const rawMimeType = data.mime_type || block.mime_type || '';
      let mimeType: { type: string; subtype: string };

      try {
        mimeType = parseMimeType(rawMimeType);
      } catch {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      if (
        mimeType.type !== 'audio' ||
        (mimeType.subtype !== 'wav' && mimeType.subtype !== 'mp3')
      ) {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      return {
        type: 'input_audio',
        input_audio: {
          format: mimeType.subtype,
          data: data.data,
        },
      };
    }

    if (block.source_type === 'base64') {
      let mimeType: { type: string; subtype: string };

      try {
        mimeType = parseMimeType(block.mime_type ?? '');
      } catch {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      if (
        mimeType.type !== 'audio' ||
        (mimeType.subtype !== 'wav' && mimeType.subtype !== 'mp3')
      ) {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      return {
        type: 'input_audio',
        input_audio: {
          format: mimeType.subtype,
          data: block.data,
        },
      };
    }

    throw new Error(
      `Audio content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },

  fromStandardFileBlock(block): ChatCompletionContentPart.File {
    if (block.source_type === 'url') {
      const data = parseBase64DataUrl({ dataUrl: block.url });
      if (!data) {
        throw new Error(
          `URL file blocks with source_type ${block.source_type} must be formatted as a data URL for ChatOpenAI`
        );
      }

      return {
        type: 'file',
        file: {
          file_data: block.url, // formatted as base64 data URL
          ...(block.metadata?.filename || block.metadata?.name
            ? {
              filename: (block.metadata.filename ||
                  block.metadata.name) as string,
            }
            : {}),
        },
      };
    }

    if (block.source_type === 'base64') {
      return {
        type: 'file',
        file: {
          file_data: `data:${block.mime_type ?? ''};base64,${block.data}`,
          ...(block.metadata?.filename ||
          block.metadata?.name ||
          block.metadata?.title
            ? {
              filename: (block.metadata.filename ||
                  block.metadata.name ||
                  block.metadata.title) as string,
            }
            : {}),
        },
      };
    }

    if (block.source_type === 'id') {
      return {
        type: 'file',
        file: {
          file_id: block.id,
        },
      };
    }

    throw new Error(
      `File content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },
};

/** Options for converting messages to OpenAI params */
export interface ConvertMessagesOptions {
  /** Include reasoning_content field for DeepSeek thinking mode with tool calls */
  includeReasoningContent?: boolean;
  /** Include reasoning_details field for OpenRouter/Gemini thinking mode with tool calls */
  includeReasoningDetails?: boolean;
  /** Convert reasoning_details to content blocks for Claude (requires content array format) */
  convertReasoningDetailsToContent?: boolean;
  /** Preserve OpenRouter's canonical cache-decorated tool text block. */
  preserveToolCacheControl?: boolean;
}

// Used in LangSmith, export is important here
export function _convertMessagesToOpenAIParams(
  messages: BaseMessage[],
  model?: string,
  options?: ConvertMessagesOptions
): OpenAICompletionParam[] {
  const projectedMessages = projectToolCallInputs(
    messages,
    MAX_PROVIDER_TOOL_CALL_INPUT_CHARS
  );
  let hasReasoningToolCallContext = false;
  // TODO: Function messages do not support array content, fix cast
  return projectedMessages.flatMap((message) => {
    let role = messageToOpenAIRole(message);
    if (role === 'system' && isReasoningModel(model)) {
      role = 'developer';
    }

    let hasReasoningBlock = false;

    let content: unknown;
    if (
      role === 'tool' &&
      typeof message.content !== 'string' &&
      !isComputerCallOutputMessage(message)
    ) {
      content =
        options?.preserveToolCacheControl === true
          ? getBoundedCacheControlledTextToolContent(
            message.content,
            HARD_MAX_TOOL_RESULT_CHARS
          )
          : undefined;
      content ??= serializeStructuredValueBounded(
        message.content,
        HARD_MAX_TOOL_RESULT_CHARS
      ).content;
    } else if (typeof message.content === 'string') {
      content = message.content;
    } else {
      content = message.content.map((m) => {
        if (isAnthropicThinkingContentBlock(m)) {
          hasReasoningBlock = true;
          return m;
        }
        if (isDataContentBlock(m)) {
          return convertToProviderContentBlock(
            m,
            completionsApiContentBlockConverter
          );
        }
        return m;
      });
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const completionParam: Record<string, any> = {
      role,
      content,
    };
    let messageHasToolCalls = false;
    let messageIsToolResult = false;
    if (message.name != null) {
      completionParam.name = message.name;
    }
    if (message.additional_kwargs.function_call != null) {
      completionParam.function_call = message.additional_kwargs.function_call;
      completionParam.content = '';
    }
    if (isAIMessage(message) && !!message.tool_calls?.length) {
      messageHasToolCalls = true;
      completionParam.tool_calls = message.tool_calls.map(
        convertLangChainToolCallToBoundedOpenAI
      );
      completionParam.content = hasReasoningBlock ? content : '';
      if (
        options?.includeReasoningDetails === true &&
        message.additional_kwargs.reasoning_details != null
      ) {
        // For Claude via OpenRouter, convert reasoning_details to content blocks
        const isClaudeModel =
          model?.includes('claude') === true ||
          model?.includes('anthropic') === true;
        if (
          options.convertReasoningDetailsToContent === true &&
          isClaudeModel
        ) {
          const reasoningDetails = message.additional_kwargs
            .reasoning_details as Record<string, unknown>[];
          const contentBlocks = [];

          // Add thinking blocks from reasoning_details
          for (const detail of reasoningDetails) {
            if (detail.type === 'reasoning.text' && detail.text != null) {
              contentBlocks.push({
                type: 'thinking',
                thinking: detail.text,
              });
            } else if (
              detail.type === 'reasoning.encrypted' &&
              detail.data != null
            ) {
              contentBlocks.push({
                type: 'redacted_thinking',
                data: detail.data,
                id: detail.id,
              });
            }
          }

          // Set content to array with thinking blocks
          if (contentBlocks.length > 0) {
            completionParam.content = contentBlocks;
          }
        } else {
          // For non-Claude models, pass as separate field
          completionParam.reasoning_details =
            message.additional_kwargs.reasoning_details;
        }
      }
    } else {
      if (message.additional_kwargs.tool_calls != null) {
        messageHasToolCalls =
          !Array.isArray(message.additional_kwargs.tool_calls) ||
          message.additional_kwargs.tool_calls.length > 0;
        completionParam.tool_calls = message.additional_kwargs.tool_calls;
        if (
          options?.includeReasoningDetails === true &&
          message.additional_kwargs.reasoning_details != null
        ) {
          // For Claude via OpenRouter, convert reasoning_details to content blocks
          const isClaudeModel =
            model?.includes('claude') === true ||
            model?.includes('anthropic') === true;
          if (
            options.convertReasoningDetailsToContent === true &&
            isClaudeModel
          ) {
            const reasoningDetails = message.additional_kwargs
              .reasoning_details as Record<string, unknown>[];
            const contentBlocks = [];

            // Add thinking blocks from reasoning_details
            for (const detail of reasoningDetails) {
              if (detail.type === 'reasoning.text' && detail.text != null) {
                contentBlocks.push({
                  type: 'thinking',
                  thinking: detail.text,
                });
              } else if (
                detail.type === 'reasoning.encrypted' &&
                detail.data != null
              ) {
                contentBlocks.push({
                  type: 'redacted_thinking',
                  data: detail.data,
                  id: detail.id,
                });
              }
            }

            // Set content to array with thinking blocks
            if (contentBlocks.length > 0) {
              completionParam.content = contentBlocks;
            }
          } else {
            // For non-Claude models, pass as separate field
            completionParam.reasoning_details =
              message.additional_kwargs.reasoning_details;
          }
        }
      }
      if ((message as ToolMessage).tool_call_id != null) {
        messageIsToolResult = true;
        completionParam.tool_call_id = (message as ToolMessage).tool_call_id;
      }
    }

    if (
      options?.includeReasoningContent === true &&
      isAIMessage(message) &&
      (hasReasoningToolCallContext || messageHasToolCalls) &&
      typeof message.additional_kwargs.reasoning_content === 'string' &&
      message.additional_kwargs.reasoning_content !== ''
    ) {
      completionParam.reasoning_content =
        message.additional_kwargs.reasoning_content;
    }

    if (messageHasToolCalls || messageIsToolResult) {
      hasReasoningToolCallContext = true;
    }

    if (
      message.additional_kwargs.audio &&
      typeof message.additional_kwargs.audio === 'object' &&
      'id' in message.additional_kwargs.audio
    ) {
      const audioMessage = {
        role: 'assistant',
        audio: {
          id: message.additional_kwargs.audio.id,
        },
      };
      return [completionParam, audioMessage] as OpenAICompletionParam[];
    }

    return completionParam as OpenAICompletionParam;
  });
}

const _FUNCTION_CALL_IDS_MAP_KEY = '__openai_function_call_ids__';

function _convertReasoningSummaryToOpenAIResponsesParams(
  reasoning: ChatOpenAIReasoningSummary
): OpenAIClient.Responses.ResponseReasoningItem {
  // combine summary parts that have the the same index and then remove the indexes
  const summary = (
    reasoning.summary.length > 1
      ? reasoning.summary.reduce(
        (acc, curr) => {
          const last = acc.at(-1);

          if (last!.index === curr.index) {
              last!.text += curr.text;
          } else {
            acc.push(curr);
          }
          return acc;
        },
        [{ ...reasoning.summary[0] }]
      )
      : reasoning.summary
  ).map((s) =>
    Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'index'))
  ) as OpenAIClient.Responses.ResponseReasoningItem.Summary[];

  return {
    ...reasoning,
    summary,
  } as OpenAIClient.Responses.ResponseReasoningItem;
}

export function _convertMessagesToOpenAIResponsesParams(
  messages: BaseMessage[],
  model?: string,
  zdrEnabled?: boolean
): ResponsesInputItem[] {
  const projectedMessages = projectToolCallInputs(
    messages,
    MAX_PROVIDER_TOOL_CALL_INPUT_CHARS
  );
  return projectedMessages.flatMap(
    (lcMsg): ResponsesInputItem | ResponsesInputItem[] => {
      const additional_kwargs =
        lcMsg.additional_kwargs as BaseMessageFields['additional_kwargs'] & {
          [_FUNCTION_CALL_IDS_MAP_KEY]?: Record<string, string>;
          reasoning?: OpenAIClient.Responses.ResponseReasoningItem;
          type?: string;
          refusal?: string;
        };
      const responseMetadata = lcMsg.response_metadata as {
        output?: ResponsesInputItem[];
      };

      let role = messageToOpenAIRole(lcMsg);
      if (role === 'system' && isReasoningModel(model)) role = 'developer';

      if (role === 'function') {
        throw new Error('Function messages are not supported in Responses API');
      }

      if (role === 'tool') {
        const toolMessage = lcMsg as ToolMessage;

        // Handle computer call output
        if (additional_kwargs.type === 'computer_call_output') {
          const screenshot = getComputerCallOutputScreenshot(
            toolMessage.content
          );
          if (screenshot == null) {
            throw new Error('Invalid computer call output');
          }

          const output: OpenAIClient.Responses.ResponseComputerToolCallOutputScreenshot =
            'image_url' in screenshot
              ? {
                type: 'computer_screenshot',
                image_url: screenshot.image_url,
              }
              : {
                type: 'computer_screenshot',
                file_id: screenshot.file_id,
              };
          const computerCallOutput: ResponsesInputItem = {
            type: 'computer_call_output',
            output,
            call_id: toolMessage.tool_call_id,
          };
          return computerCallOutput;
        }

        return {
          type: 'function_call_output',
          call_id: toolMessage.tool_call_id,
          id: toolMessage.id?.startsWith('fc_') ? toolMessage.id : undefined,
          output:
            typeof toolMessage.content !== 'string'
              ? serializeStructuredValueBounded(
                toolMessage.content,
                HARD_MAX_TOOL_RESULT_CHARS
              ).content
              : toolMessage.content,
        };
      }

      if (role === 'assistant') {
        // if we have the original response items, just reuse them
        if (
          !zdrEnabled &&
          responseMetadata.output != null &&
          Array.isArray(responseMetadata.output) &&
          responseMetadata.output.length > 0 &&
          responseMetadata.output.every((item) => 'type' in item)
        ) {
          return responseMetadata.output;
        }

        // otherwise, try to reconstruct the response from what we have

        const input: ResponsesInputItem[] = [];

        // reasoning items
        if (additional_kwargs.reasoning && !zdrEnabled) {
          const reasoningItem = _convertReasoningSummaryToOpenAIResponsesParams(
            additional_kwargs.reasoning
          );
          input.push(reasoningItem);
        }

        // ai content
        let content = lcMsg.content as
          | string
          | Array<
              | MessageContentComplex
              | OpenAIClient.Responses.ResponseOutputText
              | OpenAIClient.Responses.ResponseOutputRefusal
            >;
        if (additional_kwargs.refusal) {
          if (typeof content === 'string') {
            content = [{ type: 'output_text', text: content, annotations: [] }];
          }
          content = [
            ...content,
            { type: 'refusal', refusal: additional_kwargs.refusal },
          ];
        }

        input.push({
          type: 'message',
          role: 'assistant',
          ...(lcMsg.id && !zdrEnabled && lcMsg.id.startsWith('msg_')
            ? { id: lcMsg.id }
            : {}),
          content:
            typeof content === 'string'
              ? content
              : content.flatMap((item) => {
                if (item.type === 'text') {
                  const textItem = item as MessageContentComplex & {
                      annotations?: unknown[];
                    };
                  return {
                    type: 'output_text',
                    text: item.text,
                    annotations: textItem.annotations ?? [],
                  };
                }

                if (item.type === 'output_text' || item.type === 'refusal') {
                  return item;
                }

                return [];
              }),
        } as ResponsesInputItem);

        const functionCallIds = additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY];

        if (isAIMessage(lcMsg) && !!lcMsg.tool_calls?.length) {
          input.push(
            ...lcMsg.tool_calls.map(
              (toolCall): ResponsesInputItem => ({
                type: 'function_call',
                name: toolCall.name,
                arguments: serializeToolCallInput(
                  toolCall.args,
                  MAX_PROVIDER_TOOL_CALL_INPUT_CHARS
                ),
                call_id: toolCall.id!,
                ...(zdrEnabled ? { id: functionCallIds?.[toolCall.id!] } : {}),
              })
            )
          );
        } else if (additional_kwargs.tool_calls) {
          input.push(
            ...additional_kwargs.tool_calls.map(
              (toolCall): ResponsesInputItem => ({
                type: 'function_call',
                name: toolCall.function.name,
                call_id: toolCall.id,
                arguments: toolCall.function.arguments,
                ...(zdrEnabled ? { id: functionCallIds?.[toolCall.id] } : {}),
              })
            )
          );
        }

        const toolOutputs =
          ((responseMetadata.output as Array<ResponsesInputItem> | undefined)
            ?.length ?? 0) > 0
            ? responseMetadata.output
            : additional_kwargs.tool_outputs;

        const fallthroughCallTypes: ResponsesInputItem['type'][] = [
          'computer_call',
          /** @ts-ignore */
          'mcp_call',
          /** @ts-ignore */
          'code_interpreter_call',
          /** @ts-ignore */
          'image_generation_call',
        ];

        if (toolOutputs != null) {
          const castToolOutputs = toolOutputs as Array<ResponsesInputItem>;
          const fallthroughCalls = castToolOutputs.filter((item) =>
            fallthroughCallTypes.includes(item.type)
          );
          if (fallthroughCalls.length > 0) input.push(...fallthroughCalls);
        }

        return input;
      }

      if (role === 'user' || role === 'system' || role === 'developer') {
        if (typeof lcMsg.content === 'string') {
          return { type: 'message', role, content: lcMsg.content };
        }

        const messages: ResponsesInputItem[] = [];
        const content = (lcMsg.content as MessageContentComplex[]).flatMap(
          (item) => {
            if (item.type === 'mcp_approval_response') {
              const approvalResponse = item as MessageContentComplex & {
                approval_request_id: string;
                approve: boolean;
              };
              messages.push({
                // @ts-ignore
                type: 'mcp_approval_response',
                approval_request_id: approvalResponse.approval_request_id,
                approve: approvalResponse.approve,
              });
            }
            if (isDataContentBlock(item)) {
              return convertToProviderContentBlock(
                item,
                completionsApiContentBlockConverter
              );
            }
            if (item.type === 'text') {
              return {
                type: 'input_text',
                text: item.text,
              };
            }
            if (item.type === 'image_url') {
              const imageItem = item as MessageContentImageUrl;
              return {
                type: 'input_image',
                image_url:
                  typeof imageItem.image_url === 'string'
                    ? imageItem.image_url
                    : imageItem.image_url.url,
                detail:
                  typeof imageItem.image_url === 'string'
                    ? 'auto'
                    : imageItem.image_url.detail,
              };
            }
            if (
              item.type === 'input_text' ||
              item.type === 'input_image' ||
              item.type === 'input_file'
            ) {
              return item;
            }
            return [];
          }
        );

        if (content.length > 0) {
          messages.push({
            type: 'message',
            role,
            content,
          } as ResponsesInputItem);
        }
        return messages;
      }

      console.warn(
        `Unsupported role found when converting to OpenAI Responses API: ${role}`
      );
      return [];
    }
  );
}

export function isReasoningModel(model?: string) {
  return model != null && model !== '' && /\b(o\d|gpt-[5-9])\b/i.test(model);
}

function _convertOpenAIResponsesMessageToBaseMessage(
  response: ResponsesCreateInvoke | ResponsesParseInvoke
): BaseMessage {
  if (response.error) {
    // TODO: add support for `addLangChainErrorFields`
    const error = new Error(response.error.message);
    error.name = response.error.code;
    throw error;
  }

  let messageId: string | undefined;
  const content: MessageContentComplex[] = [];
  const tool_calls: ToolCall[] = [];
  const invalid_tool_calls: InvalidToolCall[] = [];
  const response_metadata: Record<string, unknown> = {
    model: response.model,
    created_at: response.created_at,
    id: response.id,
    incomplete_details: response.incomplete_details,
    metadata: response.metadata,
    object: response.object,
    status: response.status,
    user: response.user,
    service_tier: response.service_tier,

    // for compatibility with chat completion calls.
    model_name: response.model,
  };

  const additional_kwargs: {
    [key: string]: unknown;
    refusal?: string;
    reasoning?: OpenAIClient.Responses.ResponseReasoningItem;
    tool_outputs?: unknown[];
    parsed?: unknown;
    [_FUNCTION_CALL_IDS_MAP_KEY]?: Record<string, string>;
  } = {};

  for (const item of response.output) {
    if (item.type === 'message') {
      messageId = item.id;
      content.push(
        ...item.content.flatMap((part) => {
          if (part.type === 'output_text') {
            if ('parsed' in part && part.parsed != null) {
              additional_kwargs.parsed = part.parsed;
            }
            return {
              type: 'text',
              text: part.text,
              annotations: part.annotations,
            };
          }

          if (part.type === 'refusal') {
            additional_kwargs.refusal = part.refusal;
            return [];
          }

          return part;
        })
      );
    } else if (item.type === 'function_call') {
      const fnAdapter = {
        function: { name: item.name, arguments: item.arguments },
        id: item.call_id,
      };

      try {
        tool_calls.push(parseToolCall(fnAdapter, { returnId: true }));
      } catch (e: unknown) {
        let errMessage: string | undefined;
        if (
          typeof e === 'object' &&
          e != null &&
          'message' in e &&
          typeof e.message === 'string'
        ) {
          errMessage = e.message;
        }
        invalid_tool_calls.push(makeInvalidToolCall(fnAdapter, errMessage));
      }

      additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] ??= {};
      if (item.id) {
        additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][item.call_id] = item.id;
      }
    } else if (item.type === 'reasoning') {
      additional_kwargs.reasoning = item;
    } else {
      additional_kwargs.tool_outputs ??= [];
      additional_kwargs.tool_outputs.push(item);
    }
  }

  return new AIMessage({
    id: messageId,
    content: toLangChainContent(content),
    tool_calls,
    invalid_tool_calls,
    usage_metadata: response.usage,
    additional_kwargs,
    response_metadata,
  });
}

export function _convertOpenAIResponsesDeltaToBaseMessageChunk(
  chunk: ResponseReturnStreamEvents
) {
  const content: MessageContentComplex[] = [];
  let generationInfo: Record<string, unknown> = {};
  let usage_metadata: UsageMetadata | undefined;
  const tool_call_chunks: ToolCallChunk[] = [];
  const response_metadata: Record<string, unknown> = {};
  const additional_kwargs: {
    [key: string]: unknown;
    reasoning?: Partial<ChatOpenAIReasoningSummary>;
    tool_outputs?: unknown[];
  } = {};
  let id: string | undefined;
  if (chunk.type === 'response.output_text.delta') {
    content.push({
      type: 'text',
      text: chunk.delta,
      index: chunk.content_index,
    });
    /** @ts-ignore */
  } else if (chunk.type === 'response.output_text_annotation.added') {
    content.push({
      type: 'text',
      text: '',
      /** @ts-ignore */
      annotations: [chunk.annotation],
      /** @ts-ignore */
      index: chunk.content_index,
    });
  } else if (
    chunk.type === 'response.output_item.added' &&
    chunk.item.type === 'message'
  ) {
    id = chunk.item.id;
  } else if (
    chunk.type === 'response.output_item.added' &&
    chunk.item.type === 'function_call'
  ) {
    response_metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY] =
      OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER;
    tool_call_chunks.push({
      type: 'tool_call_chunk',
      name: chunk.item.name,
      args: chunk.item.arguments,
      id: chunk.item.call_id,
      index: chunk.output_index,
    });

    additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {
      [chunk.item.call_id]: chunk.item.id,
    };
  } else if (
    chunk.type === 'response.output_item.done' &&
    [
      'web_search_call',
      'file_search_call',
      'computer_call',
      'code_interpreter_call',
      'mcp_call',
      'mcp_list_tools',
      'mcp_approval_request',
      'image_generation_call',
    ].includes(chunk.item.type)
  ) {
    additional_kwargs.tool_outputs = [chunk.item];
  } else if (chunk.type === 'response.created') {
    response_metadata.id = chunk.response.id;
    response_metadata.model_name = chunk.response.model;
    response_metadata.model = chunk.response.model;
  } else if (chunk.type === 'response.completed') {
    const msg = _convertOpenAIResponsesMessageToBaseMessage(chunk.response);

    usage_metadata = chunk.response.usage;
    if (chunk.response.text?.format?.type === 'json_schema') {
      additional_kwargs.parsed ??= JSON.parse(msg.text);
    }
    for (const [key, value] of Object.entries(chunk.response)) {
      if (key !== 'id') response_metadata[key] = value;
    }
  } else if (chunk.type === 'response.function_call_arguments.delta') {
    response_metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY] =
      OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER;
    tool_call_chunks.push({
      type: 'tool_call_chunk',
      args: chunk.delta,
      index: chunk.output_index,
    });
  } else if (chunk.type === 'response.function_call_arguments.done') {
    response_metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY] =
      OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER;
    response_metadata[STREAMED_TOOL_CALL_SEAL_METADATA_KEY] = {
      kind: 'single',
      index: chunk.output_index,
    };
    tool_call_chunks.push({
      type: 'tool_call_chunk',
      name: chunk.name,
      args: chunk.arguments,
      index: chunk.output_index,
    });
  } else if (
    chunk.type === 'response.web_search_call.completed' ||
    chunk.type === 'response.file_search_call.completed'
  ) {
    generationInfo = {
      tool_outputs: {
        id: chunk.item_id,
        type: chunk.type.replace('response.', '').replace('.completed', ''),
        status: 'completed',
      },
    };
  } else if (chunk.type === 'response.refusal.done') {
    additional_kwargs.refusal = chunk.refusal;
  } else if (
    chunk.type === 'response.output_item.added' &&
    'item' in chunk &&
    chunk.item.type === 'reasoning'
  ) {
    const summary: ChatOpenAIReasoningSummary['summary'] | undefined = chunk
      .item.summary
      ? chunk.item.summary.map((s, index) => ({
        ...s,
        index,
      }))
      : undefined;

    additional_kwargs.reasoning = {
      // We only capture ID in the first chunk or else the concatenated result of all chunks will
      // have an ID field that is repeated once per chunk. There is special handling for the `type`
      // field that prevents this, however.
      id: chunk.item.id,
      type: chunk.item.type,
      ...(summary ? { summary } : {}),
    };
  } else if (chunk.type === 'response.reasoning_summary_part.added') {
    additional_kwargs.reasoning = {
      type: 'reasoning',
      summary: [{ ...chunk.part, index: chunk.summary_index }],
    };
  } else if (chunk.type === 'response.reasoning_summary_text.delta') {
    additional_kwargs.reasoning = {
      type: 'reasoning',
      summary: [
        { text: chunk.delta, type: 'summary_text', index: chunk.summary_index },
      ],
    };
    /** @ts-ignore */
  } else if (chunk.type === 'response.image_generation_call.partial_image') {
    // noop/fixme: retaining partial images in a message chunk means that _all_
    // partial images get kept in history, so we don't do anything here.
    return null;
  } else {
    return null;
  }

  return new ChatGenerationChunk({
    // Legacy reasons, `onLLMNewToken` should pulls this out
    text: content.map((part) => ('text' in part ? part.text : '')).join(''),
    message: new AIMessageChunk({
      id,
      content: toLangChainContent(content),
      tool_call_chunks,
      usage_metadata,
      additional_kwargs,
      response_metadata,
    }),
    generationInfo,
  });
}

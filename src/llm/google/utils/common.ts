import { v4 as uuidv4 } from 'uuid';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ToolCallChunk } from '@langchain/core/messages/tool';
import { isOpenAITool } from '@langchain/core/language_models/base';
import { isLangChainTool } from '@langchain/core/utils/function_calling';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  ChatMessage,
  ToolMessage,
  ToolMessageChunk,
  MessageContent,
  MessageContentComplex,
  UsageMetadata,
  isAIMessage,
  isBaseMessage,
  isToolMessage,
  StandardContentBlockConverter,
  parseBase64DataUrl,
  convertToProviderContentBlock,
  isDataContentBlock,
} from '@langchain/core/messages';
import {
  POSSIBLE_ROLES,
  type Part,
  type Content,
  type TextPart,
  type FileDataPart,
  type InlineDataPart,
  type FunctionCallPart,
  type GenerateContentCandidate,
  type EnhancedGenerateContentResponse,
  type FunctionDeclaration as GenerativeAIFunctionDeclaration,
  type FunctionDeclarationsTool as GoogleGenerativeAIFunctionDeclarationsTool,
} from '@google/generative-ai';
import type { ChatGeneration, ChatResult } from '@langchain/core/outputs';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  jsonSchemaToGeminiParameters,
  schemaToGenerativeAIParameters,
} from './zod_to_genai_parameters';
import { toLangChainContent } from '@/messages/langchain';
import { GoogleGenerativeAIToolType } from '../types';

export const _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY =
  '__gemini_function_call_thought_signatures__';

const DUMMY_SIGNATURE =
  'ErYCCrMCAdHtim9kOoOkrPiCNVsmlpMIKd7ZMxgiFbVQOkgp7nlLcDMzVsZwIzvuT7nQROivoXA72ccC2lSDvR0Gh7dkWaGuj7ctv6t7ZceHnecx0QYa+ix8tYpRfjhyWozQ49lWiws6+YGjCt10KRTyWsZ2h6O7iHTYJwKIRwGUHRKy/qK/6kFxJm5ML00gLq4D8s5Z6DBpp2ZlR+uF4G8jJgeWQgyHWVdx2wGYElaceVAc66tZdPQRdOHpWtgYSI1YdaXgVI8KHY3/EfNc2YqqMIulvkDBAnuMhkAjV9xmBa54Tq+ih3Im4+r3DzqhGqYdsSkhS0kZMwte4Hjs65dZzCw9lANxIqYi1DJ639WNPYihp/DCJCos7o+/EeSPJaio5sgWDyUnMGkY1atsJZ+m7pj7DD5tvQ==';

type GoogleServerSideToolPart = Part & {
  type?: 'toolCall' | 'toolResponse';
  toolCall?: object;
  toolResponse?: object;
};

type GoogleServerSideToolPartMetadata = {
  thought?: boolean;
  thoughtSignature?: string;
};

type GoogleFunctionCallWithId = FunctionCallPart['functionCall'] & {
  id?: string;
};

type GoogleFunctionResponseWithId = {
  name: string;
  response: object;
  id?: string;
};

function getGoogleFunctionId(id?: string): string | undefined {
  return id != null && id !== '' ? id : undefined;
}

function createGoogleFunctionResponsePart({
  name,
  response,
  id,
}: {
  name: string;
  response: object;
  id?: string;
}): Part {
  const functionId = getGoogleFunctionId(id);
  const functionResponse: GoogleFunctionResponseWithId = {
    name,
    response,
    ...(functionId != null ? { id: functionId } : {}),
  };
  return { functionResponse };
}

/**
 * Executes a function immediately and returns its result.
 * Functional utility similar to an Immediately Invoked Function Expression (IIFE).
 * @param fn The function to execute.
 * @returns The result of invoking fn.
 */
export const iife = <T>(fn: () => T): T => fn();

export function getMessageAuthor(message: BaseMessage): string {
  const type = message._getType();
  if (ChatMessage.isInstance(message)) {
    return message.role;
  }
  if (type === 'tool') {
    return type;
  }
  return message.name ?? type;
}

/**
 * Maps a message type to a Google Generative AI chat author.
 * @param message The message to map.
 * @param model The model to use for mapping.
 * @returns The message type mapped to a Google Generative AI chat author.
 */
export function convertAuthorToRole(
  author: string
): (typeof POSSIBLE_ROLES)[number] {
  switch (author) {
  /**
     *  Note: Gemini currently is not supporting system messages
     *  we will convert them to human messages and merge with following
     * */
  case 'supervisor':
  case 'ai':
  case 'model': // getMessageAuthor returns message.name. code ex.: return message.name ?? type;
    return 'model';
  case 'system':
    return 'system';
  case 'human':
    return 'user';
  case 'tool':
  case 'function':
    return 'function';
  default:
    throw new Error(`Unknown / unsupported author: ${author}`);
  }
}

function messageContentMedia(content: MessageContentComplex): Part {
  if ('mimeType' in content && 'data' in content) {
    return {
      inlineData: {
        mimeType: content.mimeType,
        data: content.data,
      },
    };
  }
  if ('mimeType' in content && 'fileUri' in content) {
    return {
      fileData: {
        mimeType: content.mimeType,
        fileUri: content.fileUri,
      },
    };
  }

  throw new Error('Invalid media content');
}

function isGoogleServerSideToolPart(
  content: MessageContentComplex
): content is MessageContentComplex & GoogleServerSideToolPart {
  return (
    'toolCall' in content ||
    'toolResponse' in content ||
    content.type === 'toolCall' ||
    content.type === 'toolResponse'
  );
}

function convertGoogleServerSideToolPart(
  content: MessageContentComplex & GoogleServerSideToolPart
): Part {
  const metadata: GoogleServerSideToolPartMetadata = {};
  if ('thought' in content && typeof content.thought === 'boolean') {
    metadata.thought = content.thought;
  }
  if (
    'thoughtSignature' in content &&
    typeof content.thoughtSignature === 'string'
  ) {
    metadata.thoughtSignature = content.thoughtSignature;
  }
  if ('toolCall' in content && content.toolCall != null) {
    return { toolCall: content.toolCall, ...metadata } as unknown as Part;
  }
  if ('toolResponse' in content && content.toolResponse != null) {
    return {
      toolResponse: content.toolResponse,
      ...metadata,
    } as unknown as Part;
  }

  return content as Part;
}

function convertGoogleServerSideToolResponsePart(
  part: Part
): GoogleServerSideToolPart | undefined {
  if (
    'toolCall' in part &&
    typeof part.toolCall === 'object' &&
    part.toolCall != null
  ) {
    return { ...part, type: 'toolCall', toolCall: part.toolCall };
  }
  if (
    'toolResponse' in part &&
    typeof part.toolResponse === 'object' &&
    part.toolResponse != null
  ) {
    return { ...part, type: 'toolResponse', toolResponse: part.toolResponse };
  }
  return undefined;
}

function inferToolNameFromPreviousMessages(
  message: ToolMessage | ToolMessageChunk,
  previousMessages: BaseMessage[]
): string | undefined {
  return previousMessages
    .map((msg) => {
      if (isAIMessage(msg)) {
        return msg.tool_calls ?? [];
      }
      return [];
    })
    .flat()
    .find((toolCall) => {
      return toolCall.id === message.tool_call_id;
    })?.name;
}

function _getStandardContentBlockConverter(
  isMultimodalModel: boolean
): StandardContentBlockConverter<{
  text: TextPart;
  image: FileDataPart | InlineDataPart;
  audio: FileDataPart | InlineDataPart;
  file: FileDataPart | InlineDataPart | TextPart;
}> {
  const standardContentBlockConverter: StandardContentBlockConverter<{
    text: TextPart;
    image: FileDataPart | InlineDataPart;
    audio: FileDataPart | InlineDataPart;
    file: FileDataPart | InlineDataPart | TextPart;
  }> = {
    providerName: 'Google Gemini',

    fromStandardTextBlock(block) {
      return {
        text: block.text,
      };
    },

    fromStandardImageBlock(block): FileDataPart | InlineDataPart {
      if (!isMultimodalModel) {
        throw new Error('This model does not support images');
      }
      if (block.source_type === 'url') {
        const data = parseBase64DataUrl({ dataUrl: block.url });
        if (data) {
          return {
            inlineData: {
              mimeType: data.mime_type,
              data: data.data,
            },
          };
        } else {
          return {
            fileData: {
              mimeType: block.mime_type ?? '',
              fileUri: block.url,
            },
          };
        }
      }

      if (block.source_type === 'base64') {
        return {
          inlineData: {
            mimeType: block.mime_type ?? '',
            data: block.data,
          },
        };
      }

      throw new Error(`Unsupported source type: ${block.source_type}`);
    },

    fromStandardAudioBlock(block): FileDataPart | InlineDataPart {
      if (!isMultimodalModel) {
        throw new Error('This model does not support audio');
      }
      if (block.source_type === 'url') {
        const data = parseBase64DataUrl({ dataUrl: block.url });
        if (data) {
          return {
            inlineData: {
              mimeType: data.mime_type,
              data: data.data,
            },
          };
        } else {
          return {
            fileData: {
              mimeType: block.mime_type ?? '',
              fileUri: block.url,
            },
          };
        }
      }

      if (block.source_type === 'base64') {
        return {
          inlineData: {
            mimeType: block.mime_type ?? '',
            data: block.data,
          },
        };
      }

      throw new Error(`Unsupported source type: ${block.source_type}`);
    },

    fromStandardFileBlock(block): FileDataPart | InlineDataPart | TextPart {
      if (!isMultimodalModel) {
        throw new Error('This model does not support files');
      }
      if (block.source_type === 'text') {
        return {
          text: block.text,
        };
      }
      if (block.source_type === 'url') {
        const data = parseBase64DataUrl({ dataUrl: block.url });
        if (data) {
          return {
            inlineData: {
              mimeType: data.mime_type,
              data: data.data,
            },
          };
        } else {
          return {
            fileData: {
              mimeType: block.mime_type ?? '',
              fileUri: block.url,
            },
          };
        }
      }

      if (block.source_type === 'base64') {
        return {
          inlineData: {
            mimeType: block.mime_type ?? '',
            data: block.data,
          },
        };
      }
      throw new Error(`Unsupported source type: ${block.source_type}`);
    },
  };
  return standardContentBlockConverter;
}

function _convertLangChainContentToPart(
  content: MessageContentComplex,
  isMultimodalModel: boolean
): Part | undefined {
  if (isDataContentBlock(content)) {
    return convertToProviderContentBlock(
      content,
      _getStandardContentBlockConverter(isMultimodalModel)
    );
  }

  if (isGoogleServerSideToolPart(content)) {
    return convertGoogleServerSideToolPart(content);
  }

  if (content.type === 'text') {
    return typeof content.text === 'string' && content.text !== ''
      ? {
        text: content.text,
        ...('thoughtSignature' in content &&
          typeof content.thoughtSignature === 'string'
          ? { thoughtSignature: content.thoughtSignature }
          : {}),
      }
      : undefined;
  } else if (content.type === 'executableCode') {
    return { executableCode: content.executableCode };
  } else if (content.type === 'codeExecutionResult') {
    return { codeExecutionResult: content.codeExecutionResult };
  } else if (content.type === 'image_url') {
    if (!isMultimodalModel) {
      throw new Error('This model does not support images');
    }
    let source: string;
    if (typeof content.image_url === 'string') {
      source = content.image_url;
    } else if (
      typeof content.image_url === 'object' &&
      'url' in content.image_url
    ) {
      source = content.image_url.url;
    } else {
      throw new Error('Please provide image as base64 encoded data URL');
    }
    const [dm, data] = source.split(',');
    if (!dm.startsWith('data:')) {
      throw new Error('Please provide image as base64 encoded data URL');
    }

    const [mimeType, encoding] = dm.replace(/^data:/, '').split(';');
    if (encoding !== 'base64') {
      throw new Error('Please provide image as base64 encoded data URL');
    }

    return {
      inlineData: { data, mimeType },
      ...('thoughtSignature' in content &&
      typeof content.thoughtSignature === 'string'
        ? { thoughtSignature: content.thoughtSignature }
        : {}),
    };
  } else if (content.type === 'media') {
    return messageContentMedia(content);
  } else if (content.type === 'tool_use') {
    const functionId = getGoogleFunctionId(
      typeof content.id === 'string' ? content.id : undefined
    );
    return {
      functionCall: {
        name: content.name,
        args: content.input,
        ...(functionId != null ? { id: functionId } : {}),
      },
    };
  } else if (
    content.type?.includes('/') === true &&
    // Ensure it's a single slash.
    content.type.split('/').length === 2 &&
    'data' in content &&
    typeof content.data === 'string'
  ) {
    return {
      inlineData: {
        mimeType: content.type,
        data: content.data,
      },
    };
  } else if ('functionCall' in content) {
    // No action needed here — function calls will be added later from message.tool_calls
    return undefined;
  } else {
    if ('type' in content) {
      throw new Error(`Unknown content type ${content.type}`);
    } else {
      throw new Error(`Unknown content ${JSON.stringify(content)}`);
    }
  }
}

export function convertMessageContentToParts(
  message: BaseMessage,
  isMultimodalModel: boolean,
  previousMessages: BaseMessage[],
  model?: string
): Part[] {
  if (isToolMessage(message)) {
    const messageName =
      message.name ??
      inferToolNameFromPreviousMessages(message, previousMessages);
    if (messageName === undefined) {
      throw new Error(
        `Google requires a tool name for each tool call response, and we could not infer a called tool name for ToolMessage "${message.id}" from your passed messages. Please populate a "name" field on that ToolMessage explicitly.`
      );
    }

    const result = Array.isArray(message.content)
      ? (message.content
        .map((c) => _convertLangChainContentToPart(c, isMultimodalModel))
        .filter((p) => p !== undefined) as Part[])
      : message.content;

    if (message.status === 'error') {
      return [
        createGoogleFunctionResponsePart({
          name: messageName,
          // The API expects an object with an `error` field if the function call fails.
          // `error` must be a valid object (not a string or array), so we wrap `message.content` here
          response: { error: { details: result } },
          id: message.tool_call_id,
        }),
      ];
    }

    return [
      createGoogleFunctionResponsePart({
        name: messageName,
        // again, can't have a string or array value for `response`, so we wrap it as an object here
        response: { result },
        id: message.tool_call_id,
      }),
    ];
  }

  let functionCalls: FunctionCallPart[] = [];
  const messageParts: Part[] = [];

  if (typeof message.content === 'string' && message.content) {
    messageParts.push({ text: message.content });
  }

  if (Array.isArray(message.content)) {
    messageParts.push(
      ...(message.content
        .map((c) => _convertLangChainContentToPart(c, isMultimodalModel))
        .filter((p) => p !== undefined) as Part[])
    );
  }

  const functionThoughtSignatures = (
    message.additional_kwargs as BaseMessage['additional_kwargs'] | undefined
  )?.[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] as
    | Record<string, string>
    | undefined;

  if (isAIMessage(message) && (message.tool_calls?.length ?? 0) > 0) {
    functionCalls = (message.tool_calls ?? []).map((tc) => {
      const thoughtSignature = iife(() => {
        if (tc.id != null && tc.id !== '') {
          const signature = functionThoughtSignatures?.[tc.id];
          if (signature != null && signature !== '') {
            return signature;
          }
        }
        if (model?.includes('gemini-3') === true) {
          return DUMMY_SIGNATURE;
        }
        return '';
      });
      const functionId = getGoogleFunctionId(tc.id);
      const functionCall: GoogleFunctionCallWithId = {
        name: tc.name,
        args: tc.args,
        ...(functionId != null ? { id: functionId } : {}),
      };

      return {
        functionCall,
        ...(thoughtSignature ? { thoughtSignature } : {}),
      };
    });
  }

  const parsedFunctionCallIds = new Set(
    functionCalls.flatMap((part) => {
      const functionCall = part.functionCall as GoogleFunctionCallWithId;
      return functionCall.id != null ? [functionCall.id] : [];
    })
  );
  const parsedFunctionCallNames = new Set(
    functionCalls.map((part) => part.functionCall.name)
  );
  const contentWithoutParsedMirrors = messageParts.filter((part) => {
    if (!('functionCall' in part) || part.functionCall == null) {
      return true;
    }
    const functionCall = part.functionCall as GoogleFunctionCallWithId;
    return !(
      (functionCall.id != null && parsedFunctionCallIds.has(functionCall.id)) ||
      (functionCall.id == null &&
        parsedFunctionCallNames.has(functionCall.name))
    );
  });

  return [...contentWithoutParsedMirrors, ...functionCalls];
}

export function convertBaseMessagesToContent(
  messages: BaseMessage[],
  isMultimodalModel: boolean,
  convertSystemMessageToHumanContent: boolean = false,

  model?: string
): Content[] | undefined {
  return messages.reduce<{
    content: Content[] | undefined;
    mergeWithPreviousContent: boolean;
  }>(
    (acc, message, index) => {
      if (!isBaseMessage(message)) {
        throw new Error('Unsupported message input');
      }
      const author = getMessageAuthor(message);
      if (author === 'system' && index !== 0) {
        throw new Error('System message should be the first one');
      }
      const role = convertAuthorToRole(author);

      const prevContent = acc.content?.[acc.content.length];
      if (
        !acc.mergeWithPreviousContent &&
        prevContent &&
        prevContent.role === role
      ) {
        throw new Error(
          'Google Generative AI requires alternate messages between authors'
        );
      }

      const parts = convertMessageContentToParts(
        message,
        isMultimodalModel,
        messages.slice(0, index),
        model
      );

      if (acc.mergeWithPreviousContent) {
        const prevContent = acc.content?.[acc.content.length - 1];
        if (!prevContent) {
          throw new Error(
            'There was a problem parsing your system message. Please try a prompt without one.'
          );
        }
        prevContent.parts.push(...parts);

        return {
          mergeWithPreviousContent: false,
          content: acc.content,
        };
      }
      let actualRole = role;
      if (
        actualRole === 'function' ||
        (actualRole === 'system' && !convertSystemMessageToHumanContent)
      ) {
        // GenerativeAI API will throw an error if the role is not "user" or "model."
        actualRole = 'user';
      }
      const content: Content = {
        role: actualRole,
        parts,
      };
      return {
        mergeWithPreviousContent:
          author === 'system' && !convertSystemMessageToHumanContent,
        content: [...(acc.content ?? []), content],
      };
    },
    { content: [], mergeWithPreviousContent: false }
  ).content;
}

/**
 * Gemini Flash generation from which Google rejects a request whose `contents`
 * end with a `model`-role turn (a "prefill"). Every Flash release from 3.6
 * onward enforces it, so the cutoff is derived from the model id rather than
 * enumerated - a new Flash model is covered on release with no change here.
 *
 * Scoped to Flash deliberately: dropping the turn silently degrades a working
 * prefill into a fresh generation, so the rule only widens where Google
 * documents the restriction. Lines that reject it without matching the cutoff
 * are listed in {@link NO_PREFILL_GEMINI_MODELS}.
 * @see https://ai.google.dev/gemini-api/docs/latest-model#api-changes-and-parameter-updates
 */
const NO_PREFILL_FLASH_MIN_VERSION = { major: 3, minor: 6 } as const;

/**
 * `gemini-<major>[.<minor>]-flash`, with optional suffixes (`-latest`, `-lite`).
 * Google ships both forms — `gemini-3.7-flash` and the major-only
 * `gemini-3-flash-preview` — so the minor component is optional and an omitted
 * one reads as `.0`.
 */
const GEMINI_FLASH_VERSION_PATTERN = /^gemini-(\d+)(?:\.(\d+))?-flash(?:$|-)/;

/**
 * Models that reject prefill despite predating
 * {@link NO_PREFILL_FLASH_MIN_VERSION}. Gemini 3.5 Flash-Lite enforces the
 * restriction while its sibling Gemini 3.5 Flash still accepts a trailing model
 * turn, so the 3.5 generation cannot be expressed as a version cutoff.
 */
const NO_PREFILL_GEMINI_MODELS = ['gemini-3.5-flash-lite'] as const;

export function rejectsModelTurnPrefill(model?: string): boolean {
  if (model == null || model === '') {
    return false;
  }
  const modelId = model.toLowerCase().split('/').pop() ?? '';
  const listed = NO_PREFILL_GEMINI_MODELS.some(
    (id) => modelId === id || modelId.startsWith(`${id}-`)
  );
  if (listed) {
    return true;
  }
  const match = GEMINI_FLASH_VERSION_PATTERN.exec(modelId);
  if (!match) {
    return false;
  }
  const major = Number(match[1]);
  const minor = Number(match[2] || '0');
  if (major !== NO_PREFILL_FLASH_MIN_VERSION.major) {
    return major > NO_PREFILL_FLASH_MIN_VERSION.major;
  }
  return minor >= NO_PREFILL_FLASH_MIN_VERSION.minor;
}

/**
 * Drops trailing `model`-role turns for models that reject prefill (see
 * {@link rejectsModelTurnPrefill}). Such a turn is only produced by prefill
 * flows (e.g. editing an assistant reply and resubmitting); these models return
 * HTTP 400 for it, so we drop it and let the model generate fresh from the
 * preceding user turn. No-op for every other model, preserving working prefill.
 */
export function dropUnsupportedModelTurnPrefill(
  contents: Content[] | undefined,
  model?: string
): Content[] | undefined {
  if (
    contents == null ||
    contents.length === 0 ||
    !rejectsModelTurnPrefill(model)
  ) {
    return contents;
  }
  let end = contents.length;
  while (end > 1 && contents[end - 1]?.role === 'model') {
    end -= 1;
  }
  return end === contents.length ? contents : contents.slice(0, end);
}

export function convertResponseContentToChatGenerationChunk(
  response: EnhancedGenerateContentResponse,
  extra: {
    usageMetadata?: UsageMetadata | undefined;
    index: number;
  }
): ChatGenerationChunk | null {
  if (!response.candidates || response.candidates.length === 0) {
    return null;
  }
  const [candidate] = response.candidates as [
    Partial<GenerateContentCandidate> | undefined,
  ];
  const { content: candidateContent, ...generationInfo } = candidate ?? {};

  // Extract function calls directly from parts to preserve thoughtSignature
  const functionCalls =
    (candidateContent?.parts as Part[] | undefined)?.reduce(
      (acc, p) => {
        if ('functionCall' in p && p.functionCall) {
          acc.push({
            ...p,
            id:
              'id' in p.functionCall && typeof p.functionCall.id === 'string'
                ? p.functionCall.id
                : uuidv4(),
          });
        }
        return acc;
      },
      [] as (
        | undefined
        | (FunctionCallPart & { id: string; thoughtSignature?: string })
      )[]
    ) ?? [];

  let content: MessageContent | undefined;
  // Checks if some parts do not have text. If false, it means that the content is a string.
  const reasoningParts: string[] = [];
  if (
    candidateContent != null &&
    Array.isArray(candidateContent.parts) &&
    candidateContent.parts.every(
      (p) => 'text' in p && !('thoughtSignature' in p)
    )
  ) {
    // content = candidateContent.parts.map((p) => p.text).join('');
    const textParts: string[] = [];
    for (const part of candidateContent.parts) {
      if ('thought' in part && part.thought === true) {
        reasoningParts.push(part.text ?? '');
        continue;
      }
      textParts.push(part.text ?? '');
    }
    content = textParts.join('');
  } else if (candidateContent && Array.isArray(candidateContent.parts)) {
    content = toLangChainContent(
      candidateContent.parts
        .map((p) => {
          if ('text' in p && 'thought' in p && p.thought === true) {
            reasoningParts.push(p.text ?? '');
            return undefined;
          } else if ('text' in p) {
            return {
              type: 'text',
              text: p.text,
              ...('thoughtSignature' in p
                ? { thoughtSignature: p.thoughtSignature }
                : {}),
            };
          } else if ('executableCode' in p) {
            return {
              type: 'executableCode',
              executableCode: p.executableCode,
            };
          } else if ('codeExecutionResult' in p) {
            return {
              type: 'codeExecutionResult',
              codeExecutionResult: p.codeExecutionResult,
            };
          }
          const serverSideToolPart = convertGoogleServerSideToolResponsePart(p);
          if (serverSideToolPart !== undefined) {
            return serverSideToolPart;
          }
          return p;
        })
        .filter((p) => p !== undefined)
    );
  } else {
    // no content returned - likely due to abnormal stop reason, e.g. malformed function call
    content = [];
  }

  let text = '';
  if (typeof content === 'string' && content) {
    text = content;
  } else if (Array.isArray(content)) {
    const block = content.find((b) => 'text' in b) as
      | { text: string }
      | undefined;
    text = block?.text ?? '';
  }

  const toolCallChunks: ToolCallChunk[] = [];
  if (functionCalls.length > 0) {
    toolCallChunks.push(
      ...functionCalls.map((fc) => ({
        type: 'tool_call_chunk' as const,
        id: fc?.id,
        name: fc?.functionCall.name,
        args: JSON.stringify(fc?.functionCall.args),
      }))
    );
  }

  // Extract thought signatures from function calls for Gemini 3+
  const functionThoughtSignatures = functionCalls.reduce(
    (acc, fc) => {
      if (
        fc &&
        'thoughtSignature' in fc &&
        typeof fc.thoughtSignature === 'string'
      ) {
        acc[fc.id] = fc.thoughtSignature;
      }
      return acc;
    },
    {} as Record<string, string>
  );

  const additional_kwargs: ChatGeneration['message']['additional_kwargs'] = {
    [_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY]: functionThoughtSignatures,
  };

  if (reasoningParts.length > 0) {
    additional_kwargs.reasoning = reasoningParts.join('');
  }

  if (candidate?.groundingMetadata) {
    additional_kwargs.groundingMetadata = candidate.groundingMetadata;
  }

  const isFinalChunk =
    response.candidates[0]?.finishReason === 'STOP' ||
    response.candidates[0]?.finishReason === 'MAX_TOKENS' ||
    response.candidates[0]?.finishReason === 'SAFETY';

  // The GenAI API delivers function calls as complete objects (never partial
  // arg deltas), so every call on this chunk is sealed on arrival for eager
  // tool execution.
  const response_metadata: Record<string, unknown> | undefined =
    toolCallChunks.length > 0
      ? {
        [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
            GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
      }
      : undefined;

  return new ChatGenerationChunk({
    text,
    message: new AIMessageChunk({
      content: content,
      name: !candidateContent ? undefined : candidateContent.role,
      tool_call_chunks: toolCallChunks,
      // Each chunk can have unique "generationInfo", and merging strategy is unclear,
      // so leave blank for now.
      additional_kwargs,
      response_metadata,
      usage_metadata: isFinalChunk ? extra.usageMetadata : undefined,
    }),
    generationInfo,
  });
}

/**
 * Maps a Google GenerateContentResult to a LangChain ChatResult
 */
export function mapGenerateContentResultToChatResult(
  response: EnhancedGenerateContentResponse,
  extra?: {
    usageMetadata: UsageMetadata | undefined;
  }
): ChatResult {
  if (!response.candidates || response.candidates.length === 0) {
    return {
      generations: [],
      llmOutput: {
        filters: response.promptFeedback,
      },
    };
  }
  const [candidate] = response.candidates as [
    Partial<GenerateContentCandidate> | undefined,
  ];
  const { content: candidateContent, ...generationInfo } = candidate ?? {};

  // Extract function calls directly from parts to preserve thoughtSignature
  const functionCalls =
    candidateContent?.parts.reduce(
      (acc, p) => {
        if ('functionCall' in p && p.functionCall) {
          acc.push({
            ...p,
            id:
              'id' in p.functionCall && typeof p.functionCall.id === 'string'
                ? p.functionCall.id
                : uuidv4(),
          });
        }
        return acc;
      },
      [] as (FunctionCallPart & { id: string; thoughtSignature?: string })[]
    ) ?? [];

  let content: MessageContent | undefined;
  const reasoningParts: string[] = [];
  if (
    Array.isArray(candidateContent?.parts) &&
    candidateContent.parts.length === 1 &&
    !('thoughtSignature' in candidateContent.parts[0]) &&
    (candidateContent.parts[0].text ?? '') !== '' &&
    !(
      'thought' in candidateContent.parts[0] &&
      candidateContent.parts[0].thought === true
    )
  ) {
    content = candidateContent.parts[0].text;
  } else if (
    Array.isArray(candidateContent?.parts) &&
    candidateContent.parts.length > 0
  ) {
    content = toLangChainContent(
      candidateContent.parts
        .map((p) => {
          if ('text' in p && 'thought' in p && p.thought === true) {
            reasoningParts.push(p.text ?? '');
            return undefined;
          } else if ('text' in p) {
            return {
              type: 'text',
              text: p.text,
              ...('thoughtSignature' in p
                ? { thoughtSignature: p.thoughtSignature }
                : {}),
            };
          } else if ('executableCode' in p) {
            return {
              type: 'executableCode',
              executableCode: p.executableCode,
            };
          } else if ('codeExecutionResult' in p) {
            return {
              type: 'codeExecutionResult',
              codeExecutionResult: p.codeExecutionResult,
            };
          }
          const serverSideToolPart = convertGoogleServerSideToolResponsePart(p);
          if (serverSideToolPart !== undefined) {
            return serverSideToolPart;
          }
          return p;
        })
        .filter((p) => p !== undefined)
    );
  } else {
    content = [];
  }
  let text = '';
  if (typeof content === 'string') {
    text = content;
  } else if (Array.isArray(content) && content.length > 0) {
    const block = content.find((b) => 'text' in b) as
      | { text: string }
      | undefined;
    text = block?.text ?? text;
  }

  const additional_kwargs: ChatGeneration['message']['additional_kwargs'] = {
    ...generationInfo,
  };
  if (reasoningParts.length > 0) {
    additional_kwargs.reasoning = reasoningParts.join('');
  }

  // Extract thought signatures from function calls for Gemini 3+
  const functionThoughtSignatures = functionCalls.reduce(
    (acc, fc) => {
      if ('thoughtSignature' in fc && typeof fc.thoughtSignature === 'string') {
        acc[fc.id] = fc.thoughtSignature;
      }
      return acc;
    },
    {} as Record<string, string>
  );

  const tool_calls = functionCalls.map((fc) => ({
    type: 'tool_call' as const,
    id: fc.id,
    name: fc.functionCall.name,
    args: fc.functionCall.args,
  }));

  // Store thought signatures map for later retrieval
  additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] =
    functionThoughtSignatures;

  const generation: ChatGeneration = {
    text,
    message: new AIMessage({
      content,
      tool_calls,
      additional_kwargs,
      usage_metadata: extra?.usageMetadata,
    }),
    generationInfo,
  };
  return {
    generations: [generation],
    llmOutput: {
      tokenUsage: {
        promptTokens: extra?.usageMetadata?.input_tokens,
        completionTokens: extra?.usageMetadata?.output_tokens,
        totalTokens: extra?.usageMetadata?.total_tokens,
      },
    },
  };
}

export function convertToGenerativeAITools(
  tools: GoogleGenerativeAIToolType[]
): GoogleGenerativeAIFunctionDeclarationsTool[] {
  if (
    tools.every(
      (tool) =>
        'functionDeclarations' in tool &&
        Array.isArray(tool.functionDeclarations)
    )
  ) {
    return tools as GoogleGenerativeAIFunctionDeclarationsTool[];
  }
  return [
    {
      functionDeclarations: tools.map(
        (tool): GenerativeAIFunctionDeclaration => {
          if (isLangChainTool(tool)) {
            const jsonSchema = schemaToGenerativeAIParameters(tool.schema);
            if (
              jsonSchema.type === 'object' &&
              'properties' in jsonSchema &&
              Object.keys(jsonSchema.properties).length === 0
            ) {
              return {
                name: tool.name,
                description: tool.description,
              };
            }
            return {
              name: tool.name,
              description: tool.description,
              parameters: jsonSchema,
            };
          }
          if (isOpenAITool(tool)) {
            return {
              name: tool.function.name,
              description:
                tool.function.description ?? 'A function available to call.',
              parameters: jsonSchemaToGeminiParameters(
                tool.function.parameters
              ),
            };
          }
          return tool as unknown as GenerativeAIFunctionDeclaration;
        }
      ),
    },
  ];
}

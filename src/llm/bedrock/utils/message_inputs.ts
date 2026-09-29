/**
 * Utility functions for converting LangChain messages to Bedrock Converse messages.
 * Ported from @langchain/aws common.js
 */
import {
  type BaseMessage,
  type ToolMessage,
  isAIMessage,
  type Data,
  parseBase64DataUrl,
  parseMimeType,
  MessageContentComplex,
  type StandardContentBlockConverter,
  convertToProviderContentBlock,
  isDataContentBlock,
} from '@langchain/core/messages';
import type {
  AudioFormat,
  AudioSource,
  DocumentFormat,
  DocumentSource,
  VideoFormat,
  VideoSource,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  BedrockMessage,
  BedrockSystemContentBlock,
  BedrockContentBlock,
  MessageContentReasoningBlock,
} from '../types';
import { serializeStructuredValueBounded } from '@/utils/toolContent';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { isReasoningContentBlock } from '@/messages/reasoningTypes';

/**
 * Reasoning blocks from other providers, relative to Bedrock. Bedrock's native
 * reasoning format is `reasoning_content`; these carry provider-specific
 * signatures Bedrock cannot validate, so they are dropped on a cross-provider
 * handoff (e.g. Anthropic → Bedrock) rather than crashing the conversion.
 */
/**
 * Google server-side tool blocks (`toolCall`/`toolResponse` parts from e.g.
 * URL context or Google Search). Only Google can execute these and validate
 * their thought signatures, so they are dropped on a cross-provider handoff
 * (e.g. Google → Bedrock); the assistant's answer text is kept.
 */
const FOREIGN_SERVER_TOOL_TYPES = ['toolCall', 'toolResponse'];

/**
 * Bedrock Converse rejects assistant messages with no content blocks. When
 * filtering (e.g. dropping foreign reasoning) empties an assistant turn that
 * also has no tool calls, fall back to this placeholder text.
 */
const BEDROCK_EMPTY_TEXT_PLACEHOLDER = '_';
/**
 * Whether a converted user message carries top-level text and document blocks, recorded
 * during its single conversion pass and OR-ed through merging so finalization never
 * rescans content.
 */
const userGroupShape = new WeakMap<
  BedrockMessage,
  { hasText: boolean; hasDocument: boolean }
>();

/**
 * Convert a LangChain reasoning block to a Bedrock reasoning block.
 */
export function langchainReasoningBlockToBedrockReasoningBlock(
  content: MessageContentReasoningBlock
): {
  reasoningText?: { text?: string; signature?: string };
  redactedContent?: Uint8Array;
} {
  if (content.reasoningText != null) {
    return {
      reasoningText: content.reasoningText,
    };
  }
  if (content.redactedContent != null && content.redactedContent !== '') {
    return {
      redactedContent: new Uint8Array(
        Buffer.from(content.redactedContent, 'base64')
      ),
    };
  }
  throw new Error('Invalid reasoning content');
}

/**
 * Whether a reasoning block can be serialized to a valid Bedrock
 * `reasoningContent`. Bedrock Converse rejects `reasoningText` with a null/empty
 * `text` (e.g. a signature-only block that never merged with its text), so such
 * blocks must be dropped rather than sent.
 */
function isSerializableBedrockReasoningBlock(
  content: MessageContentReasoningBlock
): boolean {
  if (content.reasoningText != null) {
    const text = content.reasoningText.text;
    return text != null && text !== '';
  }
  return content.redactedContent != null && content.redactedContent !== '';
}

function appendSerializableBedrockTextBlock(
  contentBlocks: BedrockContentBlock[],
  text: string
): boolean {
  if (text === '') {
    return false;
  }
  const cleanedText = text.replace(/\n/g, '').trim();
  if (cleanedText !== '') {
    contentBlocks.push({ text });
    return true;
  }
  const lastBlock = contentBlocks[contentBlocks.length - 1] as
    | BedrockContentBlock
    | undefined;
  if (lastBlock == null || !('text' in lastBlock)) {
    return false;
  }
  const mergedTextContent = `${lastBlock.text}${text}`;
  (lastBlock as { text: string }).text = mergedTextContent;
  return true;
}

/**
 * Concatenate consecutive reasoning blocks in content array.
 */
export function concatenateLangchainReasoningBlocks(
  content: Array<MessageContentComplex | MessageContentReasoningBlock>
): Array<MessageContentComplex | MessageContentReasoningBlock> {
  const result: Array<MessageContentComplex | MessageContentReasoningBlock> =
    [];
  let pendingReasoning: MessageContentReasoningBlock | undefined;

  const flushPendingReasoning = (): void => {
    if (pendingReasoning != null) {
      result.push(pendingReasoning);
      pendingReasoning = undefined;
    }
  };

  for (const block of content) {
    if (block.type !== 'reasoning_content') {
      flushPendingReasoning();
      result.push(block);
      continue;
    }

    const currentReasoning = block as MessageContentReasoningBlock;
    if (currentReasoning.reasoningText != null) {
      const previousText = pendingReasoning?.reasoningText?.text;
      const previousSignature = pendingReasoning?.reasoningText?.signature;
      const { text, signature } = currentReasoning.reasoningText;
      const mergedReasoningText: { text?: string; signature?: string } = {};
      if (previousText !== undefined || text !== undefined) {
        mergedReasoningText.text = (previousText ?? '') + (text ?? '');
      }
      if (previousSignature !== undefined || signature !== undefined) {
        mergedReasoningText.signature =
          (previousSignature ?? '') + (signature ?? '');
      }
      pendingReasoning = {
        type: 'reasoning_content',
        reasoningText: mergedReasoningText,
      };

      // A signature seals the accumulated reasoning text. Any following
      // reasoning block starts a new independently signed Bedrock block.
      if ('signature' in currentReasoning.reasoningText) {
        flushPendingReasoning();
      }
    }

    if (currentReasoning.redactedContent != null) {
      flushPendingReasoning();
      result.push({
        type: 'reasoning_content',
        redactedContent: currentReasoning.redactedContent,
      });
    }
  }

  flushPendingReasoning();
  return result;
}

/**
 * Extract image info from a base64 string or URL.
 */
export function extractImageInfo(base64: string): BedrockContentBlock {
  // Extract the format from the base64 string
  const formatMatch = base64.match(/^data:image\/(\w+);base64,/);
  let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
  if (formatMatch) {
    const extractedFormat = formatMatch[1].toLowerCase();
    if (['gif', 'jpeg', 'png', 'webp'].includes(extractedFormat)) {
      format = extractedFormat as typeof format;
    }
  }

  // Remove the data URL prefix if present
  const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');

  // Convert base64 to Uint8Array
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return {
    image: {
      format,
      source: {
        bytes,
      },
    },
  };
}

type MediaContentBlock = MessageContentComplex & {
  data?: string | Uint8Array;
  url?: string;
  fileId?: string;
  mimeType?: string;
};

const mimeTypeToVideoFormat: Record<string, VideoFormat> = {
  'video/flv': 'flv',
  'video/mkv': 'mkv',
  'video/mov': 'mov',
  'video/mp4': 'mp4',
  'video/mpeg': 'mpeg',
  'video/mpg': 'mpg',
  'video/three_gp': 'three_gp',
  'video/webm': 'webm',
  'video/wmv': 'wmv',
};

const mimeTypeToAudioFormat: Record<string, AudioFormat> = {
  'audio/aac': 'aac',
  'audio/flac': 'flac',
  'audio/m4a': 'm4a',
  'audio/mka': 'mka',
  'audio/mkv': 'mkv',
  'audio/mp3': 'mp3',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mpeg',
  'audio/mpga': 'mpga',
  'audio/ogg': 'ogg',
  'audio/opus': 'opus',
  'audio/pcm': 'pcm',
  'audio/wav': 'wav',
  'audio/webm': 'webm',
  'audio/x-aac': 'x-aac',
};

const mimeTypeToDocumentFormat: Partial<Record<string, DocumentFormat>> = {
  'text/csv': 'csv',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
    'docx',
  'text/html': 'html',
  'text/markdown': 'md',
  'application/pdf': 'pdf',
  'text/plain': 'txt',
  'application/vnd.ms-excel': 'xls',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
};

function base64ToBytes(data: string): Uint8Array {
  return Uint8Array.from(atob(data), (char) => char.charCodeAt(0));
}

function getMediaFormat<T extends string>(
  mimeType: string | undefined,
  formatMap: Record<string, T>
): T | undefined {
  if (mimeType == null || mimeType === '') {
    return undefined;
  }
  return formatMap[mimeType] ?? (parseMimeType(mimeType).subtype as T);
}

function resolveMediaSource(
  block: MediaContentBlock
): AudioSource | VideoSource {
  if (typeof block.data === 'string') {
    return { bytes: base64ToBytes(block.data) };
  }
  if (block.data instanceof Uint8Array) {
    return { bytes: block.data };
  }
  if (typeof block.url === 'string') {
    const parsedData = parseBase64DataUrl({
      dataUrl: block.url,
      asTypedArray: true,
    });
    if (parsedData != null) {
      return { bytes: parsedData.data as Uint8Array };
    }
    throw new Error(
      `Only base64 data URLs are supported for ${block.type} blocks with 'url' field with ChatBedrockConverse.`
    );
  }
  if (typeof block.fileId === 'string') {
    return { s3Location: { uri: block.fileId } };
  }
  throw new Error(
    `${block.type} block must include one of: 'data' (base64 string or Uint8Array), 'url' (base64 data URL), or 'fileId' (S3 URI).`
  );
}

function convertMultimodalVideoBlock(
  block: MediaContentBlock
): BedrockContentBlock {
  return {
    video: {
      format: getMediaFormat(block.mimeType, mimeTypeToVideoFormat),
      source: resolveMediaSource(block) as VideoSource,
    },
  } as BedrockContentBlock;
}

function convertMultimodalAudioBlock(
  block: MediaContentBlock
): BedrockContentBlock {
  return {
    audio: {
      format: getMediaFormat(block.mimeType, mimeTypeToAudioFormat),
      source: resolveMediaSource(block) as AudioSource,
    },
  } as BedrockContentBlock;
}

function getDocumentName(block: Data.StandardFileBlock): string {
  return (
    (block.metadata?.name as string | undefined) ??
    (block.metadata?.filename as string | undefined) ??
    (block.metadata?.title as string | undefined) ??
    globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 12)
  );
}

function getDocumentFormat(
  mimeType: string | undefined
): DocumentFormat | undefined {
  if (mimeType == null || mimeType === '') {
    return undefined;
  }
  const parsedMimeType = parseMimeType(mimeType);
  const format =
    mimeTypeToDocumentFormat[
      `${parsedMimeType.type}/${parsedMimeType.subtype}`
    ];
  if (format === undefined) {
    throw new Error(
      `Unsupported file mime type: "${mimeType}" ChatBedrockConverse only supports ${Object.keys(
        mimeTypeToDocumentFormat
      ).join(', ')} formats.`
    );
  }
  return format;
}

const standardContentBlockConverter: StandardContentBlockConverter<{
  text: BedrockContentBlock;
  image: BedrockContentBlock;
  file: BedrockContentBlock;
}> = {
  providerName: 'ChatBedrockConverse',

  fromStandardTextBlock(block: Data.StandardTextBlock): BedrockContentBlock {
    return { text: block.text };
  },

  fromStandardImageBlock(block: Data.StandardImageBlock): BedrockContentBlock {
    if (block.source_type === 'url') {
      const parsedData = parseBase64DataUrl({
        dataUrl: block.url,
        asTypedArray: true,
      });
      if (parsedData == null) {
        throw new Error(
          [
            'Only base64 data URLs are supported for image blocks with source type ',
            'url',
            ' with ChatBedrockConverse.',
          ].join(String.fromCharCode(39))
        );
      }
      return {
        image: {
          format: parseMimeType(parsedData.mime_type).subtype as
            | 'gif'
            | 'jpeg'
            | 'png'
            | 'webp',
          source: { bytes: parsedData.data as Uint8Array },
        },
      };
    }
    if (block.source_type === 'base64') {
      let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
      if (block.mime_type != null && block.mime_type !== '') {
        format = parseMimeType(block.mime_type).subtype as typeof format;
      }
      if (format != null && !['gif', 'jpeg', 'png', 'webp'].includes(format)) {
        throw new Error(
          `Unsupported image mime type: "${block.mime_type}" ChatBedrockConverse only supports "image/gif", "image/jpeg", "image/png", and "image/webp" formats.`
        );
      }
      return {
        image: {
          format,
          source: { bytes: base64ToBytes(block.data) },
        },
      };
    }
    throw new Error(
      `Image source type '${block.source_type}' not supported with ChatBedrockConverse.`
    );
  },

  fromStandardFileBlock(block: Data.StandardFileBlock): BedrockContentBlock {
    const name = getDocumentName(block);
    if (block.source_type === 'text') {
      return {
        document: {
          name,
          format: 'txt',
          source: { bytes: new TextEncoder().encode(block.text) },
        },
      } as BedrockContentBlock;
    }
    if (block.source_type === 'url') {
      const parsedData = parseBase64DataUrl({
        dataUrl: block.url,
        asTypedArray: true,
      });
      if (parsedData == null) {
        throw new Error(
          [
            'Only base64 data URLs are supported for file blocks with source type ',
            'url',
            ' with ChatBedrockConverse.',
          ].join(String.fromCharCode(39))
        );
      }
      return {
        document: {
          name,
          format: getDocumentFormat(parsedData.mime_type),
          source: { bytes: parsedData.data as Uint8Array } as DocumentSource,
        },
      } as BedrockContentBlock;
    }
    if (block.source_type === 'base64') {
      return {
        document: {
          name,
          format: getDocumentFormat(block.mime_type),
          source: { bytes: base64ToBytes(block.data) } as DocumentSource,
        },
      } as BedrockContentBlock;
    }
    throw new Error(
      `File source type '${block.source_type}' not supported with ChatBedrockConverse.`
    );
  },
};

type BedrockPromptCacheTtl = '5m' | '1h';
type NormalizedBedrockCachePoint = {
  type: 'default';
  ttl?: BedrockPromptCacheTtl;
};

/**
 * Check if a block has a default cache point and return its normalized form,
 * preserving an optional extended-TTL `ttl` (`'5m'` | `'1h'`). Returns
 * `undefined` when the block is not a default cache point.
 */
function getDefaultCachePoint(
  block: unknown
): NormalizedBedrockCachePoint | undefined {
  if (typeof block !== 'object' || block === null) {
    return undefined;
  }
  if (!('cachePoint' in block)) {
    return undefined;
  }
  const cachePoint = (block as { cachePoint?: unknown }).cachePoint;
  if (typeof cachePoint !== 'object' || cachePoint === null) {
    return undefined;
  }
  if (!('type' in cachePoint)) {
    return undefined;
  }
  if ((cachePoint as { type?: string }).type !== 'default') {
    return undefined;
  }
  const ttl = (cachePoint as { ttl?: unknown }).ttl;
  return ttl === '5m' || ttl === '1h'
    ? { type: 'default', ttl }
    : { type: 'default' };
}

/**
 * Convert a LangChain content block to a Bedrock Converse content block.
 */
function convertLangChainContentBlockToConverseContentBlock({
  block,
  onUnknown = 'throw',
}: {
  block: string | MessageContentComplex;
  onUnknown?: 'throw' | 'passthrough';
}): BedrockContentBlock {
  if (typeof block === 'string') {
    return { text: block };
  }

  if (isDataContentBlock(block)) {
    return convertToProviderContentBlock(block, standardContentBlockConverter);
  }

  if (block.type === 'text') {
    return { text: (block as { text: string }).text };
  }

  if (block.type === 'image_url') {
    const imageUrl =
      typeof (block as { image_url: string | { url: string } }).image_url ===
      'string'
        ? (block as { image_url: string }).image_url
        : (block as { image_url: { url: string } }).image_url.url;
    return extractImageInfo(imageUrl);
  }

  if (block.type === 'image') {
    // Handle standard image block format
    const imageBlock = block as {
      source_type?: string;
      url?: string;
      data?: string;
      mime_type?: string;
    };
    if (
      imageBlock.source_type === 'url' &&
      imageBlock.url != null &&
      imageBlock.url !== ''
    ) {
      const parsedData = parseBase64DataUrl({
        dataUrl: imageBlock.url,
        asTypedArray: true,
      });
      if (parsedData != null) {
        const parsedMimeType = parseMimeType(parsedData.mime_type);
        return {
          image: {
            format: parsedMimeType.subtype as 'gif' | 'jpeg' | 'png' | 'webp',
            source: {
              bytes: parsedData.data as Uint8Array,
            },
          },
        };
      }
    } else if (
      imageBlock.source_type === 'base64' &&
      imageBlock.data != null &&
      imageBlock.data !== ''
    ) {
      let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
      if (imageBlock.mime_type != null && imageBlock.mime_type !== '') {
        const parsedMimeType = parseMimeType(imageBlock.mime_type);
        format = parsedMimeType.subtype as typeof format;
      }
      return {
        image: {
          format,
          source: {
            bytes: Uint8Array.from(atob(imageBlock.data), (c) =>
              c.charCodeAt(0)
            ),
          },
        },
      };
    }
    // If it already has the Bedrock image structure, pass through
    if ((block as { image?: unknown }).image !== undefined) {
      return {
        image: (block as { image: unknown }).image,
      } as BedrockContentBlock;
    }
  }

  if (
    block.type === 'video' &&
    (block as { video?: unknown }).video !== undefined
  ) {
    return {
      video: (block as { video: unknown }).video,
    } as BedrockContentBlock;
  }

  if (block.type === 'video') {
    return convertMultimodalVideoBlock(block as MediaContentBlock);
  }

  if (
    block.type === 'audio' &&
    (block as { audio?: unknown }).audio !== undefined
  ) {
    return {
      audio: (block as { audio: unknown }).audio,
    } as BedrockContentBlock;
  }

  if (block.type === 'audio') {
    return convertMultimodalAudioBlock(block as MediaContentBlock);
  }

  if (
    block.type === 'document' &&
    (block as { document?: unknown }).document !== undefined
  ) {
    return {
      document: (block as { document: unknown }).document,
    } as BedrockContentBlock;
  }

  const cachePoint = getDefaultCachePoint(block);
  if (cachePoint != null) {
    return {
      cachePoint,
    } as BedrockContentBlock;
  }

  if (onUnknown === 'throw') {
    throw new Error(`Unsupported content block type: ${block.type}`);
  } else {
    return block as unknown as BedrockContentBlock;
  }
}

/**
 * Convert a system message to Bedrock system content blocks.
 */
function convertSystemMessageToConverseMessage(
  msg: BaseMessage
): BedrockSystemContentBlock[] {
  if (typeof msg.content === 'string') {
    return [{ text: msg.content }];
  } else if (Array.isArray(msg.content) && msg.content.length > 0) {
    const contentBlocks: BedrockSystemContentBlock[] = [];
    for (const block of msg.content) {
      if (
        typeof block === 'object' &&
        block.type === 'text' &&
        typeof (block as { text?: string }).text === 'string'
      ) {
        contentBlocks.push({
          text: (block as { text: string }).text,
        });
      } else {
        const cachePoint = getDefaultCachePoint(block);
        if (cachePoint == null) {
          break;
        }
        contentBlocks.push({
          cachePoint,
        } as BedrockSystemContentBlock);
      }
    }
    if (msg.content.length === contentBlocks.length) {
      return contentBlocks;
    }
  }
  throw new Error(
    'System message content must be either a string, or an array of text blocks, optionally including a cache point.'
  );
}

/**
 * Convert an AI message to a Bedrock message.
 */
/**
 * Bedrock Converse requires `toolUse.input` to be a JSON object document.
 * History can carry non-object values: streaming leaves the raw partial-JSON
 * string on Anthropic-shaped inline blocks, and context-pressure truncation
 * (pre-3.x `createBoundedTruncationValue`) could persist `null` onto BOTH a
 * block's inline input and its `tool_calls` args. A string is parsed when it
 * forms a complete JSON object; every other shape degrades to `{}` — the call
 * already executed, so the replayed input is informational. Twin of
 * `coerceAnthropicToolUseInput` in the Anthropic fork; duplicated so each
 * fork stays self-contained against its upstream.
 */
function coerceBedrockToolUseInput(input: unknown): Record<string, unknown> {
  let candidate: unknown = input;
  if (typeof candidate === 'string') {
    try {
      candidate = JSON.parse(candidate);
    } catch {
      return {};
    }
  }
  return typeof candidate === 'object' &&
    candidate !== null &&
    !Array.isArray(candidate)
    ? (candidate as Record<string, unknown>)
    : {};
}

function convertAIMessageToConverseMessage(
  msg: BaseMessage,
  replayReasoning = true
): BedrockMessage {
  // Check for v1 format from other providers (PR #9766 fix)
  const responseMetadata = msg.response_metadata as
    | { output_version?: string }
    | undefined;
  if (responseMetadata?.output_version === 'v1') {
    return convertFromV1ToChatBedrockConverseMessage(msg, replayReasoning);
  }

  const assistantMsg: BedrockMessage = {
    role: 'assistant',
    content: [],
  };

  if (typeof msg.content === 'string' && msg.content !== '') {
    assistantMsg.content?.push({ text: msg.content });
  } else if (Array.isArray(msg.content)) {
    const parsedToolCallIds = new Set(
      isAIMessage(msg)
        ? (msg.tool_calls ?? []).flatMap((toolCall) =>
          toolCall.id != null ? [toolCall.id] : []
        )
        : []
    );
    const concatenatedBlocks = concatenateLangchainReasoningBlocks(
      msg.content as Array<MessageContentComplex | MessageContentReasoningBlock>
    );
    const contentBlocks: BedrockContentBlock[] = [];

    concatenatedBlocks.forEach((block) => {
      if (block.type === 'text') {
        const text = (block as { text?: string }).text ?? '';
        appendSerializableBedrockTextBlock(contentBlocks, text);
      } else if (block.type === 'tool_use') {
        const toolUse = block as {
          id?: unknown;
          name?: unknown;
          input?: unknown;
        };
        if (
          typeof toolUse.id === 'string' &&
          parsedToolCallIds.has(toolUse.id)
        ) {
          return;
        }
        if (
          typeof toolUse.id !== 'string' ||
          typeof toolUse.name !== 'string'
        ) {
          throw new Error('Invalid Anthropic tool_use content block');
        }
        contentBlocks.push({
          toolUse: {
            toolUseId: toolUse.id,
            name: toolUse.name,
            input: coerceBedrockToolUseInput(toolUse.input),
          },
        } as BedrockContentBlock);
      } else if (block.type === 'reasoning_content') {
        const reasoningBlock = block as MessageContentReasoningBlock;
        // Bedrock Converse rejects reasoningContent whose reasoningText.text is
        // null/empty (a signature-only block that never merged with its text),
        // and models other than Claude reject reasoningContent outright. Drop
        // it rather than emit an invalid request; the empty-turn placeholder
        // below covers a turn left with no content.
        if (
          !replayReasoning ||
          !isSerializableBedrockReasoningBlock(reasoningBlock)
        ) {
          return;
        }
        contentBlocks.push({
          reasoningContent:
            langchainReasoningBlockToBedrockReasoningBlock(reasoningBlock),
        } as BedrockContentBlock);
      } else {
        const cachePoint = getDefaultCachePoint(block);
        if (cachePoint != null) {
          contentBlocks.push({
            cachePoint,
          } as BedrockContentBlock);
        } else if (isReasoningContentBlock(block)) {
          // Reasoning from another provider (Anthropic `thinking`/
          // `redacted_thinking`, Google `reasoning`, LibreChat `think`).
          // Bedrock's native reasoning is `reasoning_content` (handled above); a
          // foreign block carries a signature Bedrock cannot validate, so drop
          // it on a cross-provider handoff (e.g. Anthropic → Bedrock) rather
          // than crash. The Bedrock model produces its own reasoning. Anything
          // else unknown still throws below — real content must be surfaced.
          return;
        } else if (FOREIGN_SERVER_TOOL_TYPES.some((t) => t === block.type)) {
          // Google server-side tool call/response (e.g. URL context) — only
          // Google can execute it or validate its thought signature; drop it
          // on a cross-provider handoff rather than crash.
          return;
        } else {
          const blockValues = Object.fromEntries(
            Object.entries(block).filter(([key]) => key !== 'type')
          );
          throw new Error(
            `Unsupported content block type: ${block.type} with content of ${JSON.stringify(blockValues, null, 2)}`
          );
        }
      }
    });

    assistantMsg.content = [...(assistantMsg.content ?? []), ...contentBlocks];
  }

  // Important: this must be placed after any reasoning content blocks
  if (isAIMessage(msg) && msg.tool_calls != null && msg.tool_calls.length > 0) {
    const existingToolUseIds = new Set(
      (assistantMsg.content ?? [])
        .filter((content) => 'toolUse' in content)
        .map((content) => content.toolUse?.toolUseId)
    );
    const toolUseBlocks = msg.tool_calls
      .filter((toolCall) => !existingToolUseIds.has(toolCall.id))
      .map((toolCall) => ({
        toolUse: {
          toolUseId: toolCall.id,
          name: toolCall.name,
          input: coerceBedrockToolUseInput(toolCall.args),
        },
      }));
    assistantMsg.content = [
      ...(assistantMsg.content ?? []),
      ...toolUseBlocks,
    ] as BedrockContentBlock[];
  }

  // Bedrock rejects an assistant message with no content blocks; if filtering
  // (e.g. dropping foreign reasoning) left it empty, emit a placeholder.
  if (assistantMsg.content == null || assistantMsg.content.length === 0) {
    assistantMsg.content = [{ text: BEDROCK_EMPTY_TEXT_PLACEHOLDER }];
  }

  return assistantMsg;
}

/**
 * Convert a v1 format message from other providers to Bedrock format.
 * This handles messages with standard content blocks like tool_call and reasoning.
 * (Implements PR #9766 fix for output_version v1 detection)
 */
function convertFromV1ToChatBedrockConverseMessage(
  msg: BaseMessage,
  replayReasoning = true
): BedrockMessage {
  const contentBlocks: BedrockContentBlock[] = [];
  const assistantMsg: BedrockMessage = {
    role: 'assistant',
    content: contentBlocks,
  };
  let droppedUnserializableContent = false;
  let unconvertedContentType: string | undefined;

  if (Array.isArray(msg.content)) {
    const concatenatedBlocks = concatenateLangchainReasoningBlocks(
      msg.content as Array<MessageContentComplex | MessageContentReasoningBlock>
    );
    for (const block of concatenatedBlocks) {
      if (typeof block === 'string') {
        if (!appendSerializableBedrockTextBlock(contentBlocks, block)) {
          droppedUnserializableContent = true;
        }
      } else if (block.type === 'text') {
        const text = (block as { text?: string }).text ?? '';
        if (!appendSerializableBedrockTextBlock(contentBlocks, text)) {
          droppedUnserializableContent = true;
        }
      } else if (block.type === 'tool_call') {
        const toolCall = block as {
          id: string;
          name: string;
          args: Record<string, unknown>;
        };
        contentBlocks.push({
          toolUse: {
            toolUseId: toolCall.id,
            name: toolCall.name,
            input: coerceBedrockToolUseInput(toolCall.args),
          },
        } as BedrockContentBlock);
      } else if (block.type === 'reasoning') {
        const reasoning = block as { reasoning?: string };
        /** Bedrock Converse rejects `reasoningText` with a null/empty `text`
         * (`Member must not be null`), e.g. when the producing model omitted
         * reasoning text (`thinking.display: "omitted"`) — drop rather than send. */
        if (
          !replayReasoning ||
          reasoning.reasoning == null ||
          reasoning.reasoning === ''
        ) {
          droppedUnserializableContent = true;
          continue;
        }
        contentBlocks.push({
          reasoningContent: {
            reasoningText: { text: reasoning.reasoning },
          },
        } as BedrockContentBlock);
      } else if (block.type === 'reasoning_content') {
        const reasoningBlock = block as MessageContentReasoningBlock;
        if (
          !replayReasoning ||
          !isSerializableBedrockReasoningBlock(reasoningBlock)
        ) {
          droppedUnserializableContent = true;
          continue;
        }
        contentBlocks.push({
          reasoningContent:
            langchainReasoningBlockToBedrockReasoningBlock(reasoningBlock),
        } as BedrockContentBlock);
      } else {
        unconvertedContentType ??= block.type;
      }
    }
  } else if (typeof msg.content === 'string') {
    if (!appendSerializableBedrockTextBlock(contentBlocks, msg.content)) {
      droppedUnserializableContent = true;
    }
  }

  // Also handle tool_calls from the message
  if (isAIMessage(msg) && msg.tool_calls != null && msg.tool_calls.length > 0) {
    // Check if tool calls are already in content
    const existingToolUseIds = new Set(
      contentBlocks
        .filter((c) => 'toolUse' in c)
        .map((c) => (c as { toolUse: { toolUseId: string } }).toolUse.toolUseId)
    );

    for (const tc of msg.tool_calls) {
      if (!existingToolUseIds.has(tc.id ?? '')) {
        contentBlocks.push({
          toolUse: {
            toolUseId: tc.id,
            name: tc.name,
            input: coerceBedrockToolUseInput(tc.args),
          },
        } as BedrockContentBlock);
      }
    }
  }

  const hasNoContent = contentBlocks.length === 0;
  if (hasNoContent && unconvertedContentType != null) {
    throw new Error(
      `Unsupported v1 content block type: ${unconvertedContentType}`
    );
  }
  if (hasNoContent && droppedUnserializableContent) {
    contentBlocks.push({ text: BEDROCK_EMPTY_TEXT_PLACEHOLDER });
  }

  return assistantMsg;
}

/**
 * Convert a human message to a Bedrock message.
 */
function convertHumanMessageToConverseMessage(
  msg: BaseMessage
): BedrockMessage {
  const contentBlocks: BedrockContentBlock[] = [];
  let hasText = false;
  let hasDocument = false;
  // Whitespace with nothing before it to fold into is held until the next text fragment,
  // so `[' ', 'hello']` and `[image, ' ', 'hello']` keep their leading space instead of
  // losing it; whitespace left over at the end has nothing to attach to and is dropped.
  let pendingWhitespace = '';
  const appendText = (text: string): void => {
    const candidate = pendingWhitespace + text;
    if (appendSerializableBedrockTextBlock(contentBlocks, candidate)) {
      hasText = true;
      pendingWhitespace = '';
      return;
    }
    pendingWhitespace = candidate;
  };

  if (typeof msg.content === 'string') {
    appendText(msg.content);
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (
        block.type === 'text' &&
        typeof (block as { text?: unknown }).text === 'string'
      ) {
        appendText((block as { text: string }).text);
        continue;
      }
      const converted = convertLangChainContentBlockToConverseContentBlock({
        block,
      });
      if ('document' in converted) {
        hasDocument = true;
      }
      contentBlocks.push(converted);
    }
  }

  const message: BedrockMessage = { role: 'user', content: contentBlocks };
  userGroupShape.set(message, { hasText, hasDocument });
  return message;
}

/**
 * Bedrock rejects an empty user message just as it rejects a blank text block
 * (`The text field in the ContentBlock object at messages.N.content.0 is blank`), and the
 * rejection fences the whole conversation on every retry. A promptless send — attachment, no
 * typed text — serializes to exactly that. Applied only once a user group is complete: an
 * empty human turn adjacent to a tool result merges into it below, and a per-message
 * placeholder would inject a synthetic `_` into an otherwise valid message.
 */
function finalizeUserMessage(message: BedrockMessage): void {
  if (message.role !== 'user') {
    return;
  }
  if (message.content == null || message.content.length === 0) {
    message.content = [{ text: BEDROCK_EMPTY_TEXT_PLACEHOLDER }];
    return;
  }
  // Converse also requires a text block in any message that carries a document block. A
  // promptless document upload arrives as blank text plus the document; the blank is
  // dropped during conversion, so supply the placeholder if no text survived.
  const shape = userGroupShape.get(message);
  if (shape != null && shape.hasDocument && !shape.hasText) {
    message.content.push({ text: BEDROCK_EMPTY_TEXT_PLACEHOLDER });
  }
}

/**
 * Convert a tool message to a Bedrock message.
 */
function convertToolMessageToConverseMessage(msg: BaseMessage): BedrockMessage {
  const toolCallId = (msg as { tool_call_id?: string }).tool_call_id;
  let isError = (msg as ToolMessage).status === 'error';

  let content: BedrockContentBlock[];
  if (typeof msg.content === 'string') {
    content = [{ text: msg.content }];
  } else if (Array.isArray(msg.content)) {
    content = msg.content.flatMap((block) => {
      if (typeof block === 'object' && block.type === 'tool_result') {
        if ((block as { is_error?: unknown }).is_error === true) {
          isError = true;
        }
        const toolResultContent = (block as { content?: unknown }).content;
        if (typeof toolResultContent === 'string') {
          return [{ text: toolResultContent }];
        }
        if (Array.isArray(toolResultContent)) {
          return toolResultContent.map((nestedBlock) =>
            convertLangChainContentBlockToConverseContentBlock({
              block: nestedBlock,
              onUnknown: 'passthrough',
            })
          );
        }
        return [
          {
            text: serializeStructuredValueBounded(
              toolResultContent,
              HARD_MAX_TOOL_RESULT_CHARS
            ).content,
          },
        ];
      }
      return [
        convertLangChainContentBlockToConverseContentBlock({
          block,
          onUnknown: 'passthrough',
        }),
      ];
    });
  } else {
    content = [{ text: String(msg.content) }];
  }

  // A `cachePoint` is a message-level ContentBlock — it is NOT a valid
  // ToolResultContentBlock. A tail prompt-cache breakpoint that anchors on a
  // tool result therefore ends up nested inside `toolResult.content`, which
  // Bedrock silently ignores (no cache write, no cache read). Hoist any
  // cachePoint(s) out of the tool result body so they sit as siblings after
  // it, which is the only position Bedrock honors.
  const toolResultContent: BedrockContentBlock[] = [];
  const trailingCachePoints: BedrockContentBlock[] = [];
  for (const block of content) {
    const cachePoint = getDefaultCachePoint(block);
    if (cachePoint != null) {
      trailingCachePoints.push({
        cachePoint,
      } as BedrockContentBlock);
    } else {
      toolResultContent.push(block);
    }
  }

  return {
    role: 'user',
    content: [
      {
        toolResult: {
          toolUseId: toolCallId,
          content: toolResultContent as { text: string }[],
          ...(isError ? { status: 'error' as const } : {}),
        },
      },
      ...trailingCachePoints,
    ],
  };
}

/** Options for {@link convertToConverseMessages}. */
export interface ConvertToConverseMessagesOptions {
  /** The model the request is for, which decides whether prior reasoning is replayed. */
  model?: string;
}

/**
 * Bedrock replays prior reasoning (`reasoningContent`) only to Anthropic Claude. Other Bedrock
 * models reject an assistant turn that carries it (OpenAI GPT: "This model doesn't support the
 * reasoningContent.reasoningText.text field for assistant messages"), so a handoff or model
 * switch from Claude to another Bedrock model fails on its first request. A model that can't be
 * identified (none given, or an ARN) keeps the reasoning, as before.
 */
export function replaysBedrockReasoning(model?: string): boolean {
  if (model == null || model === '' || model.startsWith('arn:')) {
    return true;
  }
  const id = model.toLowerCase();
  return id.includes('anthropic') || id.includes('claude');
}

/**
 * Convert LangChain messages to Bedrock Converse messages.
 */
export function convertToConverseMessages(
  messages: BaseMessage[],
  options: ConvertToConverseMessagesOptions = {}
): {
  converseMessages: BedrockMessage[];
  converseSystem: BedrockSystemContentBlock[];
} {
  const replayReasoning = replaysBedrockReasoning(options.model);
  const converseSystem = messages
    .filter((msg) => msg._getType() === 'system')
    .flatMap((msg) => convertSystemMessageToConverseMessage(msg));

  const converseMessages = messages
    .filter((msg) => msg._getType() !== 'system')
    .map((msg) => {
      if (msg._getType() === 'ai') {
        return convertAIMessageToConverseMessage(msg, replayReasoning);
      } else if (msg._getType() === 'human' || msg._getType() === 'generic') {
        return convertHumanMessageToConverseMessage(msg);
      } else if (msg._getType() === 'tool') {
        return convertToolMessageToConverseMessage(msg);
      } else {
        throw new Error(`Unsupported message type: ${msg._getType()}`);
      }
    });

  /**
   * Combine ALL consecutive user messages into one, not just tool-result
   * pairs. The Converse API documents strict role alternation; enforcement
   * varies by model family (Claude on Converse currently tolerates adjacent
   * user messages — verified live, 2026-07-28 — but the docs promise nothing
   * for the others), so the converter never emits the shape. The case that
   * matters: a `PostToolBatch`/`PreemptBoundary` hook injection lands a text
   * turn directly after tool results, and `ToolMessage` and `HumanMessage`
   * both convert to `role: 'user'` above. Block order is preserved
   * (toolResult blocks, then text) — verified live that Converse accepts the
   * mixed user message and answers both halves — and the toolUse/toolResult
   * pairing stays intact.
   */
  const combinedConverseMessages = converseMessages.reduce<BedrockMessage[]>(
    (acc, curr) => {
      if (acc.length === 0) {
        acc.push(curr);
        return acc;
      }
      const lastMessage = acc[acc.length - 1];
      if (lastMessage.role === 'user' && curr.role === 'user') {
        lastMessage.content = lastMessage.content?.concat(curr.content ?? []);
        const lastShape = userGroupShape.get(lastMessage);
        const nextShape = userGroupShape.get(curr);
        if (lastShape != null || nextShape != null) {
          userGroupShape.set(lastMessage, {
            hasText:
              (lastShape?.hasText ?? false) || (nextShape?.hasText ?? false),
            hasDocument:
              (lastShape?.hasDocument ?? false) ||
              (nextShape?.hasDocument ?? false),
          });
        }
      } else {
        finalizeUserMessage(lastMessage);
        acc.push(curr);
      }
      return acc;
    },
    []
  );
  if (combinedConverseMessages.length > 0) {
    finalizeUserMessage(
      combinedConverseMessages[combinedConverseMessages.length - 1]
    );
  }

  return { converseMessages: combinedConverseMessages, converseSystem };
}

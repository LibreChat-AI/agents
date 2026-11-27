import { Buffer } from 'node:buffer';
import { isIDContentBlock } from '@langchain/core/messages';
import type {
  BaseMessage,
  Data,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { ToolHistoryPreparation } from '@/messages/toolHistoryProjection';
import type * as t from '@/types';
import {
  projectCacheControlledToolOutputsToText,
  projectComputerCallOutputsToText,
  projectOpenAIChatToolMessageContent,
  projectOpenAIResponsesToolMessageContent,
  projectOpenRouterToolMessageContent,
  projectSingleTextToolOutputsToText,
  projectStructuredToolOutputsToText,
  projectToolStreamContentForProvider,
} from '@/messages/core';
import {
  stripAnthropicCacheControl,
  stripBedrockCacheControl,
  isTailCacheableBlock,
  withCacheControl,
  getCacheControl,
  cloneMessage,
} from '@/messages/cache';
import {
  coalesceAdjacentUserTurns,
  appendPredecessorHandoffCue,
  appendInstructionlessHandoffCue,
  removePredecessorHandoffCue,
} from '@/messages';
import { createToolHistoryPreparation } from '@/messages/toolHistoryProjection';
import { flattenLegacyContent, isLegacyConvertible } from '@/messages/content';
import { isImageContentBlock, toOmittedImageBlock } from '@/messages/vision';
import { isAnthropicLike, isGoogleLike, isOpenAILike } from '@/utils/llm';
import { annotateMessagesForLLM } from '@/tools/toolOutputReferences';
import { providerRequiresStrictAlternation } from '@/llm/providers';
import { hasComputerCallOutputMarker } from '@/utils/toolContent';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

const preparedProviderRequestBrand = Symbol('PreparedProviderRequest');
const OMITTED_ATTACHMENT_TEXT =
  '[Attachment omitted because its binary format is unsupported by this provider.]';

const BEDROCK_DOCUMENT_MIME_TYPES: Readonly<Partial<Record<string, string>>> = {
  csv: 'text/csv',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  html: 'text/html',
  md: 'text/markdown',
  pdf: 'application/pdf',
  txt: 'text/plain',
  xls: 'application/vnd.ms-excel',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

interface SerializedBuffer {
  type: 'Buffer';
  data: number[];
}

interface BedrockDocumentBlock {
  type: 'document';
  document: {
    name: string;
    format: string;
    source: {
      bytes: Uint8Array | SerializedBuffer;
    };
  };
}

type StandardBase64FileBlock = Data.StandardFileBlock & {
  source_type: 'base64';
  mime_type: string;
  data: string;
};

type ProviderContentBlock =
  | MessageContentComplex
  | BedrockDocumentBlock
  | StandardBase64FileBlock;

export type ProviderMessageProjectionMode =
  | 'chat-messages'
  | 'openai-responses';

export interface ProviderPayloadMeasurement {
  readonly fits: boolean;
  readonly projectedMessageTokens?: number;
  readonly availableMessageTokens?: number;
  readonly contextBudget?: number;
  readonly effectiveInstructionTokens?: number;
  readonly toolMessageTokens?: number;
  readonly toolMessageTokenCounts?: Record<string, number>;
  readonly toolMessageUsageError?: Error;
}

export interface PreparedProviderRequest {
  readonly model: t.ChatModel;
  readonly modelId?: string;
  readonly provider: t.ProviderName;
  readonly projectionMode: ProviderMessageProjectionMode;
  readonly messages: BaseMessage[];
  readonly measurement?: ProviderPayloadMeasurement;
  readonly [preparedProviderRequestBrand]: true;
}

type PreparedProviderRequestData = Omit<
  PreparedProviderRequest,
  typeof preparedProviderRequestBrand
>;

export interface ProviderRequestContext {
  getOrCreateToolOutputRegistry?(): ToolOutputReferenceRegistry | undefined;
  isRunProducedMessage?(message: BaseMessage): boolean;
  resolveAttachment?: t.AttachmentResolver;
}

type ResolveReference = (
  reference: t.AttachmentReference
) => MessageContentComplex | null | undefined;

export interface PrepareProviderRequestParams {
  model: t.ChatModel;
  messages: BaseMessage[];
  provider: t.ProviderName;
  context?: ProviderRequestContext;
  config?: RunnableConfig;
  maxToolResultChars?: number;
  measure?: (messages: BaseMessage[]) => ProviderPayloadMeasurement;
  toolHistory?: ToolHistoryPreparation;
  /** `false` replaces images with a text marker for a model without image input. */
  vision?: boolean;
  /** Flattens messages the image replacement leaves text-only, as `formatContentStrings` does. */
  useLegacyContent?: boolean;
}

export function usesNativeOpenAIResponses(
  model: t.ChatModel,
  provider: t.ProviderName,
  callOptions?: unknown
): boolean {
  if (!isOpenAILike(provider)) {
    return false;
  }
  let candidate: unknown = model;
  let effectiveCallOptions = callOptions;
  const seen = new Set<object>();
  for (let depth = 0; depth < 20; depth++) {
    if (candidate == null || typeof candidate !== 'object') {
      return false;
    }
    if (seen.has(candidate)) {
      return false;
    }
    seen.add(candidate);
    const runnable = candidate as {
      _useResponsesApi?: (options?: unknown) => boolean;
      bound?: unknown;
      defaultOptions?: unknown;
      last?: unknown;
      constructor?: { name?: unknown };
    };
    try {
      if (
        runnable.defaultOptions != null &&
        typeof runnable.defaultOptions === 'object' &&
        !Array.isArray(runnable.defaultOptions) &&
        effectiveCallOptions != null &&
        typeof effectiveCallOptions === 'object' &&
        !Array.isArray(effectiveCallOptions)
      ) {
        effectiveCallOptions = {
          ...(runnable.defaultOptions as Record<string, unknown>),
          ...(effectiveCallOptions as Record<string, unknown>),
        };
      } else if (effectiveCallOptions == null) {
        effectiveCallOptions = runnable.defaultOptions;
      }
      if (typeof runnable._useResponsesApi === 'function') {
        return runnable._useResponsesApi(effectiveCallOptions) === true;
      }
    } catch {
      // Continue through RunnableSequence/RunnableBinding wrappers.
    }
    if (
      typeof runnable.constructor?.name === 'string' &&
      runnable.constructor.name.includes('Responses')
    ) {
      return true;
    }
    if (runnable.last != null && typeof runnable.last === 'object') {
      candidate = runnable.last;
      continue;
    }
    if (runnable.bound != null && typeof runnable.bound === 'object') {
      candidate = runnable.bound;
      continue;
    }
    return false;
  }
  return false;
}

function resolveProviderMessageProjectionMode(
  model: t.ChatModel,
  provider: t.ProviderName,
  callOptions?: unknown
): ProviderMessageProjectionMode {
  return usesNativeOpenAIResponses(model, provider, callOptions)
    ? 'openai-responses'
    : 'chat-messages';
}

/** How a request for a model without image input treats images. */
interface ImageOmission {
  /** Flattens a turn left with only text, as `formatContentStrings` does. */
  legacyContent: boolean;
}

interface ProjectMessagesForProviderParams {
  model: t.ChatModel;
  messages: BaseMessage[];
  provider: t.ProviderName;
  maxToolResultChars?: number;
  callOptions?: unknown;
  toolHistory?: ToolHistoryPreparation;
  resolve?: ResolveReference;
  vision?: boolean;
  useLegacyContent?: boolean;
}

function isSerializedBuffer(value: object): value is SerializedBuffer {
  return (
    'type' in value &&
    value.type === 'Buffer' &&
    'data' in value &&
    Array.isArray(value.data)
  );
}

function isBedrockDocumentBlock(
  block: ProviderContentBlock
): block is BedrockDocumentBlock {
  if (block.type !== 'document' || !('document' in block)) {
    return false;
  }
  const document = block.document;
  if (typeof document !== 'object' || document == null) {
    return false;
  }
  if (
    !('name' in document) ||
    typeof document.name !== 'string' ||
    !('format' in document) ||
    typeof document.format !== 'string' ||
    !('source' in document) ||
    typeof document.source !== 'object' ||
    document.source == null ||
    !('bytes' in document.source)
  ) {
    return false;
  }
  const bytes = document.source.bytes;
  return (
    bytes instanceof Uint8Array ||
    (typeof bytes === 'object' && bytes != null && isSerializedBuffer(bytes))
  );
}

function toStandardFileBlock(
  block: BedrockDocumentBlock
): StandardBase64FileBlock | undefined {
  const mimeType = BEDROCK_DOCUMENT_MIME_TYPES[block.document.format];
  if (mimeType == null) {
    return undefined;
  }
  const source = block.document.source.bytes;
  const bytes = source instanceof Uint8Array ? source : source.data;
  return {
    type: 'file',
    source_type: 'base64',
    mime_type: mimeType,
    data: Buffer.from(bytes).toString('base64'),
    metadata: { name: block.document.name },
  };
}

function isStandardFileBlock(
  block: ProviderContentBlock
): block is StandardBase64FileBlock {
  return (
    block.type === 'file' &&
    'source_type' in block &&
    block.source_type === 'base64' &&
    'mime_type' in block &&
    typeof block.mime_type === 'string' &&
    'data' in block &&
    typeof block.data === 'string'
  );
}

function shouldRetainStandardFile(
  provider: t.ProviderName,
  mimeType: string
): boolean {
  const normalizedMimeType = mimeType.split(';', 1)[0].trim().toLowerCase();
  if (
    provider === Providers.ANTHROPIC ||
    getProviderFamily(provider) === 'anthropic'
  ) {
    return (
      normalizedMimeType === 'application/pdf' ||
      normalizedMimeType === 'image/jpeg' ||
      normalizedMimeType === 'image/png' ||
      normalizedMimeType === 'image/gif' ||
      normalizedMimeType === 'image/webp'
    );
  }
  if (isOpenAILike(provider)) {
    return normalizedMimeType === 'application/pdf';
  }
  return true;
}

function canProjectBedrockDocument(
  provider: t.ProviderName,
  mimeType: string
): boolean {
  if (isGoogleLike(provider)) {
    return true;
  }
  if (
    provider === Providers.ANTHROPIC ||
    getProviderFamily(provider) === 'anthropic' ||
    isOpenAILike(provider)
  ) {
    return mimeType === 'application/pdf';
  }
  return false;
}

/** How one provider request takes audio and video. */
type TimedMediaInput = 'gemini' | 'openai' | 'none';

/**
 * Gemini and Vertex read audio and video as `media` parts. OpenAI-compatible
 * Chat Completions endpoints read `input_audio` and `video_url` parts, where
 * the endpoint and model accept them, which only the host can tell. Claude,
 * Bedrock's Converse converter and the Responses API read none of them.
 */
function getTimedMediaInput(
  provider: t.ProviderName,
  projectionMode: ProviderMessageProjectionMode
): TimedMediaInput {
  if (isGoogleLike(provider)) {
    return 'gemini';
  }
  const family = getProviderFamily(provider);
  if (
    projectionMode === 'openai-responses' ||
    provider === Providers.ANTHROPIC ||
    family === 'anthropic' ||
    provider === Providers.BEDROCK ||
    family === 'bedrock'
  ) {
    return 'none';
  }
  return 'openai';
}

/** The audio or video MIME type a block names, normalized; `undefined` for anything else. */
function getTimedMediaMimeType(mimeType: unknown): string | undefined {
  if (typeof mimeType !== 'string') {
    return undefined;
  }
  const normalized = mimeType.split(';', 1)[0].trim().toLowerCase();
  return normalized.startsWith('audio/') || normalized.startsWith('video/')
    ? normalized
    : undefined;
}

/** Gemini's MIME type for each `input_audio` format it plays; raw PCM names no sample rate. */
const GEMINI_AUDIO_MIME_TYPES: Readonly<Partial<Record<string, string>>> = {
  aac: 'audio/aac',
  aiff: 'audio/aiff',
  flac: 'audio/flac',
  m4a: 'audio/mp4',
  mp3: 'audio/mp3',
  ogg: 'audio/ogg',
  wav: 'audio/wav',
  webm: 'audio/webm',
};

/** The MIME type and payload of a base64 `data:` URL. */
function parseDataUrl(
  url: string
): { mimeType: string; data: string } | undefined {
  const comma = url.indexOf(',');
  if (!url.startsWith('data:') || comma < 0) {
    return undefined;
  }
  const [mimeType, ...parameters] = url.slice(5, comma).split(';');
  if (parameters.at(-1)?.trim().toLowerCase() !== 'base64') {
    return undefined;
  }
  const data = url.slice(comma + 1);
  return data === '' ? undefined : { mimeType, data };
}

/** An OpenAI-compatible audio or video part as Gemini inline media, when it carries its bytes. */
function toGeminiMedia(
  block: ProviderContentBlock
): ProviderContentBlock | null {
  if (block.type === 'input_audio') {
    const audio = (
      block as { input_audio?: { data?: unknown; format?: unknown } }
    ).input_audio;
    const mimeType =
      typeof audio?.format === 'string'
        ? GEMINI_AUDIO_MIME_TYPES[audio.format.toLowerCase()]
        : undefined;
    return mimeType != null &&
      typeof audio?.data === 'string' &&
      audio.data !== ''
      ? { type: 'media', mimeType, data: audio.data }
      : null;
  }
  const video = (block as { video_url?: string | { url?: unknown } | null })
    .video_url;
  const url = typeof video === 'string' ? video : video?.url;
  const parsed = typeof url === 'string' ? parseDataUrl(url) : undefined;
  const mimeType = getTimedMediaMimeType(parsed?.mimeType);
  return parsed != null && mimeType != null
    ? { type: 'media', mimeType, data: parsed.data }
    : null;
}

/**
 * The part a request that takes `input` gets for an audio or video part: the
 * part itself when its provider reads it, Gemini inline media rewritten from
 * an OpenAI-compatible part, or `null` to leave it out. `undefined` for any
 * other block.
 */
function projectTimedMedia(
  block: ProviderContentBlock,
  input: TimedMediaInput
): ProviderContentBlock | null | undefined {
  if (block.type === 'media') {
    const media = block as { mimeType?: unknown; mime_type?: unknown };
    if (getTimedMediaMimeType(media.mimeType ?? media.mime_type) == null) {
      return undefined;
    }
    return input === 'gemini' ? block : null;
  }
  if (block.type === 'input_audio' || block.type === 'video_url') {
    if (input === 'openai') {
      return block;
    }
    return input === 'gemini' ? toGeminiMedia(block) : null;
  }
  if (
    isStandardFileBlock(block) &&
    getTimedMediaMimeType(block.mime_type) != null
  ) {
    return input === 'gemini' ? block : null;
  }
  return undefined;
}

function isBlankTextBlock(block: ProviderContentBlock): boolean {
  return (
    block.type === 'text' &&
    'text' in block &&
    typeof block.text === 'string' &&
    block.text.trim() === ''
  );
}

function copyUsableContentPrefix(
  sourceContent: ProviderContentBlock[],
  end: number
): ProviderContentBlock[] {
  const content: ProviderContentBlock[] = [];
  for (let index = 0; index < end; index++) {
    const block = sourceContent[index];
    if (!isBlankTextBlock(block)) {
      content.push(block);
    }
  }
  return content;
}

/** Puts the breakpoint a dropped reference carried on the last block that can hold one. */
function anchorCacheControl(
  content: MessageContentComplex[],
  marker: NonNullable<ReturnType<typeof getCacheControl>>
): void {
  for (let index = content.length - 1; index >= 0; index--) {
    if (isTailCacheableBlock(content[index])) {
      content[index] = withCacheControl(content[index], marker);
      return;
    }
  }
}

/**
 * Resolves referenced attachments and reprojects persisted provider-native
 * ones, in one pass and without changing history. Audio and video reach each
 * provider in the part it reads, or not at all (`projectTimedMedia`); tool
 * results keep their own tool-output projection.
 */
function projectAttachmentsForProvider(
  messages: BaseMessage[],
  provider: t.ProviderName,
  projectionMode: ProviderMessageProjectionMode,
  resolve?: ResolveReference,
  imageOmission?: ImageOmission
): BaseMessage[] {
  const native =
    provider === Providers.BEDROCK || getProviderFamily(provider) === 'bedrock';
  const timedMediaInput = getTimedMediaInput(provider, projectionMode);
  let projected: BaseMessage[] | undefined;
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    if (!Array.isArray(message.content)) {
      continue;
    }
    const projectsTimedMedia = message.getType() !== 'tool';
    /** A native computer-call output keeps its screenshot: the Responses API takes no other output. */
    const omitsImages =
      imageOmission != null && !hasComputerCallOutputMarker(message);
    const sourceContent = message.content as ProviderContentBlock[];
    let content: ProviderContentBlock[] | undefined;
    let displaced: ReturnType<typeof getCacheControl>;
    let omittedImage = false;
    for (let blockIndex = 0; blockIndex < sourceContent.length; blockIndex++) {
      const block = sourceContent[blockIndex];
      if (omitsImages && isImageContentBlock(block as MessageContentComplex)) {
        content ??= copyUsableContentPrefix(sourceContent, blockIndex);
        content.push(toOmittedImageBlock(block as MessageContentComplex));
        omittedImage = true;
        continue;
      }
      const resolved =
        resolve != null && isIDContentBlock(block) ? resolve(block) : undefined;
      if (resolved !== undefined) {
        content ??= copyUsableContentPrefix(sourceContent, blockIndex);
        const marker = getCacheControl(block);
        if (resolved === null) {
          displaced = marker ?? displaced;
        } else if (omitsImages && isImageContentBlock(resolved)) {
          content.push(toOmittedImageBlock(block as MessageContentComplex));
          omittedImage = true;
        } else {
          content.push(
            marker == null ? resolved : withCacheControl(resolved, marker)
          );
        }
        continue;
      }
      const media = projectsTimedMedia
        ? projectTimedMedia(block, timedMediaInput)
        : undefined;
      if (media === block) {
        content?.push(block);
        continue;
      }
      if (media !== undefined) {
        content ??= copyUsableContentPrefix(sourceContent, blockIndex);
        const marker = getCacheControl(block);
        if (media === null) {
          displaced = marker ?? displaced;
        } else {
          content.push(
            marker == null ? media : withCacheControl(media, marker)
          );
        }
        continue;
      }
      if (native) {
        if (content != null && !isBlankTextBlock(block)) {
          content.push(block);
        }
        continue;
      }
      if (isStandardFileBlock(block)) {
        if (shouldRetainStandardFile(provider, block.mime_type)) {
          content?.push(block);
          continue;
        }
        content ??= copyUsableContentPrefix(sourceContent, blockIndex);
        continue;
      }
      if (!isBedrockDocumentBlock(block)) {
        if (content != null && !isBlankTextBlock(block)) {
          content.push(block);
        }
        continue;
      }
      content ??= copyUsableContentPrefix(sourceContent, blockIndex);
      const mimeType = BEDROCK_DOCUMENT_MIME_TYPES[block.document.format];
      if (mimeType == null || !canProjectBedrockDocument(provider, mimeType)) {
        continue;
      }
      const standardFile = toStandardFileBlock(block);
      if (standardFile == null) {
        continue;
      }
      content.push(standardFile);
    }
    if (content == null) {
      continue;
    }
    if (content.length === 0) {
      content.push({ type: 'text', text: OMITTED_ATTACHMENT_TEXT });
    }
    if (displaced != null) {
      anchorCacheControl(content as MessageContentComplex[], displaced);
    }
    const projectedContent = content as MessageContentComplex[];
    projected ??= [...messages];
    projected[messageIndex] = cloneMessage(
      message,
      omittedImage &&
        imageOmission?.legacyContent === true &&
        isLegacyConvertible(message, projectedContent)
        ? flattenLegacyContent(projectedContent)
        : projectedContent
    );
  }
  return projected ?? messages;
}

function projectMessagesForProviderMode(
  {
    messages,
    provider,
    maxToolResultChars,
    toolHistory,
    resolve,
    vision,
    useLegacyContent,
  }: ProjectMessagesForProviderParams,
  projectionMode: ProviderMessageProjectionMode
): BaseMessage[] {
  const providerFamily = getProviderFamily(provider);
  const nativeOpenAIResponses = projectionMode === 'openai-responses';
  /** Images go before tool results are serialized, so none lands in them as base64 text. */
  const providerInputMessages = projectAttachmentsForProvider(
    projectToolStreamContentForProvider(
      messages,
      nativeOpenAIResponses ? 'native' : 'fallback',
      maxToolResultChars,
      toolHistory
    ),
    provider,
    projectionMode,
    resolve,
    vision === false ? { legacyContent: useLegacyContent === true } : undefined
  );
  if (nativeOpenAIResponses) {
    return projectOpenAIResponsesToolMessageContent(
      stripAnthropicCacheControl(
        stripBedrockCacheControl(providerInputMessages)
      ),
      maxToolResultChars
    );
  }
  if (provider === Providers.OPENROUTER) {
    return projectComputerCallOutputsToText(
      projectOpenRouterToolMessageContent(
        stripBedrockCacheControl(providerInputMessages),
        maxToolResultChars
      )
    );
  }
  if (isOpenAILike(provider)) {
    return projectComputerCallOutputsToText(
      projectOpenAIChatToolMessageContent(
        stripAnthropicCacheControl(
          stripBedrockCacheControl(providerInputMessages)
        ),
        maxToolResultChars
      )
    );
  }
  if (provider === Providers.ANTHROPIC || providerFamily === 'anthropic') {
    return projectComputerCallOutputsToText(
      projectSingleTextToolOutputsToText(
        stripBedrockCacheControl(providerInputMessages),
        maxToolResultChars
      )
    );
  }
  if (provider === Providers.BEDROCK || providerFamily === 'bedrock') {
    return stripAnthropicCacheControl(
      projectComputerCallOutputsToText(
        projectCacheControlledToolOutputsToText(
          providerInputMessages,
          maxToolResultChars
        )
      )
    );
  }
  return projectComputerCallOutputsToText(
    projectStructuredToolOutputsToText(
      projectSingleTextToolOutputsToText(
        stripAnthropicCacheControl(
          stripBedrockCacheControl(providerInputMessages)
        ),
        maxToolResultChars
      ),
      maxToolResultChars
    )
  );
}

/** Produces the provider-facing representation before adapter serialization. */
export function projectMessagesForProvider(
  params: ProjectMessagesForProviderParams
): BaseMessage[] {
  return projectMessagesForProviderMode(
    params,
    resolveProviderMessageProjectionMode(
      params.model,
      params.provider,
      params.callOptions
    )
  );
}

/** Reads the serving model id through LangChain binding/sequence wrappers. */
export function resolveServingModelId(model: unknown): string | undefined {
  const seen = new Set<unknown>();
  let current: unknown = model;
  while (current != null && typeof current === 'object' && !seen.has(current)) {
    seen.add(current);
    const wrapper = current as {
      model?: unknown;
      bound?: unknown;
      last?: unknown;
      steps?: unknown[];
    };
    if (typeof wrapper.model === 'string' && wrapper.model !== '') {
      return wrapper.model;
    }
    current =
      wrapper.bound ??
      wrapper.last ??
      (Array.isArray(wrapper.steps)
        ? wrapper.steps[wrapper.steps.length - 1]
        : undefined);
  }
  return undefined;
}

/**
 * Finalizes one provider request and measures the exact message array that
 * will be passed to LangChain. Source messages remain untouched.
 */
export function prepareProviderRequest({
  model,
  messages,
  provider,
  context,
  config,
  maxToolResultChars,
  measure,
  toolHistory = createToolHistoryPreparation(),
  vision,
  useLegacyContent,
}: PrepareProviderRequestParams): PreparedProviderRequest {
  const projectionMode = resolveProviderMessageProjectionMode(
    model,
    provider,
    config
  );
  const modelId = resolveServingModelId(model);
  const resolveAttachment = context?.resolveAttachment;
  const projected = projectMessagesForProviderMode(
    {
      model,
      messages,
      provider,
      maxToolResultChars,
      callOptions: config,
      toolHistory,
      vision,
      useLegacyContent,
      resolve:
        resolveAttachment == null
          ? undefined
          : (reference): MessageContentComplex | null | undefined =>
            resolveAttachment(reference, {
              provider,
              modelId,
              projectionMode,
            }),
    },
    projectionMode
  );
  const registry = context?.getOrCreateToolOutputRegistry?.();
  const runId = config?.configurable?.run_id as string | undefined;
  const annotated = annotateMessagesForLLM(projected, registry, runId);
  const isRunProduced = context?.isRunProducedMessage;
  const handoffGrounded = appendInstructionlessHandoffCue(annotated, config);
  const cued = isAnthropicLike(provider, {
    model: modelId,
  })
    ? appendPredecessorHandoffCue(
      handoffGrounded,
      isRunProduced == null
        ? undefined
        : (message): boolean => isRunProduced.call(context, message)
    )
    : removePredecessorHandoffCue(handoffGrounded);
  const preparedMessages = providerRequiresStrictAlternation(provider)
    ? coalesceAdjacentUserTurns(cued)
    : cued;

  const request: PreparedProviderRequestData = {
    model,
    modelId,
    provider,
    projectionMode,
    messages: preparedMessages,
    measurement: measure?.(preparedMessages),
  };
  Object.defineProperty(request, preparedProviderRequestBrand, {
    value: true,
    enumerable: false,
  });
  return Object.freeze(request) as PreparedProviderRequest;
}

export function assertPreparedProviderRequestFor(
  request: PreparedProviderRequest,
  model: t.ChatModel,
  provider: t.ProviderName,
  config?: RunnableConfig
): void {
  if (
    !Object.prototype.hasOwnProperty.call(request, preparedProviderRequestBrand)
  ) {
    throw new Error('Invalid prepared provider request');
  }
  if (request.model !== model) {
    throw new Error('Prepared provider request does not match serving model');
  }
  if (request.provider !== provider) {
    throw new Error(
      'Prepared provider request does not match serving provider'
    );
  }
  if (
    request.projectionMode !==
    resolveProviderMessageProjectionMode(model, provider, config)
  ) {
    throw new Error(
      'Prepared provider request does not match invocation options'
    );
  }
}

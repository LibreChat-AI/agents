import { AzureOpenAI as AzureOpenAIClient } from 'openai';
import { ChatXAI as OriginalChatXAI } from '@langchain/xai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ToolDefinition } from '@langchain/core/language_models/base';
import { ChatDeepSeek as OriginalChatDeepSeek } from '@langchain/deepseek';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  AIMessage,
  AIMessageChunk,
  isAIMessage,
} from '@langchain/core/messages';
import {
  convertToOpenAITool,
  isLangChainTool,
} from '@langchain/core/utils/function_calling';
import {
  getEndpoint,
  OpenAIClient,
  wrapOpenAIClientError,
  getHeadersWithUserAgent,
  convertMessagesToResponsesInput,
  convertResponsesDeltaToChatGenerationChunk,
  ChatOpenAI as OriginalChatOpenAI,
  ChatOpenAIResponses as OriginalChatOpenAIResponses,
  ChatOpenAICompletions as OriginalChatOpenAICompletions,
  AzureChatOpenAI as OriginalAzureChatOpenAI,
  AzureChatOpenAIResponses as OriginalAzureChatOpenAIResponses,
  AzureChatOpenAICompletions as OriginalAzureChatOpenAICompletions,
} from '@langchain/openai';
import type {
  BaseMessage,
  BaseMessageChunk,
  UsageMetadata,
} from '@langchain/core/messages';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { ChatGeneration, ChatResult } from '@langchain/core/outputs';
import type { ChatXAIInput } from '@langchain/xai';
import type * as t from '@langchain/openai';
import type { SmoothItem, SmoothPiece } from '@/llm/stream/smoother';
import type { ResponsesReplayPosition } from '@/messages/core';
import type { SeenScalarMetadata } from './streamMetadata';
import type { HeaderValue, HeadersLike } from './types';
import type { PromptCacheTtl } from '@/messages/cache';
import {
  OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY,
  OPENAI_RESPONSES_REPLAY_POSITIONS_KEY,
  projectOpenAIResponsesToolMessageContent,
  projectToolStreamContentForProvider,
} from '@/messages/core';
import {
  buildAnthropicCacheControl,
  resolvePromptCacheTtl,
  stripAnthropicCacheControl,
  stripBedrockCacheControl,
} from '@/messages/cache';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  hasReasoningKwargs,
  hasToolCallChunks,
  getReasoningKwargsText,
} from '@/llm/stream/chunkAdapters';
import { isReasoningModel, _convertMessagesToOpenAIParams } from './utils';
import { smoothStream, resolveStreamDelay } from '@/llm/stream/smoother';
import { INTENT_ARG, isIntentLabelProperty } from '@/tools/intentArg';
import { dropRepeatedScalarMetadata } from './streamMetadata';

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
const iife = <T>(fn: () => T) => fn();

export function isHeaders(headers: unknown): headers is Headers {
  return (
    typeof Headers !== 'undefined' &&
    headers !== null &&
    typeof headers === 'object' &&
    Object.prototype.toString.call(headers) === '[object Headers]'
  );
}

export function normalizeHeaders(
  headers: HeadersLike
): Record<string, HeaderValue | readonly HeaderValue[]> {
  const output = iife(() => {
    // If headers is a Headers instance
    if (isHeaders(headers)) {
      return headers;
    }
    // If headers is an array of [key, value] pairs
    else if (Array.isArray(headers)) {
      return new Headers(headers);
    }
    // If headers is a NullableHeaders-like object (has 'values' property that is a Headers)
    else if (
      typeof headers === 'object' &&
      headers !== null &&
      'values' in headers &&
      isHeaders(headers.values)
    ) {
      return headers.values;
    }
    // If headers is a plain object
    else if (typeof headers === 'object' && headers !== null) {
      const entries: [string, string][] = Object.entries(headers)
        .filter(([, v]) => typeof v === 'string')
        .map(([k, v]) => [k, v as string]);
      return new Headers(entries);
    }
    return new Headers();
  });

  return Object.fromEntries(output.entries());
}

type OpenAICoreRequestOptions = OpenAIClient.RequestOptions;
type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;
type OpenAIClientConfig = NonNullable<
  ConstructorParameters<typeof OpenAIClient>[0]
>;
type LibreChatOpenAIFields = t.ChatOpenAIFields & {
  _lc_stream_delay?: number;
  firstPartyEndpoint?: boolean;
  includeReasoningContent?: boolean;
  includeReasoningDetails?: boolean;
  convertReasoningDetailsToContent?: boolean;
  preserveToolCacheControl?: boolean;
  responsesPromptCache?: boolean;
  responsesPromptCacheTtl?: PromptCacheTtl;
  promptCacheExplicit?: boolean;
  safety_identifier?: string;
};
type LibreChatAzureOpenAIFields = t.AzureOpenAIInput & {
  _lc_stream_delay?: number;
  firstPartyEndpoint?: boolean;
  promptCacheExplicit?: boolean;
  safety_identifier?: string;
};
type ReasoningCallOptions = {
  reasoning?: OpenAIClient.Reasoning;
  reasoningEffort?: OpenAIClient.Reasoning['effort'];
};
type OpenAIDeltaWithLibreChatFields = Record<string, unknown> & {
  reasoning?: unknown;
  reasoning_details?: unknown;
  provider_specific_fields?: unknown;
};
type OpenAIClientOwner = {
  client?: OpenAIClient;
  clientConfig: OpenAIClientConfig;
  timeout?: number;
};
type AbortableOpenAIClient = CustomOpenAIClient | CustomAzureOpenAIClient;
type OpenAIClientDelegate = {
  client?: AbortableOpenAIClient;
  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions;
};
type OpenAIChatCompletion = OpenAIClient.Chat.Completions.ChatCompletion;
type OpenAIChatCompletionChunk =
  OpenAIClient.Chat.Completions.ChatCompletionChunk;
type OpenAIChatCompletionStreamItem =
  | OpenAIChatCompletionChunk
  | {
      event: string;
      data?: unknown;
    };
type OpenAIChatCompletionRequest =
  | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
  | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming;
type OpenAIChatCompletionResult =
  | AsyncIterable<OpenAIChatCompletionChunk>
  | OpenAIChatCompletion;
type PromptTokensDetailsWithCacheWrite = NonNullable<
  OpenAIClient.Completions.CompletionUsage['prompt_tokens_details']
> & {
  cache_write_tokens?: number;
};
type OpenAIChatCompletionRetry = (
  request: OpenAIChatCompletionRequest,
  requestOptions?: OpenAICoreRequestOptions
) => Promise<
  AsyncIterable<OpenAIChatCompletionStreamItem> | OpenAIChatCompletion
>;
type OpenAIManagedRequestFields = {
  promptCacheExplicit?: boolean;
  safetyIdentifier?: string;
};
type OpenAIManagedRequestParams = {
  prompt_cache_options?: {
    mode: 'explicit';
    ttl: '30m';
  };
  safety_identifier?: string;
};

function stripResponsesToolCacheControl<T>(tools: T): T {
  if (!Array.isArray(tools)) {
    return tools;
  }

  return tools.map((tool) => {
    if (tool == null || typeof tool !== 'object') {
      return tool;
    }
    const clone = { ...(tool as Record<string, unknown>) };
    delete clone.cache_control;
    if (
      clone.extras != null &&
      typeof clone.extras === 'object' &&
      !Array.isArray(clone.extras)
    ) {
      clone.extras = { ...(clone.extras as Record<string, unknown>) };
      delete (clone.extras as Record<string, unknown>).cache_control;
    }
    return clone;
  }) as T;
}

function projectOpenAIResponsesProviderMessages(
  messages: BaseMessage[]
): BaseMessage[] {
  return projectOpenAIResponsesToolMessageContent(
    stripAnthropicCacheControl(
      stripBedrockCacheControl(projectToolStreamContentForProvider(messages))
    )
  );
}

type ResponsesRequest =
  | OpenAIClient.Responses.ResponseCreateParamsStreaming
  | OpenAIClient.Responses.ResponseCreateParamsNonStreaming;
type ResponsesResult =
  | AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>
  | OpenAIClient.Responses.Response;
type ResponsesStreamChunkOptions = {
  promptIndex?: number;
  signal?: AbortSignal;
};
type CacheableChatPart = {
  type: 'text' | 'image_url' | 'input_audio' | 'file' | 'refusal';
  prompt_cache_breakpoint?: { mode: 'explicit' };
  [key: string]: unknown;
};
type CacheableResponsePart = (
  | OpenAIClient.Responses.ResponseInputText
  | OpenAIClient.Responses.ResponseInputImage
  | OpenAIClient.Responses.ResponseInputFile
) & {
  prompt_cache_breakpoint?: { mode: 'explicit' };
};
// `Omit` before re-adding `input_tokens_details` as optional matters: the SDK's own
// `ResponseUsage` declares it required (true for OpenAI itself), so a plain intersection
// would keep it required in the merged type despite the `?:` here — masking, at the type
// level, that OpenAI-*compatible* servers (e.g. mlx_vlm.server) may omit it entirely.
// Mirrors `CompletionUsageWithCacheWrite`'s handling of the analogous Completions API field.
type ResponsesUsageWithCacheWrite = Omit<
  OpenAIClient.Responses.ResponseUsage,
  'input_tokens_details'
> & {
  input_tokens_details?: OpenAIClient.Responses.ResponseUsage['input_tokens_details'] & {
    cache_write_tokens?: number;
  };
};
const CACHE_WRITE_METADATA_KEY = '__librechat_cache_write_tokens';

function applyManagedRequestParams<T extends object>(
  params: T,
  fields: OpenAIManagedRequestFields
): T & OpenAIManagedRequestParams {
  return {
    ...params,
    ...(fields.promptCacheExplicit === true && {
      prompt_cache_options: {
        mode: 'explicit' as const,
        ttl: '30m' as const,
      },
    }),
    ...(fields.safetyIdentifier != null && {
      safety_identifier: fields.safetyIdentifier,
    }),
  };
}

function isCacheableChatPart(part: unknown): part is CacheableChatPart {
  if (typeof part !== 'object' || part == null || !('type' in part)) {
    return false;
  }
  return (
    part.type === 'text' ||
    part.type === 'image_url' ||
    part.type === 'input_audio' ||
    part.type === 'file' ||
    part.type === 'refusal'
  );
}

function canAddChatBreakpoint(
  message: OpenAIClient.Chat.Completions.ChatCompletionMessageParam
): boolean {
  if (
    message.role !== 'system' &&
    message.role !== 'developer' &&
    message.role !== 'user' &&
    message.role !== 'assistant' &&
    message.role !== 'tool'
  ) {
    return false;
  }
  if (typeof message.content === 'string') {
    return message.content.length > 0;
  }
  return (
    Array.isArray(message.content) &&
    message.content.some((part) => isCacheableChatPart(part))
  );
}

function addChatBreakpoint(
  message: OpenAIClient.Chat.Completions.ChatCompletionMessageParam
): OpenAIClient.Chat.Completions.ChatCompletionMessageParam {
  if (!canAddChatBreakpoint(message)) {
    return message;
  }

  if (typeof message.content === 'string') {
    return {
      ...message,
      content: [
        {
          type: 'text',
          text: message.content,
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    } as unknown as OpenAIClient.Chat.Completions.ChatCompletionMessageParam;
  }
  if (!Array.isArray(message.content) || message.content.length === 0) {
    return message;
  }

  const content = [...message.content] as unknown as CacheableChatPart[];
  let index = content.length - 1;
  while (index >= 0 && !isCacheableChatPart(content[index])) {
    index--;
  }
  if (index < 0) {
    return message;
  }
  content[index] = {
    ...content[index],
    prompt_cache_breakpoint: { mode: 'explicit' },
  };
  return {
    ...message,
    content,
  } as unknown as OpenAIClient.Chat.Completions.ChatCompletionMessageParam;
}

function selectCacheBreakpointIndexes(
  roles: Array<string | undefined>,
  cacheable: boolean[]
): number[] {
  let instructionIndex = -1;
  let latestUserIndex = -1;
  for (let index = 0; index < roles.length; index++) {
    const role = roles[index];
    if ((role === 'system' || role === 'developer') && cacheable[index]) {
      instructionIndex = index;
    }
    if (role === 'user') {
      latestUserIndex = index;
    }
  }

  const indexes = new Set<number>();
  if (instructionIndex >= 0) {
    indexes.add(instructionIndex);
  }
  for (let index = latestUserIndex - 1; index >= 0; index--) {
    if (cacheable[index]) {
      indexes.add(index);
      break;
    }
  }
  return [...indexes];
}

/** @internal */
export function addChatCacheBreakpoints(
  messages: OpenAIClient.Chat.Completions.ChatCompletionMessageParam[]
): OpenAIClient.Chat.Completions.ChatCompletionMessageParam[] {
  const indexes = new Set(
    selectCacheBreakpointIndexes(
      messages.map((message) => message.role),
      messages.map((message) => canAddChatBreakpoint(message))
    )
  );
  return messages.map((message, index) =>
    indexes.has(index) ? addChatBreakpoint(message) : message
  );
}

function isResponseMessage(
  item: OpenAIClient.Responses.ResponseInputItem
): item is OpenAIClient.Responses.ResponseInputItem.Message {
  return item.type === 'message';
}

function isResponseInputRole(role: string): boolean {
  return role === 'system' || role === 'developer' || role === 'user';
}

/** Only `input_text`/`input_image`/`input_file` accept a Responses breakpoint;
 *  `output_text`/`refusal` (replayed assistant blocks) are rejected with a 400. */
function isCacheableResponsePart(part: unknown): part is CacheableResponsePart {
  if (typeof part !== 'object' || part == null || !('type' in part)) {
    return false;
  }
  return (
    part.type === 'input_text' ||
    part.type === 'input_image' ||
    part.type === 'input_file'
  );
}

function addResponseBreakpoint(
  item: OpenAIClient.Responses.ResponseInputItem
): OpenAIClient.Responses.ResponseInputItem {
  if (!isResponseMessage(item)) {
    return item;
  }
  const rawContent = item.content as
    | string
    | OpenAIClient.Responses.ResponseInputMessageContentList;
  if (typeof rawContent === 'string') {
    if (rawContent.length === 0) {
      return item;
    }
    return {
      ...item,
      content: [
        {
          type: 'input_text',
          text: rawContent,
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    } as OpenAIClient.Responses.ResponseInputItem;
  }
  if (!Array.isArray(rawContent) || rawContent.length === 0) {
    return item;
  }

  const content = [...rawContent];
  let index = content.length - 1;
  while (index >= 0 && !isCacheableResponsePart(content[index])) {
    index--;
  }
  if (index < 0) {
    return item;
  }
  content[index] = {
    ...(content[index] as CacheableResponsePart),
    prompt_cache_breakpoint: { mode: 'explicit' },
  };
  return { ...item, content };
}

/** @internal */
export function addResponseCacheBreakpoints(
  input: OpenAIClient.Responses.ResponseCreateParams['input']
): OpenAIClient.Responses.ResponseCreateParams['input'] {
  if (!Array.isArray(input)) {
    return input;
  }
  const indexes = new Set(
    selectCacheBreakpointIndexes(
      input.map((item) => (isResponseMessage(item) ? item.role : undefined)),
      input.map((item) => {
        if (!isResponseMessage(item)) {
          return false;
        }
        /** Only input roles take a Responses breakpoint. Assistant/tool turns
         *  carry output content (string or output_text) that the API rejects
         *  under an input marker, so they're never eligible. */
        if (!isResponseInputRole(item.role)) {
          return false;
        }
        const content = item.content as
          | string
          | OpenAIClient.Responses.ResponseInputMessageContentList;
        return typeof content === 'string'
          ? content.length > 0
          : Array.isArray(content) &&
              content.some((part) => isCacheableResponsePart(part));
      })
    )
  );
  return input.map((item, index) =>
    indexes.has(index) ? addResponseBreakpoint(item) : item
  );
}

/**
 * GPT-6 Astra, matched on the model id rather than a `gpt-6` family cutoff.
 *
 * OpenAI documents these constraints for Astra specifically, and every gate
 * keyed off this helper *removes* capability — it forces the Responses API,
 * drops sampling parameters, and lowers reasoning effort. A false positive
 * therefore silently degrades a sibling model that never needed it, so the
 * match stays narrow until OpenAI documents the same rules more widely.
 *
 * Deliberately does NOT match a `provider/` prefixed id such as
 * `openai/gpt-6-astra`. A slash means a proxy is doing the routing, and the
 * proxy's contract is not OpenAI's: `ChatOpenRouter` extends this class, and on
 * OpenRouter `effort: 'none'` is a *supported* value meaning "disable
 * reasoning" (it maps to `include_reasoning: false`). Substituting it with
 * `low` there would silently turn reasoning off into reasoning on, and forcing
 * the Responses API would change the endpoint shape a proxy may not serve. Bare
 * ids reach the first-party OpenAI and Azure surfaces these rules describe.
 * @see https://developers.openai.com/api/docs/models/gpt-6-astra
 * @see https://developers.openai.com/api/docs/guides/latest-model
 */
const GPT_6_ASTRA_PATTERN = /^gpt-6-astra(?:-|$)/i;

/** @internal */
export function isGpt6AstraModel(model?: string): boolean {
  return model != null && GPT_6_ASTRA_PATTERN.test(model.toLowerCase());
}

/**
 * Reasoning efforts GPT-6 Astra does not accept, mapped to the level OpenAI
 * recommends migrating to. `none` is rejected outright and `minimal` is not
 * offered; the migration guide says to "start with `low` and compare results".
 * Substituting keeps a stored agent configuration usable instead of failing
 * the turn on a value the previous model accepted.
 */
const GPT_6_ASTRA_UNSUPPORTED_EFFORTS = new Set(['none', 'minimal']);
const GPT_6_ASTRA_EFFORT_FALLBACK = 'low' as const;

function substituteUnsupportedAstraEffort(
  astraRulesApply: boolean,
  reasoning: OpenAIClient.Reasoning | undefined
): OpenAIClient.Reasoning | undefined {
  const effort = reasoning?.effort;
  if (
    effort == null ||
    !astraRulesApply ||
    !GPT_6_ASTRA_UNSUPPORTED_EFFORTS.has(effort)
  ) {
    return reasoning;
  }
  return { ...reasoning, effort: GPT_6_ASTRA_EFFORT_FALLBACK };
}

/** Rejected inside the Responses `include` array. */
const GPT_6_ASTRA_UNSUPPORTED_INCLUDE: OpenAIClient.Responses.ResponseIncludable =
  'message.output_text.logprobs';

/**
 * The subset of a request GPT-6 Astra rejects, taken from the SDK's own request
 * types so the stripping boundary keeps their constraints. Every member is
 * optional so a key can be deleted from a request object whose concrete type
 * marks it required.
 */
type AstraStrippableParams = Partial<
  Pick<
    OpenAIClient.Chat.Completions.ChatCompletionCreateParams,
    'temperature' | 'top_p' | 'logprobs' | 'top_logprobs'
  >
> &
  Partial<Pick<OpenAIClient.Responses.ResponseCreateParams, 'include'>>;

/**
 * Removes the sampling and logprob parameters GPT-6 Astra rejects.
 *
 * LangChain's request builders emit `temperature` and `top_p` from instance
 * fields unconditionally, so leaving them unset upstream is not enough — they
 * are stripped after `super.invocationParams`. `logprobs` is Chat Completions
 * only; the Responses equivalent rides inside `include`.
 */
function stripUnsupportedAstraParams<T extends object>(
  astraRulesApply: boolean,
  params: T,
  endpoint: 'completions' | 'responses'
): T {
  if (!astraRulesApply) {
    return params;
  }
  const next = { ...params };
  const record = next as AstraStrippableParams;
  delete record.temperature;
  delete record.top_p;
  delete record.top_logprobs;
  if (endpoint === 'completions') {
    delete record.logprobs;
    return next;
  }
  const include = record.include;
  if (!Array.isArray(include)) {
    return next;
  }
  const filtered = include.filter(
    (entry) => entry !== GPT_6_ASTRA_UNSUPPORTED_INCLUDE
  );
  if (filtered.length === include.length) {
    return next;
  }
  if (filtered.length === 0) {
    delete record.include;
  } else {
    record.include = filtered;
  }
  return next;
}

/** @internal */
export function shouldIncludeEncryptedReasoning(
  model: string,
  params: {
    store?: boolean | null;
    reasoning?: unknown;
  },
  astraRulesApply = false
): boolean {
  const reasoningContext = (
    params.reasoning as
      | { context?: 'auto' | 'current_turn' | 'all_turns' }
      | undefined
  )?.context;
  return (
    (/^gpt-5\.6(?:-|$)/i.test(model) || astraRulesApply) &&
    (params.store === false || reasoningContext !== 'current_turn')
  );
}

export function getCacheWriteTokens(message: BaseMessage): number | undefined {
  const responseMetadata = message.response_metadata as {
    usage?: ResponsesUsageWithCacheWrite;
    metadata?: Record<string, string>;
  };
  const reported =
    responseMetadata.usage?.input_tokens_details?.cache_write_tokens;
  if (reported != null) {
    return reported;
  }
  const serialized = responseMetadata.metadata?.[CACHE_WRITE_METADATA_KEY];
  if (serialized == null) {
    return;
  }
  const parsed = Number(serialized);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function attachCacheWriteUsage(message: BaseMessage): void {
  const cacheWriteTokens = getCacheWriteTokens(message);
  if (
    cacheWriteTokens == null ||
    !isAIMessage(message) ||
    message.usage_metadata == null
  ) {
    return;
  }
  message.usage_metadata.input_token_details = {
    ...message.usage_metadata.input_token_details,
    cache_creation: cacheWriteTokens,
  };
  const responseMetadata = message.response_metadata as {
    metadata?: Record<string, string>;
  };
  if (responseMetadata.metadata?.[CACHE_WRITE_METADATA_KEY] == null) {
    return;
  }
  const metadata = { ...responseMetadata.metadata };
  delete metadata[CACHE_WRITE_METADATA_KEY];
  message.response_metadata = {
    ...message.response_metadata,
    metadata,
  };
}

export function attachCacheWriteMetadata(
  response: OpenAIClient.Responses.Response
): OpenAIClient.Responses.Response {
  const usage = response.usage as ResponsesUsageWithCacheWrite | undefined;
  const cacheWriteTokens = usage?.input_tokens_details?.cache_write_tokens;
  if (cacheWriteTokens == null) {
    return response;
  }
  return {
    ...response,
    metadata: {
      ...(response.metadata ?? {}),
      [CACHE_WRITE_METADATA_KEY]: String(cacheWriteTokens),
    },
  };
}

function isResponsesStream(
  result: ResponsesResult
): result is AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent> {
  return Symbol.asyncIterator in result;
}

const RESPONSES_REPLAY_OUTPUT_ITEM_TYPES = new Set([
  'local_shell_call_output',
  'shell_call_output',
  'apply_patch_call_output',
  'program_output',
]);

function isResponsesReplayOutputItem(item: unknown): boolean {
  return (
    typeof item === 'object' &&
    item != null &&
    'type' in item &&
    typeof item.type === 'string' &&
    RESPONSES_REPLAY_OUTPUT_ITEM_TYPES.has(item.type)
  );
}

type ResponsesReasoningSlot = {
  encrypted_content?: string;
  id?: string;
  status?: string;
};

function getResponsesReasoningSlot(
  reasoning: unknown
): ResponsesReasoningSlot | undefined {
  return typeof reasoning === 'object' && reasoning != null
    ? (reasoning as ResponsesReasoningSlot)
    : undefined;
}

function isSealedReasoningSlot(
  slot: ResponsesReasoningSlot | undefined
): slot is ResponsesReasoningSlot & { encrypted_content: string } {
  return (
    typeof slot?.encrypted_content === 'string' &&
    slot.encrypted_content.length > 0
  );
}

function resolveActiveReasoningItemId(
  incoming: ResponsesReasoningSlot | undefined,
  carried: unknown
): string | undefined {
  if (typeof incoming?.id === 'string' && incoming.id.length > 0) {
    return incoming.id;
  }
  return typeof carried === 'string' ? carried : undefined;
}

/**
 * A single `additional_kwargs.reasoning` slot has to stand in for a turn that
 * can emit many reasoning items, and the chunk merge folds it field by field:
 * `encrypted_content` and `status` are strings, so they concatenate, while
 * `id` takes whichever item arrived last. An interrupted turn replays from
 * that slot, handing the provider one item id welded to every item's
 * ciphertext — rejected as "Encrypted content could not be decrypted or
 * parsed". Keep the slot describing whichever item most recently sealed, so
 * the id, its ciphertext, and its status always come from the same item.
 *
 * `activeItemId` tracks the item currently streaming, which is the one a
 * terminal `encrypted_content` belongs to: the slot's own id has already been
 * pinned back to the last sealed item by an earlier merge.
 */
function resealReasoningItemBoundary(
  combined: AIMessageChunk,
  accumulated: ResponsesReasoningSlot | undefined,
  incoming: ResponsesReasoningSlot | undefined,
  activeItemId: string | undefined
): void {
  const merged = getResponsesReasoningSlot(
    combined.additional_kwargs.reasoning
  );
  if (merged == null || incoming == null) {
    return;
  }
  if (isSealedReasoningSlot(incoming)) {
    combined.additional_kwargs.reasoning = {
      ...merged,
      id: activeItemId ?? merged.id,
      encrypted_content: incoming.encrypted_content,
      status: incoming.status,
    };
    return;
  }
  if (!isSealedReasoningSlot(accumulated)) {
    return;
  }
  combined.additional_kwargs.reasoning = {
    ...merged,
    id: accumulated.id,
    encrypted_content: accumulated.encrypted_content,
    status: accumulated.status,
  };
}

/**
 * LangChain's Responses converter places the authoritative terminal output in
 * response_metadata.output. Its chunk merge has no way to delete provisional
 * tool_outputs or replay-position sidecars, so remove those preemption-only
 * captures once that terminal output arrives. An interrupted stream has no
 * terminal chunk and keeps the captures for replay.
 */
class ResponsesReplayAIMessageChunk extends AIMessageChunk {
  override get lc_id(): string[] {
    return [...this.lc_namespace, AIMessageChunk.lc_name()];
  }

  override concat(chunk: AIMessageChunk): this {
    const accumulated = getResponsesReasoningSlot(
      this.additional_kwargs.reasoning
    );
    const incoming = getResponsesReasoningSlot(
      chunk.additional_kwargs.reasoning
    );
    const activeItemId = resolveActiveReasoningItemId(
      incoming,
      this.additional_kwargs[OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY]
    );
    const combined = super.concat(chunk);
    resealReasoningItemBoundary(combined, accumulated, incoming, activeItemId);
    if (activeItemId != null) {
      combined.additional_kwargs[OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY] =
        activeItemId;
    }
    if (!Array.isArray(chunk.response_metadata.output)) {
      return combined;
    }
    delete combined.additional_kwargs[OPENAI_RESPONSES_ACTIVE_REASONING_ID_KEY];
    delete combined.additional_kwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY];
    const toolOutputs = combined.additional_kwargs.tool_outputs;
    if (!Array.isArray(toolOutputs)) {
      return combined;
    }
    const retainedToolOutputs = toolOutputs.filter(
      (item) => !isResponsesReplayOutputItem(item)
    );
    if (retainedToolOutputs.length === toolOutputs.length) {
      return combined;
    }
    if (retainedToolOutputs.length > 0) {
      combined.additional_kwargs.tool_outputs = retainedToolOutputs;
    } else {
      delete combined.additional_kwargs.tool_outputs;
    }
    return combined;
  }
}

function makeResponsesReplayAggregationSafe(
  chunk: ChatGenerationChunk
): ChatGenerationChunk {
  if (!AIMessageChunk.isInstance(chunk.message)) {
    return chunk;
  }
  const message = chunk.message;
  chunk.message = new ResponsesReplayAIMessageChunk({
    id: message.id,
    name: message.name,
    content: message.content,
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
    tool_calls: message.tool_calls,
    invalid_tool_calls: message.invalid_tool_calls,
    tool_call_chunks: message.tool_call_chunks,
    usage_metadata: message.usage_metadata,
  });
  return chunk;
}

function remapResponsesTextBlockIndex(
  chunk: ChatGenerationChunk,
  event: OpenAIClient.Responses.ResponseStreamEvent,
  textBlockIndices: Map<string, number>
): void {
  const position = iife(() => {
    if (
      event.type === 'response.output_text.delta' ||
      event.type === 'response.output_text.annotation.added'
    ) {
      return {
        contentIndex: event.content_index,
        outputIndex: event.output_index,
      };
    }
    if (
      event.type === 'response.output_item.added' &&
      event.item.type === 'message'
    ) {
      return { contentIndex: 0, outputIndex: event.output_index };
    }
    return undefined;
  });
  if (position == null || !Array.isArray(chunk.message.content)) {
    return;
  }
  const key = `${position.outputIndex}:${position.contentIndex}`;
  let blockIndex = textBlockIndices.get(key);
  if (blockIndex == null) {
    blockIndex = textBlockIndices.size;
    textBlockIndices.set(key, blockIndex);
  }
  const content = chunk.message.content.map((block) =>
    typeof block === 'object' && block.type === 'text'
      ? { ...block, index: blockIndex }
      : block
  );
  chunk.message.content = content;
  chunk.message.lc_kwargs.content = content;
}

function convertDroppedResponsesReplayOutput(
  event: OpenAIClient.Responses.ResponseStreamEvent
): ChatGenerationChunk | null {
  if (event.type !== 'response.output_item.done') {
    return null;
  }
  if (event.item.type === 'reasoning') {
    // Added/summary events already stream id, type, and summary. Only merge
    // terminal fields here so chunk concatenation does not duplicate summary.
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: [],
        additional_kwargs: {
          reasoning: {
            status: event.item.status,
            ...(typeof event.item.encrypted_content === 'string'
              ? { encrypted_content: event.item.encrypted_content }
              : {}),
          },
        },
        response_metadata: { model_provider: 'openai' },
      }),
    });
  }
  if (!RESPONSES_REPLAY_OUTPUT_ITEM_TYPES.has(event.item.type)) {
    return null;
  }
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: [],
      additional_kwargs: { tool_outputs: [event.item] },
      response_metadata: { model_provider: 'openai' },
    }),
  });
}

function attachResponsesReplayPosition(
  chunk: ChatGenerationChunk,
  event: OpenAIClient.Responses.ResponseStreamEvent,
  seenPositions: Set<string>
): void {
  let position: ResponsesReplayPosition | undefined;
  if (event.type === 'response.output_text.delta' && event.delta.length > 0) {
    position = {
      contentIndex: event.content_index,
      itemId: event.item_id,
      kind: 'text',
      outputIndex: event.output_index,
    };
  } else if (
    event.type === 'response.output_item.added' &&
    event.item.type === 'message' &&
    typeof event.item.id === 'string' &&
    event.item.id.length > 0
  ) {
    position = {
      itemId: event.item.id,
      kind: 'message',
      outputIndex: event.output_index,
    };
  } else if (
    event.type === 'response.output_item.added' &&
    event.item.type === 'reasoning' &&
    typeof event.item.id === 'string' &&
    event.item.id.length > 0
  ) {
    position = {
      itemId: event.item.id,
      kind: 'reasoning',
      outputIndex: event.output_index,
    };
  } else if (
    event.type === 'response.output_item.done' &&
    (RESPONSES_REPLAY_OUTPUT_ITEM_TYPES.has(event.item.type) ||
      Array.isArray(chunk.message.additional_kwargs.tool_outputs))
  ) {
    let itemId: string | undefined;
    if (typeof event.item.id === 'string' && event.item.id.length > 0) {
      itemId = event.item.id;
    } else if (
      'call_id' in event.item &&
      typeof event.item.call_id === 'string' &&
      event.item.call_id.length > 0
    ) {
      itemId = event.item.call_id;
    }
    if (itemId != null) {
      position = {
        itemId,
        kind: 'output',
        outputIndex: event.output_index,
      };
    }
  }
  if (position == null) {
    return;
  }
  const positionKey = `${position.kind}:${position.itemId}:${position.outputIndex}:${position.contentIndex ?? ''}`;
  if (seenPositions.has(positionKey)) {
    return;
  }
  seenPositions.add(positionKey);
  const existing = chunk.message.additional_kwargs[
    OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
  ] as unknown;
  const additionalKwargs = {
    ...chunk.message.additional_kwargs,
    [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
      ...(Array.isArray(existing) ? existing : []),
      position,
    ],
  };
  chunk.message.additional_kwargs = additionalKwargs;
  chunk.message.lc_kwargs.additional_kwargs = additionalKwargs;
}

type ResponsesAnnotationsBoundaryEvent = {
  type: OpenAIClient.Responses.ResponseStreamEvent['type'];
  response?: {
    output?: Array<{
      type: string;
      content?: Array<{
        type: string;
        annotations?: object[] | null;
      }>;
    }>;
  };
};

/**
 * The Responses API spec declares `annotations` required on `output_text`
 * content parts, but some OpenAI-compatible gateways omit the field on the
 * terminal `response.completed`/`response.incomplete` events. LangChain's
 * converter calls `part.annotations.map(...)` unconditionally, so a missing
 * field crashes the whole stream. Default it to `[]` before conversion.
 */
export function ensureResponsesOutputAnnotations(
  event: ResponsesAnnotationsBoundaryEvent
): void {
  if (event.type !== 'response.completed' && event.type !== 'response.incomplete') {
    return;
  }
  const output = event.response?.output;
  if (!Array.isArray(output)) {
    return;
  }
  for (const item of output) {
    if (item.type !== 'message' || !Array.isArray(item.content)) {
      continue;
    }
    for (const part of item.content) {
      if (part.type === 'output_text' && !Array.isArray(part.annotations)) {
        part.annotations = [];
      }
    }
  }
}

function getResponsesStreamError(
  event: OpenAIClient.Responses.ResponseStreamEvent
): Error | undefined {
  if (event.type === 'error') {
    return new OpenAIClient.APIError(
      undefined,
      event,
      event.message,
      undefined
    );
  }
  if (event.type !== 'response.failed') {
    return;
  }
  if (event.response.error != null) {
    return new OpenAIClient.APIError(
      undefined,
      event.response.error,
      event.response.error.message,
      undefined
    );
  }
  return new OpenAIClient.OpenAIError(
    `Response ${event.response.id} failed without error details.`
  );
}

async function* convertLibreChatResponsesStream(
  stream: AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>,
  options: ResponsesStreamChunkOptions,
  runManager?: CallbackManagerForLLMRun
): AsyncGenerator<ChatGenerationChunk> {
  const seenReplayPositions = new Set<string>();
  const responsesTextBlockIndices = new Map<string, number>();
  try {
    for await (const event of stream) {
      options.signal?.throwIfAborted();
      const streamError = getResponsesStreamError(event);
      if (streamError != null) {
        throw streamError;
      }
      ensureResponsesOutputAnnotations(event);
      const convertedChunk =
        convertResponsesDeltaToChatGenerationChunk(event) ??
        convertDroppedResponsesReplayOutput(event);
      if (convertedChunk == null) {
        continue;
      }
      const chunk = makeResponsesReplayAggregationSafe(convertedChunk);
      remapResponsesTextBlockIndex(chunk, event, responsesTextBlockIndices);
      attachResponsesReplayPosition(chunk, event, seenReplayPositions);
      attachCacheWriteUsage(chunk.message);
      await runManager?.handleLLMNewToken(
        chunk.text || '',
        {
          prompt: options.promptIndex ?? 0,
          completion: 0,
        },
        undefined,
        undefined,
        undefined,
        { chunk }
      );
      yield chunk;
    }
  } catch (e) {
    throw wrapOpenAIClientError(e);
  }
}

function createUsageMetadata(
  usage?: OpenAIClient.Completions.CompletionUsage
): UsageMetadata {
  const usageMetadata: UsageMetadata = {
    input_tokens: usage?.prompt_tokens ?? 0,
    output_tokens: usage?.completion_tokens ?? 0,
    total_tokens: usage?.total_tokens ?? 0,
  };

  if (usage == null) {
    return usageMetadata;
  }

  const inputTokenDetails: UsageMetadata['input_token_details'] = {};
  const outputTokenDetails: UsageMetadata['output_token_details'] = {};
  let hasInputTokenDetails = false;
  let hasOutputTokenDetails = false;
  const promptTokenDetails = usage.prompt_tokens_details as
    | PromptTokensDetailsWithCacheWrite
    | undefined;
  const audioInputTokens = promptTokenDetails?.audio_tokens;
  const cachedInputTokens = promptTokenDetails?.cached_tokens;
  const cacheWriteInputTokens = promptTokenDetails?.cache_write_tokens;
  const audioOutputTokens = usage.completion_tokens_details?.audio_tokens;
  const reasoningOutputTokens =
    usage.completion_tokens_details?.reasoning_tokens;

  if (audioInputTokens != null) {
    inputTokenDetails.audio = audioInputTokens;
    hasInputTokenDetails = true;
  }
  if (cachedInputTokens != null) {
    inputTokenDetails.cache_read = cachedInputTokens;
    hasInputTokenDetails = true;
  }
  if (cacheWriteInputTokens != null) {
    inputTokenDetails.cache_creation = cacheWriteInputTokens;
    hasInputTokenDetails = true;
  }
  if (audioOutputTokens != null) {
    outputTokenDetails.audio = audioOutputTokens;
    hasOutputTokenDetails = true;
  }
  if (reasoningOutputTokens != null) {
    outputTokenDetails.reasoning = reasoningOutputTokens;
    hasOutputTokenDetails = true;
  }

  if (hasInputTokenDetails) {
    usageMetadata.input_token_details = inputTokenDetails;
  }
  if (hasOutputTokenDetails) {
    usageMetadata.output_token_details = outputTokenDetails;
  }

  return usageMetadata;
}

function getExposedOpenAIClient(
  completions: OpenAIClientDelegate,
  responses: OpenAIClientDelegate,
  preferResponses: boolean
): AbortableOpenAIClient {
  const responsesClient = responses.client;
  if (responsesClient?.abortHandler != null) {
    return responsesClient;
  }
  const completionsClient = completions.client;
  if (completionsClient?.abortHandler != null) {
    return completionsClient;
  }

  const delegate = preferResponses ? responses : completions;
  delegate._getClientOptions(undefined);
  return delegate.client as AbortableOpenAIClient;
}

function getReasoningParams(
  astraRulesApply: boolean,
  baseReasoning: OpenAIClient.Reasoning | undefined,
  options?: ReasoningCallOptions
): OpenAIClient.Reasoning | undefined {
  let reasoning: OpenAIClient.Reasoning | undefined;
  if (baseReasoning !== undefined) {
    reasoning = {
      ...reasoning,
      ...baseReasoning,
    };
  }
  if (options?.reasoning !== undefined) {
    reasoning = {
      ...reasoning,
      ...options.reasoning,
    };
  }
  if (
    options?.reasoningEffort !== undefined &&
    reasoning?.effort === undefined
  ) {
    reasoning = {
      ...reasoning,
      effort: options.reasoningEffort,
    };
  }
  return substituteUnsupportedAstraEffort(astraRulesApply, reasoning);
}

function getGatedReasoningParams(
  model: string,
  astraRulesApply: boolean,
  baseReasoning: OpenAIClient.Reasoning | undefined,
  options?: ReasoningCallOptions
): OpenAIClient.Reasoning | undefined {
  if (!isReasoningModel(model)) {
    return;
  }
  return getReasoningParams(astraRulesApply, baseReasoning, options);
}

function isObject(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

function isOpenAIChatCompletionChunk(
  value: unknown
): value is OpenAIChatCompletionChunk {
  if (!isObject(value)) {
    return false;
  }

  // Intentionally loose: downstream handlers already tolerate empty choices.
  const { choices } = value as { choices?: unknown };
  return Array.isArray(choices);
}

function getOpenAIChatCompletionChunk(
  value: OpenAIChatCompletionStreamItem
): OpenAIChatCompletionChunk | undefined {
  if (isOpenAIChatCompletionChunk(value)) {
    return value;
  }

  const { data } = value;
  if (isOpenAIChatCompletionChunk(data)) {
    return data;
  }

  return undefined;
}

async function* filterOpenAIChatCompletionStream(
  stream: AsyncIterable<OpenAIChatCompletionStreamItem>
): AsyncGenerator<OpenAIChatCompletionChunk> {
  let sawFinishReason = false;
  for await (const item of stream) {
    const chunk = getOpenAIChatCompletionChunk(item);
    if (chunk == null) {
      continue;
    }
    if (chunk.choices.some((choice) => choice.finish_reason != null)) {
      sawFinishReason = true;
    }
    yield chunk;
  }
  if (!sawFinishReason) {
    throw new Error('OpenAI-compatible stream ended before completion');
  }
}

async function completionWithFilteredOpenAIStream(
  request: OpenAIChatCompletionRequest,
  requestOptions: OpenAICoreRequestOptions | undefined,
  completionWithRetry: OpenAIChatCompletionRetry
): Promise<OpenAIChatCompletionResult> {
  if (request.stream !== true) {
    return (await completionWithRetry(
      request,
      requestOptions
    )) as OpenAIChatCompletion;
  }

  const stream = await completionWithRetry(request, requestOptions);
  return filterOpenAIChatCompletionStream(
    stream as AsyncIterable<OpenAIChatCompletionStreamItem>
  );
}

function attachLibreChatDeltaFields(
  chunk: BaseMessageChunk,
  delta: Record<string, unknown>
): BaseMessageChunk {
  if (!AIMessageChunk.isInstance(chunk)) {
    return chunk;
  }

  const libreChatDelta = delta as OpenAIDeltaWithLibreChatFields;
  if (
    libreChatDelta.reasoning != null &&
    chunk.additional_kwargs.reasoning_content == null
  ) {
    chunk.additional_kwargs.reasoning_content = libreChatDelta.reasoning;
  }
  if (libreChatDelta.reasoning_details != null) {
    chunk.additional_kwargs.reasoning_details =
      libreChatDelta.reasoning_details;
  }
  if (libreChatDelta.provider_specific_fields != null) {
    chunk.additional_kwargs.provider_specific_fields =
      libreChatDelta.provider_specific_fields;
  }
  return chunk;
}

function attachLibreChatMessageFields(
  message: BaseMessage,
  rawMessage: Record<string, unknown>
): BaseMessage {
  if (!isAIMessage(message)) {
    return message;
  }
  if (
    rawMessage.reasoning != null &&
    message.additional_kwargs.reasoning_content == null
  ) {
    message.additional_kwargs.reasoning_content = rawMessage.reasoning;
  }
  if (rawMessage.reasoning_details != null) {
    message.additional_kwargs.reasoning_details = rawMessage.reasoning_details;
  }
  if (rawMessage.provider_specific_fields != null) {
    message.additional_kwargs.provider_specific_fields =
      rawMessage.provider_specific_fields;
  }
  return message;
}

function getCustomOpenAIClientOptions(
  owner: OpenAIClientOwner,
  options?: OpenAICoreRequestOptions
): OpenAICoreRequestOptions {
  if (!(owner.client as OpenAIClient | undefined)) {
    const openAIEndpointConfig: t.OpenAIEndpointConfig = {
      baseURL: owner.clientConfig.baseURL,
    };

    const endpoint = getEndpoint(openAIEndpointConfig);
    const params = {
      ...owner.clientConfig,
      baseURL: endpoint,
      timeout: owner.timeout,
      maxRetries: 0,
    };
    if (params.baseURL == null) {
      delete params.baseURL;
    }

    params.defaultHeaders = getHeadersWithUserAgent(params.defaultHeaders);
    owner.client = new CustomOpenAIClient(params);
  }
  const requestOptions = {
    ...owner.clientConfig,
    ...options,
  } as OpenAICoreRequestOptions;
  return requestOptions;
}

/**
 * Classifies a generation chunk for the smoothing engine:
 * - splittable: plain visible text (string content equal to `chunk.text`, no
 *   logprobs / finish_reason) — sliced adaptively at the pacing cadence.
 *   ANY logprobs value blocks splitting here (this family only attaches
 *   logprobs on request; the DeepSeek suite pins chunks with them staying
 *   intact) — deliberately stricter than `stream/chunkAdapters.ts`, where
 *   google-common's always-present empty logprobs must not block.
 * - atomic: text- or reasoning-bearing chunks whose metadata cannot survive
 *   slicing — paced as one piece, never split (legacy parity: these were
 *   emitted whole but still paced).
 * - passthrough: tool-call deltas, usage-only, finish_reason and other
 *   metadata chunks — strict FIFO, zero delay.
 */
export function toSmoothItem(
  chunk: ChatGenerationChunk
): SmoothItem<ChatGenerationChunk> {
  const { message } = chunk;
  const isMessageChunk = message instanceof AIMessageChunk;
  /** Chunks pairing visible text with a reasoning delta (reasoning_content,
   * reasoning summary, or OpenRouter reasoning_details) or with tool-call
   * deltas must pace whole: split pieces would each clone the same kwargs /
   * tool_call_chunks and downstream accumulation duplicates them per piece. */
  const splittable =
    Boolean(chunk.text) &&
    isMessageChunk &&
    typeof message.content === 'string' &&
    message.content === chunk.text &&
    chunk.generationInfo?.logprobs == null &&
    chunk.generationInfo?.finish_reason == null &&
    !hasReasoningKwargs(message) &&
    !hasToolCallChunks(message);

  if (splittable) {
    return {
      text: chunk.text,
      smooth: true,
      emit: (piece) => cloneGenerationChunkPiece(chunk, piece),
    };
  }

  const pacedText =
    chunk.text || (isMessageChunk ? getReasoningKwargsText(message) : '');
  if (pacedText !== '') {
    return {
      text: pacedText,
      smooth: true,
      atomic: true,
      emit: () => chunk,
    };
  }

  return { text: '', smooth: false, emit: () => chunk };
}

/**
 * Usage metadata, additional kwargs and response metadata survive only on
 * the first piece: the aggregator's dict merge concatenates string fields
 * and sums usage, so replication across pieces corrupts them. Unlike the
 * generic adapter, `generationInfo` stays on every piece — per-piece token
 * indices ride in it and `dropRepeatedScalarMetadata` owns repetition there.
 */
function cloneGenerationChunkPiece(
  chunk: ChatGenerationChunk,
  piece: SmoothPiece
): ChatGenerationChunk {
  if (piece.isFirst && piece.isLast) {
    return chunk;
  }
  const message = chunk.message as AIMessageChunk;
  return new ChatGenerationChunk({
    text: piece.text,
    generationInfo: chunk.generationInfo,
    message: new AIMessageChunk(
      Object.assign({}, message, {
        content: piece.text,
        usage_metadata: piece.isFirst ? message.usage_metadata : undefined,
        additional_kwargs: piece.isFirst ? message.additional_kwargs : {},
        response_metadata: piece.isFirst ? message.response_metadata : {},
      })
    ),
  });
}

export async function emitStreamChunkCallback(
  chunk: ChatGenerationChunk,
  runManager?: CallbackManagerForLLMRun
): Promise<void> {
  await runManager?.handleLLMNewToken(
    chunk.text,
    getStreamChunkTokenIndices(chunk),
    undefined,
    undefined,
    undefined,
    { chunk }
  );
}

function getStreamChunkTokenIndices(
  chunk: ChatGenerationChunk
): { prompt: number; completion: number } | undefined {
  const prompt = chunk.generationInfo?.prompt;
  const completion = chunk.generationInfo?.completion;

  if (typeof prompt === 'number' && typeof completion === 'number') {
    return { prompt, completion };
  }

  return undefined;
}

/**
 * Adaptive smoothing adapter for the OpenAI chat-model family, layered over
 * the shared `smoothStream` engine. Keeps the historical signature so every
 * `_streamResponseChunks` call site is unchanged.
 *
 * `seenScalarMetadata`: when provided, de-duplicates repeated scalar metadata
 * just before emitting, so token callbacks and the yielded chunk observe the
 * same cleaned data. Omitted by callers that wrap this stream and finalize
 * downstream (e.g. `ChatOpenRouter`, which needs the raw `finish_reason` as
 * its flush signal and de-duplicates after its own processing).
 */
async function* delayStreamChunks(
  chunks: AsyncGenerator<ChatGenerationChunk>,
  delay?: number,
  signal?: AbortSignal,
  runManager?: CallbackManagerForLLMRun,
  seenScalarMetadata?: SeenScalarMetadata
): AsyncGenerator<ChatGenerationChunk> {
  const source = (async function* (): AsyncGenerator<
    SmoothItem<ChatGenerationChunk>
    > {
    for await (const chunk of chunks) {
      yield toSmoothItem(chunk);
    }
  })();

  const smoothed = smoothStream({
    source,
    delayMs: delay != null && delay > 0 ? delay : 0,
    signal,
  });

  for await (const outputChunk of smoothed) {
    if (seenScalarMetadata != null) {
      dropRepeatedScalarMetadata(outputChunk, seenScalarMetadata);
    }
    await emitStreamChunkCallback(outputChunk, runManager);
    yield outputChunk;
  }
}

function createAbortHandler(controller: AbortController): () => void {
  return function (): void {
    controller.abort();
  };
}
/**
 * Formats a tool in either OpenAI format, or LangChain structured tool format
 * into an OpenAI tool format. If the tool is already in OpenAI format, return without
 * any changes. If it is in LangChain structured tool format, convert it to OpenAI tool format
 * using OpenAI's `zodFunction` util, falling back to `convertToOpenAIFunction` if the parameters
 * returned from the `zodFunction` util are not defined.
 *
 * @param {BindToolsInput} tool The tool to convert to an OpenAI tool.
 * @param {Object} [fields] Additional fields to add to the OpenAI tool.
 * @returns {ToolDefinition} The inputted tool in OpenAI tool format.
 */
/**
 * OpenAI strict function schemas require every property to appear in
 * `required`. The optional `intent` label (see `tools/intentArg.ts`) is
 * deliberately NOT required — the same schema is callable from programmatic
 * tool calling — so a tool auto-marked `strict: true` (the non-streaming
 * `json_schema` structured-output path) would be rejected as invalid before
 * execution. That path never streams a live label anyway, so the
 * marker-identified property is dropped there; every other path keeps it.
 */
function stripIntentFromStrictTools<T extends object>(params: T): T {
  const record = params as { tools?: unknown[] };
  const tools = record.tools;
  if (!Array.isArray(tools) || tools.length === 0) {
    return params;
  }
  const nextTools = tools.map((tool) => {
    const candidate = tool as {
      strict?: boolean;
      parameters?: { properties?: Record<string, unknown>; required?: unknown };
      function?: {
        strict?: boolean;
        parameters?: {
          properties?: Record<string, unknown>;
          required?: unknown;
        };
      };
    };
    /** Chat-completions tools nest under `function`; responses-API tools are flat. */
    const holder = candidate.function ?? candidate;
    if (holder.strict !== true) {
      return tool;
    }
    const parameters = holder.parameters;
    const properties = parameters?.properties;
    if (properties == null || !isIntentLabelProperty(properties[INTENT_ARG])) {
      return tool;
    }
    const required = Array.isArray(parameters?.required)
      ? (parameters.required as unknown[])
      : [];
    if (required.includes(INTENT_ARG)) {
      return tool;
    }
    const { [INTENT_ARG]: _omit, ...restProps } = properties;
    const nextParams = { ...parameters, properties: restProps };
    if (candidate.function != null) {
      return {
        ...candidate,
        function: { ...candidate.function, parameters: nextParams },
      };
    }
    return { ...candidate, parameters: nextParams };
  });
  if (nextTools.every((tool, index) => tool === tools[index])) {
    return params;
  }
  return { ...params, tools: nextTools } as T;
}

export function _convertToOpenAITool(
  tool: BindToolsInput,
  fields?: {
    /**
     * If `true`, model output is guaranteed to exactly match the JSON Schema
     * provided in the function definition.
     */
    strict?: boolean;
  }
): OpenAIClient.ChatCompletionTool {
  let toolDef: OpenAIClient.ChatCompletionTool | undefined;

  if (isLangChainTool(tool)) {
    toolDef = convertToOpenAITool(tool);
  } else {
    toolDef = tool as ToolDefinition;
  }

  if (fields?.strict !== undefined) {
    toolDef.function.strict = fields.strict;
  }

  return toolDef;
}
export class CustomOpenAIClient extends OpenAIClient {
  abortHandler?: () => void;
  async fetchWithTimeout(
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController
  ): Promise<Response> {
    const { signal, ...options } = init || {};
    const handler = createAbortHandler(controller);
    this.abortHandler = handler;
    if (signal) signal.addEventListener('abort', handler, { once: true });

    const timeout = setTimeout(handler, ms);

    const fetchOptions = {
      signal: controller.signal as AbortSignal,
      ...options,
    };
    if (fetchOptions.method != null) {
      // Custom methods like 'patch' need to be uppercased
      // See https://github.com/nodejs/undici/issues/2294
      fetchOptions.method = fetchOptions.method.toUpperCase();
    }

    return (
      // use undefined this binding; fetch errors if bound to something else in browser/cloudflare
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      this.fetch.call(undefined, url, fetchOptions).finally(() => {
        clearTimeout(timeout);
      })
    );
  }
}
export class CustomAzureOpenAIClient extends AzureOpenAIClient {
  abortHandler?: () => void;
  async fetchWithTimeout(
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController
  ): Promise<Response> {
    const { signal, ...options } = init || {};
    const handler = createAbortHandler(controller);
    this.abortHandler = handler;
    if (signal) signal.addEventListener('abort', handler, { once: true });

    const timeout = setTimeout(handler, ms);

    const fetchOptions = {
      signal: controller.signal as AbortSignal,
      ...options,
    };
    if (fetchOptions.method != null) {
      // Custom methods like 'patch' need to be uppercased
      // See https://github.com/nodejs/undici/issues/2294
      fetchOptions.method = fetchOptions.method.toUpperCase();
    }

    return (
      // use undefined this binding; fetch errors if bound to something else in browser/cloudflare
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      this.fetch.call(undefined, url, fetchOptions).finally(() => {
        clearTimeout(timeout);
      })
    );
  }
}

const OFFICIAL_OPENAI_HOSTNAME = 'api.openai.com';

/**
 * Official OpenAI (api.openai.com) and Azure OpenAI Chat Completions streams
 * emit tool-call deltas strictly sequentially by index: once a delta for a
 * later index appears, a prior index's arguments never change. Stamping this
 * adapter lets the stream handler seal a prior call for eager execution the
 * moment the next call begins. OpenAI-compatible endpoints (custom baseURL)
 * must NOT be stamped — e.g. live Kimi/Moonshot streams revise prior-index
 * args after advancing — so callers gate on the wire endpoint, not the class.
 */
function stampSequentialStreamedToolCallAdapter(
  message: BaseMessageChunk
): BaseMessageChunk {
  if (
    message instanceof AIMessageChunk &&
    (message.tool_call_chunks?.length ?? 0) > 0
  ) {
    message.response_metadata = {
      ...message.response_metadata,
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
    };
  }
  return message;
}

function isOfficialOpenAIBaseURL(baseURL: string | null | undefined): boolean {
  // The OpenAI SDK falls back to OPENAI_BASE_URL when the client has no
  // explicit baseURL, so an unset constructor value can still route to an
  // OpenAI-compatible endpoint.
  const effectiveBaseURL =
    baseURL != null && baseURL !== '' ? baseURL : process.env.OPENAI_BASE_URL;
  if (effectiveBaseURL == null || effectiveBaseURL === '') {
    return true;
  }
  // Compared through the URL parser rather than textually: it normalizes the
  // host case and drops the default :443, both of which spell the same
  // first-party endpoint, while keeping a lookalike host such as
  // `api.openai.com.example.net` a distinct hostname. A non-default port is
  // someone else's listener, so it stays proxied.
  let parsed: URL;
  try {
    parsed = new URL(effectiveBaseURL);
  } catch {
    return false;
  }
  return (
    parsed.protocol === 'https:' &&
    parsed.hostname === OFFICIAL_OPENAI_HOSTNAME &&
    parsed.port === ''
  );
}

const AZURE_FIRST_PARTY_HOST_SUFFIXES = [
  '.openai.azure.com',
  '.cognitiveservices.azure.com',
  '.api.cognitive.microsoft.com',
] as const;

/**
 * Azure OpenAI is first-party when requests resolve to an instance-name
 * endpoint or an *.openai.azure.com / *.cognitiveservices.azure.com /
 * regional *.api.cognitive.microsoft.com base path. A custom
 * `clientConfig.baseURL` or a non-Azure `azureOpenAIBasePath` routes through
 * a proxy or Azure-compatible endpoint whose stream contract is unknown, so
 * those are not stamped.
 */
function isFirstPartyAzureEndpoint(args: {
  baseURL: string | null | undefined;
  azureOpenAIBasePath: string | undefined;
}): boolean {
  if (args.baseURL != null && args.baseURL !== '') {
    return false;
  }
  if (args.azureOpenAIBasePath == null || args.azureOpenAIBasePath === '') {
    return true;
  }
  // Parsed rather than matched textually, for the reason given on
  // `isOfficialOpenAIBaseURL`: the host case carries no meaning, so an
  // equivalent mixed-case spelling must not read as a proxy. Any port is
  // accepted here, as the previous pattern did.
  let parsed: URL;
  try {
    parsed = new URL(args.azureOpenAIBasePath);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'https:') {
    return false;
  }
  return AZURE_FIRST_PARTY_HOST_SUFFIXES.some((suffix) =>
    parsed.hostname.endsWith(suffix)
  );
}

/**
 * Whether the GPT-6 Astra request-shaping rules apply to this request.
 *
 * Shaping only: which API serves the turn is the caller's decision, made where
 * the rest of the request is shaped. These rules cover what the model rejects
 * on either API — sampling and logprob parameters, unsupported reasoning
 * efforts — plus the encrypted reasoning it supports.
 *
 * Both halves must hold: the SDK knows the model is Astra, and the caller has
 * declared that this client talks to the first-party endpoint those rules
 * describe. The endpoint half is *declared* rather than inferred from a base
 * URL: only the caller knows whether a given URL is a faithful first-party
 * route, a gateway, or a proxy with its own semantics, and every gate here
 * removes capability — forcing Responses, dropping parameters, lowering effort
 * — so guessing wrong silently degrades an endpoint the SDK cannot see.
 *
 * Defaults to off. An undeclared client keeps its existing behavior and a
 * misconfigured Astra call fails with the provider's own error, which names the
 * remedy, rather than being silently rewritten.
 */
function astraRulesApply(
  model: string,
  firstPartyEndpoint: boolean | undefined
): boolean {
  return firstPartyEndpoint === true && isGpt6AstraModel(model);
}

class LibreChatOpenAICompletions extends OriginalChatOpenAICompletions {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  private includeReasoningContent?: boolean;
  private includeReasoningDetails?: boolean;
  private convertReasoningDetailsToContent?: boolean;
  private preserveToolCacheControl?: boolean;
  private promptCacheExplicit?: boolean;
  private safetyIdentifier?: string;

  constructor(fields?: LibreChatOpenAIFields) {
    super(fields);
    this.includeReasoningContent = fields?.includeReasoningContent;
    this.includeReasoningDetails = fields?.includeReasoningDetails;
    this.convertReasoningDetailsToContent =
      fields?.convertReasoningDetailsToContent;
    this.preserveToolCacheControl = fields?.preserveToolCacheControl;
    this.promptCacheExplicit = fields?.promptCacheExplicit;
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
    this.safetyIdentifier = fields?.safety_identifier;
  }

  invocationParams(
    options?: this['ParsedCallOptions'],
    extra?: { streaming?: boolean }
  ): ReturnType<OriginalChatOpenAICompletions['invocationParams']> {
    return stripIntentFromStrictTools(
      stripUnsupportedAstraParams(
        this.astraRulesApply,
        applyManagedRequestParams(super.invocationParams(options, extra), {
          promptCacheExplicit: this.promptCacheExplicit,
          safetyIdentifier: this.safetyIdentifier,
        }),
        'completions'
      )
    );
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.astraRulesApply, this.reasoning, options);
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    return getCustomOpenAIClientOptions(this, options);
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIChatCompletion>;
  async completionWithRetry(
    request:
      | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
      | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk> | OpenAIChatCompletion> {
    return completionWithFilteredOpenAIStream(
      this.promptCacheExplicit === true
        ? { ...request, messages: addChatCacheBreakpoints(request.messages) }
        : request,
      requestOptions,
      super.completionWithRetry.bind(this) as OpenAIChatCompletionRetry
    );
  }

  protected _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: OpenAIClient.Chat.Completions.ChatCompletionChunk,
    defaultRole?: OpenAIClient.Chat.ChatCompletionRole
  ): BaseMessageChunk {
    const message = attachLibreChatDeltaFields(
      super._convertCompletionsDeltaToBaseMessageChunk(
        delta,
        rawResponse,
        defaultRole
      ),
      delta
    );
    if (isOfficialOpenAIBaseURL(this.clientConfig.baseURL)) {
      return stampSequentialStreamedToolCallAdapter(message);
    }
    return message;
  }

  protected _convertCompletionsMessageToBaseMessage(
    message: OpenAIClient.ChatCompletionMessage,
    rawResponse: OpenAIClient.ChatCompletion
  ): BaseMessage {
    return attachLibreChatMessageFields(
      super._convertCompletionsMessageToBaseMessage(message, rawResponse),
      message as unknown as Record<string, unknown>
    );
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    if (
      this.includeReasoningContent !== true &&
      this.includeReasoningDetails !== true
    ) {
      return super._generate(messages, options, runManager);
    }

    options.signal?.throwIfAborted();
    const usageMetadata: Partial<UsageMetadata> = {};
    const params = this.invocationParams(options);
    const messagesMapped = _convertMessagesToOpenAIParams(
      messages,
      this.model,
      {
        includeReasoningContent: this.includeReasoningContent,
        includeReasoningDetails: this.includeReasoningDetails,
        convertReasoningDetailsToContent: this.convertReasoningDetailsToContent,
        preserveToolCacheControl: this.preserveToolCacheControl,
      }
    );

    if (params.stream === true) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      const finalChunks = new Map<number, ChatGenerationChunk>();
      for await (const chunk of stream) {
        chunk.message.response_metadata = {
          ...chunk.generationInfo,
          ...chunk.message.response_metadata,
        };
        const index =
          typeof chunk.generationInfo?.completion === 'number'
            ? chunk.generationInfo.completion
            : 0;
        const existingChunk = finalChunks.get(index);
        if (existingChunk == null) {
          finalChunks.set(index, chunk);
        } else {
          finalChunks.set(index, existingChunk.concat(chunk));
        }
      }
      const generations = Array.from(finalChunks.entries())
        .sort(([aKey], [bKey]) => aKey - bKey)
        .map(([, value]) => value);
      const { functions, function_call } = this.invocationParams(options);
      const promptTokenUsage = await this._getEstimatedTokenCountFromPrompt(
        messages,
        functions,
        function_call
      );
      const completionTokenUsage =
        await this._getNumTokensFromGenerations(generations);
      usageMetadata.input_tokens = promptTokenUsage;
      usageMetadata.output_tokens = completionTokenUsage;
      usageMetadata.total_tokens = promptTokenUsage + completionTokenUsage;
      return {
        generations,
        llmOutput: {
          estimatedTokenUsage: {
            promptTokens: usageMetadata.input_tokens,
            completionTokens: usageMetadata.output_tokens,
            totalTokens: usageMetadata.total_tokens,
          },
        },
      };
    }

    const data = await this.completionWithRetry(
      {
        ...params,
        stream: false,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );
    const {
      completion_tokens: completionTokens,
      prompt_tokens: promptTokens,
      total_tokens: totalTokens,
      prompt_tokens_details: promptTokensDetails,
      completion_tokens_details: completionTokensDetails,
    } = data.usage ?? {};

    if (completionTokens != null) {
      usageMetadata.output_tokens =
        (usageMetadata.output_tokens ?? 0) + completionTokens;
    }
    if (promptTokens != null) {
      usageMetadata.input_tokens =
        (usageMetadata.input_tokens ?? 0) + promptTokens;
    }
    if (totalTokens != null) {
      usageMetadata.total_tokens =
        (usageMetadata.total_tokens ?? 0) + totalTokens;
    }
    const promptTokensDetailsWithCacheWrite = promptTokensDetails as
      | PromptTokensDetailsWithCacheWrite
      | undefined;
    if (
      promptTokensDetailsWithCacheWrite?.audio_tokens != null ||
      promptTokensDetailsWithCacheWrite?.cached_tokens != null ||
      promptTokensDetailsWithCacheWrite?.cache_write_tokens != null
    ) {
      usageMetadata.input_token_details = {
        ...(promptTokensDetailsWithCacheWrite.audio_tokens != null && {
          audio: promptTokensDetailsWithCacheWrite.audio_tokens,
        }),
        ...(promptTokensDetailsWithCacheWrite.cached_tokens != null && {
          cache_read: promptTokensDetailsWithCacheWrite.cached_tokens,
        }),
        ...(promptTokensDetailsWithCacheWrite.cache_write_tokens != null && {
          cache_creation: promptTokensDetailsWithCacheWrite.cache_write_tokens,
        }),
      };
    }
    if (
      completionTokensDetails?.audio_tokens != null ||
      completionTokensDetails?.reasoning_tokens != null
    ) {
      usageMetadata.output_token_details = {
        ...(completionTokensDetails.audio_tokens != null && {
          audio: completionTokensDetails.audio_tokens,
        }),
        ...(completionTokensDetails.reasoning_tokens != null && {
          reasoning: completionTokensDetails.reasoning_tokens,
        }),
      };
    }

    const generations: ChatGeneration[] = [];
    for (const part of data.choices) {
      const generation: ChatGeneration = {
        text: part.message.content ?? '',
        message: this._convertCompletionsMessageToBaseMessage(
          part.message,
          data
        ),
      };
      generation.generationInfo = {
        finish_reason: part.finish_reason,
        ...(part.logprobs ? { logprobs: part.logprobs } : {}),
      };
      if (isAIMessage(generation.message)) {
        generation.message.usage_metadata = usageMetadata as UsageMetadata;
      }
      generation.message = new AIMessage(
        Object.fromEntries(
          Object.entries(generation.message).filter(
            ([key]) => !key.startsWith('lc_')
          )
        )
      );
      generations.push(generation);
    }
    return {
      generations,
      llmOutput: {
        tokenUsage: {
          promptTokens: usageMetadata.input_tokens,
          completionTokens: usageMetadata.output_tokens,
          totalTokens: usageMetadata.total_tokens,
        },
      },
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (
      this.includeReasoningContent !== true &&
      this.includeReasoningDetails !== true
    ) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }

    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model, {
        includeReasoningContent: this.includeReasoningContent,
        includeReasoningDetails: this.includeReasoningDetails,
        convertReasoningDetailsToContent: this.convertReasoningDetailsToContent,
        preserveToolCacheControl: this.preserveToolCacheControl,
      });

    const params = {
      ...this.invocationParams(options, {
        streaming: true,
      }),
      messages: messagesMapped,
      stream: true as const,
    };
    let defaultRole: OpenAIClient.Chat.ChatCompletionRole | undefined;

    const streamIterable = await this.completionWithRetry(params, options);
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;
    for await (const data of streamIterable) {
      if (options.signal?.aborted === true) {
        return;
      }
      type StreamChoice = Omit<
        OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice,
        'delta'
      > & {
        delta?: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['delta'];
      };
      const choices = data.choices as StreamChoice[] | undefined;
      const choice = choices?.[0];
      if (data.usage != null) {
        usage = data.usage;
      }
      if (choice == null) {
        continue;
      }

      const { delta } = choice;
      if (delta == null) {
        continue;
      }
      const chunk = this._convertCompletionsDeltaToBaseMessageChunk(
        delta as unknown as Record<string, unknown>,
        data,
        defaultRole
      );
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: options.promptIndex ?? 0,
        completion: choice.index,
      };
      if (typeof chunk.content !== 'string') {
        // eslint-disable-next-line no-console
        console.log(
          '[WARNING]: Received non-string content from OpenAI. This is currently not supported.'
        );
        continue;
      }
      const generationInfo: Record<string, unknown> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
        generationInfo.service_tier = data.service_tier;
      }
      if (this.logprobs === true) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text,
        newTokenIndices,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (usage) {
      const promptTokenDetails = usage.prompt_tokens_details as
        | PromptTokensDetailsWithCacheWrite
        | undefined;
      const inputTokenDetails = {
        ...(promptTokenDetails?.audio_tokens != null && {
          audio: promptTokenDetails.audio_tokens,
        }),
        ...(promptTokenDetails?.cached_tokens != null && {
          cache_read: promptTokenDetails.cached_tokens,
        }),
        ...(promptTokenDetails?.cache_write_tokens != null && {
          cache_creation: promptTokenDetails.cache_write_tokens,
        }),
      };
      const outputTokenDetails = {
        ...(usage.completion_tokens_details?.audio_tokens != null && {
          audio: usage.completion_tokens_details.audio_tokens,
        }),
        ...(usage.completion_tokens_details?.reasoning_tokens != null && {
          reasoning: usage.completion_tokens_details.reasoning_tokens,
        }),
      };
      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          response_metadata: { usage: { ...usage } },
          usage_metadata: {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            ...(Object.keys(inputTokenDetails).length > 0 && {
              input_token_details: inputTokenDetails,
            }),
            ...(Object.keys(outputTokenDetails).length > 0 && {
              output_token_details: outputTokenDetails,
            }),
          },
        }),
        text: '',
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text,
        {
          prompt: 0,
          completion: 0,
        },
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }
}

class LibreChatOpenAIResponses extends OriginalChatOpenAIResponses {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  private promptCacheExplicit?: boolean;
  private responsesPromptCache?: boolean;
  private responsesPromptCacheTtl?: PromptCacheTtl;
  private safetyIdentifier?: string;

  constructor(fields?: LibreChatOpenAIFields) {
    super(fields);
    this.promptCacheExplicit = fields?.promptCacheExplicit;
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
    this.responsesPromptCache = fields?.responsesPromptCache;
    this.responsesPromptCacheTtl = fields?.responsesPromptCacheTtl;
    this.safetyIdentifier = fields?.safety_identifier;
  }

  invocationParams(
    options?: this['ParsedCallOptions']
  ): ReturnType<OriginalChatOpenAIResponses['invocationParams']> {
    const cacheOptions = options as
      | {
          promptCache?: boolean;
          promptCacheTtl?: PromptCacheTtl;
        }
      | undefined;
    const promptCache = cacheOptions?.promptCache ?? this.responsesPromptCache;
    const cacheControl =
      promptCache === true
        ? buildAnthropicCacheControl(
          resolvePromptCacheTtl(
            cacheOptions?.promptCacheTtl ?? this.responsesPromptCacheTtl
          )
        )
        : undefined;
    const baseParams = applyManagedRequestParams(
      super.invocationParams(options),
      {
        promptCacheExplicit: this.promptCacheExplicit,
        safetyIdentifier: this.safetyIdentifier,
      }
    );
    const params = {
      ...baseParams,
      ...(baseParams.tools != null && {
        tools: stripResponsesToolCacheControl(baseParams.tools),
      }),
      ...(cacheControl != null && {
        cache_control: cacheControl,
      }),
    };
    if (shouldIncludeEncryptedReasoning(this.model, params, this.astraRulesApply)) {
      params.include = [
        ...new Set([
          ...(params.include ?? []),
          'reasoning.encrypted_content' as const,
        ]),
      ];
    }
    return stripIntentFromStrictTools(
      stripUnsupportedAstraParams(this.astraRulesApply, params, 'responses')
    );
  }

  async completionWithRetry(
    request: OpenAIClient.Responses.ResponseCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>>;
  async completionWithRetry(
    request: OpenAIClient.Responses.ResponseCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIClient.Responses.Response>;
  async completionWithRetry(
    request: ResponsesRequest,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<ResponsesResult> {
    const managedRequest = {
      ...request,
      input:
        this.promptCacheExplicit === true
          ? addResponseCacheBreakpoints(request.input)
          : request.input,
    };
    const result = await super.completionWithRetry(
      managedRequest as OpenAIClient.Responses.ResponseCreateParamsStreaming,
      requestOptions
    );
    return isResponsesStream(result)
      ? result
      : attachCacheWriteMetadata(result);
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const result = await super._generate(
      projectOpenAIResponsesProviderMessages(messages),
      options,
      runManager
    );
    for (const generation of result.generations) {
      attachCacheWriteUsage(generation.message);
    }
    return result;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const projectedMessages = projectOpenAIResponsesProviderMessages(messages);
    const stream = await this.completionWithRetry(
      {
        ...this.invocationParams(options),
        input: convertMessagesToResponsesInput({
          messages: projectedMessages,
          zdrEnabled: this.zdrEnabled ?? false,
          model: this.model,
        }),
        stream: true,
      },
      options
    );
    yield* convertLibreChatResponsesStream(stream, options, runManager);
  }

  async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* super._streamChatModelEvents(
      projectOpenAIResponsesProviderMessages(messages),
      options,
      runManager
    );
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.astraRulesApply, this.reasoning, options);
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    return getCustomOpenAIClientOptions(this, options);
  }
}

class LibreChatAzureOpenAICompletions extends OriginalAzureChatOpenAICompletions {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  private promptCacheExplicit?: boolean;
  private safetyIdentifier?: string;

  constructor(fields?: LibreChatAzureOpenAIFields) {
    super(fields);
    this.promptCacheExplicit = fields?.promptCacheExplicit;
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
    this.safetyIdentifier = fields?.safety_identifier;
  }

  invocationParams(
    options?: this['ParsedCallOptions'],
    extra?: { streaming?: boolean }
  ): ReturnType<OriginalAzureChatOpenAICompletions['invocationParams']> {
    return stripIntentFromStrictTools(
      stripUnsupportedAstraParams(
        this.astraRulesApply,
        applyManagedRequestParams(super.invocationParams(options, extra), {
          promptCacheExplicit: this.promptCacheExplicit,
          safetyIdentifier: this.safetyIdentifier,
        }),
        'completions'
      )
    );
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(
      this.model,
      this.astraRulesApply,
      this.reasoning,
      options
    );
  }

  protected _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: OpenAIClient.Chat.Completions.ChatCompletionChunk,
    defaultRole?: OpenAIClient.Chat.ChatCompletionRole
  ): BaseMessageChunk {
    const message = super._convertCompletionsDeltaToBaseMessageChunk(
      delta,
      rawResponse,
      defaultRole
    );
    if (
      isFirstPartyAzureEndpoint({
        baseURL: this.clientConfig.baseURL,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
      })
    ) {
      // First-party Azure OpenAI: same sequential-by-index stream contract
      // as api.openai.com.
      return stampSequentialStreamedToolCallAdapter(message);
    }
    return message;
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIChatCompletion>;
  async completionWithRetry(
    request:
      | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
      | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk> | OpenAIChatCompletion> {
    return completionWithFilteredOpenAIStream(
      this.promptCacheExplicit === true
        ? { ...request, messages: addChatCacheBreakpoints(request.messages) }
        : request,
      requestOptions,
      super.completionWithRetry.bind(this) as OpenAIChatCompletionRetry
    );
  }
}

class LibreChatAzureOpenAIResponses extends OriginalAzureChatOpenAIResponses {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  private promptCacheExplicit?: boolean;
  private safetyIdentifier?: string;

  constructor(fields?: LibreChatAzureOpenAIFields) {
    super(fields);
    this.promptCacheExplicit = fields?.promptCacheExplicit;
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
    this.safetyIdentifier = fields?.safety_identifier;
  }

  invocationParams(
    options?: this['ParsedCallOptions']
  ): ReturnType<OriginalAzureChatOpenAIResponses['invocationParams']> {
    const params = applyManagedRequestParams(super.invocationParams(options), {
      promptCacheExplicit: this.promptCacheExplicit,
      safetyIdentifier: this.safetyIdentifier,
    });
    if (shouldIncludeEncryptedReasoning(this.model, params, this.astraRulesApply)) {
      params.include = [
        ...new Set([
          ...(params.include ?? []),
          'reasoning.encrypted_content' as const,
        ]),
      ];
    }
    return stripIntentFromStrictTools(
      stripUnsupportedAstraParams(this.astraRulesApply, params, 'responses')
    );
  }

  async completionWithRetry(
    request: OpenAIClient.Responses.ResponseCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>>;
  async completionWithRetry(
    request: OpenAIClient.Responses.ResponseCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIClient.Responses.Response>;
  async completionWithRetry(
    request: ResponsesRequest,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<ResponsesResult> {
    const managedRequest = {
      ...request,
      input:
        this.promptCacheExplicit === true
          ? addResponseCacheBreakpoints(request.input)
          : request.input,
    };
    const result = await super.completionWithRetry(
      managedRequest as OpenAIClient.Responses.ResponseCreateParamsStreaming,
      requestOptions
    );
    return isResponsesStream(result)
      ? result
      : attachCacheWriteMetadata(result);
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const result = await super._generate(
      projectOpenAIResponsesProviderMessages(messages),
      options,
      runManager
    );
    for (const generation of result.generations) {
      attachCacheWriteUsage(generation.message);
    }
    return result;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const projectedMessages = projectOpenAIResponsesProviderMessages(messages);
    const stream = await this.completionWithRetry(
      {
        ...this.invocationParams(options),
        input: convertMessagesToResponsesInput({
          messages: projectedMessages,
          zdrEnabled: this.zdrEnabled ?? false,
          model: this.model,
        }),
        stream: true,
      },
      options
    );
    yield* convertLibreChatResponsesStream(stream, options, runManager);
  }

  async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* super._streamChatModelEvents(
      projectOpenAIResponsesProviderMessages(messages),
      options,
      runManager
    );
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(
      this.model,
      this.astraRulesApply,
      this.reasoning,
      options
    );
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }
}

function withLibreChatOpenAIFields(
  fields?: LibreChatOpenAIFields
): LibreChatOpenAIFields {
  const nextFields = fields ?? {};
  return {
    ...nextFields,
    completions:
      nextFields.completions ?? new LibreChatOpenAICompletions(nextFields),
    responses: nextFields.responses ?? new LibreChatOpenAIResponses(nextFields),
  };
}

export class ChatOpenAI extends OriginalChatOpenAI<t.ChatOpenAICallOptions> {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  _lc_stream_delay: number;

  constructor(
    fields?: LibreChatOpenAIFields & t.OpenAIChatInput['modelKwargs']
  ) {
    super(withLibreChatOpenAIFields(fields));
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
  }

  public get exposedClient(): CustomOpenAIClient {
    return getExposedOpenAIClient(
      this.completions as OpenAIClientDelegate,
      this.responses as OpenAIClientDelegate,
      this._useResponsesApi(undefined)
    ) as CustomOpenAIClient;
  }
  static lc_name(): string {
    return 'LibreChatOpenAI';
  }
  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  /**
   * Returns backwards compatible reasoning parameters from constructor params and call options
   * @internal
   */
  getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.astraRulesApply, this.reasoning, options);
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, undefined),
      this._lc_stream_delay,
      options.signal,
      runManager,
      new Map()
    );
  }

  /**
   * Raw variant that skips scalar-metadata de-duplication. Used by subclasses
   * (e.g. `ChatOpenRouter`) that read `finish_reason` as a control signal and
   * must de-duplicate only after their own finalization.
   */
  protected _streamRawResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    return delayStreamChunks(
      super._streamResponseChunks(messages, options, undefined),
      this._lc_stream_delay,
      options.signal,
      runManager
    );
  }
}

export class AzureChatOpenAI extends OriginalAzureChatOpenAI {
  protected firstPartyEndpoint?: boolean;

  /** @see {@link astraRulesApply} */
  protected get astraRulesApply(): boolean {
    return astraRulesApply(this.model, this.firstPartyEndpoint);
  }

  _lc_stream_delay: number;

  constructor(fields?: LibreChatAzureOpenAIFields) {
    super(fields);
    this.completions = new LibreChatAzureOpenAICompletions(fields);
    this.responses = new LibreChatAzureOpenAIResponses(fields);
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
    this.firstPartyEndpoint = fields?.firstPartyEndpoint;
  }

  public get exposedClient(): CustomOpenAIClient {
    return getExposedOpenAIClient(
      this.completions as OpenAIClientDelegate,
      this.responses as OpenAIClientDelegate,
      this._useResponsesApi(undefined)
    ) as CustomOpenAIClient;
  }

  static lc_name(): 'LibreChatAzureOpenAI' {
    return 'LibreChatAzureOpenAI';
  }
  /**
   * Returns backwards compatible reasoning parameters from constructor params and call options
   * @internal
   */
  getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(
      this.model,
      this.astraRulesApply,
      this.reasoning,
      options
    );
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, undefined),
      this._lc_stream_delay,
      options.signal,
      runManager,
      new Map()
    );
  }
}
export class ChatDeepSeek extends OriginalChatDeepSeek {
  _lc_stream_delay: number;

  constructor(
    fields?: ConstructorParameters<typeof OriginalChatDeepSeek>[0] & {
      _lc_stream_delay?: number;
    }
  ) {
    super(fields);
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }
  static lc_name(): 'LibreChatDeepSeek' {
    return 'LibreChatDeepSeek';
  }

  protected _convertDeepSeekMessages(
    messages: BaseMessage[]
  ): OpenAICompletionParam[] {
    return _convertMessagesToOpenAIParams(messages, this.model, {
      includeReasoningContent: true,
    });
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    options.signal?.throwIfAborted();
    const params = this.invocationParams(options);

    if (params.stream === true) {
      return super._generate(messages, options, runManager);
    }

    const messagesMapped = this._convertDeepSeekMessages(messages);
    const response = await this.completionWithRetry(
      {
        ...params,
        stream: false,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );

    const usageMetadata = createUsageMetadata(response.usage);

    const generations: ChatGeneration[] = response.choices.map((part) => {
      const text = part.message.content ?? '';
      const generation: ChatGeneration = {
        text,
        message: this._convertCompletionsMessageToBaseMessage(
          part.message,
          response
        ),
      };
      generation.generationInfo = {
        finish_reason: part.finish_reason,
        ...(part.logprobs != null ? { logprobs: part.logprobs } : {}),
      };
      if (isAIMessage(generation.message)) {
        generation.message.usage_metadata = usageMetadata;
      }
      generation.message = new AIMessage(
        Object.fromEntries(
          Object.entries(generation.message).filter(
            ([key]) => !key.startsWith('lc_')
          )
        )
      );
      return generation;
    });

    return {
      generations,
      llmOutput: {
        tokenUsage: {
          promptTokens: usageMetadata.input_tokens,
          completionTokens: usageMetadata.output_tokens,
          totalTokens: usageMetadata.total_tokens,
        },
      },
    };
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      this._streamResponseChunksWithReasoning(messages, options, undefined),
      this._lc_stream_delay,
      options.signal,
      runManager,
      new Map()
    );
  }

  /** Parses raw `<think>` fallback tags across chunks and emits sanitized DeepSeek stream chunks. */
  protected async *_streamResponseChunksWithReasoning(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const stream = this._streamResponseChunksFromReasoningMessages(
      messages,
      options
    );
    const thinkStartTag = '<think>';
    const thinkEndTag = '</think>';
    let tokensBuffer = '';
    let isThinking = false;

    for await (const chunk of stream) {
      if (options.signal?.aborted === true) {
        throw new Error('AbortError');
      }

      const reasoningContent =
        chunk.message.additional_kwargs.reasoning_content;
      if (reasoningContent != null && reasoningContent !== '') {
        yield* this._yieldDeepSeekStreamChunk(chunk, runManager);
        continue;
      }

      const text = chunk.text;
      if (text === '') {
        yield* this._yieldDeepSeekStreamChunk(chunk, runManager);
        continue;
      }

      tokensBuffer += text;

      while (tokensBuffer !== '') {
        if (isThinking) {
          const thinkEndIndex = tokensBuffer.indexOf(thinkEndTag);
          if (thinkEndIndex !== -1) {
            const thoughtContent = tokensBuffer.substring(0, thinkEndIndex);
            if (thoughtContent !== '') {
              yield* this._yieldDeepSeekReasoningText(
                chunk,
                thoughtContent,
                runManager
              );
            }

            tokensBuffer = tokensBuffer.substring(
              thinkEndIndex + thinkEndTag.length
            );
            isThinking = false;
            continue;
          }

          const splitIndex = this._getDeepSeekPartialTagSplitIndex(
            tokensBuffer,
            thinkEndTag
          );
          if (splitIndex !== -1) {
            const safeToYield = tokensBuffer.substring(0, splitIndex);
            if (safeToYield !== '') {
              yield* this._yieldDeepSeekReasoningText(
                chunk,
                safeToYield,
                runManager
              );
            }
            tokensBuffer = tokensBuffer.substring(splitIndex);
            break;
          }

          yield* this._yieldDeepSeekReasoningText(
            chunk,
            tokensBuffer,
            runManager
          );
          tokensBuffer = '';
          break;
        }

        const thinkStartIndex = tokensBuffer.indexOf(thinkStartTag);
        if (thinkStartIndex !== -1) {
          const beforeThink = tokensBuffer.substring(0, thinkStartIndex);
          if (beforeThink !== '') {
            yield* this._yieldDeepSeekStreamChunk(
              this._createDeepSeekStreamChunk(chunk, beforeThink),
              runManager
            );
          }

          tokensBuffer = tokensBuffer.substring(
            thinkStartIndex + thinkStartTag.length
          );
          isThinking = true;
          continue;
        }

        const splitIndex = this._getDeepSeekPartialTagSplitIndex(
          tokensBuffer,
          thinkStartTag
        );
        if (splitIndex !== -1) {
          const safeToYield = tokensBuffer.substring(0, splitIndex);
          if (safeToYield !== '') {
            yield* this._yieldDeepSeekStreamChunk(
              this._createDeepSeekStreamChunk(chunk, safeToYield),
              runManager
            );
          }
          tokensBuffer = tokensBuffer.substring(splitIndex);
          break;
        }

        yield* this._yieldDeepSeekStreamChunk(
          this._createDeepSeekStreamChunk(chunk, tokensBuffer),
          runManager
        );
        tokensBuffer = '';
        break;
      }
    }

    if (tokensBuffer === '') {
      return;
    }

    if (isThinking) {
      yield* this._yieldDeepSeekStreamChunk(
        new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: '',
            additional_kwargs: {
              reasoning_content: tokensBuffer,
            },
          }),
          text: '',
        }),
        runManager
      );
      return;
    }

    yield* this._yieldDeepSeekStreamChunk(
      new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: tokensBuffer,
        }),
        text: tokensBuffer,
      }),
      runManager
    );
  }

  protected async *_streamResponseChunksFromReasoningMessages(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): AsyncGenerator<ChatGenerationChunk> {
    const params = {
      ...this.invocationParams(options, { streaming: true }),
      stream: true as const,
    };
    const messagesMapped = this._convertDeepSeekMessages(messages);
    const streamIterable = await this.completionWithRetry(
      {
        ...params,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );

    let defaultRole:
      | OpenAIClient.Chat.Completions.ChatCompletionRole
      | undefined;
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;

    for await (const data of streamIterable) {
      if (options.signal?.aborted === true) {
        throw new Error('AbortError');
      }

      if (data.usage != null) {
        usage = data.usage;
      }

      if (data.choices.length === 0) {
        continue;
      }

      const choice = data.choices[0];
      const { delta } = choice;
      const messageChunk = this._convertCompletionsDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      defaultRole = delta.role ?? defaultRole;

      if (typeof messageChunk.content !== 'string') {
        continue;
      }

      const messageText = messageChunk.content;
      const newTokenIndices = {
        prompt: options.promptIndex ?? 0,
        completion: choice.index,
      };
      const generationInfo = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        Object.assign(generationInfo, {
          finish_reason: choice.finish_reason,
          system_fingerprint: data.system_fingerprint,
          model_name: data.model,
          service_tier: data.service_tier,
        });
      }
      if (this.logprobs === true) {
        Object.assign(generationInfo, { logprobs: choice.logprobs });
      }

      const generationChunk = new ChatGenerationChunk({
        message: messageChunk,
        text: messageText,
        generationInfo,
      });

      yield generationChunk;
    }

    if (usage != null) {
      const usageMetadata = createUsageMetadata(usage);

      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          response_metadata: {
            usage: { ...usage },
          },
          usage_metadata: usageMetadata,
        }),
        text: '',
        generationInfo: {
          prompt: 0,
          completion: 0,
        },
      });

      yield generationChunk;
    }

    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }

  protected _createDeepSeekStreamChunk(
    chunk: ChatGenerationChunk,
    content: string,
    additionalKwargs?: AIMessageChunk['additional_kwargs'],
    text = content
  ): ChatGenerationChunk {
    if (!(chunk.message instanceof AIMessageChunk)) {
      return new ChatGenerationChunk({
        message: new AIMessageChunk({
          content,
          additional_kwargs:
            additionalKwargs ?? chunk.message.additional_kwargs,
          response_metadata: chunk.message.response_metadata,
          id: chunk.message.id,
        }),
        text,
        generationInfo: chunk.generationInfo,
      });
    }

    const message = chunk.message;
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        content,
        additional_kwargs: additionalKwargs ?? message.additional_kwargs,
        response_metadata: message.response_metadata,
        tool_calls: message.tool_calls,
        tool_call_chunks: message.tool_call_chunks,
        id: message.id,
      }),
      text,
      generationInfo: chunk.generationInfo,
    });
  }

  protected _createDeepSeekReasoningStreamChunk(
    chunk: ChatGenerationChunk,
    reasoningContent: string
  ): ChatGenerationChunk {
    return this._createDeepSeekStreamChunk(
      chunk,
      '',
      {
        ...chunk.message.additional_kwargs,
        reasoning_content: reasoningContent,
      },
      ''
    );
  }

  protected async *_yieldDeepSeekReasoningText(
    chunk: ChatGenerationChunk,
    reasoningContent: string,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* this._yieldDeepSeekStreamChunk(
      this._createDeepSeekReasoningStreamChunk(chunk, reasoningContent),
      runManager
    );
  }

  protected async *_yieldDeepSeekStreamChunk(
    chunk: ChatGenerationChunk,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield chunk;
    await runManager?.handleLLMNewToken(
      chunk.text,
      this._getDeepSeekTokenIndices(chunk),
      undefined,
      undefined,
      undefined,
      { chunk }
    );
  }

  protected _getDeepSeekTokenIndices(
    chunk: ChatGenerationChunk
  ): { prompt: number; completion: number } | undefined {
    return getStreamChunkTokenIndices(chunk);
  }

  protected _getDeepSeekPartialTagSplitIndex(
    text: string,
    tag: string
  ): number {
    for (let i = tag.length - 1; i >= 1; i--) {
      if (text.endsWith(tag.substring(0, i))) {
        return text.length - i;
      }
    }

    return -1;
  }
}

/** xAI-specific usage metadata type */
export interface XAIUsageMetadata
  extends OpenAIClient.Completions.CompletionUsage {
  prompt_tokens_details?: {
    audio_tokens?: number;
    cached_tokens?: number;
    text_tokens?: number;
    image_tokens?: number;
  };
  completion_tokens_details?: {
    audio_tokens?: number;
    reasoning_tokens?: number;
    accepted_prediction_tokens?: number;
    rejected_prediction_tokens?: number;
  };
  num_sources_used?: number;
}

export class ChatMoonshot extends ChatOpenAI {
  constructor(
    fields?: LibreChatOpenAIFields & t.OpenAIChatInput['modelKwargs']
  ) {
    super({
      ...fields,
      includeReasoningContent: true,
    });
  }

  static lc_name(): 'LibreChatMoonshot' {
    return 'LibreChatMoonshot';
  }
}

export class ChatXAI extends OriginalChatXAI {
  _lc_stream_delay: number;

  constructor(
    fields?: Partial<ChatXAIInput> & {
      configuration?: { baseURL?: string };
      clientConfig?: { baseURL?: string };
      _lc_stream_delay?: number;
    }
  ) {
    super(fields);
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
    const customBaseURL =
      fields?.configuration?.baseURL ?? fields?.clientConfig?.baseURL;
    if (customBaseURL != null && customBaseURL) {
      this.clientConfig = {
        ...this.clientConfig,
        baseURL: customBaseURL,
      };
      // Reset the client to force recreation with new config
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.client = undefined as any;
    }
  }

  static lc_name(): 'LibreChatXAI' {
    return 'LibreChatXAI';
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, undefined),
      this._lc_stream_delay,
      options.signal,
      runManager,
      new Map()
    );
  }
}

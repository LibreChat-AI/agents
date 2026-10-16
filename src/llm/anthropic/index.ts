import { AIMessageChunk } from '@langchain/core/messages';
import { ChatAnthropicMessages } from '@langchain/anthropic';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseChatModelParams } from '@langchain/core/language_models/chat_models';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { AnthropicInput } from '@langchain/anthropic';
import type { Anthropic } from '@anthropic-ai/sdk';
import type {
  AnthropicMessageCreateParams,
  AnthropicStreamingMessageCreateParams,
  AnthropicOutputConfig,
  AnthropicBeta,
  ChatAnthropicToolType,
  AnthropicMCPServerURLDefinition,
  AnthropicContextManagementConfigParam,
  AnthropicRequestOptions,
  AnthropicMessageStreamEvent,
} from '@/llm/anthropic/types';
import type { SmoothItem } from '@/llm/stream/smoother';
import type { AnthropicUsageData } from './utils/message_outputs';
import {
  _convertMessagesToAnthropicPayload,
  stripUnsupportedAssistantPrefill,
} from './utils/message_inputs';
import { _makeMessageChunkFromAnthropicEvent } from './utils/message_outputs';
import { smoothStream, resolveStreamDelay, isSignalAborted } from '@/llm/stream/smoother';
import { convertAnthropicStream } from './utils/stream_events';
import { handleToolChoice } from './utils/tools';

type StreamTokenType = 'string' | 'input' | 'content';

interface AnthropicStreamUsage {
  inputTokens: number;
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
  outputTokens: number;
}

interface CumulativeUsageValue {
  cumulative: number;
  increment: number;
}

interface AnthropicEventStream
  extends AsyncIterable<AnthropicMessageStreamEvent> {
  controller?: { abort: () => void };
}

type BoundThinkingConfig = Anthropic.ThinkingConfigParam & {
  block_binding: { prefix_mismatch_behavior: 'drop_block' };
};

/** The `block_binding` field is a 400 without this beta header, so they are sent together. */
const THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01' as AnthropicBeta;

const ANTHROPIC_TOOL_BETAS: Partial<Record<string, AnthropicBeta>> = {
  tool_search_tool_regex_20251119: 'advanced-tool-use-2025-11-20',
  tool_search_tool_bm25_20251119: 'advanced-tool-use-2025-11-20',
  memory_20250818: 'context-management-2025-06-27',
  web_fetch_20250910: 'web-fetch-2025-09-10',
  code_execution_20250825: 'code-execution-2025-08-25',
  computer_20251124: 'computer-use-2025-11-24',
  computer_20250124: 'computer-use-2025-01-24',
  mcp_toolset: 'mcp-client-2025-11-20',
};

function _toolsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(params.tools && params.tools.length > 0);
}
export function _documentsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  for (const message of params.messages) {
    if (typeof message.content === 'string') {
      continue;
    }
    for (const block of message.content) {
      const maybeBlock: unknown = block;
      if (
        typeof maybeBlock === 'object' &&
        maybeBlock !== null &&
        'type' in maybeBlock &&
        maybeBlock.type === 'document' &&
        'citations' in maybeBlock &&
        maybeBlock.citations != null &&
        typeof maybeBlock.citations === 'object' &&
        'enabled' in maybeBlock.citations &&
        maybeBlock.citations.enabled === true
      ) {
        return true;
      }
    }
  }
  return false;
}

function _thinkingInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(
    params.thinking &&
    (params.thinking.type === 'enabled' || params.thinking.type === 'adaptive')
  );
}

function _compactionInParams(
  params: (
    | AnthropicMessageCreateParams
    | AnthropicStreamingMessageCreateParams
  ) & {
    context_management?: AnthropicContextManagementConfigParam;
  }
): boolean {
  return (
    params.context_management?.edits?.some(
      (edit) => edit.type === 'compact_20260112'
    ) === true
  );
}

function isThinkingEnabled(thinking: Anthropic.ThinkingConfigParam): boolean {
  return thinking.type === 'enabled' || thinking.type === 'adaptive';
}

function isOpus47Model(model?: string): boolean {
  return /^claude-opus-4-7(?:-|$)/.test(model ?? '');
}

function combineBetas(
  ...betaGroups: (AnthropicBeta[] | undefined)[]
): AnthropicBeta[] {
  const betas = new Set<AnthropicBeta>();
  for (const betaGroup of betaGroups) {
    for (const beta of betaGroup ?? []) {
      betas.add(beta);
    }
  }
  return [...betas];
}

function getToolBetas(tools?: ChatAnthropicToolType[]): AnthropicBeta[] {
  const betas = new Set<AnthropicBeta>();
  for (const tool of tools ?? []) {
    if (typeof tool !== 'object' || !('type' in tool)) {
      continue;
    }
    const beta = ANTHROPIC_TOOL_BETAS[String(tool.type)];
    if (beta != null) {
      betas.add(beta);
    }
  }
  return [...betas];
}

function getCompactionBetas(
  contextManagement?: AnthropicContextManagementConfigParam
): AnthropicBeta[] {
  return contextManagement?.edits?.some(
    (edit) => edit.type === 'compact_20260112'
  ) === true
    ? ['compact-2026-01-12']
    : [];
}

function getTaskBudgetBetas(
  model: string,
  outputConfig?: AnthropicOutputConfig
): AnthropicBeta[] {
  return isOpus47Model(model) &&
    outputConfig != null &&
    'task_budget' in outputConfig &&
    outputConfig.task_budget != null
    ? ['task-budgets-2026-03-13']
    : [];
}

function isSetSamplingValue(value?: number | null): value is number {
  return value != null && value !== -1;
}

function isNonDefaultTemperature(value?: number): boolean {
  return isSetSamplingValue(value) && value !== 1;
}

function validateInvocationParamCompatibility({
  model,
  thinking,
  topK,
  topP,
  temperature,
}: {
  model: string;
  thinking: Anthropic.ThinkingConfigParam;
  topK?: number;
  topP?: number | null;
  temperature?: number;
}): void {
  const opus47 = isOpus47Model(model);
  if (opus47 && thinking.type === 'enabled') {
    throw new Error(
      'thinking.type="enabled" is not supported for claude-opus-4-7; use thinking.type="adaptive" instead'
    );
  }
  if (opus47 && 'budget_tokens' in thinking) {
    throw new Error(
      'thinking.budget_tokens is not supported for claude-opus-4-7; use outputConfig.effort instead'
    );
  }
  if (opus47) {
    if (isSetSamplingValue(topK)) {
      throw new Error(
        'topK is not supported for claude-opus-4-7; omit topK/topP/temperature or use model prompting instead'
      );
    }
    if (isSetSamplingValue(topP) && topP !== 1) {
      throw new Error(
        'topP is not supported for claude-opus-4-7 when set to non-default values'
      );
    }
    if (isNonDefaultTemperature(temperature)) {
      throw new Error(
        'temperature is not supported for claude-opus-4-7 when set to non-default values'
      );
    }
  }
  if (!isThinkingEnabled(thinking)) {
    return;
  }
  if (isSetSamplingValue(topK)) {
    throw new Error('topK is not supported when thinking is enabled');
  }
  if (isSetSamplingValue(topP)) {
    throw new Error('topP is not supported when thinking is enabled');
  }
  if (isNonDefaultTemperature(temperature)) {
    throw new Error('temperature is not supported when thinking is enabled');
  }
}

function getSamplingParams({
  model,
  thinking,
  topK,
  topP,
  temperature,
}: {
  model: string;
  thinking: Anthropic.ThinkingConfigParam;
  topK?: number;
  topP?: number | null;
  temperature?: number;
}): {
  temperature?: number;
  top_k?: number;
  top_p?: number;
} {
  if (isThinkingEnabled(thinking) || isOpus47Model(model)) {
    return {};
  }
  return {
    ...(isSetSamplingValue(temperature) ? { temperature } : {}),
    ...(isSetSamplingValue(topK) ? { top_k: topK } : {}),
    ...(isSetSamplingValue(topP) ? { top_p: topP } : {}),
  };
}

async function* abortableAnthropicStream(
  source: AnthropicEventStream,
  signal?: AbortSignal
): AsyncGenerator<AnthropicMessageStreamEvent> {
  for await (const data of source) {
    if (isSignalAborted(signal)) {
      source.controller?.abort();
      return;
    }
    yield data;
  }
}

function extractToken(
  chunk: AIMessageChunk
): [string, StreamTokenType] | [undefined] {
  if (typeof chunk.content === 'string') {
    return [chunk.content, 'string'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'input' in chunk.content[0]
  ) {
    return typeof chunk.content[0].input === 'string'
      ? [chunk.content[0].input, 'input']
      : [JSON.stringify(chunk.content[0].input), 'input'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'text' in chunk.content[0]
  ) {
    const text = chunk.content[0].text;
    return typeof text === 'string' ? [text, 'content'] : [undefined];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'thinking' in chunk.content[0]
  ) {
    const thinking = chunk.content[0].thinking;
    return typeof thinking === 'string' ? [thinking, 'content'] : [undefined];
  }
  return [undefined];
}

function cloneChunk(
  text: string,
  tokenType: StreamTokenType,
  chunk: AIMessageChunk
): AIMessageChunk {
  if (tokenType === 'string') {
    return new AIMessageChunk(Object.assign({}, chunk, { content: text }));
  } else if (tokenType === 'input') {
    return chunk;
  }
  const content = chunk.content[0] as MessageContentComplex;
  if (content.type === 'text') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (content.type === 'text_delta') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (
    typeof content.type === 'string' &&
    content.type.startsWith('thinking')
  ) {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { thinking: text })],
      })
    );
  }

  return chunk;
}

function resolveCumulativeUsageValue(
  current: number | null | undefined,
  previous: number
): CumulativeUsageValue {
  if (current == null) {
    return { cumulative: previous, increment: 0 };
  }
  const cumulative = Math.max(previous, current);
  return { cumulative, increment: cumulative - previous };
}

function withIncrementalMessageDeltaUsage(
  chunk: AIMessageChunk,
  previousUsage: AnthropicStreamUsage,
  cumulativeUsage: AnthropicUsageData
): { chunk: AIMessageChunk; usage: AnthropicStreamUsage } {
  const usage = chunk.usage_metadata;
  if (usage == null) {
    return { chunk, usage: previousUsage };
  }

  const inputTokens = resolveCumulativeUsageValue(
    cumulativeUsage.input_tokens,
    previousUsage.inputTokens
  );
  const cacheCreationInputTokens = resolveCumulativeUsageValue(
    cumulativeUsage.cache_creation_input_tokens,
    previousUsage.cacheCreationInputTokens
  );
  const cacheReadInputTokens = resolveCumulativeUsageValue(
    cumulativeUsage.cache_read_input_tokens,
    previousUsage.cacheReadInputTokens
  );
  const outputTokens = resolveCumulativeUsageValue(
    cumulativeUsage.output_tokens,
    previousUsage.outputTokens
  );
  const incrementalInputTokens =
    inputTokens.increment +
    cacheCreationInputTokens.increment +
    cacheReadInputTokens.increment;
  const hasCacheUsage =
    cumulativeUsage.cache_creation_input_tokens != null ||
    cumulativeUsage.cache_read_input_tokens != null;
  const inputTokenDetails = {
    ...(cumulativeUsage.cache_creation_input_tokens != null && {
      cache_creation: cacheCreationInputTokens.increment,
    }),
    ...(cumulativeUsage.cache_read_input_tokens != null && {
      cache_read: cacheReadInputTokens.increment,
    }),
  };

  return {
    chunk: new AIMessageChunk(
      Object.assign({}, chunk, {
        usage_metadata: {
          ...usage,
          input_tokens: incrementalInputTokens,
          output_tokens: outputTokens.increment,
          total_tokens: incrementalInputTokens + outputTokens.increment,
          input_token_details: hasCacheUsage ? inputTokenDetails : undefined,
        },
      })
    ),
    usage: {
      inputTokens: inputTokens.cumulative,
      cacheCreationInputTokens: cacheCreationInputTokens.cumulative,
      cacheReadInputTokens: cacheReadInputTokens.cumulative,
      outputTokens: outputTokens.cumulative,
    },
  };
}

export type CustomAnthropicInput = AnthropicInput & {
  _lc_stream_delay?: number;
  outputConfig?: AnthropicOutputConfig;
  inferenceGeo?: string;
  contextManagement?: AnthropicContextManagementConfigParam;
  dropMismatchedThinkingBlocks?: boolean;
} & BaseChatModelParams;

export type CustomAnthropicCallOptions = {
  outputConfig?: AnthropicOutputConfig;
  outputFormat?: Anthropic.Messages.JSONOutputFormat;
  inferenceGeo?: string;
  betas?: AnthropicBeta[];
  container?: string;
  mcp_servers?: AnthropicMCPServerURLDefinition[];
};

type CustomAnthropicInvocationParams = {
  betas?: AnthropicBeta[];
  container?: string;
  context_management?: AnthropicContextManagementConfigParam;
  inference_geo?: string;
  mcp_servers?: AnthropicMCPServerURLDefinition[];
  output_config?: AnthropicOutputConfig;
};

type AnthropicEmittedChunk = {
  chunk: ChatGenerationChunk;
  token: string;
};

export class CustomAnthropic extends ChatAnthropicMessages {
  _lc_stream_delay: number;
  private tools_in_params?: boolean;
  top_k: number | undefined;
  outputConfig?: AnthropicOutputConfig;
  inferenceGeo?: string;
  contextManagement?: AnthropicContextManagementConfigParam;
  dropMismatchedThinkingBlocks?: boolean;
  constructor(fields?: CustomAnthropicInput) {
    super(fields);
    this.dropMismatchedThinkingBlocks = fields?.dropMismatchedThinkingBlocks;
    this.resetTokenEvents();
    this.setDirectFields(fields);
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
    this.outputConfig = fields?.outputConfig;
    this.inferenceGeo = fields?.inferenceGeo;
    this.contextManagement = fields?.contextManagement;
  }

  static lc_name(): 'LibreChatAnthropic' {
    return 'LibreChatAnthropic';
  }

  /**
   * Get the parameters used to invoke the model
   */
  override invocationParams(
    options?: this['ParsedCallOptions'] & CustomAnthropicCallOptions
  ): Omit<
    AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams,
    'messages'
  > &
    CustomAnthropicInvocationParams {
    const tool_choice:
      | Anthropic.Messages.ToolChoiceAuto
      | Anthropic.Messages.ToolChoiceAny
      | Anthropic.Messages.ToolChoiceTool
      | Anthropic.Messages.ToolChoiceNone
      | undefined = handleToolChoice(options?.tool_choice);

    const callOptions = options as CustomAnthropicCallOptions | undefined;
    const mergedOutputConfig: AnthropicOutputConfig | undefined = (():
      | AnthropicOutputConfig
      | undefined => {
      const base = {
        ...this.outputConfig,
        ...callOptions?.outputConfig,
      };
      if (callOptions?.outputFormat && !base.format) {
        base.format = callOptions.outputFormat;
      }
      return Object.keys(base).length > 0 ? base : undefined;
    })();

    const inferenceGeo = callOptions?.inferenceGeo ?? this.inferenceGeo;

    const contextManagement = this.contextManagement;
    const toolBetas = getToolBetas(options?.tools);
    const compactionBetas = getCompactionBetas(contextManagement);
    const taskBudgetBetas = getTaskBudgetBetas(this.model, mergedOutputConfig);
    const formattedTools = this.formatStructuredToolToAnthropic(
      options?.tools,
      {
        strict: options?.strict,
      }
    );

    const requestedThinking = this.thinkingExplicitlySet
      ? this.thinking
      : undefined;
    const bindsThinking =
      this.dropMismatchedThinkingBlocks === true &&
      requestedThinking != null &&
      isThinkingEnabled(requestedThinking);
    const sharedParams = {
      tools: formattedTools,
      tool_choice,
      // Match upstream: omit `thinking` unless the user set it, so we don't send
      // `{ type: 'disabled' }` (an unsupported param on some models) by default.
      thinking: bindsThinking
        ? ({
          ...requestedThinking,
          block_binding: { prefix_mismatch_behavior: 'drop_block' },
        } as BoundThinkingConfig)
        : requestedThinking,
      context_management: contextManagement,
      ...this.invocationKwargs,
      container: callOptions?.container,
      betas: combineBetas(
        this.betas,
        callOptions?.betas,
        toolBetas,
        compactionBetas,
        taskBudgetBetas,
        bindsThinking ? [THINKING_BINDING_BETA] : undefined
      ),
      output_config: mergedOutputConfig,
      inference_geo: inferenceGeo,
      mcp_servers: callOptions?.mcp_servers,
      // Top-level request cache_control (1.5.x): API auto-advances the cache
      // breakpoint across turns. Additive — independent of our block-level cache.
      cache_control: options?.cache_control,
    };
    validateInvocationParamCompatibility({
      model: this.model,
      thinking: this.thinking,
      topK: this.top_k,
      topP: this.topP,
      temperature: this.temperature,
    });

    return {
      model: this.model,
      stop_sequences: options?.stop ?? this.stopSequences,
      stream: this.streaming,
      max_tokens: this.maxTokens,
      ...getSamplingParams({
        model: this.model,
        thinking: this.thinking,
        topK: this.top_k,
        topP: this.topP,
        temperature: this.temperature,
      }),
      ...sharedParams,
    };
  }

  resetTokenEvents(): void {
    this.tools_in_params = undefined;
  }

  setDirectFields(fields?: CustomAnthropicInput): void {
    this.temperature = fields?.temperature ?? undefined;
    this.topP = fields?.topP ?? undefined;
    this.top_k = fields?.topK;
    if (this.temperature === -1 || this.temperature === 1) {
      this.temperature = undefined;
    }
    if (this.topP === -1) {
      this.topP = undefined;
    }
    if (this.top_k === -1) {
      this.top_k = undefined;
    }
  }

  private createGenerationChunk({
    token,
    chunk,
    shouldStreamUsage,
  }: {
    token?: string;
    chunk: AIMessageChunk;
    shouldStreamUsage: boolean;
  }): ChatGenerationChunk {
    const usage_metadata = shouldStreamUsage ? chunk.usage_metadata : undefined;
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        // Just yield chunk as it is and tool_use will be concat by BaseChatModel._generateUncached().
        content: chunk.content,
        additional_kwargs: chunk.additional_kwargs,
        tool_call_chunks: chunk.tool_call_chunks,
        response_metadata: chunk.response_metadata,
        usage_metadata,
        id: chunk.id,
      }),
      text: token ?? '',
    });
  }

  protected override async createStreamWithRetry(
    request: AnthropicStreamingMessageCreateParams,
    options?: AnthropicRequestOptions
  ): ReturnType<ChatAnthropicMessages['createStreamWithRetry']> {
    return super.createStreamWithRetry(
      stripUnsupportedAssistantPrefill(request),
      options
    );
  }

  protected override async completionWithRetry(
    request: AnthropicMessageCreateParams,
    options: AnthropicRequestOptions
  ): ReturnType<ChatAnthropicMessages['completionWithRetry']> {
    return super.completionWithRetry(
      stripUnsupportedAssistantPrefill(request),
      options
    );
  }

  async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    const params = this.invocationParams(options);
    const formattedMessages = _convertMessagesToAnthropicPayload(messages);
    const payload = stripUnsupportedAssistantPrefill({
      ...params,
      ...formattedMessages,
      stream: true,
    } as const);
    const stream = await this.createStreamWithRetry(payload, {
      headers: options.headers,
      signal: options.signal,
    });
    const shouldStreamUsage = options.streamUsage ?? this.streamUsage;
    yield* convertAnthropicStream(
      abortableAnthropicStream(stream, options.signal),
      { streamUsage: shouldStreamUsage }
    );
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.resetTokenEvents();
    const params = this.invocationParams(options);
    const formattedMessages = _convertMessagesToAnthropicPayload(messages);
    const payload = stripUnsupportedAssistantPrefill({
      ...params,
      ...formattedMessages,
      stream: true,
    } as const);
    const coerceContentToString =
      !_toolsInParams(payload) &&
      !_documentsInParams(payload) &&
      !_thinkingInParams(payload) &&
      !_compactionInParams(payload);

    const stream = await this.createStreamWithRetry(payload, {
      headers: options.headers,
      signal: options.signal,
    });

    const shouldStreamUsage = options.streamUsage ?? this.streamUsage;
    let streamUsage: AnthropicStreamUsage = {
      inputTokens: 0,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
      outputTokens: 0,
    };
    const toEmittedChunk = (
      token: string,
      chunk: AIMessageChunk
    ): AnthropicEmittedChunk => ({
      token,
      chunk: this.createGenerationChunk({ token, chunk, shouldStreamUsage }),
    });

    const source = (async function* (): AsyncGenerator<
      SmoothItem<AnthropicEmittedChunk>
      > {
      for await (const data of stream) {
        const result = _makeMessageChunkFromAnthropicEvent(
          data as Anthropic.Beta.Messages.BetaRawMessageStreamEvent,
          {
            streamUsage: shouldStreamUsage,
            coerceContentToString,
          }
        );
        if (!result) {
          continue;
        }

        let { chunk } = result;
        if (data.type === 'message_start') {
          streamUsage = {
            ...streamUsage,
            inputTokens: data.message.usage.input_tokens,
            outputTokens: data.message.usage.output_tokens,
            cacheCreationInputTokens:
              data.message.usage.cache_creation_input_tokens ?? 0,
            cacheReadInputTokens:
              data.message.usage.cache_read_input_tokens ?? 0,
          };
        }
        if (data.type === 'message_delta') {
          const incremental = withIncrementalMessageDeltaUsage(
            chunk,
            streamUsage,
            data.usage
          );
          chunk = incremental.chunk;
          streamUsage = incremental.usage;
        }

        const [token = '', tokenType] = extractToken(chunk);
        if (
          !tokenType ||
          tokenType === 'input' ||
          (token === '' && (chunk.usage_metadata != null || chunk.id != null))
        ) {
          yield {
            text: '',
            smooth: false,
            emit: (): AnthropicEmittedChunk => toEmittedChunk(token, chunk),
          };
          continue;
        }

        if (token === '') {
          continue;
        }

        yield {
          text: token,
          smooth: true,
          emit: (piece): AnthropicEmittedChunk => {
            if (piece.isFirst && piece.isLast) {
              return toEmittedChunk(token, chunk);
            }
            const cloned = cloneChunk(piece.text, tokenType, chunk);
            const chunkForPiece =
              !piece.isFirst && cloned.usage_metadata != null
                ? new AIMessageChunk(
                  Object.assign({}, cloned, { usage_metadata: undefined })
                )
                : cloned;
            return toEmittedChunk(piece.text, chunkForPiece);
          },
        };
      }
    })();

    const smoothed = smoothStream({
      source,
      delayMs: this._lc_stream_delay,
      signal: options.signal,
      abortUpstream: () => stream.controller.abort(),
    });

    try {
      for await (const emitted of smoothed) {
        yield emitted.chunk;
        await runManager?.handleLLMNewToken(
          emitted.token,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: emitted.chunk }
        );
      }
    } finally {
      this.resetTokenEvents();
    }
  }
}

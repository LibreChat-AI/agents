/**
 * Optimized ChatBedrockConverse wrapper that fixes content block merging for
 * streaming responses and adds support for latest @langchain/aws features:
 *
 * - Application Inference Profiles (PR #9129)
 * - Service Tiers (Priority/Standard/Flex) (PR #9785) - requires AWS SDK 3.966.0+
 *
 * Bedrock's `@langchain/aws` library does not include an `index` property on content
 * blocks (unlike Anthropic/OpenAI), which causes LangChain's `_mergeLists` to append
 * each streaming chunk as a separate array entry instead of merging by index.
 *
 * This wrapper takes full ownership of the stream by directly interfacing with the
 * AWS SDK client (`this.client`) and using custom handlers from `./utils/` that
 * include `contentBlockIndex` in response_metadata for every delta type. It then
 * promotes `contentBlockIndex` to an `index` property on each content block
 * (mirroring Anthropic's pattern) and strips it from metadata to avoid
 * `_mergeDicts` conflicts.
 *
 * When multiple content block types are present (e.g. reasoning + text), text deltas
 * are promoted from strings to array form with `index` so they merge correctly once
 * the accumulated content is already an array.
 */

import { ChatBedrockConverse } from '@langchain/aws';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';
import {
  ConverseCommand,
  ConverseStreamCommand,
  type ConverseStreamOutput,
  type GuardrailConfiguration,
  type GuardrailStreamConfiguration,
} from '@aws-sdk/client-bedrock-runtime';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, ResponseMetadata } from '@langchain/core/messages';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import type { SmoothItem } from '@/llm/stream/smoother';
import type { ContentBlockDeltaEvent } from './types';
import {
  convertToConverseMessages,
  createConverseToolUseStopChunk,
  convertConverseMessageToLangChainMessage,
  handleConverseStreamContentBlockStart,
  handleConverseStreamContentBlockDelta,
  handleConverseStreamMetadata,
} from './utils';
import {
  resolveBedrockPromptCacheTtl,
  supportsBedrockToolCache,
  type PromptCacheTtl,
} from '@/messages/cache';
import {
  applyCachePointsToConversePayload,
  stripUnsupportedConversePrefill,
} from './cachePoints';
import {
  smoothStream,
  resolveStreamDelay,
  isSignalAborted,
} from '@/llm/stream/smoother';
import { modelDisallowsAssistantPrefill } from '@/llm/anthropic/utils/message_inputs';
import { linkStreamLimitCanonical } from '@/llm/streamLimits';
import { isReasoningRejection, stripSignedReasoning } from './reasoningRetry';
import { Providers } from '@/common';
import { isBedrockOpenAIModel } from './utils/message_inputs';
import { neutralizeHarmonyToolConfig } from './utils/harmony';
import { insertBedrockToolCachePoint } from './toolCache';

/**
 * A `200 OK` ConverseStream whose event stream closes before a single event: no
 * `messageStart`, no `messageStop`, no exception event. OpenAI models do this for a
 * request their chat template cannot render (see `./utils/harmony`). Surfaced as an error
 * so the turn fails with this explanation instead of resolving to no message at all.
 */
export const BEDROCK_EMPTY_STREAM_MESSAGE =
  'Amazon Bedrock ended the response stream without sending any events. The model could not process this request; it may contain text the model rejects.';

/**
 * Service tier type for Bedrock invocations.
 * Requires AWS SDK >= 3.966.0 to actually work.
 * @see https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
 */
export type ServiceTierType = 'priority' | 'default' | 'flex' | 'reserved';

export type CustomGuardrailConfiguration = GuardrailConfiguration &
  Pick<GuardrailStreamConfiguration, 'streamProcessingMode'>;

type BedrockEmittedChunk = {
  chunk: ChatGenerationChunk;
  callbackChunk?: ChatGenerationChunk;
  callbackToken: string;
};

/**
 * Resolves the text a delta contributes to the smoothing cadence, preferring a
 * text delta over a reasoning delta and ignoring non-string payloads.
 */
function resolveVisibleText(text?: string, reasoningText?: string): string {
  if (typeof text === 'string') {
    return text;
  }

  if (typeof reasoningText === 'string') {
    return reasoningText;
  }

  return '';
}

/**
 * Extended input interface with additional features:
 * - applicationInferenceProfile: Use an inference profile ARN instead of model ID
 * - serviceTier: Specify service tier (Priority, Standard, Flex, Reserved)
 */
export interface CustomChatBedrockConverseInput
  extends ChatBedrockConverseInput {
  /**
   * Enables Bedrock prompt cache checkpoints for message and tool prefixes.
   */
  promptCache?: boolean;

  /**
   * Prompt-cache checkpoint TTL. Defaults to `'1h'` (extended cache) when
   * `promptCache` is enabled; set `'5m'` for the legacy 5-minute behavior.
   * Bedrock models that don't support the 1-hour TTL downgrade to 5m
   * server-side (verified on Sonnet/Opus 4.6), so the default is safe to leave
   * on; use `'5m'` for any model that rejects it.
   */
  promptCacheTtl?: PromptCacheTtl;

  /**
   * Minimum delay in milliseconds between visible streamed content deltas.
   */
  _lc_stream_delay?: number;

  /**
   * Retry a request once without signed reasoning blocks when Bedrock rejects
   * them as not matching the conversation. Set by the SDK for agents that
   * replay provider reasoning; Bedrock has no way to drop such blocks itself.
   */
  retryWithoutReasoningOnRejection?: boolean;

  /**
   * Guardrail configuration for Converse and ConverseStream invocations.
   * `streamProcessingMode` is only used by ConverseStream.
   */
  guardrailConfig?: CustomGuardrailConfiguration;

  /**
   * Application Inference Profile ARN to use for the model.
   * For example, "arn:aws:bedrock:eu-west-1:123456789102:application-inference-profile/fm16bt65tzgx"
   * When provided, this ARN will be used for the actual inference calls instead of the model ID.
   * Must still provide `model` as normal modelId to benefit from all the metadata.
   * @see https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html
   */
  applicationInferenceProfile?: string;

  /**
   * Service tier for model invocation.
   * Specifies the processing tier type used for serving the request.
   * Supported values are 'priority', 'default', 'flex', and 'reserved'.
   *
   * - 'priority': Prioritized processing for lower latency
   * - 'default': Standard processing tier
   * - 'flex': Flexible processing tier with lower cost
   * - 'reserved': Reserved capacity for consistent performance
   *
   * If not provided, AWS uses the default tier.
   * Note: Requires AWS SDK >= 3.966.0 to work.
   * @see https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
   */
  serviceTier?: ServiceTierType;
}

/**
 * Extended call options with serviceTier override support.
 */
export interface CustomChatBedrockConverseCallOptions {
  serviceTier?: ServiceTierType;
  guardrailConfig?: CustomGuardrailConfiguration;
}

export class CustomChatBedrockConverse extends ChatBedrockConverse {
  _lc_stream_delay: number;

  /**
   * Whether to insert Bedrock prompt cache checkpoints when available.
   */
  promptCache?: boolean;

  /**
   * Prompt-cache checkpoint TTL (`'5m'` legacy or `'1h'` extended cache).
   */
  promptCacheTtl?: PromptCacheTtl;

  /**
   * Application Inference Profile ARN to use instead of model ID.
   */
  applicationInferenceProfile?: string;

  /**
   * Service tier for model invocation.
   */
  serviceTier?: ServiceTierType;

  /** See {@link CustomChatBedrockConverseInput.retryWithoutReasoningOnRejection}. */
  retryWithoutReasoningOnRejection?: boolean;

  /**
   * The configured model id, captured at construction so it survives the
   * temporary `this.model` swap to an application-inference-profile ARN during
   * generation. Used to gate the Bedrock tool cache point to Claude models
   * (see {@link supportsBedrockToolCache}).
   */
  private readonly cacheModelId: string;

  constructor(fields?: CustomChatBedrockConverseInput) {
    super(fields);
    this.promptCache = fields?.promptCache;
    this.promptCacheTtl = fields?.promptCacheTtl;
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
    this.applicationInferenceProfile = fields?.applicationInferenceProfile;
    this.serviceTier = fields?.serviceTier;
    this.retryWithoutReasoningOnRejection =
      fields?.retryWithoutReasoningOnRejection;
    // `super(fields)` initializes `this.model` to LangChain's default Claude
    // model when `fields.model` is omitted, so fall back to it rather than ''
    // (which would treat the default Claude model as tool-cache-unsupported).
    this.cacheModelId = fields?.model ?? this.model;
  }

  static lc_name(): string {
    return 'LibreChatBedrockConverse';
  }

  /**
   * Get the model ID to use for API calls.
   * Returns applicationInferenceProfile if set and non-empty, otherwise returns this.model.
   */
  protected getModelId(): string {
    return this.applicationInferenceProfile != null &&
      this.applicationInferenceProfile !== ''
      ? this.applicationInferenceProfile
      : this.model;
  }

  /**
   * Override invocationParams to add serviceTier support.
   */
  override invocationParams(
    options?: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions
  ): ReturnType<ChatBedrockConverse['invocationParams']> & {
    serviceTier?: { type: ServiceTierType };
  } {
    const baseParams = super.invocationParams(options);
    const baseToolConfig = isBedrockOpenAIModel(this.model)
      ? neutralizeHarmonyToolConfig(baseParams.toolConfig)
      : baseParams.toolConfig;
    const toolConfig =
      this.promptCache === true && supportsBedrockToolCache(this.cacheModelId)
        ? insertBedrockToolCachePoint(
          baseToolConfig,
          true,
          resolveBedrockPromptCacheTtl(this.promptCacheTtl, this.cacheModelId)
        )
        : baseToolConfig;

    /** Service tier from options or fall back to class-level setting */
    const serviceTierType = options?.serviceTier ?? this.serviceTier;

    return {
      ...baseParams,
      toolConfig,
      serviceTier: serviceTierType ? { type: serviceTierType } : undefined,
    };
  }

  /**
   * Override _generateNonStreaming to use applicationInferenceProfile as modelId.
   * Uses the same model-swapping pattern as streaming for consistency.
   */
  override async _generateNonStreaming(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    if (
      isBedrockOpenAIModel(this.model) ||
      modelDisallowsAssistantPrefill(this.model)
    ) {
      try {
        return await this.generateWithLocalConverter(messages, options);
      } catch (error) {
        const retry = this.reasoningRetryMessages(error, messages);
        if (retry == null) {
          throw error;
        }
        return await this.generateWithLocalConverter(retry, options);
      }
    }
    const originalModel = this.model;
    if (
      this.applicationInferenceProfile != null &&
      this.applicationInferenceProfile !== ''
    ) {
      this.model = this.applicationInferenceProfile;
    }

    try {
      try {
        return await super._generateNonStreaming(messages, options, runManager);
      } catch (error) {
        const retry = this.reasoningRetryMessages(error, messages);
        if (retry == null) {
          throw error;
        }
        return await super._generateNonStreaming(retry, options, runManager);
      }
    } finally {
      this.model = originalModel;
    }
  }

  /**
   * Non-streaming Converse through this package's converter, as the stream already uses.
   * The parent's converter replays reasoning and sends harmony control tokens as written,
   * both of which OpenAI models on Bedrock fail on, and it cannot strip a trailing
   * assistant prefill that Claude 4.6 and later reject.
   */
  private async generateWithLocalConverter(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions
  ): Promise<ChatResult> {
    const { converseMessages: convertedMessages, converseSystem } =
      convertToConverseMessages(messages, { model: this.model });
    const converseMessages = stripUnsupportedConversePrefill(
      convertedMessages,
      this.model
    );
    const params = this.invocationParams(options);
    const modelId = this.getModelId();
    applyCachePointsToConversePayload({
      cacheControl: options.cache_control,
      system: converseSystem,
      messages: converseMessages,
      params,
      modelId,
    });
    const { output, ...responseMetadata } = await this.client.send(
      new ConverseCommand({
        modelId,
        messages: converseMessages,
        ...(converseSystem.length > 0 ? { system: converseSystem } : {}),
        requestMetadata: options.requestMetadata,
        ...(params as Record<string, unknown>),
      }),
      { abortSignal: options.signal }
    );
    if (output?.message == null) {
      throw new Error('No message found in Bedrock response.');
    }
    const message = convertConverseMessageToLangChainMessage(
      output.message,
      responseMetadata
    );
    return {
      generations: [
        {
          text: typeof message.content === 'string' ? message.content : '',
          message,
        },
      ],
    };
  }

  /**
   * Own the stream end-to-end so we have direct access to every
   * `contentBlockDelta.contentBlockIndex` from the AWS SDK.
   *
   * This replaces the parent's implementation which strips contentBlockIndex
   * from text and reasoning deltas, making it impossible to merge correctly.
   */
  override _streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    // Without the retry the stream is returned as is: an extra generator layer shifts the smoother's pacing
    return this.retryWithoutReasoningOnRejection === true
      ? this.streamWithReasoningRetry(messages, options, runManager)
      : this.streamConverse(messages, options, runManager);
  }

  private async *streamWithReasoningRetry(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    let emitted = false;
    try {
      for await (const chunk of this.streamConverse(
        messages,
        options,
        runManager
      )) {
        emitted = true;
        yield chunk;
      }
    } catch (error) {
      const retry = emitted
        ? undefined
        : this.reasoningRetryMessages(error, messages);
      if (retry == null) {
        throw error;
      }
      yield* this.streamConverse(retry, options, runManager);
    }
  }

  /**
   * The messages to send again once, without signed reasoning, after Bedrock
   * rejected the replayed blocks; undefined when this is not that case.
   */
  private reasoningRetryMessages(
    error: unknown,
    messages: BaseMessage[]
  ): BaseMessage[] | undefined {
    if (
      this.retryWithoutReasoningOnRejection !== true ||
      !isReasoningRejection(error)
    ) {
      return undefined;
    }
    const stripped = stripSignedReasoning(messages, Providers.BEDROCK);
    return stripped === messages ? undefined : stripped;
  }

  private async *streamConverse(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const { converseMessages: convertedMessages, converseSystem } =
      convertToConverseMessages(messages, { model: this.model });
    const converseMessages = stripUnsupportedConversePrefill(
      convertedMessages,
      this.model
    );
    const params = this.invocationParams(options);

    let { streamUsage } = this;
    if ((options as Record<string, unknown>).streamUsage !== undefined) {
      streamUsage = (options as Record<string, unknown>).streamUsage as boolean;
    }

    const modelId = this.getModelId();

    applyCachePointsToConversePayload({
      cacheControl: options.cache_control,
      system: converseSystem,
      messages: converseMessages,
      params,
      modelId,
    });

    const command = new ConverseStreamCommand({
      modelId,
      messages: converseMessages,
      system: converseSystem,
      ...(params as Record<string, unknown>),
    });

    const streamAbortController = new AbortController();
    const abortStream = (): void => streamAbortController.abort();
    options.signal?.addEventListener('abort', abortStream, { once: true });
    if (isSignalAborted(options.signal)) {
      abortStream();
    }

    try {
      const response = await this.client.send(command, {
        abortSignal: streamAbortController.signal,
      });

      const stream: AsyncIterable<ConverseStreamOutput> | undefined =
        response.stream;
      if (!stream) {
        throw new Error(BEDROCK_EMPTY_STREAM_MESSAGE);
      }
      const isStreamAborted = (): boolean =>
        streamAbortController.signal.aborted;

      const seenBlockIndices = new Set<number>();
      const toolUseBlockIndices = new Set<number>();

      /**
       * Guardrails can reject an already-streamed toolUse block at
       * `messageStop` (`guardrail_intervened`), after `contentBlockStop` has
       * passed. Only emit eager-execution seals when no guardrails are
       * configured, so a later intervention can't race an eagerly started tool.
       */
      const sealToolUseOnStop =
        options.guardrailConfig == null && this.guardrailConfig == null;

      /**
       * Builds the emission for one piece of a delta, reproducing the exact
       * per-piece pipeline (sliced delta → chunk → enrichment → stream-limit
       * link). `indicesSnapshot` is the arrival-time copy of the seen block
       * indices, so lazily emitted pieces observe the same enrichment
       * decisions the delta saw when it arrived.
       */
      const buildDeltaEmission = (
        contentBlockDelta: ContentBlockDeltaEvent,
        token: string,
        indicesSnapshot: Set<number>
      ): BedrockEmittedChunk => {
        const delta = contentBlockDelta.delta;
        const text = delta?.text;
        const reasoningContent = delta?.reasoningContent;
        const reasoningText = reasoningContent?.text;

        let splitDelta = contentBlockDelta;
        if (typeof text === 'string') {
          splitDelta = {
            ...contentBlockDelta,
            delta: { text: token },
          };
        } else if (
          typeof reasoningText === 'string' &&
          reasoningContent != null
        ) {
          splitDelta = {
            ...contentBlockDelta,
            delta: {
              reasoningContent: {
                ...reasoningContent,
                text: token,
              },
            },
          };
        }

        const deltaChunk = handleConverseStreamContentBlockDelta(splitDelta);
        const enrichedChunk = this.enrichChunk(deltaChunk, indicesSnapshot);
        if (enrichedChunk !== deltaChunk) {
          /** The callback copy is the same emission as the enriched yield;
           * without the link, stream-limit accounting charges both message
           * objects and Bedrock tool arguments falsely trip near half the
           * cap. Linked on the messages, which are what the accounting
           * observes. */
          linkStreamLimitCanonical(deltaChunk.message, enrichedChunk.message);
        }
        return {
          chunk: enrichedChunk,
          callbackChunk: deltaChunk,
          callbackToken: deltaChunk.text,
        };
      };

      const enrichChunk = this.enrichChunk.bind(this);
      const source = (async function* (): AsyncGenerator<
        SmoothItem<BedrockEmittedChunk>
        > {
        let sawEvent = false;
        for await (const event of stream) {
          sawEvent = true;
          if (event.contentBlockStart != null) {
            const startChunk = handleConverseStreamContentBlockStart(
              event.contentBlockStart
            );
            if (startChunk != null) {
              const idx = event.contentBlockStart.contentBlockIndex;
              if (idx != null) {
                seenBlockIndices.add(idx);
                if (event.contentBlockStart.start?.toolUse != null) {
                  toolUseBlockIndices.add(idx);
                }
              }
              const enrichedStart = enrichChunk(startChunk, seenBlockIndices);
              if (enrichedStart !== startChunk) {
                linkStreamLimitCanonical(
                  startChunk.message,
                  enrichedStart.message
                );
              }
              yield {
                text: '',
                smooth: false,
                emit: (): BedrockEmittedChunk => ({
                  chunk: enrichedStart,
                  callbackChunk: startChunk,
                  callbackToken: startChunk.text,
                }),
              };
            }
          } else if (event.contentBlockDelta != null) {
            const contentBlockDelta = event.contentBlockDelta;
            const delta = contentBlockDelta.delta;
            if (delta == null) {
              throw new Error('No delta found in content block.');
            }

            const idx = contentBlockDelta.contentBlockIndex;
            if (idx != null) {
              seenBlockIndices.add(idx);
            }

            const visibleText = resolveVisibleText(
              delta.text,
              delta.reasoningContent?.text
            );
            const indicesSnapshot = new Set(seenBlockIndices);

            if (visibleText === '') {
              yield {
                text: '',
                smooth: false,
                emit: (): BedrockEmittedChunk =>
                  buildDeltaEmission(
                    contentBlockDelta,
                    visibleText,
                    indicesSnapshot
                  ),
              };
              continue;
            }

            yield {
              text: visibleText,
              smooth: true,
              emit: (piece): BedrockEmittedChunk =>
                buildDeltaEmission(
                  contentBlockDelta,
                  piece.text,
                  indicesSnapshot
                ),
            };
          } else if (event.metadata != null) {
            const metadataChunk = handleConverseStreamMetadata(event.metadata, {
              streamUsage,
            });
            yield {
              text: '',
              smooth: false,
              emit: (): BedrockEmittedChunk => ({
                chunk: metadataChunk,
                callbackToken: '',
              }),
            };
          } else if (event.contentBlockStop != null) {
            const stopIdx = event.contentBlockStop.contentBlockIndex;
            if (stopIdx != null) {
              seenBlockIndices.add(stopIdx);
              if (sealToolUseOnStop && toolUseBlockIndices.has(stopIdx)) {
                const sealChunk = createConverseToolUseStopChunk(stopIdx);
                yield {
                  text: '',
                  smooth: false,
                  emit: (): BedrockEmittedChunk => ({
                    chunk: sealChunk,
                    callbackChunk: sealChunk,
                    callbackToken: sealChunk.text,
                  }),
                };
              }
            }
          } else {
            const eventChunk = new ChatGenerationChunk({
              text: '',
              message: new AIMessageChunk({
                content: '',
                response_metadata: { ...event } as ResponseMetadata,
              }),
            });
            yield {
              text: '',
              smooth: false,
              emit: (): BedrockEmittedChunk => ({
                chunk: eventChunk,
                callbackToken: '',
              }),
            };
          }
        }
        if (!sawEvent && !isStreamAborted()) {
          throw new Error(BEDROCK_EMPTY_STREAM_MESSAGE);
        }
      })();

      const smoothed = smoothStream({
        source,
        delayMs: this._lc_stream_delay,
        signal: options.signal,
        abortUpstream: abortStream,
      });

      for await (const emitted of smoothed) {
        yield emitted.chunk;

        if (emitted.callbackChunk != null) {
          await runManager?.handleLLMNewToken(
            emitted.callbackToken,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: emitted.callbackChunk }
          );
        }
      }
    } finally {
      options.signal?.removeEventListener('abort', abortStream);
      if (!streamAbortController.signal.aborted) {
        streamAbortController.abort();
      }
    }
  }

  /**
   * Inject `index` on content blocks for proper merge behaviour, then strip
   * `contentBlockIndex` from response_metadata to prevent `_mergeDicts` conflicts.
   *
   * Text string content is promoted to array form only when the stream contains
   * multiple content block indices (e.g. reasoning at index 0, text at index 1),
   * ensuring text merges correctly with the already-array accumulated content.
   */
  private enrichChunk(
    chunk: ChatGenerationChunk,
    seenBlockIndices: Set<number>
  ): ChatGenerationChunk {
    const message = chunk.message;
    if (!(message instanceof AIMessageChunk)) {
      return chunk;
    }

    const metadata = message.response_metadata as Record<string, unknown>;
    const blockIndex = this.extractContentBlockIndex(metadata);
    const hasMetadataIndex = blockIndex != null;

    let content: AIMessageChunk['content'] = message.content;
    let contentModified = false;

    if (Array.isArray(content) && blockIndex != null) {
      content = content.map((block) =>
        typeof block === 'object' && !('index' in block)
          ? { ...block, index: blockIndex }
          : block
      );
      contentModified = true;
    } else if (
      typeof content === 'string' &&
      content !== '' &&
      blockIndex != null &&
      seenBlockIndices.size > 1
    ) {
      content = [{ type: 'text', text: content, index: blockIndex }];
      contentModified = true;
    }

    if (!contentModified && !hasMetadataIndex) {
      return chunk;
    }

    const cleanedMetadata = hasMetadataIndex
      ? (this.removeContentBlockIndex(metadata) as Record<string, unknown>)
      : metadata;

    return new ChatGenerationChunk({
      text: chunk.text,
      message: new AIMessageChunk({
        ...message,
        content,
        response_metadata: cleanedMetadata,
      }),
      generationInfo: chunk.generationInfo,
    });
  }

  /**
   * Extract `contentBlockIndex` from the top level of response_metadata.
   * Our custom handlers always place it at the top level.
   */
  private extractContentBlockIndex(
    metadata: Record<string, unknown>
  ): number | undefined {
    if (
      'contentBlockIndex' in metadata &&
      typeof metadata.contentBlockIndex === 'number'
    ) {
      return metadata.contentBlockIndex;
    }
    return undefined;
  }

  private removeContentBlockIndex(obj: unknown): unknown {
    if (obj === null || obj === undefined) {
      return obj;
    }

    if (Array.isArray(obj)) {
      return obj.map((item) => this.removeContentBlockIndex(item));
    }

    if (typeof obj === 'object') {
      const cleaned: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key !== 'contentBlockIndex') {
          cleaned[key] = this.removeContentBlockIndex(value);
        }
      }
      return cleaned;
    }

    return obj;
  }
}

export type { ChatBedrockConverseInput };

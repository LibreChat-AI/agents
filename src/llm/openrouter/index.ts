import type {
  ChatOpenAICallOptions,
  OpenAIChatInput,
  OpenAIClient,
} from '@langchain/openai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { ClientOptions } from 'openai';
import type { SeenScalarMetadata } from '@/llm/openai/streamMetadata';
import type { PromptCacheTtl } from '@/messages/cache';
import { dropRepeatedScalarMetadata } from '@/llm/openai/streamMetadata';
import { ChatOpenAI, emitStreamChunkCallback } from '@/llm/openai';

export type OpenRouterReasoningEffort =
  | 'xhigh'
  | 'high'
  | 'medium'
  | 'low'
  | 'minimal'
  | 'none';

export interface OpenRouterReasoning {
  effort?: OpenRouterReasoningEffort;
  max_tokens?: number;
  exclude?: boolean;
  enabled?: boolean;
}

export interface ChatOpenRouterCallOptions
  extends Omit<ChatOpenAICallOptions, 'reasoning'> {
  /** @deprecated Use `reasoning` object instead */
  include_reasoning?: boolean;
  reasoning?: OpenRouterReasoning;
  modelKwargs?: OpenAIChatInput['modelKwargs'];
  useResponsesApi?: boolean;
  promptCache?: boolean;
  /**
   * Prompt-cache breakpoint TTL. Defaults to `'1h'` (extended cache) when
   * `promptCache` is enabled; set `'5m'` for the legacy 5-minute behavior.
   * OpenRouter forwards this to Claude upstreams (Anthropic / Bedrock / Vertex),
   * which downgrade to 5m where the extended TTL isn't supported.
   */
  promptCacheTtl?: PromptCacheTtl;
}

export type ChatOpenRouterInput = Partial<
  ChatOpenRouterCallOptions & OpenAIChatInput
> & {
  /** OpenAI-compatible client options, including a custom fetch transport. */
  configuration?: ClientOptions;
  /** Minimum delay in ms between visible streamed deltas (default 25; 0 disables). */
  _lc_stream_delay?: number;
};

/** invocationParams return type extended with OpenRouter reasoning */
export type OpenRouterInvocationParams = Omit<
  OpenAIClient.Chat.ChatCompletionCreateParams,
  'messages'
> & {
  reasoning?: OpenRouterReasoning;
  cache_control?: { type: 'ephemeral'; ttl?: '1h' };
};

type InvocationParamsExtra = {
  streaming?: boolean;
};

interface OpenRouterReasoningTextDetail {
  type: 'reasoning.text';
  text?: string;
  format?: string;
  index?: number;
}

interface OpenRouterReasoningEncryptedDetail {
  type: 'reasoning.encrypted';
  id?: string;
  data?: string;
  format?: string;
  index?: number;
}

type OpenRouterReasoningDetail =
  | OpenRouterReasoningTextDetail
  | OpenRouterReasoningEncryptedDetail;

function isReasoningTextDetail(
  value: unknown
): value is OpenRouterReasoningTextDetail {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    value.type === 'reasoning.text'
  );
}

function isReasoningEncryptedDetail(
  value: unknown
): value is OpenRouterReasoningEncryptedDetail {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    value.type === 'reasoning.encrypted'
  );
}

function getReasoningDetails(value: unknown): OpenRouterReasoningDetail[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (detail): detail is OpenRouterReasoningDetail =>
      isReasoningTextDetail(detail) || isReasoningEncryptedDetail(detail)
  );
}

export class ChatOpenRouter extends ChatOpenAI {
  private openRouterReasoning?: OpenRouterReasoning;
  /** @deprecated Use `reasoning` object instead */
  private includeReasoning?: boolean;

  constructor(_fields: ChatOpenRouterInput) {
    const fieldsWithoutPromptCache: ChatOpenRouterInput = { ..._fields };
    delete fieldsWithoutPromptCache.promptCache;
    delete fieldsWithoutPromptCache.promptCacheTtl;

    const {
      include_reasoning,
      reasoning: openRouterReasoning,
      modelKwargs = {},
      ...fields
    } = fieldsWithoutPromptCache;

    // Extract reasoning from modelKwargs if provided there (e.g., from LLMConfig)
    const { reasoning: mkReasoning, ...restModelKwargs } = modelKwargs as {
      reasoning?: OpenRouterReasoning;
    } & Record<string, unknown>;
    const mergedReasoning =
      mkReasoning != null || openRouterReasoning != null
        ? {
          ...mkReasoning,
          ...openRouterReasoning,
        }
        : undefined;
    const runtimeReasoning =
      mergedReasoning ??
      (include_reasoning === true ? { enabled: true } : undefined);
    const parentModelKwargs =
      runtimeReasoning == null
        ? restModelKwargs
        : { ...restModelKwargs, reasoning: runtimeReasoning };

    super({
      ...fields,
      modelKwargs: parentModelKwargs,
      includeReasoningDetails: true,
      convertReasoningDetailsToContent: true,
      preserveToolCacheControl: true,
      responsesPromptCache: _fields.promptCache,
      responsesPromptCacheTtl: _fields.promptCacheTtl,
    });

    // Merge reasoning config: modelKwargs.reasoning < constructor reasoning
    if (mergedReasoning != null) {
      this.openRouterReasoning = mergedReasoning;
    }

    this.includeReasoning = include_reasoning;
  }
  static lc_name(): 'LibreChatOpenRouter' {
    return 'LibreChatOpenRouter';
  }

  // OpenRouter widens OpenAI reasoning with extra effort levels ('xhigh' | 'none' | 'minimal').
  override invocationParams(
    options?: this['ParsedCallOptions'],
    extra?: InvocationParamsExtra
  ): OpenRouterInvocationParams {
    type MutableParams = Omit<
      OpenAIClient.Chat.ChatCompletionCreateParams,
      'messages'
    > & {
      reasoning_effort?: string;
      reasoning?: OpenRouterReasoning;
      cache_control?: { type: 'ephemeral'; ttl?: '1h' };
    };

    const optionsWithDefaults = this._combineCallOptions(options);
    const useResponsesApi = this._useResponsesApi(options);
    const params = (
      useResponsesApi
        ? this.responses.invocationParams(optionsWithDefaults)
        : this.completions.invocationParams(optionsWithDefaults, extra)
    ) as MutableParams;

    // Remove the OpenAI-native reasoning_effort that the parent sets;
    // OpenRouter uses a `reasoning` object instead
    delete params.reasoning_effort;

    // Build the OpenRouter reasoning config
    const reasoning = this.buildOpenRouterReasoning(optionsWithDefaults);
    if (reasoning != null) {
      params.reasoning = reasoning;
    } else {
      delete params.reasoning;
    }

    return params;
  }

  private buildOpenRouterReasoning(
    options?: this['ParsedCallOptions']
  ): OpenRouterReasoning | undefined {
    let reasoning: OpenRouterReasoning | undefined;

    // 1. Instance-level reasoning config (from constructor)
    if (this.openRouterReasoning != null) {
      reasoning = { ...this.openRouterReasoning };
    }

    // 2. LangChain-style reasoning params (from parent's `this.reasoning`)
    const lcReasoning = this.getReasoningParams(options);
    if (lcReasoning?.effort != null) {
      reasoning = {
        ...reasoning,
        effort: lcReasoning.effort as OpenRouterReasoningEffort,
      };
    }

    // 3. Call-level reasoning override
    const callReasoning = (options as ChatOpenRouterCallOptions | undefined)
      ?.reasoning;
    if (callReasoning != null) {
      reasoning = { ...reasoning, ...callReasoning };
    }

    // 4. Legacy include_reasoning backward compatibility
    if (reasoning == null && this.includeReasoning === true) {
      reasoning = { enabled: true };
    }

    return reasoning;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const reasoningTextByIndex = new Map<
      number,
      OpenRouterReasoningTextDetail
    >();
    const reasoningEncryptedById = new Map<
      string,
      OpenRouterReasoningEncryptedDetail
    >();
    // Consume the raw (un-deduped) stream so `finish_reason` survives as the
    // reasoning-flush signal below; de-duplicate after that, before emitting.
    const seenScalarMetadata: SeenScalarMetadata = new Map();

    for await (const generationChunk of super._streamRawResponseChunks(
      messages,
      options,
      undefined
    )) {
      let currentReasoningText = '';
      const reasoningDetails = getReasoningDetails(
        generationChunk.message.additional_kwargs.reasoning_details
      );

      for (const detail of reasoningDetails) {
        if (detail.type === 'reasoning.text') {
          currentReasoningText += detail.text ?? '';
          const index = detail.index ?? 0;
          const existing = reasoningTextByIndex.get(index);
          if (existing != null) {
            existing.text = `${existing.text ?? ''}${detail.text ?? ''}`;
            continue;
          }
          reasoningTextByIndex.set(index, {
            ...detail,
            text: detail.text ?? '',
          });
          continue;
        }
        if (detail.id != null) {
          reasoningEncryptedById.set(detail.id, { ...detail });
        }
      }

      if (
        currentReasoningText.length > 0 &&
        generationChunk.message.additional_kwargs.reasoning == null
      ) {
        generationChunk.message.additional_kwargs.reasoning =
          currentReasoningText;
      }

      if (generationChunk.generationInfo?.finish_reason != null) {
        const finalReasoningDetails = [
          ...reasoningTextByIndex.values(),
          ...reasoningEncryptedById.values(),
        ];
        if (finalReasoningDetails.length > 0) {
          generationChunk.message.additional_kwargs.reasoning_details =
            finalReasoningDetails;
        } else {
          delete generationChunk.message.additional_kwargs.reasoning_details;
        }
      } else {
        delete generationChunk.message.additional_kwargs.reasoning_details;
      }

      dropRepeatedScalarMetadata(generationChunk, seenScalarMetadata);
      await emitStreamChunkCallback(generationChunk, runManager);
      yield generationChunk;
    }
  }
}

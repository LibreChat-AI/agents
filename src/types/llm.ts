// src/types/llm.ts
import { ChatMistralAI } from '@langchain/mistralai';
import type {
  OpenAIChatInput,
  ChatOpenAIFields,
  AzureOpenAIInput,
  ClientOptions as OAIClientOptions,
} from '@langchain/openai';
import type {
  BindToolsInput,
  BaseChatModel,
  BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import type { GoogleGenerativeAIChatInput } from '@langchain/google-genai';
import type { ChatVertexAIInput } from '@langchain/google-vertexai';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import type { ChatMistralAIInput } from '@langchain/mistralai';
import type { ChatDeepSeekInput } from '@langchain/deepseek';
import type { RequestOptions } from '@google/generative-ai';
import type { StructuredTool } from '@langchain/core/tools';
import type { AnthropicInput } from '@langchain/anthropic';
import type { Runnable } from '@langchain/core/runnables';
import type { OpenAI as OpenAIClient } from 'openai';
import type { ChatXAIInput } from '@langchain/xai';
import type { CustomProviderOptionsMap } from '../provider-registration';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import type { NativeMediaPort } from '@/llm/google/native';
import type { PromptCacheTtl } from '@/messages/cache';
import {
  AzureChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  ChatOpenAI,
  ChatXAI,
} from '@/llm/openai';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatVertexAI } from '@/llm/vertexai';
import { Providers } from '@/common';

export type {
  NativeMediaPort,
  NativeMediaPart,
  NativeMediaContent,
  NativeMediaReference,
  NativeMediaRestoreInput,
  NativeMediaProviderOutcome,
} from '@/llm/google/native';

export type AzureClientOptions = Partial<OpenAIChatInput> &
  Partial<AzureOpenAIInput> & {
    openAIApiKey?: string;
    openAIApiVersion?: string;
    openAIBasePath?: string;
    deploymentName?: string;
  } & BaseChatModelParams & {
    configuration?: OAIClientOptions;
  } & ManagedRequestOptions &
  StreamSmoothingOptions;
/**
 * Controls whether Claude's reasoning content is returned in adaptive
 * thinking responses. Added for Claude Opus 4.7, which omits thinking by
 * default unless the caller opts in with `'summarized'`.
 * @see https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7#thinking-content-omitted-by-default
 */
export type ThinkingDisplay = 'summarized' | 'omitted';
export type ThinkingConfigAdaptive = {
  type: 'adaptive';
  display?: ThinkingDisplay;
};
export type ThinkingConfig =
  | NonNullable<AnthropicInput['thinking']>
  | ThinkingConfigAdaptive;
export type ChatOpenAIToolType =
  | BindToolsInput
  | OpenAIClient.ChatCompletionTool;
export type CommonToolType = StructuredTool | ChatOpenAIToolType;
export type AnthropicReasoning = {
  thinking?: ThinkingConfig | boolean;
  thinkingBudget?: number;
};
export type GoogleThinkingConfig = {
  thinkingBudget?: number;
  includeThoughts?: boolean;
  thinkingLevel?: 'THINKING_LEVEL_UNSPECIFIED' | 'LOW' | 'MEDIUM' | 'HIGH';
};
/** GPT-5.6 managed-request passthrough fields, shared by the OpenAI and
 *  Azure wrappers that both read them. */
export type ManagedRequestOptions = {
  promptCacheExplicit?: boolean;
  safety_identifier?: string;
  /**
   * Declares that this client talks to the first-party OpenAI or Azure surface.
   * Gates the model-specific request *shaping* documented only for it —
   * currently GPT-6 Astra's rejected sampling and logprob parameters, its
   * unsupported reasoning efforts, and the encrypted reasoning it supports —
   * and defaults to off.
   *
   * Shaping only. Which API serves the turn is not decided here: GPT-6 Astra
   * serves tool calls only from the Responses API, and a caller wanting them
   * must select it with `useResponsesApi`, alongside the rest of the request
   * shaping that depends on which API is in use.
   *
   * The shaping runs inside this SDK's own request delegates. A caller that
   * supplies its own `completions` or `responses` delegate replaces that
   * construction and owns the request shaping for it — this flag cannot reach
   * inside a delegate it did not build.
   *
   * Declared rather than inferred from a base URL: only the caller knows
   * whether a URL is a faithful first-party route, a gateway, or a proxy with
   * its own semantics, and every gate it controls removes capability, so
   * guessing wrong silently degrades an endpoint the SDK cannot see.
   */
  firstPartyEndpoint?: boolean;
};
/**
 * Adaptive stream-smoothing configuration shared by every provider client.
 */
export type StreamSmoothingOptions = {
  /**
   * Minimum delay in milliseconds between visible streamed content deltas.
   * Defaults to 25; piece sizes adapt to the backlog so render lag stays
   * bounded regardless of provider chunk size. Set 0 to disable smoothing.
   */
  _lc_stream_delay?: number;
};

export type OpenAIClientOptions = ChatOpenAIFields &
  ManagedRequestOptions &
  StreamSmoothingOptions;
export type AnthropicClientOptions = Omit<AnthropicInput, 'thinking'> &
  StreamSmoothingOptions & {
    thinking?: ThinkingConfig;
    promptCache?: boolean;
    /**
     * Prompt-cache breakpoint TTL. Defaults to `'1h'` (extended cache) when
     * `promptCache` is enabled; set `'5m'` to opt back into the legacy
     * 5-minute behavior.
     */
    promptCacheTtl?: PromptCacheTtl;
  };
export type MistralAIClientOptions = ChatMistralAIInput &
  StreamSmoothingOptions;
export type VertexAIClientOptions = ChatVertexAIInput &
  StreamSmoothingOptions & {
    includeThoughts?: boolean;
    thinkingConfig?: GoogleThinkingConfig;
  };
export type BedrockAnthropicInput = ChatBedrockConverseInput &
  StreamSmoothingOptions & {
    additionalModelRequestFields?: ChatBedrockConverseInput['additionalModelRequestFields'] &
      AnthropicReasoning;
    promptCache?: boolean;
    /**
     * Prompt-cache checkpoint TTL. Defaults to `'1h'` (extended cache) when
     * `promptCache` is enabled; set `'5m'` to opt into the legacy 5-minute
     * behavior. Bedrock models that don't support the 1-hour TTL downgrade to 5m
     * server-side, so the default is safe to leave on.
     */
    promptCacheTtl?: PromptCacheTtl;
  };
export type BedrockConverseClientOptions = BedrockAnthropicInput;
export type BedrockAnthropicClientOptions = BedrockAnthropicInput;
export type GoogleClientOptions = GoogleGenerativeAIChatInput &
  StreamSmoothingOptions & {
    nativeMedia?: NativeMediaPort;
    responseModalities?: string[];
    customHeaders?: RequestOptions['customHeaders'];
    thinkingConfig?: GoogleThinkingConfig;
    includeServerSideToolInvocations?: boolean;
  };
export type DeepSeekClientOptions = Partial<ChatDeepSeekInput> &
  StreamSmoothingOptions;
export type XAIClientOptions = ChatXAIInput & StreamSmoothingOptions;

export type BuiltInClientOptions =
  | OpenAIClientOptions
  | AzureClientOptions
  | AnthropicClientOptions
  | MistralAIClientOptions
  | VertexAIClientOptions
  | BedrockConverseClientOptions
  | GoogleClientOptions
  | DeepSeekClientOptions
  | XAIClientOptions;

type CustomProviderName = Extract<keyof CustomProviderOptionsMap, string>;

type LooseRuntimeProviderName = string & {
  readonly __runtimeProviderName?: never;
};

export type ProviderName =
  | keyof ProviderOptionsMap
  | CustomProviderName
  | LooseRuntimeProviderName;

declare const RUNTIME_PROVIDER_NAME: unique symbol;

/** A runtime provider without declaration-merged option types. */
export type RuntimeProviderName = string & {
  readonly [RUNTIME_PROVIDER_NAME]: true;
};

export type ClientOptions =
  | BuiltInClientOptions
  | CustomProviderOptionsMap[CustomProviderName];

export type SharedLLMConfig<
  P extends
    | keyof ProviderOptionsMap
    | CustomProviderName
    | RuntimeProviderName = keyof ProviderOptionsMap,
> = {
  provider: P;
  model?: string;
  _lc_stream_delay?: number;
};

type CustomProviderClientOptionsConfig<P extends CustomProviderName> = {
  provider: P;
} & (object extends CustomProviderOptionsMap[P]
  ? { clientOptions?: CustomProviderOptionsMap[P] }
  : { clientOptions: CustomProviderOptionsMap[P] });

export type ProviderClientOptionsConfig =
  | {
      provider: keyof ProviderOptionsMap;
      clientOptions?: BuiltInClientOptions;
    }
  | {
      [P in CustomProviderName]: CustomProviderClientOptionsConfig<P>;
    }[CustomProviderName]
  | {
      provider: RuntimeProviderName;
      clientOptions: ClientOptions;
    };

export type FallbackConfig = ProviderClientOptionsConfig & {
  /** Context window used to corroborate ambiguous fallback overflow errors. */
  maxContextTokens?: number;
};

type LLMConfigFor<P extends CustomProviderName> = SharedLLMConfig<P> &
  CustomProviderOptionsMap[P] & {
    /** Optional provider fallbacks in order of attempt */
    fallbacks?: FallbackConfig[];
  };

export type BuiltInLLMConfig = SharedLLMConfig &
  BuiltInClientOptions & {
    /** Optional provider fallbacks in order of attempt */
    fallbacks?: FallbackConfig[];
  };

export type LLMConfig =
  | BuiltInLLMConfig
  | {
      [P in CustomProviderName]: LLMConfigFor<P>;
    }[CustomProviderName]
  | (SharedLLMConfig<RuntimeProviderName> &
      ClientOptions & {
        /** Optional provider fallbacks in order of attempt */
        fallbacks?: FallbackConfig[];
      });

export type ProviderOptionsMap = {
  [Providers.AZURE]: AzureClientOptions;
  [Providers.OPENAI]: OpenAIClientOptions;
  [Providers.GOOGLE]: GoogleClientOptions;
  [Providers.VERTEXAI]: VertexAIClientOptions;
  [Providers.DEEPSEEK]: DeepSeekClientOptions;
  [Providers.ANTHROPIC]: AnthropicClientOptions;
  [Providers.MISTRALAI]: MistralAIClientOptions;
  [Providers.MISTRAL]: MistralAIClientOptions;
  [Providers.OPENROUTER]: ChatOpenRouterCallOptions & StreamSmoothingOptions;
  [Providers.BEDROCK]: BedrockAnthropicClientOptions;
  [Providers.XAI]: XAIClientOptions;
  [Providers.MOONSHOT]: OpenAIClientOptions;
};

export interface ChatModelMap {
  [Providers.XAI]: ChatXAI;
  [Providers.OPENAI]: ChatOpenAI;
  [Providers.AZURE]: AzureChatOpenAI;
  [Providers.DEEPSEEK]: ChatDeepSeek;
  [Providers.VERTEXAI]: ChatVertexAI;
  [Providers.ANTHROPIC]: CustomAnthropic;
  [Providers.MISTRALAI]: ChatMistralAI;
  [Providers.MISTRAL]: ChatMistralAI;
  [Providers.OPENROUTER]: ChatOpenRouter;
  [Providers.BEDROCK]: CustomChatBedrockConverse;
  [Providers.GOOGLE]: CustomChatGoogleGenerativeAI;
  [Providers.MOONSHOT]: ChatMoonshot;
}

export type ProviderOptionsFor<P extends ProviderName> =
  P extends keyof ProviderOptionsMap
    ? ProviderOptionsMap[P]
    : P extends CustomProviderName
      ? CustomProviderOptionsMap[P]
      : BuiltInClientOptions;

export type ProviderModelFor<P extends ProviderName> =
  P extends keyof ChatModelMap
    ? ChatModelMap[P] & BaseChatModel
    : BaseChatModel;

export type ProviderModelConstructor<P extends ProviderName> = new (
  config: ProviderOptionsFor<P>
) => ProviderModelFor<P>;

export type ChatModelConstructorMap = {
  [P in Providers]: new (config: ProviderOptionsMap[P]) => ChatModelMap[P];
};

export type ChatModelInstance = BaseChatModel;

export type ModelWithTools = BaseChatModel & {
  bindTools(tools: CommonToolType[]): Runnable;
};

export type { CustomProviderOptionsMap } from '../provider-registration';

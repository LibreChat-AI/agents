/* eslint-disable @typescript-eslint/ban-ts-comment */
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import {
  FunctionCallingMode,
  GoogleGenerativeAI as GenerativeAI,
} from '@google/generative-ai';
import type {
  GenerateContentRequest,
  SafetySetting,
  ToolConfig,
} from '@google/generative-ai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { GeminiApiUsageMetadata, InputTokenDetails } from './types';
import type { GoogleClientOptions, GoogleThinkingConfig } from '@/types';
import type { NativeMediaPort } from './native';
import {
  convertResponseContentToChatGenerationChunk,
  convertBaseMessagesToContent,
  dropUnsupportedModelTurnPrefill,
  mapGenerateContentResultToChatResult,
} from './utils/common';
import { smoothGenerationChunks } from '@/llm/stream/chunkAdapters';
import { resolveStreamDelay } from '@/llm/stream/smoother';
import { NativeMediaSession } from './native';

type GoogleToolConfigWithServerSideInvocations = ToolConfig & {
  includeServerSideToolInvocations?: boolean;
  functionCallingConfig?: Omit<
    NonNullable<ToolConfig['functionCallingConfig']>,
    'mode'
  > & {
    mode?:
      | NonNullable<ToolConfig['functionCallingConfig']>['mode']
      | 'VALIDATED';
  };
};

export class CustomChatGoogleGenerativeAI extends ChatGoogleGenerativeAI {
  static readonly nativeMediaProtocolVersion = 1;
  nativeMedia?: NativeMediaPort;
  _lc_stream_delay: number;
  thinkingConfig?: GoogleThinkingConfig;
  includeServerSideToolInvocations?: boolean;

  /**
   * Override to add gemini-3 model support for multimodal and function calling thought signatures
   */
  get _isMultimodalModel(): boolean {
    return (
      this.model.startsWith('gemini-1.5') ||
      this.model.startsWith('gemini-2') ||
      (this.model.startsWith('gemma-3-') &&
        !this.model.startsWith('gemma-3-1b')) ||
      this.model.startsWith('gemini-3')
    );
  }

  constructor(fields: GoogleClientOptions) {
    super(fields);
    this.nativeMedia = fields.nativeMedia;

    this._lc_stream_delay = resolveStreamDelay(fields._lc_stream_delay);
    this.model = fields.model.replace(/^models\//, '');

    this.maxOutputTokens = fields.maxOutputTokens ?? this.maxOutputTokens;

    if (this.maxOutputTokens != null && this.maxOutputTokens < 0) {
      throw new Error('`maxOutputTokens` must be a positive integer');
    }

    this.temperature = fields.temperature ?? this.temperature;
    if (
      this.temperature != null &&
      (this.temperature < 0 || this.temperature > 2)
    ) {
      throw new Error('`temperature` must be in the range of [0.0,2.0]');
    }

    this.topP = fields.topP ?? this.topP;
    if (this.topP != null && this.topP < 0) {
      throw new Error('`topP` must be a positive integer');
    }

    if (this.topP != null && this.topP > 1) {
      throw new Error('`topP` must be below 1.');
    }

    this.topK = fields.topK ?? this.topK;
    if (this.topK != null && this.topK < 0) {
      throw new Error('`topK` must be a positive integer');
    }

    this.stopSequences = fields.stopSequences ?? this.stopSequences;

    this.apiKey = fields.apiKey ?? getEnvironmentVariable('GOOGLE_API_KEY');
    if (this.apiKey == null || this.apiKey === '') {
      throw new Error(
        'Please set an API key for Google GenerativeAI ' +
          'in the environment variable GOOGLE_API_KEY ' +
          'or in the `apiKey` field of the ' +
          'ChatGoogleGenerativeAI constructor'
      );
    }

    this.safetySettings = fields.safetySettings ?? this.safetySettings;
    if (this.safetySettings && this.safetySettings.length > 0) {
      const safetySettingsSet = new Set(
        this.safetySettings.map((s) => s.category)
      );
      if (safetySettingsSet.size !== this.safetySettings.length) {
        throw new Error(
          'The categories in `safetySettings` array must be unique'
        );
      }
    }

    this.thinkingConfig = fields.thinkingConfig ?? this.thinkingConfig;
    this.includeServerSideToolInvocations =
      fields.includeServerSideToolInvocations ??
      this.includeServerSideToolInvocations;

    this.streaming = fields.streaming ?? this.streaming;
    this.json = fields.json;

    // @ts-ignore - Accessing private property from parent class
    this.client = new GenerativeAI(this.apiKey).getGenerativeModel(
      {
        model: this.model,
        safetySettings: this.safetySettings as SafetySetting[],
        generationConfig: {
          ...(this.nativeMedia && fields.responseModalities
            ? { responseModalities: fields.responseModalities }
            : {}),
          stopSequences: this.stopSequences,
          maxOutputTokens: this.maxOutputTokens,
          temperature: this.temperature,
          topP: this.topP,
          topK: this.topK,
          ...(this.json != null
            ? { responseMimeType: 'application/json' }
            : {}),
        },
      },
      {
        apiVersion: fields.apiVersion,
        baseUrl: fields.baseUrl,
        customHeaders: fields.customHeaders,
      }
    );
    this.streamUsage = fields.streamUsage ?? this.streamUsage;
  }

  static lc_name(): 'LibreChatGoogleGenerativeAI' {
    return 'LibreChatGoogleGenerativeAI';
  }

  /**
   * Helper function to convert Gemini API usage metadata to LangChain format
   * Includes support for cached tokens and tier-based tracking for gemini-3-pro-preview
   */
  private _convertToUsageMetadata(
    usageMetadata: GeminiApiUsageMetadata | undefined,
    model: string
  ): UsageMetadata | undefined {
    if (!usageMetadata) {
      return undefined;
    }

    const output: UsageMetadata = {
      input_tokens: usageMetadata.promptTokenCount ?? 0,
      output_tokens:
        (usageMetadata.candidatesTokenCount ?? 0) +
        (usageMetadata.thoughtsTokenCount ?? 0),
      total_tokens: usageMetadata.totalTokenCount ?? 0,
    };

    const hasCachedInput = Boolean(usageMetadata.cachedContentTokenCount);
    if (hasCachedInput) {
      output.input_token_details ??= {};
      output.input_token_details.cache_read =
        usageMetadata.cachedContentTokenCount;
    }

    // gemini-3-pro-preview has bracket based tracking of tokens per request
    if (model === 'gemini-3-pro-preview') {
      const over200k = Math.max(
        0,
        (usageMetadata.promptTokenCount ?? 0) - 200000
      );
      const cachedOver200k = Math.max(
        0,
        (usageMetadata.cachedContentTokenCount ?? 0) - 200000
      );
      if (over200k) {
        output.input_token_details = {
          ...output.input_token_details,
          over_200k: over200k,
        } as InputTokenDetails;
      }
      if (cachedOver200k) {
        output.input_token_details = {
          ...output.input_token_details,
          cache_read_over_200k: cachedOver200k,
        } as InputTokenDetails;
      }
    }

    return output;
  }

  invocationParams(
    options?: this['ParsedCallOptions']
  ): Omit<GenerateContentRequest, 'contents'> {
    const params = super.invocationParams(options);
    if (this.thinkingConfig) {
      /** @ts-ignore */
      this.client.generationConfig = {
        /** @ts-ignore */
        ...this.client.generationConfig,
        /** @ts-ignore */
        thinkingConfig: this.thinkingConfig,
      };
    }
    if (
      this.includeServerSideToolInvocations === true &&
      Array.isArray(params.tools) &&
      params.tools.length > 0
    ) {
      const toolConfig = params.toolConfig as
        | GoogleToolConfigWithServerSideInvocations
        | undefined;
      const functionCallingConfig = toolConfig?.functionCallingConfig;
      params.toolConfig = {
        ...toolConfig,
        ...(functionCallingConfig?.mode === FunctionCallingMode.AUTO
          ? {
            functionCallingConfig: {
              ...functionCallingConfig,
              mode: 'VALIDATED',
            },
          }
          : {}),
        includeServerSideToolInvocations: true,
      } as ToolConfig;
    }
    return params;
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<import('@langchain/core/outputs').ChatResult> {
    const native = new NativeMediaSession(
      this.nativeMedia,
      this.model,
      runManager?.runId,
      options.signal
    );
    try {
      const admitted = await native.start();
      if (admitted?.responseModalities) {
        /** @ts-ignore Google types predate responseModalities. */
        this.client.generationConfig.responseModalities =
          admitted.responseModalities;
      }
      const prompt = convertBaseMessagesToContent(
        await native.messages(messages),
        this._isMultimodalModel,
        this.useSystemInstruction,
        this.model
      );
      let actualPrompt = prompt;
      if (prompt?.[0].role === 'system') {
        const [systemInstruction] = prompt;
        /** @ts-ignore */
        this.client.systemInstruction = systemInstruction;
        actualPrompt = prompt.slice(1);
      }
      actualPrompt = dropUnsupportedModelTurnPrefill(actualPrompt, this.model);
      const parameters = this.invocationParams(options);
      const request = {
        ...parameters,
        contents: actualPrompt,
      };

      const res = await this.caller.callWithOptions(
        { signal: options.signal },
        async () =>
          /** @ts-ignore */
          this.client.generateContent(request)
      );

      const response = res.response;
      const usageMetadata = this._convertToUsageMetadata(
        /** @ts-ignore */
        response.usageMetadata,
        this.model
      );

      /** @ts-ignore */
      const generationResult = mapGenerateContentResultToChatResult(response, {
        usageMetadata,
      });

      const generation = generationResult.generations.at(0);
      if (generation != null) {
        generation.message.content = await native.content(
          generation.message.content,
          0
        );
        generation.message.lc_kwargs.content = generation.message.content;
      }
      await native.complete();
      await runManager?.handleLLMNewToken(
        generationResult.generations[0].text || '',
        undefined,
        undefined,
        undefined,
        undefined,
        undefined
      );
      return generationResult;
    } catch (error) {
      await native.fail();
      throw error;
    }
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const native = new NativeMediaSession(
      this.nativeMedia,
      this.model,
      runManager?.runId,
      options.signal
    );
    try {
      const admitted = await native.start();
      if (admitted?.responseModalities) {
        /** @ts-ignore Google types predate responseModalities. */
        this.client.generationConfig.responseModalities =
          admitted.responseModalities;
      }
      yield* smoothGenerationChunks({
        chunks: this._streamProviderChunks(messages, options, native),
        delayMs: this._lc_stream_delay,
        signal: options.signal,
        runManager,
      });
      await native.complete();
    } finally {
      await native.fail();
    }
  }

  private async *_streamProviderChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    native: NativeMediaSession
  ): AsyncGenerator<ChatGenerationChunk> {
    const prompt = convertBaseMessagesToContent(
      await native.messages(messages),
      this._isMultimodalModel,
      this.useSystemInstruction,
      this.model
    );
    let actualPrompt = prompt;
    if (prompt?.[0].role === 'system') {
      const [systemInstruction] = prompt;
      /** @ts-ignore */
      this.client.systemInstruction = systemInstruction;
      actualPrompt = prompt.slice(1);
    }
    actualPrompt = dropUnsupportedModelTurnPrefill(actualPrompt, this.model);
    const parameters = this.invocationParams(options);
    const request = {
      ...parameters,
      contents: actualPrompt,
    };
    const stream = await this.caller.callWithOptions(
      { signal: options.signal },
      async () => {
        /** @ts-ignore */
        const { stream } = await this.client.generateContentStream(request);
        return stream;
      }
    );

    let index = 0;
    let lastUsageMetadata: UsageMetadata | undefined;
    for await (const response of stream) {
      if (
        'usageMetadata' in response &&
        this.streamUsage !== false &&
        options.streamUsage !== false
      ) {
        lastUsageMetadata = this._convertToUsageMetadata(
          response.usageMetadata as GeminiApiUsageMetadata | undefined,
          this.model
        );
      }

      const chunk = convertResponseContentToChatGenerationChunk(response, {
        usageMetadata: undefined,
        index,
      });
      index += 1;
      if (!chunk) {
        continue;
      }

      chunk.message.content = await native.content(
        chunk.message.content,
        index - 1
      );
      chunk.message.lc_kwargs.content = chunk.message.content;
      yield chunk;
    }

    if (lastUsageMetadata) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          usage_metadata: lastUsageMetadata,
        }),
      });
    }
  }
}

import { ChatGoogle } from '@langchain/google-gauth';
import { ChatConnection } from '@langchain/google-common';
import { AIMessageChunk, isAIMessage, isToolMessage } from '@langchain/core/messages';
import type {
  GeminiContent,
  GoogleAIAPI,
  GenerateContentResponseData,
  GeminiResponseCandidate,
  GeminiRequest,
  GoogleAIModelRequestParams,
  GoogleAbstractedClient,
} from '@langchain/google-common';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatResult, ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { GoogleThinkingConfig, VertexAIClientOptions } from '@/types';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  GEMINI_PARTS,
  GEMINI_REPLAY_UNSUPPORTED,
  GEMINI_SIGNATURES,
  captureGeminiReplayParts,
  restoreGeminiReplayParts,
} from '@/llm/google/utils/signatures';
import { dropUnsupportedModelTurnPrefill } from '@/llm/google/utils/common';
import { smoothGenerationChunks } from '@/llm/stream/chunkAdapters';
import { resolveStreamDelay } from '@/llm/stream/smoother';

/**
 * `@langchain/google-common`'s `_streamResponseChunks` emits usage on TWO
 * different paths within the same stream:
 *
 *   - Streaming chunks set `chunk.generationInfo.usage_metadata` via
 *     `responseToUsageMetadata`, which correctly sums
 *     `candidatesTokenCount + thoughtsTokenCount` and includes
 *     `output_token_details.reasoning`.
 *   - The trailing fallback chunk (emitted after the API stream exhausts)
 *     attaches its own `chunk.message.usage_metadata` built inline as
 *     `output_tokens = candidatesTokenCount` only — dropping
 *     `thoughtsTokenCount` and `output_token_details` entirely.
 *
 * After `AIMessageChunk.concat`, only `message.usage_metadata` survives —
 * which is the buggy fallback value. This breaks the documented
 * `total_tokens === input_tokens + output_tokens` invariant and silently
 * undercharges thinking models for reasoning tokens.
 *
 * The repair: track the last `generationInfo.usage_metadata` we see, and
 * when the fallback chunk arrives with its buggy `message.usage_metadata`,
 * replace it with the tracked good value. `CustomChatGoogleGenerativeAI`
 * solves the same problem for the Google API path differently — by
 * overriding `_convertToUsageMetadata`.
 */
export function repairStreamUsageMetadata(
  current: UsageMetadata | undefined,
  generationInfoUsage: UsageMetadata | undefined
): UsageMetadata | undefined {
  if (!current) return current;
  if (!generationInfoUsage) return current;
  if (generationInfoUsage.total_tokens !== current.total_tokens) return current;
  if (generationInfoUsage.output_tokens <= current.output_tokens)
    return current;
  return generationInfoUsage;
}

/**
 * The Gemini API delivers function calls as complete objects — never as
 * partial arg deltas. `@langchain/google-common` pre-parses each streamed
 * functionCall part into `tool_calls` (invalid args land in
 * `invalid_tool_calls` instead), so a chunk whose tool-call chunks all parsed
 * cleanly is sealed on arrival for eager tool execution. Anything that fails
 * the parse check is left unstamped and falls back to the lazy path.
 */
export function sealCompleteStreamedToolCalls(message: AIMessageChunk): void {
  const chunkCount = message.tool_call_chunks?.length ?? 0;
  if (
    chunkCount === 0 ||
    (message.invalid_tool_calls?.length ?? 0) > 0 ||
    (message.tool_calls?.length ?? 0) !== chunkCount
  ) {
    return;
  }
  message.response_metadata = {
    ...message.response_metadata,
    [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
      GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
    [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
  };
}

type AdditionalKwargs =
  | undefined
  | (BaseMessage['additional_kwargs'] & {
      signatures?: Array<string | undefined>;
    });

/** Restores native part order. Legacy text with merged positional slots has lost its boundaries and requires regeneration. */
export function fixThoughtSignatures(
  contents: GeminiContent[],
  input: BaseMessage[]
): void {
  // All AI messages, in order, non-signature ones still consume positional
  // slots so later messages line up with their model content blocks.
  const aiMessages = input.filter(isAIMessage);
  const modelContents = contents.filter((c) => c.role === 'model');

  const count = Math.min(aiMessages.length, modelContents.length);
  for (let i = 0; i < count; i++) {
    const replay = restoreGeminiReplayParts(aiMessages[i]);
    if (replay != null) {
      modelContents[i].parts = replay;
      continue;
    }
    if (Object.hasOwn(aiMessages[i].additional_kwargs, GEMINI_PARTS) && aiMessages[i].additional_kwargs[GEMINI_REPLAY_UNSUPPORTED] !== true) {
      for (const part of modelContents[i].parts) delete part.thoughtSignature;
      continue;
    }
    const signatureMap = aiMessages[i].additional_kwargs[GEMINI_SIGNATURES] as Record<string, string> | undefined;
    if (signatureMap != null) {
      let callIndex = 0;
      for (const part of modelContents[i].parts) {
        if (!('functionCall' in part)) continue;
        const id = aiMessages[i].tool_calls?.[callIndex++]?.id;
        const signature = id != null ? signatureMap[id] : undefined;
        if (typeof signature === 'string' && signature !== '') part.thoughtSignature = signature;
        else delete part.thoughtSignature;
      }
      continue;
    }
    const signatures = (aiMessages[i].additional_kwargs as AdditionalKwargs)
      ?.signatures;
    if (!Array.isArray(signatures) || signatures.length === 0) continue;

    const content = modelContents[i];
    const callCount = aiMessages[i].tool_calls?.length ?? 0;
    const mismatchedSlots = signatures.length !== content.parts.length;
    if (mismatchedSlots && content.parts.some((part) => 'text' in part)) continue;
    if (mismatchedSlots && signatures.slice(callCount).some((signature) => signature != null && signature !== '')) continue;
    for (let index = 0; index < content.parts.length; index++) {
      const part = content.parts[index];
      const signature = signatures[index];
      if ((part.thoughtSignature == null || part.thoughtSignature === '') &&
          typeof signature === 'string' && signature !== '') {
        part.thoughtSignature = signature;
      }
    }
  }
}

type VertexResponseData = Omit<Partial<GenerateContentResponseData>, 'candidates'> & {
  candidates?: Array<Omit<Partial<GeminiResponseCandidate>, 'content'> & {
    content?: GeminiResponseCandidate['content'];
  }>;
};

class CustomChatConnection extends ChatConnection<VertexAIClientOptions> {
  thinkingConfig?: GoogleThinkingConfig;

  buildGeminiAPI(): GoogleAIAPI {
    const api = super.buildGeminiAPI();
    return {
      ...api,
      responseToChatResult: (response): ChatResult => {
        const result = api.responseToChatResult(response);
        const data = response.data as VertexResponseData;
        const message = result.generations[0]?.message;
        if (result.generations.length > 0)
          captureGeminiReplayParts(
            message,
            data.candidates?.[0]?.content?.parts ?? [],
            this.model
          );
        return result;
      },
      responseToChatGeneration: (response): ChatGenerationChunk | null => {
        const data = response.data as VertexResponseData;
        const candidate = data.candidates?.[0];
        if (candidate?.content == null) return api.responseToChatGeneration(response);
        const parts = candidate.content.parts;
        if (parts.length === 0)
          return api.responseToChatGeneration(response);
        let combined: ChatGenerationChunk | undefined;
        let generationInfo: ChatGenerationChunk['generationInfo'];
        for (const part of parts) {
          const chunk = api.responseToChatGeneration({
            data: {
              ...data,
              candidates: [
                {
                  ...candidate,
                  content: { ...candidate.content, parts: [part] },
                },
              ],
            } as GenerateContentResponseData,
          });
          if (chunk != null) {
            generationInfo ??= chunk.generationInfo;
            chunk.generationInfo = undefined;
            combined = combined == null ? chunk : combined.concat(chunk);
          }
        }
        if (combined == null) return null;
        combined.generationInfo = generationInfo;
        captureGeminiReplayParts(combined.message, parts, this.model);
        return combined;
      },
    };
  }

  async formatData(
    input: BaseMessage[],
    parameters: GoogleAIModelRequestParams
  ): Promise<unknown> {
    const formattedData = (await super.formatData(
      input,
      parameters
    )) as GeminiRequest;
    if (formattedData.generationConfig?.thinkingConfig?.thinkingBudget === -1) {
      // -1 means "let the model decide" - delete the property so the API doesn't receive an invalid value
      if (
        formattedData.generationConfig.thinkingConfig.includeThoughts === false
      ) {
        formattedData.generationConfig.thinkingConfig.includeThoughts = true;
      }
      delete formattedData.generationConfig.thinkingConfig.thinkingBudget;
    }
    if (this.thinkingConfig?.thinkingLevel != null) {
      formattedData.generationConfig ??= {};
      // thinkingLevel and thinkingBudget cannot coexist, the API rejects the request.
      // Remove thinkingBudget when thinkingLevel is set.
      const { thinkingBudget: _, ...existingThinkingConfig } =
        (formattedData.generationConfig.thinkingConfig as
          | Record<string, unknown>
          | undefined) ?? {};
      (
        formattedData.generationConfig as Record<string, unknown>
      ).thinkingConfig = {
        ...existingThinkingConfig,
        thinkingLevel: this.thinkingConfig.thinkingLevel,
        ...(this.thinkingConfig.includeThoughts != null && {
          includeThoughts: this.thinkingConfig.includeThoughts,
        }),
      };
    }
    if (formattedData.contents) {
      fixThoughtSignatures(formattedData.contents, input);
      formattedData.contents = dropUnsupportedModelTurnPrefill(
        formattedData.contents,
        this.model
      );
      const functionResponseNames: Array<string | undefined> = [];
      const callNames = new Map<string, string>();
      for (const message of input) {
        if (isAIMessage(message)) {
          callNames.clear();
          for (const call of message.tool_calls ?? []) {
            if (call.id != null) callNames.set(call.id, call.name);
          }
        } else if (isToolMessage(message)) {
          functionResponseNames.push(callNames.get(message.tool_call_id) ?? message.name);
        }
      }
      let responseIndex = 0;
      // gemini-3.1+ models reject role="function"; convert to role="user"
      for (const content of formattedData.contents ?? []) {
        for (const part of content.parts) {
          if (!('functionResponse' in part)) continue;
          const name = functionResponseNames[responseIndex++];
          if (name != null && name !== '') part.functionResponse.name = name;
        }
        if (content.role === 'function') {
          (content as { role: string }).role = 'user';
        }
      }
    }
    return formattedData;
  }
}

/**
 * Integration with Google Vertex AI chat models.
 *
 * Setup:
 * Install `@langchain/google-vertexai` and set your stringified
 * Vertex AI credentials as an environment variable named `GOOGLE_APPLICATION_CREDENTIALS`.
 *
 * ```bash
 * npm install @langchain/google-vertexai
 * export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials"
 * ```
 *
 * ## [Constructor args](https://api.js.langchain.com/classes/_langchain_google_vertexai.index.ChatVertexAI.html#constructor.new_ChatVertexAI)
 *
 * ## [Runtime args](https://api.js.langchain.com/interfaces/langchain_google_common_types.GoogleAIBaseLanguageModelCallOptions.html)
 *
 * Runtime args can be passed as the second argument to any of the base runnable methods `.invoke`. `.stream`, `.batch`, etc.
 * They can also be passed via `.withConfig`, or the second arg in `.bindTools`, like shown in the examples below:
 *
 * ```typescript
 * // When calling `.withConfig`, call options should be passed via the first argument
 * const llmWithArgsBound = llm.withConfig({
 *   stop: ["\n"],
 *   tools: [...],
 * });
 *
 * // When calling `.bindTools`, call options should be passed via the second argument
 * const llmWithTools = llm.bindTools(
 *   [...],
 *   {
 *     tool_choice: "auto",
 *   }
 * );
 * ```
 *
 * ## Examples
 *
 * <details open>
 * <summary><strong>Instantiate</strong></summary>
 *
 * ```typescript
 * import { ChatVertexAI } from '@langchain/google-vertexai';
 *
 * const llm = new ChatVertexAI({
 *   model: "gemini-1.5-pro",
 *   temperature: 0,
 *   // other params...
 * });
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Invoking</strong></summary>
 *
 * ```typescript
 * const input = `Translate "I love programming" into French.`;
 *
 * // Models also accept a list of chat messages or a formatted prompt
 * const result = await llm.invoke(input);
 * console.log(result);
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"J'adore programmer\" \n\nHere's why this is the best translation:\n\n* **J'adore** means \"I love\" and conveys a strong passion.\n* **Programmer** is the French verb for \"to program.\"\n\nThis translation is natural and idiomatic in French. \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 63,
 *     "total_tokens": 72
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Streaming Chunks</strong></summary>
 *
 * ```typescript
 * for await (const chunk of await llm.stream(input)) {
 *   console.log(chunk);
 * }
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "J'adore programmer\" \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "",
 *   "additional_kwargs": {},
 *   "response_metadata": {
 *     "finishReason": "stop"
 *   },
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 8,
 *     "total_tokens": 17
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Aggregate Streamed Chunks</strong></summary>
 *
 * ```typescript
 * import { AIMessageChunk } from '@langchain/core/messages';
 * import { concat } from '@langchain/core/utils/stream';
 *
 * const stream = await llm.stream(input);
 * let full: AIMessageChunk | undefined;
 * for await (const chunk of stream) {
 *   full = !full ? chunk : concat(full, chunk);
 * }
 * console.log(full);
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"J'adore programmer\" \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {
 *     "finishReason": "stop"
 *   },
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 8,
 *     "total_tokens": 17
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Bind tools</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const GetWeather = {
 *   name: "GetWeather",
 *   description: "Get the current weather in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const GetPopulation = {
 *   name: "GetPopulation",
 *   description: "Get the current population in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const llmWithTools = llm.bindTools([GetWeather, GetPopulation]);
 * const aiMsg = await llmWithTools.invoke(
 *   "Which city is hotter today and which is bigger: LA or NY?"
 * );
 * console.log(aiMsg.tool_calls);
 * ```
 *
 * ```txt
 * [
 *   {
 *     name: 'GetPopulation',
 *     args: { location: 'New York City, NY' },
 *     id: '33c1c1f47e2f492799c77d2800a43912',
 *     type: 'tool_call'
 *   }
 * ]
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Structured Output</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const Joke = z.object({
 *   setup: z.string().describe("The setup of the joke"),
 *   punchline: z.string().describe("The punchline to the joke"),
 *   rating: z.number().optional().describe("How funny the joke is, from 1 to 10")
 * }).describe('Joke to tell user.');
 *
 * const structuredLlm = llm.withStructuredOutput(Joke, { name: "Joke" });
 * const jokeResult = await structuredLlm.invoke("Tell me a joke about cats");
 * console.log(jokeResult);
 * ```
 *
 * ```txt
 * {
 *   setup: 'What do you call a cat that loves to bowl?',
 *   punchline: 'An alley cat!'
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Usage Metadata</strong></summary>
 *
 * ```typescript
 * const aiMsgForMetadata = await llm.invoke(input);
 * console.log(aiMsgForMetadata.usage_metadata);
 * ```
 *
 * ```txt
 * { input_tokens: 9, output_tokens: 8, total_tokens: 17 }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Stream Usage Metadata</strong></summary>
 *
 * ```typescript
 * const streamForMetadata = await llm.stream(
 *   input,
 *   {
 *     streamUsage: true
 *   }
 * );
 * let fullForMetadata: AIMessageChunk | undefined;
 * for await (const chunk of streamForMetadata) {
 *   fullForMetadata = !fullForMetadata ? chunk : concat(fullForMetadata, chunk);
 * }
 * console.log(fullForMetadata?.usage_metadata);
 * ```
 *
 * ```txt
 * { input_tokens: 9, output_tokens: 8, total_tokens: 17 }
 * ```
 * </details>
 *
 * <br />
 */
export class ChatVertexAI extends ChatGoogle {
  lc_namespace = ['langchain', 'chat_models', 'vertexai'];
  _lc_stream_delay: number;
  dynamicThinkingBudget = false;
  thinkingConfig?: GoogleThinkingConfig;

  static lc_name(): 'LibreChatVertexAI' {
    return 'LibreChatVertexAI';
  }

  constructor(model: string, fields?: Omit<VertexAIClientOptions, 'model'>);
  constructor(fields?: VertexAIClientOptions);
  constructor(
    modelOrFields?: string | VertexAIClientOptions,
    params?: Omit<VertexAIClientOptions, 'model'>
  ) {
    const fields =
      typeof modelOrFields === 'string'
        ? { ...(params ?? {}), model: modelOrFields }
        : modelOrFields;
    const dynamicThinkingBudget = fields?.thinkingBudget === -1;
    super({
      ...fields,
      platformType: 'gcp',
    });
    this.dynamicThinkingBudget = dynamicThinkingBudget;
    this.thinkingConfig = fields?.thinkingConfig;
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
  }
  invocationParams(
    options?: this['ParsedCallOptions'] | undefined
  ): GoogleAIModelRequestParams {
    const params = super.invocationParams(options);
    if (this.dynamicThinkingBudget) {
      params.maxReasoningTokens = -1;
    }
    return params;
  }
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* smoothGenerationChunks({
      chunks: this._streamRepairedChunks(messages, options),
      delayMs: this._lc_stream_delay,
      signal: options.signal,
      runManager,
    });
  }

  private async *_streamRepairedChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): AsyncGenerator<ChatGenerationChunk> {
    let lastGoodUsage: UsageMetadata | undefined;
    for await (const chunk of super._streamResponseChunks(
      messages,
      options,
      undefined
    )) {
      const genUsage = (
        chunk.generationInfo as { usage_metadata?: UsageMetadata } | undefined
      )?.usage_metadata;
      if (genUsage) {
        lastGoodUsage = genUsage;
      }
      if (chunk.message instanceof AIMessageChunk) {
        const repaired = repairStreamUsageMetadata(
          chunk.message.usage_metadata,
          lastGoodUsage
        );
        if (repaired !== chunk.message.usage_metadata) {
          chunk.message.usage_metadata = repaired;
        }
        sealCompleteStreamedToolCalls(chunk.message);
      }
      yield chunk;
    }
  }
  buildConnection(
    fields: VertexAIClientOptions | undefined,
    client: GoogleAbstractedClient
  ): void {
    // Note: buildConnection is called from super() BEFORE this.thinkingConfig is set,
    // so we must read thinkingConfig from `fields` directly.
    const thinkingConfig = fields?.thinkingConfig ?? this.thinkingConfig;

    const connection = new CustomChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      false
    );
    connection.thinkingConfig = thinkingConfig;
    this.connection = connection;

    const streamedConnection = new CustomChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      true
    );
    streamedConnection.thinkingConfig = thinkingConfig;
    this.streamedConnection = streamedConnection;
  }
}

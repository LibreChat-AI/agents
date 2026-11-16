import { concat } from '@langchain/core/utils/stream';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import {
  AIMessageChunk,
  HumanMessage,
  isAIMessage,
} from '@langchain/core/messages';
import type { ChatGeneration, LLMResult } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatOpenAI } from './index';

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;

class InspectableChatOpenAI extends ChatOpenAI {
  get delegate() {
    return this.completions;
  }
}

class InspectableChatOpenRouter extends ChatOpenRouter {
  get delegate() {
    return this.completions;
  }
}

class MetadataCallback extends BaseCallbackHandler {
  name = 'provider-metadata-test';
  override awaitHandlers = true;
  lc_prefer_streaming = false;
  message?: BaseMessage;

  override handleLLMEnd(output: LLMResult): void {
    const generation = output.generations.at(0)?.at(0);
    if (generation != null && 'message' in generation) {
      this.message = generation.message as BaseMessage;
    }
  }
}

const usage = {
  prompt_tokens: 21,
  completion_tokens: 39,
  total_tokens: 60,
  cost: 0.0000216,
  cost_details: { upstream_inference_cost: 0.0000216 },
};
const base = {
  id: 'gen-provider-metadata',
  object: 'chat.completion.chunk' as const,
  created: 1,
  model: 'test-model',
  provider: 'Google',
};

function rawChunk(finish: boolean, extra: Partial<RawChunk> = {}): RawChunk {
  return {
    ...base,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant', content: finish ? '' : 'Hi' },
        finish_reason: finish ? 'stop' : null,
        logprobs: null,
      },
    ],
    ...extra,
  };
}

function createModel(provider: 'openai' | 'openrouter', streaming = true) {
  const fields = {
    model: base.model,
    apiKey: 'offline-test',
    streaming,
    _lc_stream_delay: 0,
  };
  const model =
    provider === 'openrouter'
      ? new InspectableChatOpenRouter(fields)
      : new InspectableChatOpenAI(fields);
  const client: {
    completionWithRetry(
      request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
    ): Promise<AsyncIterable<RawChunk>>;
  } = model.delegate;
  client.completionWithRetry = async () => ({
    async *[Symbol.asyncIterator]() {
      yield rawChunk(false);
      yield rawChunk(true);
      yield rawChunk(true, { usage });
    },
  });
  // Token estimation otherwise downloads tokenizer data.
  jest.spyOn(model.delegate, 'getNumTokens').mockResolvedValue(1);
  return model;
}

function expectMetadata(message: BaseMessage | undefined): void {
  expect(message?.content).toBe('Hi');
  expect(message != null && isAIMessage(message)).toBe(true);
  if (message == null || !isAIMessage(message)) return;
  expect(message.response_metadata).toMatchObject({
    provider: 'Google',
    usage,
  });
  expect(message.usage_metadata).toMatchObject({
    input_tokens: 21,
    output_tokens: 39,
    total_tokens: 60,
  });
}

afterEach(() => jest.restoreAllMocks());

describe.each(['openai', 'openrouter'] as const)(
  '%s provider metadata',
  (provider) => {
    test.each(['stream', 'invoke', 'callback-stream'] as const)(
      'deduplicates repeated finish snapshots through %s and handleLLMEnd',
      async (entry) => {
        const model = createModel(provider);
        const callback = new MetadataCallback();
        callback.lc_prefer_streaming = entry === 'callback-stream';
        for (let run = 0; run < 2; run++) {
          if (entry !== 'stream') {
            expectMetadata(await model.invoke('hi', { callbacks: [callback] }));
          } else {
            const stream = await model.stream('hi', { callbacks: [callback] });
            let result: AIMessageChunk | undefined;
            for await (const chunk of stream) {
              result = result == null ? chunk : concat(result, chunk);
            }
            expectMetadata(result);
          }
          expectMetadata(callback.message);
        }
      }
    );

    test('deduplicates provider metadata in tool-bound streaming invoke', async () => {
      const model = createModel(provider);
      const bound = model.bindTools([
        {
          type: 'function',
          function: {
            name: 'noop',
            parameters: { type: 'object', properties: {} },
          },
        },
      ]);
      const callback = new MetadataCallback();
      expectMetadata(await bound.invoke('hi', { callbacks: [callback] }));
      expectMetadata(callback.message);
    });

    test('deduplicates each completion independently in streaming generate', async () => {
      const model = createModel(provider);
      const client: {
        completionWithRetry(
          request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
        ): Promise<AsyncIterable<RawChunk>>;
      } = model.delegate;
      client.completionWithRetry = async () => ({
        async *[Symbol.asyncIterator]() {
          for (const index of [0, 1]) {
            for (const finish of [false, true, true]) {
              const chunk = rawChunk(finish);
              chunk.choices[0].index = index;
              yield chunk;
            }
          }
        },
      });
      const result = await model.generate([[new HumanMessage('hi')]]);
      expect(result.generations[0]).toHaveLength(2);
      for (const generation of result.generations[0] as ChatGeneration[]) {
        expect(generation.message.response_metadata).toMatchObject({
          provider: 'Google',
          model_name: base.model,
          finish_reason: 'stop',
        });
      }
    });

    test('preserves non-streaming provider metadata', async () => {
      const model = createModel(provider, false);
      const client: {
        completionWithRetry(
          request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming
        ): Promise<OpenAIClient.Chat.ChatCompletion>;
      } = model.delegate;
      client.completionWithRetry = async () => ({
        ...base,
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'Hi', refusal: null },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage,
      });
      const callback = new MetadataCallback();
      expectMetadata(await model.invoke('hi', { callbacks: [callback] }));
      expectMetadata(callback.message);
    });
  }
);

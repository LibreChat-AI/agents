import { concat } from '@langchain/core/utils/stream';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessageChunk, isAIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { OpenAIClient } from '@langchain/openai';
import { ChatXAI, AzureChatOpenAI } from './index';

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;

type StreamingClient = {
  completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
  ): Promise<AsyncIterable<RawChunk>>;
};

function stubStream(client: StreamingClient, chunks: RawChunk[]): void {
  client.completionWithRetry = async () => ({
    async *[Symbol.asyncIterator]() {
      yield* chunks;
    },
  });
}

class OfflineAzureChatOpenAI extends AzureChatOpenAI {
  setStream(chunks: RawChunk[]): void {
    stubStream(this.completions, chunks);
  }
}

const usage = {
  prompt_tokens: 21,
  completion_tokens: 39,
  total_tokens: 60,
  prompt_tokens_details: { cached_tokens: 7 },
  completion_tokens_details: { reasoning_tokens: 3 },
};

class UsageCallback extends BaseCallbackHandler {
  name = 'stream-usage-test';
  override awaitHandlers = true;
  message?: BaseMessage;

  override handleLLMEnd(output: LLMResult): void {
    const generation = output.generations.at(0)?.at(0);
    if (generation != null && 'message' in generation) {
      this.message = generation.message as BaseMessage;
    }
  }
}

function createModel(
  provider: 'azure' | 'xai',
  usageLocation: 'tail' | 'choice'
): AzureChatOpenAI | ChatXAI {
  const base: Omit<RawChunk, 'choices'> = {
    id: 'chatcmpl-usage',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'gpt-4o',
  };
  const chunks: RawChunk[] = [
    {
      ...base,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant', content: 'Hi' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      ...base,
      choices: [
        {
          index: 0,
          delta: { content: '' },
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
      ...(usageLocation === 'choice' && { usage }),
    },
    ...(usageLocation === 'tail' ? [{ ...base, choices: [], usage }] : []),
  ];
  if (provider === 'xai') {
    const model = new ChatXAI({ apiKey: 'offline-test', _lc_stream_delay: 0 });
    stubStream(model, chunks);
    return model;
  }
  const model = new OfflineAzureChatOpenAI({
    azureOpenAIApiKey: 'offline-test',
    azureOpenAIApiInstanceName: 'offline-instance',
    azureOpenAIApiDeploymentName: 'offline-deployment',
    azureOpenAIApiVersion: '2024-08-01-preview',
    _lc_stream_delay: 0,
  });
  model.setStream(chunks);
  return model;
}

function expectUsage(message: BaseMessage | undefined): void {
  expect(message?.content).toBe('Hi');
  expect(message != null && isAIMessage(message)).toBe(true);
  if (message == null || !isAIMessage(message)) return;
  expect(message.response_metadata.usage).toEqual(usage);
  expect(message.usage_metadata).toMatchObject({
    input_tokens: 21,
    output_tokens: 39,
    total_tokens: 60,
    input_token_details: { cache_read: 7 },
    output_token_details: { reasoning: 3 },
  });
}

describe('Chat Completions usage', () => {
  test.each([
    ['azure', 'tail'],
    ['azure', 'choice'],
    ['xai', 'tail'],
  ] as const)(
    'preserves %s usage on %s through stream concat and handleLLMEnd',
    async (provider, usageLocation) => {
      const model = createModel(provider, usageLocation);
      const callback = new UsageCallback();
      const stream = await model.stream('hi', { callbacks: [callback] });
      let result: AIMessageChunk | undefined;
      for await (const chunk of stream) {
        result = result == null ? chunk : concat(result, chunk);
      }
      expectUsage(result);
      expectUsage(callback.message);
    }
  );
});

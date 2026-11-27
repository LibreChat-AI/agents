import { concat } from '@langchain/core/utils/stream';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessageChunk, isAIMessage } from '@langchain/core/messages';
import type {
  HandleLLMNewTokenCallbackFields,
  NewTokenIndices,
} from '@langchain/core/callbacks/base';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import { AzureChatOpenAI, ChatOpenAI } from './index';

const rawUsage = {
  input_tokens: 100,
  output_tokens: 10,
  total_tokens: 110,
  input_tokens_details: {
    cached_tokens: 20,
    cache_write_tokens: 60,
    cache_write_tokens_details: { ttl_30m_tokens: 15, ttl_12h_tokens: 45 },
  },
  output_tokens_details: { reasoning_tokens: 2 },
};
const expectedUsage = {
  input_tokens: 100,
  output_tokens: 10,
  total_tokens: 110,
  input_token_details: {
    cache_read: 20,
    cache_creation: 60,
    cache_creation_30m: 15,
    cache_creation_12h: 45,
  },
  output_token_details: { reasoning: 2 },
};
const userMetadata = { user: 'kept', nested: { user: 'also kept' } };

class UsageCallback extends BaseCallbackHandler {
  name = 'responses-cache-usage-test';
  override awaitHandlers = true;
  lc_prefer_streaming = false;
  message?: BaseMessage;
  tokenMessages: BaseMessage[] = [];

  override handleLLMEnd(output: LLMResult): void {
    const generation = output.generations.at(0)?.at(0);
    if (generation != null && 'message' in generation) {
      this.message = generation.message as BaseMessage;
    }
  }

  override handleLLMNewToken(
    _token: string,
    _indices: NewTokenIndices,
    _runId: string,
    _parentRunId?: string,
    _tags?: string[],
    fields?: HandleLLMNewTokenCallbackFields
  ): void {
    if (fields?.chunk != null && 'message' in fields.chunk) {
      this.tokenMessages.push(fields.chunk.message);
    }
  }
}

function fixture(
  provider: 'openai' | 'azure',
  streaming: boolean | undefined,
  usage: object | null = rawUsage
) {
  const response = {
    id: 'resp_cache_usage',
    object: 'response',
    model: 'cache-test-model',
    created_at: 0,
    status: 'completed',
    error: null,
    incomplete_details: null,
    metadata: userMetadata,
    output: [
      {
        id: 'msg_cache_usage',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text: 'hi', annotations: [] }],
      },
    ],
    output_text: 'hi',
    usage,
  };
  const fetch = jest.fn<
    ReturnType<typeof globalThis.fetch>,
    Parameters<typeof globalThis.fetch>
  >(async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as { stream?: boolean };
    if (request.stream !== true) {
      return Response.json(response);
    }
    const events = [
      {
        type: 'response.created',
        response: { ...response, usage: null, output: [], metadata: {} },
      },
      {
        type: 'response.output_text.delta',
        delta: 'hi',
        item_id: 'msg_cache_usage',
        output_index: 0,
        content_index: 0,
        logprobs: [],
      },
      { type: 'response.completed', response },
    ];
    return new Response(
      events
        .map(
          (event, sequence_number) =>
            `event: ${event.type}\ndata: ${JSON.stringify({ ...event, sequence_number })}\n\n`
        )
        .join(''),
      { headers: { 'content-type': 'text/event-stream' } }
    );
  });
  const fields = {
    apiKey: 'offline-test',
    model: 'cache-test-model',
    useResponsesApi: true,
    streaming,
    maxRetries: 0,
    configuration: {
      fetch,
      baseURL: 'https://cache-usage.invalid/v1',
      maxRetries: 0,
    },
  };
  if (provider === 'openai') {
    return { model: new ChatOpenAI(fields), fetch };
  }
  return {
    model: new AzureChatOpenAI({
      ...fields,
      azureOpenAIApiKey: 'offline-test',
      azureOpenAIApiInstanceName: 'offline-instance',
      azureOpenAIApiDeploymentName: 'offline-deployment',
      azureOpenAIApiVersion: '2025-04-01-preview',
    }),
    fetch,
  };
}

function expectUsage(message: BaseMessage | undefined): void {
  expect(message).toBeDefined();
  expect(isAIMessage(message!)).toBe(true);
  if (message == null || !isAIMessage(message)) return;
  expect(message.usage_metadata).toEqual(expectedUsage);
  expect(message.response_metadata.metadata).toEqual(userMetadata);
  expect(JSON.stringify(message.response_metadata)).not.toContain(
    '__librechat_cache_write'
  );
}

describe.each(['openai', 'azure'] as const)(
  '%s Responses cache TTL usage invocation',
  (provider) => {
    it.each([false, true])(
      'preserves TTLs on invoke and handleLLMEnd with streaming=%p',
      async (streaming) => {
        const { model, fetch } = fixture(provider, streaming);
        const callback = new UsageCallback();
        const result = await model.invoke('hi', { callbacks: [callback] });
        expectUsage(result);
        expectUsage(callback.message);
        expect(fetch).toHaveBeenCalledTimes(1);
      }
    );

    it('preserves TTLs through host concat, token callbacks, and streamed handleLLMEnd', async () => {
      const { model, fetch } = fixture(provider, undefined);
      const callback = new UsageCallback();
      const stream = await model.stream('hi', { callbacks: [callback] });
      let result: AIMessageChunk | undefined;
      for await (const chunk of stream) {
        result = result == null ? chunk : concat(result, chunk);
      }
      expectUsage(result);
      expectUsage(result?.concat(new AIMessageChunk('')));
      expectUsage(callback.message);
      const usageChunks = callback.tokenMessages.filter(
        (message) => isAIMessage(message) && message.usage_metadata != null
      );
      expect(usageChunks).toHaveLength(1);
      expectUsage(usageChunks[0]);
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('preserves TTLs when a streaming callback bypasses _generate', async () => {
      const { model } = fixture(provider, undefined);
      const callback = new UsageCallback();
      callback.lc_prefer_streaming = true;
      expectUsage(await model.invoke('hi', { callbacks: [callback] }));
      expectUsage(callback.message);
    });

    it('preserves explicit zero TTLs in streamed callbacks and final usage', async () => {
      const { model } = fixture(provider, true, {
        ...rawUsage,
        input_tokens_details: {
          cached_tokens: 20,
          cache_write_tokens: 0,
          cache_write_tokens_details: { ttl_30m_tokens: 0, ttl_12h_tokens: 0 },
        },
      });
      const callback = new UsageCallback();
      const result = await model.invoke('hi', { callbacks: [callback] });
      const usage = {
        ...expectedUsage,
        input_token_details: {
          cache_read: 20,
          cache_creation: 0,
          cache_creation_30m: 0,
          cache_creation_12h: 0,
        },
      };
      expect(result.usage_metadata).toEqual(usage);
      expect(isAIMessage(callback.message!)).toBe(true);
      if (callback.message != null && isAIMessage(callback.message)) {
        expect(callback.message.usage_metadata).toEqual(usage);
      }
    });

    it.each([false, true])(
      'accepts absent optional usage details with streaming=%p',
      async (streaming) => {
        const { model } = fixture(provider, streaming, {
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
        });
        const callback = new UsageCallback();
        const result = await model.invoke('hi', { callbacks: [callback] });
        expect(result.usage_metadata).toMatchObject({
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
        });
        expect(result.usage_metadata?.input_token_details).toEqual({});
        expect(isAIMessage(callback.message!)).toBe(true);
      }
    );

    it.each([false, true])(
      'accepts absent usage with streaming=%p',
      async (streaming) => {
        const { model } = fixture(provider, streaming, null);
        const callback = new UsageCallback();
        const result = await model.invoke('hi', { callbacks: [callback] });
        expect(result.usage_metadata?.input_token_details).not.toHaveProperty(
          'cache_creation_12h'
        );
        expect(callback.message).toBeDefined();
      }
    );
  }
);

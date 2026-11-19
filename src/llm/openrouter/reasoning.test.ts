import { concat } from '@langchain/core/utils/stream';
import { tools as openAITools } from '@langchain/openai';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { OpenAIChatInput } from '@langchain/openai';
import type { OpenRouterReasoning, ChatOpenRouterCallOptions } from './index';
import type { GraphTools } from '@/types';
import { partitionAndMarkOpenRouterToolCache } from './toolCache';
import { addTailCacheControl } from '@/messages/cache';
import { ChatOpenRouter } from './index';

type CreateRouterOptions = Partial<
  ChatOpenRouterCallOptions &
    Pick<OpenAIChatInput, 'model' | 'apiKey' | 'streamUsage'>
>;

type RuntimeInvocationParams = {
  reasoning?: OpenRouterReasoning;
  reasoning_effort?: string;
};

class RuntimeInspectableChatOpenRouter extends ChatOpenRouter {
  getRuntimeInvocationParams(): RuntimeInvocationParams {
    return this.completions.invocationParams() as RuntimeInvocationParams;
  }
}

function createRouter(overrides: CreateRouterOptions = {}): ChatOpenRouter {
  return new ChatOpenRouter({
    model: 'openrouter/test-model',
    apiKey: 'test-key',
    ...overrides,
  });
}

describe('ChatOpenRouter reasoning handling', () => {
  // ---------------------------------------------------------------
  // 1. Constructor reasoning config
  // ---------------------------------------------------------------
  describe('constructor reasoning config', () => {
    it('stores reasoning when passed directly', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });
  });

  // ---------------------------------------------------------------
  // 2. modelKwargs reasoning extraction
  // ---------------------------------------------------------------
  describe('modelKwargs reasoning extraction', () => {
    it('extracts reasoning from modelKwargs and places it into params.reasoning', () => {
      const router = createRouter({
        modelKwargs: { reasoning: { effort: 'medium' } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });

    it('does not leak reasoning into modelKwargs that reach the parent', () => {
      const router = createRouter({
        modelKwargs: {
          reasoning: { effort: 'medium' },
        },
      });
      const params = router.invocationParams();
      // reasoning should be the structured OpenRouter object, not buried in modelKwargs
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });
  });

  // ---------------------------------------------------------------
  // 3. Reasoning merge precedence
  // ---------------------------------------------------------------
  describe('reasoning merge precedence', () => {
    it('constructor reasoning overrides modelKwargs.reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'high' },
        modelKwargs: { reasoning: { effort: 'low' } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });

    it('merges non-overlapping keys from modelKwargs.reasoning and constructor reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'high' },
        modelKwargs: { reasoning: { max_tokens: 5000 } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', max_tokens: 5000 });
    });
  });

  // ---------------------------------------------------------------
  // 4. invocationParams output
  // ---------------------------------------------------------------
  describe('invocationParams output', () => {
    it('includes reasoning object in params', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning).toBeDefined();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });

    it('does NOT include reasoning_effort in params', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning_effort).toBeUndefined();
    });

    it('passes reasoning to the runtime completions delegate', () => {
      const router = new RuntimeInspectableChatOpenRouter({
        model: 'openrouter/test-model',
        apiKey: 'test-key',
        reasoning: { max_tokens: 1024 },
      });
      const params = router.getRuntimeInvocationParams();
      expect(params.reasoning).toEqual({ max_tokens: 1024 });
      expect(params.reasoning_effort).toBeUndefined();
    });

    it('passes legacy include_reasoning to the runtime completions delegate', () => {
      const router = new RuntimeInspectableChatOpenRouter({
        model: 'openrouter/test-model',
        apiKey: 'test-key',
        include_reasoning: true,
      });
      const params = router.getRuntimeInvocationParams();
      expect(params.reasoning).toEqual({ enabled: true });
      expect(params.reasoning_effort).toBeUndefined();
    });

    it('does not include reasoning when none is configured', () => {
      const router = createRouter();
      const params = router.invocationParams();
      expect(params.reasoning).toBeUndefined();
      expect(params.reasoning_effort).toBeUndefined();
    });

    it('preserves streaming extras from parent invocation params', () => {
      const router = createRouter({ streamUsage: true });
      const params = router.invocationParams(undefined, { streaming: true });
      expect(params.stream_options).toEqual({ include_usage: true });
    });
  });

  // ---------------------------------------------------------------
  // 5. Legacy include_reasoning
  // ---------------------------------------------------------------
  describe('legacy include_reasoning', () => {
    it('produces { enabled: true } when only include_reasoning is true', () => {
      const router = createRouter({ include_reasoning: true });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ enabled: true });
    });

    it('does not produce reasoning when include_reasoning is false', () => {
      const router = createRouter({ include_reasoning: false });
      const params = router.invocationParams();
      expect(params.reasoning).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 6. Legacy include_reasoning ignored when reasoning is provided
  // ---------------------------------------------------------------
  describe('legacy include_reasoning ignored when reasoning provided', () => {
    it('reasoning wins over include_reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'medium' },
        include_reasoning: true,
      });
      const params = router.invocationParams();
      // Should use the structured reasoning, NOT fall back to { enabled: true }
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });

    it('reasoning from modelKwargs also wins over include_reasoning', () => {
      const router = createRouter({
        modelKwargs: { reasoning: { effort: 'low' } },
        include_reasoning: true,
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'low' });
    });
  });

  // ---------------------------------------------------------------
  // 7. Various effort levels (OpenRouter-specific)
  // ---------------------------------------------------------------
  describe('various effort levels', () => {
    const efforts: Array<{
      effort: OpenRouterReasoning['effort'];
    }> = [
      { effort: 'xhigh' },
      { effort: 'none' },
      { effort: 'minimal' },
      { effort: 'high' },
      { effort: 'medium' },
      { effort: 'low' },
    ];

    it.each(efforts)('supports effort level "$effort"', ({ effort }) => {
      const router = createRouter({ reasoning: { effort } });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort });
      expect(params.reasoning_effort).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 8. max_tokens reasoning
  // ---------------------------------------------------------------
  describe('max_tokens reasoning', () => {
    it('passes max_tokens in reasoning object', () => {
      const router = createRouter({
        reasoning: { max_tokens: 8000 },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ max_tokens: 8000 });
    });

    it('combines max_tokens with effort', () => {
      const router = createRouter({
        reasoning: { effort: 'high', max_tokens: 8000 },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', max_tokens: 8000 });
      expect(params.reasoning_effort).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 9. exclude reasoning
  // ---------------------------------------------------------------
  describe('exclude reasoning', () => {
    it('passes exclude flag in reasoning object', () => {
      const router = createRouter({
        reasoning: { effort: 'high', exclude: true },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', exclude: true });
    });

    it('supports exclude without effort', () => {
      const router = createRouter({
        reasoning: { exclude: true },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ exclude: true });
    });
  });
});

describe('ChatOpenRouter Responses prompt caching', () => {
  it('maps promptCache to the Responses top-level cache control', () => {
    const router = new ChatOpenRouter({
      model: 'anthropic/claude-sonnet-4',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: true,
      promptCacheTtl: '1h',
    });

    expect(router.invocationParams().cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('sends cache control on the actual Responses request', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: true,
      promptCacheTtl: '1h',
      streaming: false,
    });
    const sentinel = new Error('captured request');
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (request: unknown) => Promise<never>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      throw sentinel;
    };

    await expect(router.invoke([new HumanMessage('hello')])).rejects.toThrow(
      sentinel
    );
    expect(capturedRequest).toMatchObject({
      cache_control: { type: 'ephemeral', ttl: '1h' },
      input: [{ type: 'message', role: 'user', content: 'hello' }],
    });
  });

  it('honors per-call Responses cache overrides', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: false,
      streaming: false,
    });
    const sentinel = new Error('captured override');
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (request: unknown) => Promise<never>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      throw sentinel;
    };

    await expect(
      router.invoke([new HumanMessage('hello')], {
        promptCache: true,
        promptCacheTtl: '5m',
      } as unknown as Parameters<typeof router.invoke>[1])
    ).rejects.toThrow(sentinel);
    expect(capturedRequest).toMatchObject({
      cache_control: { type: 'ephemeral' },
    });
    expect(JSON.stringify(capturedRequest)).not.toContain('"ttl"');
  });

  it('removes tail block markers added after invocation normalization', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: true,
      streaming: false,
    });
    const sentinel = new Error('captured tail cache');
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (request: unknown) => Promise<never>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      throw sentinel;
    };
    const messages = addTailCacheControl([
      new HumanMessage('search'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_cached_result',
            name: 'search',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'result body',
        tool_call_id: 'call_cached_result',
        name: 'search',
      }),
    ]);

    await expect(router.invoke(messages)).rejects.toThrow(sentinel);

    const request = capturedRequest as {
      cache_control?: unknown;
      input?: Array<{ type?: string; output?: unknown }>;
    };
    expect(
      request.input?.find((item) => item.type === 'function_call_output')
    ).toMatchObject({ output: 'result body' });
    expect(request.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(JSON.stringify(request.input)).not.toContain('cache_control');
  });

  it('normalizes cache markers for native Responses event streaming', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: true,
      streaming: true,
    });
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (
            request: unknown
          ) => Promise<AsyncIterable<never>>;
          _streamChatModelEvents: (
            messages: unknown[],
            options: Record<string, unknown>
          ) => AsyncIterable<unknown>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      return {
        async *[Symbol.asyncIterator]() {
          yield* [];
        },
      };
    };
    const messages = addTailCacheControl([
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_stream_cache',
            name: 'search',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'stream result',
        tool_call_id: 'call_stream_cache',
        name: 'search',
      }),
    ]);

    for await (const _event of responses._streamChatModelEvents(messages, {})) {
      // The stubbed provider stream is empty; consuming it triggers request creation.
    }

    const request = capturedRequest as {
      input?: Array<{ type?: string; output?: unknown }>;
      cache_control?: unknown;
    };
    expect(
      request.input?.find((item) => item.type === 'function_call_output')
    ).toMatchObject({ output: 'stream result' });
    expect(request.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(JSON.stringify(request.input)).not.toContain('cache_control');
  });

  it('removes Chat-style tool cache markers from Responses tools', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: true,
      promptCache: true,
      streaming: false,
    });
    const sentinel = new Error('captured tool cache');
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (request: unknown) => Promise<never>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      throw sentinel;
    };
    const markedTools = partitionAndMarkOpenRouterToolCache(
      [
        {
          type: 'function',
          function: {
            name: 'search',
            description: 'Search records',
            parameters: { type: 'object', properties: {} },
          },
        },
      ] as GraphTools,
      () => false,
      '1h'
    )!;
    const bound = router.bindTools(markedTools);

    await expect(bound.invoke([new HumanMessage('search')])).rejects.toThrow(
      sentinel
    );

    const request = capturedRequest as {
      cache_control?: unknown;
      tools?: unknown[];
    };
    expect(request.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(JSON.stringify(request.tools)).not.toContain('cache_control');
  });

  it('keeps top-level cache control when a computer tool auto-selects Responses', async () => {
    const router = new ChatOpenRouter({
      model: 'openai/gpt-4.1',
      apiKey: 'test-key',
      useResponsesApi: false,
      promptCache: true,
      streaming: false,
    });
    const sentinel = new Error('captured computer cache');
    let capturedRequest: unknown;
    const responses = (
      router as unknown as {
        responses: {
          completionWithRetry: (request: unknown) => Promise<never>;
        };
      }
    ).responses;
    responses.completionWithRetry = async (request) => {
      capturedRequest = request;
      throw sentinel;
    };
    const bound = router.bindTools([
      openAITools.computerUse({
        displayWidth: 1024,
        displayHeight: 768,
        environment: 'browser',
        execute: async () => 'data:image/png;base64,AA==',
      }),
    ]);

    await expect(bound.invoke([new HumanMessage('inspect')])).rejects.toThrow(
      sentinel
    );

    expect(capturedRequest).toMatchObject({
      cache_control: { type: 'ephemeral', ttl: '1h' },
    });
  });

  it('keeps Chat prompt caching on content blocks', () => {
    const router = new ChatOpenRouter({
      model: 'anthropic/claude-sonnet-4',
      apiKey: 'test-key',
      useResponsesApi: false,
      promptCache: true,
    });

    expect(router.invocationParams().cache_control).toBeUndefined();
  });

  it('does not double response_metadata.usage and preserves top-level provider when usage arrives on a choice chunk', async () => {
    const router = new ChatOpenRouter({
      model: 'anthropic/claude-haiku-5.5',
      apiKey: 'test-key',
      streaming: true,
    });
    const usage = {
      prompt_tokens: 21,
      completion_tokens: 39,
      total_tokens: 60,
      cost: 0.0000216,
      is_byok: false,
      cost_details: { upstream_inference_cost: 0.0000216 },
    };
    const base = {
      id: 'gen-123',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'anthropic/claude-haiku-5.5',
      provider: 'Google',
    };
    const rawChunks = [
      {
        ...base,
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'Hi' },
            finish_reason: null,
          },
        ],
      },
      {
        ...base,
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: '' },
            finish_reason: 'stop',
          },
        ],
        usage,
      },
    ];
    const completions = (
      router as unknown as {
        completions: {
          completionWithRetry: (
            request: unknown
          ) => Promise<AsyncIterable<unknown>>;
        };
      }
    ).completions;
    completions.completionWithRetry = async () => ({
      async *[Symbol.asyncIterator]() {
        for (const chunk of rawChunks) {
          yield chunk;
        }
      },
    });

    const stream = await router.stream([new HumanMessage('hi')]);
    let result: AIMessageChunk | undefined;
    for await (const chunk of stream) {
      result = result == null ? chunk : concat(result, chunk);
    }
    expect(result?.response_metadata.provider).toBe('Google');
    expect(result?.response_metadata.usage).toEqual(usage);
    expect(result?.usage_metadata).toMatchObject({
      input_tokens: 21,
      output_tokens: 39,
      total_tokens: 60,
    });
  });

  it('preserves ChatOpenRouter class, reasoning params, and metadata dedup after bindTools and withConfig', async () => {
    const router = new ChatOpenRouter({
      model: 'anthropic/claude-haiku-5.5',
      apiKey: 'test-key',
      reasoning: { effort: 'high' },
      _lc_stream_delay: 0,
    });

    const tool = {
      type: 'function' as const,
      function: {
        name: 'noop',
        description: 'noop',
        parameters: { type: 'object', properties: {} },
      },
    };

    const bound = router.bindTools([tool]) as unknown as ChatOpenRouter;
    expect(bound).toBeInstanceOf(ChatOpenRouter);
    expect((bound.constructor as { lc_name?: () => string }).lc_name?.()).toBe(
      'LibreChatOpenRouter'
    );
    expect(bound._streamResponseChunks).toBe(router._streamResponseChunks);
    expect(bound.invocationParams({})).toMatchObject({
      reasoning: { effort: 'high' },
    });

    const configured = bound.withConfig({
      stop: ['END'],
    }) as unknown as ChatOpenRouter;
    expect(configured).toBeInstanceOf(ChatOpenRouter);
    expect(configured.invocationParams({})).toMatchObject({
      reasoning: { effort: 'high' },
      stop: ['END'],
    });

    const completions = (
      configured as unknown as {
        completions: {
          completionWithRetry: (
            request: unknown
          ) => Promise<AsyncIterable<unknown>>;
        };
      }
    ).completions;
    const rawChunks = [
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'anthropic/claude-haiku-5.5',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'Hello ' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'anthropic/claude-haiku-5.5',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'world' },
            finish_reason: 'stop',
          },
        ],
      },
    ];
    completions.completionWithRetry = async () => ({
      async *[Symbol.asyncIterator]() {
        for (const chunk of rawChunks) {
          yield chunk;
        }
      },
    });

    const stream = await configured.stream([new HumanMessage('hi')]);
    let result: Awaited<ReturnType<typeof stream.next>>['value'] | undefined;
    for await (const chunk of stream) {
      result = result == null ? chunk : result.concat(chunk);
    }
    expect(result?.response_metadata.model_name).toBe(
      'anthropic/claude-haiku-5.5'
    );
    expect(result?.response_metadata.finish_reason).toBe('stop');
  });
});

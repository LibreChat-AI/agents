import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect, jest } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import {
  convertMessagesToCompletionsMessageParams,
  convertMessagesToResponsesInput,
  tools as openAITools,
} from '@langchain/openai';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { convertMessageContentToParts } from '@/llm/google/utils/common';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { toLangChainContent } from '@/messages/langchain';
import { Constants, Providers } from '@/common';
import { ToolNode } from '@/tools/ToolNode';
import { ChatOpenAI } from '@/llm/openai';

/**
 * Minimal stub model shape `attemptInvoke` reads. Either `invoke` or
 * `stream` is populated depending on which path the test exercises;
 * extending the real `BaseChatModel` would pull in too much surface.
 */
type StubModel = {
  model?: string;
  _useResponsesApi?: (options?: unknown) => boolean;
  defaultOptions?: unknown;
  last?: unknown;
  invoke?: (messages: BaseMessage[], config?: unknown) => Promise<AIMessage>;
  stream?: (
    messages: BaseMessage[],
    config?: unknown
  ) => AsyncGenerator<AIMessageChunk>;
};

type CapturingModel = {
  invokeMessages: BaseMessage[][];
  model: StubModel;
};

type StreamingCapturingModel = {
  streamMessages: BaseMessage[][];
  model: StubModel;
};

function buildCapturingModel(): CapturingModel {
  const invokeMessages: BaseMessage[][] = [];
  const responseMsg = new AIMessage({ content: 'ok' });
  const model: StubModel = {
    invoke: jest.fn(async (messages: BaseMessage[]): Promise<AIMessage> => {
      invokeMessages.push(messages);
      return responseMsg;
    }),
  };
  return { invokeMessages, model };
}

function buildStreamingCapturingModel(): StreamingCapturingModel {
  const streamMessages: BaseMessage[][] = [];
  const model: StubModel = {
    stream: jest.fn(async function* (
      messages: BaseMessage[]
    ): AsyncGenerator<AIMessageChunk> {
      streamMessages.push(messages);
      yield new AIMessageChunk({ content: 'ok' });
    }),
  };
  return { streamMessages, model };
}

describe('attemptInvoke applies lazy ref annotation', () => {
  it('annotates ToolMessages with live _refKey before sending to provider (non-streaming)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-1', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-1' } }
    );

    expect(invokeMessages).toHaveLength(1);
    const sent = invokeMessages[0];
    expect(sent[1].content).toBe('[ref: tool0turn0]\noutput');

    const original = messages[1] as ToolMessage;
    expect(original.content).toBe('output');
    expect(original.additional_kwargs._refKey).toBe('tool0turn0');
    expect(messages[1]).not.toBe(sent[1]);
  });

  it('annotates messages passed to model.stream (streaming path)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-2', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { streamMessages, model } = buildStreamingCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
        onChunk: () => {
          /* swallow */
        },
      },
      { configurable: { run_id: 'run-2' } }
    );

    expect(streamMessages).toHaveLength(1);
    expect(streamMessages[0][0].content).toBe('[ref: tool0turn0]\noutput');
    expect(messages[0].content).toBe('output');
  });

  it('passes messages unchanged when no registry is exposed on context (e.g. summarization)', async () => {
    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
    });

    expect(invokeMessages).toHaveLength(1);
    expect(invokeMessages[0][0].content).toBe('output');
  });

  it('replaces native computer screenshots for non-Responses providers', async () => {
    const screenshot = `data:image/png;base64,${'A'.repeat(100_000)}`;
    const computerOutput = new ToolMessage({
      content: screenshot,
      tool_call_id: 'computer-output',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages: [computerOutput],
      provider: Providers.ANTHROPIC,
    });

    expect(invokeMessages[0][0].content).toBe(
      '[Computer screenshot omitted for this provider]'
    );
    expect(computerOutput.content).toBe(screenshot);
  });

  it('scopes cache blocks to providers that support them', async () => {
    const cachedOutput = new ToolMessage({
      content: [
        {
          type: 'text',
          text: 'result body',
          cache_control: { type: 'ephemeral' },
        },
      ] as ToolMessage['content'],
      tool_call_id: 'call_cached_tool',
    });
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_cached_tool', name: 'search', args: {} }],
      }),
      cachedOutput,
    ];
    const openRouter = buildCapturingModel();
    const openAI = buildCapturingModel();
    const anthropic = buildCapturingModel();
    const google = buildCapturingModel();

    await attemptInvoke({
      model: openRouter.model as t.ChatModel,
      messages,
      provider: Providers.OPENROUTER,
    });
    await attemptInvoke({
      model: openAI.model as t.ChatModel,
      messages,
      provider: Providers.OPENAI,
    });
    await attemptInvoke({
      model: anthropic.model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
    });
    await attemptInvoke({
      model: google.model as t.ChatModel,
      messages,
      provider: Providers.GOOGLE,
    });

    expect(openRouter.invokeMessages[0][1].content).toEqual(
      cachedOutput.content
    );
    expect(openAI.invokeMessages[0][1].content).toBe('result body');
    expect(anthropic.invokeMessages[0][1].content).toEqual(
      cachedOutput.content
    );
    expect(google.invokeMessages[0][1].content).toBe('result body');
    expect(cachedOutput.content).toEqual([
      {
        type: 'text',
        text: 'result body',
        cache_control: { type: 'ephemeral' },
      },
    ]);
  });

  it('removes foreign cache markers from official OpenAI Chat payloads', async () => {
    const messages = [
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'question',
            cache_control: { type: 'ephemeral' },
          },
        ]),
      }),
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'answer',
            cache_control: { type: 'ephemeral' },
          },
        ]),
      }),
    ];
    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.OPENAI,
    });

    const payload = convertMessagesToCompletionsMessageParams({
      messages: invokeMessages[0],
      model: 'gpt-4o',
    });
    expect(JSON.stringify(payload)).not.toContain('cache_control');
    expect(JSON.stringify(messages)).toContain('cache_control');
  });

  it('removes Bedrock cache points before an Anthropic fallback', async () => {
    const messages = [
      new HumanMessage('search'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_bedrock_cache', name: 'search', args: {} }],
      }),
      new ToolMessage({
        content: [
          { type: 'text', text: 'result body' },
          { cachePoint: { type: 'default' } },
        ] as ToolMessage['content'],
        tool_call_id: 'call_bedrock_cache',
      }),
    ];
    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
    });

    expect((invokeMessages[0][2] as ToolMessage).content).toBe('result body');
    expect(() =>
      _convertMessagesToAnthropicPayload(invokeMessages[0])
    ).not.toThrow();
    expect(JSON.stringify(invokeMessages[0])).not.toContain('cachePoint');
    expect(JSON.stringify(messages)).toContain('cachePoint');
  });

  it('preserves error tool metadata while stripping foreign cache markers', async () => {
    const cachedError = new ToolMessage({
      content: toLangChainContent([
        {
          type: 'text',
          text: 'boom',
          cache_control: { type: 'ephemeral' },
        },
      ]),
      tool_call_id: 'call_error',
      name: 'search',
      status: 'error',
      artifact: { source: 'test' },
      metadata: { trace: 'trace-1' },
    });
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_error', name: 'search', args: {} }],
      }),
      cachedError,
    ];
    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.GOOGLE,
    });

    const sent = invokeMessages[0][1] as ToolMessage;
    expect(sent).toMatchObject({
      content: 'boom',
      status: 'error',
      artifact: { source: 'test' },
      metadata: { trace: 'trace-1' },
    });
    expect(convertMessageContentToParts(sent, false, messages)).toEqual([
      expect.objectContaining({
        functionResponse: expect.objectContaining({
          response: { error: { details: 'boom' } },
        }),
      }),
    ]);
    expect(cachedError.content).not.toBe('boom');
  });

  it('removes per-block cache markers from OpenRouter Responses input', async () => {
    const messages = [
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'question',
            cache_control: { type: 'ephemeral' },
          },
        ]),
      }),
    ];
    const { invokeMessages, model } = buildCapturingModel();
    model.last = { _useResponsesApi: () => true };

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.OPENROUTER,
    });

    const input = convertMessagesToResponsesInput({
      messages: invokeMessages[0],
      model: 'openai/gpt-5',
      zdrEnabled: false,
    });
    expect(JSON.stringify(input)).not.toContain('cache_control');
    expect(JSON.stringify(messages)).toContain('cache_control');
  });

  it.each([
    [Providers.OPENAI, false],
    [Providers.OPENAI, true],
    [Providers.OPENROUTER, false],
    [Providers.OPENROUTER, true],
    [Providers.ANTHROPIC, false],
    [Providers.BEDROCK, false],
    [Providers.GOOGLE, false],
  ])(
    'drops incomplete streamed tool-input fragments for %s (Responses=%s)',
    async (provider, responses) => {
      let typeGetterCalls = 0;
      const accessorBlock = {};
      Object.defineProperty(accessorBlock, 'type', {
        enumerable: true,
        get() {
          typeGetterCalls++;
          throw new Error('type getter must not run');
        },
      });
      const proxyBlock = new Proxy(
        {},
        {
          getOwnPropertyDescriptor() {
            throw new Error('proxy descriptor trap must not run');
          },
        }
      );
      const streamedToolCall = new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'calling' },
          {
            type: 'tool_use',
            id: 'call_streamed',
            name: 'search',
            input: '',
            index: 0,
          },
          {
            type: 'text',
            index: 0,
            input: '{"query":"records"}',
          },
          accessorBlock,
          proxyBlock,
        ]),
        tool_calls: [
          {
            id: 'call_streamed',
            name: 'search',
            args: { query: 'records' },
            type: 'tool_call',
          },
        ],
      });
      const messages = [
        new HumanMessage('run the search'),
        streamedToolCall,
        new ToolMessage({
          content: 'result',
          tool_call_id: 'call_streamed',
          name: 'search',
        }),
      ];
      const { invokeMessages, model } = buildCapturingModel();
      if (responses) {
        model.last = { _useResponsesApi: () => true };
      }

      await attemptInvoke({
        model: model as t.ChatModel,
        messages,
        provider,
      });

      const sent = invokeMessages[0][1] as AIMessage;
      expect(sent.content).toEqual([
        { type: 'text', text: 'calling' },
        {
          type: 'tool_use',
          id: 'call_streamed',
          name: 'search',
          input: '',
          index: 0,
        },
      ]);
      expect(sent.tool_calls).toEqual(streamedToolCall.tool_calls);
      expect(typeGetterCalls).toBe(0);
      expect(streamedToolCall.content).toHaveLength(5);
    }
  );

  it.each([Providers.OPENAI, Providers.OPENROUTER])(
    'canonicalizes computer screenshots for an actual %s Responses attempt',
    async (provider) => {
      const { invokeMessages, model } = buildCapturingModel();
      model.last = { _useResponsesApi: () => true };
      const computerCall = new AIMessage({
        content: '',
        response_metadata: {
          output: [
            {
              type: 'computer_call',
              call_id: 'computer-output',
              action: { type: 'screenshot' },
            },
          ],
        },
      });
      const computerOutput = new ToolMessage({
        content: 'data:image/png;base64,AA==',
        tool_call_id: 'computer-output',
        additional_kwargs: { type: 'computer_call_output' },
      });

      await attemptInvoke({
        model: model as t.ChatModel,
        messages: [computerCall, computerOutput],
        provider,
      });

      expect(invokeMessages[0][1].content).toEqual([
        {
          type: 'input_image',
          image_url: 'data:image/png;base64,AA==',
        },
      ]);
    }
  );

  it('merges per-call options when selecting Chat versus Responses', async () => {
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            call_id: 'call_per_call_computer',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_per_call_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const responses = buildCapturingModel();
    responses.model.defaultOptions = { tools: [] };
    responses.model._useResponsesApi = (options) =>
      Array.isArray((options as { tools?: unknown[] } | undefined)?.tools) &&
      (options as { tools: unknown[] }).tools.length > 0;

    await attemptInvoke(
      {
        model: responses.model as t.ChatModel,
        messages: [computerCall, computerOutput],
        provider: Providers.OPENAI,
      },
      {
        tools: [{ type: 'computer_20251124' }],
      } as unknown as Parameters<typeof attemptInvoke>[1]
    );

    expect(responses.invokeMessages[0][1].content).toEqual([
      {
        type: 'input_image',
        image_url: 'data:image/png;base64,AA==',
      },
    ]);

    const chat = buildCapturingModel();
    chat.model.defaultOptions = {
      tools: [{ type: 'computer_20251124' }],
    };
    chat.model._useResponsesApi = responses.model._useResponsesApi;

    await attemptInvoke(
      {
        model: chat.model as t.ChatModel,
        messages: [computerCall, computerOutput],
        provider: Providers.OPENAI,
      },
      { tools: [] } as unknown as Parameters<typeof attemptInvoke>[1]
    );

    expect(chat.invokeMessages[0][1].content).toBe(
      '[Computer screenshot omitted for this provider]'
    );
  });

  it('detects Responses selected by a bound built-in computer tool', async () => {
    const boundModel = new ChatOpenAI({
      apiKey: 'test-key',
      model: 'computer-use-preview',
      useResponsesApi: false,
    }).bindTools([
      openAITools.computerUse({
        displayWidth: 1024,
        displayHeight: 768,
        environment: 'browser',
        execute: async () => 'data:image/png;base64,AA==',
      }),
    ]);
    const invokeMessages: BaseMessage[][] = [];
    Object.defineProperty(boundModel, 'stream', {
      configurable: true,
      value: undefined,
    });
    Object.defineProperty(boundModel, 'invoke', {
      configurable: true,
      value: jest.fn(async (messages: BaseMessage[]) => {
        invokeMessages.push(messages);
        return new AIMessage({ content: 'ok' });
      }),
    });
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            call_id: 'bound-computer-output',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'bound-computer-output',
      additional_kwargs: { type: 'computer_call_output' },
    });

    await attemptInvoke({
      model: boundModel as t.ChatModel,
      messages: [computerCall, computerOutput],
      provider: Providers.OPENAI,
    });

    expect(invokeMessages[0][1].content).toEqual([
      {
        type: 'input_image',
        image_url: 'data:image/png;base64,AA==',
      },
    ]);
  });

  it('skips annotation for stale _refKey not present in current run registry (cross-run scenario)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    // run-3 registry holds tool0turn0 - the current run's live ref
    registry.set('run-3', 'tool0turn0', 'live-stored');

    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      // Stale ToolMessage from a hydrated prior run - its _refKey points
      // at a key that exists in registry, but conceptually different
      // semantics. For this test, use a key that doesn't exist in the
      // current registry to demonstrate the no-op behavior.
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'old',
        status: 'success',
        content: 'old-output',
        additional_kwargs: { _refKey: 'tool5turn5' },
      }),
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'new',
        status: 'success',
        content: 'new-output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-3' } }
    );

    const sent = invokeMessages[0];
    expect(sent[0].content).toBe('old-output');
    expect(sent[1].content).toBe('[ref: tool0turn0]\nnew-output');
  });

  it('applies unresolved-refs annotation regardless of registry presence', async () => {
    const registry = new ToolOutputReferenceRegistry();
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'error',
        content: 'Error: bad ref',
        additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-err' } }
    );

    expect(invokeMessages[0][0].content).toBe(
      'Error: bad ref\n[unresolved refs: tool9turn9]'
    );
  });

  it('annotates refs registered under an anonymous-batch scope (no run_id)', async () => {
    /**
     * Regression: anonymous ToolNode invocations register refs under
     * a synthetic per-batch scope (`\0anon-<n>`) that
     * `config.configurable.run_id` cannot recover. The transform must
     * read the message-stamped `_refScope` rather than relying on the
     * config-derived runId, otherwise the registry lookup misses and
     * the LLM never sees the `[ref: …]` marker.
     */
    const registry = new ToolOutputReferenceRegistry();
    const anonScope = '\0anon-0';
    registry.set(anonScope, 'tool0turn0', 'stored');

    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: {
          _refKey: 'tool0turn0',
          _refScope: anonScope,
        },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
      context,
    });

    expect(invokeMessages[0][0].content).toBe('[ref: tool0turn0]\noutput');
  });
});

describe('tryFallbackProviders applies the same lazy annotation transform', () => {
  it('threads context through to attemptInvoke so fallback messages are annotated', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-fb', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();
    /**
     * Mock `initializeModel` indirectly by stubbing the LLM init via
     * Jest's manual `mock` so the fallback path returns our capturing
     * model. Skipping this here would require pulling in the real
     * provider init chain (Anthropic, etc.) which the rest of this
     * test layer does not bring in.
     */
    jest.doMock('@/llm/init', () => ({
      initializeModel: (): unknown => model,
    }));

    // Reset the module so the doMock takes effect.
    jest.resetModules();
    const { tryFallbackProviders: freshTry } = (await import(
      '@/llm/invoke'
    )) as { tryFallbackProviders: typeof tryFallbackProviders };

    await freshTry({
      fallbacks: [{ provider: Providers.ANTHROPIC }],
      messages,
      primaryError: new Error('primary failed'),
      context,
      config: { configurable: { run_id: 'run-fb' } },
    });

    expect(invokeMessages.length).toBeGreaterThanOrEqual(1);
    expect(invokeMessages[invokeMessages.length - 1][0].content).toBe(
      '[ref: tool0turn0]\noutput'
    );

    jest.dontMock('@/llm/init');
    jest.resetModules();
  });

  it('stops draining a cancellation-ignoring stream once the breaker trips', async () => {
    const abort = new AbortController();
    /** Statically imported class: neighboring tests reset the module
     * registry, and a dynamically imported instance would fail the
     * statically-linked drain loop's instanceof check. */
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    let yields = 0;
    const model: StubModel = {
      stream: jest.fn(async function* (): AsyncGenerator<AIMessageChunk> {
        for (let i = 0; i < 5; i++) {
          yields += 1;
          yield new AIMessageChunk({ content: `c${i}` });
        }
      }),
    };

    /** The stub ignores the abort signal entirely; only the per-chunk
     * breaker check in the drain loop can stop it. */
    await expect(
      attemptInvoke(
        {
          model: model as t.ChatModel,
          messages: [new HumanMessage('hi')],
          provider: Providers.ANTHROPIC,
          onChunk: () => {
            abort.abort(trip);
          },
        },
        { signal: abort.signal }
      )
    ).rejects.toBe(trip);
    expect(yields).toBeLessThanOrEqual(2);
  });

  it('rejects before invoking a fallback when the breaker tripped during preparation', async () => {
    const abort = new AbortController();
    const { invokeMessages, model } = buildCapturingModel();
    await jest.isolateModulesAsync(async () => {
      jest.doMock('@/llm/init', () => ({
        initializeModel: (): unknown => model,
      }));
      const { tryFallbackProviders: freshTry } = (await import(
        '@/llm/invoke'
      )) as { tryFallbackProviders: typeof tryFallbackProviders };
      const { StreamLimitExceededError: FreshStreamLimitError } = (await import(
        '@/llm/streamLimits'
      )) as typeof import('@/llm/streamLimits');
      const trip = new FreshStreamLimitError({
        kind: 'tool_call_args',
        limit: 10,
        observed: 11,
        toolName: 'db_query',
      });

      /** The sibling's trip lands while fallback preparation is awaited; the
       * fallback must not be invoked at all — a provider that ignores the
       * aborted signal and SUCCEEDS would resolve a run that must reject. */
      await expect(
        freshTry({
          fallbacks: [{ provider: Providers.ANTHROPIC }],
          messages: [
            new ToolMessage({
              name: 'echo',
              tool_call_id: 'tc1',
              status: 'success',
              content: 'output',
            }),
          ],
          primaryError: new Error('primary failed'),
          config: { signal: abort.signal },
          prepareProviderMessages: async ({ messages: preparedMessages }) => {
            abort.abort(trip);
            return preparedMessages;
          },
        })
      ).rejects.toBe(trip);
    });
    expect(invokeMessages).toHaveLength(0);

    jest.dontMock('@/llm/init');
  });

  it('neutralizes preempted Responses history before an OpenAI Chat fallback', async () => {
    const reasoning = {
      id: 'rs_fallback',
      type: 'reasoning',
      status: 'completed',
      summary: [],
      encrypted_content: 'opaque-fallback-reasoning',
    };
    const toolOutput = {
      id: 'ci_fallback',
      type: 'code_interpreter_call',
      status: 'completed',
      code: 'print("fallback result")',
      outputs: [{ type: 'logs', logs: 'fallback result' }],
    };
    const translatedMessage = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: {
        reasoning,
        tool_outputs: [toolOutput],
      },
      response_metadata: { model_provider: 'openai' },
    });
    const message = new AIMessage({
      contentBlocks: [
        ...translatedMessage.contentBlocks,
        {
          type: 'non_standard',
          value: {
            id: 'rs_bare_fallback',
            type: 'reasoning',
            summary: [],
          },
        },
        {
          type: 'image',
          mimeType: 'image/png',
          data: 'AA==',
          id: 'ig_fallback',
          metadata: { status: 'completed' },
        },
      ],
      additional_kwargs: {
        reasoning,
        tool_outputs: [toolOutput],
      },
      response_metadata: {
        model_provider: 'openai',
        output_version: 'v1',
        preempted: true,
      },
    });
    const { invokeMessages, model } = buildCapturingModel();
    model._useResponsesApi = () => false;
    jest.doMock('@/llm/init', () => ({
      initializeModel: (): unknown => model,
    }));
    jest.resetModules();
    const { tryFallbackProviders: freshTry } = (await import(
      '@/llm/invoke'
    )) as { tryFallbackProviders: typeof tryFallbackProviders };

    await freshTry({
      fallbacks: [
        {
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-5.6' },
        },
      ],
      messages: [message],
      primaryError: new Error('primary failed'),
    });

    const sent = invokeMessages[invokeMessages.length - 1][0] as AIMessage;
    expect(sent).not.toBe(message);
    expect(sent.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      { type: 'text', text: expect.stringContaining('fallback result') },
    ]);
    expect(sent.additional_kwargs).toEqual({});
    expect(JSON.stringify(sent.toJSON())).not.toMatch(/rs_|ci_|ig_/);
    const chatPayload = _convertMessagesToOpenAIParams([sent], 'gpt-5.6');
    expect(chatPayload).toEqual([
      {
        role: 'assistant',
        content: sent.content,
      },
    ]);
    expect(JSON.stringify(chatPayload)).not.toMatch(/rs_|ci_|ig_/);
    expect(JSON.stringify(message.toJSON())).toContain('rs_fallback');
    expect(JSON.stringify(message.toJSON())).toContain('ci_fallback');
    expect(JSON.stringify(message.toJSON())).toContain('ig_fallback');

    jest.dontMock('@/llm/init');
    jest.resetModules();
  });
});

describe('invocation attribution metadata', () => {
  it('stamps INVOKED_PROVIDER on the config passed to the model', async () => {
    const capturedConfigs: unknown[] = [];
    const model: StubModel = {
      model: 'claude-opus-4-8',
      invoke: jest.fn(
        async (_m: BaseMessage[], config?: unknown): Promise<AIMessage> => {
          capturedConfigs.push(config);
          return new AIMessage({ content: 'ok' });
        }
      ),
    };

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages: [new HumanMessage('hi')],
        /** A ChatOpenAI-derived provider — `ls_provider` would lie here. */
        provider: Providers.DEEPSEEK,
      },
      {
        configurable: { run_id: 'run-attr' },
        metadata: {
          existing: true,
          endpoint: 'DWAINE',
          rootAgentId: 'sales-copilot',
          activeAgentId: 'dwaine',
        },
      }
    );

    const config = capturedConfigs[0] as {
      metadata?: Record<string, unknown>;
    };
    expect(config.metadata?.[Constants.INVOKED_PROVIDER]).toBe(
      Providers.DEEPSEEK
    );
    expect(config.metadata).toMatchObject({
      endpoint: 'DWAINE',
      model: 'claude-opus-4-8',
      provider: Providers.DEEPSEEK,
      resolvedProvider: Providers.DEEPSEEK,
      rootAgentId: 'sales-copilot',
      activeAgentId: 'dwaine',
    });
    /** Pre-existing metadata is preserved, not replaced. */
    expect(config.metadata?.existing).toBe(true);
  });

  it('stamps INVOKED_MODEL from the fallback clientOptions in tryFallbackProviders', async () => {
    const capturedConfigs: unknown[] = [];
    const model: StubModel = {
      invoke: jest.fn(
        async (_m: BaseMessage[], config?: unknown): Promise<AIMessage> => {
          capturedConfigs.push(config);
          return new AIMessage({ content: 'ok' });
        }
      ),
    };

    jest.doMock('@/llm/init', () => ({
      initializeModel: (): unknown => model,
    }));
    jest.resetModules();
    const { tryFallbackProviders: freshTry } = (await import(
      '@/llm/invoke'
    )) as { tryFallbackProviders: typeof tryFallbackProviders };

    await freshTry({
      fallbacks: [
        {
          provider: Providers.ANTHROPIC,
          clientOptions: { model: 'claude-fallback-1' },
        },
      ],
      messages: [new HumanMessage('hi')],
      primaryError: new Error('primary failed'),
      config: { configurable: { run_id: 'run-attr-fb' } },
    });

    const config = capturedConfigs[0] as {
      metadata?: Record<string, unknown>;
    };
    expect(config.metadata?.[Constants.INVOKED_MODEL]).toBe(
      'claude-fallback-1'
    );
    expect(config.metadata?.[Constants.INVOKED_PROVIDER]).toBe(
      Providers.ANTHROPIC
    );
    expect(config.metadata?.model).toBe('claude-fallback-1');
    expect(config.metadata?.provider).toBe(Providers.ANTHROPIC);
    expect(config.metadata?.resolvedProvider).toBe(Providers.ANTHROPIC);

    jest.dontMock('@/llm/init');
    jest.resetModules();
  });
});

describe('cross-run hydration through ToolNode + attemptInvoke', () => {
  it('annotates run 2 refs but leaves hydrated run 1 ToolMessages untouched', async () => {
    /**
     * Smoke test for the headline scenario: ToolMessages produced in
     * run 1 are persisted with clean content + `_refKey`/`_refScope`
     * metadata. When those messages are hydrated into run 2's state
     * and run 2 produces its own tool output, the annotation transform
     * must (a) annotate run 2's fresh tool message because its
     * `_refScope` is live in run 2's registry, and (b) leave run 1's
     * tool message clean because run 1's scope is not in run 2's
     * registry. Same `tool0turn0` key collides across runs without any
     * confusion.
     */
    const echo = tool(async (input) => (input as { command: string }).command, {
      name: 'echo',
      description: 'echoes its command back',
      schema: z.object({ command: z.string() }),
    }) as unknown as StructuredToolInterface;

    /* Run 1 */
    const run1Node = new ToolNode({
      tools: [echo],
      toolOutputReferences: { enabled: true },
    });
    const run1Result = (await run1Node.invoke(
      {
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'r1c1', name: 'echo', args: { command: 'run-1-output' } },
            ],
          }),
        ],
      },
      { configurable: { run_id: 'run-1' } }
    )) as { messages: ToolMessage[] };

    const run1ToolMsg = run1Result.messages[0];
    expect(run1ToolMsg.content).toBe('run-1-output');
    expect(run1ToolMsg.additional_kwargs._refKey).toBe('tool0turn0');
    expect(run1ToolMsg.additional_kwargs._refScope).toBe('run-1');

    /* Run 2 - fresh ToolNode and registry, simulating a new session */
    const run2Node = new ToolNode({
      tools: [echo],
      toolOutputReferences: { enabled: true },
    });
    const run2Result = (await run2Node.invoke(
      {
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'r2c1', name: 'echo', args: { command: 'run-2-output' } },
            ],
          }),
        ],
      },
      { configurable: { run_id: 'run-2' } }
    )) as { messages: ToolMessage[] };

    const run2ToolMsg = run2Result.messages[0];
    expect(run2ToolMsg.content).toBe('run-2-output');
    expect(run2ToolMsg.additional_kwargs._refKey).toBe('tool0turn0');
    expect(run2ToolMsg.additional_kwargs._refScope).toBe('run-2');

    /* Hydrate run 1's message + run 2's message into a single state */
    const hydrated: BaseMessage[] = [
      new HumanMessage('first request'),
      run1ToolMsg,
      new HumanMessage('second request'),
      run2ToolMsg,
    ];

    /* attemptInvoke with run 2's registry */
    const context = {
      getOrCreateToolOutputRegistry: () =>
        run2Node._unsafeGetToolOutputRegistry(),
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const { invokeMessages, model } = buildCapturingModel();
    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages: hydrated,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-2' } }
    );

    const sent = invokeMessages[0];
    /* Run 1's hydrated tool message stays clean — its scope is stale */
    expect(sent[1].content).toBe('run-1-output');
    /* Run 2's tool message gets annotated — its scope is live */
    expect(sent[3].content).toBe('[ref: tool0turn0]\nrun-2-output');

    /* Persisted state is unchanged */
    expect(hydrated[1].content).toBe('run-1-output');
    expect(hydrated[3].content).toBe('run-2-output');
  });
});

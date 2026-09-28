import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { RunnableLambda } from '@langchain/core/runnables';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { OpenAIChatCompletionChunkChoice } from '@/openai';
import type * as t from '@/types';
import {
  createOpenAIToolCallStream,
  createOpenAIStreamTracker,
  createOpenAIHandlers,
  createChatCompletionChunk,
  sendOpenAIFinalChunk,
} from '@/openai';
import { composeEventHandlers, ModelEndHandler } from '@/events';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import * as init from '@/llm/init';
import { Run } from '@/run';

const toolsReply = (): AIMessageChunk =>
  new AIMessageChunk({
    content: '',
    tool_calls: [
      { id: 'a', name: 'lookup', args: { city: 'Paris' } },
      { id: 'b', name: 'lookup', args: { city: 'Madrid' } },
    ],
  });

/** A deterministic provider boundary. Everything after invoke/stream is the real SDK. */
class InvokeModel implements t.ChatModel {
  calls = 0;
  constructor(readonly response: AIMessageChunk = toolsReply()) {}
  async invoke(
    messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AIMessageChunk> {
    this.calls++;
    return messages.some((message) => message.getType() === 'tool')
      ? new AIMessageChunk('done')
      : this.response;
  }
}

class OneChunkModel extends InvokeModel {
  async stream(): Promise<AsyncIterable<AIMessageChunk>> {
    const response = this.response;
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<AIMessageChunk> {
        yield response;
      },
    };
  }
}

class StreamModel extends InvokeModel {
  async stream(
    messages: BaseMessage[],
    config?: RunnableConfig
  ): Promise<AsyncIterable<AIMessageChunk>> {
    return this.chunks(messages, config);
  }
  async *chunks(
    messages: BaseMessage[],
    config?: RunnableConfig
  ): AsyncGenerator<AIMessageChunk> {
    if (messages.some((message) => message.getType() === 'tool')) {
      yield await this.invoke(messages, config);
      return;
    }
    this.calls++;
    yield new AIMessageChunk({
      content: 'thinking',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          id: 'a',
          index: 0,
          name: 'lookup',
          args: '{"city":',
        },
        {
          type: 'tool_call_chunk',
          id: 'b',
          index: 1,
          name: 'lookup',
          args: '{"city":',
        },
      ],
    });
    yield new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        { type: 'tool_call_chunk', index: 1, args: '"Madrid"}' },
        { type: 'tool_call_chunk', index: 0, args: '"Paris"}' },
      ],
    });
  }
}

class FailedStream extends InvokeModel {
  async stream(): Promise<AsyncIterable<AIMessageChunk>> {
    return this.chunks();
  }
  async *chunks(): AsyncGenerator<AIMessageChunk> {
    this.calls++;
    yield new AIMessageChunk({
      content: 'failed attempt',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          id: 'discard',
          index: 0,
          name: 'lookup',
          args: '{"city":',
        },
      ],
    });
    throw Object.assign(new Error('503 unavailable'), { status: 503 });
  }
}

/** Real LangChain callback lifecycle, with only the remote provider transport replaced. */
class CallbackModel extends FakeListChatModel {
  constructor() {
    super({ responses: [''] });
  }
  async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const message = messages.some((entry) => entry.getType() === 'tool')
      ? new AIMessageChunk('done')
      : toolsReply();
    message.usage_metadata = {
      input_tokens: 5,
      output_tokens: 3,
      total_tokens: 8,
    };
    yield new ChatGenerationChunk({ text: '', message });
    await runManager?.handleLLMNewToken('');
  }
}

async function setup(
  model: t.ChatModel,
  options: {
    fallbacks?: t.FallbackConfig[];
    signal?: AbortSignal;
    observer?: t.EventHandler;
    claimObserver?: t.EventHandler;
    observerBeforeProjection?: boolean;
    failTool?: boolean;
    toolEnd?: boolean;
    noProjection?: boolean;
    clientDelegatedToolNames?: readonly string[];
    usage?: t.EventHandler;
  } = {}
) {
  const tracker = createOpenAIStreamTracker();
  const projected = tracker.toolCalls;
  const writes: string[] = [];
  const wireConfig = {
    tracker,
    context: { requestId: 'parity', model: 'agent', created: 1 },
    writer: {
      write: (frame: string): void => {
        writes.push(frame);
      },
    },
  };
  const frames: OpenAIChatCompletionChunkChoice['delta'][] = [];
  const accepted: t.ModelResponseEvent[] = [];
  const claimed: t.ModelToolsClaimedEvent[] = [];
  const executed: string[] = [];
  // Compose only content handlers. The legacy raw tool handlers must not also
  // publish calls owned by the accepted-result projector.
  const contentHandlers = createOpenAIHandlers(wireConfig);
  const projection = createOpenAIToolCallStream({
    tracker,
    signal: options.signal,
    emit: (delta) => {
      frames.push(delta);
      wireConfig.writer.write(
        `data: ${JSON.stringify(createChatCompletionChunk(wireConfig.context, delta))}\n\n`
      );
    },
  });
  const run = await Run.create<t.IState>({
    runId: 'accepted-projection',
    skipCleanup: true,
    tokenCounter: () => 1,
    clientDelegatedToolNames: options.clientDelegatedToolNames,
    graphConfig: {
      type: 'standard',
      toolEnd: options.toolEnd,
      maxContextTokens: 100_000,
      llmConfig: {
        provider: Providers.OPENAI,
        streaming: true,
        streamUsage: false,
        fallbacks: options.fallbacks,
      },
      tools: [
        tool(
          async ({ city }) => {
            executed.push(city);
            if (options.failTool === true) throw new Error('tool failed');
            return 'sunny';
          },
          {
            name: 'lookup',
            description: 'Weather',
            schema: z.object({ city: z.string() }),
          }
        ),
      ],
    },
    customHandlers:
      options.noProjection === true
        ? {}
        : composeEventHandlers(
          {
            [GraphEvents.ON_MODEL_RESPONSE]: {
              handle: (_event, data): void => {
                if (
                  data != null &&
                    'type' in data &&
                    data.type === 'model_response'
                )
                  accepted.push(data);
              },
            },
          },
          options.observerBeforeProjection === true &&
              options.observer != null
            ? { [GraphEvents.ON_MODEL_RESPONSE]: options.observer }
            : undefined,
          {
            [GraphEvents.ON_MESSAGE_DELTA]:
                contentHandlers[GraphEvents.ON_MESSAGE_DELTA],
            [GraphEvents.ON_REASONING_DELTA]:
                contentHandlers[GraphEvents.ON_REASONING_DELTA],
          },
          options.claimObserver == null
            ? undefined
            : { [GraphEvents.ON_MODEL_TOOLS_CLAIMED]: options.claimObserver },
          projection.handlers,
          {
            [GraphEvents.ON_MODEL_TOOLS_CLAIMED]: {
              handle: (_event, data): void => {
                if (
                  data != null &&
                    'type' in data &&
                    data.type === 'model_tools_claimed'
                )
                  claimed.push(data);
              },
            },
          },
          options.usage != null
            ? { [GraphEvents.CHAT_MODEL_END]: options.usage }
            : undefined,
          options.observerBeforeProjection !== true &&
              options.observer != null
            ? { [GraphEvents.ON_MODEL_RESPONSE]: options.observer }
            : undefined
        ),
  });
  if (!run.Graph) throw new Error('Missing graph');
  run.Graph.overrideModel = model;
  const execute = async (): Promise<void> => {
    try {
      await run.processStream(
        { messages: [new HumanMessage('weather')] },
        {
          configurable: { thread_id: 'parity' },
          version: 'v2',
          signal: options.signal,
        }
      );
      options.signal?.throwIfAborted();
      if (run.getInterrupt() != null || run.getHaltReason() != null) {
        throw new Error('Run did not complete naturally');
      }
      projection.finish();
      await sendOpenAIFinalChunk(wireConfig);
    } catch (error) {
      projection.abort();
      throw error;
    }
  };
  return {
    run,
    execute,
    projection,
    projected,
    frames,
    accepted,
    claimed,
    executed,
    writes,
  };
}

afterEach(() => jest.restoreAllMocks());

describe('accepted tool calls through the real execution boundary', () => {
  it.each([
    new Map([['x', 1]]),
    new Set([1]),
    new Date('2026-01-01'),
    /x/,
    new Uint8Array([1]),
  ])(
    'rejects structured-cloneable non-JSON arguments before tool execution (%s)',
    async (value) => {
      const reply = toolsReply();
      reply.tool_calls![0].args = { city: 'Paris', value };
      const fixture = await setup(new InvokeModel(reply));
      await expect(fixture.execute()).rejects.toThrow('not JSON serializable');
      expect(fixture.executed).toHaveLength(0);
      expect(fixture.frames).toHaveLength(0);
      expect(fixture.writes).toHaveLength(0);
    }
  );

  it('sanitizes clone errors without leaking malformed argument contents', async () => {
    const reply = toolsReply();
    reply.tool_calls![0].args = { credential: () => 'SENSITIVE' };
    const fixture = await setup(new InvokeModel(reply));
    await expect(fixture.execute()).rejects.toThrow(
      'Accepted tool call arguments are not JSON serializable'
    );
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.frames).toHaveLength(0);
  });

  it.each(['invoke', 'stream'] as const)(
    'rejects collection, ID and name accessors before %s run steps',
    async (mode) => {
      for (const field of ['tool_calls', 'id', 'name', 'args'] as const) {
        const reply = toolsReply();
        const getter = jest.fn(() => 'SENSITIVE');
        const target = field === 'tool_calls' ? reply : reply.tool_calls![0];
        Object.defineProperty(target, field, { get: getter });
        const fixture = await setup(
          mode === 'stream' ? new OneChunkModel(reply) : new InvokeModel(reply)
        );
        await expect(fixture.execute()).rejects.toThrow(
          'non-serializable tool calls'
        );
        expect(getter).not.toHaveBeenCalled();
        expect(
          fixture.run.Graph?.getRunSteps().filter(
            (step) => step.type === StepTypes.TOOL_CALLS
          )
        ).toHaveLength(0);
        expect(fixture.executed).toHaveLength(0);
        expect(fixture.accepted).toHaveLength(0);
        expect(fixture.frames).toHaveLength(0);
      }
    }
  );

  it('does not retry invalid descriptors through a fallback provider', async () => {
    const reply = toolsReply();
    const getter = jest.fn(() => 'SENSITIVE');
    Object.defineProperty(reply.tool_calls![0], 'id', { get: getter });
    const fixture = await setup(new InvokeModel(reply), {
      fallbacks: [{ provider: Providers.OPENAI, maxContextTokens: 100_000 }],
    });
    const fallback = jest.spyOn(init, 'initializeModel');
    await expect(fixture.execute()).rejects.toThrow(
      'non-serializable tool calls'
    );
    expect(fallback).not.toHaveBeenCalled();
    expect(getter).not.toHaveBeenCalled();
    expect(
      fixture.run.Graph?.getRunSteps().filter(
        (step) => step.type === StepTypes.TOOL_CALLS
      )
    ).toHaveLength(0);
    expect(fixture.executed).toHaveLength(0);
  });

  it('rejects accessor arguments before clone without executing a stateful getter', async () => {
    const getter = jest.fn(() => 'other');
    const reply = toolsReply();
    Object.defineProperty(reply.tool_calls![0].args, 'city', {
      enumerable: true,
      get: getter,
    });
    const fixture = await setup(new InvokeModel(reply));
    await expect(fixture.execute()).rejects.toThrow('not JSON serializable');
    expect(getter).not.toHaveBeenCalled();
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.writes).toHaveLength(0);
  });

  it('rejects custom instances and hidden keys before clone normalizes them', async () => {
    const reply = toolsReply();
    reply.tool_calls![0].args = {
      city: 'Paris',
      custom: new (class {
        secret = true;
      })(),
    };
    const fixture = await setup(new InvokeModel(reply));
    await expect(fixture.execute()).rejects.toThrow('not JSON serializable');
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.writes).toHaveLength(0);
  });

  it('isolates an earlier mutating observer from the projector and tool execution', async () => {
    const fixture = await setup(new InvokeModel(), {
      observerBeforeProjection: true,
      observer: {
        handle: (_event, data): void => {
          if (
            data != null &&
            'type' in data &&
            data.type === 'model_response'
          ) {
            if (data.toolCalls.length > 1) {
              data.toolCalls[0].args.city = 'changed';
              delete data.toolCalls[1].args.city;
            }
          }
        },
      },
    });
    await fixture.execute();
    expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
    expect(fixture.projected.size).toBe(0);
    expect(fixture.frames.every((delta) => delta.tool_calls == null)).toBe(
      true
    );
  });

  it('does not accept provider/tool custom events as authoritative graph results', async () => {
    class SpoofModel extends InvokeModel {
      async invoke(
        messages: BaseMessage[],
        config?: RunnableConfig
      ): Promise<AIMessageChunk> {
        await dispatchCustomEvent(
          GraphEvents.ON_MODEL_RESPONSE,
          {
            type: 'model_response',
            id: 'spoof',
            agentId: 'default',
            toolCalls: [
              { id: 'spoof', name: 'lookup', args: { city: 'injected' } },
            ],
            toolCallDispositions: ['client'],
            invalidToolCalls: [],
          },
          config
        );
        await dispatchCustomEvent(
          GraphEvents.ON_MODEL_TOOLS_CLAIMED,
          {
            type: 'model_tools_claimed',
            agentId: 'default',
            messageId: 'spoof',
          },
          config
        );
        return super.invoke(messages, config);
      }
    }
    const fixture = await setup(new SpoofModel());
    await fixture.execute();
    expect(fixture.accepted.map((entry) => entry.id)).not.toContain('spoof');
    expect(fixture.claimed).toHaveLength(1);
    expect(fixture.claimed[0].messageId).not.toBe('spoof');
    expect(fixture.projected.size).toBe(0);
  });

  it('rejects malformed finalized results before the tool node can run', async () => {
    const invalid = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          type: 'tool_call_chunk',
          index: 0,
          id: 'bad',
          name: 'lookup',
          args: 'SENSITIVE',
        },
      ],
    });
    const fixture = await setup(new InvokeModel(invalid));
    await expect(fixture.execute()).rejects.toThrow('invalid tool calls');
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.frames).toHaveLength(0);
  });

  it('uses real LangChain callbacks without counting model usage twice', async () => {
    const usage: UsageMetadata[] = [];
    const fixture = await setup(new CallbackModel(), {
      usage: new ModelEndHandler(usage),
    });
    await fixture.execute();
    expect(fixture.projected.size).toBe(0);
    expect(fixture.accepted).toHaveLength(2);
    expect(usage).toHaveLength(2);
    expect(usage.reduce((total, entry) => total + entry.total_tokens, 0)).toBe(
      16
    );
  });

  it('isolates two concurrent real runs even when providers reuse IDs', async () => {
    const a = await setup(new StreamModel());
    const b = await setup(new StreamModel());
    await Promise.all([a.execute(), b.execute()]);
    expect(a.projected.size).toBe(0);
    expect(b.projected.size).toBe(0);
    expect(
      a.accepted
        .map((event) => event.id)
        .some((id) => b.accepted.some((event) => event.id === id))
    ).toBe(false);
  });

  it.each(['stream', 'invoke'])(
    'omits completed parallel %s tool calls from the final answer',
    async (mode) => {
      const fixture = await setup(
        mode === 'stream' ? new StreamModel() : new InvokeModel()
      );
      await fixture.execute();
      // The SDK executed these calls before producing the text answer. Never
      // reconstruct client-facing tool_calls, irrespective of finish_reason.
      expect(fixture.projected.size).toBe(0);
      expect(fixture.frames.every((delta) => delta.tool_calls == null)).toBe(
        true
      );
      expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
      const terminal = fixture.writes
        .slice(0, -1)
        .map((frame) => JSON.parse(frame.slice(6)))
        .find((chunk) => chunk.choices[0]?.finish_reason != null);
      // These tools already ran server-side; the accepted final response is text.
      expect(terminal.choices[0].finish_reason).toBe('stop');
      expect(
        fixture.writes.some((frame) => frame.includes('"content":"done"'))
      ).toBe(true);
      expect(fixture.writes.at(-1)).toBe('data: [DONE]\n\n');
      expect(fixture.accepted).toHaveLength(2); // tools, then the final answer
      expect(new Set(fixture.accepted.map((result) => result.id)).size).toBe(2);
    }
  );

  it.each(['tools', 'text'])(
    'publishes no completed tools after the successful %s fallback',
    async (kind) => {
      const fixture = await setup(new FailedStream(), {
        fallbacks: [
          {
            provider: Providers.OPENAI,
            maxContextTokens: 100_000,
          },
        ],
      });
      const fallback = new InvokeModel(
        kind === 'tools' ? toolsReply() : new AIMessageChunk('no tools needed')
      );
      jest
        .spyOn(init, 'initializeModel')
        .mockReturnValue(
          RunnableLambda.from<BaseMessage[], AIMessageChunk>(
            (messages, config) => fallback.invoke(messages, config)
          )
        );
      await fixture.execute();
      expect(fallback.calls).toBeGreaterThan(0);
      expect(fixture.projected.size).toBe(0);
      expect(
        fixture.accepted
          .flatMap((event) => event.toolCalls)
          .some((call) => call.id === 'discard')
      ).toBe(false);
    }
  );

  it('does not deliver a failed primary result or publish any partial calls', async () => {
    const fixture = await setup(new FailedStream());
    await expect(fixture.execute()).rejects.toThrow('503 unavailable');
    expect(fixture.accepted).toHaveLength(0);
    expect(fixture.frames).toHaveLength(0);
    expect(fixture.projected.size).toBe(0);
  });

  it('keeps host observer mutations out of execution arguments and projected output', async () => {
    const fixture = await setup(new InvokeModel(), {
      observer: {
        handle: (_event, data): void => {
          if (
            data != null &&
            'type' in data &&
            data.type === 'model_response'
          ) {
            for (const call of data.toolCalls) call.args.city = 'tampered';
          }
        },
      },
    });
    await fixture.execute();
    expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
    expect(fixture.projected.size).toBe(0);
  });

  it('awaits the accepted-result observer and propagates rejection before tool execution', async () => {
    let seen = false;
    const fixture = await setup(new InvokeModel(), {
      fallbacks: [{ provider: Providers.OPENAI, maxContextTokens: 100_000 }],
      observer: {
        handle: async (): Promise<void> => {
          await Promise.resolve();
          seen = true;
          throw new Error('projection failed');
        },
      },
    });
    const fallback = jest.spyOn(init, 'initializeModel');
    await expect(fixture.execute()).rejects.toThrow('projection failed');
    expect(seen).toBe(true);
    expect(fallback).not.toHaveBeenCalled();
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.frames).toHaveLength(0);
  });

  it('honors cancellation from the accepted response before the tool node runs', async () => {
    const controller = new AbortController();
    const fixture = await setup(new InvokeModel(), {
      signal: controller.signal,
      observer: {
        handle: (): void => {
          controller.abort(new Error('cancelled'));
        },
      },
    });
    await expect(fixture.execute()).rejects.toThrow();
    expect(fixture.frames).toHaveLength(0);
    expect(fixture.executed).toHaveLength(0);
  });
});

describe('accepted tool ownership and compatibility', () => {
  it.each(['invoke', 'stream'] as const)(
    'never resends graph-owned tools after a terminal %s tool node',
    async (mode) => {
      const fixture = await setup(
        mode === 'stream' ? new StreamModel() : new InvokeModel(),
        { toolEnd: true }
      );
      await fixture.execute();
      expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
      expect(fixture.accepted).toHaveLength(1);
      expect(fixture.projected.size).toBe(0);
      expect(fixture.frames.every((delta) => delta.tool_calls == null)).toBe(
        true
      );
      const choices = fixture.writes
        .filter((frame) => frame.startsWith('data: {'))
        .flatMap((frame) => JSON.parse(frame.slice(6)).choices);
      expect(
        choices.find((choice) => choice.finish_reason != null).finish_reason
      ).toBe('stop');
    }
  );

  it.each(['depth', 'bytes'] as const)(
    'does not apply projection %s limits to an ordinary run',
    async (limit) => {
      let tree: unknown = 'leaf';
      if (limit === 'depth') {
        for (let i = 0; i < 65; i++) tree = { child: tree };
      } else {
        tree = 'x'.repeat(4 * 1024 * 1024);
      }
      const chunk = new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'ordinary', name: 'lookup', args: { city: 'Paris', tree } },
        ],
      });
      const fixture = await setup(new InvokeModel(chunk), {
        noProjection: true,
      });
      await fixture.execute();
      expect(fixture.accepted).toHaveLength(0);
      expect(fixture.executed).toEqual(['Paris']);
    }
  );
});

describe('provider tool fragment descriptor boundary', () => {
  it.each(['iterator', 'callback'] as const)(
    'rejects raw accessors before %s accounting or dispatch',
    async (delivery) => {
      for (const field of [
        'tool_call_chunks',
        'id',
        'name',
        'args',
        'index',
      ] as const) {
        const chunk = new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              index: 0,
              id: 'a',
              name: 'lookup',
              args: '{"city":"Paris"}',
              type: 'tool_call_chunk',
            },
          ],
        });
        const getter = jest.fn(() => {
          throw new Error('DO NOT READ');
        });
        Object.defineProperty(
          field === 'tool_call_chunks' ? chunk : chunk.tool_call_chunks![0],
          field,
          { get: getter }
        );
        const fixture = await setup(new OneChunkModel(chunk));
        if (delivery === 'callback' && fixture.run.Graph != null)
          fixture.run.Graph.config = {};
        const invoke =
          delivery === 'iterator'
            ? fixture.execute()
            : new ChatModelStreamHandler().handle(
              GraphEvents.CHAT_MODEL_STREAM,
              { chunk },
              {},
              fixture.run.Graph
            );
        await expect(invoke).rejects.toThrow('non-serializable tool calls');
        expect(getter).not.toHaveBeenCalled();
        expect(fixture.executed).toHaveLength(0);
        expect(fixture.accepted).toHaveLength(0);
        expect(
          fixture.run.Graph?.getRunSteps().filter(
            (step) => step.type === StepTypes.TOOL_CALLS
          )
        ).toHaveLength(0);
      }
    }
  );
});

describe('tool ownership observation', () => {
  it('isolates a mutating claim observer from the projector in a terminal tool run', async () => {
    const fixture = await setup(new InvokeModel(), {
      toolEnd: true,
      claimObserver: {
        handle: (_event, data): void => {
          if (
            data != null &&
            'type' in data &&
            data.type === 'model_tools_claimed'
          ) {
            data.agentId = 'unrelated';
            data.messageId = 'unrelated';
          }
        },
      },
    });
    await fixture.execute();
    expect(fixture.claimed).toHaveLength(1);
    expect(fixture.claimed[0].messageId).toBe(fixture.accepted[0].messageId);
    expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
    expect(fixture.projected.size).toBe(0);
  });

  it('awaits claim observers and never executes a rejected batch or retries the model', async () => {
    const model = new InvokeModel();
    const fixture = await setup(model, {
      claimObserver: {
        handle: async (): Promise<void> => {
          await Promise.resolve();
          throw new Error('claim observer failed');
        },
      },
    });
    await expect(fixture.execute()).rejects.toThrow('claim observer failed');
    expect(model.calls).toBe(1);
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.projected.size).toBe(0);
    expect(fixture.frames).toHaveLength(0);
  });
});

describe('reused provider chunk compatibility', () => {
  it('executes the complete arguments when a provider reuses and mutates the same fragment record', async () => {
    class ReusedModel extends InvokeModel {
      async stream(
        messages: BaseMessage[]
      ): Promise<AsyncIterable<AIMessageChunk>> {
        const last = messages.some((message) => message.getType() === 'tool');
        const record = {
          index: 0,
          id: 'reuse',
          name: 'lookup',
          args: '{"city":"Pa',
          type: 'tool_call_chunk' as const,
        };
        const chunk = new AIMessageChunk({
          content: '',
          tool_call_chunks: [record],
        });
        return {
          async *[Symbol.asyncIterator](): AsyncGenerator<AIMessageChunk> {
            if (last) {
              yield new AIMessageChunk('done');
              return;
            }
            yield chunk;
            record.args = 'ris"}';
            record.id = '';
            record.name = '';
            chunk.tool_calls = [];
            chunk.invalid_tool_calls = [];
            yield chunk;
          },
        };
      }
    }
    const fixture = await setup(new ReusedModel());
    await fixture.execute();
    expect(fixture.executed).toEqual(['Paris']);
    expect(fixture.projected.size).toBe(0);
  });
});

describe('graph ownership across provider and invoke boundaries', () => {
  it('never delegates externally completed calls when the router bypasses ToolNode', async () => {
    const reply = new AIMessageChunk({
      id: 'server-answer',
      content: '',
      tool_calls: [
        { id: 'srvtoolu_1', name: 'web_search', args: { q: 'weather' } },
      ],
    });
    class ServerExecutedModel extends InvokeModel {
      async invoke(
        messages: BaseMessage[],
        config?: RunnableConfig
      ): Promise<AIMessageChunk> {
        const result = await super.invoke(messages, config);
        if (result === reply)
          fixture.run.Graph!.invokedToolIds = new Set(['srvtoolu_1']);
        return result;
      }
    }
    const fixture = await setup(new ServerExecutedModel(reply));
    await fixture.execute();
    expect(fixture.accepted).toHaveLength(1);
    expect(fixture.claimed).toHaveLength(0); // no ToolNode on this path
    expect(fixture.projected.size).toBe(0);
    expect(fixture.frames.every((delta) => delta.tool_calls == null)).toBe(
      true
    );
  });

  it.each(['invoke', 'stream'] as const)(
    'emits only an explicitly delegated terminal %s tool call',
    async (mode) => {
      const reply = new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'for-client', name: 'lookup', args: { city: 'Paris' } },
        ],
      });
      const fixture = await setup(
        mode === 'stream' ? new OneChunkModel(reply) : new InvokeModel(reply),
        { clientDelegatedToolNames: ['lookup'] }
      );
      await fixture.execute();
      expect(fixture.executed).toHaveLength(0);
      expect(fixture.accepted[0].toolCallDispositions).toEqual(['client']);
      expect([...fixture.projected.values()].map((call) => call.id)).toEqual([
        'for-client',
      ]);
      expect(fixture.frames.some((delta) => delta.tool_calls != null)).toBe(
        true
      );
      const finish = fixture.writes
        .filter((frame) => frame.startsWith('data: {'))
        .flatMap((frame) => JSON.parse(frame.slice(6)).choices)
        .find((choice) => choice.finish_reason != null);
      expect(finish.finish_reason).toBe('tool_calls');
    }
  );

  it('fails a mixed client/SDK batch before either side executes', async () => {
    const reply = new AIMessageChunk({
      content: '',
      tool_calls: [
        { id: 'for-client', name: 'lookup', args: { city: 'Paris' } },
        { id: 'other', name: 'unregistered', args: {} },
      ],
    });
    const fixture = await setup(new InvokeModel(reply), {
      clientDelegatedToolNames: ['lookup'],
    });
    await expect(fixture.execute()).rejects.toThrow(
      'Mixed client and graph-owned tool calls'
    );
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.projected.size).toBe(0);
  });

  it('keeps a provider-owned server call and an SDK tool call off the client wire', async () => {
    const reply = new AIMessageChunk({
      content: '',
      tool_calls: [
        { id: 'srvtoolu_2', name: 'web_search', args: { q: 'weather' } },
        { id: 'local', name: 'lookup', args: { city: 'Paris' } },
      ],
    });
    class MixedModel extends InvokeModel {
      async invoke(
        messages: BaseMessage[],
        config?: RunnableConfig
      ): Promise<AIMessageChunk> {
        const result = await super.invoke(messages, config);
        if (result === reply)
          fixture.run.Graph!.invokedToolIds = new Set(['srvtoolu_2']);
        return result;
      }
    }
    const fixture = await setup(new MixedModel(reply), { toolEnd: true });
    await fixture.execute();
    expect(fixture.accepted[0].toolCallDispositions).toEqual([
      'provider',
      'sdk',
    ]);
    expect(fixture.executed).toEqual(['Paris']);
    expect(fixture.projected.size).toBe(0);
    expect(fixture.frames.every((delta) => delta.tool_calls == null)).toBe(
      true
    );
  });

  it('assigns SDK message identity without mutating a frozen model response or its kwargs', async () => {
    const reply = new AIMessageChunk('hello');
    Object.freeze(reply.lc_kwargs);
    Object.freeze(reply);
    const fixture = await setup(new InvokeModel(reply));
    await fixture.execute();
    expect(reply.id).toBeUndefined();
    expect(fixture.accepted[0].messageId).toBeTruthy();
    expect(fixture.projected.size).toBe(0);
  });

  it.each([true, false])(
    'accepts immutable invoke responses with tool calls=%s without changing provider data',
    async (withTools) => {
      const reply = new AIMessageChunk(
        withTools
          ? {
            id: 'immutable',
            content: '',
            tool_calls: [
              { id: 'a', name: 'lookup', args: { city: 'Paris' } },
            ],
          }
          : { id: 'immutable', content: 'hello' }
      );
      const originalToolCalls = reply.tool_calls;
      Object.freeze(reply);
      const fixture = await setup(new InvokeModel(reply));
      await fixture.execute();
      expect(reply.tool_calls).toBe(originalToolCalls);
      expect(fixture.executed).toEqual(withTools ? ['Paris'] : []);
      expect(fixture.projected.size).toBe(0);
    }
  );
});

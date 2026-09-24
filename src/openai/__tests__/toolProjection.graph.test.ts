import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import { RunnableLambda } from '@langchain/core/runnables';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { OpenAIToolCall, OpenAIChatCompletionChunkChoice } from '@/openai';
import type * as t from '@/types';
import { createOpenAIToolCallStream } from '@/openai';
import { composeEventHandlers, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
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
    failTool?: boolean;
    usage?: t.EventHandler;
  } = {}
) {
  const projected = new Map<number, OpenAIToolCall>();
  const frames: OpenAIChatCompletionChunkChoice['delta'][] = [];
  const accepted: t.ModelResponseEvent[] = [];
  const executed: string[] = [];
  const projection = createOpenAIToolCallStream({
    toolCalls: projected,
    signal: options.signal,
    emit: (delta) => {
      frames.push(delta);
    },
  });
  const run = await Run.create<t.IState>({
    runId: 'accepted-projection',
    skipCleanup: true,
    tokenCounter: () => 1,
    graphConfig: {
      type: 'standard',
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
    customHandlers: composeEventHandlers(
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
      projection.handlers,
      options.usage != null
        ? { [GraphEvents.CHAT_MODEL_END]: options.usage }
        : undefined,
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
    } catch (error) {
      projection.abort();
      throw error;
    }
  };
  return { run, execute, projection, projected, frames, accepted, executed };
}

afterEach(() => jest.restoreAllMocks());

describe('accepted tool calls through the real execution boundary', () => {
  it('sanitizes clone errors without leaking malformed argument contents', async () => {
    const reply = toolsReply();
    reply.tool_calls![0].args = { credential: () => 'SENSITIVE' };
    const fixture = await setup(new InvokeModel(reply));
    await expect(fixture.execute()).rejects.toThrow(
      'Accepted model response contains non-serializable tool calls'
    );
    expect(fixture.executed).toHaveLength(0);
    expect(fixture.frames).toHaveLength(0);
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
            invalidToolCalls: [],
          },
          config
        );
        return super.invoke(messages, config);
      }
    }
    const fixture = await setup(new SpoofModel());
    await fixture.execute();
    expect(fixture.accepted.map((entry) => entry.id)).not.toContain('spoof');
    expect([...fixture.projected.values()].map((call) => call.id)).toEqual([
      'a',
      'b',
    ]);
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
    expect(fixture.projected.size).toBe(2);
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
    expect(a.projected.size).toBe(2);
    expect(b.projected.size).toBe(2);
    expect(
      a.accepted
        .map((event) => event.id)
        .some((id) => b.accepted.some((event) => event.id === id))
    ).toBe(false);
  });

  it.each(['stream', 'invoke'])(
    'projects the final parallel calls through %s without manual event metadata',
    async (mode) => {
      const fixture = await setup(
        mode === 'stream' ? new StreamModel() : new InvokeModel()
      );
      await fixture.execute();
      expect([...fixture.projected.values()]).toEqual([
        {
          id: 'a',
          type: 'function',
          function: { name: 'lookup', arguments: '{"city":"Paris"}' },
        },
        {
          id: 'b',
          type: 'function',
          function: { name: 'lookup', arguments: '{"city":"Madrid"}' },
        },
      ]);
      expect(fixture.executed.sort()).toEqual(['Madrid', 'Paris']);
      expect(fixture.accepted).toHaveLength(2); // tools, then the final answer
      expect(new Set(fixture.accepted.map((result) => result.id)).size).toBe(2);
    }
  );

  it.each(['tools', 'text'])(
    'projects only the successful %s fallback after a partial primary stream',
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
      expect([...fixture.projected.values()].map((call) => call.id)).toEqual(
        kind === 'tools' ? ['a', 'b'] : []
      );
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
    expect(fixture.projected.get(0)?.function.arguments).toBe(
      '{"city":"Paris"}'
    );
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

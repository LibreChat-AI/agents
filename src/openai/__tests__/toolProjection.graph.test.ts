import { AIMessageChunk } from '@langchain/core/messages';
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type { OpenAIChatCompletionChunkChoice, OpenAIToolCall } from '@/openai';
import { createOpenAIToolCallStream } from '@/openai';
import { STREAM_LIMIT_ATTEMPT_KEY } from '@/llm/streamLimits';
import { ChatModelStreamHandler } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { HandlerRegistry } from '@/events';
import { StandardGraph } from '@/graphs';

function setup() {
  const toolCalls = new Map<number, OpenAIToolCall>();
  const emitted: OpenAIChatCompletionChunkChoice['delta'][] = [];
  const stream = createOpenAIToolCallStream({
    toolCalls,
    emit: (delta) => {
      emitted.push(delta);
    },
  });
  const graph = new StandardGraph({
    runId: 'parity',
    agents: ['agent', 'a', 'b'].map((agentId) => ({
      agentId,
      provider: Providers.OPENAI,
      tools: [],
    })),
  });
  graph.config = { configurable: { run_id: 'parity', thread_id: 'thread' } };
  graph.handlerRegistry = new HandlerRegistry();
  graph.handlerRegistry.register(GraphEvents.ON_RUN_STEP, {
    handle: (_event, data, metadata): void => {
      if (data != null && 'stepDetails' in data)
        stream.onRunStep(data, metadata, graph);
    },
  });
  graph.handlerRegistry.register(GraphEvents.ON_RUN_STEP_DELTA, {
    handle: (_event, data, metadata): void => {
      if (
        data != null &&
        'delta' in data &&
        'id' in data &&
        'type' in data.delta &&
        data.delta.type === 'tool_calls'
      ) {
        stream.onRunStepDelta(
          {
            id: data.id,
            delta: { type: data.delta.type, tool_calls: data.delta.tool_calls },
          },
          metadata,
          graph
        );
      }
    },
  });
  const producer = new ChatModelStreamHandler();
  const metadata = { langgraph_node: 'agent=agent', langgraph_step: 1 };
  const send = async (
    fragment: ToolCallChunk,
    extra: Record<string, unknown> = {}
  ): Promise<void> => {
    stream.observeModelAttempt({ ...metadata, ...extra }, graph);
    await producer.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: new AIMessageChunk({
          content: '',
          tool_call_chunks: [{ ...fragment, type: 'tool_call_chunk' }],
        }),
      },
      { ...metadata, ...extra },
      graph
    );
  };
  return { stream, toolCalls, emitted, graph, metadata, send };
}

describe('tool projection through real SDK stream handlers', () => {
  it('reconciles cumulative argument snapshots and repeated IDs/names', async () => {
    const { stream, toolCalls, emitted, send } = setup();
    for (const args of [
      '{"city":"N',
      '{"city":"N',
      '{"city":"NYC"}',
      '{"city":"NYC"}',
    ]) {
      await send({ index: 0, id: 'call_weather', name: 'weather', args });
    }
    expect(emitted).toHaveLength(0);
    stream.finish();
    expect([...toolCalls.values()]).toEqual([
      {
        id: 'call_weather',
        type: 'function',
        function: { name: 'weather', arguments: '{"city":"NYC"}' },
      },
    ]);
  });

  it('keeps repeated incremental characters rather than deduplicating them', async () => {
    const { stream, toolCalls, send } = setup();
    await send({ index: 0, id: 'call', name: 'lookup', args: '{"word":"' });
    for (const args of ['a', 'a', 'a', '"}']) await send({ index: 0, args });
    stream.finish();
    expect(toolCalls.get(0)?.function.arguments).toBe('{"word":"aaa"}');
  });

  it('reuses invocation identity when handleToolCalls declares a later parallel call', async () => {
    const { stream, toolCalls, emitted, send } = setup();
    await send({ index: 0, id: 'a', name: 'lookup', args: '' });
    await send({ index: 1, id: 'b', name: 'lookup', args: '' });
    await send({ index: 0, args: '{"city":"Madrid"}' });
    await send({ index: 1, id: 'b', name: 'lookup', args: '{"city":"Paris"}' });
    stream.finish();
    expect(
      [...toolCalls.values()].map((call) => [call.id, call.function.arguments])
    ).toEqual([
      ['a', '{"city":"Madrid"}'],
      ['b', '{"city":"Paris"}'],
    ]);
    expect(emitted).toHaveLength(4);
  });

  it('replaces a failed attempt and drops its late chunks without deleting other invocations', async () => {
    const { stream, toolCalls, send } = setup();
    await send(
      { index: 0, id: 'kept', name: 'lookup', args: '{}' },
      { langgraph_step: 0, [STREAM_LIMIT_ATTEMPT_KEY]: 9 }
    );
    await send(
      { index: 0, id: 'primary', name: 'lookup', args: '{"city":' },
      { [STREAM_LIMIT_ATTEMPT_KEY]: 10 }
    );
    await send(
      { index: 0, id: 'fallback', name: 'lookup', args: '{"city":"Paris"}' },
      { [STREAM_LIMIT_ATTEMPT_KEY]: 11 }
    );
    await send(
      { index: 0, args: '"NYC"}' },
      { [STREAM_LIMIT_ATTEMPT_KEY]: 10 }
    );
    stream.finish();
    expect([...toolCalls.values()].map((call) => call.id)).toEqual([
      'kept',
      'fallback',
    ]);
    expect(toolCalls.get(1)?.function.arguments).toBe('{"city":"Paris"}');
  });

  it('keeps indexed arguments until identity arrives in a later chunk', async () => {
    const { stream, toolCalls, send } = setup();
    await send({ index: 0, args: '{"city":' });
    await send({ index: 0, id: 'call', name: 'lookup', args: '"Paris"}' });
    stream.finish();
    expect([...toolCalls.values()]).toEqual([
      {
        id: 'call',
        type: 'function',
        function: { name: 'lookup', arguments: '{"city":"Paris"}' },
      },
    ]);
  });

  it('discards failed tools when the fallback produces only text', async () => {
    const { stream, toolCalls, emitted, graph, metadata, send } = setup();
    await send(
      { index: 0, id: 'primary', name: 'lookup', args: '{}' },
      { [STREAM_LIMIT_ATTEMPT_KEY]: 10 }
    );
    const fallback = { ...metadata, [STREAM_LIMIT_ATTEMPT_KEY]: 11 };
    stream.observeModelAttempt(fallback, graph);
    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: new AIMessageChunk({ content: 'No tool needed' }),
      },
      fallback,
      graph
    );
    stream.finish();
    expect(toolCalls.size).toBe(0);
    expect(emitted).toHaveLength(0);
  });

  it('does not mix identical provider indexes across parallel graph nodes', async () => {
    const { stream, toolCalls, send } = setup();
    await send(
      { index: 0, id: 'a', name: 'lookup', args: '{}' },
      { langgraph_node: 'agent=a' }
    );
    await send(
      { index: 0, id: 'b', name: 'lookup', args: '{}' },
      { langgraph_node: 'agent=b' }
    );
    stream.finish();
    expect([...toolCalls.values()].map((call) => call.id)).toEqual(['a', 'b']);
  });

  it('retains the declaring graph segment when earlier chunks drain late', () => {
    const { stream, toolCalls, graph, metadata } = setup();
    for (let i = 0; i < 2; i++) {
      stream.onRunStep(
        {
          id: `step_${i}`,
          stepDetails: {
            type: 'tool_calls',
            tool_calls: [{ id: 'reused', name: 'lookup' }],
          },
        },
        metadata,
        graph
      );
      stream.onRunStepDelta(
        {
          id: `step_${i}`,
          delta: {
            type: 'tool_calls',
            tool_calls: [{ index: 0, id: 'reused', name: 'lookup' }],
          },
        },
        metadata,
        graph
      );
      graph.advanceStreamSegment();
    }
    for (let i = 0; i < 2; i++)
      stream.onRunStepDelta(
        {
          id: `step_${i}`,
          delta: {
            type: 'tool_calls',
            tool_calls: [{ index: 0, args: JSON.stringify({ i }) }],
          },
        },
        metadata,
        graph
      );
    stream.finish();
    expect(
      [...toolCalls.values()].map((call) => call.function.arguments)
    ).toEqual(['{"i":0}', '{"i":1}']);
  });
});

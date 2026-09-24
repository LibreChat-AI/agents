import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  OpenAIChatCompletionChunkChoice,
  OpenAIToolCall,
  OpenAIToolCallStreamConfig,
} from '@/openai';
import type { ModelResponseEvent } from '@/types';
import { createOpenAIToolCallStream } from '@/openai';
import { GraphEvents } from '@/common';

function setup(
  options: Partial<
    Omit<OpenAIToolCallStreamConfig, 'tracker' | 'toolCalls'>
  > = {}
) {
  const toolCalls = new Map<number, OpenAIToolCall>();
  const deltas: OpenAIChatCompletionChunkChoice['delta'][] = [];
  const stream = createOpenAIToolCallStream({
    toolCalls,
    emit: (delta) => {
      deltas.push(delta);
    },
    ...options,
  });
  const accept = (
    calls: ToolCall[],
    id = 'accepted',
    invalid: ModelResponseEvent['invalidToolCalls'] = []
  ) =>
    stream.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
      GraphEvents.ON_MODEL_RESPONSE,
      {
        type: 'model_response',
        id,
        agentId: 'agent',
        toolCalls: calls,
        toolCallDispositions: calls.map(() => 'client' as const),
        invalidToolCalls: invalid,
      }
    );
  return { stream, toolCalls, deltas, accept };
}

describe('accepted tool-call projection', () => {
  it('rejects a reused nonempty result map without destroying prior output', () => {
    const previous: OpenAIToolCall = {
      id: 'old',
      type: 'function',
      function: { name: 'lookup', arguments: '{}' },
    };
    const toolCalls = new Map([[42, previous]]);
    expect(() => createOpenAIToolCallStream({ toolCalls })).toThrow('empty');
    expect(toolCalls.get(42)).toBe(previous);
  });

  it('fails closed if another writer populates the map during collection', () => {
    const { stream, toolCalls, deltas } = setup();
    toolCalls.set(42, {
      id: 'other',
      type: 'function',
      function: { name: 'lookup', arguments: '{}' },
    });
    expect(() => stream.finish()).toThrow('modified');
    expect(deltas).toHaveLength(0);
    expect(toolCalls.size).toBe(0);
    expect(() => stream.finish()).toThrow('aborted');
  });

  it('rejects invalid runtime IDs without leaking the value', () => {
    const { accept, stream } = setup();
    // @ts-expect-error Custom JS models can violate the native tool-call contract.
    expect(() => accept([{ id: 42, name: 'lookup', args: {} }])).toThrow(
      'ID must be a string'
    );
    expect(() => stream.finish()).toThrow('aborted');
  });

  it.each([
    ['Map', new Map([['x', 1]])],
    ['Set', new Set([1])],
    ['Date', new Date('2026-01-01')],
    ['RegExp', /pattern/],
    ['typed array', new Uint8Array([1, 2])],
    ['boxed number', Object(4)],
  ])(
    'rejects nested %s without silently changing executed arguments',
    (_name, value) => {
      const { stream, accept, deltas } = setup();
      expect(() => accept([{ name: 'lookup', args: { value } }])).toThrow(
        'not JSON serializable'
      );
      expect(() => stream.finish()).toThrow('aborted');
      expect(deltas).toHaveLength(0);
    }
  );

  it('rejects an async emitter and observes its rejection instead of reporting completion', async () => {
    const { accept, stream, toolCalls } = setup({
      emit: async () => {
        throw new Error('write failed');
      },
    });
    await accept([{ name: 'lookup', args: {} }]);
    expect(() => stream.finish()).toThrow('synchronous emitter');
    await Promise.resolve();
    expect(() => stream.finish()).toThrow('aborted');
    expect(toolCalls.size).toBe(0);
  });

  it('ignores unrelated events rather than treating them as completed tool calls', async () => {
    const { stream, deltas } = setup();
    await stream.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
      GraphEvents.ON_RUN_STEP,
      { output: undefined }
    );
    stream.finish();
    expect(deltas).toHaveLength(0);
  });

  it('rejects argument toJSON hooks that replace the required object', () => {
    const { accept, stream } = setup();
    expect(() =>
      accept([{ name: 'lookup', args: { toJSON: () => 'SENSITIVE' } }])
    ).toThrow('not JSON serializable');
    expect(() => stream.finish()).toThrow('aborted');
  });

  it.each([NaN, Infinity, undefined, BigInt(1)])(
    'rejects lossy JSON encoding of argument %s',
    (value) => {
      const { accept, stream } = setup();
      expect(() => accept([{ name: 'lookup', args: { value } }])).toThrow(
        'not JSON serializable'
      );
      expect(() => stream.finish()).toThrow('aborted');
    }
  );

  it('fails closed on unserializable arguments without exposing their contents', () => {
    const { stream, accept, deltas } = setup();
    const args: Record<string, unknown> = {};
    args.self = args;
    expect(() => accept([{ id: 'a', name: 'lookup', args }])).toThrow(
      'Accepted tool call arguments are not JSON serializable'
    );
    expect(() => stream.finish()).toThrow('aborted');
    expect(deltas).toHaveLength(0);
  });

  it('rejects missing names on finalized calls rather than guessing from earlier deltas', () => {
    const { accept, stream } = setup();
    expect(() => accept([{ id: 'a', name: '', args: {} }])).toThrow(
      'missing its name'
    );
    expect(() => stream.finish()).toThrow('aborted');
  });

  it('rejects non-object JSON arguments', () => {
    const { accept } = setup();
    expect(() => accept([{ id: 'a', name: 'lookup', args: [] }])).toThrow(
      'must be an object'
    );
  });

  it('formats final calls once with dense indexes across accepted responses', async () => {
    const { stream, deltas, toolCalls, accept } = setup();
    await accept(
      [{ id: 'a', name: 'lookup', args: { city: 'Madrid' } }],
      'first'
    );
    await accept(
      [{ id: 'b', name: 'lookup', args: { city: 'Paris' } }],
      'second'
    );
    expect(deltas).toHaveLength(0);
    expect(toolCalls.size).toBe(0);
    stream.finish();
    expect(deltas).toEqual([
      {
        tool_calls: [
          {
            index: 0,
            id: 'a',
            type: 'function',
            function: { name: 'lookup', arguments: '' },
          },
        ],
      },
      {
        tool_calls: [
          { index: 0, function: { arguments: '{"city":"Madrid"}' } },
        ],
      },
      {
        tool_calls: [
          {
            index: 1,
            id: 'b',
            type: 'function',
            function: { name: 'lookup', arguments: '' },
          },
        ],
      },
      {
        tool_calls: [{ index: 1, function: { arguments: '{"city":"Paris"}' } }],
      },
    ]);
    expect([...toolCalls.keys()]).toEqual([0, 1]);
  });

  it('drops pending calls and releases their budget when a later text answer is accepted', async () => {
    const { stream, accept, toolCalls, deltas } = setup({ maxToolCalls: 1 });
    await accept([{ id: 'already-run', name: 'lookup', args: {} }], 'first');
    await accept([], 'answer');
    stream.finish();
    expect(toolCalls.size).toBe(0);
    expect(deltas).toHaveLength(0);
  });

  it.each([[[undefined, 'call_0']], [['dup', 'dup', 'call_1']]])(
    'reserves future provider IDs before allocating synthetic IDs (%j)',
    async (ids) => {
      const { stream, accept, toolCalls } = setup();
      for (let i = 0; i < ids.length; i++) {
        await accept(
          [{ id: ids[i], name: 'lookup', args: { i } }],
          `accepted-${i}`
        );
      }
      stream.finish();
      const outputs = [...toolCalls.values()].map((call) => call.id);
      expect(new Set(outputs).size).toBe(ids.length);
      ids.forEach((id, index) => {
        if (id != null && ids.indexOf(id) === index)
          expect(outputs[index]).toBe(id);
      });
    }
  );

  it('preserves a later provider ID inside the same accepted response', async () => {
    const { accept, stream, toolCalls } = setup();
    await accept([
      { name: 'lookup', args: { i: 0 } },
      { id: 'call_0', name: 'lookup', args: { i: 1 } },
      { id: 'call_1', name: 'lookup', args: { i: 2 } },
    ]);
    stream.finish();
    expect([...toolCalls.values()].map((call) => call.id)).toEqual([
      'call_2',
      'call_0',
      'call_1',
    ]);
  });

  it('rejects a generated ID that would exceed the byte limit before emitting any frames', async () => {
    const { accept, stream, toolCalls, deltas } = setup({
      maxBufferedBytes: 8,
    });
    await accept([{ name: 'lookup', args: {} }]);
    expect(() => stream.finish()).toThrow('buffer limit');
    expect(deltas).toHaveLength(0);
    expect(toolCalls.size).toBe(0);
  });

  it('does not merge calls by repeated names, provider IDs or argument contents', async () => {
    const { stream, toolCalls, accept } = setup();
    await accept([
      { id: 'call_0', name: 'x', args: { text: 'aaa' } },
      { id: 'call_0', name: 'x', args: { text: 'aaa' } },
      { name: 'x', args: {} },
    ]);
    stream.finish();
    expect(new Set([...toolCalls.values()].map((call) => call.id)).size).toBe(
      3
    );
    expect(toolCalls.get(1)?.function.arguments).toBe('{"text":"aaa"}');
  });

  it('detaches projected state from later mutations and makes published calls immutable', async () => {
    const { stream, toolCalls, accept } = setup();
    const args = { city: 'Paris' };
    await accept([{ id: 'a', name: 'lookup', args }]);
    args.city = 'unsafe';
    stream.finish();
    expect(toolCalls.get(0)?.function.arguments).toBe('{"city":"Paris"}');
    expect(Object.isFrozen(toolCalls.get(0)?.function)).toBe(true);
  });

  it('rejects invalid final calls without publishing a previously accepted call or sensitive arguments', async () => {
    const { stream, accept, deltas, toolCalls } = setup();
    await accept([{ id: 'good', name: 'lookup', args: {} }]);
    expect(() =>
      accept([], 'bad', [
        {
          name: 'lookup',
          args: 'SECRET',
          error: 'SECRET',
          type: 'invalid_tool_call',
        },
      ])
    ).toThrow('Accepted model response contains invalid tool calls');
    expect(() => stream.finish()).toThrow('Agent response aborted');
    expect(deltas).toHaveLength(0);
    expect(toolCalls.size).toBe(0);
  });

  it.each(['', '   '])('rejects malformed accepted identity %j', (id) => {
    const { accept } = setup();
    expect(() => accept([{ name: 'lookup', args: {} }], id)).toThrow(
      'identity'
    );
  });

  it('fails closed on duplicate delivery of an accepted response', async () => {
    const { stream, accept, deltas } = setup();
    await accept([{ name: 'lookup', args: {} }]);
    expect(() => accept([{ name: 'lookup', args: {} }])).toThrow('identity');
    expect(() => stream.finish()).toThrow('aborted');
    expect(deltas).toHaveLength(0);
  });

  it.each(['explicit', 'signal', 'writer'])(
    'stops output permanently on %s failure',
    async (mode) => {
      const controller = new AbortController();
      const frames: OpenAIChatCompletionChunkChoice['delta'][] = [];
      const { stream, accept, toolCalls } = setup({
        signal: controller.signal,
        emit: (delta) => {
          frames.push(delta);
          stream.finish();
          if (mode === 'writer') throw new Error('write failed');
          if (mode === 'signal') controller.abort();
          else stream.abort();
        },
      });
      await accept([
        { name: 'a', args: {} },
        { name: 'b', args: {} },
      ]);
      expect(() => stream.finish()).toThrow(
        mode === 'writer' ? 'write failed' : 'aborted'
      );
      expect(frames).toHaveLength(1);
      expect(() => stream.finish()).toThrow('aborted');
      expect(toolCalls.size).toBe(0);
    }
  );

  it('seals successfully, ignores late accepted results, and leaves results readable', async () => {
    const { stream, accept, toolCalls, deltas } = setup();
    await accept([{ id: 'a', name: 'lookup', args: {} }]);
    stream.finish();
    await accept([{ id: 'late', name: 'lookup', args: {} }], 'late');
    stream.finish();
    stream.abort();
    expect(deltas).toHaveLength(2);
    expect(toolCalls.size).toBe(1);
  });

  it('does not infer acceptance from a run-step or model-end callback', () => {
    const { stream, deltas } = setup();
    expect(Object.keys(stream.handlers)).toEqual([
      GraphEvents.ON_MODEL_TOOLS_CLAIMED,
      GraphEvents.ON_MODEL_RESPONSE,
    ]);
    stream.finish();
    expect(deltas).toHaveLength(0);
  });

  it('bounds buffered call count across responses', async () => {
    const { accept, stream, deltas } = setup({ maxToolCalls: 1 });
    await accept([{ name: 'a', args: {} }]);
    expect(() => accept([{ name: 'b', args: {} }], 'second')).toThrow(
      'call limit'
    );
    expect(() => stream.finish()).toThrow('aborted');
    expect(deltas).toHaveLength(0);
  });

  it('bounds retained UTF-8 bytes including names and IDs', () => {
    const { accept, stream } = setup({ maxBufferedBytes: 20 });
    expect(() =>
      accept([{ name: 'x', id: 'id', args: { data: '🌍🌍🌍' } }])
    ).toThrow('buffer limit');
    expect(() => stream.finish()).toThrow('aborted');
  });

  it('does not retain text-only responses in its replay index', async () => {
    const { accept, stream, toolCalls } = setup({ maxToolCalls: 1 });
    for (let i = 0; i < 1000; i++) await accept([], 'text');
    await accept([{ name: 'lookup', args: {} }], 'text');
    stream.finish();
    expect(toolCalls.size).toBe(1);
  });

  it('bounds id-collision work while processing many accepted calls', async () => {
    const { accept, stream, toolCalls } = setup();
    const calls = Array.from({ length: 500 }, (_, i) => ({
      id: `call_${i}`,
      name: 'lookup',
      args: {},
    }));
    await accept(calls, 'first');
    await accept(calls, 'second');
    stream.finish();
    expect(toolCalls.size).toBe(1000);
    expect(new Set([...toolCalls.values()].map((call) => call.id)).size).toBe(
      1000
    );
  });

  it.each([0, -1, NaN, Infinity, 1.5])('rejects invalid limit %s', (limit) => {
    expect(() => setup({ maxToolCalls: limit })).toThrow('positive safe');
    expect(() => setup({ maxBufferedBytes: limit })).toThrow('positive safe');
  });
});

describe('graph tool ownership', () => {
  it('retires only the claimed agent/message, even when sibling provider IDs collide', async () => {
    const f = setup();
    const accept = (
      agentId: string,
      messageId: string,
      id: string,
      calls: ToolCall[]
    ) =>
      f.stream.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
        GraphEvents.ON_MODEL_RESPONSE,
        {
          type: 'model_response',
          agentId,
          messageId,
          id,
          toolCalls: calls,
          toolCallDispositions: calls.map(() => 'client' as const),
          invalidToolCalls: [],
        }
      );
    const claim = (agentId: string, messageId: string) =>
      f.stream.handlers[GraphEvents.ON_MODEL_TOOLS_CLAIMED].handle(
        GraphEvents.ON_MODEL_TOOLS_CLAIMED,
        { type: 'model_tools_claimed', agentId, messageId }
      );
    await accept('a', 'shared', 'a1', [
      { id: 'same', name: 'lookup', args: { city: 'Paris' } },
    ]);
    await accept('b', 'shared', 'b1', [
      { id: 'same', name: 'lookup', args: { city: 'Madrid' } },
    ]);
    await accept('b', 'next', 'b2', [
      { id: 'next', name: 'lookup', args: { city: 'Rome' } },
    ]);
    await claim('b', 'missing');
    await claim('b', 'shared');
    await claim('b', 'shared'); // duplicate delivery is harmless
    await accept('a', 'answer', 'a2', []);
    f.stream.finish();
    expect(
      [...f.toolCalls.values()].map((call) => call.function.arguments)
    ).toEqual(['{"city":"Rome"}']);
  });

  it('releases both byte and call budgets after each graph-owned batch', async () => {
    const f = setup({ maxToolCalls: 1, maxBufferedBytes: 14 });
    for (let i = 0; i < 1100; i++) {
      await f.stream.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
        GraphEvents.ON_MODEL_RESPONSE,
        {
          type: 'model_response',
          id: String(i),
          messageId: String(i),
          agentId: 'a',
          toolCalls: [{ id: 'x', name: 'lookup', args: {} }],
          toolCallDispositions: ['client'],
          invalidToolCalls: [],
        }
      );
      await f.stream.handlers[GraphEvents.ON_MODEL_TOOLS_CLAIMED].handle(
        GraphEvents.ON_MODEL_TOOLS_CLAIMED,
        { type: 'model_tools_claimed', messageId: String(i), agentId: 'a' }
      );
    }
    f.stream.finish();
    expect(f.toolCalls.size).toBe(0);
    expect(f.deltas).toHaveLength(0);
  });
});

describe('explicit call disposition', () => {
  const deliver = (
    f: ReturnType<typeof setup>,
    calls: ToolCall[],
    dispositions?: Array<'client' | 'sdk' | 'provider'>
  ) =>
    f.stream.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
      GraphEvents.ON_MODEL_RESPONSE,
      {
        type: 'model_response',
        id: 'decision',
        agentId: 'agent',
        toolCalls: calls,
        ...(dispositions === undefined
          ? {}
          : { toolCallDispositions: dispositions }),
        invalidToolCalls: [],
      } as ModelResponseEvent
    );

  it('fails closed on missing, mismatched, and unknown disposition', async () => {
    for (const ownership of [
      undefined,
      [],
      ['unknown'] as unknown as Array<'client'>,
    ]) {
      const f = setup();
      expect(() =>
        deliver(f, [{ id: 'one', name: 'lookup', args: {} }], ownership)
      ).toThrow('trusted execution ownership');
      expect(f.toolCalls.size).toBe(0);
      expect(f.deltas).toHaveLength(0);
      expect(() => f.stream.finish()).toThrow('aborted');
    }
  });

  it('includes only explicit client delegation in a mixed accepted result', async () => {
    const f = setup();
    await deliver(
      f,
      [
        { id: 'server', name: 'web_search', args: {} },
        { id: 'internal', name: 'lookup', args: {} },
        { id: 'client', name: 'external', args: { x: 1 } },
      ],
      ['provider', 'sdk', 'client']
    );
    f.stream.finish();
    expect([...f.toolCalls.values()].map((call) => call.id)).toEqual([
      'client',
    ]);
    expect(f.deltas.filter((delta) => delta.tool_calls != null)).toHaveLength(
      2
    );
  });

  it('never projects internal or provider calls without a later tool claim', async () => {
    const f = setup();
    await deliver(
      f,
      [
        { id: 'provider', name: 'web_search', args: {} },
        { id: 'sdk', name: 'lookup', args: {} },
      ],
      ['provider', 'sdk']
    );
    f.stream.finish();
    expect(f.toolCalls.size).toBe(0);
    expect(f.deltas).toHaveLength(0);
  });
});

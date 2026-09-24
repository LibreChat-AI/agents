import type { OpenAIToolCall, OpenAIChatCompletionChunkChoice } from '@/openai';
import type { RunStep, RunStepDeltaEvent } from '@/types';
import { createOpenAIToolCallStream } from '@/openai';
import { StepTypes } from '@/common';

type Delta = OpenAIChatCompletionChunkChoice['delta'];

function setup() {
  const toolCalls = new Map<number, OpenAIToolCall>();
  const deltas: Delta[] = [];
  const stream = createOpenAIToolCallStream({
    toolCalls,
    emit: (delta) => {
      deltas.push(delta);
    },
  });
  return { toolCalls, deltas, stream };
}

describe('complete-before-publish Chat Completions tool projection', () => {
  it('projects parallel calls from native SDK RunStep and RunStepDeltaEvent payloads', () => {
    const { toolCalls, deltas, stream } = setup();
    const step: RunStep = {
      id: 'step',
      index: 4,
      type: StepTypes.TOOL_CALLS,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          { id: 'call_a', name: 'lookup', args: {} },
          { id: 'call_b', name: 'lookup', args: {} },
        ],
      },
    };
    const fragments: RunStepDeltaEvent = {
      id: 'step',
      delta: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          { index: 0, id: 'call_a', name: 'lookup', args: '{"city":"Madrid"}' },
          { index: 1, id: 'call_b', name: 'lookup', args: '{"city":"Paris"}' },
        ],
      },
    };
    stream.onRunStep(step);
    stream.onRunStepDelta(fragments);
    expect(deltas).toHaveLength(0);
    stream.finish();
    expect(
      [...toolCalls.values()].map((call) => call.function.arguments)
    ).toEqual(['{"city":"Madrid"}', '{"city":"Paris"}']);
    expect(
      deltas
        .filter((delta) => delta.tool_calls?.[0].id !== undefined)
        .map((delta) => delta.tool_calls?.[0].index)
    ).toEqual([0, 1]);
  });

  it('uses outward call ordinals after text and across later run steps, not step or provider indexes', () => {
    const { toolCalls, deltas, stream } = setup();
    stream.onRunStep({
      id: 'step_one',
      index: 3,
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ id: 'call_a', name: 'first', args: {}, index: 2 }],
      },
    });
    stream.onRunStepDelta({
      id: 'step_one',
      delta: {
        type: 'tool_calls',
        tool_calls: [{ index: 2, id: 'call_a', args: '{}' }],
      },
    });
    stream.onRunStep({
      id: 'step_two',
      index: 4,
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ id: 'call_b', name: 'second', args: {} }],
      },
    });
    stream.onRunStepDelta({
      id: 'step_two',
      delta: {
        type: 'tool_calls',
        tool_calls: [{ index: 0, id: 'call_b', args: '{}' }],
      },
    });
    expect(deltas).toHaveLength(0);
    stream.finish();
    expect([...toolCalls.keys()]).toEqual([0, 1]);
    expect(
      deltas
        .flatMap((delta) => delta.tool_calls ?? [])
        .map((call) => call.index)
    ).toEqual([0, 0, 1, 1]);
    expect(
      deltas
        .filter((delta) => (delta.tool_calls?.[0].id ?? '') !== '')
        .map((delta) => delta.tool_calls?.[0].id)
    ).toEqual(['call_a', 'call_b']);
  });

  it('scopes reused provider indexes to their model invocation', () => {
    const { toolCalls, stream } = setup();
    const graph = {
      getStepBaseKey: (metadata: Record<string, unknown> | undefined): string =>
        String(metadata?.run_id),
    };
    stream.onRunStepDelta(
      {
        id: 'a',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, id: 'call_a', name: 'one', args: '{}' }],
        },
      },
      { run_id: 'first' },
      graph
    );
    stream.onRunStepDelta(
      {
        id: 'b',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, id: 'call_b', name: 'two', args: '{}' }],
        },
      },
      { run_id: 'second' },
      graph
    );
    stream.finish();
    expect([...toolCalls.values()].map((call) => call.id)).toEqual([
      'call_a',
      'call_b',
    ]);
  });

  it('assembles split IDs and names before any client can freeze them', () => {
    const { toolCalls, deltas, stream } = setup();
    for (const [id, name, args] of [
      ['call_', 'get_', '{"city":'],
      ['123', 'weather', '"Paris"}'],
    ]) {
      stream.onRunStepDelta({
        id: 'step',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 4, id, name, args }],
        },
      });
    }
    expect(deltas).toHaveLength(0);
    stream.finish();
    expect(toolCalls.get(0)).toEqual({
      id: 'call_123',
      type: 'function',
      function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
    });
    expect(deltas[0].tool_calls?.[0]).toMatchObject({
      index: 0,
      id: 'call_123',
      function: { name: 'get_weather', arguments: '' },
    });
    expect(deltas[1].tool_calls?.[0].function?.arguments).toBe(
      '{"city":"Paris"}'
    );
  });

  it('serializes object snapshots once and assigns unique IDs to id-less calls', () => {
    const { toolCalls, stream } = setup();
    stream.onRunStep({
      id: 'step',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [
          { name: 'lookup', args: { city: 'Madrid' } },
          { name: 'lookup', args: { city: 'Paris' } },
        ],
      },
    });
    stream.finish();
    expect(
      [...toolCalls.values()].map((call) => call.function.arguments)
    ).toEqual(['{"city":"Madrid"}', '{"city":"Paris"}']);
    expect(new Set([...toolCalls.values()].map((call) => call.id)).size).toBe(
      2
    );
  });

  it('validates all calls before publishing any identity and refuses malformed arguments', () => {
    const { toolCalls, deltas, stream } = setup();
    stream.onRunStep({
      id: 'step',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [
          { name: 'valid', args: {} },
          { name: 'invalid', args: 'NOT_JSON' },
        ],
      },
    });
    expect(() => stream.finish()).toThrow('Invalid tool call arguments');
    expect(deltas).toHaveLength(0);
    expect(toolCalls.size).toBe(0);
    stream.onRunStep({
      id: 'late',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'ignored', args: {} }],
      },
    });
    expect(() => stream.finish()).toThrow('Agent response aborted');
  });

  it('seals before transport callbacks so reentrant finish or writer failure cannot duplicate calls', () => {
    const toolCalls = new Map<number, OpenAIToolCall>();
    const emitted: Delta[] = [];
    const stream = createOpenAIToolCallStream({
      toolCalls,
      emit: (delta) => {
        emitted.push(delta);
        stream.finish();
        throw new Error('transport failed');
      },
    });
    stream.onRunStep({
      id: 'step',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'lookup', args: {} }],
      },
    });
    expect(() => stream.finish()).toThrow('transport failed');
    expect(() => stream.finish()).not.toThrow();
    expect(emitted).toHaveLength(1);
  });

  it('aborts before output and does not emit on a later finish', () => {
    const { toolCalls, deltas, stream } = setup();
    stream.onRunStep({
      id: 'step',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'lookup', args: {} }],
      },
    });
    stream.abort();
    expect(() => stream.finish()).toThrow('Agent response aborted');
    expect(toolCalls.size).toBe(0);
    expect(deltas).toHaveLength(0);
  });
});

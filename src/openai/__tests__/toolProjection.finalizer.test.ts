import type { ModelResponseEvent } from '@/types';
import {
  createOpenAIToolCallStream,
  createOpenAIStreamTracker,
  createOpenAIHandlers,
  createChatCompletionChunk,
  sendOpenAIFinalChunk,
} from '@/openai';
import { GraphEvents } from '@/common';

function setup(mode?: 'throw' | 'abort') {
  const writes: string[] = [];
  const tracker = createOpenAIStreamTracker();
  const context = { requestId: 'request', model: 'agent', created: 1 };
  const config = {
    tracker,
    context,
    writer: {
      write: (frame: string): void => {
        writes.push(frame);
      },
    },
  };
  // The tracker is the shared state owner for projection and terminal formatting.
  const projection = createOpenAIToolCallStream({
    tracker,
    emit: (delta) => {
      config.writer.write(
        `data: ${JSON.stringify(createChatCompletionChunk(context, delta))}\n\n`
      );
      if (mode === 'throw') throw new Error('writer failed');
      if (mode === 'abort') projection.abort();
    },
  });
  let acceptedSequence = 0;
  const accept = (toolCalls: ModelResponseEvent['toolCalls']) =>
    projection.handlers[GraphEvents.ON_MODEL_RESPONSE].handle(
      GraphEvents.ON_MODEL_RESPONSE,
      {
        type: 'model_response',
        id: `accepted-${++acceptedSequence}`,
        agentId: 'agent',
        toolCalls,
        invalidToolCalls: [],
      }
    );
  const text = async (): Promise<void> => {
    await createOpenAIHandlers(config)[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      { id: 'msg', delta: { content: [{ type: 'text', text: 'done' }] } }
    );
  };
  return { writes, tracker, config, projection, accept, text };
}

function terminal(writes: string[]) {
  return writes
    .slice(0, -1)
    .map((frame) => JSON.parse(frame.slice(6)))
    .find((chunk) => chunk.choices[0]?.finish_reason != null);
}

describe('accepted projector with public OpenAI finalizer', () => {
  it('omits already-executed calls from a final text response', async () => {
    const f = setup();
    await f.accept([{ id: 'call', name: 'lookup', args: {} }]);
    await f.text();
    await f.accept([]); // final-answer acceptance, before processStream returns
    f.projection.finish();
    await sendOpenAIFinalChunk(f.config);
    expect(terminal(f.writes).choices[0].finish_reason).toBe('stop');
    expect(f.tracker.toolCalls.size).toBe(0);
    const deltas = f.writes
      .slice(0, -1)
      .map((frame) => JSON.parse(frame.slice(6)));
    expect(
      deltas
        .flatMap((chunk) => chunk.choices)
        .every((choice) => choice.delta.tool_calls == null)
    ).toBe(true);
  });

  it('retains tool_calls when the last accepted response requests a new tool', async () => {
    const f = setup();
    await f.text();
    await f.accept([]);
    await f.accept([{ id: 'call', name: 'lookup', args: {} }]);
    f.projection.finish();
    await sendOpenAIFinalChunk(f.config);
    expect(terminal(f.writes).choices[0].finish_reason).toBe('tool_calls');
  });

  it('requires a real sink before taking ownership of streaming tracker state', () => {
    const tracker = createOpenAIStreamTracker();
    expect(() =>
      // @ts-expect-error Streaming mode must not advance role/finish state without output.
      createOpenAIToolCallStream({ tracker })
    ).toThrow('requires an emitter');
    expect(tracker.hasRole).toBe(false);
  });

  it.each(['throw', 'abort'] as const)(
    'does not leave a false tool-call finish marker after %s',
    async (mode) => {
      const f = setup(mode);
      await f.text();
      await f.accept([{ name: 'lookup', args: {} }]);
      expect(() => f.projection.finish()).toThrow();
      expect(f.tracker.toolCalls.size).toBe(0);
      expect(f.tracker.lastChunkKind).toBe('text');
      expect(() => f.projection.finish()).toThrow('aborted');
      expect(f.writes).not.toContain('data: [DONE]\n\n');
    }
  );

  it('rejects two independent storage owners', () => {
    const tracker = createOpenAIStreamTracker();
    expect(() =>
      // @ts-expect-error The unreleased API permits a tracker or a map, never both.
      createOpenAIToolCallStream({
        tracker,
        toolCalls: new Map(),
        emit: () => undefined,
      })
    ).toThrow('not both');
  });

  it.each([false, true])(
    'sets tool_calls after projected chunks (text before=%s)',
    async (before) => {
      const f = setup();
      if (before) await f.text();
      await f.accept([{ id: 'call', name: 'lookup', args: {} }]);
      f.projection.finish();
      await sendOpenAIFinalChunk(f.config);
      expect(terminal(f.writes).choices[0].finish_reason).toBe('tool_calls');
      expect(f.writes.at(-1)).toBe('data: [DONE]\n\n');
      expect(JSON.parse(f.writes[0].slice(6)).choices[0].delta).toEqual({
        role: 'assistant',
      });
      expect(
        f.writes.filter((frame) => frame.includes('"role":"assistant"'))
      ).toHaveLength(1);
    }
  );

  it('still ends with stop when later assistant text follows published tool calls', async () => {
    const f = setup();
    await f.accept([{ id: 'call', name: 'lookup', args: {} }]);
    f.projection.finish();
    await f.text();
    await sendOpenAIFinalChunk(f.config);
    expect(terminal(f.writes).choices[0].finish_reason).toBe('stop');
  });

  it.each([false, true])(
    'does not fabricate tool_calls for an empty projection (text=%s)',
    async (text) => {
      const f = setup();
      if (text) await f.text();
      f.projection.finish();
      await sendOpenAIFinalChunk(f.config);
      expect(terminal(f.writes).choices[0].finish_reason).toBe('stop');
    }
  );

  it('preserves explicit terminal reasons and usage framing', async () => {
    const f = setup();
    f.tracker.usage = {
      promptTokens: 5,
      completionTokens: 3,
      reasoningTokens: 1,
    };
    await f.accept([{ name: 'lookup', args: {} }]);
    f.projection.finish();
    await sendOpenAIFinalChunk(f.config, 'length');
    expect(terminal(f.writes).choices[0].finish_reason).toBe('length');
    expect(JSON.parse(f.writes.at(-2)!.slice(6))).toMatchObject({
      choices: [],
      usage: {
        prompt_tokens: 5,
        completion_tokens: 3,
        total_tokens: 8,
        completion_tokens_details: { reasoning_tokens: 1 },
      },
    });
  });
});

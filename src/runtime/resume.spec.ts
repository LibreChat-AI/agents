import type { RunStep } from '@/index';
import {
  ContentTypes,
  GraphEvents,
  StepTypes,
  createContentAggregator,
} from '@/index';
import {
  createContentIndexOffsetHandlers,
  hydrateResumeRunSteps,
} from './resume';

describe('createContentIndexOffsetHandlers', () => {
  const textSeed = (n: number) =>
    Array.from({ length: n }, () => ({ type: 'text' }));
  const makeRecorder = () => {
    const calls: Array<{ event: string; data: unknown }> = [];
    const handler = {
      handle: (event: string, data: unknown) =>
        void calls.push({ event, data }),
    };
    return { calls, handler };
  };

  it('returns the input untouched for offset 0 / undefined handlers', () => {
    const { handler } = makeRecorder();
    const handlers = { on_run_step: handler };
    expect(createContentIndexOffsetHandlers(handlers, [])).toBe(handlers);
    expect(
      createContentIndexOffsetHandlers(undefined, textSeed(3))
    ).toBeUndefined();
  });

  it('shifts ON_RUN_STEP index by the offset without mutating the original payload', () => {
    const { calls, handler } = makeRecorder();
    const wrapped = createContentIndexOffsetHandlers(
      { on_run_step: handler },
      textSeed(3)
    )!;
    const runStep = {
      id: 'step_1',
      index: 0,
      stepDetails: { type: 'message_creation' },
    };
    wrapped.on_run_step.handle('on_run_step', runStep as never);
    expect((calls[0].data as { index: number }).index).toBe(3);
    expect(runStep.index).toBe(0); // caller's object untouched
  });

  it('shifts ON_AGENT_UPDATE nested index', () => {
    const { calls, handler } = makeRecorder();
    const wrapped = createContentIndexOffsetHandlers(
      { on_agent_update: handler },
      textSeed(2)
    )!;
    wrapped.on_agent_update.handle('on_agent_update', {
      agent_update: { index: 1, runId: 'r' },
    } as never);
    expect(
      (calls[0].data as { agent_update: { index: number } }).agent_update.index
    ).toBe(3);
  });

  it('passes every other event handler through by reference (stateful instances intact)', () => {
    const { handler } = makeRecorder();
    const deltaHandler = { handle: jest.fn() };
    const wrapped = createContentIndexOffsetHandlers(
      { on_run_step: handler, on_message_delta: deltaHandler },
      textSeed(5)
    )!;
    expect(wrapped.on_message_delta).toBe(deltaHandler);
    // Deltas carry no index — they resolve through the (shifted) stepMap entry.
    wrapped.on_message_delta.handle('on_message_delta', {
      id: 'step_1',
      delta: {},
    } as never);
    expect(deltaHandler.handle).toHaveBeenCalledTimes(1);
  });

  it('rebinds a resumed tool step to its seeded unresolved slot by tool_call id', () => {
    const { calls, handler } = makeRecorder();
    const seed = [
      { type: 'text' },
      { type: 'tool_call', tool_call: { id: 'tc_paused', output: '' } },
      {
        type: 'tool_call',
        tool_call: { id: 'tc_done', output: 'already resolved' },
      },
    ];
    const wrapped = createContentIndexOffsetHandlers(
      { on_run_step: handler },
      seed
    )!;

    // The paused call's re-execution must land on its seeded slot (1), not 0+3.
    wrapped.on_run_step.handle('on_run_step', {
      id: 'step_t',
      index: 0,
      stepDetails: { type: 'tool_calls', tool_calls: [{ id: 'tc_paused' }] },
    } as never);
    expect((calls[0].data as { index: number }).index).toBe(1);

    // Resolved seeded calls rebind too: ids are minted per call, so a same-id
    // step can only be the interrupted batch re-executing (e.g. an ask part
    // whose answer the resume controller pre-stamped onto the seed).
    wrapped.on_run_step.handle('on_run_step', {
      id: 'step_u',
      index: 1,
      stepDetails: { type: 'tool_calls', tool_calls: [{ id: 'tc_done' }] },
    } as never);
    expect((calls[1].data as { index: number }).index).toBe(2);

    // Message steps always offset.
    wrapped.on_run_step.handle('on_run_step', {
      id: 'step_m',
      index: 2,
      stepDetails: { type: 'message_creation' },
    } as never);
    expect((calls[2].data as { index: number }).index).toBe(5);
  });

  it('leaves a runStep without a numeric index unshifted (defensive)', () => {
    const { calls, handler } = makeRecorder();
    const wrapped = createContentIndexOffsetHandlers(
      { on_run_step: handler },
      textSeed(4)
    )!;
    const weird = { id: 'step_x' };
    wrapped.on_run_step.handle('on_run_step', weird as never);
    expect(calls[0].data).toBe(weird);
  });
});

describe('hydrateResumeRunSteps', () => {
  it('restores step and tool-call identity so a resumed completion updates its seeded card', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call-approval',
          name: 'approval_probe',
          args: '{"value":"before"}',
        },
      }
    );
    const runStep: RunStep = {
      id: 'step-approval',
      runId: 'response-1',
      type: StepTypes.TOOL_CALLS,
      index: 1,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          {
            id: 'call-approval',
            name: 'approval_probe',
            args: { value: 'before' },
          },
        ],
      },
      usage: null,
    };
    const toolCallStepIds = new Map<string, string>();

    hydrateResumeRunSteps(
      [runStep],
      stepMap,
      { toolCallStepIds },
      contentParts
    );
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step-approval',
          index: 1,
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'call-approval',
            name: 'approval_probe',
            args: { value: 'before' },
            output: 'approved output',
          },
        },
      },
    });

    expect(stepMap.get('step-approval')).toBe(runStep);
    expect(toolCallStepIds.get('call-approval')).toBe('step-approval');
    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call-approval',
        output: 'approved output',
        progress: 1,
      },
    });
  });

  it('realigns a stale persisted index to the seeded tool card by tool-call id', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      {
        type: ContentTypes.TEXT,
        text: 'Prepended after the step was recorded',
      },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call-shifted',
          name: 'approval_probe',
          args: '{"value":"shifted"}',
        },
      }
    );
    const staleRunStep: RunStep = {
      id: 'step-shifted',
      runId: 'response-1',
      type: StepTypes.TOOL_CALLS,
      index: 0,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          {
            id: 'call-shifted',
            name: 'approval_probe',
            args: { value: 'shifted' },
          },
        ],
      },
      usage: null,
    };
    const toolCallStepIds = new Map<string, string>();

    hydrateResumeRunSteps(
      [staleRunStep],
      stepMap,
      { toolCallStepIds },
      contentParts
    );
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step-shifted',
          index: 0,
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'call-shifted',
            name: 'approval_probe',
            args: { value: 'shifted' },
            output: 'shifted output',
          },
        },
      },
    });

    expect(staleRunStep.index).toBe(0);
    expect(stepMap.get('step-shifted')?.index).toBe(1);
    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Prepended after the step was recorded',
    });
    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call-shifted',
        output: 'shifted output',
        progress: 1,
      },
    });
  });
});

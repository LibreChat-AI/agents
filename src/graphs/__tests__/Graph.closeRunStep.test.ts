// src/graphs/__tests__/Graph.closeRunStep.test.ts
import { CallbackManager } from '@langchain/core/callbacks/manager';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type * as t from '@/types';
import { GraphEvents, StepTypes, Providers } from '@/common';
import { HandlerRegistry } from '@/events';
import { StandardGraph } from '../Graph';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

function createGraph(): {
  graph: StandardGraph;
  closed: t.RunStepClosedEvent[];
  } {
  const graph = new StandardGraph({
    runId: 'run_1',
    agents: [makeAgent('agent')],
  });
  const closed: t.RunStepClosedEvent[] = [];
  const registry = new HandlerRegistry();
  registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
    handle: (_event, data): void => {
      closed.push(data as t.RunStepClosedEvent);
    },
  });
  graph.handlerRegistry = registry;
  return { graph, closed };
}

function seedStep(
  graph: StandardGraph,
  id: string,
  type: StepTypes = StepTypes.MESSAGE_CREATION
): t.RunStep {
  const index = graph.contentData.length;
  const stepDetails: t.StepDetails =
    type === StepTypes.TOOL_CALLS
      ? { type: StepTypes.TOOL_CALLS, tool_calls: [] }
      : {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: `msg_${id}` },
      };
  const step: t.RunStep = {
    id,
    type,
    index,
    stepIndex: index,
    stepDetails,
    usage: null,
    created_at: 1_000,
    status: 'in_progress',
  };
  graph.contentData.push(step);
  graph.contentIndexMap.set(id, index);
  return step;
}

describe('StandardGraph.dispatchRunStepDelta', () => {
  it('delivers the producer receipt time rather than timing a slow consumer', async () => {
    const { graph } = createGraph();
    const received: t.RunStepDeltaEvent[] = [];
    graph.config = { configurable: { run_id: 'run_1' } };
    graph.handlerRegistry?.register(GraphEvents.ON_RUN_STEP_DELTA, {
      handle: async (_event, data): Promise<void> => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        received.push(data as t.RunStepDeltaEvent);
      },
    });
    await graph.dispatchRunStepDelta(
      'step_tool',
      { type: StepTypes.TOOL_CALLS, tool_calls: [{ index: 0, args: '{', type: 'tool_call_chunk' }] },
      undefined,
      1_000
    );
    expect(received).toEqual([
      { id: 'step_tool', delta: expect.objectContaining({ type: StepTypes.TOOL_CALLS }), observed_at: 1_000 },
    ]);
  });
});

describe('StandardGraph.closeRunStep', () => {
  it('stamps the terminal status + timestamp and emits ON_RUN_STEP_CLOSED once', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a');

    const first = await graph.closeRunStep('step_a', 'completed', {
      at: 2_000,
    });
    expect(first).toBe(true);
    expect(step.status).toBe('completed');
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
    expect(closed[0]).toMatchObject({
      id: 'step_a',
      index: 0,
      type: StepTypes.MESSAGE_CREATION,
      status: 'completed',
      created_at: 1_000,
      closed_at: 2_000,
    });

    const second = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
    });
    expect(second).toBe(false);
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
  });

  it('keeps a terminal status immutable across later close attempts', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);

    await graph.closeRunStep('step_a', 'cancelled', { at: 2_000 });
    const reclosed = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
    });
    expect(reclosed).toBe(false);
    expect(step.status).toBe('cancelled');
    expect(step.cancelled_at).toBe(2_000);
    expect(step.completed_at).toBeUndefined();
    expect(closed).toHaveLength(1);
  });

  it('keeps the stored stamp and the emitted event in agreement', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);

    await graph.closeRunStep('step_a', 'completed', { at: 2_000 });
    const reclosed = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
    });

    /** No silent correction: RunStep.completed_at can never disagree with the
     *  closed_at already delivered to subscribers. */
    expect(reclosed).toBe(false);
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
    expect(closed[0].closed_at).toBe(2_000);
  });

  it('tolerates empty and unknown step ids', async () => {
    const { graph, closed } = createGraph();
    expect(await graph.closeRunStep('', 'completed')).toBe(false);
    expect(await graph.closeRunStep('step_missing', 'completed')).toBe(false);
    expect(closed).toHaveLength(0);
  });
});

describe('StandardGraph.recordStepCompletion', () => {
  it('closes a TOOL_CALLS step only after every registered call completes', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');
    graph.registerPendingToolCall('call_2', 'step_a');

    await graph.recordStepCompletion('step_a', { toolCallId: 'call_1' });
    expect(step.status).toBe('in_progress');
    expect(closed).toHaveLength(0);

    await graph.recordStepCompletion('step_a', { toolCallId: 'call_2' });
    expect(step.status).toBe('completed');
    expect(typeof step.completed_at).toBe('number');
    expect(closed).toHaveLength(1);
    expect(closed[0].status).toBe('completed');
  });

  it('closes steps without pending tracking on the first completion', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a');

    await graph.recordStepCompletion('step_a');
    expect(step.status).toBe('completed');
    expect(closed).toHaveLength(1);
  });

  it('absorbs duplicate completions without re-emitting', async () => {
    const { graph, closed } = createGraph();
    seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');

    await graph.recordStepCompletion('step_a', { toolCallId: 'call_1' });
    await graph.recordStepCompletion('step_a', { toolCallId: 'call_1' });
    expect(closed).toHaveLength(1);
  });

  it('leaves a closed step untouched if a late call registers and completes', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');

    await graph.recordStepCompletion('step_a', {
      toolCallId: 'call_1',
      at: 2_000,
    });
    expect(closed).toHaveLength(1);

    graph.registerPendingToolCall('call_2', 'step_a');
    await graph.recordStepCompletion('step_a', {
      toolCallId: 'call_2',
      at: 3_000,
    });
    /** First close wins outright — no second event and no divergence between
     *  the stored stamp and what subscribers were told. */
    expect(closed).toHaveLength(1);
    expect(step.completed_at).toBe(2_000);
    expect(closed[0].closed_at).toBe(2_000);
  });
  it('closes a multi-call step at the latest producer timestamp', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');
    graph.registerPendingToolCall('call_2', 'step_a');

    /** call_2 finished later, but its host handler was faster, so the
     *  earlier-finishing call_1 is what drains the pending set. */
    await graph.recordStepCompletion('step_a', {
      toolCallId: 'call_2',
      at: 5_000,
    });
    await graph.recordStepCompletion('step_a', {
      toolCallId: 'call_1',
      at: 4_000,
    });

    expect(step.completed_at).toBe(5_000);
    expect(closed).toHaveLength(1);
    expect(closed[0].closed_at).toBe(5_000);
  });

  it('uses the producer\'s completion timestamp when one is supplied', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');

    await graph.recordStepCompletion('step_a', {
      toolCallId: 'call_1',
      at: 4_000,
    });

    /** A slow host handler must not inflate the recorded duration. */
    expect(step.completed_at).toBe(4_000);
    expect(closed).toHaveLength(1);
    expect(closed[0].closed_at).toBe(4_000);
  });
});

describe('StandardGraph.closeUnfinishedRunSteps', () => {
  it('closes only non-terminal steps with the sweep status', async () => {
    const { graph, closed } = createGraph();
    const done = seedStep(graph, 'step_done', StepTypes.TOOL_CALLS);
    const openMessage = seedStep(graph, 'step_msg');
    const openTool = seedStep(graph, 'step_tool', StepTypes.TOOL_CALLS);
    await graph.closeRunStep('step_done', 'completed', { at: 2_000 });
    closed.length = 0;

    await graph.closeUnfinishedRunSteps('cancelled', 5_000);

    expect(done.completed_at).toBe(2_000);
    expect(openMessage.status).toBe('cancelled');
    expect(openMessage.cancelled_at).toBe(5_000);
    expect(openTool.status).toBe('cancelled');
    expect(openTool.cancelled_at).toBe(5_000);
    expect(closed.map((event) => event.id)).toEqual(['step_msg', 'step_tool']);
    expect(closed.every((event) => event.status === 'cancelled')).toBe(true);
    expect(graph.pendingToolCallsByStep.size).toBe(0);
    expect(graph.openMessageStepByAgent.size).toBe(0);
  });

  it('stamps failed_at for failed sweeps', async () => {
    const { graph } = createGraph();
    const step = seedStep(graph, 'step_a');

    await graph.closeUnfinishedRunSteps('failed', 5_000);
    expect(step.status).toBe('failed');
    expect(step.failed_at).toBe(5_000);
    expect(step.cancelled_at).toBeUndefined();
    expect(step.completed_at).toBeUndefined();
  });
  it('keeps sweeping when a closure handler throws', async () => {
    const graph = new StandardGraph({
      runId: 'run_1',
      agents: [makeAgent('agent')],
    });
    const seen: string[] = [];
    const registry = new HandlerRegistry();
    registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
      handle: (_event, data): void => {
        const event = data as t.RunStepClosedEvent;
        seen.push(event.id);
        if (event.id === 'step_1') {
          throw new Error('host handler blew up');
        }
      },
    });
    graph.handlerRegistry = registry;
    const first = seedStep(graph, 'step_1');
    const second = seedStep(graph, 'step_2');
    const third = seedStep(graph, 'step_3');

    await graph.closeUnfinishedRunSteps('cancelled', 5_000);

    expect(seen).toEqual(['step_1', 'step_2', 'step_3']);
    for (const step of [first, second, third]) {
      expect(step.status).toBe('cancelled');
      expect(step.cancelled_at).toBe(5_000);
    }
  });

  describe('host handler isolation', () => {
    function createThrowingGraph(): {
      graph: StandardGraph;
      seen: string[];
      } {
      const graph = new StandardGraph({
        runId: 'run_1',
        agents: [makeAgent('agent')],
      });
      const seen: string[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
        handle: (_event, data): void => {
          seen.push((data as t.RunStepClosedEvent).id);
          throw new Error('host handler blew up');
        },
      });
      graph.handlerRegistry = registry;
      return { graph, seen };
    }

    /**
     * The step is stamped terminal before dispatch, so a delivery failure
     * reports on state that is already committed. Propagating would fail the
     * whole run over an observational event — `closeOpenMessageStep` awaits
     * this inside the stream loop on every CHAT_MODEL_END.
     */
    it('does not propagate a throwing host handler', async () => {
      const { graph, seen } = createThrowingGraph();
      const step = seedStep(graph, 'step_a');

      await expect(
        graph.closeRunStep('step_a', 'completed', { at: 2_000 })
      ).resolves.toBe(true);

      expect(seen).toEqual(['step_a']);
      expect(step.status).toBe('completed');
      expect(step.completed_at).toBe(2_000);
    });

    /**
     * The hot path from `run.ts`: a rejection here sets `streamThrew` and
     * fires the StopFailure hooks for a response that fully delivered.
     */
    it('does not fail the stream loop when closing the open message step', async () => {
      const { graph, seen } = createThrowingGraph();
      const step = seedStep(graph, 'step_msg');
      graph.openMessageStepByAgent.set('', 'step_msg');

      await expect(
        graph.closeOpenMessageStep(undefined, 3_000)
      ).resolves.toBeUndefined();

      expect(seen).toEqual(['step_msg']);
      expect(step.status).toBe('completed');
    });

    /**
     * The secondary channel is the documented fallback for when the primary
     * dispatch does not deliver, so a throwing primary must not suppress it —
     * that would remove the redundancy exactly when it is needed.
     */
    it('still dispatches the secondary custom event after the handler throws', async () => {
      const { graph } = createThrowingGraph();
      seedStep(graph, 'step_b');
      /**
       * A real subscriber on the real dispatch path. Asserting that
       * `dispatchCustomEvent` was merely called would still pass if the
       * config or the callback propagation dropped the event — the exact
       * failure this test exists to catch — so the assertion is that a
       * handler actually received it.
       *
       * The manager needs a parent run id: `dispatchCustomEvent` reads it
       * from `getParentRunId()`, not from the config, and silently skips the
       * dispatch when there is none.
       */
      const received: Array<{ name: string; payload: unknown }> = [];
      const callbacks = new CallbackManager('parent-run-1');
      callbacks.addHandler(
        BaseCallbackHandler.fromMethods({
          handleCustomEvent(name: string, payload: unknown): void {
            received.push({ name, payload });
          },
        }),
        true,
      );
      graph.config = { configurable: { thread_id: 'test' }, callbacks };

      await graph.closeRunStep('step_b', 'failed', { at: 4_000 });

      expect(received).toHaveLength(1);
      expect(received[0].name).toBe(GraphEvents.ON_RUN_STEP_CLOSED);
      expect(received[0].payload).toMatchObject({
        id: 'step_b',
        status: 'failed',
        closed_at: 4_000,
      });
    });

    /** A terminal status stays immutable even when delivery failed. */
    it('keeps the close authoritative despite the delivery failure', async () => {
      const { graph, seen } = createThrowingGraph();
      const step = seedStep(graph, 'step_c');

      await graph.closeRunStep('step_c', 'cancelled', { at: 5_000 });
      const second = await graph.closeRunStep('step_c', 'completed', {
        at: 6_000,
      });

      expect(second).toBe(false);
      expect(seen).toEqual(['step_c']);
      expect(step.status).toBe('cancelled');
      expect(step.cancelled_at).toBe(5_000);
      expect(step.completed_at).toBeUndefined();
    });
  });
});

describe('StandardGraph.trackDispatchedRunStep', () => {
  it('reserves distinct content indexes when parallel lanes dispatch concurrently', async () => {
    const graph = new StandardGraph({
      runId: 'run_1',
      agents: [makeAgent('agent')],
    });
    /** A host handler that yields — the window in which two lanes could
     *  otherwise both read the same contentData.length. */
    const registry = new HandlerRegistry();
    registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
      handle: async (): Promise<void> => {
        await new Promise((resolve) => setTimeout(resolve, 5));
      },
    });
    graph.handlerRegistry = registry;

    const openA = seedStep(graph, 'open_a');
    openA.agentId = 'agent_a';
    const openB = seedStep(graph, 'open_b');
    openB.agentId = 'agent_b';
    graph.openMessageStepByAgent.set('agent_a', 'open_a');
    graph.openMessageStepByAgent.set('agent_b', 'open_b');

    const makeSuccessor = (id: string, agentId: string): t.RunStep => ({
      id,
      agentId,
      type: StepTypes.MESSAGE_CREATION,
      /** Both lanes computed the same stale index before dispatching. */
      index: graph.contentData.length,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: `msg_${id}` },
      },
      usage: null,
      created_at: 1_000,
      status: 'in_progress',
    });
    const successorA = makeSuccessor('next_a', 'agent_a');
    const successorB = makeSuccessor('next_b', 'agent_b');

    const track = (
      graph as unknown as {
        trackDispatchedRunStep: (step: t.RunStep) => Promise<void>;
      }
    ).trackDispatchedRunStep.bind(graph);
    await Promise.all([track(successorA), track(successorB)]);

    expect(successorA.index).not.toBe(successorB.index);
    expect(graph.getRunStep('next_a')).toBe(successorA);
    expect(graph.getRunStep('next_b')).toBe(successorB);
    expect(graph.contentData.filter((s) => s.id === 'next_a')).toHaveLength(1);
    const indexes = graph.contentData.map((step) => step.index);
    expect(new Set(indexes).size).toBe(indexes.length);
  });
  it('stamps created_at after the predecessor closure is delivered', async () => {
    const graph = new StandardGraph({
      runId: 'run_1',
      agents: [makeAgent('agent')],
    });
    let predecessorClosedAt = 0;
    const registry = new HandlerRegistry();
    registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
      handle: async (): Promise<void> => {
        /** A slow predecessor handler must not be charged to the successor. */
        await new Promise((resolve) => setTimeout(resolve, 20));
        predecessorClosedAt = Date.now();
      },
    });
    graph.handlerRegistry = registry;

    seedStep(graph, 'open_a');
    graph.openMessageStepByAgent.set('', 'open_a');

    const successor: t.RunStep = {
      id: 'next_a',
      type: StepTypes.MESSAGE_CREATION,
      index: 0,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'msg_next_a' },
      },
      usage: null,
      status: 'in_progress',
    };
    await (
      graph as unknown as {
        trackDispatchedRunStep: (step: t.RunStep) => Promise<void>;
      }
    ).trackDispatchedRunStep.bind(graph)(successor);

    expect(predecessorClosedAt).toBeGreaterThan(0);
    expect(successor.created_at as number).toBeGreaterThanOrEqual(
      predecessorClosedAt
    );
  });
});

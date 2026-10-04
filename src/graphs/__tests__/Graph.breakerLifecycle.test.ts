import { describe, it, expect, jest } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { StandardGraph } from '../Graph';
import { Providers } from '@/common';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

const makeGraph = (): StandardGraph =>
  new StandardGraph({
    runId: 'run_1',
    agents: [makeAgent('agent')],
  });

const makeTrip = (): StreamLimitExceededError =>
  new StreamLimitExceededError({
    kind: 'tool_call_args',
    limit: 10,
    observed: 11,
    toolName: 'db_query',
  });

type CompiledRegistrationInternals = {
  _compiledToolNodes: Set<{ clearDirectPathTurns(): void }>;
  _subagentExecutors: Set<{
    clearHeavyState(): void;
    resetCheckpointThreadIds(): void;
  }>;
};

describe('run breaker lifecycle', () => {
  it('retains compiled cleanup registrations across graph reuse', () => {
    const graph = makeGraph();
    const internals = graph as unknown as CompiledRegistrationInternals;
    const clearDirectPathTurns = jest.fn();
    const clearSubagentState = jest.fn();
    const resetCheckpointThreadIds = jest.fn();
    internals._compiledToolNodes.add({ clearDirectPathTurns });
    internals._subagentExecutors.add({
      clearHeavyState: clearSubagentState,
      resetCheckpointThreadIds,
    });

    graph.clearHeavyState();
    graph.clearHeavyState();

    expect(clearDirectPathTurns).toHaveBeenCalledTimes(2);
    expect(clearSubagentState).toHaveBeenCalledTimes(2);
    expect(internals._compiledToolNodes.size).toBe(1);
    expect(internals._subagentExecutors.size).toBe(1);
  });

  it('resets subagent checkpoint ownership at each fresh run start', () => {
    const graph = makeGraph();
    const internals = graph as unknown as CompiledRegistrationInternals;
    const resetCheckpointThreadIds = jest.fn();
    internals._subagentExecutors.add({
      clearHeavyState: jest.fn(),
      resetCheckpointThreadIds,
    });

    graph.resetValues();
    graph.resetValues();

    expect(resetCheckpointThreadIds).toHaveBeenCalledTimes(2);
  });

  it('replaces the breaker at every run start, even when un-aborted', () => {
    const graph = makeGraph();
    const previous = graph.breakerAbort;
    expect(previous.signal.aborted).toBe(false);

    graph.resetValues();

    expect(graph.breakerAbort).not.toBe(previous);
    /** A straggler from the failed run trips the controller it captured;
     * the run starting now must not observe it. */
    previous.abort(makeTrip());
    expect(graph.breakerAbort.signal.aborted).toBe(false);
  });

  it('keeps straggler accounting through cleanup and one reset, then sweeps it', () => {
    const graph = makeGraph();
    const epoch = graph.breakerEpoch;
    graph.streamedToolCallArgTallies.set('gen:i:0', { bytes: 42, epoch });
    graph.streamDeltaEventCounts.set('gen', { count: 7, epoch });
    graph.streamLimitChargeCredits = new WeakMap();

    /** Cleanup runs while sibling attempts can still be unwinding on the
     * retained breaker; clearing here would hand a cancellation-ignoring
     * provider's late chunks a fresh budget. */
    graph.clearHeavyState();
    expect(graph.streamedToolCallArgTallies.get('gen:i:0')?.bytes).toBe(42);
    expect(graph.streamDeltaEventCounts.get('gen')?.count).toBe(7);
    expect(graph.streamLimitChargeCredits).toBeDefined();

    /** Producer loops of straggling attempts sit OUTSIDE the consumer-only
     * epoch gate, so their entries must survive the next run start too —
     * one grace reset keeps them on their original budgets. */
    graph.resetValues();
    expect(graph.streamedToolCallArgTallies.get('gen:i:0')?.bytes).toBe(42);
    expect(graph.streamDeltaEventCounts.get('gen')?.count).toBe(7);
    expect(graph.streamLimitChargeCredits).toBeUndefined();

    /** A second reset sweeps entries from older epochs. */
    graph.resetValues();
    expect(graph.streamedToolCallArgTallies.size).toBe(0);
    expect(graph.streamDeltaEventCounts.size).toBe(0);
  });

  it('holds a leased producer straggler to its budget across two run resets', async () => {
    const { attemptInvoke } = await import('@/llm/invoke');
    const { resolveStreamLimits } = await import('@/llm/streamLimits');
    const { AIMessageChunk, HumanMessage } = await import(
      '@langchain/core/messages'
    );
    const graph = makeGraph();
    graph.streamLimits = resolveStreamLimits({ maxToolCallArgBytes: 100 });
    const chunkOf = (args: string): InstanceType<typeof AIMessageChunk> =>
      new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            type: 'tool_call_chunk',
            id: 'call_1',
            name: 'writer',
            args,
            index: 0,
          },
        ],
      });
    let yields = 0;
    const model = {
      stream: async function* (): AsyncGenerator<
        InstanceType<typeof AIMessageChunk>
        > {
        yields += 1;
        yield chunkOf('x'.repeat(60));
        /** Two prompt runs start and reset while this attempt is still
         * draining. Its lease must keep the tally alive — reset counting
         * cannot prove the attempt finished. */
        graph.resetValues();
        graph.resetValues();
        yields += 1;
        yield chunkOf('x'.repeat(60));
        yields += 1;
        yield chunkOf('x');
      },
    };

    await expect(
      attemptInvoke(
        {
          model: model as unknown as t.ChatModel,
          messages: [new HumanMessage('hi')],
          provider: Providers.OPENAI,
          context: graph as Parameters<typeof attemptInvoke>[0]['context'],
          onChunk: () => {
            /* producer loop enforcement is under test */
          },
        },
        { metadata: {} }
      )
    ).rejects.toMatchObject({ kind: 'tool_call_args', limit: 100 });
    expect(yields).toBe(2);
    /** The attempt's finally released its lease and entries. */
    expect(graph.activeStreamLimitGenerations?.size ?? 0).toBe(0);
  });

  it('takes no attempt lease when every guard is disabled', async () => {
    const { attemptInvoke } = await import('@/llm/invoke');
    const { resolveStreamLimits } = await import('@/llm/streamLimits');
    const { AIMessage, HumanMessage } = await import(
      '@langchain/core/messages'
    );
    const graph = makeGraph();
    graph.streamLimits = resolveStreamLimits({
      maxToolCallArgBytes: 0,
      maxToolCallArgBytesByTool: { writer: 0 },
    });
    const model = {
      invoke: async (): Promise<InstanceType<typeof AIMessage>> =>
        new AIMessage({ content: 'ok' }),
    };

    await attemptInvoke(
      {
        model: model as unknown as t.ChatModel,
        messages: [new HumanMessage('hi')],
        provider: Providers.OPENAI,
        context: graph as Parameters<typeof attemptInvoke>[0]['context'],
      },
      { metadata: {} }
    );

    /** The lease only protects accounting entries; with no guard able to
     * fire there are none, and per-attempt bookkeeping must not allocate. */
    expect(graph.activeStreamLimitGenerations).toBeUndefined();
  });

  it('holds a post-reset producer straggler to its original byte budget', async () => {
    const { enforceStreamLimitsForWireChunk } = await import(
      '@/llm/streamLimits'
    );
    const graph = makeGraph();
    graph.streamLimits = {
      maxToolCallArgBytes: 100,
      maxDeltaEventsPerTurn: 0,
      hasEnforceableToolCallArgLimit: true,
    };
    const metadata = {
      langgraph_checkpoint_ns: '',
      langgraph_node: 'agent',
      langgraph_step: 1,
    };
    const chunkOf = (
      args: string
    ): { tool_call_chunks: Array<Record<string, unknown>> } => ({
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args, index: 0 }],
    });

    enforceStreamLimitsForWireChunk({
      graph,
      metadata,
      chunk: chunkOf('x'.repeat(60)) as Parameters<
        typeof enforceStreamLimitsForWireChunk
      >[0]['chunk'],
    });

    /** The next run starts while this producer is still draining. Its next
     * chunk must land on the ORIGINAL 60-byte tally and trip, not open a
     * fresh allowance. */
    graph.resetValues();
    expect(() =>
      enforceStreamLimitsForWireChunk({
        graph,
        metadata,
        chunk: chunkOf('x'.repeat(60)) as Parameters<
          typeof enforceStreamLimitsForWireChunk
        >[0]['chunk'],
      })
    ).toThrow(StreamLimitExceededError);
  });

  it('rejects a model node at entry when the breaker has already tripped', async () => {
    const graph = makeGraph();
    const trip = makeTrip();
    graph.breakerAbort.abort(trip);

    const node = graph.createCallModel('agent');
    await expect(
      node(
        { messages: [] } as unknown as t.AgentSubgraphState,
        {} as RunnableConfig
      )
    ).rejects.toBe(trip);
  });

  it.each(['own', 'parent'])('rejects model entry after a protection trip in the %s breaker', async (scope) => {
    const controller = new AbortController();
    const trip = new ProviderTextProtectionError('blocked');
    const graph = new StandardGraph({ runId: 'protection-entry', agents: [makeAgent('agent')], signal: scope === 'parent' ? controller.signal : undefined });
    if (scope === 'own') graph.breakerAbort.abort(trip);
    else controller.abort(trip);
    await expect(graph.createCallModel('agent')({ messages: [] } as t.AgentSubgraphState, {})).rejects.toBe(trip);
  });

  it('lets a model node proceed past entry when the breaker is live', async () => {
    const graph = makeGraph();
    const node = graph.createCallModel('agent');
    /** With a live breaker the entry guard must not fire; the node then
     * fails later, on the missing config, proving it got past the guard. */
    await expect(
      node({ messages: [] } as unknown as t.AgentSubgraphState, undefined)
    ).rejects.toThrow('No config provided');
  });
});

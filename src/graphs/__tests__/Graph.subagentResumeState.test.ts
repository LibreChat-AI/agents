import { describe, it, expect } from '@jest/globals';
import type {
  SubagentGraphResumeState,
  SubagentToolNodeResumeState,
} from '@/tools/subagent/SubagentReplay';
import { Constants, Providers } from '@/common';
import { StandardGraph } from '../Graph';

type ResumeToolNode = {
  createSubagentResumeState(): SubagentToolNodeResumeState;
  restoreSubagentResumeState(state: SubagentToolNodeResumeState): void;
};

const makeGraph = (
  runId: string,
  codeSessionKey?: string,
  agentIds = ['child-agent']
): StandardGraph =>
  new StandardGraph({
    runId,
    subagentScope: true,
    agents: agentIds.map((agentId) => ({
      agentId,
      codeSessionKey,
      provider: Providers.OPENAI,
      instructions: 'Test child state restoration.',
    })),
  });

const initializeToolNode = (
  graph: StandardGraph,
  agentId = 'child-agent'
): ResumeToolNode =>
  graph.initializeTools({
    currentTools: [],
    agentContext: graph.agentContexts.get(agentId),
  }) as ResumeToolNode;

const legacyKey = (stateKey: string): string =>
  JSON.stringify((JSON.parse(stateKey) as string[]).slice(0, 3));

const legacyState = (
  state: SubagentGraphResumeState
): SubagentGraphResumeState => ({
  ...state,
  toolNodes: state.toolNodes.map((node) => ({
    ...node,
    stateKey: legacyKey(node.stateKey),
  })),
});

const initializePartitions = (
  graph: StandardGraph,
  partitions: Array<string | undefined>
): ResumeToolNode[] =>
  partitions.map((partition) => {
    graph.agentContexts.get('child-agent')!.codeSessionKey = partition;
    return initializeToolNode(graph);
  });

const initializePartitionState = (
  graph: StandardGraph,
  partitions: Array<string | undefined>
): void => {
  initializePartitions(graph, partitions).forEach((node, index) => {
    node.restoreSubagentResumeState({
      ...node.createSubagentResumeState(),
      toolUsageCounts: [
        { toolName: `${graph.runId}-tool-${index}`, count: index + 1 },
      ],
      directPathTurns: [
        { toolCallId: `${graph.runId}-call-${index}`, turn: index },
      ],
    });
    graph.sessions.set(partitions[index] ?? Constants.EXECUTE_CODE, {
      session_id: `${graph.runId}-session-${index}`,
      lastUpdated: index,
    });
  });
};

const expectRejectedWithoutMutation = (
  target: StandardGraph,
  state: SubagentGraphResumeState,
  expectedError = 'Cannot restore'
): void => {
  target.sessions.set('sentinel', {
    session_id: 'unchanged',
    lastUpdated: 2,
  });
  target.toolCallStepIds.set('existing', 'unchanged-step');
  target.getEagerEventToolUsageCount('child-agent').set('sentinel', 7);
  target.eagerEventToolSuppressions.add('sentinel');
  target.toolOutputReferences = { enabled: true };
  target
    .getOrCreateToolOutputRegistry()
    ?.set('target-scope', 'tool0turn0', 'unchanged output');
  const before = target.createSubagentResumeState('target-scope');
  const savedStateBefore = JSON.stringify(state);
  expect(() =>
    target.restoreSubagentResumeState(state, 'target-scope')
  ).toThrow(expectedError);
  expect(target.createSubagentResumeState('target-scope')).toEqual(before);
  expect(JSON.stringify(state)).toBe(savedStateBefore);
};

describe('subagent graph resume state', () => {
  describe.each(['legacy', 'current'])('%s identities', (format) => {
    const checkpoint = (graph: StandardGraph): SubagentGraphResumeState => {
      const state = graph.createSubagentResumeState('source-scope');
      return format === 'legacy' ? legacyState(state) : state;
    };

    it.each([undefined, Constants.EXECUTE_CODE, 'partition-A'])(
      'restores identity, turns, references, and sessions for %s',
      (codeSessionKey) => {
        const source = makeGraph('source', codeSessionKey);
        source.toolOutputReferences = { enabled: true };
        const sourceNode = initializeToolNode(source);
        const sessionKey = codeSessionKey ?? Constants.EXECUTE_CODE;
        source.toolCallStepIds.set('call_tool', 'step_tool');
        source.sessions.set(sessionKey, {
          session_id: 'sandbox-session',
          lastUpdated: 42,
        });
        source
          .getOrCreateToolOutputRegistry()
          ?.set('source-scope', 'tool0turn0', 'raw output');
        source.getEagerEventToolUsageCount('child-agent').set('calculator', 3);
        source.eagerEventToolSuppressions.add('unstable_search');
        sourceNode.restoreSubagentResumeState({
          ...sourceNode.createSubagentResumeState(),
          toolUsageCounts: [{ toolName: 'calculator', count: 3 }],
          directPathTurns: [{ toolCallId: 'call_tool', turn: 2 }],
        });
        const savedState = checkpoint(source);
        const serialized = JSON.stringify(savedState);

        const target = makeGraph('target', codeSessionKey);
        target.toolOutputReferences = { enabled: true };
        const targetNode = initializeToolNode(target);
        const restoredState = JSON.parse(serialized) as SubagentGraphResumeState;
        target.restoreSubagentResumeState(restoredState, 'target-scope');

        expect(target.createSubagentResumeState('target-scope')).toEqual(
          source.createSubagentResumeState('source-scope')
        );
        expect(targetNode.createSubagentResumeState()).toEqual(
          sourceNode.createSubagentResumeState()
        );
        expect(target.sessions.get(sessionKey)?.session_id).toBe(
          'sandbox-session'
        );
        expect(target.sessions.size).toBe(1);
        expect(JSON.stringify(restoredState)).toBe(serialized);
      }
    );

    it('matches multiple agents without relying on checkpoint order', () => {
      const agentIds = ['child-agent', 'other-agent'];
      const source = makeGraph('source', 'partition-A', agentIds);
      const target = makeGraph('target', 'partition-A', agentIds);
      for (const agentId of agentIds) {
        const sourceNode = initializeToolNode(source, agentId);
        sourceNode.restoreSubagentResumeState({
          ...sourceNode.createSubagentResumeState(),
          toolUsageCounts: [{ toolName: agentId, count: 3 }],
        });
        initializeToolNode(target, agentId);
      }
      const state = checkpoint(source);
      state.toolNodes.reverse();
      target.restoreSubagentResumeState(state, 'target-scope');
      expect(target.createSubagentResumeState('target-scope')).toEqual(
        source.createSubagentResumeState('source-scope')
      );
    });

    it.each(['agent', 'scope', 'name', 'added node', 'removed node'])(
      'rejects changed topology: %s',
      (change) => {
        const source = makeGraph('source', 'partition-A');
        initializeToolNode(source);
        const target = makeGraph(
          'target',
          'partition-A',
          change === 'agent' ? ['other-agent'] : ['child-agent']
        );
        if (change === 'scope') target.subagentScope = false;
        if (change !== 'removed node') {
          const node = target.initializeTools({
            currentTools: [],
            agentContext: [...target.agentContexts.values()][0],
          });
          if (change === 'name') node.name = 'changed-tools';
        }
        if (change === 'added node') {
          initializePartitions(target, ['partition-B']);
        }
        expectRejectedWithoutMutation(target, checkpoint(source));
      }
    );

    it('rejects duplicate saved identities', () => {
      const agentIds = ['child-agent', 'other-agent'];
      const source = makeGraph('source', 'partition-A', agentIds);
      const target = makeGraph('target', 'partition-A', agentIds);
      for (const agentId of agentIds) {
        initializeToolNode(source, agentId);
        initializeToolNode(target, agentId);
      }
      const state = checkpoint(source);
      state.toolNodes[1] = { ...state.toolNodes[0] };
      expectRejectedWithoutMutation(target, state);
    });
  });

  it('keeps the legacy default partition identity', () => {
    const implicit = initializeToolNode(makeGraph('implicit'));
    const explicit = initializeToolNode(
      makeGraph('explicit', Constants.EXECUTE_CODE)
    );
    expect(explicit.createSubagentResumeState().stateKey).toBe(
      implicit.createSubagentResumeState().stateKey
    );
    expect(
      JSON.parse(implicit.createSubagentResumeState().stateKey)
    ).toHaveLength(3);
  });

  it.each([undefined, 'partition-B'])(
    'rejects modern partition-A restoration into %s',
    (codeSessionKey) => {
      const source = makeGraph('source', 'partition-A');
      initializeToolNode(source);
      source.sessions.set('partition-A', {
        session_id: 'session-A',
        lastUpdated: 1,
      });
      const target = makeGraph('target', codeSessionKey);
      initializeToolNode(target);
      expectRejectedWithoutMutation(
        target,
        source.createSubagentResumeState('source-scope')
      );
    }
  );

  describe.each([false, true])('reversed saved order: %s', (reversed) => {
    it.each([
      {
        description: 'current default/custom',
        partitions: [undefined, 'partition-A'],
        legacyIndex: -1,
      },
      {
        description: 'current default/custom with JSON whitespace',
        partitions: [undefined, 'partition-A'],
        legacyIndex: -1,
        formattedKeys: true,
      },
      {
        description: 'current default/multiple custom',
        partitions: [undefined, 'partition-A', 'partition-B'],
        legacyIndex: -1,
      },
      {
        description: 'current custom/custom',
        partitions: ['partition-A', 'partition-B'],
        legacyIndex: -1,
      },
      {
        description: 'mixed exact/legacy',
        partitions: ['partition-A', 'partition-B'],
        legacyIndex: 0,
      },
      {
        description: 'multiple exact/one legacy remainder',
        partitions: ['partition-A', 'partition-B', 'partition-C'],
        legacyIndex: 0,
      },
    ])('restores $description identities', ({ partitions, legacyIndex, formattedKeys }) => {
      const source = makeGraph('source');
      const target = makeGraph('target');
      initializePartitionState(source, partitions);
      initializePartitionState(target, partitions);
      const state = source.createSubagentResumeState('source-scope');
      if (legacyIndex >= 0) {
        state.toolNodes[legacyIndex].stateKey = legacyKey(
          state.toolNodes[legacyIndex].stateKey
        );
      }
      if (formattedKeys === true) {
        for (const nodeState of state.toolNodes) {
          nodeState.stateKey = JSON.stringify(JSON.parse(nodeState.stateKey), null, 2);
        }
      }
      if (reversed) state.toolNodes.reverse();
      const serialized = JSON.stringify(state);
      const restoredState = JSON.parse(serialized) as SubagentGraphResumeState;
      target.restoreSubagentResumeState(restoredState, 'target-scope');
      expect(target.createSubagentResumeState('target-scope')).toEqual(
        source.createSubagentResumeState('source-scope')
      );
      expect(JSON.stringify(restoredState)).toBe(serialized);
    });

    it.each([
      ['legacy', 'compact'],
      ['legacy', 'whitespace'],
      ['legacy', 'escaped'],
      ['modern', 'compact'],
      ['modern', 'whitespace'],
      ['modern', 'escaped'],
    ])(
      'rejects duplicate canonical %s identities (%s)',
      (format, encoding) => {
        const source = makeGraph('source');
        const target = makeGraph('target');
        const partitions = format === 'legacy'
          ? [undefined, 'partition-A']
          : ['partition-A', 'partition-B'];
        initializePartitionState(source, partitions);
        initializePartitionState(target, partitions);
        const checkpoint = source.createSubagentResumeState('source-scope');
        const state = format === 'legacy' ? legacyState(checkpoint) : checkpoint;
        state.toolNodes[1].stateKey = state.toolNodes[0].stateKey;
        const stateKey = state.toolNodes[1].stateKey;
        if (encoding === 'whitespace') {
          state.toolNodes[1].stateKey = JSON.stringify(JSON.parse(stateKey), null, 2);
        }
        if (encoding === 'escaped') {
          state.toolNodes[1].stateKey = stateKey.replace('child-agent', '\\u0063hild-agent');
        }
        if (encoding !== 'compact') {
          expect(state.toolNodes[0].stateKey).not.toBe(state.toolNodes[1].stateKey);
          expect(JSON.parse(state.toolNodes[0].stateKey)).toEqual(
            JSON.parse(state.toolNodes[1].stateKey)
          );
        }
        if (reversed) state.toolNodes.reverse();
        expectRejectedWithoutMutation(target, state);
      }
    );

    it('rejects a changed modern partition after reserving the default identity', () => {
      const source = makeGraph('source');
      const target = makeGraph('target');
      initializePartitionState(source, [undefined, 'partition-A']);
      initializePartitionState(target, [undefined, 'partition-B']);
      const state = source.createSubagentResumeState('source-scope');
      if (reversed) state.toolNodes.reverse();
      expectRejectedWithoutMutation(target, state);
    });
  });

  it('rejects a legacy identity with multiple remaining live candidates', () => {
    const source = makeGraph('source', 'partition-A', ['child-agent', 'removed-agent']);
    initializeToolNode(source);
    initializeToolNode(source, 'removed-agent');
    const target = makeGraph('target');
    initializePartitionState(target, ['partition-A', 'partition-B']);
    const state = legacyState(source.createSubagentResumeState('source-scope'));
    expectRejectedWithoutMutation(
      target,
      state,
      `Cannot restore subagent tool state for "${state.toolNodes[0].stateKey}".`
    );
  });

  it.each([false, true])(
    'rejects a legacy/modern collision (legacy first: %s)',
    (legacyFirst) => {
      const agentIds = ['child-agent', 'other-agent'];
      const source = makeGraph('source', 'partition-A', agentIds);
      const target = makeGraph('target', 'partition-A', agentIds);
      for (const agentId of agentIds) {
        const node = initializeToolNode(source, agentId);
        node.restoreSubagentResumeState({
          ...node.createSubagentResumeState(),
          directPathTurns: [{ toolCallId: agentId, turn: 4 }],
        });
        initializeToolNode(target, agentId);
      }
      const state = source.createSubagentResumeState('source-scope');
      state.toolNodes[1].stateKey = legacyKey(state.toolNodes[0].stateKey);
      if (legacyFirst) state.toolNodes.reverse();
      expectRejectedWithoutMutation(target, state);
    }
  );

  it.each([undefined, 'partition-A'])(
    'rejects duplicate live identities in partition %s',
    (codeSessionKey) => {
      const source = makeGraph('source', codeSessionKey);
      initializeToolNode(source);
      const target = makeGraph('target', codeSessionKey);
      initializeToolNode(target);
      initializeToolNode(target);
      expectRejectedWithoutMutation(
        target,
        source.createSubagentResumeState('source-scope')
      );
    }
  );

  const malformedKeys: Record<string, (identity: string[]) => string> = {
    'invalid JSON': () => 'not json',
    null: () => 'null',
    object: () => '{}',
    string: () => '"identity"',
    'empty array': () => '[]',
    'missing name': (identity) => JSON.stringify(identity.slice(0, 2)),
    'null name': (identity) => JSON.stringify([...identity.slice(0, 2), null]),
    'numeric name': (identity) => JSON.stringify([...identity.slice(0, 2), 7]),
    'null partition': (identity) => JSON.stringify([...identity, null]),
    'object partition': (identity) => JSON.stringify([...identity, {}]),
    'extra element': (identity) =>
      JSON.stringify([...identity, 'partition-A', 'extra']),
  };

  it.each(Object.entries(malformedKeys))(
    'rejects malformed saved key (%s) after a valid match',
    (_description, createKey) => {
      const agentIds = ['child-agent', 'other-agent'];
      const source = makeGraph('source', 'partition-A', agentIds);
      const target = makeGraph('target', 'partition-A', agentIds);
      for (const agentId of agentIds) {
        const node = initializeToolNode(source, agentId);
        node.restoreSubagentResumeState({
          ...node.createSubagentResumeState(),
          toolUsageCounts: [{ toolName: 'calculator', count: 3 }],
          directPathTurns: [{ toolCallId: 'call_tool', turn: 2 }],
        });
        initializeToolNode(target, agentId);
      }
      const state = legacyState(source.createSubagentResumeState('source-scope'));
      state.toolNodes[1].stateKey = createKey(
        JSON.parse(state.toolNodes[1].stateKey) as string[]
      );
      expectRejectedWithoutMutation(target, state);
    }
  );
});

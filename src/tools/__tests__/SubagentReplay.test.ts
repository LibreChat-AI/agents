import {
  getSubagentResumeManifest,
  attachSubagentResumeManifest,
  requireValidSubagentResumeManifest,
  stripSubagentResumeManifest,
} from '@/tools/subagent/SubagentReplay';
import {
  attachRunStepResumeState,
  getRunStepResumeState,
  stripRunStepResumeState,
} from '@/tools/runStepResume';
import type { RunStepResumeState } from '@/types';
import { ToolMessage } from '@langchain/core/messages';
import {
  attachToolBatchReplayState,
  getPublicToolInterruptPayload,
  restoreToolBatchReplayState,
  restoreToolReplayConfig,
  TOOL_BATCH_REPLAY_KEY,
} from '@/tools/toolBatchReplay';

describe('SubagentReplay manifest', () => {
  const execution = {
    parentToolCallId: 'call_parent',
    childRunId: 'child-run',
    subagentType: 'researcher',
    configId: 'researcher@v1',
    approvalExecutionScope: 'child-approval-scope',
    checkpoints: [
      {
        threadId: 'child-thread',
        checkpointId: 'checkpoint-root',
        checkpointNs: '',
      },
      {
        threadId: 'child-thread',
        checkpointId: 'checkpoint-agent',
        checkpointNs: 'agent:task',
      },
    ],
    graphState: {
      toolCallSteps: [{ toolCallId: 'call_tool', stepId: 'step_tool' }],
      toolSessions: [
        {
          toolName: 'execute_code',
          context: { session_id: 'session', lastUpdated: 1 },
        },
      ],
      toolNodes: [
        {
          stateKey: 'child-agent',
          toolUsageCounts: [{ toolName: 'calculator', count: 1 }],
          directPathTurns: [{ toolCallId: 'call_tool', turn: 0 }],
        },
      ],
      eagerToolUsage: [
        {
          agentId: 'child-agent',
          toolUsageCounts: [{ toolName: 'calculator', count: 1 }],
        },
      ],
      eagerToolSuppressions: ['unstable_search'],
      toolOutputReferences: {
        entries: [{ key: 'tool0turn0', value: '42' }],
        turnCounter: 1,
        warnedNonStringTools: [],
      },
    },
    approvalReplays: [
      {
        key: {
          executionScope: 'child-approval-scope',
          agentId: 'child-agent',
          toolUseId: 'call_tool',
        },
        result: {
          decision: 'ask' as const,
          reason: 'review tool',
          additionalContexts: [],
          injectedMessages: [],
          errors: [],
        },
      },
    ],
  };

  it.each([null, 'confirm', ['a', 'b']])(
    'preserves nested custom payload %j with parent replay state',
    async (payload) => {
      const nested = attachSubagentResumeManifest(payload, {
        version: 1,
        executions: [execution],
      });
      const wrapped = await attachToolBatchReplayState(
        nested,
        'parent',
        new Map([['batch', new Map()]])
      );
      const persisted = JSON.parse(JSON.stringify(wrapped));
      expect(getPublicToolInterruptPayload(persisted)).toEqual(payload);
      const config: Record<string, unknown> = {};
      restoreToolReplayConfig(config, 'interrupt', persisted);
      expect(config[TOOL_BATCH_REPLAY_KEY]).toMatchObject({
        records: [{ owner: 'parent' }],
      });
    }
  );

  it('round-trips code-session baselines with settled direct results', async () => {
    const result = {
      proposal: { name: 'execute_code', args: { code: 'remove()' } },
      output: new ToolMessage({
        content: 'removed data.csv',
        tool_call_id: 'call_delete',
      }),
      additionalContexts: [],
      codeSessionBaseline: [['data.csv', 'storage\0file']] as Array<
        [string, string]
      >,
    };
    const payload = await attachToolBatchReplayState(
      null,
      'parent',
      new Map([['batch', new Map([['call_delete', result]])]])
    );
    const config: Record<string, unknown> = {};
    restoreToolReplayConfig(
      config,
      'interrupt',
      JSON.parse(JSON.stringify(payload))
    );

    const restored = await restoreToolBatchReplayState(
      { configurable: config },
      'parent'
    );

    expect(restored[0]?.results[0]?.[1].codeSessionBaseline).toEqual(
      result.codeSessionBaseline
    );
  });

  it('round-trips a private manifest without exposing it publicly', () => {
    const manifest = { version: 1 as const, executions: [execution] };
    const payload = attachSubagentResumeManifest(
      { type: 'tool_approval', visible: true },
      manifest
    );

    expect(getSubagentResumeManifest(payload)).toEqual(manifest);
    expect(stripSubagentResumeManifest(payload)).toEqual({
      type: 'tool_approval',
      visible: true,
    });
  });

  it('composes with private run-step state without changing the public payload', () => {
    const manifest = { version: 1 as const, executions: [execution] };
    const runStepState: RunStepResumeState = {
      version: 1,
      revision: 1,
      nextIndex: 0,
      toolCallSteps: [],
      steps: [],
    };
    const visiblePayload = {
      type: 'tool_approval',
      subagent: { parent_tool_call_id: 'call_parent' },
    };
    const payload = attachSubagentResumeManifest(
      attachRunStepResumeState(visiblePayload, runStepState),
      manifest
    );
    const manifestPayload = stripRunStepResumeState(payload);

    expect(getRunStepResumeState(payload)).toEqual(runStepState);
    expect(getSubagentResumeManifest(manifestPayload)).toEqual(manifest);
    expect(stripSubagentResumeManifest(manifestPayload)).toEqual(
      visiblePayload
    );
  });

  it('accepts legacy executions without a bound subagent type', () => {
    const { subagentType: _subagentType, ...legacyExecution } = execution;
    const manifest = { version: 1 as const, executions: [legacyExecution] };
    const payload = attachSubagentResumeManifest(
      { type: 'tool_approval' },
      manifest
    );

    expect(getSubagentResumeManifest(payload)).toEqual(manifest);
  });

  it.each(['', 42, { type: 'researcher' }])(
    'rejects malformed bound subagent type %#',
    (subagentType) => {
      const manifest = {
        version: 1 as const,
        executions: [{ ...execution, subagentType }],
      };

      expect(
        getSubagentResumeManifest({
          __librechat_subagent_resume_manifest: manifest,
        })
      ).toBeUndefined();
    }
  );

  it.each([
    ['string', 'approve this child'],
    ['number', 42],
    ['null', null],
    ['array', ['approve', { call: 1 }]],
  ])(
    'round-trips a private manifest around a %s interrupt payload',
    (_label, visiblePayload) => {
      const manifest = { version: 1 as const, executions: [execution] };
      const payload = attachSubagentResumeManifest(visiblePayload, manifest);

      expect(getSubagentResumeManifest(payload)).toEqual(manifest);
      expect(stripSubagentResumeManifest(payload)).toEqual(visiblePayload);
    }
  );

  it.each([
    ['Map', new Map([['decision', 'approve']])],
    ['Set', new Set(['approve'])],
    ['RegExp', /^approve$/i],
  ])(
    'round-trips a private manifest around a %s interrupt payload',
    (_label, visiblePayload) => {
      const manifest = { version: 1 as const, executions: [execution] };
      const payload = attachSubagentResumeManifest(visiblePayload, manifest);

      expect(getSubagentResumeManifest(payload)).toEqual(manifest);
      expect(stripSubagentResumeManifest(payload)).toBe(visiblePayload);
    }
  );

  it.each([
    {
      __librechat_subagent_resume_manifest: { visible: true },
      value: 'manifest collision',
    },
    {
      __librechat_subagent_resume_wrapper: 1,
      __librechat_subagent_resume_payload: 'visible nested value',
      value: 'wrapper collision',
    },
  ])(
    'preserves a custom payload containing private transport keys',
    (value) => {
      const manifest = { version: 1 as const, executions: [execution] };
      const payload = attachSubagentResumeManifest(value, manifest);

      expect(stripSubagentResumeManifest(value)).toBe(value);
      expect(getSubagentResumeManifest(payload)).toEqual(manifest);
      expect(stripSubagentResumeManifest(payload)).toBe(value);
    }
  );

  it('updates the manifest on an existing inline payload', () => {
    const firstManifest = { version: 1 as const, executions: [execution] };
    const nextManifest = {
      version: 1 as const,
      executions: [{ ...execution, childRunId: 'next-inline-child-run' }],
    };
    const firstPayload = attachSubagentResumeManifest(
      { type: 'tool_approval', visible: true },
      firstManifest
    );
    const nextPayload = attachSubagentResumeManifest(
      firstPayload,
      nextManifest
    );

    expect(getSubagentResumeManifest(nextPayload)).toEqual(nextManifest);
    expect(stripSubagentResumeManifest(nextPayload)).toEqual({
      type: 'tool_approval',
      visible: true,
    });
  });

  it('updates the manifest on an existing private wrapper', () => {
    const firstManifest = { version: 1 as const, executions: [execution] };
    const nextManifest = {
      version: 1 as const,
      executions: [{ ...execution, childRunId: 'next-child-run' }],
    };
    const firstPayload = attachSubagentResumeManifest(
      'visible payload',
      firstManifest
    );
    const nextPayload = attachSubagentResumeManifest(
      firstPayload,
      nextManifest
    );

    expect(getSubagentResumeManifest(nextPayload)).toEqual(nextManifest);
    expect(stripSubagentResumeManifest(nextPayload)).toBe('visible payload');
  });

  it('rejects ambiguous or cross-execution replay data', () => {
    const mismatchedScope = {
      ...execution,
      approvalReplays: [
        {
          ...execution.approvalReplays[0],
          key: {
            ...execution.approvalReplays[0].key,
            executionScope: 'different-child',
          },
        },
      ],
    };
    const duplicateParent = {
      version: 1 as const,
      executions: [execution, { ...execution }],
    };
    const duplicateToolCall = {
      ...execution,
      graphState: {
        ...execution.graphState,
        toolCallSteps: [
          ...execution.graphState.toolCallSteps,
          { ...execution.graphState.toolCallSteps[0] },
        ],
      },
    };

    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest(
          { type: 'tool_approval' },
          { version: 1, executions: [mismatchedScope] }
        )
      )
    ).toBeUndefined();
    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest({ type: 'tool_approval' }, duplicateParent)
      )
    ).toBeUndefined();
    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest(
          { type: 'tool_approval' },
          { version: 1, executions: [duplicateToolCall] }
        )
      )
    ).toBeUndefined();
  });

  it('keeps a manifest carrying a pre-v2 fading tier resumable', () => {
    const legacyTier = {
      v: 1,
      budgetTokens: 6_250,
      masked: false,
      latched: true,
    };
    const manifest = {
      __librechat_subagent_resume_manifest: {
        version: 1,
        executions: [
          {
            ...execution,
            graphState: {
              ...execution.graphState,
              fadingTier: legacyTier,
              fadingTiers: { 'child-agent': legacyTier },
            },
          },
        ],
      },
    };

    expect(() => requireValidSubagentResumeManifest(manifest)).not.toThrow();
    expect(getSubagentResumeManifest(manifest)).toBeDefined();
    expect(
      getSubagentResumeManifest({
        __librechat_subagent_resume_manifest: {
          version: 1,
          executions: [
            {
              ...execution,
              graphState: {
                ...execution.graphState,
                fadingTier: { v: 1, budgetTokens: 0, masked: false },
              },
            },
          ],
        },
      })
    ).toBeUndefined();
  });

  it('fails closed for malformed nested references', () => {
    const malformed = {
      __librechat_subagent_resume_manifest: {
        version: 1,
        executions: [{ ...execution, checkpoints: [null] }],
      },
    };
    expect(getSubagentResumeManifest(malformed)).toBeUndefined();
    expect(() => requireValidSubagentResumeManifest(malformed)).toThrow(
      'Invalid subagent resume manifest.'
    );
    expect(
      getSubagentResumeManifest({
        __librechat_subagent_resume_manifest: {
          version: 1,
          executions: [
            {
              ...execution,
              graphState: {
                ...execution.graphState,
                toolCallSteps: [null],
              },
            },
          ],
        },
      })
    ).toBeUndefined();
  });

  it('rejects duplicate checkpoint namespaces and approval replay keys', () => {
    const duplicateCheckpointNamespace = {
      ...execution,
      checkpoints: [execution.checkpoints[0], { ...execution.checkpoints[0] }],
    };
    const duplicateApprovalReplay = {
      ...execution,
      approvalReplays: [
        execution.approvalReplays[0],
        { ...execution.approvalReplays[0] },
      ],
    };

    for (const invalidExecution of [
      duplicateCheckpointNamespace,
      duplicateApprovalReplay,
    ]) {
      expect(
        getSubagentResumeManifest(
          attachSubagentResumeManifest(
            { type: 'tool_approval' },
            { version: 1, executions: [invalidExecution] }
          )
        )
      ).toBeUndefined();
    }
  });

  it('rejects cyclic descendant manifests', () => {
    const cyclic: {
      version: number;
      executions: Array<typeof execution & { descendant?: object }>;
    } = { version: 1, executions: [{ ...execution }] };
    cyclic.executions[0].descendant = cyclic;

    expect(
      getSubagentResumeManifest({
        __librechat_subagent_resume_manifest: cyclic,
      })
    ).toBeUndefined();
  });
});

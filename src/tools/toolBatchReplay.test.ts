import { Command } from '@langchain/langgraph';
import { ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { SubagentResumeManifest } from '@/tools/subagent/SubagentReplay';
import {
  TOOL_BATCH_REPLAY_KEY,
  TOOL_REPLAY_CONFIGS_KEY,
  getToolReplayConfig,
  restoreToolReplayConfigs,
  attachToolBatchReplayState,
  clearStaleToolApprovalConfig,
  getToolBatchReplayOwner,
  getToolReplayResumeStatus,
  getToolBatchReplayState,
  restoreToolBatchReplayState,
  stripToolBatchReplayState,
  restoreToolReplayConfig,
  getToolBatchReplayScope,
  getPublicToolInterruptPayload,
  rebindToolBatchReplayPayload,
} from './toolBatchReplay';
import { attachSubagentResumeManifest, SUBAGENT_RESUME_MANIFEST_CONFIG_KEY } from '@/tools/subagent/SubagentReplay';
import { getToolApprovalReviewEvidence } from '@/hitl/approvalReview';
import { attachRunStepResumeState } from '@/tools/runStepResume';

const approval = {
  type: 'tool_approval',
  action_requests: [{ tool_call_id: 'call', name: 'echo', arguments: {} }],
  review_configs: [
    {
      tool_call_id: 'call',
      action_name: 'echo',
      allowed_decisions: ['approve'],
    },
  ],
};

describe('checkpoint-owned tool batch replay', () => {
  it.each([false, true])('routes each checkpoint owner independently across a fork, question=%s', async question => {
    const interrupts = await Promise.all(['left', 'right'].map(async member => ({
      id: `${member}-interrupt`,
      value: await attachToolBatchReplayState(
        question ? { type: 'ask_user_question', question: { question: 'Continue?' } } : approval,
        JSON.stringify(['source', '', member, 'user', 'source-thread']),
        new Map([[JSON.stringify(['source', member]), new Map()]]),
      ),
    })));
    const config: RunnableConfig = { configurable: {
      user_id: 'user', thread_id: 'child-thread',
      __pregel_scratchpad: { resume: [] },
      __librechat_tool_approval_execution_scope: 'fork',
    } };
    restoreToolReplayConfigs(config.configurable!, interrupts, 'source', 'fork', 'child-thread', new Set(interrupts.map(entry => entry.id)));
    const original = config.configurable?.[TOOL_BATCH_REPLAY_KEY];
    for (const member of ['left', 'right']) {
      const owner = getToolBatchReplayOwner(config, member);
      const selected = getToolReplayConfig(config, owner);
      expect(selected).not.toBe(config);
      expect(selected.configurable?.[TOOL_REPLAY_CONFIGS_KEY]).toBeUndefined();
      expect(getToolBatchReplayState(selected.configurable)?.interruptId).toBe(`${member}-interrupt`);
      expect((await restoreToolBatchReplayState(selected, owner))[0].batch).toBe(JSON.stringify(['fork', member]));
      if (!question) expect(getToolApprovalReviewEvidence(selected)?.owner).toBe(owner);
      const foreign = {
        ...config,
        configurable: { ...config.configurable, user_id: 'different-user' },
      };
      const neutral = getToolReplayConfig(foreign, getToolBatchReplayOwner(foreign, member));
      expect(neutral.configurable?.[TOOL_REPLAY_CONFIGS_KEY]).toBeUndefined();
      expect(neutral.configurable?.[TOOL_BATCH_REPLAY_KEY]).toBeUndefined();
      expect(getToolApprovalReviewEvidence(neutral)).toBeUndefined();
      expect(neutral.configurable?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]).toBeUndefined();
    }
    expect(config.configurable?.[TOOL_BATCH_REPLAY_KEY]).toBe(original);
    expect(getToolBatchReplayState(interrupts[0].value)?.records[0].owner).toBe(JSON.stringify(['source', '', 'left', 'user', 'source-thread']));
  });

  it.each([false, true].flatMap(question => [false, true].map(resumeRight => ({ question, resumeRight }))))('restores unaddressed settled records without approval authority: %j', async ({ question, resumeRight }) => {
    const payload = question ? { type: 'ask_user_question', question: { question: 'Continue?' } } : approval;
    const pending = await Promise.all(['left', 'right'].map(async member => ({
      id: `${member}-interrupt`,
      value: await attachToolBatchReplayState(payload,
        JSON.stringify(['source', '', member, 'user', 'source-thread']),
        new Map([[JSON.stringify(['source', member]), new Map([['shared-effect', {
          proposal: { name: `${member}_effect`, args: {} },
          output: new ToolMessage({ content: `${member} completed`, tool_call_id: 'shared-effect' }),
          additionalContexts: [],
        }]])]]),
      ),
    })));
    const config: RunnableConfig = { configurable: { user_id: 'user', thread_id: 'child-thread', __librechat_tool_approval_execution_scope: 'fork' } };
    restoreToolReplayConfigs(config.configurable!, pending, 'source', 'fork', 'child-thread', new Set(resumeRight ? ['right-interrupt'] : []));
    const leftOwner = getToolBatchReplayOwner(config, 'left');
    const left = getToolReplayConfig(config, leftOwner);
    expect(getToolApprovalReviewEvidence(left)).toBeUndefined();
    expect(getToolBatchReplayState(left.configurable)?.approvalOwner).toBeUndefined();
    expect((await restoreToolBatchReplayState(left, leftOwner))[0].results[0][1].output).toMatchObject({ content: 'left completed' });
    expect(getToolApprovalReviewEvidence(config)?.interruptId).toBe(question || !resumeRight ? undefined : 'right-interrupt');
    const right = getToolReplayConfig(config, getToolBatchReplayOwner(config, 'right'));
    expect(getToolApprovalReviewEvidence(right)?.interruptId).toBe(question || !resumeRight ? undefined : 'right-interrupt');
    expect(getToolBatchReplayState(pending[0].value)?.approvalOwner).toBe(question ? undefined : JSON.stringify(['source', '', 'left', 'user', 'source-thread']));
  });

  it.each([false, true].flatMap(questionFirst => ['approval', 'question', 'both'].map(resumed => ({ questionFirst, resumed }))))('isolates ownerless question evidence without leaking decisions: %j', async ({ questionFirst, resumed }) => {
    const approve = { id: 'approval', value: await attachToolBatchReplayState(approval, JSON.stringify(['source', '', 'left', 'user', 'source-thread']), new Map()) };
    const question = { id: 'question', value: { type: 'ask_user_question', question: { question: 'Explain?' } } };
    const config: RunnableConfig = { configurable: { user_id: 'user', thread_id: 'child-thread', __librechat_tool_approval_execution_scope: 'fork' } };
    restoreToolReplayConfigs(config.configurable!, questionFirst ? [question, approve] : [approve, question], 'source', 'fork', 'child-thread', new Set(resumed === 'both' ? ['approval', 'question'] : [resumed]));
    const selectedApproval = getToolReplayConfig(config, getToolBatchReplayOwner(config, 'left'));
    const selectedQuestion = getToolReplayConfig(config, getToolBatchReplayOwner(config, 'right'));
    expect(getToolApprovalReviewEvidence(selectedApproval)?.interruptId).toBe(resumed === 'question' ? undefined : 'approval');
    expect(getToolApprovalReviewEvidence(selectedQuestion)).toBeUndefined();
    expect(getToolBatchReplayState(selectedQuestion.configurable)).toBeUndefined();
    const foreign = { configurable: { ...config.configurable, user_id: 'different-user' } };
    expect(() => getToolReplayConfig(foreign, getToolBatchReplayOwner(foreign, 'left'))).toThrow('execution identity changed');
    const foreignThread = { configurable: { ...config.configurable, thread_id: 'other-thread' } };
    expect(() => getToolReplayConfig(foreignThread, getToolBatchReplayOwner(foreignThread, 'right'))).toThrow('execution identity changed');
  });

  it.each([false, true].flatMap(reverse => [false, true].flatMap(nested => ['left', 'right', 'both'].map(addressed => ({ reverse, nested, addressed })))))(
    'merges inherited parent records without conflating question members: %j',
    async ({ reverse, nested, addressed }) => {
      const parentOwner = JSON.stringify([nested ? 'middle-scope' : 'root-scope', 'root-ns', 'parent', 'user', 'thread']);
      const manifest: SubagentResumeManifest = { version: 1, executions: [{
        parentToolCallId: 'spawn', childRunId: 'child-run', approvalExecutionScope: 'child-scope',
        checkpoints: [{ threadId: 'child-thread', checkpointNs: '', checkpointId: 'checkpoint' }],
        graphState: { toolCallSteps: [], toolSessions: [], toolNodes: [], eagerToolUsage: [], eagerToolSuppressions: [] },
        approvalReplays: [],
      }] };
      if (nested) {
        manifest.executions = [{
          ...manifest.executions[0],
          childRunId: 'middle-run',
          approvalExecutionScope: 'middle-scope',
          descendant: { version: 1, executions: [{ ...manifest.executions[0] }] },
        }];
      }
      const pending = await Promise.all(['left', 'right'].map(async member => {
        const leaf = await attachToolBatchReplayState(
          { type: 'ask_user_question', question: { question: member } },
          JSON.stringify(['child-scope', '', member, 'user', 'child-thread']),
          new Map([[JSON.stringify(['child-scope', 'child-batch']), new Map()]]),
        );
        return { id: member, value: await attachToolBatchReplayState(
          attachSubagentResumeManifest(leaf, manifest), parentOwner,
          new Map([[JSON.stringify(['root-scope', 'parent-batch']), new Map()]]),
        ) };
      }));
      const config: RunnableConfig = { configurable: { user_id: 'user', thread_id: 'thread', __librechat_tool_approval_execution_scope: 'root-scope' } };
      restoreToolReplayConfigs(config.configurable!, reverse ? [...pending].reverse() : pending,
        'root-scope', 'root-scope', 'thread', new Set(addressed === 'both' ? ['left', 'right'] : [addressed]));
      const restoredParent = getToolReplayConfig(config, parentOwner);
      expect(await restoreToolBatchReplayState(restoredParent, parentOwner)).toHaveLength(1);
      expect(restoredParent.configurable?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]).toEqual(manifest);
      expect(getToolApprovalReviewEvidence(restoredParent)).toBeUndefined();
      for (const member of ['left', 'right']) {
        const owner = JSON.stringify(['child-scope', '', member, 'user', 'child-thread']);
        const leaf = getToolReplayConfig(config, owner);
        expect(await restoreToolBatchReplayState(leaf, owner)).toHaveLength(1);
        expect(getToolApprovalReviewEvidence(leaf)).toBeUndefined();
        expect(getToolBatchReplayState(leaf.configurable)?.approvalOwner).toBeUndefined();
      }
      const state = getToolBatchReplayState(pending[1].value)!;
      const inherited = state.records.find(record => record.owner === parentOwner)!;
      inherited.data = 'conflicting';
      pending[1].value = attachSubagentResumeManifest({
        type: 'ask_user_question', question: { question: 'right' },
        [TOOL_BATCH_REPLAY_KEY]: state,
      }, manifest);
      expect(() => restoreToolReplayConfigs({}, pending, 'root-scope', 'root-scope', 'thread', new Set(['left', 'right']))).toThrow('Conflicting parent tool replay');
    },
  );

  it('rejects ambiguous member evidence rather than accepting the last checkpoint', async () => {
    const payload = await attachToolBatchReplayState(approval, JSON.stringify(['source', '', 'member']), new Map());
    expect(() => restoreToolReplayConfigs({}, [{ id: 'first', value: payload }, { id: 'second', value: payload }], 'source', 'fork', 'child-thread', new Set(['first', 'second']))).toThrow('Ambiguous tool replay checkpoint owner');
  });

  it('does not let one restored owner select authorization for a different member', async () => {
    const payload = await attachToolBatchReplayState(approval, JSON.stringify(['source', '', 'left']), new Map());
    const config: RunnableConfig = { configurable: { thread_id: 'child-thread', __librechat_tool_approval_execution_scope: 'fork' } };
    restoreToolReplayConfigs(config.configurable!, [{ id: 'only-left', value: payload }], 'source', 'fork', 'child-thread', new Set(['only-left']));
    const selected = getToolReplayConfig(config, getToolBatchReplayOwner(config, 'right'));
    expect(selected).toBe(config);
    expect(getToolApprovalReviewEvidence(selected)?.owner).not.toBe(getToolBatchReplayOwner(config, 'right'));
  });

  it.each([
    { interruptId: 'current', resume: [[]], isChildReplay: false, expected: 'active' },
    { interruptId: 'prior', resume: [[]], isChildReplay: false, expected: 'unverifiable' },
    { interruptId: 'current', resume: [], isChildReplay: false, expected: 'stale' },
    { interruptId: 'current', resume: [], isChildReplay: true, expected: 'active' },
    { interruptId: 'prior', resume: [], isChildReplay: true, expected: 'stale' },
  ])('scopes replay to the active interrupt and consuming task: %j', ({ interruptId, resume, isChildReplay, expected }) => {
    const status = getToolReplayResumeStatus({ configurable: {
      __pregel_resume_map: { current: [{ type: 'approve' }] },
      __pregel_scratchpad: { resume },
    } }, interruptId, isChildReplay);
    expect(status).toBe(expected);
  });

  it.each([
    {},
    { __pregel_resume_map: { current: [] } },
    { __pregel_resume_map: { current: [] }, __pregel_scratchpad: {} },
  ])('fails closed when resume internals are unavailable or malformed: %j', (configurable) => {
    expect(getToolReplayResumeStatus({ configurable }, 'current', false)).toBe('unverifiable');
  });

  it('distinguishes stale execution from an unverifiable active resume', () => {
    expect(getToolReplayResumeStatus({ configurable: {
      __pregel_scratchpad: { resume: [] },
    } }, 'current', false)).toBe('stale');
    expect(getToolReplayResumeStatus({ configurable: {
      __pregel_scratchpad: { resume: [{ type: 'approve' }] },
    } }, 'current', false)).toBe('unverifiable');
  });

  it('accepts only an unambiguous task-local legacy resume without a restored interrupt id', () => {
    expect(getToolReplayResumeStatus({ configurable: {
      __pregel_resume_map: { current: [{ type: 'approve' }] },
      __pregel_scratchpad: { resume: [[{ type: 'approve' }]] },
    } }, undefined, false)).toBe('active');
    expect(getToolReplayResumeStatus({ configurable: {
      __pregel_resume_map: { first: [], second: [] },
      __pregel_scratchpad: { resume: [[{ type: 'approve' }]] },
    } }, undefined, false)).toBe('unverifiable');
  });

  it('stamps question replay state with the checkpoint-owned interrupt id', async () => {
    const payload = await attachToolBatchReplayState(
      { type: 'ask_user_question', question: { question: 'Continue?' } },
      'owner', new Map([['batch', new Map()]])
    );
    const configurable = {};
    restoreToolReplayConfig(configurable, 'question-interrupt', payload);
    expect(getToolBatchReplayState(configurable)?.interruptId).toBe('question-interrupt');
    expect(getToolBatchReplayState(payload)?.interruptId).toBeUndefined();
  });

  it('does not downgrade a versioned approval with a missing owner to legacy evidence', () => {
    expect(() => restoreToolReplayConfig({}, 'interrupt', {
      ...approval, [TOOL_BATCH_REPLAY_KEY]: { version: 1, records: [] },
    })).toThrow('Invalid tool approval checkpoint');
    expect(() => restoreToolReplayConfig({}, 'interrupt', approval)).not.toThrow();
  });
  it('durably rebinds only the proven owner across consecutive unconsumed forks', async () => {
    const owner = JSON.stringify(['source', '', 'agent']);
    const batch = JSON.stringify(['source', 'assistant', 'proposal']);
    const unrelated = JSON.stringify(['other-child', '', 'agent']);
    const original = await attachToolBatchReplayState(
      approval,
      owner,
      new Map([[batch, new Map()]])
    );
    const wrapped = await attachToolBatchReplayState(
      original,
      unrelated,
      new Map([[JSON.stringify(['other-child', 'assistant']), new Map()]])
    );
    const first = rebindToolBatchReplayPayload(wrapped, 'source', 'fork-1');
    const second = rebindToolBatchReplayPayload(
      JSON.parse(JSON.stringify(first)),
      'fork-1',
      'fork-2'
    );
    const state = getToolBatchReplayState(second);
    expect(state?.approvalOwner).toBe(JSON.stringify(['fork-2', '', 'agent']));
    expect(state?.records.map(({ owner, batch }) => [owner, batch])).toEqual([
      [
        JSON.stringify(['fork-2', '', 'agent']),
        JSON.stringify(['fork-2', 'assistant', 'proposal']),
      ],
      [unrelated, JSON.stringify(['other-child', 'assistant'])],
    ]);
    expect(getToolBatchReplayState(wrapped)?.approvalOwner).toBe(owner);
    expect(getPublicToolInterruptPayload(second)).toEqual(approval);
  });

  it('rebinds conversation identity only through the trusted fork adapter', async () => {
    const owner = JSON.stringify(['source', '', 'agent', 'user', 'parent-thread']);
    const payload = await attachToolBatchReplayState(
      approval,
      owner,
      new Map([[JSON.stringify(['source', 'assistant']), new Map()]])
    );

    const rebound = rebindToolBatchReplayPayload(
      payload,
      'source',
      'child',
      'child-thread'
    );

    expect(getToolBatchReplayState(rebound)?.approvalOwner).toBe(
      JSON.stringify(['child', '', 'agent', 'user', 'child-thread'])
    );
  });

  it('rejects corrupt reference counters instead of restarting numbering', async () => {
    const wrapped = await attachToolBatchReplayState(
      approval,
      'owner',
      new Map([['batch', new Map()]])
    );
    const state = getToolBatchReplayState(wrapped)!;
    state.records[0].referenceState = {
      entries: [],
      turnCounter: -1,
      warnedNonStringTools: [],
    };
    expect(() =>
      getToolBatchReplayState({ [TOOL_BATCH_REPLAY_KEY]: state })
    ).toThrow('Invalid tool batch replay checkpoint');
  });

  it.each([undefined, '', 0])('does not cache a missing scope %j', (scope) => {
    expect(
      getToolBatchReplayScope({ configurable: { thread_id: scope } })
    ).toBeUndefined();
  });
  it('binds replay ownership to principal and conversation identity', () => {
    const owner = (user_id: string, thread_id: string) => getToolBatchReplayOwner({ configurable: {
      user_id,
      thread_id,
      __librechat_tool_approval_execution_scope: 'run',
    } }, 'agent');
    expect(owner('user-a', 'thread')).not.toBe(owner('user-b', 'thread'));
    expect(owner('user', 'thread-a')).not.toBe(owner('user', 'thread-b'));
  });
  it('binds legacy checkpoint owners during trusted restoration', async () => {
    const legacyOwner = JSON.stringify(['run', '', 'agent']);
    const payload = await attachToolBatchReplayState(approval, legacyOwner, new Map([
      ['batch', new Map([['call', {
        proposal: { name: 'echo', args: {} },
        output: new ToolMessage({ content: 'done', tool_call_id: 'call' }),
        additionalContexts: [],
      }]])],
    ]));
    const configurable = { user_id: 'user', thread_id: 'thread' };

    restoreToolReplayConfig(configurable, 'interrupt', payload);

    const owner = JSON.stringify(['run', '', 'agent', 'user', 'thread']);
    const restored = getToolBatchReplayState(configurable);
    expect(restored?.approvalOwner).toBe(owner);
    expect(restored?.records[0]?.owner).toBe(owner);
    await expect(restoreToolBatchReplayState({ configurable }, owner)).resolves.toHaveLength(1);
  });
  it('removes stale approval evidence without discarding settled results', async () => {
    const output = new ToolMessage({ content: 'checkpointed mutation', tool_call_id: 'call' });
    const payload = await attachToolBatchReplayState(approval, 'owner', new Map([
      ['batch', new Map([['call', {
        proposal: { name: 'echo', args: {} },
        output,
        additionalContexts: [],
      }]])],
    ]));
    const configurable: Record<string, unknown> = {};
    restoreToolReplayConfig(configurable, 'interrupt', payload);

    clearStaleToolApprovalConfig(configurable, 'owner', 'batch');

    expect(configurable).not.toHaveProperty('__librechat_tool_approval_review');
    const restored = await restoreToolBatchReplayState({ configurable }, 'owner');
    expect(restored).toHaveLength(1);
    expect(restored[0].results[0][1].output).toEqual(output);
  });
  it('preserves objects resembling the primitive wrapper', async () => {
    const payload = {
      __librechat_tool_batch_wrapper: 1,
      __librechat_tool_batch_payload: 'user data',
    };
    const wrapped = await attachToolBatchReplayState(
      payload,
      'owner',
      new Map([['batch', new Map()]])
    );
    expect(
      getPublicToolInterruptPayload(JSON.parse(JSON.stringify(wrapped)))
    ).toEqual(payload);
  });
  it.each([undefined, {}, { name: 'echo' }])(
    'rejects a missing or incomplete proposal %j',
    async (proposal) => {
      const wrapped = await attachToolBatchReplayState(
        approval,
        'owner',
        new Map([
          [
            'batch',
            new Map([
              [
                'call',
                {
                  proposal,
                  output: new ToolMessage({
                    content: 'done',
                    tool_call_id: 'call',
                  }),
                  additionalContexts: [],
                },
              ],
            ]),
          ],
        ])
      );
      await expect(
        restoreToolBatchReplayState(
          {
            configurable: {
              [TOOL_BATCH_REPLAY_KEY]: getToolBatchReplayState(wrapped),
            },
          },
          'owner'
        )
      ).rejects.toThrow('Invalid settled tool batch results');
    }
  );
  it.each([null, 'confirm', ['one', 'two']])(
    'preserves custom interrupt payload %j with settled results',
    async (payload) => {
      const wrapped = await attachToolBatchReplayState(
        payload,
        'owner',
        new Map([
          [
            'batch',
            new Map([
              [
                'call',
                {
                  output: new ToolMessage({
                    content: 'done',
                    tool_call_id: 'call',
                  }),
                  additionalContexts: [],
                },
              ],
            ]),
          ],
        ])
      );
      expect(getToolBatchReplayState(wrapped)?.records).toHaveLength(1);
      expect(stripToolBatchReplayState(wrapped)).toEqual(payload);
    }
  );
  it('does not reinterpret corrupt approval evidence as permission to execute', () => {
    const configurable = {
      forged: 'retained',
      [TOOL_BATCH_REPLAY_KEY]: { version: 1, records: [] },
    };
    expect(() =>
      restoreToolReplayConfig(configurable, 'interrupt', {
        ...approval,
        action_requests: [null],
      })
    ).toThrow('Invalid tool approval checkpoint');
    expect(configurable).toEqual({ forged: 'retained' });
  });
  it.each([undefined, 'child:task'])(
    'round-trips complete results through the checkpoint codec (%s)',
    async (namespace) => {
      const config = {
        configurable: { thread_id: 'thread', checkpoint_ns: namespace },
      };
      const owner = getToolBatchReplayOwner(config, 'agent');
      const results = new Map([
        [
          'message',
          {
            proposal: { name: 'echo', args: {} },
            output: new ToolMessage({ content: 'saved', tool_call_id: 'call' }),
            additionalContexts: ['context'],
            completionHandled: true,
          },
        ],
        [
          'command',
          {
            proposal: { name: 'handoff', args: {} },
            output: new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: 'handoff',
                    tool_call_id: 'handoff',
                  }),
                ],
              },
            }),
            additionalContexts: [],
          },
        ],
      ]);
      const payload = await attachToolBatchReplayState(
        approval,
        owner,
        new Map([['batch', results]])
      );
      const state = getToolBatchReplayState(
        JSON.parse(JSON.stringify(payload))
      );
      const restored = await restoreToolBatchReplayState(
        { configurable: { [TOOL_BATCH_REPLAY_KEY]: state } },
        owner
      );
      expect(restored[0].batch).toBe('batch');
      expect(restored[0].results[0]).toEqual([
        'message',
        results.get('message'),
      ]);
      const command = restored[0].results[1][1].output;
      expect(command).toBeInstanceOf(Command);
      expect(command).toMatchObject({
        update: { messages: [expect.any(ToolMessage)] },
      });
      expect(JSON.stringify(command)).toEqual(
        JSON.stringify(results.get('command')?.output)
      );
      expect(stripToolBatchReplayState(payload)).toEqual(approval);
    }
  );

  it('keeps the child approval owner while accumulating distinct parent results', async () => {
    const child = await attachToolBatchReplayState(
      approval,
      'child',
      new Map()
    );
    const parent = await attachToolBatchReplayState(
      child,
      'parent',
      new Map([
        [
          'batch',
          new Map([
            [
              'call',
              {
                output: new ToolMessage({
                  content: 'result',
                  tool_call_id: 'call',
                }),
                additionalContexts: [],
              },
            ],
          ]),
        ],
      ])
    );
    const state = getToolBatchReplayState(parent);
    expect(state?.approvalOwner).toBe('child');
    expect(
      await restoreToolBatchReplayState(
        { configurable: { [TOOL_BATCH_REPLAY_KEY]: state } },
        'unrelated'
      )
    ).toEqual([]);
    expect(state?.records.map((record) => record.owner)).toEqual(['parent']);
  });

  it.each([
    null,
    {},
    { version: 2, records: [] },
    { version: 1, records: [null] },
  ])('fails closed for malformed checkpoint state %j', (state) => {
    expect(() =>
      getToolBatchReplayState({ [TOOL_BATCH_REPLAY_KEY]: state })
    ).toThrow('Invalid tool batch replay checkpoint');
  });

  it('does not require private state in legacy checkpoints', () => {
    expect(getToolBatchReplayState(approval)).toBeUndefined();
    expect(
      getToolBatchReplayState({ [TOOL_BATCH_REPLAY_KEY]: undefined })
    ).toBeUndefined();
  });

  it.each([false, true])('rejects an unaddressed malformed sibling manifest, wrapped=%s', (wrapped) => {
    const malformed = {
      type: 'ask_user_question',
      question: { question: 'Continue?' },
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: { version: 2, executions: [] },
    };
    const value = wrapped ? attachRunStepResumeState(malformed, {
      version: 1, revision: 0, nextIndex: 0, steps: [], toolCallSteps: [],
    }) : malformed;
    expect(() => restoreToolReplayConfigs({}, [
      { id: 'addressed', value: approval }, { id: 'unaddressed', value },
    ], 'source', 'target', 'thread', new Set(['addressed'])))
      .toThrow('Invalid subagent resume manifest');
  });

  it.each(['approval', 'ownerless', 'unmatched'].flatMap((kind) => [false, true].map((reverse) => ({ kind, reverse }))))(
    'does not inherit a sibling manifest: %j', async ({ kind, reverse }) => {
      const manifest: SubagentResumeManifest = { version: 1, executions: [{
        parentToolCallId: 'shared-spawn', childRunId: 'existing-child', approvalExecutionScope: 'child-scope',
        checkpoints: [{ threadId: 'child-thread', checkpointId: 'checkpoint', checkpointNs: '' }],
        graphState: { toolCallSteps: [], toolSessions: [], toolNodes: [], eagerToolUsage: [], eagerToolSuppressions: [] },
        approvalReplays: [],
      }] };
      const owner = (member: string) => JSON.stringify(['scope', '', member, 'user', 'thread']);
      const child = { id: 'child', value: attachSubagentResumeManifest(await attachToolBatchReplayState(approval, owner('left'), new Map()), manifest) };
      const other = { id: 'other', value: kind === 'ownerless' ? { type: 'ask_user_question' } : await attachToolBatchReplayState(approval, owner('right'), new Map()) };
      const config: RunnableConfig = { configurable: { user_id: 'user', thread_id: 'thread', __librechat_tool_approval_execution_scope: 'scope', __pregel_scratchpad: { resume: [] } } };
      restoreToolReplayConfigs(config.configurable!, reverse ? [other, child] : [child, other], 'scope', 'scope', 'thread', new Set(['child']));
      const selected = getToolReplayConfig(config, owner(kind === 'unmatched' ? 'joined' : 'right'));
      expect(selected.configurable?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]).toBeUndefined();
      expect(selected.configurable?.[TOOL_REPLAY_CONFIGS_KEY]).toBeUndefined();
      expect(getToolApprovalReviewEvidence(selected)).toBeUndefined();
      expect(getToolReplayConfig(config, owner('left')).configurable?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]).toEqual(manifest);
    }
  );

  it.each([
    { runtime: {}, expected: 'reject' },
    { runtime: { __pregel_scratchpad: { resume: [] }, __pregel_resume_map: { pending: [] } }, expected: 'stale' },
    { runtime: { __pregel_scratchpad: { resume: [[]] }, __pregel_resume_map: { pending: [], sibling: [] } }, expected: 'reject' },
    { runtime: { __pregel_scratchpad: { resume: [], nullResume: [] } }, expected: 'reject' },
    { runtime: { __pregel_scratchpad: { resume: [[]] } }, expected: 'reject' },
  ])('distinguishes active or ambiguous foreign approval from stale downstream context: $expected %j', async ({ runtime, expected }) => {
    const config: RunnableConfig = { configurable: { thread_id: 'thread', user_id: 'original', __librechat_tool_approval_execution_scope: 'scope', ...runtime } };
    const pending = await Promise.all(['left', 'right'].map(async (member) => ({ id: member, value: await attachToolBatchReplayState(approval,
      JSON.stringify(['scope', '', member, 'original', 'thread']), new Map()) })));
    restoreToolReplayConfigs(config.configurable!, pending, 'scope', 'scope', 'thread', new Set(['left', 'right']));
    const changed = { ...config, configurable: { ...config.configurable, thread_id: 'other-thread', user_id: 'other-user' } };
    const select = () => getToolReplayConfig(changed, getToolBatchReplayOwner(changed, 'left'));
    if (expected === 'reject') expect(select).toThrow('execution owner changed');
    else {
      const neutral = select();
      expect(neutral.configurable?.[TOOL_REPLAY_CONFIGS_KEY]).toBeUndefined();
      expect(getToolApprovalReviewEvidence(neutral)).toBeUndefined();
    }
  });

  it('does not invent approval authority for an opaque custom resume', async () => {
    const config: RunnableConfig = { configurable: { user_id: 'original', thread_id: 'thread', __librechat_tool_approval_execution_scope: 'scope',
      __pregel_scratchpad: { resume: ['custom answer'] }, __pregel_resume_map: { left: 'custom answer', right: 'custom answer' },
    } };
    const pending = await Promise.all(['left', 'right'].map(async (member) => ({ id: member, value: await attachToolBatchReplayState({ question: 'Custom question' },
      JSON.stringify(['scope', '', member, 'original', 'thread']), new Map([[JSON.stringify(['scope', 'completed']), new Map()]])) })));
    restoreToolReplayConfigs(config.configurable!, pending, 'scope', 'scope', 'thread', new Set(['left', 'right']));
    const changed = { ...config, configurable: { ...config.configurable, user_id: 'other-user' } };
    const neutral = getToolReplayConfig(changed, getToolBatchReplayOwner(changed, 'left'));
    expect(getToolApprovalReviewEvidence(neutral)).toBeUndefined();
    expect(neutral.configurable?.[TOOL_REPLAY_CONFIGS_KEY]).toBeUndefined();
  });

});

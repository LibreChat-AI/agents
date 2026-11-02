import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { IState } from '@/types';
import {
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
  HookRegistry,
} from '@/hooks';
import { TOOL_APPROVAL_REVIEW_CONFIG_KEY } from '@/hitl/approvalReview';
import { TOOL_BATCH_REPLAY_KEY } from '../toolBatchReplay';
import { askUserQuestion } from '@/hitl';
import { ToolNode } from '../ToolNode';
import { Providers } from '@/common';
import { Run } from '@/run';

function createHarness(
  carryCheckpointConfig = false,
  question = false,
  completedSibling = false,
  resumeInternalMode: 'supported' | 'missing' | 'mismatched' = 'supported'
) {
  let carriedConfigurable: RunnableConfig['configurable'];
  const checkpointer = new MemorySaver();
  const executions: string[] = [];
  const namespaces: string[] = [];
  const resumeInternals: Array<{
    resumeMap: boolean;
    scratchpad: boolean;
  }> = [];
  const echo = tool(
    async ({ value }) => {
      if (question && value.endsWith('-0')) {
        askUserQuestion({ question: 'Continue?' });
      }
      executions.push(value);
      return value;
    },
    {
      name: 'echo',
      description: 'echo',
      schema: z.object({ value: z.string() }),
    }
  );
  const sibling = tool(
    async ({ value }) => {
      executions.push(value);
      return value;
    },
    {
      name: 'sibling',
      description: 'sibling',
      schema: z.object({ value: z.string() }),
    }
  );
  const createRun = async (agentId: string, runId: string, cycles = 1) => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async ({ toolName }) => ({
          decision: question || toolName === 'sibling' ? 'allow' : 'ask',
        }),
      ],
    });
    const node = new ToolNode({
      tools: completedSibling ? [echo, sibling] : [echo],
      agentId,
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
      eventDrivenMode: true,
      directToolNames: new Set(
        completedSibling ? ['echo', 'sibling'] : ['echo']
      ),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('agent', (state) => {
        const promptIndex = state.messages.findLastIndex(
          (message) => message instanceof HumanMessage
        );
        const completed = state.messages
          .slice(promptIndex)
          .filter((message) => message instanceof ToolMessage).length;
        return {
          messages: [
            new AIMessage(
              completed >= cycles
                ? 'done'
                : {
                  content: '',
                  tool_calls: [
                    ...(completedSibling
                      ? [
                        {
                          id: `${runId}-sibling-${completed}`,
                          name: 'sibling',
                          args: { value: `${runId}-sibling-${completed}` },
                        },
                      ]
                      : []),
                    {
                      id: `${runId}-${completed}`,
                      name: 'echo',
                      args: { value: `${runId}-${completed}` },
                    },
                  ],
                }
            ),
          ],
        };
      })
      .addNode('tools', async (state, config) => {
        namespaces.push(String(config.configurable?.checkpoint_ns));
        if (
          config.configurable?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY] != null
        ) {
          resumeInternals.push({
            resumeMap: Object.prototype.hasOwnProperty.call(
              config.configurable,
              '__pregel_resume_map'
            ),
            scratchpad: Object.prototype.hasOwnProperty.call(
              config.configurable,
              '__pregel_scratchpad'
            ),
          });
        }
        const configurable = { ...carriedConfigurable, ...config.configurable };
        if (configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY] != null &&
          resumeInternalMode !== 'supported') {
          if (resumeInternalMode === 'missing') {
            delete configurable.__pregel_resume_map;
            delete configurable.__pregel_scratchpad;
          } else {
            configurable.__pregel_resume_map = {};
          }
        }
        const nodeConfig = {
          ...config,
          configurable,
        };
        if (
          carryCheckpointConfig &&
          config.configurable?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY] != null
        ) {
          carriedConfigurable = {
            [TOOL_APPROVAL_REVIEW_CONFIG_KEY]:
              config.configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY],
            [TOOL_BATCH_REPLAY_KEY]: config.configurable[TOOL_BATCH_REPLAY_KEY],
          };
        }
        return node.invoke(state, nodeConfig);
      })
      .addEdge(START, 'agent')
      .addConditionalEdges('agent', (state) =>
        ((state.messages.at(-1) as AIMessage).tool_calls?.length ?? 0) > 0
          ? 'tools'
          : END
      )
      .addEdge('tools', 'agent')
      .compile({ checkpointer });
    const run = await Run.create<IState>({
      runId,
      hooks,
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId,
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
        compileOptions: { checkpointer },
      },
    });
    run.graphRunnable = graph as typeof run.graphRunnable;
    return run;
  };
  return { createRun, executions, namespaces, resumeInternals };
}

function config(scope?: string, userId = 'user-a'): RunnableConfig & { version: 'v2' } {
  return {
    configurable: {
      user_id: userId,
      thread_id: 'same-conversation',
      ...(scope == null
        ? {}
        : { [TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY]: scope }),
    },
    version: 'v2',
  };
}

describe('approval execution scope', () => {
  it.each([
    { rebuilt: false, scoped: false },
    { rebuilt: false, scoped: true },
    { rebuilt: true, scoped: true },
  ])(
    'completes two approval cycles despite changing task namespaces (rebuilt=$rebuilt, scoped=$scoped)',
    async ({ rebuilt, scoped }) => {
      const { createRun, executions, namespaces } = createHarness();
      const run = await createRun('a', 'run-1', 2);
      const runConfig = config(scoped ? 'run-1' : undefined);
      await run.processStream(
        { messages: [new HumanMessage('start')] },
        runConfig
      );
      const firstInterrupt = run.getInterrupt()?.interruptId;
      expect(firstInterrupt).toBeTruthy();
      await run.resume([{ type: 'approve' }], runConfig);
      expect(executions).toEqual(['run-1-0']);
      expect(run.getInterrupt()?.interruptId).toBeTruthy();
      expect(run.getInterrupt()?.interruptId).not.toBe(firstInterrupt);
      const secondResume = rebuilt ? await createRun('a', 'run-1', 2) : run;
      await secondResume.resume([{ type: 'approve' }], runConfig);
      expect(executions).toEqual(['run-1-0', 'run-1-1']);
      expect(secondResume.getInterrupt()).toBeUndefined();
      expect(new Set(namespaces).size).toBeGreaterThan(1);
    }
  );

  it('replays a completed sibling exactly once across an approval resume', async () => {
    const { createRun, executions, resumeInternals } = createHarness(
      false,
      false,
      true
    );
    const run = await createRun('a', 'run-1');
    const runConfig = config('run-1');

    await run.processStream(
      { messages: [new HumanMessage('start')] },
      runConfig
    );
    expect(executions).toEqual(['run-1-sibling-0']);

    await run.resume([{ type: 'approve' }], runConfig);
    expect(executions).toEqual(['run-1-sibling-0', 'run-1-0']);
    expect(resumeInternals).toEqual([
      { resumeMap: true, scratchpad: true },
    ]);
    expect(run.getInterrupt()).toBeUndefined();
  });

  it.each(['missing', 'mismatched'] as const)(
    'fails closed without replaying a completed mutation when resume internals are %s',
    async (resumeInternalMode) => {
      const { createRun, executions } = createHarness(false, false, true, resumeInternalMode);
      const runId = `changed-internals-${resumeInternalMode}`;
      const run = await createRun('a', runId);
      const runConfig = config(runId);
      await run.processStream({ messages: [new HumanMessage('start')] }, runConfig);
      expect(executions).toEqual([`${runId}-sibling-0`]);

      await expect(run.resume([{ type: 'approve' }], runConfig)).rejects.toThrow(
        'Cannot verify the active tool approval resume'
      );
      expect(executions).toEqual([`${runId}-sibling-0`]);
    }
  );

  it('preserves completed results through two consecutive approval cycles', async () => {
    const { createRun, executions } = createHarness(true, false, true);
    const run = await createRun('a', 'two-cycle-replay', 4);
    const runConfig = config('two-cycle-replay');
    await run.processStream({ messages: [new HumanMessage('start')] }, runConfig);
    await run.resume([{ type: 'approve' }], runConfig);
    expect(run.getInterrupt()?.payload).toMatchObject({ type: 'tool_approval' });
    await run.resume([{ type: 'approve' }], runConfig);

    expect(executions).toEqual([
      'two-cycle-replay-sibling-0',
      'two-cycle-replay-0',
      'two-cycle-replay-sibling-2',
      'two-cycle-replay-2',
    ]);
    expect(run.getInterrupt()).toBeUndefined();
  });

  it('continues to a new tool batch after a question resume with fresh instances', async () => {
    const { createRun, executions } = createHarness(false, true);
    const run = await createRun('a', 'question-run', 2);
    const runConfig = config('question-run');
    await run.processStream({ messages: [new HumanMessage('start')] }, runConfig);
    expect(run.getInterrupt()?.payload).toMatchObject({ type: 'ask_user_question' });
    expect(executions).toEqual([]);

    const rebuilt = await createRun('a', 'question-run', 2);
    await rebuilt.resume({ answer: 'yes' }, runConfig);
    expect(executions).toEqual(['question-run-0', 'question-run-1']);
    expect(rebuilt.getInterrupt()).toBeUndefined();
  });

  it.each(['resume', 'command'].flatMap((entrypoint) =>
    ['agent', 'scope', 'principal'].map((changedIdentity) => ({ entrypoint, changedIdentity }))
  ))(
    'fails closed when $changedIdentity changes during $entrypoint',
    async ({ entrypoint, changedIdentity }) => {
      const { createRun, executions } = createHarness();
      const original = await createRun('a', 'run-1');
      await original.processStream(
        { messages: [new HumanMessage('start')] },
        config('run-1')
      );
      const resumed = await createRun(
        changedIdentity === 'agent' ? 'b' : 'a',
        'run-1'
      );
      const resumeConfig = config(
        changedIdentity === 'scope' ? 'run-2' : 'run-1',
        changedIdentity === 'principal' ? 'user-b' : 'user-a'
      );
      const decisions = [{ type: 'approve' as const }];
      await expect(entrypoint === 'command'
        ? resumed.processStream(new Command({ resume: decisions }), resumeConfig)
        : resumed.resume(decisions, resumeConfig)
      ).rejects.toThrow('Tool approval execution owner changed');
      expect(executions).toEqual([]);
    }
  );

  it.each([false, true])(
    'allows an agent switch after a finalized run (explicit scope=%s)',
    async (scoped) => {
      const { createRun, executions } = createHarness(true);
      const first = await createRun('a', 'run-1');
      await first.processStream(
        { messages: [new HumanMessage('first')] },
        config(scoped ? 'run-1' : undefined)
      );
      await first.resume(
        [{ type: 'approve' }],
        config(scoped ? 'run-1' : undefined)
      );
      expect(first.getInterrupt()).toBeUndefined();
      const second = await createRun('b', 'run-2');
      await second.processStream(
        { messages: [new HumanMessage('second')] },
        config(scoped ? 'run-2' : undefined)
      );
      expect(second.getInterrupt()?.payload).toMatchObject({
        type: 'tool_approval',
      });
      expect(executions).toEqual(['run-1-0']);
      await second.resume(
        [{ type: 'approve' }],
        config(scoped ? 'run-2' : undefined)
      );
      expect(executions).toEqual(['run-1-0', 'run-2-0']);
    }
  );
});

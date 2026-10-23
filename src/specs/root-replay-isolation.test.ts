import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type * as t from '@/types';
import { SubagentExecutionRegistry } from '@/tools/subagent/SubagentExecutionRegistry';
import { getSubagentResumeManifest } from '@/tools/subagent/SubagentReplay';
import { getPublicToolInterruptPayload } from '@/tools/toolBatchReplay';
import { Constants, GraphEvents, Providers } from '@/common';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

type Root = 'left' | 'right';
const privateKeys = [
  '__librechat_tool_replay_configs',
  '__librechat_tool_batch_replay',
  '__librechat_tool_approval_review',
  '__librechat_subagent_resume_manifest',
  '__librechat_subagent_resume_attempt',
  '__librechat_subagent_parent_batch',
];

class IsolatedTaskModel extends FakeListChatModel {
  private readonly owners: Map<string, string>;
  constructor() {
    const owners = new Map<string, string>();
    const capture = BaseCallbackHandler.fromMethods({
      handleChatModelStart(
        _model,
        _messages,
        runId,
        _parent,
        _extra,
        _tags,
        metadata
      ) {
        if (typeof metadata?.activeAgentId === 'string')
          owners.set(runId, metadata.activeAgentId);
      },
    });
    capture.awaitHandlers = true;
    super({ responses: ['unused'], callbacks: [capture] });
    this.owners = owners;
  }
  override async *_streamResponseChunks(
    ...args: Parameters<FakeListChatModel['_streamResponseChunks']>
  ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
    const owner = this.owners.get(args[2]?.runId ?? '');
    if (owner == null) throw new Error('Missing isolated task owner');
    const root = owner === 'left' || owner === 'right';
    let name = 'final_tool';
    if (root) name = Constants.SUBAGENT;
    else if (owner === 'worker') name = 'member_tool';
    if (
      (root || owner === 'worker' || owner === 'result') &&
      !args[0].some(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).name === name
      )
    ) {
      const task = root
        ? `task-${owner}`
        : ['task-left', 'task-right'].find((candidate) =>
          args[0].some((message) =>
            String(message.content).includes(candidate)
          )
        );
      if (owner === 'worker' && task == null)
        throw new Error('Missing child task description');
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              name,
              id: root ? 'shared-spawn' : `${owner}-call`,
              index: 0,
              type: 'tool_call_chunk',
              args: JSON.stringify(
                root
                  ? { description: task, subagent_type: 'team' }
                  : { value: task ?? 'joined' }
              ),
            },
          ],
        }),
      });
      return;
    }
    if (root) {
      const output = args[0].findLast(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).name === Constants.SUBAGENT
      );
      if (!String(output?.content).includes(`worker task-${owner} done`))
        throw new Error('Foreign child result');
    }
    const task = ['task-left', 'task-right'].find((candidate) =>
      args[0].some((message) => String(message.content).includes(candidate))
    );
    const text = owner === 'worker' ? `worker ${task} done` : `${owner} done`;
    yield new ChatGenerationChunk({ text, message: new AIMessageChunk(text) });
  }
}

function harness(blockedRoot?: Root, primary: 'child' | 'root' = 'child') {
  const checkpointer = new MemorySaver();
  const effects = jest.fn(async (value: string) => `completed ${value}`);
  const hostEffects = jest.fn(async () => 'joined results');
  const directPrivateKeys: string[][] = [];
  const hostPrivateKeys: string[][] = [];
  let childReady: () => void = () => {};
  let rootReady: () => void = () => {};
  const childPaused = new Promise<void>((resolve) => {
    childReady = resolve;
  });
  const rootPaused = new Promise<void>((resolve) => {
    rootReady = resolve;
  });
  const putWrites = checkpointer.putWrites.bind(checkpointer);
  jest.spyOn(checkpointer, 'putWrites').mockImplementation(async (...args) => {
    await putWrites(...args);
    for (const [channel, entry] of args[1]) {
      if (
        channel !== '__interrupt__' ||
        entry == null ||
        typeof entry !== 'object' ||
        !('value' in entry)
      )
        continue;
      const payload = getPublicToolInterruptPayload(entry.value);
      if (!isToolApprovalInterrupt(payload)) continue;
      if (
        payload.action_requests[0]?.name === 'member_tool' &&
        getSubagentResumeManifest(entry.value) != null
      )
        childReady();
      if (payload.action_requests[0]?.name === Constants.SUBAGENT) rootReady();
    }
  });
  const agent = (agentId: string): t.AgentInputs => ({
    agentId,
    provider: Providers.OPENAI,
    clientOptions: { modelName: agentId, apiKey: 'unused' },
  });
  const build = async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [
        async ({ toolName, executingAgentId }) => {
          if (toolName === 'member_tool') {
            if (blockedRoot != null && primary === 'root') await rootPaused;
            return { decision: 'ask' as const };
          }
          if (
            toolName === Constants.SUBAGENT &&
            blockedRoot != null &&
            executingAgentId === blockedRoot
          ) {
            if (primary === 'child') await childPaused;
            return { decision: 'ask' as const };
          }
          return { decision: 'allow' as const };
        },
      ],
    });
    const work = tool(
      async ({ value }: { value: string }, runtime?: ToolRunnableConfig) => {
        directPrivateKeys.push(
          privateKeys.filter((key) =>
            Object.hasOwn(runtime?.configurable ?? {}, key)
          )
        );
        return effects(value);
      },
      {
        name: 'member_tool',
        description: 'Complete the assigned task',
        schema: z.object({ value: z.string() }),
      }
    );
    const team: t.GraphSubagentConfig = {
      kind: 'graph',
      type: 'team',
      name: 'Team',
      description: 'Work',
      configId: 'same-implementation',
      agents: [{ ...agent('worker'), graphTools: [work] }],
      edges: [],
      entryAgentId: 'worker',
      resultAgentId: 'worker',
    };
    const parents = (['left', 'right'] as const).map((id) => ({
      ...agent(id),
      subagentConfigs: [team],
      maxSubagentDepth: 1,
    }));
    const run = await Run.create<t.IState>({
      runId: `root-isolation-${Math.random()}`,
      hooks,
      humanInTheLoop: { enabled: true },
      skipCleanup: true,
      graphConfig: {
        type: 'multi-agent',
        entryAgentId: 'entry',
        agents: [
          agent('entry'),
          ...parents,
          {
            ...agent('result'),
            tools: [
              tool(async () => 'host only', {
                name: 'final_tool',
                description: 'Join',
                schema: z.object({ value: z.string() }),
              }),
            ],
            toolDefinitions: [
              {
                name: 'final_tool',
                description: 'Join',
                parameters: {
                  type: 'object',
                  properties: { value: { type: 'string' } },
                  required: ['value'],
                },
              },
            ],
          },
        ],
        edges: [
          { from: 'entry', to: ['left', 'right'], edgeType: 'direct' },
          { from: ['left', 'right'], to: 'result', edgeType: 'direct' },
        ],
        compileOptions: { checkpointer },
      },
      customHandlers: {
        [GraphEvents.ON_TOOL_EXECUTE]: {
          async handle(_event, data) {
            const request = data as t.ToolExecuteBatchRequest;
            hostPrivateKeys.push(
              privateKeys.filter((key) =>
                Object.hasOwn(request.configurable ?? {}, key)
              )
            );
            request.resolve(
              await Promise.all(
                request.toolCalls.map(async (call) => ({
                  toolCallId: call.id,
                  status: 'success' as const,
                  content: await hostEffects(),
                }))
              )
            );
          },
        },
      },
    });
    run.Graph!.overrideModel = new IsolatedTaskModel();
    run.Graph!.setSubagentModelOverride(new IsolatedTaskModel());
    return run;
  };
  return { build, effects, hostEffects, hostPrivateKeys, directPrivateKeys };
}
const config = {
  configurable: { thread_id: 'root-replay-isolation', user_id: 'user' },
  version: 'v2' as const,
};
function approve(run: Run<t.IState>): Record<string, t.ToolApprovalDecision[]> {
  return Object.fromEntries(
    (run.getInterrupt()?.pendingInterrupts ?? []).map((entry) => [
      entry.interruptId,
      [{ type: 'approve' }],
    ])
  );
}

describe('parallel root replay isolation', () => {
  afterEach(() => jest.restoreAllMocks());
  test.each(
    [false, true].flatMap((rebuilt) =>
      (['left', 'right'] as const).flatMap((blockedRoot) =>
        (['child', 'root'] as const).flatMap((primary) =>
          ['blocked', 'both'].map((addressed) => ({
            rebuilt,
            blockedRoot,
            primary,
            addressed,
          }))
        )
      )
    )
  )(
    'keeps pre-spawn approval separate from an existing child: %j',
    async ({ rebuilt, blockedRoot, primary, addressed }) => {
      const opens = jest.spyOn(SubagentExecutionRegistry.prototype, 'open');
      const h = harness(blockedRoot, primary);
      const run = await h.build();
      await run.processStream(
        { messages: [new HumanMessage('Do both tasks')] },
        config
      );
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const primaryPayload = run.getInterrupt()?.payload;
      if (!isToolApprovalInterrupt(primaryPayload))
        throw new Error('Missing primary approval');
      expect(primaryPayload.action_requests[0].name).toBe(
        primary === 'child' ? 'member_tool' : Constants.SUBAGENT
      );
      const root = pending?.find(
        (entry) =>
          isToolApprovalInterrupt(entry.payload) &&
          entry.payload.action_requests[0].name === Constants.SUBAGENT
      );
      if (root == null) throw new Error('Missing pre-spawn approval');
      const initialChild = opens.mock.results
        .filter((result) => result.type === 'return')
        .map((result) => result.value.snapshot)
        .find(
          (snapshot) =>
            snapshot.invocation?.description ===
            `task-${blockedRoot === 'left' ? 'right' : 'left'}`
        )?.identity;
      expect(initialChild).toBeDefined();
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(
        addressed === 'both'
          ? approve(run)
          : { [root.interruptId]: [{ type: 'approve' }] },
        config
      );
      const remaining = resumed.getInterrupt()?.pendingInterrupts;
      expect(remaining).toHaveLength(addressed === 'both' ? 1 : 2);
      const pendingTasks = (remaining ?? []).map((entry) => {
        if (!isToolApprovalInterrupt(entry.payload))
          throw new Error('Missing task approval');
        return entry.payload.action_requests[0].arguments.value;
      });
      expect(pendingTasks).toContain(`task-${blockedRoot}`);
      const newChild = opens.mock.results
        .filter((result) => result.type === 'return')
        .map((result) => result.value.snapshot)
        .find(
          (snapshot) =>
            snapshot.invocation?.description === `task-${blockedRoot}`
        );
      expect(newChild?.resumeExecution).toBeUndefined();
      expect(newChild?.identity).toBeDefined();
      expect(
        newChild?.identity?.childThreadId === initialChild?.childThreadId
      ).toBe(false);
      const finished = rebuilt ? await h.build() : resumed;
      await finished.resume(approve(resumed), config);
      expect(finished.getInterrupt()).toBeUndefined();
      expect(h.effects.mock.calls.map(([value]) => value).sort()).toEqual([
        'task-left',
        'task-right',
      ]);
      expect(h.hostEffects).toHaveBeenCalledTimes(1);
      expect(h.directPrivateKeys).toEqual([[], []]);
      expect(h.hostPrivateKeys).toEqual([[]]);
    }
  );

  test.each([false, true])(
    'strips sibling replay state from the actual converged host request, rebuilt=%s',
    async (rebuilt) => {
      const h = harness();
      const run = await h.build();
      await run.processStream(
        { messages: [new HumanMessage('Do both tasks')] },
        config
      );
      expect(run.getInterrupt()?.pendingInterrupts).toHaveLength(2);
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(approve(run), config);
      expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.effects.mock.calls.map(([value]) => value).sort()).toEqual([
        'task-left',
        'task-right',
      ]);
      expect(h.hostEffects).toHaveBeenCalledTimes(1);
      expect(h.hostPrivateKeys).toEqual([[]]);
    }
  );
});

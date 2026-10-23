import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { Constants, Providers } from '@/common';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

class DelegatedApprovalModel extends FakeListChatModel {
  private readonly owners: Map<string, string>;
  constructor(private readonly calls: string[]) {
    const owners = new Map<string, string>();
    super({
      responses: ['unused'],
      callbacks: [
        {
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
        },
      ],
    });
    this.owners = owners;
  }
  override async *_streamResponseChunks(
    ...args: Parameters<FakeListChatModel['_streamResponseChunks']>
  ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
    const owner = this.owners.get(args[2]?.runId ?? '');
    if (owner == null) throw new Error('Missing delegated member');
    this.calls.push(owner);
    const parent = owner === 'left' || owner === 'right';
    const worker = owner.endsWith('_worker');
    const name = parent ? Constants.SUBAGENT : 'member_tool';
    if (
      (parent || worker) &&
      !args[0].some(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).name === name
      )
    ) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              name,
              args: JSON.stringify(
                parent
                  ? { description: 'Review work', subagent_type: 'team' }
                  : { value: 'original' }
              ),
              id: parent ? 'shared-parent-call' : 'shared-member-call',
              index: 0,
              type: 'tool_call_chunk',
            },
          ],
        }),
      });
      return;
    }
    yield new ChatGenerationChunk({
      text: `${owner} complete`,
      message: new AIMessageChunk(`${owner} complete`),
    });
  }
}
const config = {
  configurable: { thread_id: 'parallel-root-subagents', user_id: 'user' },
  version: 'v2' as const,
  streamMode: 'values',
};
function harness() {
  const checkpointer = new MemorySaver();
  const calls: string[] = [];
  const left = jest.fn(async ({ value }: { value: string }) => `left ${value}`);
  const right = jest.fn(
    async ({ value }: { value: string }) => `right ${value}`
  );
  const work = { left, right };
  const inputs = (agentId: string): t.AgentInputs => ({
    agentId,
    provider: Providers.OPENAI,
    clientOptions: { modelName: agentId, apiKey: 'unused' },
    instructions: agentId,
  });
  const build = async () => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      pattern: 'member_tool',
      hooks: [async () => ({ decision: 'ask' as const })],
    });
    const parents = (['left', 'right'] as const).map((id) => {
      const workerId = `${id}_worker`;
      const team: t.GraphSubagentConfig = {
        kind: 'graph',
        type: 'team',
        name: 'Team',
        description: 'Work',
        configId: 'implementation-v1',
        agents: [
          {
            ...inputs(workerId),
            graphTools: [
              tool(work[id], {
                name: 'member_tool',
                description: 'Review',
                schema: z.object({ value: z.string() }),
              }),
            ],
          },
        ],
        edges: [],
        entryAgentId: workerId,
        resultAgentId: workerId,
      };
      return { ...inputs(id), subagentConfigs: [team], maxSubagentDepth: 1 };
    });
    const run = await Run.create<t.IState>({
      runId: `parallel-parents-${Math.random()}`,
      humanInTheLoop: { enabled: true },
      hooks,
      skipCleanup: true,
      returnContent: true,
      graphConfig: {
        type: 'multi-agent',
        agents: [inputs('entry'), ...parents, inputs('result')],
        entryAgentId: 'entry',
        edges: [
          { from: 'entry', to: ['left', 'right'], edgeType: 'direct' },
          { from: ['left', 'right'], to: 'result', edgeType: 'direct' },
        ],
        compileOptions: { checkpointer },
      },
    });
    run.Graph!.overrideModel = new DelegatedApprovalModel(calls);
    run.Graph!.setSubagentModelOverride(new DelegatedApprovalModel(calls));
    return run;
  };
  return { build, left, right, calls };
}

describe('parallel roots delegated graph approval', () => {
  test.each(
    [false, true].flatMap((rebuilt) =>
      ['both', 'left', 'right'].map((addressed) => ({ rebuilt, addressed }))
    )
  )(
    'keeps parent records and child manifests scoped: %j',
    async ({ rebuilt, addressed }) => {
      const h = harness();
      const run = await h.build();
      await run.processStream(
        { messages: [new HumanMessage('Review work')] },
        config
      );
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const entry of pending ?? []) {
        if (!isToolApprovalInterrupt(entry.payload))
          throw new Error('Missing delegated proposal');
        const member = entry.payload.subagent?.agent_id;
        if (addressed === 'both' || member === `${addressed}_worker`)
          decisions[entry.interruptId] = [{ type: 'approve' }];
      }
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(decisions, config);
      expect(h.left).toHaveBeenCalledTimes(addressed === 'right' ? 0 : 1);
      expect(h.right).toHaveBeenCalledTimes(addressed === 'left' ? 0 : 1);
      if (addressed !== 'both') {
        const last = resumed.getInterrupt()?.pendingInterrupts?.[0];
        expect(last).toBeDefined();
        const again = await h.build();
        await again.resume(
          { [last!.interruptId]: [{ type: 'approve' }] },
          config
        );
        expect(again.getInterrupt()).toBeUndefined();
      } else expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.left).toHaveBeenCalledTimes(1);
      expect(h.right).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );
});

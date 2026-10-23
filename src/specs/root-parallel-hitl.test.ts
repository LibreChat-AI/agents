import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { Command, MemorySaver } from '@langchain/langgraph';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { GraphEvents, Providers } from '@/common';
import { askUserQuestion } from '@/hitl';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

class RootApprovalModel extends FakeListChatModel {
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
    if (owner == null) throw new Error('Missing root member identity');
    this.calls.push(owner);
    if (
      (owner === 'left' || owner === 'right') &&
      !args[0].some(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).name === `${owner}_tool`
      )
    ) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              name: `${owner}_effect`,
              args: '{}',
              id: 'shared-effect',
              index: 0,
              type: 'tool_call_chunk',
            },
            {
              name: `${owner}_tool`,
              args: '{"value":"original"}',
              id: 'shared-call',
              index: 1,
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
  configurable: { thread_id: 'root-parallel-hitl', user_id: 'user' },
  version: 'v2' as const,
  streamMode: 'values',
};
function harness(eventDriven: boolean, questionMember?: 'left' | 'right') {
  const checkpointer = new MemorySaver();
  const calls: string[] = [];
  const reviews: string[] = [];
  const leftEffect = jest.fn(async () => 'left effect');
  const rightEffect = jest.fn(async () => 'right effect');
  const effects = { left: leftEffect, right: rightEffect };
  const left = jest.fn(async ({ value }: { value: string }) =>
    questionMember === 'left'
      ? `left ${askUserQuestion({ question: 'Clarify left' }).answer}`
      : `left ${value}`
  );
  const right = jest.fn(async ({ value }: { value: string }) =>
    questionMember === 'right'
      ? `right ${askUserQuestion({ question: 'Clarify right' }).answer}`
      : `right ${value}`
  );
  const execute = { left, right };
  const build = async () => {
    const hooks = new HookRegistry();
    for (const id of ['left', 'right'])
      if (id !== questionMember)
        hooks.register('PreToolUse', {
          pattern: `${id}_tool`,
          hooks: [async () => { reviews.push(id); return { decision: 'ask' as const }; }],
        });
    const agents: t.AgentInputs[] = ['entry', 'left', 'right', 'result'].map(
      (id) => ({
        agentId: id,
        provider: Providers.OPENAI,
        clientOptions: { modelName: `${id}-model`, apiKey: 'unused' },
        instructions: id,
      })
    );
    for (const id of ['left', 'right'] as const) {
      const member = agents.find((agent) => agent.agentId === id)!;
      const work = tool(execute[id], {
        name: `${id}_tool`,
        description: 'Review work',
        schema: z.object({ value: z.string() }),
      });
      member.graphTools = [
        tool(effects[id], {
          name: `${id}_effect`,
          description: 'Complete effect',
          schema: z.object({}),
        }),
      ];
      if (id === questionMember) member.graphTools.push(work);
      else member.tools = [work];
      if (eventDriven && id !== questionMember)
        member.toolDefinitions = [
          {
            name: `${id}_tool`,
            description: 'Review work',
            parameters: {
              type: 'object',
              properties: { value: { type: 'string' } },
              required: ['value'],
            },
          },
        ];
    }
    const run = await Run.create<t.IState>({
      runId: `root-approval-${Math.random()}`,
      hooks,
      humanInTheLoop: { enabled: true },
      returnContent: true,
      skipCleanup: true,
      graphConfig: {
        type: 'multi-agent',
        agents,
        entryAgentId: 'entry',
        compileOptions: { checkpointer },
        edges: [
          { from: 'entry', to: ['left', 'right'], edgeType: 'direct' },
          { from: ['left', 'right'], to: 'result', edgeType: 'direct' },
        ],
      },
      ...(eventDriven
        ? {
          customHandlers: {
            [GraphEvents.ON_TOOL_EXECUTE]: {
              async handle(_event, data) {
                const batch = data as t.ToolExecuteBatchRequest;
                batch.resolve(
                  await Promise.all(
                    batch.toolCalls.map(async (call) => ({
                      toolCallId: call.id,
                      status: 'success' as const,
                      content: await execute[
                        batch.agentId === 'left' ? 'left' : 'right'
                      ]({ value: String(call.args.value) }),
                    }))
                  )
                );
              },
            },
          },
        }
        : {}),
    });
    run.Graph!.overrideModel = new RootApprovalModel(calls);
    return run;
  };
  return { build, calls, reviews, left, right, leftEffect, rightEffect };
}
async function start(run: Run<t.IState>) {
  await run.processStream(
    { messages: [new HumanMessage('Review work')] },
    config
  );
}

describe('root parallel HITL', () => {
  test.each(
    [false, true].flatMap((eventDriven) =>
      [false, true].flatMap((rebuilt) =>
        ['both', 'left', 'right'].map((addressed) => ({
          eventDriven,
          rebuilt,
          addressed,
        }))
      )
    )
  )(
    'restores root evidence independently: %j',
    async ({ eventDriven, rebuilt, addressed }) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const entry of pending ?? []) {
        if (!isToolApprovalInterrupt(entry.payload))
          throw new Error('Missing root proposal');
        const member = entry.payload.action_requests[0].name.split('_')[0];
        if (addressed === 'both' || member === addressed)
          decisions[entry.interruptId] = [{ type: 'approve' }];
      }
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(decisions, config);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.left).toHaveBeenCalledTimes(addressed === 'right' ? 0 : 1);
      expect(h.right).toHaveBeenCalledTimes(addressed === 'left' ? 0 : 1);
      if (addressed !== 'both') {
        const remaining = resumed.getInterrupt()?.pendingInterrupts;
        expect(remaining).toHaveLength(1);
        const last = remaining?.[0];
        if (last == null) throw new Error('Missing pending sibling');
        const again = await h.build();
        await again.resume(
          { [last.interruptId]: [{ type: 'approve' }] },
          config
        );
        expect(again.getInterrupt()).toBeUndefined();
      } else expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.left).toHaveBeenCalledTimes(1);
      expect(h.right).toHaveBeenCalledTimes(1);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );

  test.each(['left', 'right'] as const)(
    'resumes root question and approval without sharing authorization: %s',
    async (questionMember) => {
      const h = harness(false, questionMember);
      const run = await h.build();
      await start(run);
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions = Object.fromEntries(
        (pending ?? []).map((entry) => [
          entry.interruptId,
          isToolApprovalInterrupt(entry.payload)
            ? [{ type: 'approve' }]
            : { answer: 'clarified' },
        ])
      );
      const rebuilt = await h.build();
      await rebuilt.resume(decisions, config);
      expect(rebuilt.getInterrupt()).toBeUndefined();
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      const question = questionMember === 'left' ? h.left : h.right;
      expect(await question.mock.results.at(-1)?.value).toBe(
        `${questionMember} clarified`
      );
    }
  );

  test.each([false, true])(
    'routes edited and rejected root proposals independently, eventDriven=%s',
    async (eventDriven) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions = Object.fromEntries(
        (pending ?? []).map((entry) => {
          if (!isToolApprovalInterrupt(entry.payload))
            throw new Error('Missing approval');
          return [
            entry.interruptId,
            [
              entry.payload.action_requests[0].name === 'left_tool'
                ? { type: 'edit', updatedInput: { value: 'reviewed-left' } }
                : { type: 'reject', reason: 'No right side effect' },
            ],
          ];
        })
      );
      const rebuilt = await h.build();
      await rebuilt.resume(decisions, config);
      expect(rebuilt.getInterrupt()).toBeUndefined();
      expect(h.left.mock.calls[0][0]).toEqual({ value: 'reviewed-left' });
      expect(h.right).not.toHaveBeenCalled();
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );

  test('direct processStream Commands restore all root review evidence', async () => {
    const h = harness(false);
    const run = await h.build();
    await start(run);
    const pending = run.getInterrupt()?.pendingInterrupts;
    expect(pending).toHaveLength(2);
    const decisions = Object.fromEntries(
      (pending ?? []).map((entry) => [entry.interruptId, [{ type: 'approve' }]])
    );
    const rebuilt = await h.build();
    await rebuilt.processStream(new Command({ resume: decisions }), config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.left).toHaveBeenCalledTimes(1);
    expect(h.right).toHaveBeenCalledTimes(1);
    expect(h.leftEffect).toHaveBeenCalledTimes(1);
    expect(h.rightEffect).toHaveBeenCalledTimes(1);
  });
});

describe('parallel approval resume authority', () => {
  test.each([false, true].flatMap((eventDriven) => [false, true].flatMap((rebuilt) =>
    ['resume', 'command'].flatMap((entrypoint) => ['left', 'both'].map((addressed) => ({ eventDriven, rebuilt, entrypoint, addressed })))
  )))('rejects another principal before reusable hooks or effects: %j', async ({ eventDriven, rebuilt, entrypoint, addressed }) => {
    const h = harness(eventDriven);
    const run = await h.build();
    await start(run);
    const pending = run.getInterrupt()?.pendingInterrupts;
    expect(pending).toHaveLength(2);
    const decisions = Object.fromEntries((pending ?? []).filter((entry) => {
      if (!isToolApprovalInterrupt(entry.payload)) throw new Error('Missing root approval');
      return addressed === 'both' || entry.payload.action_requests[0].name === 'left_tool';
    }).map((entry) => [entry.interruptId, [{ type: 'approve' }]]));
    const resumed = rebuilt ? await h.build() : run;
    const foreignConfig = { ...config, configurable: { ...config.configurable, user_id: 'other-user' } };
    const reviews = h.reviews.length;
    const resume = entrypoint === 'command'
      ? resumed.processStream(new Command({ resume: decisions }), foreignConfig)
      : resumed.resume(decisions, foreignConfig);
    const failure = await resume.then(() => undefined, (error: Error) => error);
    const errors = failure instanceof AggregateError ? failure.errors : [failure];
    expect(errors.length).toBeGreaterThan(0);
    for (const error of errors) expect(error instanceof Error && error.message.includes('execution owner changed')).toBe(true);
    expect(h.reviews).toHaveLength(reviews);
    expect(h.left).not.toHaveBeenCalled();
    expect(h.right).not.toHaveBeenCalled();
    expect(h.leftEffect).toHaveBeenCalledTimes(1);
    expect(h.rightEffect).toHaveBeenCalledTimes(1);
  });
  test.each([false, true])('rejects an execution-scope change on a real pending root, eventDriven=%s', async (eventDriven) => {
    const h = harness(eventDriven);
    const run = await h.build();
    await start(run);
    const pending = run.getInterrupt()?.pendingInterrupts;
    const decisions = Object.fromEntries((pending ?? []).map((entry) => [entry.interruptId, [{ type: 'approve' }]]));
    const resumed = await h.build();
    const failure = await resumed.resume(decisions, { ...config, configurable: { ...config.configurable, __librechat_tool_approval_execution_scope: 'other-scope' } })
      .then(() => undefined, (error: Error) => error);
    const errors = failure instanceof AggregateError ? failure.errors : [failure];
    for (const error of errors) expect(error instanceof Error && error.message.includes('execution owner changed')).toBe(true);
    expect(h.left).not.toHaveBeenCalled();
    expect(h.right).not.toHaveBeenCalled();
  });

});

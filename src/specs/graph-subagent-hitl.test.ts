import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { Interrupt } from '@langchain/langgraph';
import type { SubagentCheckpointReference } from '@/tools/subagent/SubagentReplay';
import type * as t from '@/types';
import { getSubagentResumeManifest } from '@/tools/subagent/SubagentReplay';
import { getPublicToolInterruptPayload } from '@/tools/toolBatchReplay';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { createFakeStreamingLLM } from '@/llm/fake';
import { askUserQuestion } from '@/hitl';
import { HookRegistry } from '@/hooks';
import { Constants, GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const member = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: `${agentId}-model`, apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
});

class ApprovalModel extends FakeListChatModel {
  private readonly owners: Map<string, string>;
  constructor(
    private readonly calls: string[],
    private readonly programmatic = false,
    private readonly sessionTools = false,
    private readonly completedEffects = false
  ) {
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
    const [messages] = args;
    const owner = ['entry', 'left', 'right', 'result'].find(
      (id) => this.owners.get(args[2]?.runId ?? '') === id
    );
    if (owner == null) throw new Error('Missing graph member identity');
    this.calls.push(owner);
    const memberTool =
      this.programmatic && owner === 'left'
        ? Constants.PROGRAMMATIC_TOOL_CALLING
        : `${owner}_tool`;
    const toolName = this.sessionTools ? Constants.EXECUTE_CODE : memberTool;
    const prior = messages.some(
      (message) =>
        message._getType() === 'tool' &&
        (message as ToolMessage).name === toolName
    );
    if ((owner === 'left' || owner === 'right') && !prior) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            ...(this.completedEffects
              ? [
                {
                  name: `${owner}_effect`,
                  args: '{}',
                  id: 'shared-completed-call',
                  index: 0,
                  type: 'tool_call_chunk' as const,
                },
              ]
              : []),
            {
              name: toolName,
              args: JSON.stringify(
                this.programmatic && owner === 'left'
                  ? { code: 'left_tool({value: "programmatic"})' }
                  : { value: 'original' }
              ),
              id: 'shared-member-call',
              index: this.completedEffects ? 1 : 0,
              type: 'tool_call_chunk',
            },
          ],
        }),
      });
      return;
    }
    const text = `${owner} completed`;
    yield new ChatGenerationChunk({ text, message: new AIMessageChunk(text) });
  }
}

function harness(
  eventDriven = false,
  question = false,
  programmatic = false,
  sessionTools = false,
  completedEffects = false
) {
  const checkpointer = new MemorySaver();
  const calls: string[] = [];
  const leftExecute = jest.fn(
    async ({ value }: { value: string }, _config?: ToolRunnableConfig) =>
      question
        ? `left ${askUserQuestion({ question: 'Clarify the review' }, { toolCallId: 'shared-member-call' }).answer}`
        : `left ${value}`
  );
  const rightExecute = jest.fn(
    async ({ value }: { value: string }, _config?: ToolRunnableConfig) =>
      `right ${value}`
  );
  const tools = { left: leftExecute, right: rightExecute };
  const leftEffect = jest.fn(async () => 'left effect');
  const rightEffect = jest.fn(async () => 'right effect');
  const effects = { left: leftEffect, right: rightEffect };
  const eventCalls: t.ToolCallRequest[] = [];
  const programmaticExecute = jest.fn(
    async (_args: { code: string }, runnable?: ToolRunnableConfig) => {
      const cache = runnable?.toolCall as
        | (ToolCall & Partial<t.ProgrammaticCache>)
        | undefined;
      const target = cache?.toolMap?.get('left_tool');
      return target == null
        ? 'left unavailable'
        : target.invoke({ value: 'programmatic' });
    }
  );
  const graph: t.GraphSubagentConfig = {
    kind: 'graph',
    type: 'team',
    name: 'Team',
    description: 'Review in parallel',
    agents: [
      member('entry'),
      ...(['left', 'right'] as const).map((id) => ({
        ...member(id),
        ...(completedEffects
          ? {
            graphTools: [
              tool(effects[id], {
                name: `${id}_effect`,
                description: 'Complete one effect',
                schema: z.object({}),
              }),
            ],
          }
          : {}),
        ...(eventDriven
          ? {
            toolDefinitions: [
              {
                name: sessionTools ? Constants.EXECUTE_CODE : `${id}_tool`,
                description: 'Requires review',
                parameters: {
                  type: 'object' as const,
                  properties: { value: { type: 'string' as const } },
                  required: ['value'],
                },
              },
            ],
          }
          : {}),
        tools: [
          tool(tools[id], {
            name: sessionTools ? Constants.EXECUTE_CODE : `${id}_tool`,
            description: 'Requires review',
            schema: z.object({ value: z.string() }),
          }),
        ],
      })),
      member('result'),
    ],
    edges: [
      { from: 'entry', to: ['left', 'right'], edgeType: 'direct' },
      { from: ['left', 'right'], to: 'result', edgeType: 'direct' },
    ],
    entryAgentId: 'entry',
    resultAgentId: 'result',
  };
  if (programmatic) {
    const target = graph.agents[1].tools?.[0];
    if (target == null) throw new Error('Missing target tool');
    graph.agents[1].tools = [
      target,
      tool(programmaticExecute, {
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        description: 'Run tools with code',
        schema: z.object({ code: z.string() }),
      }),
    ];
    graph.agents[1].toolRegistry = new Map([
      ['left_tool', { name: 'left_tool', allowed_callers: ['direct'] }],
      [
        Constants.PROGRAMMATIC_TOOL_CALLING,
        {
          name: Constants.PROGRAMMATIC_TOOL_CALLING,
          allowed_callers: ['direct'],
        },
      ],
    ]);
  }
  const build = async (
    definition = graph,
    signal?: AbortSignal,
    subagentContext?: t.SubagentContextAdapter
  ) => {
    const hookRegistry = new HookRegistry();
    for (const id of ['left', 'right']) {
      const memberTool =
        programmatic && id === 'left'
          ? Constants.PROGRAMMATIC_TOOL_CALLING
          : `${id}_tool`;
      hookRegistry.register('PreToolUse', {
        pattern: sessionTools ? Constants.EXECUTE_CODE : memberTool,
        hooks: [async () => ({ decision: 'ask' as const })],
      });
    }
    const run = await Run.create<t.IState>({
      runId: `graph-approval-${Math.random()}`,
      graphConfig: {
        type: 'standard',
        signal,
        compileOptions: { checkpointer },
        agents: [
          {
            ...member('parent'),
            subagentConfigs: [definition],
            maxSubagentDepth: 1,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
      subagentContext,
      ...(eventDriven
        ? {
          customHandlers: {
            [GraphEvents.ON_TOOL_EXECUTE]: {
              async handle(_event, rawData) {
                const batch = rawData as t.ToolExecuteBatchRequest;
                eventCalls.push(...batch.toolCalls);
                batch.resolve(
                  await Promise.all(
                    batch.toolCalls.map(async (call) => {
                      const executor =
                          call.name === 'left_tool' ||
                          (sessionTools && batch.agentId === 'left')
                            ? leftExecute
                            : rightExecute;
                      const result = await executor({
                        value: String(call.args.value),
                      });
                      return {
                        toolCallId: call.id,
                        content: result,
                        status: 'success' as const,
                      };
                    })
                  )
                );
              },
            },
          },
        }
        : {}),
      hooks: hookRegistry,
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph?.setSubagentModelOverride(
      new ApprovalModel(calls, programmatic, sessionTools, completedEffects)
    );
    run.Graph!.overrideModel = createFakeStreamingLLM({
      responses: ['', 'parent completed'],
      toolCalls: [
        {
          id: 'parent-spawn',
          name: Constants.SUBAGENT,
          args: { description: 'Review the task', subagent_type: 'team' },
          type: 'tool_call',
        },
      ],
    });
    return run;
  };
  return {
    checkpointer,
    graph,
    calls,
    build,
    leftExecute,
    rightExecute,
    programmaticExecute,
    leftEffect,
    rightEffect,
    eventCalls,
  };
}
const config = {
  configurable: { thread_id: 'graph-approval-thread' },
  version: 'v2' as const,
  streamMode: 'values',
};
async function start(run: Run<t.IState>) {
  await run.processStream(
    { messages: [new HumanMessage('Review work')] },
    config
  );
}

async function pendingMemberInterrupts(
  checkpointer: MemorySaver
): Promise<Interrupt[]> {
  let reference: SubagentCheckpointReference | undefined;
  for await (const parent of checkpointer.list(config)) {
    const interrupted = parent.pendingWrites?.find(
      ([, channel]) => channel === '__interrupt__'
    )?.[2] as Interrupt | undefined;
    const manifest = getSubagentResumeManifest(interrupted?.value);
    reference = manifest?.executions[0].checkpoints.find(
      (checkpoint) => checkpoint.checkpointNs === ''
    );
    if (reference != null) break;
  }
  if (reference == null) throw new Error('Missing child checkpoint');
  const child = await checkpointer.getTuple({
    configurable: {
      thread_id: reference.threadId,
      checkpoint_ns: reference.checkpointNs,
      checkpoint_id: reference.checkpointId,
    },
  });
  const pending =
    child?.pendingWrites?.flatMap(([, channel, value]) =>
      channel === '__interrupt__' ? [value as Interrupt] : []
    ) ?? [];
  expect(pending.map((interrupt) => interrupt.id)).toHaveLength(2);
  return pending;
}

describe('graph subagent foreground HITL', () => {
  test('keeps parallel approvals pending and resumes a rebuilt run without repeating completed members', async () => {
    const h = harness();
    const run = await h.build();
    await start(run);
    const first = run.getInterrupt();
    expect(first?.payload.type).toBe('tool_approval');
    if (first?.payload.type !== 'tool_approval')
      throw new Error('Missing member approval');
    expect(['left', 'right']).toContain(first.payload.subagent?.agent_id);
    expect(first.payload.subagent).toMatchObject({
      subagent_type: 'team',
      parent_tool_call_id: 'parent-spawn',
    });
    expect(h.leftExecute).not.toHaveBeenCalled();
    expect(h.rightExecute).not.toHaveBeenCalled();
    await run.resume([{ type: 'approve' }], config);
    expect(
      h.leftExecute.mock.calls.length + h.rightExecute.mock.calls.length
    ).toBe(1);
    expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
    const rebuilt = await h.build();
    await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    const returned = rebuilt
      .getRunMessages()
      ?.filter(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).name === Constants.SUBAGENT
      );
    expect(returned?.[0].content).toBe('result completed');
  });

  test.each(
    [false, true].flatMap((eventDriven) =>
      [false, true].flatMap((rebuilt) =>
        ['left', 'right', 'both'].map((addressed) => ({
          eventDriven,
          rebuilt,
          addressed,
        }))
      )
    )
  )(
    'preserves settled effects for unresumed members: eventDriven=$eventDriven rebuilt=$rebuilt addressed=$addressed',
    async ({ eventDriven, rebuilt, addressed }) => {
      const h = harness(eventDriven, false, false, false, true);
      const run = await h.build();
      await start(run);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
      const pending = await pendingMemberInterrupts(h.checkpointer);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const entry of pending) {
        if (entry.id == null) throw new Error('Missing interrupt ID');
        const payload = getPublicToolInterruptPayload(entry.value);
        if (!isToolApprovalInterrupt(payload))
          throw new Error('Missing approval');
        const memberId = payload.subagent?.agent_id;
        if (addressed === 'both' || memberId === addressed)
          decisions[entry.id] = [{ type: 'approve' }];
      }
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(decisions, config);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.leftExecute).toHaveBeenCalledTimes(
        addressed === 'right' ? 0 : 1
      );
      expect(h.rightExecute).toHaveBeenCalledTimes(
        addressed === 'left' ? 0 : 1
      );
      let final = resumed;
      if (addressed !== 'both') {
        expect(resumed.getInterrupt()?.payload.type).toBe('tool_approval');
        expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
        final = await h.build();
        await final.resume([{ type: 'approve' }], config);
      }
      expect(final.getInterrupt()).toBeUndefined();
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
      expect(
        final
          .getRunMessages()
          ?.find(
            (message) =>
              message._getType() === 'tool' &&
              (message as ToolMessage).name === Constants.SUBAGENT
          )?.content
      ).toBe('result completed');
    }
  );

  test.each(
    [false, true].flatMap((eventDriven) =>
      [false, true].map((rebuilt) => ({ eventDriven, rebuilt }))
    )
  )(
    'resumes both member approvals together: eventDriven=$eventDriven rebuilt=$rebuilt',
    async ({ eventDriven, rebuilt }) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const pending = await pendingMemberInterrupts(h.checkpointer);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const interrupt of pending) {
        if (interrupt.id == null) throw new Error('Missing interrupt ID');
        const payload = getPublicToolInterruptPayload(interrupt.value);
        if (!isToolApprovalInterrupt(payload))
          throw new Error('Missing member approval');
        expect(payload.action_requests[0].tool_call_id).toBe(
          'shared-member-call'
        );
        decisions[interrupt.id] = [{ type: 'approve' }];
      }
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(decisions, config);
      expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
      expect(
        resumed
          .getRunMessages()
          ?.find(
            (message) =>
              message._getType() === 'tool' &&
              (message as ToolMessage).name === Constants.SUBAGENT
          )?.content
      ).toBe('result completed');
    }
  );

  test.each(
    [false, true].flatMap((eventDriven) =>
      ['edit-reject', 'respond-approve', 'partial'].map((mode) => ({
        eventDriven,
        mode,
      }))
    )
  )(
    'keeps batched decisions member-scoped: eventDriven=$eventDriven mode=$mode',
    async ({ eventDriven, mode }) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const pending = await pendingMemberInterrupts(h.checkpointer);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const entry of pending) {
        if (entry.id == null) throw new Error('Missing interrupt ID');
        const payload = getPublicToolInterruptPayload(entry.value);
        if (!isToolApprovalInterrupt(payload))
          throw new Error('Missing approval');
        const isLeft = payload.subagent?.agent_id === 'left';
        if (mode === 'partial' && !isLeft) continue;
        let decision: t.ToolApprovalDecision = { type: 'approve' };
        if (mode === 'edit-reject')
          decision = isLeft
            ? { type: 'edit', updatedInput: { value: 'edited-left' } }
            : { type: 'reject', reason: 'No right effect' };
        if (mode === 'respond-approve' && isLeft)
          decision = { type: 'respond', responseText: 'Manual left result' };
        decisions[entry.id] = [decision];
      }
      const rebuilt = await h.build();
      await rebuilt.resume(decisions, config);
      if (mode === 'partial') {
        expect(h.leftExecute).toHaveBeenCalledTimes(1);
        expect(h.rightExecute).not.toHaveBeenCalled();
        expect(rebuilt.getInterrupt()?.payload.type).toBe('tool_approval');
        const again = await h.build();
        await again.resume([{ type: 'approve' }], config);
        expect(again.getInterrupt()).toBeUndefined();
        expect(h.leftExecute).toHaveBeenCalledTimes(1);
        expect(h.rightExecute).toHaveBeenCalledTimes(1);
      } else {
        expect(rebuilt.getInterrupt()).toBeUndefined();
        expect(h.leftExecute).toHaveBeenCalledTimes(
          mode === 'edit-reject' ? 1 : 0
        );
        expect(h.rightExecute).toHaveBeenCalledTimes(
          mode === 'edit-reject' ? 0 : 1
        );
        if (mode === 'edit-reject')
          expect(h.leftExecute.mock.calls[0][0]).toEqual({
            value: 'edited-left',
          });
      }
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );

  test('a question after batched approval resumes without repeating its completed sibling', async () => {
    const h = harness(false, true);
    const run = await h.build();
    await start(run);
    const pending = await pendingMemberInterrupts(h.checkpointer);
    const decisions = Object.fromEntries(
      pending.map((entry) => [entry.id!, [{ type: 'approve' }]])
    );
    await run.resume(decisions, config);
    expect(run.getInterrupt()?.payload.type).toBe('ask_user_question');
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    const rebuilt = await h.build();
    await rebuilt.resume({ answer: 'batch clarified' }, config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    expect(await h.leftExecute.mock.results.at(-1)?.value).toBe(
      'left batch clarified'
    );
  });

  test.each([false, true])(
    'approve, edit, reject, and respond preserve member side effects (eventDriven=%s)',
    async (eventDriven) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const first = run.getInterrupt();
      if (first?.payload.type !== 'tool_approval')
        throw new Error('Missing initial member approval');
      const firstOwner = first.payload.subagent?.agent_id;
      const edited = firstOwner === 'left' ? h.leftExecute : h.rightExecute;
      const rejected = firstOwner === 'left' ? h.rightExecute : h.leftExecute;
      await run.resume(
        [{ type: 'edit', updatedInput: { value: 'reviewed' } }],
        config
      );
      expect(edited).toHaveBeenCalledTimes(1);
      expect(edited.mock.calls[0][0]).toEqual({ value: 'reviewed' });
      const second = run.getInterrupt();
      if (second?.payload.type !== 'tool_approval')
        throw new Error('Sibling approval was lost');
      expect(second.payload.subagent?.agent_id).not.toBe(firstOwner);
      const rebuilt = await h.build();
      await rebuilt.resume(
        [{ type: 'reject', reason: 'Denied by reviewer' }],
        config
      );
      expect(rejected).not.toHaveBeenCalled();
      expect(edited).toHaveBeenCalledTimes(1);
      expect(rebuilt.getInterrupt()).toBeUndefined();

      const alternate = harness(eventDriven);
      const alternateRun = await alternate.build();
      await start(alternateRun);
      await alternateRun.resume(
        [{ type: 'respond', responseText: 'Manual result' }],
        config
      );
      expect(alternate.leftExecute).not.toHaveBeenCalled();
      expect(alternate.rightExecute).not.toHaveBeenCalled();
      await alternateRun.resume([{ type: 'approve' }], config);
      expect(
        alternate.leftExecute.mock.calls.length +
          alternate.rightExecute.mock.calls.length
      ).toBe(1);
      expect(alternateRun.getInterrupt()).toBeUndefined();
    }
  );

  test.each([
    'prompt-kind',
    'topology',
    'instructions',
    'model',
    'revision',
    'tool-schema',
    'direct-tool-schema',
    'registry-callers',
    'registry-schema',
    'registry-deferred',
    'registry-response-format',
    'registry-server',
    'registry-classification',
    'registry-removal',
    'registry-key',
    'tool-map-schema',
    'direct-tool-mode',
    'session-partition',
    'tool-end',
    'summarize-only',
  ] as const)(
    'a rebuilt run rejects a changed graph %s before approved execution',
    async (change) => {
      const h = harness();
      const originalTool = h.graph.agents[1].tools?.[0];
      if (originalTool == null) throw new Error('Missing member tool');
      h.graph.agents[1].toolRegistry = new Map([
        [
          'left_tool',
          {
            name: 'left_tool',
            description: 'Requires review',
            parameters: {
              type: 'object',
              properties: { value: { type: 'string' } },
            },
            allowed_callers: ['direct'],
            defer_loading: false,
            responseFormat: 'content',
            serverName: 'original-server',
            toolType: 'mcp',
          },
        ],
      ]);
      if (change === 'prompt-kind') {
        h.graph.configId = 'stable-revision';
        h.graph.edges[1].prompt = () => 'Original prompt';
      }
      const run = await h.build();
      await start(run);
      expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
      const graph: t.GraphSubagentConfig = {
        ...h.graph,
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      if (change === 'prompt-kind') graph.edges = h.graph.edges.map((edge, index) => index === 1 ? { ...edge, prompt: 'Different static prompt' } : edge);
      if (change === 'topology')
        graph.edges = [
          { from: 'entry', to: 'right', edgeType: 'direct' },
          { from: 'right', to: 'left', edgeType: 'direct' },
          { from: 'left', to: 'result', edgeType: 'direct' },
        ];
      if (change === 'session-partition')
        graph.agents[1].codeSessionKey = 'different-partition';
      if (change === 'tool-end') graph.agents[1].toolEnd = true;
      if (change === 'summarize-only') graph.agents[1].summarizeOnly = true;
      if (change === 'instructions')
        graph.agents[1].instructions = 'Changed member permissions';
      if (change === 'model')
        graph.agents[1].clientOptions = {
          ...graph.agents[1].clientOptions,
          modelName: 'different-model',
        };
      if (change === 'revision') graph.configId = 'updated-host-member-version';
      if (change === 'tool-schema')
        graph.agents[1].toolDefinitions = [
          {
            name: 'left_tool',
            description: 'New tool',
            parameters: {
              type: 'object' as const,
              properties: { path: { type: 'string' as const } },
            },
          },
        ];
      if (change === 'direct-tool-schema')
        graph.agents[1].tools = [
          tool(h.leftExecute, {
            name: 'left_tool',
            description: 'Changed schema',
            schema: z.object({ value: z.string(), path: z.string() }),
          }),
        ];
      const originalDeclaration =
        h.graph.agents[1].toolRegistry.get('left_tool');
      if (originalDeclaration == null)
        throw new Error('Missing registry declaration');
      const declaration = { ...originalDeclaration };
      if (change === 'registry-callers')
        declaration.allowed_callers = ['direct', 'code_execution'];
      if (change === 'registry-schema')
        declaration.parameters = {
          type: 'object',
          properties: { path: { type: 'string' } },
        };
      if (change === 'registry-deferred') declaration.defer_loading = true;
      if (change === 'registry-response-format')
        declaration.responseFormat = 'content_and_artifact';
      if (change === 'registry-server') declaration.serverName = 'new-server';
      if (change === 'registry-classification') declaration.toolType = 'action';
      graph.agents[1].toolRegistry = new Map([
        [change === 'registry-key' ? 'renamed-key' : 'left_tool', declaration],
      ]);
      if (change === 'registry-removal') graph.agents[1].toolRegistry.clear();
      if (change === 'tool-map-schema')
        graph.agents[1].toolMap = new Map([
          [
            'left_tool',
            tool(h.leftExecute, {
              name: 'left_tool',
              description: 'New map implementation',
              schema: z.object({ value: z.string(), path: z.string() }),
            }),
          ],
        ]);
      if (change === 'direct-tool-mode') {
        graph.agents[1].graphTools = graph.agents[1].tools as t.GenericTool[];
        graph.agents[1].tools = [];
      }
      const rebuilt = await h.build(graph);
      await rebuilt.resume([{ type: 'approve' }], config);
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(
        rebuilt
          .getRunMessages()
          ?.some(
            (message) =>
              message._getType() === 'tool' &&
              String(message.content).includes('changed')
          )
      ).toBe(true);
    }
  );

  test('graph children retain the foreground requirement for approvals', async () => {
    const h = harness();
    const run = await h.build();
    const executorTool = (
      run.Graph?.agentContexts.get('parent')?.graphTools as
        | t.GenericTool[]
        | undefined
    )?.find((candidate) => candidate.name === Constants.SUBAGENT);
    if (!executorTool) throw new Error('Missing subagent tool');
    const output = await executorTool.invoke(
      {
        type: 'tool_call',
        name: Constants.SUBAGENT,
        id: 'background-spawn',
        args: {
          description: 'Work',
          subagent_type: 'team',
          run_in_background: true,
        },
      },
      config
    );
    expect(String(output.content)).toMatch(/background/i);
    expect(h.leftExecute).not.toHaveBeenCalled();
    expect(h.rightExecute).not.toHaveBeenCalled();
  });

  test('question interrupts inside a team survive a fresh Run without bypassing member approvals', async () => {
    const h = harness(false, true);
    let run = await h.build();
    await start(run);
    let questions = 0;
    for (let attempt = 0; attempt < 6 && run.getInterrupt(); attempt++) {
      const interrupt = run.getInterrupt();
      if (interrupt?.payload.type === 'ask_user_question') {
        expect(interrupt.payload.question.question).toBe('Clarify the review');
        expect(interrupt.payload.tool_call_id).toBe('shared-member-call');
        questions++;
        run = await h.build();
        await run.resume({ answer: 'clarified' }, config);
      } else await run.resume([{ type: 'approve' }], config);
    }
    expect(questions).toBe(1);
    expect(run.getInterrupt()).toBeUndefined();
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(
      await h.leftExecute.mock.results[h.leftExecute.mock.results.length - 1]
        .value
    ).toBe('left clarified');
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
  });

  test('cancelling a paused foreground team cannot execute its waiting tool calls', async () => {
    const h = harness();
    const abort = new AbortController();
    const run = await h.build(h.graph, abort.signal);
    await start(run);
    expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
    abort.abort(new Error('Cancelled graph review'));
    await run.resume([{ type: 'approve' }], config);
    expect(h.leftExecute).not.toHaveBeenCalled();
    expect(h.rightExecute).not.toHaveBeenCalled();
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
  });

  test('one decision cannot approve pending calls in both graph branches', async () => {
    const h = harness();
    const run = await h.build();
    await start(run);
    const firstId = run.getInterrupt()?.interruptId;
    await run.resume({ 'shared-member-call': { type: 'approve' } }, config);
    expect(
      h.leftExecute.mock.calls.length + h.rightExecute.mock.calls.length
    ).toBe(1);
    expect(run.getInterrupt()?.interruptId).not.toBe(firstId);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
  });

  test('registry entry and allowed-caller order do not invalidate unchanged capabilities', async () => {
    const h = harness();
    const definitions: Array<[string, t.LCTool]> = [
      [
        'left_tool',
        { name: 'left_tool', allowed_callers: ['direct', 'code_execution'] },
      ],
      ['other_tool', { name: 'other_tool', allowed_callers: ['direct'] }],
    ];
    h.graph.agents[1].toolRegistry = new Map(definitions);
    const run = await h.build();
    await start(run);
    const reordered = {
      ...h.graph,
      agents: h.graph.agents.map((agent) => ({ ...agent })),
    };
    reordered.agents[1].toolRegistry = new Map(
      definitions
        .slice()
        .reverse()
        .map(([key, declaration]) => [
          key,
          {
            ...declaration,
            allowed_callers: declaration.allowed_callers?.slice().reverse(),
          },
        ])
    );
    const rebuilt = await h.build(reordered);
    for (let attempt = 0; attempt < 2; attempt++)
      await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    expect(definitions[0][1].allowed_callers).toEqual([
      'direct',
      'code_execution',
    ]);
  });

  test.each([false, true])(
    'programmatic approval cannot gain an expanded registry, changed=%s',
    async (changed) => {
      const h = harness(false, false, true);
      const run = await h.build();
      await start(run);
      expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
      expect(h.programmaticExecute).not.toHaveBeenCalled();
      const resumed = {
        ...h.graph,
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      if (changed) {
        resumed.agents[1].toolRegistry = new Map(
          h.graph.agents[1].toolRegistry
        );
        resumed.agents[1].toolRegistry.set('left_tool', {
          name: 'left_tool',
          allowed_callers: ['direct', 'code_execution'],
        });
      }
      const rebuilt = await h.build(resumed);
      await rebuilt.resume([{ type: 'approve' }], config);
      if (changed) {
        expect(h.programmaticExecute).not.toHaveBeenCalled();
        expect(h.rightExecute).not.toHaveBeenCalled();
        expect(
          rebuilt
            .getRunMessages()
            ?.some(
              (message) =>
                message._getType() === 'tool' &&
                String(message.content).includes('changed')
            )
        ).toBe(true);
      } else {
        await rebuilt.resume([{ type: 'approve' }], config);
        expect(rebuilt.getInterrupt()).toBeUndefined();
        expect(h.programmaticExecute).toHaveBeenCalledTimes(1);
        expect(h.rightExecute).toHaveBeenCalledTimes(1);
      }
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(
        h.graph.agents[1].toolRegistry?.get('left_tool')?.allowed_callers
      ).toEqual(['direct']);
    }
  );

  test.each([
    ...[false, true].flatMap((eventDriven) => [
      { eventDriven, source: 'declaration', rebuilt: true, changed: false },
      { eventDriven, source: 'declaration', rebuilt: true, changed: true },
      ...[false, true].flatMap((rebuilt) => [
        { eventDriven, source: 'adapter', rebuilt, changed: false },
        { eventDriven, source: 'adapter', rebuilt, changed: true },
        { eventDriven, source: 'removed-adapter', rebuilt, changed: true },
      ]),
    ]),
  ])(
    'session authority survives resume: eventDriven=$eventDriven source=$source rebuilt=$rebuilt changed=$changed',
    async ({ eventDriven, source, rebuilt, changed }) => {
      const h = harness(eventDriven, false, false, true);
      const seed = (suffix: string): t.ToolSessionMap =>
        new Map([
          [
            Constants.EXECUTE_CODE,
            {
              session_id: `session-${suffix}`,
              lastUpdated: 1,
              files: [
                {
                  id: `file-${suffix}`,
                  name: `${suffix}.txt`,
                  storage_session_id: `storage-${suffix}`,
                },
              ],
            },
          ],
        ]);
      const leftSeed = seed('A');
      const rightSeed = seed('B');
      h.graph.agents[1].codeSessionKey =
        source === 'declaration' ? 'partition-A' : 'declared-A';
      h.graph.agents[1].initialSessions = leftSeed;
      h.graph.agents[2].codeSessionKey = 'partition-B';
      h.graph.agents[2].initialSessions = rightSeed;
      let partition = 'partition-A';
      let removed = false;
      const adapter: t.SubagentContextAdapter = {
        prepare: async () =>
          removed
            ? {}
            : {
              agentSessions: {
                left: {
                  codeSessionKey: partition,
                  initialSessions: leftSeed,
                },
                right: {
                  codeSessionKey: 'partition-B',
                  initialSessions: rightSeed,
                },
              },
            },
      };
      const context = source === 'declaration' ? undefined : adapter;
      const run = await h.build(h.graph, undefined, context);
      await start(run);
      expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
      const definition = {
        ...h.graph,
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      if (changed && source === 'declaration')
        definition.agents[1].codeSessionKey = 'partition-B';
      if (changed && source === 'adapter') partition = 'partition-B';
      if (source === 'removed-adapter') removed = true;
      const resumed = rebuilt
        ? await h.build(definition, undefined, context)
        : run;
      await resumed.resume([{ type: 'approve' }], config);
      if (changed) {
        expect(h.leftExecute).not.toHaveBeenCalled();
        expect(h.rightExecute).not.toHaveBeenCalled();
        expect(h.eventCalls).toHaveLength(0);
        expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
        return;
      }
      await resumed.resume([{ type: 'approve' }], config);
      expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
      if (eventDriven) {
        expect(h.eventCalls.map((call) => call.codeSessionContext)).toEqual(
          expect.arrayContaining([
            {
              session_id: 'session-A',
              files: [
                expect.objectContaining({
                  id: 'file-A',
                  storage_session_id: 'storage-A',
                }),
              ],
            },
            {
              session_id: 'session-B',
              files: [
                expect.objectContaining({
                  id: 'file-B',
                  storage_session_id: 'storage-B',
                }),
              ],
            },
          ])
        );
      } else {
        expect(h.leftExecute.mock.calls[0][1]).toEqual(
          expect.objectContaining({
            toolCall: expect.objectContaining({
              session_id: 'session-A',
              _injected_files: [
                expect.objectContaining({
                  id: 'file-A',
                  storage_session_id: 'storage-A',
                }),
              ],
            }),
          })
        );
        expect(h.rightExecute.mock.calls[0][1]).toEqual(
          expect.objectContaining({
            toolCall: expect.objectContaining({
              session_id: 'session-B',
              _injected_files: [
                expect.objectContaining({
                  id: 'file-B',
                  storage_session_id: 'storage-B',
                }),
              ],
            }),
          })
        );
      }
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );

  test('an explicit default member partition resumes the same saved declaration', async () => {
    const h = harness();
    const run = await h.build();
    await start(run);
    const definition = {
      ...h.graph,
      agents: h.graph.agents.map((agent) => ({
        ...agent,
        codeSessionKey: Constants.EXECUTE_CODE,
      })),
    };
    const rebuilt = await h.build(definition);
    for (let attempt = 0; attempt < 2; attempt++)
      await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
  });

  test.each([false, true])(
    'functional prompt revisions protect rebuilt approval, changed=%s',
    async (changed) => {
      const h = harness();
      h.graph.configId = 'prompt-v1';
      h.graph.edges[1].prompt = () => 'Original transition';
      const run = await h.build();
      await start(run);
      const definition = {
        ...h.graph,
        configId: changed ? 'prompt-v2' : 'prompt-v1',
        edges: h.graph.edges.map((edge) => ({ ...edge })),
      };
      definition.edges[1].prompt = () =>
        changed ? 'Changed transition' : 'Original transition';
      const rebuilt = await h.build(definition);
      await rebuilt.resume([{ type: 'approve' }], config);
      if (changed) {
        expect(h.leftExecute).not.toHaveBeenCalled();
        expect(h.rightExecute).not.toHaveBeenCalled();
        return;
      }
      await rebuilt.resume([{ type: 'approve' }], config);
      expect(rebuilt.getInterrupt()).toBeUndefined();
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
    }
  );

  test('functional prompts require a host revision before resumable graph work starts', async () => {
    const h = harness();
    h.graph.edges[1].prompt = () => 'Unversioned transition';
    const run = await h.build();
    await start(run);
    expect(h.calls).toHaveLength(0);
    expect(h.leftExecute).not.toHaveBeenCalled();
    expect(h.rightExecute).not.toHaveBeenCalled();
    expect(
      run
        .getRunMessages()
        ?.some((message) => String(message.content).includes('configId'))
    ).toBe(true);
  });

  test.each(
    [false, true].flatMap((eventDriven) =>
      [false, true].map((rebuilt) => ({ eventDriven, rebuilt }))
    )
  )(
    'refreshed authorized seeds merge after checkpoint restore: eventDriven=$eventDriven rebuilt=$rebuilt',
    async ({ eventDriven, rebuilt }) => {
      const h = harness(eventDriven, false, false, true);
      let refresh = false;
      const original: t.ToolSessionMap = new Map([
        [
          Constants.EXECUTE_CODE,
          {
            session_id: 'original-session',
            lastUpdated: 1,
            files: [
              {
                id: 'original-file',
                name: 'original.txt',
                storage_session_id: 'original-storage',
              },
            ],
          },
        ],
      ]);
      const refreshed: t.ToolSessionMap = new Map([
        [
          Constants.EXECUTE_CODE,
          {
            session_id: 'fresh-session',
            lastUpdated: 2,
            files: [
              {
                id: 'fresh-file',
                name: 'fresh.txt',
                storage_session_id: 'fresh-storage',
              },
            ],
          },
        ],
      ]);
      const adapter: t.SubagentContextAdapter = {
        prepare: async () => ({
          agentSessions: {
            left: {
              codeSessionKey: 'left-partition',
              initialSessions: refresh ? refreshed : original,
            },
            right: {
              codeSessionKey: 'right-partition',
              initialSessions: original,
            },
          },
        }),
      };
      const run = await h.build(h.graph, undefined, adapter);
      await start(run);
      refresh = true;
      const resumed = rebuilt
        ? await h.build(h.graph, undefined, adapter)
        : run;
      await resumed.resume([{ type: 'approve' }], config);
      await resumed.resume([{ type: 'approve' }], config);
      expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
      const files = eventDriven
        ? h.eventCalls.find(
          (call) =>
            call.name === Constants.EXECUTE_CODE &&
              call.codeSessionContext?.files?.some(
                (file) => file.id === 'fresh-file'
              ) === true
        )?.codeSessionContext
        : h.leftExecute.mock.calls[0][1]?.toolCall;
      expect(files).toEqual(
        expect.objectContaining(
          eventDriven
            ? {
              session_id: 'original-session',
              files: [
                expect.objectContaining({ id: 'original-file' }),
                expect.objectContaining({ id: 'fresh-file' }),
              ],
            }
            : {
              session_id: 'original-session',
              _injected_files: [
                expect.objectContaining({ id: 'original-file' }),
                expect.objectContaining({ id: 'fresh-file' }),
              ],
            }
        )
      );
      expect(original.get(Constants.EXECUTE_CODE)?.files).toHaveLength(1);
      expect(refreshed.get(Constants.EXECUTE_CODE)?.files).toHaveLength(1);
    }
  );
});

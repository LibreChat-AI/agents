import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { END, START, Command, Annotation, StateGraph, MemorySaver, MessagesAnnotation, interrupt } from '@langchain/langgraph';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { Interrupt, StateSnapshot } from '@langchain/langgraph';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { getSubagentResumeManifest, SUBAGENT_RESUME_MANIFEST_CONFIG_KEY, SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY } from '@/tools/subagent/SubagentReplay';
import { attachRunStepResumeState, getRunStepResumeState, stripRunStepResumeState } from '@/tools/runStepResume';
import { getPublicToolInterruptPayload } from '@/tools/toolBatchReplay';
import { Constants, GraphEvents, Providers } from '@/common';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { createFakeStreamingLLM } from '@/llm/fake';
import { askUserQuestion } from '@/hitl';
import { HookRegistry } from '@/hooks';
import { Graph } from '@/graphs/Graph';
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
  completedEffects = false,
  mixedQuestion?: 'left' | 'right',
  approvalBarrier?: Promise<void>
) {
  let policy: 'ask' | 'allow' | 'none' = 'ask';
  const checkpointer = new MemorySaver();
  const calls: string[] = [];
  const leftExecute = jest.fn(
    async ({ value }: { value: string }, _config?: ToolRunnableConfig) =>
      question || mixedQuestion === 'left'
        ? `left ${askUserQuestion({ question: 'Clarify the review' }, { toolCallId: 'shared-member-call' }).answer}`
        : `left ${value}`
  );
  const rightExecute = jest.fn(
    async ({ value }: { value: string }, _config?: ToolRunnableConfig) =>
      mixedQuestion === 'right'
        ? `right ${askUserQuestion({ question: 'Clarify the review' }, { toolCallId: 'shared-member-call' }).answer}`
        : `right ${value}`
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
    configId: 'team-v1',
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
  if (mixedQuestion != null) {
    const member = graph.agents.find(
      (agent) => agent.agentId === mixedQuestion
    );
    if (member == null) throw new Error('Missing question member');
    member.graphTools = [
      tool(tools[mixedQuestion], {
        name: `${mixedQuestion}_tool`,
        description: 'Ask a question',
        schema: z.object({ value: z.string() }),
      }),
    ];
    member.tools = [];
    member.toolDefinitions = undefined;
  }
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
    subagentContext?: t.SubagentContextAdapter,
    runId = `graph-approval-${Math.random()}`
  ) => {
    const hookRegistry = new HookRegistry();
    for (const id of ['left', 'right']) {
      if (id === mixedQuestion || policy === 'none') continue;
      const memberTool =
        programmatic && id === 'left'
          ? Constants.PROGRAMMATIC_TOOL_CALLING
          : `${id}_tool`;
      hookRegistry.register('PreToolUse', {
        pattern: sessionTools ? Constants.EXECUTE_CODE : memberTool,
        hooks: [async () => {
          if (approvalBarrier != null) await approvalBarrier;
          return { decision: policy === 'ask' ? 'ask' as const : 'allow' as const };
        }],
      });
    }
    const run = await Run.create<t.IState>({
      runId,
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
      ...(policy === 'none' ? {} : { hooks: hookRegistry }),
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
    setPolicy: (next: typeof policy) => { policy = next; },
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
  run: Run<t.IState>
): Promise<Interrupt[]> {
  const pending = run.getInterrupt()?.pendingInterrupts;
  expect(pending).toHaveLength(2);
  return pending!.map(({ interruptId, payload }) => ({
    id: interruptId,
    value: payload,
  }));
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
      const pending = await pendingMemberInterrupts(run);
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
      const pending = await pendingMemberInterrupts(run);
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
      const pending = await pendingMemberInterrupts(run);
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
    const pending = await pendingMemberInterrupts(run);
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
    'discovered-tools',
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
      if (change === 'prompt-kind')
        graph.edges = h.graph.edges.map((edge, index) =>
          index === 1 ? { ...edge, prompt: 'Different static prompt' } : edge
        );
      if (change === 'discovered-tools')
        graph.agents[1].discoveredTools = ['new-deferred-tool'];
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
    delete h.graph.configId;
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

  test('normalizes discovered-tool declarations without exposing parent discovery state', async () => {
    const h = harness();
    h.graph.agents[1].discoveredTools = ['first', 'second', 'first'];
    const run = await h.build();
    await start(run);
    const definition = {
      ...h.graph,
      agents: h.graph.agents.map((agent) => ({ ...agent })),
    };
    definition.agents[1].discoveredTools = ['second', 'first'];
    const rebuilt = await h.build(definition);
    await rebuilt.resume([{ type: 'approve' }], config);
    await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
  });

  test.each([false, true])(
    'public pending proposals are detached and survive partial rebuilt resume, eventDriven=%s',
    async (eventDriven) => {
      const h = harness(eventDriven);
      const run = await h.build();
      await start(run);
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions: Record<string, t.ToolApprovalDecision[]> = {};
      for (const item of pending ?? []) {
        if (!isToolApprovalInterrupt(item.payload))
          throw new Error('Missing public approval');
        expect(item.payload.action_requests[0].tool_call_id).toBe(
          'shared-member-call'
        );
        expect(item.payload.review_configs[0].allowed_decisions).toContain(
          'approve'
        );
        decisions[item.interruptId] = [{ type: 'approve' }];
        item.payload.action_requests[0].arguments.value = 'mutated';
      }
      expect(JSON.stringify(run.getInterrupt())).not.toMatch(
        /__librechat|approvalReplays|checkpoints/
      );
      const first = pending?.[0];
      if (first == null) throw new Error('Missing first pending approval');
      const rebuilt = await h.build();
      await rebuilt.resume(
        { [first.interruptId]: decisions[first.interruptId] },
        config
      );
      const remaining = rebuilt.getInterrupt()?.pendingInterrupts;
      expect(remaining).toHaveLength(1);
      expect(remaining?.[0].interruptId).not.toBe(first.interruptId);
      const last = remaining?.[0];
      if (last == null) throw new Error('Missing last pending approval');
      const again = await h.build();
      await again.resume({ [last.interruptId]: [{ type: 'approve' }] }, config);
      expect(again.getInterrupt()).toBeUndefined();
      expect(h.leftExecute.mock.calls[0][0]).toEqual({ value: 'original' });
      expect(h.rightExecute.mock.calls[0][0]).toEqual({ value: 'original' });
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
    }
  );

  test.each([
    { label: 'temperature', options: { temperature: 0.8 } },
    {
      label: 'thinking',
      options: {
        modelKwargs: { thinking: { type: 'enabled', budget_tokens: 2048 } },
      },
    },
    {
      label: 'fallback',
      options: {
        fallbacks: [
          {
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'other', apiKey: 'unused' },
          },
        ],
      },
    },
    {
      label: 'endpoint',
      options: { configuration: { baseURL: 'https://changed.invalid/v1' } },
    },
    {
      label: 'request headers',
      options: { configuration: { defaultHeaders: { 'x-route': 'changed' } } },
    },
  ])(
    'rejects changed model runtime options before resumed effects: $label',
    async ({ options }) => {
      const h = harness();
      const run = await h.build();
      await start(run);
      const definition = {
        ...h.graph,
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      definition.agents[1].clientOptions = {
        ...definition.agents[1].clientOptions,
        ...options,
      };
      const rebuilt = await h.build(definition);
      await rebuilt.resume([{ type: 'approve' }], config);
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
    }
  );

  test('runtime option key order and omitted undefined values remain compatible', async () => {
    const h = harness();
    h.graph.agents[1].clientOptions = {
      modelName: 'left-model',
      apiKey: 'test-key',
      temperature: 0.4,
      configuration: {
        baseURL: 'https://stable.invalid/v1',
        defaultHeaders: { 'x-route': 'stable' },
      },
    };
    const run = await h.build();
    await start(run);
    const definition = {
      ...h.graph,
      agents: h.graph.agents.map((agent) => ({ ...agent })),
    };
    definition.agents[1].clientOptions = {
      configuration: {
        defaultHeaders: { 'x-route': 'stable' },
        baseURL: 'https://stable.invalid/v1',
      },
      temperature: 0.4,
      apiKey: 'test-key',
      modelName: 'left-model',
      maxRetries: undefined,
    };
    const rebuilt = await h.build(definition);
    await rebuilt.resume([{ type: 'approve' }], config);
    await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
  });

  test('executable runtime options require a host revision before any member starts', async () => {
    const h = harness();
    delete h.graph.configId;
    h.graph.agents[1].clientOptions = {
      ...h.graph.agents[1].clientOptions,
      configuration: { fetch: globalThis.fetch },
    };
    const run = await h.build();
    await start(run);
    expect(h.calls).toHaveLength(0);
    expect(
      run
        .getRunMessages()
        ?.some((message) => String(message.content).includes('configId'))
    ).toBe(true);
  });

  test.each([false, true])(
    'versioned executable runtime options resume with explicit host authority, changed=%s',
    async (changed) => {
      const h = harness();
      h.graph.configId = 'runtime-v1';
      h.graph.agents[1].clientOptions = {
        ...h.graph.agents[1].clientOptions,
        configuration: { fetch: globalThis.fetch },
      };
      const run = await h.build();
      await start(run);
      const definition = {
        ...h.graph,
        configId: changed ? 'runtime-v2' : 'runtime-v1',
      };
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

  test.each(
    [false, true].flatMap((eventDriven) =>
      [false, true].flatMap((rebuilt) =>
        ['left', 'right'].map((questionMember) => ({
          eventDriven,
          rebuilt,
          questionMember: questionMember as 'left' | 'right',
        }))
      )
    )
  )(
    'mixed question and approval decisions remain task-local: %j',
    async ({ eventDriven, rebuilt, questionMember }) => {
      const h = harness(
        eventDriven,
        false,
        false,
        false,
        false,
        questionMember
      );
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
      const resumed = rebuilt ? await h.build() : run;
      await resumed.resume(decisions, config);
      expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
      const approved =
        questionMember === 'left' ? h.rightExecute : h.leftExecute;
      const questioned =
        questionMember === 'left' ? h.leftExecute : h.rightExecute;
      expect(approved).toHaveBeenCalledTimes(1);
      expect(await questioned.mock.results.at(-1)?.value).toBe(
        `${questionMember} clarified`
      );
    }
  );

  test.each(['cycle', 'getter', 'array-getter', 'opaque-instance'] as const)(
    'opaque model options fail before execution without inspecting hidden values: %s',
    async (kind) => {
      const h = harness();
      delete h.graph.configId;
      const getter = jest.fn(() => 'hidden');
      const metadata = {};
      if (kind === 'cycle')
        Object.defineProperty(metadata, 'self', {
          value: metadata,
          enumerable: true,
        });
      if (kind === 'getter')
        Object.defineProperty(metadata, 'hidden', {
          enumerable: true,
          get: getter,
        });
      if (kind === 'array-getter') {
        const values: string[] = [];
        Object.defineProperty(values, 0, { enumerable: true, get: getter });
        Object.defineProperty(metadata, 'values', {
          enumerable: true,
          value: values,
        });
      }
      if (kind === 'opaque-instance')
        Object.setPrototypeOf(metadata, Date.prototype);
      h.graph.agents[1].clientOptions = {
        ...h.graph.agents[1].clientOptions,
        metadata,
      };
      const run = await h.build();
      await start(run);
      expect(h.calls).toHaveLength(0);
      expect(getter).not.toHaveBeenCalled();
      expect(
        run
          .getRunMessages()
          ?.some((message) => String(message.content).includes('configId'))
      ).toBe(true);
    }
  );

  test.each([
    { setting: 'result truncation', change: { maxToolResultChars: 400 } },
    { setting: 'cached tool schema budget', change: { toolSchemaTokens: 120 } },
    {
      setting: 'summarization enablement',
      change: { summarizationEnabled: true },
    },
    {
      setting: 'summarization configuration',
      change: { summarizationConfig: { reserveRatio: 0.4 } },
    },
    {
      setting: 'context pruning',
      change: {
        contextPruningConfig: { enabled: true, keepLastAssistants: 1 },
      },
    },
    { setting: 'legacy content format', change: { useLegacyContent: true } },
    {
      setting: 'reasoning format',
      change: { reasoningKey: 'reasoning_content' as const },
    },
    { setting: 'stream buffer', change: { streamBuffer: 16 } },
    { setting: 'member identity', change: { name: 'Different identity' } },
  ])(
    'rejects execution-affecting member setting changes: $setting',
    async ({ change }) => {
      const h = harness();
      const run = await h.build();
      await start(run);
      const definition = {
        ...h.graph,
        agents: h.graph.agents.map((agent, index) =>
          index === 1 ? { ...agent, ...change } : agent
        ),
      };
      const rebuilt = await h.build(definition);
      await rebuilt.resume([{ type: 'approve' }], config);
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(0);
    }
  );

  test.each(['tools', 'toolMap', 'graphTools'] as const)(
    'executable %s requires a host revision before effects',
    async (source) => {
      const h = harness();
      delete h.graph.configId;
      const executable = tool(h.leftExecute, {
        name: 'left_tool',
        description: 'Requires review',
        schema: z.object({ value: z.string() }),
      });
      h.graph.agents[1].tools = [];
      if (source === 'toolMap')
        h.graph.agents[1].toolMap = new Map([['left_tool', executable]]);
      else h.graph.agents[1][source] = [executable];
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
    }
  );

  test.each(['toolMap', 'graphTools'] as const)(
    'changed %s implementation revision rejects old approval',
    async (source) => {
      const h = harness();
      const original = tool(h.leftExecute, {
        name: 'left_tool',
        description: 'Requires review',
        schema: z.object({ value: z.string() }),
      });
      h.graph.agents[1].tools = [];
      if (source === 'toolMap')
        h.graph.agents[1].toolMap = new Map([['left_tool', original]]);
      else h.graph.agents[1].graphTools = [original];
      const run = await h.build();
      await start(run);
      const replacement = jest.fn(async () => 'replacement effect');
      const executable = tool(replacement, {
        name: 'left_tool',
        description: 'Requires review',
        schema: z.object({ value: z.string() }),
      });
      const definition = {
        ...h.graph,
        configId: 'implementation-v2',
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      if (source === 'toolMap')
        definition.agents[1].toolMap = new Map([['left_tool', executable]]);
      else definition.agents[1].graphTools = [executable];
      const rebuilt = await h.build(definition);
      await rebuilt.resume([{ type: 'approve' }], config);
      expect(replacement).not.toHaveBeenCalled();
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
    }
  );

  test('schema-only graph members do not require an implementation revision', async () => {
    const h = harness(true);
    delete h.graph.configId;
    h.graph.agents = h.graph.agents.map((agent) => ({ ...agent, tools: [] }));
    const run = await h.build();
    await start(run);
    expect(run.getInterrupt()?.pendingInterrupts).toHaveLength(2);
    await run.resume([{ type: 'approve' }], config);
    const rebuilt = await h.build();
    await rebuilt.resume([{ type: 'approve' }], config);
    expect(rebuilt.getInterrupt()).toBeUndefined();
    expect(h.leftExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
  });
});

class OpaqueReviewQuestion {
  readonly question = 'Continue custom work?';
  describe(): string {
    return this.question;
  }
}

function installStandaloneTeamWorkflow(run: Run<t.IState>, checkpointer: MemorySaver): void {
  const graphTools = run.Graph?.agentContexts.get('parent')?.graphTools as t.GenericTool[] | undefined;
  const spawn = graphTools?.find((candidate) => candidate.name === Constants.SUBAGENT);
  if (spawn == null) throw new Error('Missing real graph subagent tool');
  const State = Annotation.Root({
    ...MessagesAnnotation.spec,
    runStepState: Annotation<t.RunStepResumeState | undefined>({ reducer: (current, update) => update ?? current, default: () => undefined }),
  });
  run.graphRunnable = new StateGraph(State)
    .addNode('team', async (_state, runnable) => {
      const content = await spawn.invoke({
        type: 'tool_call', name: Constants.SUBAGENT, id: 'parent-spawn',
        args: { description: 'Review the task', subagent_type: 'team' },
      }, { ...runnable, callbacks: [] });
      return { messages: [new AIMessage(String(content))] };
    })
    .addEdge(START, 'team').addEdge('team', END)
    .compile({ checkpointer: checkpointer });
  const v2StreamEvents: (
      input: t.BaseGraphState,
      runnable: t.RunStreamConfig & { version: 'v2' },
      options?: Parameters<t.CompiledStateWorkflow['streamEvents']>[2]
    ) => IterableReadableStream<StreamEvent> = run.graphRunnable.streamEvents;
  const streamEvents = v2StreamEvents.bind(run.graphRunnable);
  Object.defineProperty(run.graphRunnable, 'streamEvents', {
    value: (input: t.BaseGraphState, runnable: t.RunStreamConfig, options?: Parameters<t.CompiledStateWorkflow['streamEvents']>[2]) =>
      IterableReadableStream.fromAsyncGenerator((async function* () {
        for await (const event of streamEvents(input, { ...runnable, version: 'v2' }, options)) {
          if (event.run_id === run.id) yield event;
        }
      })()),
  });
}

describe('graph subagent resume contracts', () => {
  afterEach(() => jest.restoreAllMocks());
  test.each(
    (['resume', 'command'] as const).flatMap((entrypoint) =>
      (['array', 'map'] as const).flatMap((shape) =>
        (['unchanged', 'implementation', 'partition'] as const).flatMap((change) => [
          { entrypoint, shape, change, rebuilt: true },
          ...(change === 'unchanged' ? [{ entrypoint, shape, change, rebuilt: false }] : []),
        ])
      )
    )
  )('singleton checkpoint identity and sidecars: %j', async ({ entrypoint, shape, change, rebuilt }) => {
    const h = harness(false, false, false, false, true);
    h.graph.agents = [{ ...h.graph.agents[1], codeSessionKey: 'original-partition' }];
    h.graph.edges = [];
    h.graph.entryAgentId = 'left';
    h.graph.resultAgentId = 'left';
    const runId = 'singleton-command-resume';
    const run = await h.build(h.graph, undefined, undefined, runId);
    await start(run);
    const pending = run.getInterrupt()?.pendingInterrupts;
    expect(pending).toHaveLength(1);
    const first = pending?.[0];
    if (first == null) throw new Error('Missing singleton approval');
    expect(h.leftEffect).toHaveBeenCalledTimes(1);
    const checkpoint = await h.checkpointer.getTuple(config);
    const rootInterrupt = checkpoint?.pendingWrites?.find(([, channel]) => channel === '__interrupt__')?.[2] as Interrupt | undefined;
    const saved = getSubagentResumeManifest(rootInterrupt?.value)?.executions[0];
    if (saved == null) throw new Error('Missing saved child manifest');
    expect(saved.graphState.toolCallSteps.length).toBeGreaterThan(0);
    const replacement = jest.fn(async () => 'replacement effect');
    const definition = {
      ...h.graph,
      configId: change === 'implementation' ? 'implementation-v2' : h.graph.configId,
      agents: h.graph.agents.map((agent) => ({ ...agent })),
    };
    if (change === 'implementation') definition.agents[0].tools = [tool(replacement, {
      name: 'left_tool', description: 'Requires review', schema: z.object({ value: z.string() }),
    })];
    if (change === 'partition') definition.agents[0].codeSessionKey = 'changed-partition';
    const resumed = rebuilt ? await h.build(definition, undefined, undefined, runId) : run;
    const restoreState = jest.spyOn(Graph.prototype, 'restoreSubagentResumeState');
    try {
      const decisions: t.ToolApprovalDecision[] = [{ type: 'approve' }];
      const resume = shape === 'map' ? { [first.interruptId]: decisions } : decisions;
      if (entrypoint === 'command') await resumed.processStream(new Command({ resume }), config);
      else await resumed.resume(resume, config);
      expect(replacement).not.toHaveBeenCalled();
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      if (change !== 'unchanged') {
        expect(h.leftExecute).not.toHaveBeenCalled();
        expect(restoreState).not.toHaveBeenCalled();
        expect(resumed.getRunMessages()?.some((message) => String(message.content).includes('changed'))).toBe(true);
        return;
      }
      expect(restoreState).toHaveBeenCalledWith(saved.graphState, runId);
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.leftExecute.mock.calls[0][0]).toEqual({ value: 'original' });
      expect(h.calls.filter((id) => id === 'left')).toHaveLength(2);
      expect(resumed.getInterrupt()).toBeUndefined();
    } finally {
      restoreState.mockRestore();
    }
  });

  test.each([
    { label: 'function', payload: { question: 'Continue?', resolve: () => 'resolved' } },
    { label: 'symbol', payload: { question: 'Continue?', resolution: Symbol('resolved') } },
    { label: 'class', payload: new OpaqueReviewQuestion() },
    { label: 'serializable', payload: { question: 'Continue?', resolution: 'resolved' } },
  ].flatMap((sample) => (['resume', 'command'] as const).flatMap((entrypoint) => [
    { ...sample, entrypoint, rebuilt: false },
    ...(sample.label === 'serializable' ? [{ ...sample, entrypoint, rebuilt: true }] : []),
  ])))('standalone parallel payload contract: $label $entrypoint rebuilt=$rebuilt', async ({ payload, entrypoint, rebuilt }) => {
    let releaseCustom: () => void = () => { throw new Error('Custom barrier not initialized'); };
    const customReady = new Promise<void>((resolve) => { releaseCustom = resolve; });
    const h = harness(false, false, false, false, false, 'left', customReady);
    const putWrites = h.checkpointer.putWrites.bind(h.checkpointer);
    jest.spyOn(h.checkpointer, 'putWrites').mockImplementation(async (...args) => {
      await putWrites(...args);
      if (args[1].some(([channel, value]) => channel === '__interrupt__' &&
        !isToolApprovalInterrupt(getPublicToolInterruptPayload((value as Interrupt).value)))) releaseCustom();
    });
    const finished = jest.fn((answer: string) => `custom ${answer}`);
    h.graph.agents[1].graphTools = [tool(async () => {
      return finished(interrupt<typeof payload, string>(payload));
    }, { name: 'left_tool', description: 'Ask a question', schema: z.object({ value: z.string() }) })];
    const run = await h.build();
    installStandaloneTeamWorkflow(run, h.checkpointer);
    await start(run);
    const result = run.getInterrupt<typeof payload | t.ToolApprovalInterruptPayload>();
    expect(isToolApprovalInterrupt(result?.payload)).toBe(true);
    expect(result?.pendingInterrupts).toHaveLength(2);
    const approval = result?.pendingInterrupts?.find((entry) => isToolApprovalInterrupt(entry.payload));
    const custom = result?.pendingInterrupts?.find((entry) => !isToolApprovalInterrupt(entry.payload));
    if (approval == null || custom == null || !isToolApprovalInterrupt(approval.payload)) throw new Error('Missing mixed proposals');
    expect(custom.payload).toStrictEqual(payload);
    expect(JSON.stringify(result)).not.toMatch(/__librechat|approvalReplays|checkpoints/);
    approval.payload.action_requests[0].arguments.value = 'host mutation';
    approval.payload.review_configs[0].allowed_decisions.length = 0;
    if (approval.payload.subagent != null) approval.payload.subagent.agent_id = 'foreign-member';
    const reread = run.getInterrupt();
    const isolated = reread?.pendingInterrupts?.find((entry) => entry.interruptId === approval.interruptId)?.payload;
    if (!isToolApprovalInterrupt(isolated)) throw new Error('Missing isolated approval');
    expect(isolated.action_requests[0].arguments).toEqual({ value: 'original' });
    expect(isolated.review_configs[0].allowed_decisions).toContain('approve');
    expect(isolated.subagent?.agent_id).toBe('right');
    const resumed = rebuilt ? await h.build() : run;
    if (rebuilt) installStandaloneTeamWorkflow(resumed, h.checkpointer);
    const decisions = { [approval.interruptId]: [{ type: 'approve' }], [custom.interruptId]: 'resolved' };
    if (entrypoint === 'command') await resumed.processStream(new Command({ resume: decisions }), config);
    else await resumed.resume(decisions, config);
    expect(resumed.getInterrupt()).toBeUndefined();
    expect(h.rightExecute).toHaveBeenCalledTimes(1);
    expect(h.rightExecute.mock.calls[0][0]).toEqual({ value: 'original' });
    expect(finished).toHaveBeenCalledWith('resolved');
    expect(finished).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
  });

  test.each([false, true].flatMap((rebuilt) => ['left', 'right', 'both'].map((addressed) => ({ rebuilt, addressed }))))(
    'direct parallel Commands preserve pending authority and effects: %j', async ({ rebuilt, addressed }) => {
      const h = harness(false, false, false, false, true);
      const run = await h.build();
      await start(run);
      const pending = run.getInterrupt()?.pendingInterrupts;
      expect(pending).toHaveLength(2);
      const decisions = Object.fromEntries((pending ?? []).filter((entry) => {
        if (!isToolApprovalInterrupt(entry.payload)) throw new Error('Missing approval');
        return addressed === 'both' || entry.payload.subagent?.agent_id === addressed;
      }).map((entry) => [entry.interruptId, [{ type: 'approve' }]]));
      const resumed = rebuilt ? await h.build() : run;
      if (resumed.graphRunnable == null) throw new Error('Missing graph');
      const stream = jest.spyOn(resumed.graphRunnable, 'streamEvents');
      const copySession = jest.spyOn(HookRegistry.prototype, 'copySession');
      const callerConfig = { ...config, configurable: {
        ...config.configurable,
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: { version: 999 },
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'untrusted-attempt',
      } };
      await resumed.processStream(new Command({ resume: decisions }), callerConfig);
      for (const entry of pending ?? []) {
        if (isToolApprovalInterrupt(entry.payload) && entry.payload.hook_session_id != null)
          expect(copySession).toHaveBeenCalledWith(entry.payload.hook_session_id, resumed.id);
      }
      const restored = stream.mock.calls[0][1].configurable;
      expect(restored?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]).toMatchObject({ version: 1 });
      const firstAttempt = restored?.[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY];
      expect(typeof firstAttempt).toBe('string');
      expect(firstAttempt).not.toBe('untrusted-attempt');
      expect(callerConfig.configurable[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]).toBe('untrusted-attempt');
      expect(h.leftExecute).toHaveBeenCalledTimes(addressed === 'right' ? 0 : 1);
      expect(h.rightExecute).toHaveBeenCalledTimes(addressed === 'left' ? 0 : 1);
      if (addressed !== 'both') {
        const remaining = resumed.getInterrupt()?.pendingInterrupts;
        expect(remaining).toHaveLength(1);
        const last = remaining?.[0];
        if (last == null) throw new Error('Missing pending sibling');
        const again = await h.build();
        if (again.graphRunnable == null) throw new Error('Missing graph');
        const nextStream = jest.spyOn(again.graphRunnable, 'streamEvents');
        await again.processStream(new Command({ resume: { [last.interruptId]: [{ type: 'approve' }] } }), config);
        expect(again.getInterrupt()).toBeUndefined();
        expect(nextStream.mock.calls[0][1].configurable?.[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]).not.toBe(firstAttempt);
      } else expect(resumed.getInterrupt()).toBeUndefined();
      expect(h.leftExecute).toHaveBeenCalledTimes(1);
      expect(h.rightExecute).toHaveBeenCalledTimes(1);
      expect(h.leftEffect).toHaveBeenCalledTimes(1);
      expect(h.rightEffect).toHaveBeenCalledTimes(1);
      expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
      expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
    }
  );

  test.each(['manifest-version', 'user'] as const)(
    'direct Commands reject foreign or malformed checkpoint authority: %s', async (change) => {
      const h = harness();
      h.graph.agents = [h.graph.agents[1]];
      h.graph.edges = [];
      h.graph.entryAgentId = 'left';
      h.graph.resultAgentId = 'left';
      const runId = 'direct-command-authority';
      const run = await h.build(h.graph, undefined, undefined, runId);
      await start(run);
      const first = run.getInterrupt();
      if (first == null) throw new Error('Missing approval');
      const rebuilt = await h.build(h.graph, undefined, undefined, runId);
      let injected = false;
      if (change === 'manifest-version') {
        const workflow = rebuilt.graphRunnable as t.CompiledStateWorkflow &
          Pick<ReturnType<StateGraph<t.IState>['compile']>, 'getState'>;
        const getState = workflow.getState.bind(workflow);
        const corruptSnapshot = (snapshot: StateSnapshot): StateSnapshot => ({
          ...snapshot,
          tasks: snapshot.tasks.map((task) => ({
            ...task,
            ...(task.state != null && 'tasks' in task.state ? { state: corruptSnapshot(task.state) } : {}),
            interrupts: task.interrupts.map((stored) => {
              let body = stored.value;
              while (getRunStepResumeState(body) != null) body = stripRunStepResumeState(body);
              if (body == null || typeof body !== 'object') throw new Error('Missing checkpoint envelope');
              const corrupted = { ...body, [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: { version: 999, executions: [] } };
              const state = getRunStepResumeState(stored.value);
              const value = state == null ? corrupted : attachRunStepResumeState(corrupted, state);
              expect(stripRunStepResumeState(value)).toHaveProperty([SUBAGENT_RESUME_MANIFEST_CONFIG_KEY, 'version'], 999);
              injected = true;
              return { ...stored, value };
            }),
          })),
        });
        jest.spyOn(workflow, 'getState').mockImplementation(async (...args) =>
          corruptSnapshot(await getState(...args))
        );
      }
      const callerConfig = change === 'user'
        ? { ...config, configurable: { ...config.configurable, user_id: 'foreign-user' } } : config;
      const resume = rebuilt.processStream(new Command({ resume: { [first.interruptId]: [{ type: 'approve' }] } }), callerConfig);
      if (change === 'manifest-version') {
        await expect(resume).rejects.toThrow('Invalid subagent resume manifest');
        expect(injected).toBe(true);
        expect(() => rebuilt.getInterrupt()).toThrow('Invalid subagent resume manifest');
      } else await expect(resume).rejects.toThrow('execution owner changed');
      expect(h.leftExecute).not.toHaveBeenCalled();
      expect(h.rightExecute).not.toHaveBeenCalled();
    }
  );

});

describe('graph member checkpoint approval barriers', () => {
  test.each([
    ...[false, true].flatMap((eventDriven) => [false, true].flatMap((rebuilt) =>
      ['resume', 'command'].map((entrypoint) => ({ eventDriven, rebuilt, entrypoint, policy: 'allow', decision: 'approve' }))
    )),
    ...[false, true].flatMap((eventDriven) => ['reject', 'edit', 'respond'].map((decision) => ({ eventDriven, rebuilt: true, entrypoint: 'resume', policy: 'none', decision }))),
  ])('keeps the unaddressed member paused and resumes it explicitly: %j', async ({ eventDriven, rebuilt, entrypoint, policy, decision }) => {
    const h = harness(eventDriven, false, false, false, true);
    const run = await h.build();
    await start(run);
    const pending = run.getInterrupt()?.pendingInterrupts;
    expect(pending).toHaveLength(2);
    const selectedName = entrypoint === 'command' ? 'right' : 'left';
    const selected = pending?.find((entry) => isToolApprovalInterrupt(entry.payload) && entry.payload.subagent?.agent_id === selectedName);
    const sibling = pending?.find((entry) => entry !== selected);
    if (selected == null || sibling == null) throw new Error('Missing member approvals');
    h.setPolicy(policy === 'none' ? 'none' : 'allow');
    const resumed = rebuilt ? await h.build() : run;
    let value: t.ToolApprovalDecision = { type: 'approve' };
    if (decision === 'reject') value = { type: 'reject', reason: 'Not this task' };
    else if (decision === 'edit') value = { type: 'edit', updatedInput: { value: 'reviewed' } };
    else if (decision === 'respond') value = { type: 'respond', responseText: 'Human result' };
    const decisions = { [selected.interruptId]: [value] };
    if (entrypoint === 'command') await resumed.processStream(new Command({ resume: decisions }), config);
    else await resumed.resume(decisions, config);
    const selectedEffect = selectedName === 'left' ? h.leftExecute : h.rightExecute;
    const siblingEffect = selectedName === 'left' ? h.rightExecute : h.leftExecute;
    expect(selectedEffect.mock.calls.length).toBe(decision === 'reject' || decision === 'respond' ? 0 : 1);
    expect(siblingEffect.mock.calls.length).toBe(0);
    if (decision === 'edit') expect(selectedEffect.mock.calls[0][0]).toEqual({ value: 'reviewed' });
    expect(resumed.getInterrupt()?.pendingInterrupts).toEqual([sibling]);
    const finished = rebuilt ? await h.build() : resumed;
    await finished.resume({ [sibling.interruptId]: [{ type: 'approve' }] }, config);
    expect(finished.getInterrupt()).toBeUndefined();
    expect(siblingEffect).toHaveBeenCalledTimes(1);
    expect(h.leftEffect).toHaveBeenCalledTimes(1);
    expect(h.rightEffect).toHaveBeenCalledTimes(1);
    expect(h.calls.filter((id) => id === 'entry')).toHaveLength(1);
    expect(h.calls.filter((id) => id === 'result')).toHaveLength(1);
  });
});

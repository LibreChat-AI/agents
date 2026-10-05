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
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type * as t from '@/types';
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
    private readonly programmatic = false
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
    const toolName =
      this.programmatic && owner === 'left'
        ? Constants.PROGRAMMATIC_TOOL_CALLING
        : `${owner}_tool`;
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
            {
              name: toolName,
              args: JSON.stringify(
                this.programmatic && owner === 'left'
                  ? { code: 'left_tool({value: "programmatic"})' }
                  : { value: 'original' }
              ),
              id: 'shared-member-call',
              index: 0,
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

function harness(eventDriven = false, question = false, programmatic = false) {
  const checkpointer = new MemorySaver();
  const calls: string[] = [];
  const leftExecute = jest.fn(async ({ value }: { value: string }) =>
    question
      ? `left ${askUserQuestion({ question: 'Clarify the review' }, { toolCallId: 'shared-member-call' }).answer}`
      : `left ${value}`
  );
  const rightExecute = jest.fn(
    async ({ value }: { value: string }) => `right ${value}`
  );
  const tools = { left: leftExecute, right: rightExecute };
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
        ...(eventDriven
          ? {
            toolDefinitions: [
              {
                name: `${id}_tool`,
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
            name: `${id}_tool`,
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
  const build = async (definition = graph, signal?: AbortSignal) => {
    const hookRegistry = new HookRegistry();
    for (const id of ['left', 'right'])
      hookRegistry.register('PreToolUse', {
        pattern:
          programmatic && id === 'left'
            ? Constants.PROGRAMMATIC_TOOL_CALLING
            : `${id}_tool`,
        hooks: [async () => ({ decision: 'ask' as const })],
      });
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
      ...(eventDriven
        ? {
          customHandlers: {
            [GraphEvents.ON_TOOL_EXECUTE]: {
              async handle(_event, rawData) {
                const batch = rawData as t.ToolExecuteBatchRequest;
                batch.resolve(
                  await Promise.all(
                    batch.toolCalls.map(async (call) => {
                      const executor =
                          call.name === 'left_tool'
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
    run.Graph?.setSubagentModelOverride(new ApprovalModel(calls, programmatic));
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
      const run = await h.build();
      await start(run);
      expect(run.getInterrupt()?.payload.type).toBe('tool_approval');
      const graph: t.GraphSubagentConfig = {
        ...h.graph,
        agents: h.graph.agents.map((agent) => ({ ...agent })),
      };
      if (change === 'topology')
        graph.edges = [
          { from: 'entry', to: 'right', edgeType: 'direct' },
          { from: 'right', to: 'left', edgeType: 'direct' },
          { from: 'left', to: 'result', edgeType: 'direct' },
        ];
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
});

import { MongoClient } from 'mongodb';
import { tool } from '@langchain/core/tools';
import { MongoMemoryServer } from 'mongodb-memory-server-core';
import { MongoDBSaver } from '@langchain/langgraph-checkpoint-mongodb';
import { Command, MemorySaver, interrupt } from '@langchain/langgraph';
import { HumanMessage, getBufferString } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { HandoffLimitError } from '@/graphs/handoff';
import { Constants, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

class RoutingModel extends FakeChatModel {
  readonly visits: string[] = [];
  private readonly agentsByRun = new Map<string, string>();
  constructor(
    private readonly script: (
      agent: string,
      visit: number
    ) => ToolCall[] | Error
  ) {
    super({ responses: ['done'] });
    this.callbacks = [
      {
        handleChatModelStart: (
          _model,
          _messages,
          runId,
          _parentId,
          _extra,
          _tags,
          metadata
        ) => {
          if (typeof metadata?.activeAgentId === 'string')
            this.agentsByRun.set(runId, metadata.activeAgentId);
        },
      },
    ];
  }
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    manager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const prompt = getBufferString(messages);
    const activeAgentId =
      manager == null ? undefined : this.agentsByRun.get(manager.runId);
    const agent =
      typeof activeAgentId === 'string'
        ? activeAgentId
        : (/IDENTITY:(\w+)/.exec(prompt)?.[1] ?? 'missing');
    this.visits.push(agent);
    const result = this.script(
      agent,
      this.visits.filter((id) => id === agent).length
    );
    if (result instanceof Error) throw result;
    yield* new FakeChatModel({
      responses: [`${agent} reply`],
      toolCalls: result,
    })._streamResponseChunks(messages, options, manager);
  }
}

const transfer = (target: string, id = `to-${target}`): ToolCall => ({
  id,
  name: `${Constants.LC_TRANSFER_TO_}${target}`,
  args: {},
});
const agent = (agentId: string): t.AgentInputs => ({
  agentId,
  instructions: `IDENTITY:${agentId}`,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  maxContextTokens: 8000,
});
const edge = (
  from: string,
  to: string,
  scope?: 'turn' | 'conversation'
): t.GraphEdge => ({
  from,
  to,
  edgeType: 'handoff',
  handoffScope: scope,
});
const streamConfig = (thread = 'routing'): t.RunStreamConfig => ({
  version: 'v2',
  configurable: { thread_id: thread },
  recursionLimit: 40,
});
async function setup(
  edges: t.GraphEdge[],
  script: ConstructorParameters<typeof RoutingModel>[0],
  options: Partial<Omit<t.MultiAgentGraphConfig, 'type' | 'edges'>> = {}
): Promise<{ run: Run<t.IState>; model: RoutingModel }> {
  const run = await Run.create({
    runId: 'persistent-handoff-test',
    graphConfig: {
      type: 'multi-agent',
      agents: ['A', 'B', 'C'].map(agent),
      edges,
      ...options,
    },
    returnContent: true,
  });
  const model = new RoutingModel(script);
  run.Graph!.overrideModel = model;
  return { run, model };
}
const execute = (run: Run<t.IState>, thread?: string) =>
  run.processStream(
    { messages: [new HumanMessage('help')] },
    streamConfig(thread)
  );

describe('persistent handoff execution contract', () => {
  it.each([undefined, 'turn', 'conversation'] as const)(
    'reports scope %s without changing the run identity',
    async (scope) => {
      const { run, model } = await setup(
        [edge('A', 'B', scope)],
        (id) => (id === 'A' ? [transfer('B')] : []),
        { entryAgentId: 'A' }
      );
      await execute(run);
      expect(model.visits).toEqual(['A', 'B']);
      expect(run.Graph!.defaultAgentId).toBe('A');
      expect(run.getHandoffOutcome()).toMatchObject({
        status: scope === 'conversation' ? 'candidate' : 'unchanged',
        entryAgentId: 'A',
      });
      expect(run.getHandoffOutcome()?.transitions).toHaveLength(1);
      if (scope === 'conversation')
        expect(run.getHandoffOutcome()).toMatchObject({ agentId: 'B' });
    }
  );

  it.each(['turn', 'conversation'] as const)(
    'keeps the last explicit persistent destination through a %s continuation',
    async (scope) => {
      const { run } = await setup(
        [edge('A', 'B', 'conversation'), edge('B', 'C', scope)],
        (id) => {
          if (id === 'A') return [transfer('B')];
          return id === 'B' ? [transfer('C')] : [];
        }
      );
      await execute(run);
      expect(run.getHandoffOutcome()).toMatchObject({
        status: 'candidate',
        agentId: scope === 'turn' ? 'B' : 'C',
      });
      expect(run.getHandoffOutcome()?.transitions).toHaveLength(2);
    }
  );

  it('starts at an explicitly selected node with incoming edges, not the inferred router', async () => {
    const { run, model } = await setup(
      [edge('A', 'B'), edge('B', 'C')],
      () => [],
      { entryAgentId: 'B' }
    );
    await execute(run);
    expect(model.visits).toEqual(['B']);
    expect(run.getHandoffOutcome()).toMatchObject({
      entryAgentId: 'B',
      status: 'unchanged',
    });
  });

  it('does not select a winner from parallel handoffs', async () => {
    const { run } = await setup(
      [edge('A', 'B', 'conversation'), edge('A', 'C', 'conversation')],
      (id) => (id === 'A' ? [transfer('B'), transfer('C')] : [])
    );
    await execute(run);
    expect(run.getHandoffOutcome()).toMatchObject({ status: 'ambiguous' });
    expect(run.getHandoffOutcome()?.transitions).toHaveLength(2);
  });

  it('rejects an oversized parallel batch before either recipient runs', async () => {
    const { run, model } = await setup(
      [edge('A', 'B', 'conversation'), edge('A', 'C')],
      (id) => (id === 'A' ? [transfer('B'), transfer('C')] : []),
      { maxHandoffs: 1 }
    );
    await expect(execute(run)).rejects.toThrow(HandoffLimitError);
    expect(model.visits).toEqual(['A']);
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'incomplete',
      reason: 'handoff_limit',
    });
  });

  it('bounds cycles and retains only admitted transitions', async () => {
    const { run, model } = await setup(
      [edge('A', 'B', 'conversation'), edge('B', 'A', 'conversation')],
      (id) => [transfer(id === 'A' ? 'B' : 'A')],
      { entryAgentId: 'A', maxHandoffs: 3 }
    );
    await expect(execute(run)).rejects.toThrow(HandoffLimitError);
    expect(model.visits).toEqual(['A', 'B', 'A', 'B']);
    expect(run.getHandoffOutcome()?.transitions).toHaveLength(3);
    expect(run.getHandoffOutcome()?.status).toBe('incomplete');
  });

  it('does not promote a candidate after a provider failure', async () => {
    const { run } = await setup(
      [edge('A', 'B', 'conversation')],
      (id) => (id === 'A' ? [transfer('B')] : new Error('provider down')),
      { entryAgentId: 'A' }
    );
    await expect(execute(run)).rejects.toThrow('provider down');
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'incomplete',
      reason: 'error',
    });
  });

  it('resets the ledger on a fresh turn of the same checkpointed thread', async () => {
    const checkpointer = new MemorySaver();
    const { run } = await setup(
      [edge('A', 'B', 'conversation')],
      (id) => (id === 'A' ? [transfer('B')] : []),
      { entryAgentId: 'A', maxHandoffs: 1, compileOptions: { checkpointer } }
    );
    await execute(run);
    const first = run.getHandoffOutcome();
    run.Graph!.overrideModel = new RoutingModel(() => []);
    await execute(run);
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'unchanged',
      transitions: [],
    });
    expect(run.getHandoffOutcome()?.executionId).not.toBe(first?.executionId);
  });

  it.each(['memory', 'mongo'])(
    'preserves candidates and budgets through a rebuilt pause (%s)',
    async (adapter) => {
      const mongod =
        adapter === 'mongo'
          ? await MongoMemoryServer.create({
            instance: { args: ['--nounixsocket'] },
          })
          : undefined;
      const mongoClient =
        mongod == null ? undefined : new MongoClient(mongod.getUri());
      try {
        await mongoClient?.connect();
        const checkpointer =
          mongoClient == null
            ? new MemorySaver()
            : new MongoDBSaver({ client: mongoClient, dbName: 'handoff' });
        const pause = tool(
          () => {
            interrupt({ question: 'continue?' });
            return 'approved';
          },
          {
            name: 'pause',
            description: 'Pause',
            schema: { type: 'object', properties: {} },
          }
        );
        const agents = ['A', 'B', 'C'].map((id) => ({
          ...agent(id),
          ...(id === 'B' ? { tools: [pause] } : {}),
        }));
        const edges = [
          edge('A', 'B', 'conversation'),
          edge('B', 'C', 'conversation'),
        ];
        const options = {
          agents,
          entryAgentId: 'A',
          maxHandoffs: 2,
          compileOptions: { checkpointer },
        };
        const first = await setup(
          edges,
          (id) =>
            id === 'A'
              ? [transfer('B')]
              : [{ id: 'pause-call', name: 'pause', args: {} }],
          options
        );
        await execute(first.run, 'resume');
        expect(first.run.getInterrupt()).toBeDefined();
        expect(first.run.getHandoffOutcome()).toMatchObject({
          status: 'incomplete',
          reason: 'interrupted',
        });
        const before = first.run.getHandoffOutcome();
        const rebuilt = await setup(
          edges,
          (id) => (id === 'B' ? [transfer('C')] : []),
          options
        );
        await rebuilt.run.processStream(
          new Command({ resume: true }),
          streamConfig('resume')
        );
        expect(rebuilt.model.visits).toEqual(['B', 'C']);
        expect(rebuilt.run.getHandoffOutcome()).toMatchObject({
          status: 'candidate',
          agentId: 'C',
          executionId: before?.executionId,
        });
        expect(rebuilt.run.getHandoffOutcome()?.transitions).toHaveLength(2);
        expect(rebuilt.run.getHandoffOutcome()?.transitions[0].id).toBe(
          before?.transitions[0].id
        );
      } finally {
        await mongoClient?.close();
        await mongod?.stop();
      }
    },
    120000
  );

  it('restores completed parallel sibling handoffs before resuming a paused sibling', async () => {
    const checkpointer = new MemorySaver();
    const pause = tool(
      () => {
        interrupt({ question: 'continue?' });
        return 'approved';
      },
      {
        name: 'pause',
        description: 'Pause',
        schema: { type: 'object', properties: {} },
      }
    );
    const agents = ['A', 'B', 'C', 'D', 'E'].map((id) => ({
      ...agent(id),
      ...(id === 'C' ? { tools: [pause] } : {}),
    }));
    const edges = [
      edge('A', 'B', 'conversation'),
      edge('A', 'C'),
      edge('B', 'D', 'conversation'),
      edge('C', 'E', 'conversation'),
    ];
    const options = {
      agents,
      maxHandoffs: 3,
      compileOptions: { checkpointer },
    };
    const first = await setup(
      edges,
      (id) => {
        if (id === 'A') return [transfer('B'), transfer('C')];
        if (id === 'B') return [transfer('D')];
        return id === 'C'
          ? [{ id: 'pause-call', name: 'pause', args: {} }]
          : [];
      },
      options
    );
    await execute(first.run, 'parallel-resume');
    expect(first.run.getInterrupt()).toBeDefined();
    expect(first.run.getHandoffOutcome()?.transitions).toHaveLength(3);
    const rebuilt = await setup(
      edges,
      (id) => (id === 'C' ? [transfer('E')] : []),
      options
    );
    await expect(
      rebuilt.run.processStream(
        new Command({ resume: true }),
        streamConfig('parallel-resume')
      )
    ).rejects.toThrow(HandoffLimitError);
    expect(rebuilt.model.visits).not.toContain('E');
    expect(rebuilt.run.getHandoffOutcome()).toMatchObject({
      status: 'incomplete',
      reason: 'handoff_limit',
    });
  });

  it('reports the actual conditional destination, not a tool-name suffix', async () => {
    const { run } = await setup(
      [
        {
          from: 'A',
          to: ['B', 'C'],
          edgeType: 'handoff',
          handoffScope: 'conversation',
          condition: () => 'C',
        },
      ],
      (id) =>
        id === 'A'
          ? [{ id: 'conditional', name: 'conditional_transfer', args: {} }]
          : []
    );
    await execute(run);
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'candidate',
      agentId: 'C',
    });
  });

  it('does not count a false conditional handoff', async () => {
    const { run } = await setup(
      [
        {
          from: 'A',
          to: 'B',
          condition: () => false,
          handoffScope: 'conversation',
        },
      ],
      (id, visit) =>
        id === 'A' && visit === 1
          ? [{ id: 'conditional', name: 'conditional_transfer', args: {} }]
          : [],
      { entryAgentId: 'A', maxHandoffs: 0 }
    );
    await execute(run);
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'unchanged',
      transitions: [],
    });
  });

  it('does not accept a conditional destination outside its declared edge', async () => {
    const { run, model } = await setup(
      [
        {
          from: 'A',
          to: 'B',
          condition: () => 'C',
          handoffScope: 'conversation',
        },
      ],
      (id, visit) =>
        id === 'A' && visit === 1
          ? [{ id: 'conditional', name: 'conditional_transfer', args: {} }]
          : [],
      { entryAgentId: 'A' }
    );
    await execute(run);
    expect(model.visits).not.toContain('C');
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'unchanged',
      transitions: [],
    });
  });

  it('does not promote a candidate after caller cancellation', async () => {
    const controller = new AbortController();
    const { run } = await setup(
      [edge('A', 'B', 'conversation')],
      (id) => {
        if (id === 'A') return [transfer('B')];
        controller.abort();
        return [];
      },
      { entryAgentId: 'A' }
    );
    await run
      .processStream(
        { messages: [new HumanMessage('help')] },
        { ...streamConfig(), signal: controller.signal }
      )
      .catch(() => undefined);
    expect(run.getHandoffOutcome()?.status).toBe('incomplete');
  });

  it('rejects an unsatisfiable all-of prerequisite at an explicit entry', async () => {
    await expect(
      setup([{ from: ['A', 'B'], to: 'C', edgeType: 'direct' }], () => [], {
        entryAgentId: 'B',
      })
    ).rejects.toThrow('prerequisite');
  });

  it('does not treat cyclic all-of prerequisites as reachable', async () => {
    await expect(
      setup(
        [
          { from: ['A', 'B'], to: 'C', edgeType: 'direct' },
          { from: 'C', to: 'A', edgeType: 'direct' },
        ],
        () => [],
        { entryAgentId: 'B' }
      )
    ).rejects.toThrow('prerequisite');
  });

  it('rejects an unknown entry and scope on direct edges', async () => {
    await expect(
      setup([], () => [], { entryAgentId: 'missing' })
    ).rejects.toThrow('entryAgentId');
    await expect(
      setup(
        [
          {
            from: 'A',
            to: 'B',
            edgeType: 'direct',
            handoffScope: 'conversation',
          },
        ],
        () => []
      )
    ).rejects.toThrow('handoffScope');
  });

  it.each([false, true])(
    'shares the handoff budget with stop-hook continuations (checkpointed: %s)',
    async (checkpointed) => {
      const hooks = new HookRegistry();
      hooks.register('Stop', {
        hooks: [
          async () => ({
            decision: 'block',
            injectedMessages: [
              { role: 'user', content: 'continue', source: 'steer' },
            ],
          }),
        ],
      });
      const run = await Run.create<t.IState>({
        runId: 'continuation-budget',
        graphConfig: {
          type: 'multi-agent',
          agents: ['A', 'B'].map(agent),
          edges: [edge('A', 'B', 'conversation')],
          maxHandoffs: 1,
          compileOptions: checkpointed
            ? { checkpointer: new MemorySaver() }
            : undefined,
        },
        hooks,
        maxStopContinuations: 1,
      });
      const model = new RoutingModel((id, visit) =>
        id === 'A' ? [transfer('B', `transfer-${visit}`)] : []
      );
      run.Graph!.overrideModel = model;
      await expect(execute(run)).rejects.toThrow(HandoffLimitError);
      expect(model.visits).toEqual(['A', 'B', 'A']);
      expect(run.getHandoffOutcome()).toMatchObject({
        status: 'incomplete',
        reason: 'handoff_limit',
      });
      expect(run.getHandoffOutcome()?.transitions).toHaveLength(1);
    }
  );

  it('reports the inferred entry rather than the first configured agent', async () => {
    const { run, model } = await setup(
      [edge('B', 'A', 'conversation')],
      (id) => (id === 'B' ? [transfer('A')] : []),
      { agents: ['A', 'B'].map(agent) }
    );
    await execute(run);
    expect(model.visits).toEqual(['B', 'A']);
    expect(run.getHandoffOutcome()).toMatchObject({
      status: 'candidate',
      entryAgentId: 'B',
      agentId: 'A',
    });
  });

  it('rejects conflicting scopes for the same transfer tool', async () => {
    await expect(
      setup([edge('A', 'B'), edge('A', 'B', 'conversation')], () => [])
    ).rejects.toThrow('Conflicting handoffScope');
  });

  it.each([-1, 1.5, NaN, Infinity])(
    'rejects invalid budget %s',
    async (maxHandoffs) => {
      await expect(setup([], () => [], { maxHandoffs })).rejects.toThrow(
        'maxHandoffs'
      );
    }
  );
});

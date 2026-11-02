import { MemorySaver } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { GraphFactory, GraphFactoryRequest } from '@/graphs/graphFactory';
import type { AgentContext } from '@/agents/AgentContext';
import type * as t from '@/types';
import {
  Constants,
  Providers,
  STANDARD_GRAPH_RUN_NAME,
  MULTI_AGENT_GRAPH_RUN_NAME,
} from '@/common';
import { TOOL_OUTPUT_READER } from '@/tools/ToolOutputReader';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { createFakeStreamingLLM } from '@/llm/fake';
import { createGraph } from '@/graphs/createGraph';
import { StandardGraph } from '@/graphs/Graph';

const invokeConfig: RunnableConfig = {
  configurable: { thread_id: 'graph-factory-test' },
};

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

const getSubagentTool = (
  graph: StandardGraph,
  agentId: string
): t.GenericTool => {
  graph.createWorkflow();
  const tools = graph.agentContexts.get(agentId)?.graphTools as
    | t.GenericTool[]
    | undefined;
  const tool = tools?.find(
    (candidate) => 'name' in candidate && candidate.name === Constants.SUBAGENT
  );
  if (tool == null) {
    throw new Error(`Expected subagent tool for ${agentId}`);
  }
  return tool;
};

describe('graph factory', () => {
  it('constructs standard and multi-agent graph adapters', () => {
    const standard = createGraph({
      kind: 'standard',
      input: { runId: 'standard-factory', agents: [makeAgent('standard')] },
    });
    const multiAgent = createGraph({
      kind: 'multi-agent',
      input: {
        runId: 'multi-agent-factory',
        agents: [makeAgent('multi')],
        edges: [],
      },
    });

    expect(standard).toBeInstanceOf(StandardGraph);
    expect(standard).not.toBeInstanceOf(MultiAgentGraph);
    expect(multiAgent).toBeInstanceOf(MultiAgentGraph);
  });

  it('keeps graph names and checkpoint methods on compiled workflows', () => {
    const standard = createGraph({
      kind: 'standard',
      input: { runId: 'standard-compiled', agents: [makeAgent('standard')] },
    });
    const multiAgent = createGraph({
      kind: 'multi-agent',
      input: {
        runId: 'multi-agent-compiled',
        agents: [makeAgent('multi')],
        edges: [],
      },
    });
    standard.compileOptions = { checkpointer: new MemorySaver() };
    multiAgent.compileOptions = { checkpointer: new MemorySaver() };

    const standardWorkflow = standard.createWorkflow();
    const multiAgentWorkflow = multiAgent.createWorkflow();

    expect(standardWorkflow).toHaveProperty('name', STANDARD_GRAPH_RUN_NAME);
    expect(standardWorkflow).toHaveProperty('getState', expect.any(Function));
    expect(standardWorkflow).toHaveProperty(
      'getStateHistory',
      expect.any(Function)
    );
    expect(multiAgentWorkflow).toHaveProperty(
      'name',
      MULTI_AGENT_GRAPH_RUN_NAME
    );
    expect(multiAgentWorkflow).toHaveProperty('getState', expect.any(Function));
    expect(multiAgentWorkflow).toHaveProperty(
      'getStateHistory',
      expect.any(Function)
    );
  });

  it('accepts a union-typed graph factory request', () => {
    const construct = (request: GraphFactoryRequest): StandardGraph =>
      createGraph(request);
    const graph = construct({
      kind: 'standard',
      input: { runId: 'union-factory', agents: [makeAgent('union')] },
    });

    expect(graph).toBeInstanceOf(StandardGraph);
  });

  it('rejects invalid per-member recursion limits', () => {
    expect(() =>
      createGraph({
        kind: 'multi-agent',
        input: {
          runId: 'invalid-member-limit',
          agents: [makeAgent('member')],
          edges: [],
          memberRecursionLimit: 0,
        },
      })
    ).toThrow(/memberRecursionLimit must be a positive safe integer/);
  });

  it('keeps direct StandardGraph construction compatible with agent children', async () => {
    const parent = {
      ...makeAgent('parent'),
      subagentConfigs: [
        {
          type: 'worker',
          name: 'Worker',
          description: 'Completes one isolated task.',
          agentInputs: makeAgent('worker'),
        },
      ],
    } satisfies t.AgentInputs;
    const graph = new StandardGraph({
      runId: 'direct-standard-factory',
      agents: [parent],
    });
    graph.setSubagentModelOverride(
      createFakeStreamingLLM({ responses: ['worker complete'] })
    );

    const result = await getSubagentTool(graph, 'parent').invoke(
      { description: 'Complete the task.', subagent_type: 'worker' },
      invokeConfig
    );

    expect(result).toBe('worker complete');
  });

  it('fails early when direct construction cannot instantiate a graph child', () => {
    const parent = {
      ...makeAgent('parent'),
      subagentConfigs: [
        {
          kind: 'graph' as const,
          type: 'team',
          name: 'Team',
          description: 'Runs one member.',
          agents: [makeAgent('member')],
          edges: [],
          entryAgentId: 'member',
          resultAgentId: 'member',
        },
      ],
    } satisfies t.AgentInputs;
    const graph = new StandardGraph({
      runId: 'direct-graph-child-factory',
      agents: [parent],
    });

    expect(() => graph.createWorkflow()).toThrow(
      /constructing the parent with createGraph\(\)/
    );
  });

  it.each([undefined, false, true])('propagates an injected factory and readTool=%s through child and grandchild graphs', async (readTool) => {
    const requests: GraphFactoryRequest[] = [];
    const childContexts: AgentContext[] = [];
    const graphFactory: GraphFactory = (request) => {
      requests.push(request);
      const child = request.kind === 'multi-agent'
        ? new MultiAgentGraph(request.input, { graphFactory })
        : new StandardGraph(request.input, { graphFactory });
      childContexts.push(...child.agentContexts.values());
      return child;
    };
    const nestedToolCall: ToolCall = {
      id: 'call_nested_specialist',
      name: Constants.SUBAGENT,
      args: {
        description: 'Complete the specialist step.',
        subagent_type: 'specialist',
      },
      type: 'tool_call',
    };
    const router = {
      ...makeAgent('router'),
      subagentConfigs: [
        {
          type: 'specialist',
          name: 'Specialist',
          description: 'Completes the nested specialist step.',
          agentInputs: makeAgent('specialist'),
        },
      ],
    } satisfies t.AgentInputs;
    const root = {
      ...makeAgent('root'),
      maxSubagentDepth: 2,
      subagentConfigs: [
        {
          type: 'router',
          name: 'Router',
          description: 'Delegates one nested specialist step.',
          allowNested: true,
          agentInputs: router,
        },
      ],
    } satisfies t.AgentInputs;
    const graph = new StandardGraph(
      { runId: 'recursive-graph-factory', agents: [root] },
      { graphFactory }
    );
    graph.toolOutputReferences = { enabled: true, readTool };
    graph.setSubagentModelOverride(
      createFakeStreamingLLM({
        responses: ['delegating', 'specialist complete', 'router complete'],
        toolCalls: [nestedToolCall],
      })
    );

    const result = await getSubagentTool(graph, 'root').invoke(
      { description: 'Route the task.', subagent_type: 'router' },
      invokeConfig
    );

    expect(result).toBe('router complete');
    expect(
      requests.map(({ kind, input }) => ({
        kind,
        agentIds: input.agents.map(({ agentId }) => agentId),
      }))
    ).toEqual([
      { kind: 'standard', agentIds: ['router'] },
      { kind: 'standard', agentIds: ['specialist'] },
    ]);
    expect(childContexts).toHaveLength(2);
    for (const context of childContexts) {
      expect(context.graphTools?.filter((entry) => 'name' in entry && entry.name === TOOL_OUTPUT_READER) ?? [])
        .toHaveLength(readTool === true ? 1 : 0);
      expect(context.toolRegistry?.get(TOOL_OUTPUT_READER)?.allowed_callers)
        .toEqual(readTool === true ? ['direct'] : undefined);
    }
  });
});

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { GraphSubagentConfig } from '@/types';
import type * as t from '@/types';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { createFakeStreamingLLM } from '@/llm/fake';
import { Constants, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { Run } from '@/run';

const invokeConfig: RunnableConfig = {
  configurable: { thread_id: 'graph-subagent-test' },
};

const usage = {
  input_tokens: 5,
  output_tokens: 3,
  total_tokens: 8,
};

class UsageFakeChatModel extends FakeListChatModel {
  constructor() {
    super({ responses: ['member response'] });
  }

  override async *_streamResponseChunks(
    ...args: Parameters<FakeListChatModel['_streamResponseChunks']>
  ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
    yield* super._streamResponseChunks(...args);
    yield new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        usage_metadata: { ...usage },
      }),
    });
  }
}

class LoopingFakeChatModel extends FakeListChatModel {
  private invocationCount = 0;

  constructor() {
    super({ responses: ['unused'] });
  }

  override async *_streamResponseChunks(
    ..._args: Parameters<FakeListChatModel['_streamResponseChunks']>
  ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
    this.invocationCount++;
    if (this.invocationCount <= 2) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          tool_call_chunks: [
            {
              name: 'loop_tool',
              args: '{}',
              id: `loop-${this.invocationCount}`,
              index: 0,
              type: 'tool_call_chunk',
            },
          ],
        }),
      });
      return;
    }
    yield new ChatGenerationChunk({
      text: 'escaped member turn limit',
      message: new AIMessageChunk('escaped member turn limit'),
    });
  }
}

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: `${agentId}-model`, apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
});

const makeGraphConfig = (): GraphSubagentConfig => ({
  kind: 'graph',
  type: 'research-team',
  name: 'Research Team',
  description: 'Coordinates parallel research and synthesis.',
  entryAgentId: 'coordinator',
  resultAgentId: 'synthesizer',
  agents: [
    makeAgent('coordinator'),
    makeAgent('left'),
    makeAgent('right'),
    makeAgent('synthesizer'),
  ],
  edges: [
    {
      from: 'coordinator',
      to: ['left', 'right'],
      edgeType: 'direct',
    },
    {
      from: ['left', 'right'],
      to: 'synthesizer',
      edgeType: 'direct',
    },
  ],
});

const createRun = async (
  graphConfig: GraphSubagentConfig,
  overrides: Partial<t.RunConfig> = {},
  maxSubagentDepth = 2
): Promise<Run<t.IState>> =>
  Run.create<t.IState>({
    runId: `graph-subagent-${Date.now()}-${Math.random()}`,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          ...makeAgent('parent'),
          maxSubagentDepth,
          subagentConfigs: [graphConfig],
        },
      ],
    },
    returnContent: true,
    skipCleanup: true,
    ...overrides,
  });

const getGraphSubagentTool = (run: Run<t.IState>): t.GenericTool => {
  const tools = (run.Graph as StandardGraph).agentContexts.get('parent')
    ?.graphTools as t.GenericTool[] | undefined;
  const tool = tools?.find(
    (candidate) => 'name' in candidate && candidate.name === Constants.SUBAGENT
  );
  if (tool == null) {
    throw new Error('Expected graph subagent tool');
  }
  return tool;
};

describe('Graph subagent integration', () => {
  it('runs a convergent direct DAG and returns only the designated result', async () => {
    const graphConfig = { ...makeGraphConfig(), maxTurns: 1 };
    const run = await createRun(graphConfig);
    (run.Graph as StandardGraph).setSubagentModelOverride(
      createFakeStreamingLLM({
        responses: [
          'coordinator plan',
          'left research',
          'right research',
          'synthesized answer',
        ],
      })
    );

    const result = await getGraphSubagentTool(run).invoke(
      {
        description: 'Research and synthesize the question.',
        subagent_type: 'research-team',
      },
      invokeConfig
    );

    expect(result).toBe('synthesized answer');
  });

  it('runs a depth-one parallel team without giving any member a grandchild tool', async () => {
    const config = makeGraphConfig();
    config.agents = config.agents.map((agent) => ({
      ...agent,
      maxSubagentDepth: 99,
      subagentConfigs: [
        {
          type: 'grandchild',
          name: 'Grandchild',
          description: 'Must not be spawned.',
          agentInputs: makeAgent('grandchild'),
        },
      ],
    }));
    const originalCreateWorkflow = MultiAgentGraph.prototype.createWorkflow;
    const observedMembers: Array<{
      agentId: string;
      maxSubagentDepth?: number;
      nestedConfigs?: t.SubagentConfigEntry[];
      hasSubagentTool: boolean;
    }> = [];
    const createWorkflowSpy = jest
      .spyOn(MultiAgentGraph.prototype, 'createWorkflow')
      .mockImplementation(function (this: MultiAgentGraph) {
        const workflow = originalCreateWorkflow.call(this);
        for (const [agentId, context] of this.agentContexts) {
          observedMembers.push({
            agentId,
            maxSubagentDepth: context.maxSubagentDepth,
            nestedConfigs: context.subagentConfigs,
            hasSubagentTool:
              (context.graphTools as t.GenericTool[] | undefined)?.some(
                (memberTool) =>
                  'name' in memberTool && memberTool.name === Constants.SUBAGENT
              ) ?? false,
          });
        }
        return workflow;
      });

    try {
      const run = await createRun(config, {}, 1);
      (run.Graph as StandardGraph).setSubagentModelOverride(
        createFakeStreamingLLM({
          responses: ['plan', 'left work', 'right work', 'team answer'],
        })
      );

      const result = await getGraphSubagentTool(run).invoke(
        {
          description: 'Complete the parallel team.',
          subagent_type: config.type,
        },
        invokeConfig
      );

      expect(result).toBe('team answer');
      expect(observedMembers).toEqual(
        config.agents.map(({ agentId }) => ({
          agentId,
          maxSubagentDepth: undefined,
          nestedConfigs: undefined,
          hasSubagentTool: false,
        }))
      );
    } finally {
      createWorkflowSpy.mockRestore();
    }
  });

  it('does not fall back to worker text when the result member is textless', async () => {
    const graphConfig: GraphSubagentConfig = {
      ...makeGraphConfig(),
      agents: [makeAgent('worker'), makeAgent('result')],
      edges: [{ from: 'worker', to: 'result', edgeType: 'direct' }],
      entryAgentId: 'worker',
      resultAgentId: 'result',
    };
    const run = await createRun(graphConfig);
    (run.Graph as StandardGraph).setSubagentModelOverride(
      createFakeStreamingLLM({ responses: ['worker text', ''] })
    );

    const result = await getGraphSubagentTool(run).invoke(
      {
        description: 'Complete the two-step task.',
        subagent_type: 'research-team',
      },
      invokeConfig
    );

    expect(result).toBe('Task completed');
  });

  it('keeps each member turn budget independent from the outer graph budget', async () => {
    const loopTool = tool(async (): Promise<string> => 'continue', {
      name: 'loop_tool',
      description: 'Continue the member loop.',
      schema: z.object({}),
    });
    const graphConfig: GraphSubagentConfig = {
      ...makeGraphConfig(),
      maxTurns: 1,
      agents: [
        { ...makeAgent('entry'), graphTools: [loopTool] },
        makeAgent('result'),
      ],
      edges: [{ from: 'entry', to: 'result', edgeType: 'direct' }],
      entryAgentId: 'entry',
      resultAgentId: 'result',
    };
    const run = await createRun(graphConfig);
    (run.Graph as StandardGraph).setSubagentModelOverride(
      new LoopingFakeChatModel()
    );

    const output = await getGraphSubagentTool(run).invoke(
      {
        description: 'Try to exceed the member turn budget.',
        subagent_type: 'research-team',
      },
      invokeConfig
    );

    expect(output).toMatch(/Subagent error: Recursion limit of 3 reached/);
  });

  it('executes a graph with HITL enabled when its members need no approval', async () => {
    const run = await createRun(makeGraphConfig(), {
      humanInTheLoop: { enabled: true },
    });
    (run.Graph as StandardGraph).setSubagentModelOverride(
      createFakeStreamingLLM({ responses: ['plan', 'left', 'right', 'answer'] })
    );
    const result = await getGraphSubagentTool(run).invoke(
      {
        type: 'tool_call',
        name: 'subagent',
        id: 'graph-spawn',
        args: { description: 'Research', subagent_type: 'research-team' },
      },
      invokeConfig
    );
    expect(result).toMatchObject({ content: 'answer' });
  });

  it('attributes usage to each graph member', async () => {
    const graphConfig: GraphSubagentConfig = {
      ...makeGraphConfig(),
      agents: [makeAgent('entry'), makeAgent('worker'), makeAgent('result')],
      edges: [
        { from: 'entry', to: 'worker', edgeType: 'direct' },
        { from: 'worker', to: 'result', edgeType: 'direct' },
      ],
      entryAgentId: 'entry',
      resultAgentId: 'result',
    };
    const usageEvents: t.SubagentUsageEvent[] = [];
    const run = await createRun(graphConfig, {
      subagentUsageSink: (event) => {
        usageEvents.push(event);
      },
    });
    (run.Graph as StandardGraph).setSubagentModelOverride(
      new UsageFakeChatModel()
    );

    await getGraphSubagentTool(run).invoke(
      {
        description: 'Complete the measured graph.',
        subagent_type: 'research-team',
      },
      invokeConfig
    );

    expect(usageEvents).toHaveLength(3);
    expect(
      usageEvents.map((event) => ({
        memberAgentId: event.memberAgentId,
        model: event.model,
        provider: event.provider,
        subagentKind: event.subagentKind,
      }))
    ).toEqual([
      {
        memberAgentId: 'entry',
        model: 'entry-model',
        provider: Providers.OPENAI,
        subagentKind: 'graph',
      },
      {
        memberAgentId: 'worker',
        model: 'worker-model',
        provider: Providers.OPENAI,
        subagentKind: 'graph',
      },
      {
        memberAgentId: 'result',
        model: 'result-model',
        provider: Providers.OPENAI,
        subagentKind: 'graph',
      },
    ]);
    expect(
      new Set(usageEvents.map((event) => event.subagentAgentId)).size
    ).toBe(1);
    const rootRunId = (run.Graph as StandardGraph).runId;
    for (const event of usageEvents) {
      expect(event).toMatchObject({
        runId: rootRunId,
        parentRunId: rootRunId,
        depth: 1,
      });
      expect(event.ancestry).toHaveLength(1);
      expect(event.ancestry?.[0]).toMatchObject({
        subagentType: 'research-team',
        subagentKind: 'graph',
        parentRunId: rootRunId,
      });
    }
  });
});

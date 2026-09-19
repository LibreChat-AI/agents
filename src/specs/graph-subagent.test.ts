import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { GraphSubagentConfig } from '@/types';
import type * as t from '@/types';
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
  overrides: Partial<t.RunConfig> = {}
): Promise<Run<t.IState>> =>
  Run.create<t.IState>({
    runId: `graph-subagent-${Date.now()}-${Math.random()}`,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          ...makeAgent('parent'),
          maxSubagentDepth: 2,
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

  it('fails closed when human-in-the-loop is enabled', async () => {
    const run = await createRun(makeGraphConfig(), {
      humanInTheLoop: { enabled: true },
    });

    const result = await getGraphSubagentTool(run).invoke(
      {
        description: 'Do not start this graph.',
        subagent_type: 'research-team',
      },
      invokeConfig
    );

    expect(result).toBe(
      'Error: Human-in-the-loop execution is not yet supported for graph subagents.'
    );
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
    expect(new Set(usageEvents.map((event) => event.modelRunId)).size).toBe(3);
    expect(
      usageEvents.every((event) => typeof event.modelRunId === 'string')
    ).toBe(true);
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

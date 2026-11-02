import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { Constants, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { FakeChatModel } from '@/llm/fake';

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const calls: ToolCall[] = ['first', 'second'].map((id) => ({
  id,
  name: Constants.SUBAGENT,
  args: { description: `Research ${id}`, subagent_type: 'researcher' },
}));

class ParentModel extends FakeChatModel {
  private invoked = false;
  readonly observed: BaseMessage[][] = [];
  constructor(
    private readonly started: Promise<void>,
    private readonly fail: boolean,
    private readonly sealStyle: 'responses' | 'bedrock' | 'google'
  ) {
    super({ responses: ['unused'] });
  }
  override async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    this.observed.push(messages);
    if (this.invoked) {
      yield this._createResponseChunk('Both results received.');
      return;
    }
    this.invoked = true;
    for (let index = 0; index < calls.length; index++) {
      const call = calls[index];
      const adapter = {
        bedrock: BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
        google: GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
        responses: OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
      }[this.sealStyle];
      const seal =
        this.sealStyle === 'google'
          ? { kind: 'all' }
          : { kind: 'single', index };
      yield this._createResponseChunk(
        '',
        [
          {
            id: call.id,
            name: call.name,
            args: JSON.stringify(call.args),
            index,
          },
        ],
        {
          [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]: adapter,
          ...(this.sealStyle === 'bedrock'
            ? {}
            : { [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: seal }),
        }
      );
      if (this.sealStyle === 'bedrock') {
        yield this._createResponseChunk('', [{ index, args: '' }], {
          [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]: adapter,
          [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: seal,
        });
      }
      if (index === 0) {
        let timeout!: ReturnType<typeof setTimeout>;
        try {
          await Promise.race([
            this.started,
            new Promise<never>((_, reject) => {
              timeout = setTimeout(
                () =>
                  reject(
                    new Error('Child did not start before sibling prompt')
                  ),
                3000
              );
            }),
          ]);
        } finally {
          clearTimeout(timeout);
        }
        if (this.fail) {
          throw new Error('provider disconnected');
        }
      }
    }
  }
}

class ChildModel extends FakeChatModel {
  starts = 0;
  constructor(private readonly started: () => void) {
    super({ responses: ['unused'] });
  }
  override async *_streamResponseChunks(): AsyncGenerator<ChatGenerationChunk> {
    this.starts++;
    this.started();
    yield this._createResponseChunk('Research result.');
  }
}

function fixture(
  fail = false,
  sealStyle: 'responses' | 'bedrock' | 'google' = 'responses'
) {
  const started = deferred();
  const parent = new ParentModel(started.promise, fail, sealStyle);
  const child = new ChildModel(started.resolve);
  const parentOptions = {
    model: 'fake-parent',
    fallbacks: [
      {
        provider: Providers.OPENAI,
        clientOptions: { model: 'unused-fallback' },
      },
    ],
  };
  const graph = new StandardGraph({
    runId: 'eager-subagent-integration',
    agents: [
      {
        agentId: 'parent',
        provider: Providers.OPENAI,
        instructions: 'Delegate work.',
        clientOptions: parentOptions,
        toolDefinitions: [
          {
            name: 'unused',
            description: 'unused',
            parameters: { type: 'object', properties: {} },
          },
        ],
        subagentConfigs: [
          {
            type: 'researcher',
            name: 'Researcher',
            description: 'Research tasks',
            agentInputs: {
              agentId: 'researcher',
              provider: Providers.OPENAI,
              instructions: 'Research.',
            },
          },
        ],
      },
    ],
  });
  graph.eagerEventToolExecution = { enabled: true };
  graph.overrideModel = parent;
  graph.setSubagentModelOverride(child);
  return { graph, parent, child };
}

describe('Eager subagents through the real graph and provider stream', () => {
  it.each(['responses', 'bedrock', 'google'] as const)(
    'starts the first child before the second prompt using %s seals',
    async (style) => {
      const { graph, parent, child } = fixture(false, style);
      try {
        const result = await graph.createWorkflow().invoke(
          { messages: [new HumanMessage('Research two things')] },
          {
            configurable: { thread_id: 'thread', run_id: 'run' },
          }
        );
        expect(child.starts).toBe(2);
        expect(parent.observed).toHaveLength(2);
        expect(
          result.messages.filter((message) => message instanceof ToolMessage)
        ).toHaveLength(2);
      } finally {
        graph.clearHeavyState();
      }
    }
  );

  it('fails closed instead of retrying a provider after the first child starts', async () => {
    const { graph, child } = fixture(true);
    try {
      await expect(
        graph.createWorkflow().invoke(
          { messages: [new HumanMessage('Research two things')] },
          {
            configurable: { thread_id: 'thread', run_id: 'run' },
          }
        )
      ).rejects.toBeInstanceOf(PreparedSubagentError);
      expect(child.starts).toBe(1);
    } finally {
      graph.clearHeavyState();
    }
  });
  it('preserves eager prestart with only the optional internal output reader', () => {
    const { graph } = fixture();
    graph.toolOutputReferences = { enabled: true, readTool: true };
    graph.createWorkflow();
    const agent = graph.agentContexts.get('parent')!;
    expect(graph.canPrestartSubagents(agent)).toBe(true);
    const customTool = tool(async () => 'ordinary', {
      name: 'ordinary_graph_tool', description: 'Ordinary graph tool', schema: z.object({}),
    });
    agent.graphTools = [...(agent.graphTools ?? []), customTool];
    expect(graph.canPrestartSubagents(agent)).toBe(false);
    graph.clearHeavyState();
  });

  it('keeps durable checkpoint execution on the normal path', () => {
    const { graph } = fixture();
    graph.createWorkflow();
    const agent = graph.agentContexts.get('parent');
    expect(graph.canPrestartSubagents(agent)).toBe(true);
    graph.compileOptions = { checkpointer: new MemorySaver() };
    expect(graph.canPrestartSubagents(agent)).toBe(false);
    graph.clearHeavyState();
  });

  it('keeps interrupting or disabled configurations on the normal path', () => {
    const { graph } = fixture();
    graph.createWorkflow();
    const agent = graph.agentContexts.get('parent');
    graph.interruptingToolNames = ['ask_user_question'];
    expect(graph.canPrestartSubagents(agent)).toBe(false);
    graph.interruptingToolNames = undefined;
    graph.eagerEventToolExecution = { enabled: true, maxPendingSubagents: 0 };
    expect(graph.canPrestartSubagents(agent)).toBe(false);
    graph.clearHeavyState();
  });
});

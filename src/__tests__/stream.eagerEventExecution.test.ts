import { AIMessageChunk } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
  OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
  ToolCallTypes,
} from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { HandlerRegistry } from '@/events';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';

function createResultAlteringRegistry(): HookRegistry {
  const registry = new HookRegistry();
  registry.register('PreToolUse', { hooks: [async () => ({})] });
  return registry;
}

function createGraph(overrides: Partial<StandardGraph> = {}): StandardGraph {
  const runSteps = new Map<string, t.RunStep>();
  const stepIdsByKey = new Map<string, string>();
  let stepCounter = 0;
  const handlerRegistry = new HandlerRegistry();
  handlerRegistry.register(GraphEvents.ON_TOOL_EXECUTE, {
    handle: async () => undefined,
  });
  const eagerUsageCount = new Map<string, number>();

  const graph = {
    config: {
      configurable: { user_id: 'user_1' },
      metadata: { run_id: 'run_1' },
    },
    breakerAbort: new AbortController(),
    eagerEventToolExecution: { enabled: true },
    eagerEventToolExecutions: new Map(),
    eagerEventToolUsageCount: eagerUsageCount,
    getEagerEventToolUsageCount: jest.fn(() => eagerUsageCount),
    eagerEventToolCallChunks: new Map(),
    handlerRegistry,
    hookRegistry: undefined,
    humanInTheLoop: undefined,
    toolOutputReferences: undefined,
    sessions: new Map(),
    toolCallStepIds: new Map(),
    stepKeyIds: new Map(),
    messageIdsByStepKey: new Map(),
    messageStepHasToolCalls: new Map(),
    prelimMessageIdsByStepKey: new Map(),
    getAgentContext: jest.fn(
      (): Partial<AgentContext> => ({
        provider: Providers.ANTHROPIC,
        reasoningKey: 'reasoning',
        toolDefinitions: [{ name: 'weather' }],
        graphTools: [],
        agentId: 'agent_1',
        getCallerCapabilityProjectionSnapshot: jest.fn(() => ({
          version: 1 as const,
          directToolNames: ['weather'],
          codeExecutionToolNames: [],
          directOnlyToolNames: ['weather'],
          codeExecutionOnlyToolNames: [],
        })),
      })
    ),
    getStepKey: jest.fn(() => 'step-key'),
    getStepIdByKey: jest.fn((stepKey: string) => {
      const stepId = stepIdsByKey.get(stepKey);
      if (stepId == null) {
        throw new Error('no current step');
      }
      return stepId;
    }),
    getRunStep: jest.fn((stepId: string) => runSteps.get(stepId)),
    dispatchRunStep: jest.fn(async (stepKey: string, details: unknown) => {
      const id = `step_${++stepCounter}`;
      if (
        (details as t.StepDetails).type === StepTypes.TOOL_CALLS &&
        Array.isArray((details as t.ToolCallsDetails).tool_calls)
      ) {
        for (const toolCall of (details as t.ToolCallsDetails).tool_calls ??
          []) {
          if (toolCall.id != null && toolCall.id !== '') {
            graph.toolCallStepIds.set(toolCall.id, id);
          }
        }
      }
      stepIdsByKey.set(stepKey, id);
      runSteps.set(id, {
        id,
        type: (details as { type: t.RunStep['type'] }).type,
        stepDetails: details as t.RunStep['stepDetails'],
      } as t.RunStep);
      return id;
    }),
    dispatchRunStepDelta: jest.fn(async () => undefined),
    ...overrides,
  };

  return graph as unknown as StandardGraph;
}

function chunkStateKey(stepKey: string, chunkKey: string | number): string {
  return `${stepKey}\u0000${String(chunkKey)}`;
}

function attachContentAggregator(graph: StandardGraph): {
  contentParts: Array<t.MessageContentComplex | undefined>;
} {
  const { contentParts, aggregateContent } = createContentAggregator();
  const dispatchRunStep = graph.dispatchRunStep.bind(graph);
  const stepIndices = new Map<string, number>();
  graph.dispatchRunStep = jest.fn<StandardGraph['dispatchRunStep']>(
    async (stepKey, stepDetails, metadata) => {
      const stepId = await dispatchRunStep(stepKey, stepDetails, metadata);
      const runStep = graph.getRunStep(stepId);
      if (runStep != null) {
        if (!stepIndices.has(stepId)) {
          stepIndices.set(stepId, stepIndices.size);
        }
        aggregateContent({
          event: GraphEvents.ON_RUN_STEP,
          data: {
            ...runStep,
            index: stepIndices.get(stepId) ?? 0,
          },
        });
      }
      return stepId;
    }
  );
  graph.dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
    async (id, delta) => {
      aggregateContent({
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: { id, delta } as t.MessageDeltaEvent,
      });
    }
  );
  return { contentParts };
}

const finalToolCallResponseMetadata = { finish_reason: 'tool_calls' };
const openAIResponsesToolCallMetadata = {
  [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
    OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
};
const bedrockConverseToolCallMetadata = {
  [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
    BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
};
const googleOnArrivalToolCallMetadata = {
  [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]: GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
  [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
};
const openAIChatSequentialToolCallMetadata = {
  [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
    OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
};

describe('ChatModelStreamHandler eager event tool execution', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('prestarts a complete event-driven tool call from the stream', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(toolExecuteCalls[0].callerCapabilityProjection).toEqual({
      version: 1,
      directToolNames: ['weather'],
      codeExecutionToolNames: [],
      directOnlyToolNames: ['weather'],
      codeExecutionOnlyToolNames: [],
    });
    expect(graph.eagerEventToolExecutions.get('call_weather')).toMatchObject({
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
    });
    expect(graph.toolCallStepIds.has('call_weather')).toBe(true);
  });

  it('captures normalized code-session identity before eager dispatch', async () => {
    const graph = createGraph({
      sessions: new Map([
        [
          Constants.EXECUTE_CODE,
          {
            session_id: 'exec-session',
            files: [{ id: 'old-file', name: 'report.csv' }],
            lastUpdated: 1,
          },
        ],
      ]),
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          toolDefinitions: [{ name: Constants.EXECUTE_CODE }],
          graphTools: [],
          agentId: 'agent_1',
          codeSessionKey: Constants.EXECUTE_CODE,
          getCallerCapabilityProjectionSnapshot: jest.fn(() => ({
            version: 1 as const,
            directToolNames: [],
            codeExecutionToolNames: [Constants.EXECUTE_CODE],
            directOnlyToolNames: [],
            codeExecutionOnlyToolNames: [Constants.EXECUTE_CODE],
          })),
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
        const batch = data as t.ToolExecuteBatchRequest;
        batch.toolCalls[0].codeSessionContext = {
          session_id: 'new-session',
          files: [
            {
              id: 'new-file',
              resource_id: 'new-file',
              name: 'report.csv',
              storage_session_id: 'new-storage',
              kind: 'user',
            },
          ],
        };
        batch.resolve([
          {
            toolCallId: 'call_code',
            status: 'success',
            content: '',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            { id: 'call_code', name: Constants.EXECUTE_CODE, args: {} },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    const record = graph.eagerEventToolExecutions.get('call_code');
    expect(record?.codeSessionBaselineByName?.get('report.csv')).toBe(
      'exec-session\0old-file'
    );
    expect(record?.request.codeSessionContext?.files?.[0].id).toBe('new-file');
  });

  it('prestarts when subagent callback forwarding can execute tools without a handler registry', async () => {
    const graph = createGraph({
      handlerRegistry: undefined,
      eventToolExecutionAvailable: true,
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
  });

  it('does not prestart parseable tool calls before a final tool-call signal', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'N' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(false);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('prestarts multiple complete event-driven tool calls as one batch', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'calendar' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.name} result`,
          }))
        );
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_calendar',
              name: 'calendar',
              args: { date: 'today' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toHaveLength(2);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
      expect.objectContaining({
        id: 'call_calendar',
        name: 'calendar',
        args: { date: 'today' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
    ]);
    const weatherExecution = graph.eagerEventToolExecutions.get('call_weather');
    const calendarExecution =
      graph.eagerEventToolExecutions.get('call_calendar');
    expect(weatherExecution).toMatchObject({
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
    });
    expect(calendarExecution).toMatchObject({
      toolCallId: 'call_calendar',
      toolName: 'calendar',
      args: { date: 'today' },
    });
    expect(weatherExecution?.promise).toBe(calendarExecution?.promise);
  });

  it('assigns same-tool eager turns in model emission order', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.args.city} weather`,
          }))
        );
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls.map((call) => call.turn)).toEqual([
      0, 1,
    ]);
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
    expect(
      graph.eagerEventToolExecutions.get('call_weather_1')?.request.turn
    ).toBe(0);
    expect(
      graph.eagerEventToolExecutions.get('call_weather_2')?.request.turn
    ).toBe(1);
  });

  it('scopes eager turn reservation by agent', async () => {
    const usageByAgent = new Map<string, Map<string, number>>();
    const getUsageCount = (agentId?: string): Map<string, number> => {
      const key = agentId ?? 'default';
      let usage = usageByAgent.get(key);
      if (usage == null) {
        usage = new Map<string, number>();
        usageByAgent.set(key, usage);
      }
      return usage;
    };
    const graph = createGraph({
      getEagerEventToolUsageCount: jest.fn(getUsageCount),
      getAgentContext: jest.fn(
        (metadata?: Record<string, unknown>): AgentContext =>
          ({
            provider: Providers.ANTHROPIC,
            reasoningKey: 'reasoning',
            toolDefinitions: [{ name: 'weather' }],
            graphTools: [],
            agentId:
              metadata?.langgraph_node === 'agent_2' ? 'agent_2' : 'agent_1',
          }) as unknown as AgentContext
      ),
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: 'ok',
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_1_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_1' },
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_2_weather',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_2' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls.map((call) => call.toolCalls[0].turn)).toEqual([
      0, 0,
    ]);
    expect(usageByAgent.get('agent_1')?.get('weather')).toBe(1);
    expect(usageByAgent.get('agent_2')?.get('weather')).toBe(1);
    expect(
      graph.eagerEventToolExecutions.get('call_agent_1_weather')?.request.turn
    ).toBe(0);
    expect(
      graph.eagerEventToolExecutions.get('call_agent_2_weather')?.request.turn
    ).toBe(0);
  });

  it('skips eager for the whole batch if any call is not request-plannable', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_bad',
              name: 'weather',
              args: '{"city":',
            },
            {
              id: 'call_weather_good',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolUsageCount.size).toBe(0);
  });

  it('records complete chunk-only tool calls after creating a tool step', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.toolCallStepIds.has('call_weather')).toBe(true);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"city":"NYC"}');
  });

  it('prestarts OpenAI Responses streamed tool calls on explicit arguments done', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '',
              index: 0,
            },
          ],
          response_metadata: openAIResponsesToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"N',
              index: 0,
            },
          ],
          response_metadata: openAIResponsesToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
          response_metadata: {
            ...openAIResponsesToolCallMetadata,
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: {
              kind: 'single',
              id: 'call_weather',
              index: 0,
            },
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(
      graph.eagerEventToolCallChunks.has(chunkStateKey('step-key', 0))
    ).toBe(false);
  });

  it('keeps OpenAI Chat Completions streamed chunks on the final tool_calls path', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"CH"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        args: { city: 'NYC' },
        turn: 0,
      }),
      expect.objectContaining({
        id: 'call_stock',
        args: { ticker: 'CH' },
        turn: 0,
      }),
    ]);
  });

  it('prestarts final tool calls even when the final chunk also has tool-call chunks', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              index: 0,
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      turn: 0,
    });
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
  });

  it('waits for final tool calls before prestarting streamed chunk calls', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: {},
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city"',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: ':"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: {},
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"ticker":"CH"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('preserves repeated adjacent argument deltas', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_repeat',
            status: 'success',
            content: 'ok',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: '{"word":"b',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'k"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"word":"book"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: { word: 'book' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_repeat',
      name: 'weather',
      args: { word: 'book' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('preserves identical incremental argument fragments', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('oo');
  });

  it('deduplicates repeated observed multi-character fragments', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: '{"titl',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"titl',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"titl');
  });

  it('does not prestart from cumulative streamed args before final tool calls', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"ci',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"N',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"NYC","unit":"C"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"city":"NYC","unit":"C"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC', unit: 'C' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC', unit: 'C' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('merges overlapping cumulative streamed args without duplicating suffixes', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"tit',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"title":"alpha"',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'le":"alpha","city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"title":"alpha","city":"NYC"}');
  });

  it('preserves repeated deltas from a reused stream chunk object', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableToolChunk: {
      id?: string;
      name?: string;
      args: string;
      index: number;
    } = {
      id: 'call_repeat',
      name: 'weather',
      args: '{"word":"b',
      index: 0,
    };
    const reusableChunk = {
      content: '',
      tool_call_chunks: [reusableToolChunk],
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.id = undefined;
    reusableToolChunk.name = undefined;
    reusableToolChunk.args = 'o';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.args = 'k"}';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"word":"book"}');
  });

  it('preserves repeated text deltas from a reused stream chunk object', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableChunk = { content: 'ha' } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).toHaveBeenCalledTimes(2);
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      1,
      expect.stringMatching(/^step_/),
      { content: [{ type: ContentTypes.TEXT, text: 'ha' }] },
      metadata
    );
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      2,
      expect.stringMatching(/^step_/),
      { content: [{ type: ContentTypes.TEXT, text: 'ha' }] },
      metadata
    );
  });

  it('defers disableStreaming mixed hidden reasoning and visible text chunks', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const dispatchReasoningDelta = jest.fn<
      StandardGraph['dispatchReasoningDelta']
    >(async () => undefined);
    const agentContext: Partial<AgentContext> = {
      provider: Providers.OPENAI,
      reasoningKey: 'reasoning_content',
      clientOptions: { disableStreaming: true } as t.OpenAIClientOptions,
      currentTokenType: ContentTypes.TEXT,
      toolDefinitions: [],
      graphTools: [],
      agentId: 'agent_1',
    };
    const graph = createGraph({
      dispatchMessageDelta,
      dispatchReasoningDelta,
      getAgentContext: jest.fn(
        () => agentContext
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    const mixedChunk = {
      content: 'final answer',
      additional_kwargs: {
        reasoning_content: 'hidden reasoning',
      },
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: mixedChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: mixedChunk },
      metadata,
      graph
    );

    expect(dispatchReasoningDelta).not.toHaveBeenCalled();
    expect(dispatchMessageDelta).not.toHaveBeenCalled();
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
  });

  it('leaves mixed hidden reasoning and visible text chunks on the streaming path', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const dispatchReasoningDelta = jest.fn<
      StandardGraph['dispatchReasoningDelta']
    >(async () => undefined);
    const graph = createGraph({
      dispatchMessageDelta,
      dispatchReasoningDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const metadata = { langgraph_node: 'agent' };

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: 'streamed answer',
          additional_kwargs: {
            reasoning_content: 'hidden reasoning',
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchReasoningDelta).not.toHaveBeenCalled();
    expect(dispatchMessageDelta).toHaveBeenCalledTimes(1);
    expect(dispatchMessageDelta).toHaveBeenCalledWith(
      expect.stringMatching(/^step_/),
      {
        content: [{ type: ContentTypes.TEXT, text: 'streamed answer' }],
      },
      metadata
    );
  });

  it('dispatches Gemini server-side tool context blocks from mixed stream content', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };
    const textPart: t.MessageContentComplex = {
      type: ContentTypes.TEXT,
      text: 'Search complete.',
    };
    const toolResponsePart: t.MessageContentComplex = {
      type: 'toolResponse',
      toolResponse: {
        id: 'server-search-1',
        name: 'google_search',
        response: { results: [] },
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart, textPart, toolResponsePart],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).toHaveBeenCalledTimes(3);
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      1,
      expect.stringMatching(/^step_/),
      { content: [toolCallPart] },
      metadata
    );
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      2,
      expect.stringMatching(/^step_/),
      { content: [textPart] },
      metadata
    );
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      3,
      expect.stringMatching(/^step_/),
      { content: [toolResponsePart] },
      metadata
    );
  });

  it('dispatches Gemini server-side tool context onto message steps after tool chunks', async () => {
    const messageDeltas: Array<{
      id: string;
      delta: t.MessageDelta;
    }> = [];
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async (id, delta) => {
        messageDeltas.push({
          id,
          delta,
        });
      }
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).toHaveBeenCalledTimes(1);
    expect(graph.getRunStep(messageDeltas[0]?.id ?? '')?.type).toBe(
      StepTypes.MESSAGE_CREATION
    );
    expect(messageDeltas[0]?.delta).toEqual({ content: [toolCallPart] });
  });

  it('dispatches Gemini server-side tool context when reasoning is present', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const dispatchReasoningDelta = jest.fn<
      StandardGraph['dispatchReasoningDelta']
    >(async () => undefined);
    const graph = createGraph({
      dispatchMessageDelta,
      dispatchReasoningDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart],
          additional_kwargs: {
            reasoning: 'Need to search before answering.',
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).toHaveBeenCalledTimes(1);
    expect(dispatchMessageDelta).toHaveBeenCalledWith(
      expect.stringMatching(/^step_/),
      { content: [toolCallPart] },
      metadata
    );
    expect(dispatchReasoningDelta).toHaveBeenCalledWith(
      expect.stringMatching(/^step_/),
      {
        content: [
          {
            type: ContentTypes.THINK,
            think: 'Need to search before answering.',
          },
        ],
      },
      metadata
    );
  });

  it('dispatches Gemini server-side tool context before client tool calls in mixed chunks', async () => {
    const dispatchOrder: string[] = [];
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async (_id, delta) => {
        if (
          delta.content?.some(
            (contentPart) => contentPart.type === 'toolCall'
          ) === true
        ) {
          dispatchOrder.push('server-context');
        }
      }
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchRunStep = graph.dispatchRunStep.bind(graph);
    graph.dispatchRunStep = jest.fn<StandardGraph['dispatchRunStep']>(
      async (stepKey, stepDetails, metadata) => {
        const stepId = await dispatchRunStep(stepKey, stepDetails, metadata);
        if ((stepDetails as t.StepDetails).type === StepTypes.TOOL_CALLS) {
          dispatchOrder.push('client-tool-calls');
        }
        return stepId;
      }
    );
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart],
          tool_calls: [
            {
              type: ToolCallTypes.TOOL_CALL,
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchOrder).toEqual(['server-context', 'client-tool-calls']);
  });

  it('does not dispatch Gemini server-side tool context blocks for non-Google providers', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).not.toHaveBeenCalled();
  });

  it('keeps separately streamed Gemini server-side tool context blocks on separate message steps', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const { contentParts } = attachContentAggregator(graph);
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };
    const toolResponsePart: t.MessageContentComplex = {
      type: 'toolResponse',
      toolResponse: {
        id: 'server-search-1',
        name: 'google_search',
        response: { results: [] },
      },
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolCallPart],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [toolResponsePart],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'Done',
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: ' again.',
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(contentParts).toEqual([
      toolCallPart,
      toolResponsePart,
      {
        type: ContentTypes.TEXT,
        text: 'Done again.',
      },
    ]);
  });

  it('processes a reused chunk object when its streamed payload changes', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableToolChunk = {
      id: 'call_weather',
      name: 'weather',
      args: '{"city"',
      index: 0,
    };
    const reusableChunk = {
      content: '',
      tool_call_chunks: [reusableToolChunk],
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.args = ':"NYC"}';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))?.argsText
    ).toBe('{"city":"NYC"}');
  });

  it('does not share chunk object de-duplication across graphs', async () => {
    const graphA = createGraph();
    const graphB = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.id} result`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const sharedChunk = {
      content: '',
      tool_calls: [
        {
          id: 'call_weather',
          name: 'weather',
          args: { city: 'NYC' },
        },
      ],
      response_metadata: finalToolCallResponseMetadata,
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: sharedChunk },
      { langgraph_node: 'agent' },
      graphA
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: sharedChunk },
      { langgraph_node: 'agent' },
      graphB
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(graphA.eagerEventToolExecutions.has('call_weather')).toBe(true);
    expect(graphB.eagerEventToolExecutions.has('call_weather')).toBe(true);
  });

  it('prestarts a completed streamed tool when a later tool call begins', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
    ]);
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
    expect(graph.eagerEventToolExecutions.has('call_stock')).toBe(false);
    expect(
      graph.eagerEventToolCallChunks.has(chunkStateKey('step-key', 0))
    ).toBe(false);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 1))?.argsText
    ).toBe('{"ticker":"C');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'H"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('emits a completed event when an eager streamed tool result settles', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const completedEvents: Array<{ result: t.ToolEndEvent }> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completedEvents.push(data as { result: t.ToolEndEvent });
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await graph.eagerEventToolExecutions.get('call_weather')?.promise;

    expect(completedEvents).toHaveLength(1);
    expect(completedEvents[0].result).toMatchObject({
      id: expect.stringMatching(/^step_/),
      type: 'tool_call',
      eager: true,
      tool_call: {
        id: 'call_weather',
        name: 'weather',
        args: '{"city":"NYC"}',
        output: 'ok weather',
        progress: 1,
      },
    });
    expect(
      graph.eagerEventToolExecutions.get('call_weather')?.completionDispatched
    ).toBe(true);
  });

  it('serializes bigint output before eager completion dispatch', async () => {
    const graph = createGraph();
    const completedEvents: Array<{ result: t.ToolEndEvent }> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completedEvents.push(data as { result: t.ToolEndEvent });
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: [{ rowsRead: BigInt(42) }],
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await graph.eagerEventToolExecutions.get('call_weather')?.promise;

    expect(completedEvents).toHaveLength(1);
    expect(
      (
        completedEvents[0].result as {
          tool_call?: { output?: string };
        }
      ).tool_call?.output
    ).toBe('[{"rowsRead":"42"}]');
  });

  it('does not emit a stale eager completion after the active execution is invalidated', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const completedEvents: Array<{ result: t.ToolEndEvent }> = [];
    let pendingBatch: t.ToolExecuteBatchRequest | undefined;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completedEvents.push(data as { result: t.ToolEndEvent });
          return;
        }
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          pendingBatch = data as t.ToolExecuteBatchRequest;
        }
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    const staleRecord = graph.eagerEventToolExecutions.get('call_weather');
    expect(staleRecord).toBeDefined();
    expect(pendingBatch?.toolCalls).toHaveLength(1);

    graph.eagerEventToolExecutions.delete('call_weather');
    pendingBatch?.resolve([
      {
        toolCallId: 'call_weather',
        status: 'success',
        content: 'stale weather',
      },
    ]);
    await staleRecord?.promise;

    expect(completedEvents).toHaveLength(0);
    expect(staleRecord?.completionDispatched).toBeUndefined();
  });

  it('does not mark eager completion dispatched when event delivery is swallowed', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<boolean | void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          return false;
        }
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          const batch = data as t.ToolExecuteBatchRequest;
          batch.resolve([
            {
              toolCallId: 'call_weather',
              status: 'success',
              content: 'weather result',
            },
          ]);
        }
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    const record = graph.eagerEventToolExecutions.get('call_weather');
    await record?.promise;

    expect(record?.completionDispatched).toBeUndefined();
  });

  it('does not overwrite a completed tool output with later streamed deltas', () => {
    const { contentParts, aggregateContent } = createContentAggregator();

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_weather',
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: {},
            },
          ],
        },
        usage: null,
      } as t.RunStep,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step_weather',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_weather',
            name: 'weather',
            args: '{"city":"NYC"}',
            output: 'sunny',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_weather',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC revised"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts[0]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_weather',
        name: 'weather',
        args: '{"city":"NYC"}',
        output: 'sunny',
        progress: 1,
      },
    });
  });

  it('does not use next-index sealing for Moonshot streamed tool chunks', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.MOONSHOT,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC revised' },
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC revised' },
      }),
      expect.objectContaining({
        id: 'call_stock',
        name: 'stock',
        args: { ticker: 'CH' },
      }),
    ]);
  });

  it('does not seal a streamed tool when the same chunk also carries its own index', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '',
              index: 0,
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'H"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
    });
  });

  it('preserves same-tool turns across per-call streamed eager starts', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.args.city}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather_2',
              name: 'weather',
              args: '{"city":"B',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather_1',
      name: 'weather',
      args: { city: 'NYC' },
      turn: 0,
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'oston"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_weather_2',
      name: 'weather',
      args: { city: 'Boston' },
      turn: 1,
    });
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
  });

  it('scopes streamed chunk accumulation by step key', async () => {
    const graph = createGraph({
      getStepKey: jest.fn((metadata?: Record<string, unknown>) =>
        String(metadata?.langgraph_node ?? 'step-key')
      ),
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_agent_a',
              name: 'weather',
              args: '{"city":"N',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_agent_b',
              name: 'weather',
              args: '{"city":"S',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'F"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'YC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_a', 0))?.argsText
    ).toBe('{"city":"NYC"}');
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_b', 0))?.argsText
    ).toBe('{"city":"SF"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_b',
              name: 'weather',
              args: { city: 'SF' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.has(chunkStateKey('agent_b', 0))
    ).toBe(false);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_a', 0))?.argsText
    ).toBe('{"city":"NYC"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_a',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_agent_b',
      name: 'weather',
      args: { city: 'SF' },
    });
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_agent_a',
      name: 'weather',
      args: { city: 'NYC' },
    });
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('does not prestart when parent-run session hooks alter results', async () => {
    // A subagent child graph has its own runId, but ToolNode executes hooks
    // under the PARENT run id from `configurable.run_id` ONLY — the prestart
    // gate must read the same source (a differing metadata run id here is a
    // deliberate distractor pinning the precedence).
    const sessionRegistry = new HookRegistry();
    sessionRegistry.registerSession('parent-run', 'PreToolUse', {
      hooks: [async () => ({ decision: 'allow' as const })],
    });
    const graph = createGraph({
      runId: 'child-run',
      hookRegistry: sessionRegistry,
      config: {
        configurable: { user_id: 'user_1', run_id: 'parent-run' },
        metadata: { run_id: 'run_1' },
      },
    } as Partial<StandardGraph>);
    const sessionDispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent', run_id: 'some-other-run' },
      graph
    );

    expect(sessionDispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart when batch-sensitive hooks are configured', async () => {
    const graph = createGraph({
      hookRegistry: createResultAlteringRegistry(),
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('prestarts when the registry only has observation hooks (PostToolBatch)', async () => {
    const observationRegistry = new HookRegistry();
    observationRegistry.register('PostToolBatch', {
      hooks: [async () => ({})],
    });
    const graph = createGraph({ hookRegistry: observationRegistry });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(graph.eagerEventToolExecutions.get('call_weather')).toMatchObject({
      toolCallId: 'call_weather',
      toolName: 'weather',
    });
  });

  it('does not buffer streamed chunks when eager execution is disabled', async () => {
    const graph = createGraph({
      eagerEventToolExecution: { enabled: false },
    } as Partial<StandardGraph>);
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('does not prestart local-engine direct coding tools', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: Constants.EXECUTE_CODE }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: { code: 'print(1)' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('does not prestart streamed local-engine direct coding tools', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [
            { name: Constants.EXECUTE_CODE },
            { name: 'weather' },
          ],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: '{"code":"print(1)"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('does not prestart streamed Cloudflare sandbox direct coding tools', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'cloudflare-sandbox',
        cloudflare: { sandbox: {} },
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: Constants.BASH_TOOL }, { name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_bash',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo ok"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('prestarts streamed remote bash tools when the next Anthropic tool call begins', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: Constants.READ_FILE },
          ],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_env',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo env"}',
              index: 2,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_net',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo net"}',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'toolu_env',
        name: Constants.BASH_TOOL,
        args: { command: 'echo env' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
    ]);
    expect(graph.eagerEventToolExecutions.has('toolu_env')).toBe(true);
    expect(graph.eagerEventToolExecutions.has('toolu_net')).toBe(false);
  });

  it('does not prestart streamed remote tools when graph tools may appear later', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: Constants.READ_FILE },
          ],
          graphTools: [
            { name: 'transfer_to_researcher' } as unknown as t.GenericTool,
          ],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_env',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo env"}',
              index: 2,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_net',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo net"}',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('prestarts streamed remote bash tools when code output references are enabled', async () => {
    const graph = createGraph({
      toolOutputReferences: { enabled: true },
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: Constants.READ_FILE },
          ],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    for (const args of ['{"command":"echo ', 'env && ', 'pwd"}']) {
      await handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        {
          chunk: {
            content: '',
            tool_call_chunks: [
              {
                id: 'toolu_env',
                name: Constants.BASH_TOOL,
                args,
                index: 2,
              },
            ],
          } as unknown as t.StreamChunk,
        },
        metadata,
        graph
      );
    }

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_net',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo net"}',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'toolu_env',
        name: Constants.BASH_TOOL,
        args: { command: 'echo env && pwd' },
      }),
    ]);
  });

  it('does not prestart streamed code tools whose args contain output references', async () => {
    const graph = createGraph({
      toolOutputReferences: { enabled: true },
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: Constants.READ_FILE },
          ],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_ref',
              name: Constants.BASH_TOOL,
              args: '{"command":"cat <<EOF\\n{{tool0turn0}}\\nEOF"}',
              index: 2,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_next',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo next"}',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
  });

  it('does not prestart streamed tools when the next Anthropic tool call is a graph tool', async () => {
    const handoffToolName = `${Constants.LC_TRANSFER_TO_}researcher`;
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: handoffToolName },
          ],
          graphTools: [{ name: handoffToolName } as unknown as t.GenericTool],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_env',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo env"}',
              index: 2,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_handoff',
              name: handoffToolName,
              args: '{"message":"check this"}',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart streamed tools after a graph tool appeared earlier in the same step', async () => {
    const handoffToolName = `${Constants.LC_TRANSFER_TO_}researcher`;
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.ANTHROPIC,
          reasoningKey: 'reasoning',
          toolDefinitions: [
            { name: Constants.BASH_TOOL },
            { name: handoffToolName },
          ],
          graphTools: [{ name: handoffToolName } as unknown as t.GenericTool],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_env',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo env"}',
              index: 2,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_handoff',
              name: handoffToolName,
              args: '{"message":"partial',
              index: 3,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'toolu_next',
              name: Constants.BASH_TOOL,
              args: '{"command":"echo next"}',
              index: 4,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart event tools in a mixed direct-tool batch', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [
            { name: Constants.EXECUTE_CODE },
            { name: 'weather' },
          ],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: { code: 'print(1)' },
            },
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('continues eager turns after normal event-dispatch usage', async () => {
    const graph = createGraph();
    graph.eagerEventToolUsageCount.set('weather', 1);
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather_2',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather_2',
      name: 'weather',
      turn: 1,
    });
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
  });

  it('prestarts Bedrock Converse streamed tool calls on content block stop seals', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.BEDROCK,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              index: 1,
            },
          ],
          response_metadata: bedrockConverseToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"N',
              index: 1,
            },
          ],
          response_metadata: bedrockConverseToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'YC"}',
              index: 1,
            },
          ],
          response_metadata: bedrockConverseToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '',
              index: 1,
            },
          ],
          response_metadata: {
            ...bedrockConverseToolCallMetadata,
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: {
              kind: 'single',
              index: 1,
            },
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(
      graph.eagerEventToolCallChunks.has(chunkStateKey('step-key', 1))
    ).toBe(false);
  });

  it('prestarts each Bedrock tool call at its own stop seal in multi-call turns', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.BEDROCK,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const sendChunk = async (chunk: Record<string, unknown>): Promise<void> =>
      handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        { chunk: chunk as unknown as t.StreamChunk },
        metadata,
        graph
      );

    await sendChunk({
      content: '',
      tool_call_chunks: [{ id: 'call_weather', name: 'weather', index: 1 }],
      response_metadata: bedrockConverseToolCallMetadata,
    });
    await sendChunk({
      content: '',
      tool_call_chunks: [{ args: '{"city":"NYC"}', index: 1 }],
      response_metadata: bedrockConverseToolCallMetadata,
    });
    await sendChunk({
      content: '',
      tool_call_chunks: [{ args: '', index: 1 }],
      response_metadata: {
        ...bedrockConverseToolCallMetadata,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 1 },
      },
    });

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      args: { city: 'NYC' },
    });

    await sendChunk({
      content: '',
      tool_call_chunks: [{ id: 'call_stock', name: 'stock', index: 2 }],
      response_metadata: bedrockConverseToolCallMetadata,
    });
    await sendChunk({
      content: '',
      tool_call_chunks: [{ args: '{"ticker":"CH"}', index: 2 }],
      response_metadata: bedrockConverseToolCallMetadata,
    });

    expect(toolExecuteCalls).toHaveLength(1);

    await sendChunk({
      content: '',
      tool_call_chunks: [{ args: '', index: 2 }],
      response_metadata: {
        ...bedrockConverseToolCallMetadata,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 2 },
      },
    });

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      args: { ticker: 'CH' },
      turn: 0,
    });
  });

  it('prestarts Google tool calls sealed on arrival without a final tool-call signal', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });

    // A later chunk with another sealed call prestarts independently.
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"CH"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
      turn: 0,
    });
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
    expect(graph.eagerEventToolExecutions.has('call_stock')).toBe(true);
  });

  it('keeps Google tool calls lazy without an on-arrival seal', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        toolExecuteCalls.push(data as t.ToolExecuteBatchRequest);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart on-arrival sealed calls when batch-sensitive hooks are configured', async () => {
    const graph = createGraph({
      hookRegistry: createResultAlteringRegistry(),
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart on-arrival sealed calls for programmatic tool calling batches', async () => {
    const graph = createGraph();
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      {
        langgraph_node: 'agent',
        [Constants.PROGRAMMATIC_TOOL_CALLING]: true,
      },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart on-arrival sealed calls when eager execution is disabled', async () => {
    const graph = createGraph({
      eagerEventToolExecution: { enabled: false },
    } as Partial<StandardGraph>);
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('prestarts official OpenAI sequential streamed tool calls when a later index begins', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      });

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
          response_metadata: openAIChatSequentialToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              index: 1,
            },
          ],
          response_metadata: openAIChatSequentialToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"ticker":"CH"}',
              index: 1,
            },
          ],
          response_metadata: openAIChatSequentialToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    // The final cumulative tool_calls chunk dispatches only the unstarted call.
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls).toHaveLength(1);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
      turn: 0,
    });
  });

  it('does not prestart on-arrival sealed calls when direct tools may stream', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.GOOGLE,
          reasoningKey: 'reasoning',
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [{ name: 'lookup' }],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        toolExecuteCalls.push(data as t.ToolExecuteBatchRequest);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: googleOnArrivalToolCallMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart tools listed in excludeToolNames', async () => {
    const graph = createGraph({
      eagerEventToolExecution: {
        enabled: true,
        excludeToolNames: ['create_file'],
      },
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          { toolCallId: 'call_file', status: 'success', content: 'ok' },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_file',
              name: 'create_file',
              args: { path: '/mnt/data/x.py', content: 'print(1)' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    // Excluded: no eager execution started; the call falls through to ToolNode.
    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_file')).toBe(false);
  });

  it('does not prestart code execution for a custom agent session partition', async () => {
    const graph = createGraph({
      eagerEventToolExecution: { enabled: true },
      getAgentContext: jest.fn(() => ({
        provider: Providers.ANTHROPIC,
        reasoningKey: 'reasoning',
        toolDefinitions: [{ name: Constants.EXECUTE_CODE }],
        graphTools: [],
        agentId: 'stateful-agent',
        codeSessionKey: 'execute_code:stateful:user-1',
      })) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_TOOL_EXECUTE) {
          toolExecuteCalls.push(data as t.ToolExecuteBatchRequest);
        }
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: { lang: 'python', code: 'print(1)' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_code')).toBe(false);
  });

  it('does not prestart codeSessionToolNames tools even without excludeToolNames', async () => {
    // A declared session-writing host tool is side-effecting, so it must not be
    // eagerly prestarted even when the host didn't also list it in excludeToolNames.
    const graph = createGraph({
      eagerEventToolExecution: { enabled: true },
      codeSessionToolNames: ['create_file', 'edit_file'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          { toolCallId: 'call_cf2', status: 'success', content: 'ok' },
        ]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_cf2',
              name: 'create_file',
              args: { path: '/mnt/data/y.py', content: 'print(2)' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_cf2')).toBe(false);
  });

  it('keeps the direct-tool batch guard when an excluded tool is also direct', async () => {
    // edit_file is both a direct graph tool AND excluded from eager. A mixed
    // batch with a direct tool must suppress eager for the whole batch, so the
    // sibling event tool must NOT prestart — excluding edit_file must not hide
    // it from the batch-level direct-tool guard.
    const graph = createGraph({
      eagerEventToolExecution: {
        enabled: true,
        excludeToolNames: ['edit_file'],
      },
      getAgentContext: jest.fn(() => ({
        provider: Providers.ANTHROPIC,
        reasoningKey: 'reasoning',
        toolDefinitions: [{ name: 'weather' }, { name: 'edit_file' }],
        graphTools: [{ name: 'edit_file' }],
        agentId: 'agent_1',
      })) as unknown as StandardGraph['getAgentContext'],
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        toolExecuteCalls.push(data as t.ToolExecuteBatchRequest);
        (data as t.ToolExecuteBatchRequest).resolve([]);
      });

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
            {
              id: 'call_edit',
              name: 'edit_file',
              args: { path: '/mnt/data/x.py' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    // Direct tool in batch suppresses eager for the whole batch.
    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(false);
    expect(graph.eagerEventToolExecutions.has('call_edit')).toBe(false);
  });
});

describe('executable argument readiness', () => {
  it('prevents eager host execution for string-valued parsed calls even with a final seal', async () => {
    const graph = createGraph();
    const requests: t.ToolExecuteBatchRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
        const request = data as t.ToolExecuteBatchRequest;
        requests.push(request);
        request.resolve([
          { toolCallId: 'call_weather', status: 'success', content: 'sunny' },
        ]);
      });
    const raw = new AIMessageChunk({
      content: '',
      tool_calls: [
        {
          id: 'call_weather',
          name: 'weather',
          args: '{"city":"NYC"}' as unknown as Record<string, unknown>,
        },
      ],
      response_metadata: finalToolCallResponseMetadata,
    });
    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: raw },
      { langgraph_node: 'agent' },
      graph
    );
    expect(requests).toHaveLength(0);
  });
});

describe('client delegation batch barrier', () => {
  it('does not pre-execute an SDK call before a possible client-owned call arrives', async () => {
    const graph = createGraph();
    graph.clientDelegatedToolNames = new Set(['client_tool']);
    const requests: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
        const request = data as t.ToolExecuteBatchRequest;
        requests.push(request);
        request.resolve([{ toolCallId: 'call_weather', status: 'success', content: 'sunny' }]);
      }
    );
    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: new AIMessageChunk({
        content: '',
        tool_calls: [{ id: 'call_weather', name: 'weather', args: { city: 'NYC' } }],
        response_metadata: finalToolCallResponseMetadata,
      }) },
      { langgraph_node: 'agent' },
      graph
    );
    expect(requests).toHaveLength(0);
  });
});

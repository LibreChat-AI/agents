import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
  StandardGraph,
} from '@/index';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

const CHILD_RESPONSE = 'Research result: Paris is the capital of France.';
const OVERRIDDEN_CHILD_RESPONSE = 'Deterministic child override result.';

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'subagent-test-thread' },
  streamMode: 'values',
  version: 'v2' as const,
};

const createParentAgent = (): t.AgentInputs => ({
  agentId: 'parent',
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions:
    'You are a supervisor. Delegate research tasks using the subagent tool.',
  maxContextTokens: 8000,
  subagentConfigs: [
    {
      type: 'researcher',
      name: 'Research Agent',
      description: 'Researches and summarizes information',
      agentInputs: {
        agentId: 'researcher',
        provider: Providers.OPENAI,
        clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
        instructions: 'You are a research agent. Answer concisely.',
        maxContextTokens: 8000,
      },
    },
  ],
});

describe('Subagent Integration', () => {
  jest.setTimeout(30000);

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            constructor(_options: any) {
              super({ responses: [CHILD_RESPONSE] });
            }
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  it('should create subagent tool on agent context', async () => {
    const run = await Run.create<t.IState>({
      runId: `subagent-test-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
    });

    expect(run.Graph).toBeDefined();
    const parentContext = (run.Graph as StandardGraph).agentContexts.get(
      'parent'
    );
    expect(parentContext).toBeDefined();
    expect(parentContext?.graphTools).toBeDefined();

    const subagentTool = (parentContext?.graphTools as t.GenericTool[]).find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();
  });

  it('should execute subagent and return filtered result to parent', async () => {
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    };

    const run = await Run.create<t.IState>({
      runId: `subagent-exec-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    const subagentToolCall: ToolCall = {
      id: 'call_subagent_1',
      name: Constants.SUBAGENT,
      args: {
        description: 'What is the capital of France?',
        subagent_type: 'researcher',
      },
      type: 'tool_call',
    };

    run.Graph?.overrideTestModel(
      [
        'Let me delegate this research task.',
        `Based on the research: ${CHILD_RESPONSE}`,
      ],
      10,
      [subagentToolCall]
    );

    const result = await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      callerConfig
    );

    expect(result).toBeDefined();

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();
    expect(runMessages!.length).toBeGreaterThan(0);

    const toolMessages = runMessages!.filter(
      (msg) => msg._getType() === 'tool'
    );
    const subagentResult = toolMessages.find(
      (msg) => 'name' in msg && msg.name === Constants.SUBAGENT
    );
    expect(subagentResult).toBeDefined();
    expect(String(subagentResult!.content)).toContain('Paris');
  });

  it('should not create subagent tool when no subagentConfigs', async () => {
    const agentWithoutSubagents: t.AgentInputs = {
      agentId: 'plain',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Plain agent without subagents.',
      maxContextTokens: 8000,
    };

    const run = await Run.create<t.IState>({
      runId: `no-subagent-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithoutSubagents],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get('plain');
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeUndefined();
  });

  it('should handle self-spawn subagent config', async () => {
    const agentWithSelfSpawn: t.AgentInputs = {
      agentId: 'self-parent',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Agent with self-spawn for context isolation.',
      maxContextTokens: 8000,
      subagentConfigs: [
        {
          type: 'isolated',
          name: 'Isolated Worker',
          description: 'Runs a task with isolated context',
          self: true,
        },
      ],
    };

    const run = await Run.create<t.IState>({
      runId: `self-spawn-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSelfSpawn],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'self-parent'
    );
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();
  });

  it('only applies an explicitly configured subagent model override', async () => {
    const invokeSubagent = async (
      overrideSubagents: boolean
    ): Promise<string> => {
      const run = await Run.create<t.IState>({
        runId: `subagent-model-override-${overrideSubagents}-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgent()],
        },
        returnContent: true,
        skipCleanup: true,
      });
      const graph = run.Graph as StandardGraph;
      const model = new FakeListChatModel({
        responses: [OVERRIDDEN_CHILD_RESPONSE],
      });
      graph.overrideModel = model;
      if (overrideSubagents) {
        graph.setSubagentModelOverride(model);
      }

      const context = graph.agentContexts.get('parent');
      const subagentTool = (context?.graphTools as t.GenericTool[]).find(
        (tool) => 'name' in tool && tool.name === Constants.SUBAGENT
      );
      expect(subagentTool).toBeDefined();

      return String(
        await subagentTool!.invoke(
          {
            description: 'What is the capital of France?',
            subagent_type: 'researcher',
          },
          callerConfig
        )
      );
    };

    await expect(invokeSubagent(false)).resolves.toBe(CHILD_RESPONSE);
    await expect(invokeSubagent(true)).resolves.toBe(OVERRIDDEN_CHILD_RESPONSE);
  });

  it('inherits eager event-tool settings into self-spawn child graphs', async () => {
    const originalCreateWorkflow = StandardGraph.prototype.createWorkflow;
    const observedChildGraphs: Array<{
      eagerEventToolExecution: StandardGraph['eagerEventToolExecution'];
      toolOutputReferences: StandardGraph['toolOutputReferences'];
      eventToolExecutionAvailable: boolean;
    }> = [];
    const createWorkflowSpy = jest
      .spyOn(StandardGraph.prototype, 'createWorkflow')
      .mockImplementation(function (this: StandardGraph) {
        if (this.runId?.includes('_sub_') === true) {
          observedChildGraphs.push({
            eagerEventToolExecution: this.eagerEventToolExecution,
            toolOutputReferences: this.toolOutputReferences,
            eventToolExecutionAvailable: this.eventToolExecutionAvailable,
          });
          return {
            invoke: jest.fn(async () => ({
              messages: [new AIMessage('child done')],
            })),
          } as unknown as ReturnType<StandardGraph['createWorkflow']>;
        }
        return originalCreateWorkflow.call(this);
      });

    const agentWithSelfSpawn: t.AgentInputs = {
      agentId: 'self-parent',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Agent with self-spawn for context isolation.',
      maxContextTokens: 8000,
      toolDefinitions: [{ name: 'mcp_lookup' }],
      subagentConfigs: [
        {
          type: 'isolated',
          name: 'Isolated Worker',
          description: 'Runs a task with isolated context',
          self: true,
        },
      ],
    };

    const run = await Run.create<t.IState>({
      runId: `self-spawn-eager-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSelfSpawn],
      },
      customHandlers: {
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: async () => undefined,
        },
      },
      eagerEventToolExecution: { enabled: true },
      toolOutputReferences: { enabled: true },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'self-parent'
    );
    const subagentTool = (context?.graphTools as t.GenericTool[]).find(
      (tool) => 'name' in tool && tool.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();

    await subagentTool!.invoke(
      {
        description: 'Use your MCP tool.',
        subagent_type: 'isolated',
      },
      callerConfig
    );

    expect(observedChildGraphs).toEqual([
      {
        eagerEventToolExecution: { enabled: true },
        toolOutputReferences: { enabled: true },
        eventToolExecutionAvailable: true,
      },
    ]);

    createWorkflowSpy.mockRestore();
  });

  it('forwards event-driven tools through nested child graphs', async () => {
    const originalCreateWorkflow = StandardGraph.prototype.createWorkflow;
    const parentToolHandler = jest.fn(
      (_event: string, rawData: unknown): void => {
        const request = rawData as t.ToolExecuteBatchRequest;
        request.resolve(
          request.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success' as const,
            content: `ran ${call.name}`,
          }))
        );
      }
    );
    const parentUpdateHandler = jest.fn();
    const rootModelEndHandler = jest.fn();
    const nestedUsageEvents: t.SubagentUsageEvent[] = [];
    let specialistToolDefinitions: t.LCTool[] | undefined;
    let forwardedToolResults: t.ToolExecuteResult[] | undefined;

    const createWorkflowSpy = jest
      .spyOn(StandardGraph.prototype, 'createWorkflow')
      .mockImplementation(function (this: StandardGraph) {
        const workflow = originalCreateWorkflow.call(this);
        if (this.defaultAgentId === 'router') {
          return {
            invoke: jest.fn(async () => {
              const routerContext = this.agentContexts.get('router');
              const nestedTool = (
                routerContext?.graphTools as t.GenericTool[] | undefined
              )?.find(
                (tool) => 'name' in tool && tool.name === Constants.SUBAGENT
              );
              if (nestedTool == null) {
                throw new Error('Nested subagent tool was not created');
              }
              await nestedTool.invoke(
                {
                  description: 'Use the event-driven lookup tool.',
                  subagent_type: 'specialist',
                },
                callerConfig
              );
              return { messages: [new AIMessage('router done')] };
            }),
          } as unknown as ReturnType<StandardGraph['createWorkflow']>;
        }
        if (this.defaultAgentId === 'specialist') {
          specialistToolDefinitions =
            this.agentContexts.get('specialist')?.toolDefinitions;
          return {
            invoke: jest.fn(async (_state, options) => {
              const invokeOptions = options as
                | { callbacks?: unknown[] }
                | undefined;
              const callbacks = invokeOptions?.callbacks ?? [];
              const forwarder = callbacks.find(
                (callback) =>
                  typeof (callback as { handleCustomEvent?: unknown })
                    .handleCustomEvent === 'function'
              ) as
                | {
                    handleCustomEvent?: (
                      eventName: string,
                      data: unknown,
                      runId: string
                    ) => Promise<void> | void;
                  }
                | undefined;
              const usageHandler = callbacks.find(
                (callback) =>
                  typeof (callback as { handleLLMEnd?: unknown })
                    .handleLLMEnd === 'function'
              ) as
                | {
                    handleChatModelStart?: (
                      llm: unknown,
                      messages: BaseMessage[][],
                      runId: string,
                      parentRunId?: string,
                      extraParams?: unknown,
                      tags?: string[],
                      metadata?: Record<string, unknown>
                    ) => Promise<void> | void;
                    handleLLMEnd?: (
                      output: unknown,
                      runId: string
                    ) => Promise<void> | void;
                  }
                | undefined;
              if (forwarder?.handleCustomEvent != null) {
                forwardedToolResults = await new Promise<t.ToolExecuteResult[]>(
                  (resolve, reject) => {
                    const request: t.ToolExecuteBatchRequest = {
                      toolCalls: [
                        { id: 'nested-call', name: 'mcp_lookup', args: {} },
                      ],
                      agentId: 'specialist',
                      resolve,
                      reject,
                    };
                    void forwarder.handleCustomEvent?.(
                      GraphEvents.ON_TOOL_EXECUTE,
                      request,
                      'specialist-run'
                    );
                  }
                );
                await forwarder.handleCustomEvent(
                  GraphEvents.ON_RUN_STEP,
                  { id: 'specialist-step', type: 'tool_calls' },
                  'specialist-run'
                );
              }
              await usageHandler?.handleChatModelStart?.(
                {},
                [[]],
                'specialist-model-call',
                undefined,
                undefined,
                undefined,
                {
                  agentId: 'specialist',
                  ls_model_name: 'gpt-4o-mini',
                }
              );
              await usageHandler?.handleLLMEnd?.(
                {
                  generations: [
                    [
                      {
                        text: 'specialist done',
                        message: new AIMessage({
                          content: 'specialist done',
                          usage_metadata: {
                            input_tokens: 5,
                            output_tokens: 3,
                            total_tokens: 8,
                          },
                        }),
                      },
                    ],
                  ],
                },
                'specialist-model-call'
              );
              return { messages: [new AIMessage('specialist done')] };
            }),
          } as unknown as ReturnType<StandardGraph['createWorkflow']>;
        }
        return workflow;
      });

    const rootAgent: t.AgentInputs = {
      agentId: 'root',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Delegate through the router.',
      maxContextTokens: 8000,
      maxSubagentDepth: 2,
      subagentConfigs: [
        {
          type: 'router',
          name: 'Router',
          description: 'Routes work to specialists.',
          allowNested: true,
          agentInputs: {
            agentId: 'router',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'Delegate to the specialist.',
            maxContextTokens: 8000,
            subagentConfigs: [
              {
                type: 'specialist',
                name: 'Specialist',
                description: 'Uses an event-driven tool.',
                agentInputs: {
                  agentId: 'specialist',
                  provider: Providers.OPENAI,
                  clientOptions: {
                    modelName: 'gpt-4o-mini',
                    apiKey: 'test-key',
                  },
                  instructions: 'Use the lookup tool.',
                  maxContextTokens: 8000,
                  toolDefinitions: [{ name: 'mcp_lookup' }],
                },
              },
            ],
          },
        },
      ],
    };

    try {
      const rootRunId = `nested-event-tools-${Date.now()}`;
      const run = await Run.create<t.IState>({
        runId: rootRunId,
        graphConfig: { type: 'standard', agents: [rootAgent] },
        customHandlers: {
          [GraphEvents.ON_TOOL_EXECUTE]: { handle: parentToolHandler },
          [GraphEvents.CHAT_MODEL_END]: { handle: rootModelEndHandler },
          [GraphEvents.ON_SUBAGENT_UPDATE]: {
            handle: parentUpdateHandler,
          },
        },
        returnContent: true,
        skipCleanup: true,
        subagentUsageSink: (event) => {
          nestedUsageEvents.push(event);
        },
      });
      const rootContext = (run.Graph as StandardGraph).agentContexts.get(
        'root'
      );
      const rootSubagentTool = (
        rootContext?.graphTools as t.GenericTool[] | undefined
      )?.find((tool) => 'name' in tool && tool.name === Constants.SUBAGENT);
      expect(rootSubagentTool).toBeDefined();

      await rootSubagentTool!.invoke(
        { description: 'Route this task.', subagent_type: 'router' },
        callerConfig
      );

      expect(specialistToolDefinitions).toEqual([{ name: 'mcp_lookup' }]);
      expect(parentToolHandler).toHaveBeenCalledTimes(1);
      expect(forwardedToolResults).toEqual([
        {
          toolCallId: 'nested-call',
          status: 'success',
          content: 'ran mcp_lookup',
        },
      ]);
      const forwardedSubagentTypes = parentUpdateHandler.mock.calls.map(
        ([, data]) => (data as t.SubagentUpdateEvent).subagentType
      );
      expect(forwardedSubagentTypes).toContain('router');
      expect(forwardedSubagentTypes).toContain('specialist');
      const specialistStepUpdates = parentUpdateHandler.mock.calls.filter(
        ([, data]) => {
          const event = data as t.SubagentUpdateEvent;
          return (
            event.subagentType === 'specialist' &&
            (event.data as { id?: string } | undefined)?.id ===
              'specialist-step'
          );
        }
      );
      expect(specialistStepUpdates).toHaveLength(1);
      const routerEvent = parentUpdateHandler.mock.calls
        .map(([, data]) => data as t.SubagentUpdateEvent)
        .find((event) => event.subagentType === 'router');
      const specialistEvent =
        specialistStepUpdates[0][1] as t.SubagentUpdateEvent;
      expect(routerEvent).toMatchObject({
        runId: rootRunId,
        parentRunId: rootRunId,
        depth: 1,
      });
      expect(routerEvent?.ancestry?.map((entry) => entry.subagentType)).toEqual(
        ['router']
      );
      expect(specialistEvent).toMatchObject({
        runId: rootRunId,
        parentRunId: routerEvent?.subagentRunId,
        depth: 2,
      });
      expect(
        specialistEvent.ancestry?.map((entry) => ({
          type: entry.subagentType,
          kind: entry.subagentKind,
        }))
      ).toEqual([
        { type: 'router', kind: 'agent' },
        { type: 'specialist', kind: 'agent' },
      ]);
      expect(nestedUsageEvents).toHaveLength(1);
      expect(rootModelEndHandler).not.toHaveBeenCalled();
      expect(nestedUsageEvents[0]).toMatchObject({
        runId: rootRunId,
        parentRunId: routerEvent?.subagentRunId,
        subagentType: 'specialist',
        subagentAgentId: 'specialist',
        depth: 2,
        usage: {
          input_tokens: 5,
          output_tokens: 3,
          total_tokens: 8,
        },
      });
      expect(
        nestedUsageEvents[0].ancestry?.map((entry) => ({
          type: entry.subagentType,
          kind: entry.subagentKind,
        }))
      ).toEqual([
        { type: 'router', kind: 'agent' },
        { type: 'specialist', kind: 'agent' },
      ]);
    } finally {
      createWorkflowSpy.mockRestore();
    }
  });

  it('should not create subagent tool when maxSubagentDepth is 0', async () => {
    const agentWithZeroDepth: t.AgentInputs = {
      ...createParentAgent(),
      agentId: 'zero-depth',
      maxSubagentDepth: 0,
    };

    const run = await Run.create<t.IState>({
      runId: `zero-depth-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithZeroDepth],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'zero-depth'
    );
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeUndefined();
  });

  it('should account for subagent tool schema in toolSchemaTokens', async () => {
    /** Simple char-count tokenizer — deterministic, lets us assert presence. */
    const tokenCounter: t.TokenCounter = (message) => {
      const content = message.content;
      if (typeof content === 'string') return content.length;
      if (Array.isArray(content)) return JSON.stringify(content).length;
      return JSON.stringify(content).length;
    };

    const agentWithSubagent = createParentAgent();
    const runWith = await Run.create<t.IState>({
      runId: `with-sub-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSubagent],
      },
      tokenCounter,
      returnContent: true,
      skipCleanup: true,
    });

    const agentWithoutSubagent: t.AgentInputs = {
      agentId: 'plain',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions:
        'You are a supervisor. Delegate research tasks using the subagent tool.',
      maxContextTokens: 8000,
    };
    const runWithout = await Run.create<t.IState>({
      runId: `without-sub-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithoutSubagent],
      },
      tokenCounter,
      returnContent: true,
      skipCleanup: true,
    });

    const contextWith = (runWith.Graph as StandardGraph).agentContexts.get(
      'parent'
    );
    const contextWithout = (
      runWithout.Graph as StandardGraph
    ).agentContexts.get('plain');

    await contextWith?.tokenCalculationPromise;
    await contextWithout?.tokenCalculationPromise;

    /** Subagent tool schema is ~600 chars; expect measurable difference. */
    expect(contextWith!.toolSchemaTokens).toBeGreaterThan(
      contextWithout!.toolSchemaTokens
    );
  });

  it('reports child model usage through subagentUsageSink', async () => {
    const childModelRunIds: string[] = [];
    const CHILD_USAGE = {
      input_tokens: 11,
      output_tokens: 7,
      total_tokens: 18,
    };
    /**
     * The default mock (FakeListChatModel) reports no usage. Re-mock with a
     * subclass that reports `usage_metadata` the way live providers do:
     * stamped on the generation in the invoke path, and carried on a final
     * zero-content chunk in the stream path (the graph's `attemptInvoke`
     * prefers `model.stream()`, and chunk concatenation folds the usage
     * into the aggregated message that `handleLLMEnd` receives).
     */
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeListChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({ responses: [CHILD_RESPONSE] });
          }
          async _generate(
            ...args: Parameters<FakeListChatModel['_generate']>
          ): ReturnType<FakeListChatModel['_generate']> {
            if (args[2]?.runId != null) childModelRunIds.push(args[2].runId);
            const result = await super._generate(...args);
            for (const generation of result.generations) {
              (generation.message as AIMessage).usage_metadata = {
                ...CHILD_USAGE,
              };
            }
            return result;
          }
          async *_streamResponseChunks(
            ...args: Parameters<FakeListChatModel['_streamResponseChunks']>
          ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
            if (args[2]?.runId != null) childModelRunIds.push(args[2].runId);
            yield* super._streamResponseChunks(...args);
            yield new ChatGenerationChunk({
              text: '',
              message: new AIMessageChunk({
                content: '',
                usage_metadata: { ...CHILD_USAGE },
              }),
            });
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const collectedUsage: UsageMetadata[] = [];
    const sunkEvents: t.SubagentUsageEvent[] = [];
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    };

    const runId = `subagent-usage-${Date.now()}`;
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      subagentUsageSink: (event) => {
        sunkEvents.push(event);
      },
    });

    const subagentToolCall: ToolCall = {
      id: 'call_subagent_usage',
      name: Constants.SUBAGENT,
      args: {
        description: 'What is the capital of France?',
        subagent_type: 'researcher',
      },
      type: 'tool_call',
    };

    run.Graph?.overrideTestModel(
      [
        'Let me delegate this research task.',
        `Based on the research: ${CHILD_RESPONSE}`,
      ],
      10,
      [subagentToolCall]
    );

    await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      callerConfig
    );

    /** Child made exactly one model call; all events are child-tagged. */
    expect(sunkEvents).toHaveLength(1);
    const event = sunkEvents[0];
    /** Chunk concat adds empty `*_token_details` — match on the counts. */
    expect(event.usage).toMatchObject(CHILD_USAGE);
    expect(event.subagentType).toBe('researcher');
    expect(event.subagentAgentId).toBe('researcher');
    expect(event.provider).toBe(Providers.OPENAI);
    /** FakeListChatModel emits no ls_model_name → config fallback. */
    expect(event.model).toBe('gpt-4o-mini');
    expect(event.runId).toBe(runId);
    expect(childModelRunIds).toContain(event.modelRunId);
    expect(event.modelRunId).not.toBe(runId);
    expect(event.subagentRunId).toContain(`${runId}_sub_`);
    /**
     * The parent's own calls must NOT be routed through the sink — they
     * flow through the registered CHAT_MODEL_END handler. (The fake
     * override model reports no usage, so collectedUsage stays empty;
     * the load-bearing assertion is that the sink saw no parent calls.)
     */
    expect(sunkEvents.every((e) => e.subagentType === 'researcher')).toBe(true);
  });
});

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { MemorySaver, interrupt } from '@langchain/langgraph';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  HookCallback,
  PermissionDeniedHookOutput,
  PostToolBatchHookOutput,
  PostToolUseHookOutput,
  PreToolUseHookOutput,
  SubagentStartHookInput,
  SubagentStartHookOutput,
  SubagentStopHookInput,
  SubagentStopHookOutput,
} from '@/hooks/types';
import type * as t from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
} from '@/index';
import { InMemorySubagentTaskStore } from '@/tools/subagent';
import { HookRegistry } from '@/hooks/HookRegistry';
import * as providers from '@/llm/providers';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const CHILD_RESPONSE = 'Hook test child response.';
const PRIMITIVE_INTERRUPT_TOOL_NAME = 'confirm_child';

const calculatorDef: t.LCTool = {
  name: 'calculator',
  description: 'Evaluate a math expression.',
  parameters: {
    type: 'object',
    properties: {
      expression: { type: 'string' },
    },
    required: ['expression'],
  },
};

const referenceToolDefs: t.LCTool[] = [
  {
    name: 'produce_value',
    description: 'Produce a value for a later tool.',
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'consume_value',
    description: 'Consume a value from an earlier tool.',
    parameters: {
      type: 'object',
      properties: { value: { type: 'string' } },
      required: ['value'],
    },
  },
];

const callerConfig = {
  configurable: { thread_id: 'hook-test-thread' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

const originalGetChatModelClass = providers.getChatModelClass;

function makeSubagentToolCall(
  id = `call_sub_${Date.now()}`,
  description = 'Test task for hook verification'
): ToolCall {
  return {
    name: Constants.SUBAGENT,
    args: {
      description,
      subagent_type: 'researcher',
    },
    id,
    type: 'tool_call',
  };
}

function createParentAgent(): t.AgentInputs {
  return {
    agentId: 'hook-parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: 'Delegate research tasks to subagents.',
    maxContextTokens: 8000,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher-child',
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Answer concisely.',
          maxContextTokens: 8000,
        },
      },
    ],
  };
}

function createParentAgentWithChildTool(): t.AgentInputs {
  return {
    agentId: 'hook-parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: 'Delegate research tasks to subagents.',
    maxContextTokens: 8000,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher-child',
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Use calculator for arithmetic, then answer concisely.',
          maxContextTokens: 8000,
          toolDefinitions: [calculatorDef],
        },
      },
    ],
  };
}

function createParentAgentWithPrimitiveInterruptTool(
  primitiveInterruptTool: t.GenericTool
): t.AgentInputs {
  const parent = createParentAgent();
  const child = parent.subagentConfigs?.[0];
  const childAgent = child?.agentInputs;
  if (child == null || childAgent == null) {
    throw new Error('Expected a child agent configuration.');
  }
  return {
    ...parent,
    subagentConfigs: [
      {
        ...child,
        agentInputs: {
          ...childAgent,
          instructions: 'Request confirmation, then answer concisely.',
          graphTools: [primitiveInterruptTool],
        },
      },
    ],
  };
}

function createParentAgentWithNestedChildTool(): t.AgentInputs {
  return {
    agentId: 'nested-hook-parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'parent-model', apiKey: 'test-key' },
    instructions: 'Delegate nested calculations.',
    maxContextTokens: 8000,
    maxSubagentDepth: 2,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Delegates calculations',
        allowNested: true,
        agentInputs: {
          agentId: 'nested-researcher-child',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'nested-child-model',
            apiKey: 'test-key',
          },
          instructions: 'Delegate arithmetic to the calculator worker.',
          maxContextTokens: 8000,
          subagentConfigs: [
            {
              type: 'calculator-worker',
              name: 'Calculator Worker',
              description: 'Runs calculator tools',
              agentInputs: {
                agentId: 'calculator-grandchild',
                provider: Providers.OPENAI,
                clientOptions: {
                  modelName: 'grandchild-model',
                  apiKey: 'test-key',
                },
                instructions: 'Use calculator, then answer.',
                maxContextTokens: 8000,
                toolDefinitions: [calculatorDef],
              },
            },
          ],
        },
      },
    ],
  };
}

function createParentAgentWithReferenceTools(): t.AgentInputs {
  const parent = createParentAgentWithChildTool();
  const child = parent.subagentConfigs?.[0];
  const childAgent = child?.agentInputs;
  if (child == null || childAgent == null) {
    throw new Error('Expected a child agent configuration.');
  }
  return {
    ...parent,
    subagentConfigs: [
      {
        ...child,
        agentInputs: {
          ...childAgent,
          toolDefinitions: referenceToolDefs,
        },
      },
    ],
  };
}

function createCalculatorToolCall(): ToolCall {
  return {
    name: 'calculator',
    args: { expression: '21 * 2' },
    id: 'call_child_calculator',
    type: 'tool_call',
  };
}

class HitlChildFakeChatModel extends FakeChatModel {
  constructor(_options: object) {
    super({ responses: [CHILD_RESPONSE], sleep: 1 });
  }

  _streamResponseChunks(
    messages: Parameters<FakeChatModel['_streamResponseChunks']>[0],
    options: Parameters<FakeChatModel['_streamResponseChunks']>[1],
    runManager?: Parameters<FakeChatModel['_streamResponseChunks']>[2]
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const hasToolResult = messages.some(
      (message) => message._getType() === 'tool'
    );
    return new FakeChatModel({
      responses: [hasToolResult ? CHILD_RESPONSE : 'Using calculator.'],
      sleep: 1,
      toolCalls: hasToolResult ? [] : [createCalculatorToolCall()],
    })._streamResponseChunks(messages, options, runManager);
  }

  bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
    const config = {
      tools,
    } as Parameters<FakeChatModel['withConfig']>[0];
    return this.withConfig(config);
  }
}

class PrimitiveInterruptFakeChatModel extends FakeChatModel {
  constructor(_options: object) {
    super({ responses: [CHILD_RESPONSE], sleep: 1 });
  }

  _streamResponseChunks(
    messages: Parameters<FakeChatModel['_streamResponseChunks']>[0],
    options: Parameters<FakeChatModel['_streamResponseChunks']>[1],
    runManager?: Parameters<FakeChatModel['_streamResponseChunks']>[2]
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const hasToolResult = messages.some(
      (message) => message._getType() === 'tool'
    );
    return new FakeChatModel({
      responses: [hasToolResult ? CHILD_RESPONSE : 'Requesting confirmation.'],
      sleep: 1,
      toolCalls: hasToolResult
        ? []
        : [
          {
            name: PRIMITIVE_INTERRUPT_TOOL_NAME,
            args: {},
            id: 'call_child_primitive_interrupt',
            type: 'tool_call',
          },
        ],
    })._streamResponseChunks(messages, options, runManager);
  }

  bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
    return this.withConfig({ tools } as Parameters<
      FakeChatModel['withConfig']
    >[0]);
  }
}

class NestedHitlFakeChatModel extends FakeChatModel {
  private readonly nestedChild: boolean;

  constructor(options: { modelName?: string }) {
    super({ responses: [CHILD_RESPONSE], sleep: 1 });
    this.nestedChild = options.modelName === 'nested-child-model';
  }

  _streamResponseChunks(
    messages: Parameters<FakeChatModel['_streamResponseChunks']>[0],
    options: Parameters<FakeChatModel['_streamResponseChunks']>[1],
    runManager?: Parameters<FakeChatModel['_streamResponseChunks']>[2]
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const hasToolResult = messages.some(
      (message) => message._getType() === 'tool'
    );
    let toolCalls: ToolCall[] = [];
    if (!hasToolResult && this.nestedChild) {
      toolCalls = [
        {
          name: Constants.SUBAGENT,
          args: {
            description: 'Calculate the result',
            subagent_type: 'calculator-worker',
          },
          id: 'call_nested_calculator_worker',
          type: 'tool_call',
        },
      ];
    } else if (!hasToolResult) {
      toolCalls = [createCalculatorToolCall()];
    }
    return new FakeChatModel({
      responses: [hasToolResult ? CHILD_RESPONSE : 'Delegating calculation.'],
      sleep: 1,
      toolCalls,
    })._streamResponseChunks(messages, options, runManager);
  }

  bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
    return this.withConfig({ tools } as Parameters<
      FakeChatModel['withConfig']
    >[0]);
  }
}

class ReferenceHitlFakeChatModel extends FakeChatModel {
  constructor(_options: object) {
    super({ responses: [CHILD_RESPONSE], sleep: 1 });
  }

  _streamResponseChunks(
    messages: Parameters<FakeChatModel['_streamResponseChunks']>[0],
    options: Parameters<FakeChatModel['_streamResponseChunks']>[1],
    runManager?: Parameters<FakeChatModel['_streamResponseChunks']>[2]
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const toolResultCount = messages.filter(
      (message) => message._getType() === 'tool'
    ).length;
    let toolCalls: ToolCall[] = [];
    if (toolResultCount === 0) {
      toolCalls = [
        {
          name: 'produce_value',
          args: {},
          id: 'call_produce_value',
          type: 'tool_call',
        },
      ];
    } else if (toolResultCount === 1) {
      toolCalls = [
        {
          name: 'consume_value',
          args: { value: '{{tool0turn0}}' },
          id: 'call_consume_value',
          type: 'tool_call',
        },
      ];
    }
    return new FakeChatModel({
      responses: [toolCalls.length > 0 ? 'Using a tool.' : CHILD_RESPONSE],
      sleep: 1,
      toolCalls,
    })._streamResponseChunks(messages, options, runManager);
  }

  bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
    return this.withConfig({ tools } as Parameters<
      FakeChatModel['withConfig']
    >[0]);
  }
}

async function createSubagentRun(
  hooks: HookRegistry,
  runId = `subagent-hook-${Date.now()}`
): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [createParentAgent()],
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    },
    hooks,
  });
}

describe('Subagent hook integration (end-to-end via Run)', () => {
  jest.setTimeout(15000);

  let getChatModelClassSpy: jest.SpyInstance;

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

  it('SubagentStart fires with correct payload through real Run pipeline', async () => {
    const registry = new HookRegistry();
    let captured: SubagentStartHookInput | undefined;

    const hook: HookCallback<'SubagentStart'> = async (
      input
    ): Promise<SubagentStartHookOutput> => {
      captured = input;
      return {};
    };
    registry.register('SubagentStart', { hooks: [hook] });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    expect(captured).toBeDefined();
    expect(captured!.hook_event_name).toBe('SubagentStart');
    expect(captured!.agentType).toBe('researcher');
    expect(captured!.parentAgentId).toBe('hook-parent');
    expect(captured!.threadId).toBe('hook-test-thread');
    expect(captured!.inputs).toHaveLength(1);
    expect(captured!.inputs[0].content).toContain(
      'Test task for hook verification'
    );
  });

  it('SubagentStop fires with messages from child execution', async () => {
    const registry = new HookRegistry();
    let captured: SubagentStopHookInput | undefined;

    const hook: HookCallback<'SubagentStop'> = async (
      input
    ): Promise<SubagentStopHookOutput> => {
      captured = input;
      return {};
    };
    registry.register('SubagentStop', { hooks: [hook] });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    expect(captured).toBeDefined();
    expect(captured!.hook_event_name).toBe('SubagentStop');
    expect(captured!.agentType).toBe('researcher');
    expect(captured!.threadId).toBe('hook-test-thread');
    expect(captured!.messages.length).toBeGreaterThan(0);
  });

  it('propagates a caller-only Run signal into lazy subagent resolution', async () => {
    const callerAbort = new AbortController();
    let resolverSignal: AbortSignal | undefined;
    let markResolverStarted = (): void => undefined;
    const resolverStarted = new Promise<void>((resolve) => {
      markResolverStarted = resolve;
    });
    let releaseResolver = (): void => undefined;
    const resolverRelease = new Promise<void>((resolve) => {
      releaseResolver = resolve;
    });
    const parent = createParentAgent();
    const eagerChild = parent.subagentConfigs?.[0]?.agentInputs;
    if (eagerChild == null) {
      throw new Error('Expected eager child inputs.');
    }
    parent.subagentConfigs = [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Researches topics',
        configId: 'caller-signal-researcher@v1',
        resolveAgentInputs: async (context) => {
          resolverSignal = context.signal;
          markResolverStarted();
          await resolverRelease;
          return eagerChild;
        },
      },
    ];
    const run = await Run.create<t.IState>({
      runId: `subagent-caller-signal-${Date.now()}`,
      graphConfig: { type: 'standard', agents: [parent] },
      returnContent: true,
      skipCleanup: true,
    });
    run.Graph!.overrideTestModel(['Delegating...', 'Caller cancelled.'], 5, [
      makeSubagentToolCall('call_caller_signal'),
    ]);

    const processing = run.processStream(
      { messages: [new HumanMessage('cancel this delegation')] },
      { ...callerConfig, signal: callerAbort.signal }
    );
    await resolverStarted;
    callerAbort.abort(new Error('caller cancelled'));
    await Promise.resolve();
    const resolverWasAborted = resolverSignal?.aborted === true;
    releaseResolver();
    await processing.catch(() => undefined);

    expect(resolverWasAborted).toBe(true);
  });

  it('SubagentStart deny blocks subagent execution and returns blocked message', async () => {
    const registry = new HookRegistry();
    const denyHook: HookCallback<
      'SubagentStart'
    > = async (): Promise<SubagentStartHookOutput> => ({
      decision: 'deny',
      reason: 'policy violation',
    });
    registry.register('SubagentStart', {
      pattern: '^researcher$',
      hooks: [denyHook],
    });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(
      ['Delegating...', 'The subagent was blocked.'],
      5,
      [tc]
    );

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();

    const toolMessages = runMessages!.filter(
      (msg) =>
        msg._getType() === 'tool' &&
        'name' in msg &&
        msg.name === Constants.SUBAGENT
    );
    expect(toolMessages.length).toBe(1);
    expect(String(toolMessages[0].content)).toContain(
      'Blocked: policy violation'
    );
  });

  it('PreToolUse and PostToolUse fire for event-driven tools inside subagents', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({
              responses: ['Using calculator.', CHILD_RESPONSE],
              sleep: 1,
              toolCalls: [createCalculatorToolCall()],
            });
          }
          bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
            const config = {
              tools,
            } as Parameters<FakeChatModel['withConfig']>[0];
            return this.withConfig(config);
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const preToolEvents: string[] = [];
    const postToolEvents: string[] = [];
    // `executingAgentId` is always the owning agent (incl. top-level), unlike
    // `agentId` which stays undefined outside a subagent scope.
    const preExecEvents: string[] = [];
    const postExecEvents: string[] = [];

    const preHook: HookCallback<'PreToolUse'> = async (
      input
    ): Promise<PreToolUseHookOutput> => {
      preToolEvents.push(`${input.agentId ?? '-'}:${input.toolName}`);
      preExecEvents.push(`${input.executingAgentId ?? '-'}:${input.toolName}`);
      return { decision: 'allow' };
    };
    registry.register('PreToolUse', { hooks: [preHook] });

    const postHook: HookCallback<'PostToolUse'> = async (
      input
    ): Promise<PostToolUseHookOutput> => {
      postToolEvents.push(`${input.agentId ?? '-'}:${input.toolName}`);
      postExecEvents.push(`${input.executingAgentId ?? '-'}:${input.toolName}`);
      return {};
    };
    registry.register('PostToolUse', { hooks: [postHook] });

    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          const results: t.ToolExecuteResult[] = request.toolCalls.map(
            (call) => ({
              toolCallId: call.id,
              status: 'success',
              content: '42',
            })
          );
          request.resolve(results);
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: `subagent-tool-hook-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgentWithChildTool()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      hooks: registry,
    });

    const tc = makeSubagentToolCall();
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('calculate something')] },
      callerConfig
    );

    expect(preToolEvents).toContain('-:subagent');
    expect(preToolEvents).toContain('researcher-child:calculator');
    expect(postToolEvents).toContain('-:subagent');
    expect(postToolEvents).toContain('researcher-child:calculator');

    // `agentId` stays the subagent-scope marker (undefined at the top level),
    // but `executingAgentId` attributes every batch to its owning agent — incl.
    // the top-level parent, which a multi-agent host needs and `agentId` can't give.
    expect(preExecEvents).toContain('hook-parent:subagent');
    expect(preExecEvents).toContain('researcher-child:calculator');
    expect(postExecEvents).toContain('hook-parent:subagent');
    expect(postExecEvents).toContain('researcher-child:calculator');
  });

  it('top-level event-driven dispatches leave agentId unset (subagent-scope marker)', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({
              responses: ['Calculating.', 'All done.'],
              sleep: 1,
              toolCalls: [createCalculatorToolCall()],
            });
          }
          bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
            const config = {
              tools,
            } as Parameters<FakeChatModel['withConfig']>[0];
            return this.withConfig(config);
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const scopeEvents: string[] = [];

    const preHook: HookCallback<'PreToolUse'> = async (
      input
    ): Promise<PreToolUseHookOutput> => {
      scopeEvents.push(
        `pre:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return { decision: 'allow' };
    };
    registry.register('PreToolUse', { hooks: [preHook] });

    const postHook: HookCallback<'PostToolUse'> = async (
      input
    ): Promise<PostToolUseHookOutput> => {
      scopeEvents.push(
        `post:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return {};
    };
    registry.register('PostToolUse', { hooks: [postHook] });

    const batchHook: HookCallback<'PostToolBatch'> = async (
      input
    ): Promise<PostToolBatchHookOutput> => {
      scopeEvents.push(
        `batch:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return {};
    };
    registry.register('PostToolBatch', { hooks: [batchHook] });

    const dispatchAgentIds: Array<string | undefined> = [];
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          dispatchAgentIds.push(request.agentId);
          const results: t.ToolExecuteResult[] = request.toolCalls.map(
            (call) => ({
              toolCallId: call.id,
              status: 'success',
              content: '42',
            })
          );
          request.resolve(results);
        },
      },
    };

    /**
     * The LibreChat shape: a TOP-LEVEL agent whose tools ride
     * `toolDefinitions` (event-driven ToolNode, host executes via
     * ON_TOOL_EXECUTE). Hook inputs here must NOT carry the subagent-scope
     * marker — a host hook keying on `agentId != null` (e.g. a steering
     * drain that must never inject into child state) would otherwise skip
     * every top-level batch. The DISPATCH payload is the opposite: hosts
     * key tool/credential lookup on `request.agentId`, so it must keep
     * identifying the owning agent.
     */
    const run = await Run.create<t.IState>({
      runId: `toplevel-event-hook-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'hook-parent',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'Use the calculator.',
            maxContextTokens: 8000,
            toolDefinitions: [calculatorDef],
          },
        ],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      hooks: registry,
    });

    run.Graph!.overrideTestModel(['Calculating.', 'All done.'], 5, [
      createCalculatorToolCall(),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('what is 21 * 2?')] },
      callerConfig
    );

    expect(scopeEvents).toContain('pre:-:hook-parent');
    expect(scopeEvents).toContain('post:-:hook-parent');
    expect(scopeEvents).toContain('batch:-:hook-parent');
    expect(dispatchAgentIds).toEqual(['hook-parent']);
  });

  it('runs a background child with HITL enabled and denies a tool requiring approval', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return HitlChildFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const hooks = new HookRegistry();
    const attempted: string[] = [];
    const executed: string[] = [];
    hooks.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (input.toolName === 'calculator') {
            attempted.push(input.toolName);
            return { decision: 'ask', reason: 'BYOM requires approval' };
          }
          return { decision: 'allow' };
        },
      ],
    });
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          executed.push(...request.toolCalls.map((call) => call.name));
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: '42',
            }))
          );
        },
      },
    };
    const store = new InMemorySubagentTaskStore();
    const scopeId = 'owner:background-hitl';
    const run = await Run.create<t.IState>({
      runId: `subagent-background-hitl-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgentWithChildTool()],
        compileOptions: { checkpointer: new MemorySaver() },
      },
      subagentTasks: { store, scopeId },
      humanInTheLoop: { enabled: true },
      customHandlers,
      hooks,
      returnContent: true,
      skipCleanup: true,
    });
    const call = makeSubagentToolCall('call_background_hitl');
    call.args = { ...call.args, run_in_background: true };
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [call]);

    await run.processStream(
      { messages: [new HumanMessage('calculate in the background')] },
      callerConfig
    );
    expect(run.getInterrupt()).toBeUndefined();
    let tasks = store.list(scopeId);
    for (let attempt = 0; attempt < 200 && tasks[0]?.status === 'running'; attempt++) {
      await new Promise<void>((resolve) => setTimeout(resolve, 20));
      tasks = store.list(scopeId);
    }
    expect(tasks).toHaveLength(1);
    expect(tasks[0].status).toBe('completed');
    expect(attempted).toEqual(['calculator']);
    expect(executed).not.toContain('calculator');
    expect(store.claim(scopeId, tasks[0].taskId)).toMatchObject({
      status: 'completed',
      result: expect.stringContaining('Background approval denied for: calculator'),
    });
  });

  it.each([
    {
      label: 'approve',
      resumeDecision: { type: 'approve' } as const,
      shouldExecute: true,
      deniedReason: undefined,
    },
    {
      label: 'reject',
      resumeDecision: {
        type: 'reject',
        reason: 'host rejected child tool',
      } as const,
      shouldExecute: false,
      deniedReason: 'host rejected child tool',
    },
    {
      label: 'deny',
      resumeDecision: undefined,
      shouldExecute: false,
      deniedReason: 'policy denied child tool',
    },
  ])(
    'handles a child subagent tool $label through the parent Run',
    async ({ resumeDecision, shouldExecute, deniedReason }) => {
      getChatModelClassSpy.mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return HitlChildFakeChatModel;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

      const registry = new HookRegistry();
      const deniedTools: string[] = [];
      const executedTools: string[] = [];
      let calculatorPreToolCalls = 0;
      let calculatorPostToolCalls = 0;

      const preHook: HookCallback<'PreToolUse'> = async (
        input
      ): Promise<PreToolUseHookOutput> => {
        if (input.toolName === 'calculator') {
          calculatorPreToolCalls += 1;
          if (resumeDecision == null) {
            return {
              decision: 'deny',
              reason: 'policy denied child tool',
            };
          }
          return { decision: 'ask', reason: 'review calculator' };
        }
        return { decision: 'allow' };
      };
      registry.register('PreToolUse', { hooks: [preHook] });
      registry.register('PostToolUse', {
        hooks: [
          async (input): Promise<PostToolUseHookOutput> => {
            if (input.toolName === 'calculator') {
              calculatorPostToolCalls += 1;
            }
            return {};
          },
        ],
      });

      const deniedHook: HookCallback<'PermissionDenied'> = async (
        input
      ): Promise<PermissionDeniedHookOutput> => {
        deniedTools.push(
          `${input.agentId ?? '-'}:${input.toolName}:${input.reason}`
        );
        return {};
      };
      registry.register('PermissionDenied', { hooks: [deniedHook] });

      const customHandlers: Record<string, t.EventHandler> = {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: (_event, rawData): void => {
            const request = rawData as t.ToolExecuteBatchRequest;
            executedTools.push(...request.toolCalls.map((call) => call.name));
            const results: t.ToolExecuteResult[] = request.toolCalls.map(
              (call) => ({
                toolCallId: call.id,
                status: 'success',
                content: '42',
              })
            );
            request.resolve(results);
          },
        },
      };

      const run = await Run.create<t.IState>({
        runId: `subagent-tool-ask-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });

      const tc = makeSubagentToolCall();
      run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

      await run.processStream(
        { messages: [new HumanMessage('calculate something')] },
        callerConfig
      );

      if (resumeDecision == null) {
        expect(run.getInterrupt()).toBeUndefined();
        expect(calculatorPreToolCalls).toBe(1);
        expect(executedTools).not.toContain('calculator');
        expect(deniedTools).toEqual([
          `researcher-child:calculator:${deniedReason}`,
        ]);
        return;
      }

      const pending = run.getInterrupt();
      expect(pending?.payload).toMatchObject({
        type: 'tool_approval',
        subagent: {
          agent_id: 'researcher-child',
          parent_tool_call_id: tc.id,
          subagent_type: 'researcher',
        },
      });
      expect(executedTools).not.toContain('calculator');

      await run.resume([resumeDecision], callerConfig);

      expect(run.getInterrupt()).toBeUndefined();
      expect(calculatorPreToolCalls).toBe(2);
      expect(
        executedTools.filter((name) => name === 'calculator')
      ).toHaveLength(shouldExecute ? 1 : 0);
      expect(calculatorPostToolCalls).toBe(shouldExecute ? 1 : 0);
      expect(deniedTools).toEqual(
        shouldExecute ? [] : [`researcher-child:calculator:${deniedReason}`]
      );
    }
  );

  it('resumes approvals across multiple subagents and keeps updates sanitized', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return HitlChildFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const deniedToolIds: string[] = [];
    const executedToolIds: string[] = [];
    const updates: t.SubagentUpdateEvent[] = [];
    const completedSubagentCalls: string[] = [];

    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> =>
          input.toolName === 'calculator'
            ? { decision: 'ask', reason: 'review child calculator' }
            : { decision: 'allow' },
      ],
    });
    registry.register('PostToolUse', {
      hooks: [
        async (input): Promise<PostToolUseHookOutput> => {
          if (input.toolName === Constants.SUBAGENT) {
            completedSubagentCalls.push(input.toolUseId);
            return { additionalContext: `context:${input.toolUseId}` };
          }
          return {};
        },
      ],
    });
    registry.register('PermissionDenied', {
      hooks: [
        async (input): Promise<PermissionDeniedHookOutput> => {
          deniedToolIds.push(input.toolUseId);
          return {};
        },
      ],
    });

    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          executedToolIds.push(...request.toolCalls.map((call) => call.id));
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: '42',
            }))
          );
        },
      },
      [GraphEvents.ON_SUBAGENT_UPDATE]: {
        handle: (_event, data): void => {
          updates.push(data as t.SubagentUpdateEvent);
        },
      },
    };

    const checkpointer = new MemorySaver();
    const baseRunId = `subagent-multi-hitl-${Date.now()}`;
    const createRun = (runId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
          compileOptions: { checkpointer, interruptBefore: [] },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
    const run = await createRun(`${baseRunId}-initial`);
    const first = makeSubagentToolCall(
      'call_sub_first',
      'Run the first calculation'
    );
    const second = makeSubagentToolCall(
      'call_sub_second',
      'Run the second calculation'
    );
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      first,
      second,
    ]);
    const multiCallerConfig = {
      ...callerConfig,
      configurable: {
        thread_id: 'multi-subagent-hitl-thread',
        access_token: 'must-not-leak',
        requestBody: { currentTaskInput: 'must-not-leak-task' },
        userMCPAuthMap: { private: { token: 'must-not-leak-mcp' } },
      },
    };

    await run.processStream(
      { messages: [new HumanMessage('calculate twice')] },
      multiCallerConfig
    );
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: first.id },
    });
    expect(updates.some((event) => event.parentToolCallId === second.id)).toBe(
      true
    );

    await run.resume([{ type: 'approve' }], multiCallerConfig);
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: second.id },
    });

    const rebuiltRun = await createRun(`${baseRunId}-rebuilt`);
    rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
    await rebuiltRun.resume(
      [{ type: 'reject', reason: 'reject second child' }],
      multiCallerConfig
    );

    expect(rebuiltRun.getInterrupt()).toBeUndefined();
    expect(updates.filter((event) => event.phase === 'error')).toEqual([]);
    expect(executedToolIds).toHaveLength(1);
    expect(deniedToolIds).toHaveLength(1);
    expect(completedSubagentCalls).toEqual([first.id, second.id]);
    const firstPhases = updates
      .filter((event) => event.parentToolCallId === first.id)
      .map((event) => event.phase);
    const secondPhases = updates
      .filter((event) => event.parentToolCallId === second.id)
      .map((event) => event.phase);
    expect(firstPhases[0]).toBe('start');
    expect(firstPhases[firstPhases.length - 1]).toBe('stop');
    expect(secondPhases[0]).toBe('start');
    expect(secondPhases[secondPhases.length - 1]).toBe('stop');
    const serializedUpdates = JSON.stringify(updates);
    expect(serializedUpdates).not.toContain('must-not-leak');
    expect(serializedUpdates).not.toContain('currentTaskInput');
    expect(serializedUpdates).not.toContain('userMCPAuthMap');
    expect(serializedUpdates).not.toContain('checkpoint_');
  });

  it('resumes a primitive child interrupt through a rebuilt Run', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return PrimitiveInterruptFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const checkpointer = new MemorySaver();
    const resumedValues: string[] = [];
    const primitiveInterruptTool = tool(
      async () => {
        const resumeValue = interrupt<string, string>('confirm child');
        resumedValues.push(resumeValue);
        return `confirmed:${resumeValue}`;
      },
      {
        name: PRIMITIVE_INTERRUPT_TOOL_NAME,
        description: 'Pause for primitive confirmation.',
        schema: z.object({}),
      }
    );
    const resolutionContexts: t.SubagentResolveContext[] = [];
    const preparedContexts: t.SubagentContextInput[] = [];
    let completedMessages: t.SubagentContextResult['messages'] = [];
    const createLazyParent = (): t.AgentInputs => {
      const parent = createParentAgentWithPrimitiveInterruptTool(
        primitiveInterruptTool
      );
      const child = parent.subagentConfigs?.[0];
      const childInputs = child?.agentInputs;
      if (child == null || childInputs == null) {
        throw new Error('Expected a child agent configuration.');
      }
      return {
        ...parent,
        subagentConfigs: [
          {
            ...child,
            agentInputs: undefined,
            configId: 'primitive-interrupt-child@v1',
            resolveAgentInputs: async (context) => {
              resolutionContexts.push(context);
              return childInputs;
            },
          },
        ],
      };
    };
    const createRun = (currentRunId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId: currentRunId,
        graphConfig: {
          type: 'standard',
          agents: [createLazyParent()],
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: {
          [GraphEvents.TOOL_END]: new ToolEndHandler(),
          [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        },
        humanInTheLoop: { enabled: true },
        subagentContext: {
          prepare: async (input) => {
            preparedContexts.push(input);
            return {
              messages: [
                new HumanMessage({
                  id: 'shared-file-context',
                  content: 'Shared file input',
                }),
              ],
            };
          },
          complete: async (_input, result) => {
            completedMessages = result.messages;
            return { content: result.content };
          },
        },
      });
    const runId = `primitive-subagent-rebuild-${Date.now()}`;
    const parentCall = makeSubagentToolCall('call_primitive_rebuild');
    const initialRun = await createRun(`${runId}-initial`);
    initialRun.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      parentCall,
    ]);

    await initialRun.processStream(
      { messages: [new HumanMessage('confirm inside the child')] },
      callerConfig
    );
    const paused = initialRun.getInterrupt<string>();
    expect(paused?.payload).toBe('confirm child');
    expect(JSON.stringify(paused)).not.toContain(
      '__librechat_subagent_resume_manifest'
    );
    if (paused?.checkpointId == null) {
      throw new Error('Expected a persisted parent checkpoint.');
    }

    const rebuiltRun = await createRun(`${runId}-rebuilt`);
    rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
    const injected = new HumanMessage('host edit on resume');
    const rebuiltContent = await rebuiltRun.resume<string>(
      'approved after restart',
      {
        ...callerConfig,
        configurable: {
          ...callerConfig.configurable,
          checkpoint_id: paused.checkpointId,
          checkpoint_ns: paused.checkpointNs ?? '',
        },
      },
      undefined,
      { update: { messages: [injected] } }
    );

    expect(rebuiltRun.getInterrupt()).toBeUndefined();
    expect(resolutionContexts).toHaveLength(2);
    expect(preparedContexts).toHaveLength(2);
    expect(preparedContexts[1].resumed).toBe(true);
    expect(
      preparedContexts[0].executionContext.ancestry.at(-1)?.subagentRunId
    ).toBe(preparedContexts[1].executionContext.ancestry.at(-1)?.subagentRunId);
    expect(
      completedMessages.filter(
        (message) => message.id === 'shared-file-context'
      )
    ).toHaveLength(1);
    expect(resolutionContexts[0].executionId).toBe(
      resolutionContexts[1].executionId
    );
    expect(resolutionContexts[0].descriptor.configId).toBe(
      'primitive-interrupt-child@v1'
    );
    expect(resolutionContexts[1].descriptor.configId).toBe(
      'primitive-interrupt-child@v1'
    );
    expect(resumedValues).toEqual(['approved after restart']);
    expect(JSON.stringify(rebuiltContent)).toContain('Final answer.');
    expect(JSON.stringify(rebuiltContent)).not.toContain('host edit on resume');
    expect(
      rebuiltRun
        .getRunMessages()
        ?.some(
          (message) =>
            message._getType() === 'ai' &&
            JSON.stringify(message.content).includes('Final answer.')
        )
    ).toBe(true);
    expect(
      rebuiltRun.getRunMessages()?.some((message) => message.id === injected.id)
    ).toBe(false);
  });

  it('forks every resume from the paused child snapshot', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return HitlChildFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const checkpointer = new MemorySaver();
    const registry = new HookRegistry();
    const executedTools: string[] = [];
    const updates: t.SubagentUpdateEvent[] = [];
    let approvalHookCalls = 0;
    const runId = `subagent-rebuild-hitl-${Date.now()}`;
    registry.registerSession(`${runId}-initial`, 'PreToolUse', {
      once: true,
      pattern: '^calculator$',
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          approvalHookCalls += 1;
          return { decision: 'ask', reason: 'review calculator' };
        },
      ],
    });
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          executedTools.push(...request.toolCalls.map((call) => call.name));
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: '42',
            }))
          );
        },
      },
      [GraphEvents.ON_SUBAGENT_UPDATE]: {
        handle: (_event, data): void => {
          updates.push(data as t.SubagentUpdateEvent);
        },
      },
    };
    const createRun = (currentRunId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId: currentRunId,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
    const tc = makeSubagentToolCall('call_sub_rebuild');
    const initialRun = await createRun(`${runId}-initial`);
    initialRun.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      tc,
    ]);

    await initialRun.processStream(
      { messages: [new HumanMessage('calculate after restart')] },
      callerConfig
    );
    const persistedInterrupt = initialRun.getInterrupt();
    expect(persistedInterrupt?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: tc.id },
    });
    expect(JSON.stringify(persistedInterrupt)).not.toContain(
      '__librechat_subagent_resume_manifest'
    );
    expect(approvalHookCalls).toBe(1);
    const sourceChildRunId =
      persistedInterrupt?.payload.type === 'tool_approval'
        ? persistedInterrupt.payload.subagent?.run_id
        : undefined;
    expect(sourceChildRunId).toBeDefined();
    const branchConfig = {
      ...callerConfig,
      configurable: {
        ...callerConfig.configurable,
        checkpoint_id: persistedInterrupt?.checkpointId,
        checkpoint_ns: persistedInterrupt?.checkpointNs ?? '',
      },
    };
    const earlyApprovedRun = await createRun(`${runId}-early-approved`);
    const sharedRebuiltRunId = `${runId}-same-rebuilt-id`;
    const approvedRun = await createRun(sharedRebuiltRunId);
    const rejectedRun = await createRun(sharedRebuiltRunId);
    let armRestoredCheckpointBarrier = false;
    let blockNextRestoredCheckpointRead = false;
    const originalRestorePendingToolApprovals =
      registry.restorePendingToolApprovals.bind(registry);
    const restoreApprovalSpy = jest
      .spyOn(registry, 'restorePendingToolApprovals')
      .mockImplementation((sessionId, targetExecutionScope, snapshots) => {
        originalRestorePendingToolApprovals(
          sessionId,
          targetExecutionScope,
          snapshots
        );
        if (armRestoredCheckpointBarrier) {
          armRestoredCheckpointBarrier = false;
          blockNextRestoredCheckpointRead = true;
        }
      });
    let markCheckpointReadBlocked = (): void => undefined;
    const checkpointReadBlocked = new Promise<void>((resolve) => {
      markCheckpointReadBlocked = resolve;
    });
    let releaseBlockedCheckpointRead = (): void => undefined;
    const blockedCheckpointReadRelease = new Promise<void>((resolve) => {
      releaseBlockedCheckpointRead = resolve;
    });
    const originalGetTuple = checkpointer.getTuple.bind(checkpointer);
    const checkpointReadSpy = jest
      .spyOn(checkpointer, 'getTuple')
      .mockImplementation(async (config) => {
        const tuple = await originalGetTuple(config);
        const configurable = config.configurable;
        const threadId = configurable?.thread_id;
        if (
          blockNextRestoredCheckpointRead &&
          typeof threadId === 'string' &&
          threadId.startsWith('subagent:') &&
          configurable?.checkpoint_id == null
        ) {
          blockNextRestoredCheckpointRead = false;
          markCheckpointReadBlocked();
          await blockedCheckpointReadRelease;
        }
        return tuple;
      });
    earlyApprovedRun.Graph!.overrideTestModel(['Final answer.'], 1);
    approvedRun.Graph!.overrideTestModel(['Final answer.'], 1);
    rejectedRun.Graph!.overrideTestModel(['Final answer.'], 1);
    const warningSpy = jest
      .spyOn(console, 'warn')
      .mockImplementation((): void => undefined);
    let rejectedResume: Promise<t.MessageContentComplex[] | undefined> | null =
      null;
    try {
      await earlyApprovedRun.resume([{ type: 'approve' }], branchConfig);
      await initialRun.resume([{ type: 'approve' }], callerConfig);
      armRestoredCheckpointBarrier = true;
      rejectedResume = rejectedRun.resume(
        [{ type: 'reject', reason: 'deny after restart' }],
        branchConfig
      );
      await checkpointReadBlocked;
      await approvedRun.resume([{ type: 'approve' }], branchConfig);
      releaseBlockedCheckpointRead();
      await rejectedResume;
      expect(warningSpy).not.toHaveBeenCalledWith(
        expect.stringContaining('toolCallStepIds missing entry')
      );
    } finally {
      releaseBlockedCheckpointRead();
      await rejectedResume?.catch(() => undefined);
      checkpointReadSpy.mockRestore();
      warningSpy.mockRestore();
    }

    expect(earlyApprovedRun.getInterrupt()).toBeUndefined();
    expect(initialRun.getInterrupt()).toBeUndefined();
    expect(approvedRun.getInterrupt()).toBeUndefined();
    expect(rejectedRun.getInterrupt()).toBeUndefined();
    expect(executedTools).toEqual(['calculator', 'calculator', 'calculator']);
    expect(approvalHookCalls).toBe(1);
    const restoredApprovalScopes = new Set(
      restoreApprovalSpy.mock.calls.map(([, executionScope]) => executionScope)
    );
    expect(restoredApprovalScopes.size).toBeGreaterThanOrEqual(4);
    const initialChildThreads = initialRun.getChildCheckpointThreadIds();
    const earlyApprovedChildThreads =
      earlyApprovedRun.getChildCheckpointThreadIds();
    const approvedChildThreads = approvedRun.getChildCheckpointThreadIds();
    const rejectedChildThreads = rejectedRun.getChildCheckpointThreadIds();
    expect(initialChildThreads).toHaveLength(2);
    expect(earlyApprovedChildThreads).toHaveLength(1);
    expect(approvedChildThreads).toHaveLength(1);
    expect(rejectedChildThreads).toHaveLength(1);
    expect(approvedChildThreads[0]).not.toBe(rejectedChildThreads[0]);
    expect(earlyApprovedChildThreads[0]).not.toBe(approvedChildThreads[0]);
    expect(earlyApprovedChildThreads[0]).not.toBe(rejectedChildThreads[0]);
    expect(initialChildThreads).not.toContain(approvedChildThreads[0]);
    expect(initialChildThreads).not.toContain(rejectedChildThreads[0]);
    expect(
      updates
        .filter((event) => event.parentToolCallId === tc.id)
        .map((event) => event.subagentRunId)
    ).toEqual(expect.arrayContaining([sourceChildRunId]));
    expect(
      updates
        .filter((event) => event.parentToolCallId === tc.id)
        .every((event) => event.subagentRunId === sourceChildRunId)
    ).toBe(true);
  });

  it.each([true, false])(
    'forks nested resumes from each checkpoint in the manifest chain (restore hooks=%s)',
    async (restoreHooks) => {
      getChatModelClassSpy.mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return NestedHitlFakeChatModel;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

      const checkpointer = new MemorySaver();
      const registry = new HookRegistry();
      const executedTools: string[] = [];
      const baseRunId = `nested-subagent-rebuild-${Date.now()}`;
      registry.registerSession(`${baseRunId}-initial`, 'PreToolUse', {
        once: true,
        pattern: '^calculator$',
        hooks: [
          async (): Promise<PreToolUseHookOutput> => ({
            decision: 'ask',
            reason: 'review nested calculator',
          }),
        ],
      });
      const customHandlers: Record<string, t.EventHandler> = {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: (_event, rawData): void => {
            const request = rawData as t.ToolExecuteBatchRequest;
            executedTools.push(...request.toolCalls.map((call) => call.name));
            request.resolve(
              request.toolCalls.map((call) => ({
                toolCallId: call.id,
                status: 'success' as const,
                content: '42',
              }))
            );
          },
        },
      };
      const createRun = (runId: string): Promise<Run<t.IState>> =>
        Run.create<t.IState>({
          runId,
          graphConfig: {
            type: 'standard',
            agents: [createParentAgentWithNestedChildTool()],
            compileOptions: { checkpointer },
          },
          returnContent: true,
          skipCleanup: true,
          customHandlers,
          hooks:
            restoreHooks || runId.endsWith('-initial') ? registry : undefined,
          humanInTheLoop: { enabled: true },
        });
      const parentCall = makeSubagentToolCall(
        'call_nested_researcher',
        'Delegate a nested calculation'
      );
      const initialRun = await createRun(`${baseRunId}-initial`);
      initialRun.Graph!.overrideTestModel(
        ['Delegating...', 'Final answer.'],
        5,
        [parentCall]
      );

      await initialRun.processStream(
        { messages: [new HumanMessage('calculate through two agents')] },
        callerConfig
      );
      const paused = initialRun.getInterrupt();
      expect(paused?.payload).toMatchObject({
        type: 'tool_approval',
        subagent: {
          agent_id: 'calculator-grandchild',
          parent_tool_call_id: 'call_nested_calculator_worker',
        },
      });
      const oldParentCheckpoint = {
        ...callerConfig,
        configurable: {
          ...callerConfig.configurable,
          checkpoint_id: paused?.checkpointId,
          checkpoint_ns: paused?.checkpointNs ?? '',
        },
      };
      const rebuiltRun = await createRun(`${baseRunId}-rebuilt`);
      rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
      const warningSpy = jest
        .spyOn(console, 'warn')
        .mockImplementation((): void => undefined);
      try {
        await initialRun.resume([{ type: 'approve' }], callerConfig);
        await rebuiltRun.resume(
          [{ type: 'reject', reason: 'reject stale nested branch' }],
          oldParentCheckpoint
        );
        expect(warningSpy).not.toHaveBeenCalledWith(
          expect.stringContaining('toolCallStepIds missing entry')
        );
      } finally {
        warningSpy.mockRestore();
      }

      expect(initialRun.getInterrupt()).toBeUndefined();
      expect(rebuiltRun.getInterrupt()).toBeUndefined();
      expect(executedTools).toEqual(['calculator']);
      expect(initialRun.getChildCheckpointThreadIds().length).toBeGreaterThan(
        2
      );
      expect(rebuiltRun.getChildCheckpointThreadIds().length).toBeGreaterThan(
        1
      );
    }
  );

  it('restores child tool-output references into rebuilt branches', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return ReferenceHitlFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const referenceValue = 'checkpoint-scoped producer output';
    const consumedValues: unknown[] = [];
    const checkpointer = new MemorySaver();
    const registry = new HookRegistry();
    const baseRunId = `subagent-reference-rebuild-${Date.now()}`;
    registry.registerSession(`${baseRunId}-initial`, 'PreToolUse', {
      once: true,
      pattern: '^consume_value$',
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review referenced input',
        }),
      ],
    });
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          request.resolve(
            request.toolCalls.map((call) => {
              if (call.name === 'consume_value') {
                consumedValues.push(call.args.value);
              }
              return {
                toolCallId: call.id,
                status: 'success' as const,
                content:
                  call.name === 'produce_value' ? referenceValue : 'consumed',
              };
            })
          );
        },
      },
    };
    const createRun = (runId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithReferenceTools()],
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
        toolOutputReferences: { enabled: true },
      });
    const parentCall = makeSubagentToolCall('call_reference_researcher');
    const initialRun = await createRun(`${baseRunId}-initial`);
    initialRun.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      parentCall,
    ]);

    await initialRun.processStream(
      { messages: [new HumanMessage('produce and consume a value')] },
      callerConfig
    );
    const paused = initialRun.getInterrupt();
    expect(paused?.payload).toMatchObject({
      type: 'tool_approval',
      action_requests: [
        { tool_call_id: 'call_consume_value', name: 'consume_value' },
      ],
    });
    const rebuiltRun = await createRun(`${baseRunId}-rebuilt`);
    rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
    await rebuiltRun.resume([{ type: 'approve' }], {
      ...callerConfig,
      configurable: {
        ...callerConfig.configurable,
        checkpoint_id: paused?.checkpointId,
        checkpoint_ns: paused?.checkpointNs ?? '',
      },
    });

    expect(rebuiltRun.getInterrupt()).toBeUndefined();
    expect(consumedValues).toEqual([referenceValue]);
  });
});

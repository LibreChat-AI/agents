import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  describe,
  it,
  expect,
  jest,
  afterEach,
  beforeEach,
} from '@jest/globals';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  PreToolUseHookOutput,
  PostToolUseHookOutput,
  PostToolUseFailureHookOutput,
  PostToolBatchEntry,
  PostToolBatchHookInput,
  PostToolBatchHookOutput,
  RunStartHookOutput,
  StopHookInput,
  StopHookOutput,
  UserPromptSubmitHookOutput,
} from '@/hooks';
import type { ReplayableSubagentTool } from '@/tools/subagent/SubagentReplay';
import type * as t from '@/types';
import {
  TOOL_APPROVAL_REVIEW_CONFIG_KEY,
  createToolApprovalReviewEvidence,
} from '@/hitl/approvalReview';
import { TOOL_BATCH_REPLAY_KEY, getToolBatchReplayState } from '../toolBatchReplay';
import {
  SUBAGENT_REPLAY_CONTROLLER,
} from '@/tools/subagent/SubagentReplay';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { Constants, Providers as providers, GraphEvents } from '@/common';
import { getProviderMessageProvenance } from '@/messages/provenance';
import { getRunStepResumeState } from '@/tools/runStepResume';
import { HookRegistry, createToolPolicyHook } from '@/hooks';
import * as events from '@/utils/events';
import { askUserQuestion } from '@/hitl';
import { ToolNode } from '../ToolNode';

async function flushAsyncWork(): Promise<void> {
  await Promise.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setImmediate(resolve));
  await Promise.resolve();
}

afterEach(async () => {
  await flushAsyncWork();
  jest.restoreAllMocks();
  await flushAsyncWork();
});

/**
 * Schema-only tool stub. ToolNode in event-driven mode uses the schema
 * for binding/discovery but routes execution through the host via
 * `ON_TOOL_EXECUTE`, so the actual `func` here is never called.
 */
function createSchemaStub(name: string): StructuredToolInterface {
  return tool(async () => 'unused', {
    name,
    description: 'schema-only stub; host executes via ON_TOOL_EXECUTE',
    schema: z.object({ command: z.string() }),
  }) as unknown as StructuredToolInterface;
}

/**
 * Wires a fake host that responds to every `ON_TOOL_EXECUTE` event by
 * resolving the request promise with `mockResults`. Mirrors the pattern
 * used in `ToolNode.outputReferences.test.ts` so the event-driven path
 * actually returns ToolMessages without spinning up a real host.
 */
function mockEventDispatch(mockResults: t.ToolExecuteResult[]): void {
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      if (event !== 'on_tool_execute') {
        return;
      }
      const request = data as Record<string, unknown>;
      if (typeof request.resolve === 'function') {
        (request.resolve as (r: t.ToolExecuteResult[]) => void)(mockResults);
      }
    });
}

async function stripStopContinuationExecutionIds(
  checkpointer: MemorySaver,
  threadId: string
): Promise<void> {
  for (const checkpoints of Object.values(
    checkpointer.storage[threadId] ?? {}
  )) {
    for (const stored of Object.values(checkpoints)) {
      const checkpoint = await checkpointer.serde.loadsTyped('json', stored[0]);
      const runStepState = (
        checkpoint as {
          channel_values?: { runStepState?: t.RunStepResumeState };
        }
      ).channel_values?.runStepState;
      if (runStepState != null) {
        delete runStepState.stopContinuationExecutionId;
        const [, serialized] = await checkpointer.serde.dumpsTyped(checkpoint);
        stored[0] = serialized;
      }
    }
  }
  for (const [key, writes] of Object.entries(checkpointer.writes)) {
    const [storedThreadId] = JSON.parse(key) as [string, string, string];
    if (storedThreadId !== threadId) {
      continue;
    }
    for (const stored of Object.values(writes)) {
      const value = await checkpointer.serde.loadsTyped('json', stored[2]);
      const runStepState = getRunStepResumeState(value);
      if (runStepState == null) {
        continue;
      }
      delete runStepState.stopContinuationExecutionId;
      const [, serialized] = await checkpointer.serde.dumpsTyped(value);
      stored[2] = serialized;
    }
  }
}

type MessagesUpdate = { messages: BaseMessage[] };
type InterruptStateSnapshot = {
  config?: RunnableConfig;
  tasks?: Array<{
    interrupts?: Array<{ id?: string }>;
  }>;
};
type CompiledMessagesGraph = Runnable<unknown, { messages: BaseMessage[] }> & {
  invoke(input: unknown, config?: RunnableConfig): Promise<unknown>;
  getState?(
    config: RunnableConfig
  ): Promise<{ config?: RunnableConfig } | undefined>;
  getStateHistory?(
    config: RunnableConfig
  ): AsyncIterableIterator<InterruptStateSnapshot>;
};

/** Factory for a minimal `agent → tools → END` graph wrapping the ToolNode. */
function buildHITLGraph(
  toolNode: ToolNode,
  toolCalls: Array<{ id?: string; name: string; args: Record<string, unknown> }>,
  checkpointer = new MemorySaver()
): CompiledMessagesGraph {
  const toolCallIds = new Set(toolCalls.map((call) => call.id));
  const builder = new StateGraph(MessagesAnnotation)
    .addNode('agent', (state: { messages?: BaseMessage[] }): MessagesUpdate => {
      /**
       * Emit the AIMessage carrying tool_calls until this test graph
       * actually has a matching ToolMessage in state. LangGraph usually
       * resumes at the interrupted `tools` node, but under full-suite
       * async callback pressure it can re-enter this tiny test graph from
       * START while still carrying the resume value. A call-count based
       * fake agent then returned "done" too early and made HITL resume
       * assertions order-dependent. State is the stable contract here:
       * no tool result means the tool node still needs work.
       */
      const hasMatchingToolResult =
        state.messages?.some(
          (message): boolean =>
            message._getType() === 'tool' &&
            toolCallIds.has((message as ToolMessage).tool_call_id)
        ) === true;
      if (!hasMatchingToolResult) {
        return {
          messages: [new AIMessage({ content: '', tool_calls: toolCalls })],
        };
      }
      return { messages: [new AIMessage({ content: 'done' })] };
    })
    .addNode('tools', toolNode)
    .addEdge(START, 'agent')
    .addEdge('agent', 'tools')
    .addEdge('tools', END);
  return builder.compile({
    checkpointer,
  }) as unknown as CompiledMessagesGraph;
}

function makeHookRegistry(
  decision: 'allow' | 'deny' | 'ask',
  reason?: string
): HookRegistry {
  const registry = new HookRegistry();
  registry.register('PreToolUse', {
    hooks: [
      async (): Promise<PreToolUseHookOutput> => ({
        decision,
        ...(reason != null ? { reason } : {}),
      }),
    ],
  });
  return registry;
}

function resumeFromInterrupt<TResume>(
  interrupted: unknown,
  resume: TResume
): Command {
  if (isInterrupted<unknown>(interrupted)) {
    const interruptId = interrupted.__interrupt__[0]?.id;
    if (typeof interruptId === 'string' && interruptId.length > 0) {
      return new Command({ resume: { [interruptId]: resume } });
    }
  }
  return new Command({ resume });
}

async function resumeGraph<TResume>(
  graph: CompiledMessagesGraph,
  interrupted: unknown,
  resume: TResume,
  config: RunnableConfig
): Promise<unknown> {
  const interruptId = isInterrupted<unknown>(interrupted)
    ? interrupted.__interrupt__[0]?.id
    : undefined;
  let checkpointConfig = config;
  if (typeof interruptId === 'string' && graph.getStateHistory != null) {
    for await (const snapshot of graph.getStateHistory(config)) {
      const hasMatchingInterrupt =
        snapshot.tasks?.some(
          (task) =>
            task.interrupts?.some(
              (interrupt) => interrupt.id === interruptId
            ) === true
        ) === true;
      if (hasMatchingInterrupt && snapshot.config != null) {
        checkpointConfig = snapshot.config;
        break;
      }
    }
  } else {
    checkpointConfig = (await graph.getState?.(config))?.config ?? config;
  }
  return graph.invoke(
    resumeFromInterrupt(interrupted, resume),
    {
      ...checkpointConfig,
      configurable: {
        ...checkpointConfig.configurable,
        [TOOL_BATCH_REPLAY_KEY]: isInterrupted<unknown>(interrupted)
          ? getToolBatchReplayState(interrupted.__interrupt__[0]?.value)
          : undefined,
      },
    }
  );
}

describe('ToolNode HITL — `ask` decision raises interrupt() when humanInTheLoop is enabled', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('raises a tool_approval interrupt with the pending tool call payload', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'should-not-run', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'review tool args'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-1' } };

    const result = await graph.invoke({ messages: [] }, config);

    expect(isInterrupted<t.HumanInterruptPayload>(result)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(result)) {
      throw new Error('expected interrupt');
    }
    const interrupts = result.__interrupt__;
    expect(interrupts).toHaveLength(1);
    const payload = interrupts[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.action_requests).toEqual([
      {
        tool_call_id: 'call_1',
        name: 'echo',
        arguments: { command: 'list /' },
        description: 'review tool args',
      },
    ]);
    expect(payload.review_configs).toEqual([
      {
        action_name: 'echo',
        tool_call_id: 'call_1',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
    ]);
  });

  it('waits for approval before executing an explicit ask rule in bypass mode', async () => {
    let toolExecuted = false;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        toolExecuted = true;
        const request = data as {
          resolve: (results: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          { toolCallId: 'call_1', content: 'deleted', status: 'success' },
        ]);
      });
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        createToolPolicyHook({
          mode: 'bypass',
          ask: ['dangerous_*'],
        }),
      ],
    });
    const node = new ToolNode({
      tools: [createSchemaStub('dangerous_tool')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      {
        id: 'call_1',
        name: 'dangerous_tool',
        args: { command: 'delete data' },
      },
    ]);
    const config = {
      configurable: { thread_id: 'thread-bypass-explicit-ask' },
    };

    const interrupted = await graph.invoke({ messages: [] }, config);

    expect(isInterrupted<t.HumanInterruptPayload>(interrupted)).toBe(true);
    expect(toolExecuted).toBe(false);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'approve' }],
      config
    )) as { messages: BaseMessage[] };

    expect(toolExecuted).toBe(true);
    expect(
      resumed.messages.some(
        (message) =>
          message._getType() === 'tool' &&
          (message as ToolMessage).tool_call_id === 'call_1' &&
          message.content === 'deleted'
      )
    ).toBe(true);
  });

  it('executes an unmatched tool without interruption in bypass mode', async () => {
    let toolExecuted = false;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        toolExecuted = true;
        const request = data as {
          resolve: (results: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          { toolCallId: 'call_1', content: 'read result', status: 'success' },
        ]);
      });
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        createToolPolicyHook({
          mode: 'bypass',
          ask: ['dangerous_*'],
        }),
      ],
    });
    const node = new ToolNode({
      tools: [createSchemaStub('read_tool')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'read_tool', args: { command: 'read data' } },
    ]);

    const result = await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'thread-bypass-unmatched' } }
    );

    expect(isInterrupted(result)).toBe(false);
    expect(toolExecuted).toBe(true);
  });

  it('resume with approve runs the tool through the host event path', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do-it' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-approve' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted(interrupted)).toBe(true);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'approve' }],
      config
    )) as {
      messages: BaseMessage[];
    };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].content).toBe('host-result');
    expect(toolMessages[0].status).not.toBe('error');
  });

  it('resume with reject blocks the tool and emits an error ToolMessage', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'rm -rf /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-reject' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'reject', reason: 'destructive command' }],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('destructive command');
  });

  it('resume with edit substitutes the tool input before invocation', async () => {
    const capturedRequests: t.ToolCallRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        capturedRequests.push(...request.toolCalls);
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-edit' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    await resumeGraph(
      graph,
      interrupted,
      [{ type: 'edit', updatedInput: { command: 'patched' } }],
      config
    );

    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].args).toEqual({ command: 'patched' });
  });

  it('resume with respond emits the user-supplied text as a successful ToolMessage and skips host execution', async () => {
    const dispatchSpy = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'search' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-respond' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    const dispatchCallsBefore = dispatchSpy.mock.calls.filter(
      ([event]) => event === 'on_tool_execute'
    ).length;

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'respond', responseText: 'no relevant results' }],
      config
    )) as { messages: BaseMessage[] };

    const dispatchCallsAfter = dispatchSpy.mock.calls.filter(
      ([event]) => event === 'on_tool_execute'
    ).length;

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].content).toBe('no relevant results');
    expect(toolMessages[0].status).not.toBe('error');
    expect(dispatchCallsAfter).toBe(dispatchCallsBefore);
  });

  it('advertises respond in review_configs.allowed_decisions', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = {
      configurable: { thread_id: 'thread-hitl-allowed-decisions' },
    };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.review_configs[0].allowed_decisions).toEqual([
      'approve',
      'reject',
      'edit',
      'respond',
    ]);
  });

  it('resume with a record keyed by tool_call_id is accepted', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do-it' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-map' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      { call_1: { type: 'approve' } },
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].content).toBe('host-result');
  });
});

describe('ToolNode HITL — background deny policy', () => {
  it('blocks event-driven calls requesting approval with actionable feedback', async () => {
    mockEventDispatch([]);
    const dispatch = jest.spyOn(events, 'safeDispatchCustomEvent');
    const node = new ToolNode({
      tools: [createSchemaStub('write_file')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_write', 'step_write']]),
      hookRegistry: makeHookRegistry('ask', 'write requires approval'),
      humanInTheLoop: { enabled: false, backgroundPausePolicy: 'deny' },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_write', name: 'write_file', args: { command: 'write' } },
    ]);

    const result = (await graph.invoke(
      { messages: [] },
      {
        configurable: { thread_id: 'background-deny-event' },
      }
    )) as { messages: BaseMessage[] };
    const toolMessages = result.messages.filter(
      (message): message is ToolMessage => message instanceof ToolMessage
    );
    expect(isInterrupted(result)).toBe(false);
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(toolMessages[0].content).toContain(
      'Approval required for "write_file"'
    );
    expect(toolMessages[0].content).toContain('write requires approval');
    expect(
      dispatch.mock.calls.filter(([event]) => event === 'on_tool_execute')
    ).toHaveLength(0);
  });

  it('blocks a direct tool requesting approval before its body runs', async () => {
    const execute = jest.fn(async () => 'should not run');
    const writeTool = tool(execute, {
      name: 'write_file',
      description: 'Writes files',
      schema: z.object({ command: z.string() }),
    });
    const node = new ToolNode({
      tools: [writeTool],
      toolMap: new Map([['write_file', writeTool]]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: false, backgroundPausePolicy: 'deny' },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_write', name: 'write_file', args: { command: 'write' } },
    ]);

    const result = (await graph.invoke(
      { messages: [] },
      {
        configurable: { thread_id: 'background-deny-direct' },
      }
    )) as { messages: BaseMessage[] };
    const toolMessages = result.messages.filter(
      (message): message is ToolMessage => message instanceof ToolMessage
    );
    expect(isInterrupted(result)).toBe(false);
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(toolMessages[0].content).toContain(
      'Approval required for "write_file"'
    );
    expect(execute).not.toHaveBeenCalled();
  });
});

describe('ToolNode HITL — opt-out (`humanInTheLoop: { enabled: false }`) is fail-closed', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('blocks the tool with a ToolMessage error and never raises an interrupt', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'HITL explicitly disabled'),
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-optout' } };

    const result = (await graph.invoke({ messages: [] }, config)) as {
      messages: BaseMessage[];
    };

    expect(isInterrupted(result)).toBe(false);
    const toolMessages = result.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'HITL explicitly disabled'
    );
  });

  it('blocks the tool when `humanInTheLoop` is omitted (default-off)', async () => {
    /**
     * Default is OFF until host UIs (notably LibreChat) ship the
     * approval-rendering affordances. With HITL omitted, an `ask`
     * decision must collapse into a synchronous block — same fail-
     * closed behavior as the explicit `{ enabled: false }` opt-out.
     * This test guards against accidentally re-enabling the default-on
     * path before the consumer ecosystem is ready.
     */
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'default-off-blocks'),
      // humanInTheLoop intentionally omitted — should default to disabled
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-default' } };

    const out = (await graph.invoke({ messages: [] }, config)) as {
      messages: BaseMessage[];
    };
    expect(isInterrupted<t.HumanInterruptPayload>(out)).toBe(false);
    const toolMessages = out.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('default-off-blocks');
  });
});

describe('ToolNode HITL — multi-tool batches', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('bundles multiple ask decisions into a single interrupt and resolves per call', async () => {
    const capturedRequests: t.ToolCallRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        capturedRequests.push(...request.toolCalls);
        request.resolve(
          request.toolCalls.map(
            (c): t.ToolExecuteResult => ({
              toolCallId: c.id,
              content: `ran:${c.name}`,
              status: 'success',
            })
          )
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo'), createSchemaStub('cat')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_1', 'step_call_1'],
        ['call_2', 'step_call_2'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'one' } },
      { id: 'call_2', name: 'cat', args: { command: 'two' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-batch' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(interrupted)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.action_requests.map((r) => r.tool_call_id)).toEqual([
      'call_1',
      'call_2',
    ]);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'approve' }, { type: 'reject', reason: 'too risky' }],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(2);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_1')!.content).toBe('ran:echo');
    expect(byId.get('call_1')!.status).not.toBe('error');
    expect(byId.get('call_2')!.status).toBe('error');
    expect(String(byId.get('call_2')!.content)).toContain('too risky');

    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].id).toBe('call_1');
  });
});

describe('Run integration — HITL fallback checkpointer + resume', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('Run.create does NOT install a MemorySaver fallback by default (HITL is off until host UI ships)', async () => {
    /**
     * Default-off rationale: HITL ships the interrupt machinery but
     * stays opt-in until host UIs (notably LibreChat) can render and
     * resolve `tool_approval` interrupts. With HITL omitted, the SDK
     * must NOT silently install a checkpointer — that would suggest
     * the run can pause/resume when in fact the `ask` path will
     * fail-closed. Plan of record: flip the default to ON in a future
     * minor once the consumer ecosystem is ready.
     */
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-default-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      // humanInTheLoop intentionally omitted — default is OFF
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeUndefined();
  });

  it('Run.create installs a MemorySaver fallback when HITL is explicitly enabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-explicit-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);
    expect(run.Graph?.humanInTheLoop?.enabled).toBe(true);
  });

  it('Run.create preserves a host-supplied checkpointer when HITL is explicitly enabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const hostCheckpointer = new MemorySaver();
    const run = await Run.create<t.IState>({
      runId: 'hitl-host-checkpointer',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
        compileOptions: { checkpointer: hostCheckpointer },
      },
      humanInTheLoop: { enabled: true },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBe(hostCheckpointer);
  });

  it('processStream defaults durability to "exit" when a checkpointer is active', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'durability-exit-default',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });
    expect(run.Graph?.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);

    const graph = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END)
      .compile();
    const spy = jest.spyOn(graph, 'streamEvents');
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      { version: 'v2', configurable: { thread_id: 't' } }
    );

    const streamedConfig = spy.mock.calls[0]?.[1] as
      | t.RunStreamConfig
      | undefined;
    expect(streamedConfig?.durability).toBe('exit');
  });

  it('processStream respects an explicit caller durability over the checkpointer default', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'durability-explicit-override',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });

    const graph = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END)
      .compile();
    const spy = jest.spyOn(graph, 'streamEvents');
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      { version: 'v2', durability: 'sync', configurable: { thread_id: 't' } }
    );

    const streamedConfig = spy.mock.calls[0]?.[1] as
      | t.RunStreamConfig
      | undefined;
    expect(streamedConfig?.durability).toBe('sync');
  });

  it('processStream leaves durability unset when no checkpointer is active', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'durability-no-checkpointer',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      // humanInTheLoop omitted — no checkpointer installed
    });
    expect(run.Graph?.compileOptions?.checkpointer).toBeUndefined();

    const graph = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END)
      .compile();
    const spy = jest.spyOn(graph, 'streamEvents');
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      { version: 'v2', configurable: { thread_id: 't' } }
    );

    const streamedConfig = spy.mock.calls[0]?.[1] as
      | t.RunStreamConfig
      | undefined;
    expect(streamedConfig?.durability).toBeUndefined();
  });

  it('defaults durability to "exit" when HITL installs the fallback checkpointer but caller compileOptions omit it', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'durability-hitl-fallback-compileopts',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
        // Caller compileOptions without a checkpointer: HITL adds a MemorySaver
        // fallback while preserving the caller's remaining compile options.
        compileOptions: { interruptBefore: [] },
      },
      humanInTheLoop: { enabled: true },
    });
    expect(run.Graph?.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);
    expect(run.Graph?.compileOptions?.interruptBefore).toEqual([]);

    const graph = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END)
      .compile();
    const spy = jest.spyOn(graph, 'streamEvents');
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      { version: 'v2', configurable: { thread_id: 't' } }
    );

    const streamedConfig = spy.mock.calls[0]?.[1] as
      | t.RunStreamConfig
      | undefined;
    expect(streamedConfig?.durability).toBe('exit');
  });

  it('Run.resume forwards update + goto into the resume Command (langgraph 1.4.5)', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-resume-update-goto',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
    });

    const spy = jest.spyOn(run, 'processStream').mockResolvedValue(undefined);

    const decision = [{ type: 'approve' as const }];
    const update = { messages: [new AIMessage('host-edit')] };
    await run.resume(
      decision,
      { version: 'v1', configurable: { thread_id: 't' } },
      undefined,
      { update, goto: 'agent' }
    );

    const cmd = spy.mock.calls[0]?.[0] as Command;
    expect(cmd).toBeInstanceOf(Command);
    // No interrupt was captured, so the resume value passes through unscoped.
    expect(cmd.resume).toEqual(decision);
    expect(cmd.update).toEqual(update);
    expect(cmd.goto).toEqual(['agent']); // langgraph normalizes goto to an array

    // Backward-compat: omitting commandOptions leaves update unset, goto empty.
    await run.resume(decision, {
      version: 'v1',
      configurable: { thread_id: 't' },
    });
    const cmd2 = spy.mock.calls[1]?.[0] as Command;
    expect(cmd2.update).toBeUndefined();
    expect(cmd2.goto).toEqual([]);
  });

  it('Run.resume restores a persisted interrupt before scoping a rebuilt resume', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const registry = new HookRegistry();
    const persistedMatcher = {
      hooks: [async (): Promise<PreToolUseHookOutput> => ({ decision: 'ask' })],
    };
    registry.registerSession(
      'persisted-hook-session',
      'PreToolUse',
      persistedMatcher
    );

    const run = await Run.create<t.IState>({
      runId: 'hitl-rebuilt-scope',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: {
              modelName: 'gpt-4o-mini',
              apiKey: 'test-key',
            },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    const persistedState = {
      config: {
        configurable: {
          thread_id: 'durable-thread',
          checkpoint_id: 'interrupted-checkpoint',
          checkpoint_ns: '',
        },
      },
      tasks: [
        {
          interrupts: [
            {
              id: 'persisted-interrupt',
              value: {
                type: 'tool_approval',
                hook_session_id: 'persisted-hook-session',
                action_requests: [],
                review_configs: [],
              },
            },
          ],
        },
      ],
    };
    const getState = jest.fn(
      async (_config: RunnableConfig, _options?: { subgraphs?: boolean }) =>
        persistedState
    );
    run.graphRunnable = { getState } as unknown as t.CompiledStateWorkflow;
    const processSpy = jest
      .spyOn(run, 'processStream')
      .mockResolvedValue(undefined);
    const callerConfig = {
      version: 'v2' as const,
      configurable: { thread_id: 'durable-thread' },
    };
    const decision = [{ type: 'approve' as const }];

    await run.resume(decision, callerConfig);

    expect(getState).toHaveBeenCalledWith(callerConfig, { subgraphs: true });
    const command = processSpy.mock.calls[0]?.[0] as Command;
    expect(command.resume).toEqual({ 'persisted-interrupt': decision });
    expect(processSpy.mock.calls[0]?.[1].configurable).toMatchObject({
      thread_id: 'durable-thread',
      checkpoint_id: 'interrupted-checkpoint',
      checkpoint_ns: '',
    });
    expect(run.getInterrupt()).toMatchObject({
      interruptId: 'persisted-interrupt',
      threadId: 'durable-thread',
      payload: {
        type: 'tool_approval',
        hook_session_id: 'persisted-hook-session',
      },
    });
    expect(
      registry.getMatchers('PreToolUse', 'persisted-hook-session')
    ).toEqual([persistedMatcher]);
    expect(registry.getMatchers('PreToolUse', run.id)).toEqual([
      persistedMatcher,
    ]);
  });

  it('re-exports langgraph HITL primitives from the SDK barrel for host use', async () => {
    const indexExports = await import('@/index');
    expect(indexExports.MemorySaver).toBe(MemorySaver);
    expect(indexExports.Command).toBe(Command);
    expect(indexExports.INTERRUPT).toBeDefined();
    expect(typeof indexExports.interrupt).toBe('function');
    expect(typeof indexExports.isInterrupted).toBe('function');
    expect(typeof indexExports.BaseCheckpointSaver).toBe('function');
  });

  it('Run.create does not attach a checkpointer when HITL is explicitly disabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-optout-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: false },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeUndefined();
  });

  it('Run.resume() drives the host all the way through the resume command path', async () => {
    /** End-to-end on the Run wrapper: build a HITL graph that
     * interrupts on first invoke, then drive resume via the Run's
     * own `resume()` method (not raw graph.invoke + Command).
     * Validates the full Run.resume → processStream(Command) path. */
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const hexToolCallId = '0123456789abcdef0123456789abcdef';
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([[hexToolCallId, 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: hexToolCallId, name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'run-resume-direct',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'run-resume-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);
    expect(run.getInterrupt()).toBeDefined();
    expect(dispatchCount).toBe(0);

    /** This is the API contract under test: Run.resume() with a
     * tool_call_id-keyed decision map (not graph.invoke + Command).
     * The tool_call_id intentionally looks like a LangGraph interrupt
     * id; Run.resume must still wrap it under the real interrupt id. */
    await run.resume({ [hexToolCallId]: { type: 'approve' } }, callerConfig);

    expect(dispatchCount).toBe(1);
    /** Resume completed naturally: interrupt cleared, no halt
     * reason carried over from the previous pass. */
    expect(run.getInterrupt()).toBeUndefined();
    expect(run.getHaltReason()).toBeUndefined();
  });

  it('preserves the warm continuation budget across a HITL resume', async () => {
    const { Run: RunClass } = await import('@/run');
    const registry = new HookRegistry();
    const stopInputs: StopHookInput[] = [];
    let run: Awaited<ReturnType<typeof RunClass.create<t.IState>>>;
    const checkpointer = new MemorySaver();
    const toolCallId = 'abcdef0123456789abcdef0123456789';
    const threadConfig = {
      configurable: { thread_id: 'warm-hitl-budget-thread' },
      version: 'v2' as const,
    };
    const seedRun = await RunClass.create<t.IState>({
      runId: 'warm-hitl-budget-seed',
      graphConfig: {
        type: 'standard',
        compileOptions: { checkpointer },
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
            tools: [createSchemaStub('echo')],
          },
        ],
      },
      skipCleanup: true,
    });
    seedRun.Graph?.overrideTestModel(['seed answer']);
    await seedRun.processStream(
      { messages: [new HumanMessage('seed prompt')] },
      threadConfig
    );
    const seedTuple = await checkpointer.getTuple(threadConfig);
    const seedCheckpointId = seedTuple?.config.configurable?.checkpoint_id;
    expect(typeof seedCheckpointId).toBe('string');
    const config = {
      ...threadConfig,
      configurable: {
        ...threadConfig.configurable,
        checkpoint_id: seedCheckpointId,
      },
    };
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });
    registry.register('Stop', {
      hooks: [
        async (input): Promise<StopHookOutput> => {
          stopInputs.push(input);
          if (input.continuationCount > 1) {
            return { decision: 'continue' };
          }
          if (input.continuationCount === 1) {
            return {
              decision: 'block',
              injectedMessages: [
                {
                  role: 'user',
                  content: 'post-upgrade steer',
                  source: 'steer',
                },
              ],
            };
          }
          run.Graph?.overrideTestModel(
            ['tool request', 'final answer'],
            undefined,
            [
              {
                id: toolCallId,
                name: 'echo',
                args: { command: 'continue' },
                type: 'tool_call',
              },
            ]
          );
          return {
            decision: 'block',
            injectedMessages: [
              { role: 'user', content: 'late steer', source: 'steer' },
            ],
          };
        },
      ],
    });
    run = await RunClass.create<t.IState>({
      runId: 'warm-hitl-budget',
      graphConfig: {
        type: 'standard',
        compileOptions: { checkpointer },
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
            tools: [createSchemaStub('echo')],
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
      maxStopContinuations: 2,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel(['first answer']);

    await run.processStream(
      { messages: [new HumanMessage('initial prompt')] },
      config
    );
    const interrupt = run.getInterrupt();
    expect(interrupt).toBeDefined();
    expect(interrupt?.checkpointId).toEqual(expect.any(String));
    expect(interrupt?.checkpointId).not.toBe(seedCheckpointId);
    expect(stopInputs.map((input) => input.continuationCount)).toEqual([0]);
    await stripStopContinuationExecutionIds(
      checkpointer,
      threadConfig.configurable.thread_id
    );

    run = await RunClass.create<t.IState>({
      runId: 'warm-hitl-budget',
      graphConfig: {
        type: 'standard',
        compileOptions: { checkpointer },
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
            tools: [createSchemaStub('echo')],
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
      maxStopContinuations: 2,
      skipCleanup: true,
    });
    run.Graph?.overrideTestModel([
      'final answer',
      'post-upgrade continued answer',
    ]);
    await run.resume([{ type: 'approve' }], {
      ...config,
      configurable: {
        ...config.configurable,
        checkpoint_id: interrupt?.checkpointId,
      },
    });

    expect(run.getInterrupt()).toBeUndefined();
    expect(stopInputs.map((input) => input.continuationCount)).toEqual([
      0, 1, 2,
    ]);
    expect(
      stopInputs.map((input) => input.continuationBudgetRemaining)
    ).toEqual([2, 1, 0]);
  });

  it('Run.resume() forwards `update` so langgraph applies the channel edit through streamEvents', async () => {
    /** Executing proof (not a spy): interrupt on a tool, resume with an
     * injected message via `update`, then read the committed checkpoint.
     * langgraph 1.4.5 maps an INPUT resume Command through `mapCommand`
     * (pregel/io.js), which applies resume AND update AND goto, so the
     * injected message must land in the messages channel. */
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const hexToolCallId = '0123456789abcdef0123456789abcdef';
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([[hexToolCallId, 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: hexToolCallId, name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const { HumanMessage } = await import('@langchain/core/messages');
    const run = await Run.create<t.IState>({
      runId: 'run-resume-update',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'run-resume-update-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);
    expect(run.getInterrupt()).toBeDefined();

    const injected = new HumanMessage({ content: 'human-injected-on-resume' });
    await run.resume(
      { [hexToolCallId]: { type: 'approve' } },
      callerConfig,
      undefined,
      { update: { messages: [injected] } }
    );

    expect(run.getInterrupt()).toBeUndefined();

    const state = await graph.getState(callerConfig);
    const contents = (state.values.messages as BaseMessage[]).map(
      (m) => m.content
    );
    /** Proves langgraph honored `update` on the INPUT resume Command. */
    expect(contents).toContain('human-injected-on-resume');
    /** Resume itself still completed: the approved tool produced its result. */
    expect(contents).toContain('host-result');
  });

  it('Run.getHaltReason() reports prompt_denied when UserPromptSubmit denies the prompt', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'deny',
          reason: 'PII detected',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-deny-haltreason',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('please tell me their SSN')] },
      { configurable: { thread_id: 'prompt-deny-thread' }, version: 'v2' }
    );

    /** Hook denied the prompt — run returns undefined AND
     * `getHaltReason()` carries the reason so the host can
     * distinguish "blocked" from "natural empty completion". */
    expect(result).toBeUndefined();
    expect(run.getHaltReason()).toBe('PII detected');
  });

  it('Run.getHaltReason() falls back to canonical prompt_denied when deny carries no reason', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'deny',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-deny-canonical',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'prompt-deny-canonical-thread' },
        version: 'v2',
      }
    );

    /** Hook returned `deny` without a reason — host gets the
     * canonical 'prompt_denied' string so it can route on a stable
     * discriminator. */
    expect(run.getHaltReason()).toBe('prompt_denied');
  });

  it('Run.getHaltReason() reports prompt_requires_approval when UserPromptSubmit asks', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'ask',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-ask-haltreason',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('hello')] },
      { configurable: { thread_id: 'prompt-ask-thread' }, version: 'v2' }
    );

    /** Default reason when the hook didn't supply one — host can
     * route on the canonical string. */
    expect(run.getHaltReason()).toBe('prompt_requires_approval');
  });
});

describe('ToolNode HITL — additionalContext injection from hooks', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('injects PreToolUse + PostToolUse additionalContexts as a single HumanMessage', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'allow',
          additionalContext: 'pre-context: be careful',
        }),
      ],
    });
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          additionalContext: 'post-context: tool ran',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'ctx-thread-1' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain('pre-context: be careful');
    expect(String(injected!.content)).toContain('post-context: tool ran');
  });

  it('does not inject anything when no hook returns additionalContext', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({ decision: 'allow' }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'ctx-thread-2' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeUndefined();
  });

  it('marks mixed direct/event batch context as synthetic provenance', async () => {
    mockEventDispatch([
      { toolCallId: 'event_1', content: 'host-result', status: 'success' },
    ]);
    const direct = tool(async () => 'direct-result', {
      name: 'direct_echo',
      description: 'direct echo',
      schema: z.object({ command: z.string() }),
    }) as unknown as StructuredToolInterface;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => ({
          decision: 'allow',
          ...(input.toolName === 'direct_echo' && {
            additionalContext: 'direct policy context',
          }),
        }),
      ],
    });
    const node = new ToolNode({
      tools: [direct, createSchemaStub('event_echo')],
      eventDrivenMode: true,
      directToolNames: new Set(['direct_echo']),
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['direct_1', 'step_direct_1'],
        ['event_1', 'step_event_1'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });
    const graph = buildHITLGraph(node, [
      { id: 'direct_1', name: 'direct_echo', args: { command: 'a' } },
      { id: 'event_1', name: 'event_echo', args: { command: 'b' } },
    ]);

    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'mixed-context-thread' } }
    )) as { messages: BaseMessage[] };
    const context = result.messages.find(
      (message) =>
        message.getType() === 'human' &&
        String(message.content).includes('direct policy context')
    );

    expect(context).toBeDefined();
    expect(getProviderMessageProvenance(context!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
  });
});

describe('ToolNode HITL — PostToolBatch hook', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('fires once per dispatch with all entries (success + error mix), in batch order', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          { toolCallId: 'call_1', content: 'ok', status: 'success' },
          {
            toolCallId: 'call_2',
            content: '',
            status: 'error',
            errorMessage: 'boom',
          },
        ]);
      });

    const registry = new HookRegistry();
    let captured: PostToolBatchEntry[] | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          captured = (input as PostToolBatchHookInput).entries;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo'), createSchemaStub('cat')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_1', 'step_1'],
        ['call_2', 'step_2'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
      { id: 'call_2', name: 'cat', args: { command: 'b' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-thread' } }
    );

    expect(captured).toBeDefined();
    expect(captured!).toHaveLength(2);
    expect(captured![0].toolUseId).toBe('call_1');
    expect(captured![0].status).toBe('success');
    expect(captured![0].toolOutput).toBe('ok');
    expect(captured![1].toolUseId).toBe('call_2');
    expect(captured![1].status).toBe('error');
    expect(captured![1].error).toContain('boom');
  });

  it('a PostToolBatch additionalContext gets injected as a HumanMessage', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          additionalContext: 'remember to format the response as JSON',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-ctx-thread' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain('format the response as JSON');
    expect(getProviderMessageProvenance(injected!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
  });

  it('PostToolBatch injectedMessages land as individual HumanMessages after the consolidated context', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          additionalContext: 'batch convention',
          injectedMessages: [
            { role: 'user', content: 'steer one', source: 'steer' },
            { role: 'user', content: 'steer two', source: 'steer' },
          ],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-steer-thread' } }
    )) as { messages: BaseMessage[] };

    type KwargMessage = {
      additional_kwargs?: { source?: string; role?: string };
    };
    const humanMessages = result.messages.filter(
      (m) => m._getType() === 'human'
    );
    const contextIndex = humanMessages.findIndex(
      (m) => (m as KwargMessage).additional_kwargs?.source === 'hook'
    );
    const steerMessages = humanMessages.filter(
      (m) => (m as KwargMessage).additional_kwargs?.source === 'steer'
    );

    expect(contextIndex).toBeGreaterThanOrEqual(0);
    expect(steerMessages).toHaveLength(2);
    expect(String(steerMessages[0].content)).toBe('steer one');
    expect(String(steerMessages[1].content)).toBe('steer two');
    for (const steer of steerMessages) {
      expect((steer as KwargMessage).additional_kwargs?.role).toBe('user');
      expect(humanMessages.indexOf(steer)).toBeGreaterThan(contextIndex);
    }
    const toolIndex = result.messages.findIndex((m) => m._getType() === 'tool');
    const firstSteerIndex = result.messages.indexOf(steerMessages[0]);
    expect(firstSteerIndex).toBeGreaterThan(toolIndex);
  });

  it('PostToolBatch injectedMessages work without additionalContext', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          injectedMessages: [
            { role: 'user', content: 'solo steer', source: 'steer' },
          ],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'solo-steer-thread' } }
    )) as { messages: BaseMessage[] };

    type KwargMessage = {
      additional_kwargs?: { source?: string; role?: string };
    };
    const consolidated = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as KwargMessage).additional_kwargs?.source === 'hook'
    );
    const steer = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as KwargMessage).additional_kwargs?.source === 'steer'
    );
    expect(consolidated).toBeUndefined();
    expect(steer).toBeDefined();
    expect(String(steer!.content)).toBe('solo steer');
  });
});

describe('ToolNode HITL — per-hook allowedDecisions override', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('restricts the interrupt review_configs.allowed_decisions to the hook-supplied subset', async () => {
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const interrupted = await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'allowed-thread' } }
    );
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }
    expect(payload.review_configs[0].allowed_decisions).toEqual([
      'approve',
      'reject',
    ]);
  });
});

describe('Run — preventContinuation honored for pre-stream hooks', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('marks consolidated pre-stream hook context as synthetic provenance', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');
    const { HumanMessage: HM } = await import('@langchain/core/messages');
    const registry = new HookRegistry();
    registry.register('RunStart', {
      hooks: [
        async (): Promise<RunStartHookOutput> => ({
          additionalContext: 'runtime policy context',
        }),
      ],
    });
    const run = await Run.create<t.IState>({
      runId: 'prestream-provenance',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });
    const stateInputs = {
      messages: [new HM('hello')],
    } as t.IState;
    const halted = await (
      run as unknown as {
        runPreStreamHooks: (
          state: t.IState,
          threadId: string,
          config: Record<string, unknown>
        ) => Promise<boolean>;
      }
    ).runPreStreamHooks(stateInputs, 'prestream-thread', {});

    expect(halted).toBe(false);
    expect(stateInputs.messages).toHaveLength(2);
    expect(
      getProviderMessageProvenance(stateInputs.messages[1])?.parts
    ).toEqual([{ attribution: 'synthetic' }]);
  });

  it('returns undefined without invoking the graph when RunStart hook returns preventContinuation', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const registry = new HookRegistry();
    let runStartFired = false;
    registry.register('RunStart', {
      hooks: [
        async (): Promise<RunStartHookOutput> => {
          runStartFired = true;
          return {
            preventContinuation: true,
            stopReason: 'pre-flight policy halted run',
          };
        },
      ],
    });

    const run = await Run.create<t.IState>({
      runId: 'pc-runstart',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'pc-thread-1' },
        version: 'v2',
      }
    );

    expect(runStartFired).toBe(true);
    expect(result).toBeUndefined();
    /** Graph should not have been run — no messages added beyond the input. */
    expect(run.getInterrupt()).toBeUndefined();
  });

  it('returns undefined when UserPromptSubmit hook returns preventContinuation', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const registry = new HookRegistry();
    let promptFired = false;
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => {
          promptFired = true;
          return {
            preventContinuation: true,
            stopReason: 'rate limit reached',
          };
        },
      ],
    });

    const run = await Run.create<t.IState>({
      runId: 'pc-prompt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'pc-thread-2' },
        version: 'v2',
      }
    );

    expect(promptFired).toBe(true);
    expect(result).toBeUndefined();
  });
});

describe('Mid-flight preventContinuation halts the run after the current step', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('PostToolBatch hook with preventContinuation breaks the stream loop and skips Stop', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    let stopFired = false;
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          preventContinuation: true,
          stopReason: 'rate-limit policy halt',
        }),
      ],
    });
    registry.register('Stop', {
      hooks: [
        async (): Promise<Record<string, never>> => {
          stopFired = true;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('agent', () => ({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'echo', args: { command: 'x' } },
            ],
          }),
        ],
      }))
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'halt-mid-flight-1',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });
    /** Replace the SDK-built graph runnable with our handcrafted one so the
     * PostToolBatch hook fires under a real LangGraph stream. */
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'halt-thread-1' },
        version: 'v2',
      }
    );

    expect(run.getHaltReason()).toBe('rate-limit policy halt');
    expect(stopFired).toBe(false);
  });

  it('clears halt signal between processStream invocations', async () => {
    const registry = new HookRegistry();
    registry.register('RunStart', {
      hooks: [
        async (): Promise<RunStartHookOutput> => ({
          preventContinuation: true,
          stopReason: 'first run halted',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'halt-clear-1',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('first')] },
      { configurable: { thread_id: 't-1' }, version: 'v2' }
    );
    /** RunStart preventContinuation is a pre-stream early return, but
     * `processStream` should still have cleared the registry signal
     * for this run id so a subsequent call starts fresh. */
    expect(registry.getHaltSignal('halt-clear-1')).toBeUndefined();
  });
});

describe('Async fire-and-forget hooks ignore decision/context fields', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('PreToolUse with `async: true` does not block the tool even when decision is `deny`', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ran', status: 'success' },
    ]);

    let bgFired = false;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          /** Side effect runs in background; agent doesn't wait. */
          void Promise.resolve().then(() => {
            bgFired = true;
          });
          return {
            async: true,
            decision: 'deny',
            reason: 'this should be ignored',
            additionalContext: 'this should also be ignored',
          };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'async-1' } }
    )) as { messages: BaseMessage[] };

    const toolMsg = result.messages.find(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMsg).toBeDefined();
    /** Tool ran (no Blocked: prefix) — async output's `decision: 'deny'` was
     * ignored as documented. */
    expect(toolMsg!.status).not.toBe('error');
    expect(toolMsg!.content).toBe('ran');
    /** Background work runs even though we ignored the output. */
    await new Promise((r) => setImmediate(r));
    expect(bgFired).toBe(true);
    /** No injected context message — `additionalContext` was also ignored. */
    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeUndefined();
  });

  it('PostToolUse with `async: true` does not halt the run even when preventContinuation is set', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ran', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          async: true,
          preventContinuation: true,
          stopReason: 'should not halt',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'async-2' } }
    );

    /** preventContinuation was on an async output → ignored → no halt
     * signal raised under any session id. The standalone graph here
     * runs with `runId = ''` (no `config.configurable.run_id` set),
     * so check that key explicitly. */
    expect(registry.getHaltSignal('')).toBeUndefined();
  });
});

describe('Codex review fixes', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('reruns live hooks while replaying a consumed event-tool approval', async () => {
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          resolve: (results: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          { toolCallId: 'call_event', content: 'ran', status: 'success' },
        ]);
      });

    const registry = new HookRegistry();
    let onceCalls = 0;
    let liveCalls = 0;
    registry.register('PreToolUse', {
      once: true,
      pattern: '^echo$',
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          onceCalls += 1;
          return { decision: 'ask', reason: 'review once' };
        },
      ],
    });
    registry.register('PreToolUse', {
      pattern: '^echo$',
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          liveCalls += 1;
          return liveCalls === 1
            ? { decision: 'allow' }
            : { decision: 'deny', reason: 'policy changed before resume' };
        },
      ],
    });
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-event-replay',
      toolCallStepIds: new Map([['call_event', 'step_event']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_event', name: 'echo', args: { command: 'run' } },
    ]);
    const config = {
      configurable: { thread_id: 'thread-event-hook-replay' },
    };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted(interrupted)).toBe(true);
    const resumed = await graph.invoke(
      new Command({ resume: [{ type: 'approve' }] }),
      config
    );

    expect(onceCalls).toBe(1);
    expect(liveCalls).toBe(2);
    expect(dispatchCalls).toBe(0);
    const messages = (resumed as { messages: ToolMessage[] }).messages;
    const result = messages.find(
      (message) =>
        message instanceof ToolMessage && message.tool_call_id === 'call_event'
    );
    expect(result?.status).toBe('error');
    expect(String(result?.content)).toContain('policy changed before resume');
  });

  it('preserves session-scoped hooks across HITL interrupt so the policy still fires on resume', async () => {
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    let preCallCount = 0;
    /**
     * Register the policy hook against the runId via `registerSession`
     * (mirrors how a host scopes per-run policy without leaking it to
     * concurrent runs). The fix under test: this matcher MUST still be
     * present when `Run.resume()` re-runs the node so the policy
     * decision applies the second time too.
     */
    const runId = 'session-hook-preserve';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          preCallCount += 1;
          return { decision: 'ask', reason: 'session policy' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'session-thread-1' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);

    /** Interrupt fired; one hook invocation so far. Session matcher
     * MUST still be present — the regression was that finally cleared
     * it, leaving the resume to bypass the policy entirely. */
    expect(run.getInterrupt()).toBeDefined();
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'tool_approval',
      hook_session_id: runId,
    });
    expect(preCallCount).toBe(1);
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(true);
    expect(dispatchCalls).toBe(0);

    await run.resume([{ type: 'approve' }], callerConfig);

    /** Hook fired AGAIN on resume — policy was actually applied a
     * second time, not skipped. Tool then executed. */
    expect(preCallCount).toBe(2);
    expect(dispatchCalls).toBe(1);
    /** After natural completion, session matchers ARE cleared so the
     * next run on this registry starts clean. */
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('persists the parent hook session when a direct subagent approval interrupts', async () => {
    const runId = 'parent-subagent-policy';
    const executeSubagent = jest.fn(async () => 'child result');
    const subagentTool = tool(executeSubagent, {
      name: Constants.SUBAGENT,
      description: 'execute a child agent',
      schema: z.object({
        description: z.string(),
        subagent_type: z.string(),
      }),
    }) as unknown as StructuredToolInterface;
    const registry = new HookRegistry();
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'approve child delegation',
        }),
      ],
    });
    const node = new ToolNode({
      tools: [subagentTool],
      hookRegistry: registry,
      directToolNames: new Set([Constants.SUBAGENT]),
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      {
        id: 'call_subagent',
        name: Constants.SUBAGENT,
        args: { description: 'inspect logs', subagent_type: 'researcher' },
      },
    ]);

    const interrupted = await graph.invoke(
      { messages: [] },
      {
        configurable: {
          thread_id: 'parent-subagent-thread',
          run_id: runId,
        },
      }
    );

    if (!isInterrupted<t.ToolApprovalInterruptPayload>(interrupted)) {
      throw new Error('expected a subagent approval interrupt');
    }
    expect(interrupted.__interrupt__[0].value).toMatchObject({
      type: 'tool_approval',
      hook_session_id: runId,
      action_requests: [
        { tool_call_id: 'call_subagent', name: Constants.SUBAGENT },
      ],
    });
    expect(executeSubagent).not.toHaveBeenCalled();
  });

  it('denied tool in a deny+ask batch dispatches ON_RUN_STEP_COMPLETED exactly once across interrupt + resume', async () => {
    const stepCompletedDispatches: string[] = [];
    /** Spy on the underlying custom event dispatcher to capture every
     * ON_RUN_STEP_COMPLETED event with its tool_call_id. Without the
     * blockEntry deferral, this would record `call_a` twice for one
     * logical denial (once before interrupt, once after resume
     * re-execution). */
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as {
            result?: { tool_call?: { id?: string } };
          };
          const id = payload.result?.tool_call?.id;
          if (id != null) {
            stepCompletedDispatches.push(id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (input.toolName === 'tool_a') {
            return { decision: 'deny', reason: 'policy:a' };
          }
          return { decision: 'ask', reason: 'policy:b-needs-review' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('tool_a'), createSchemaStub('tool_b')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_a', 'step_a'],
        ['call_b', 'step_b'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_a', name: 'tool_a', args: { command: 'a' } },
      { id: 'call_b', name: 'tool_b', args: { command: 'b' } },
    ]);
    const config = { configurable: { thread_id: 'dedup-thread' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    /** First pass: interrupt() threw, so the deferred denial side
     * effects were not flushed. Zero step-completed events for the
     * denied tool yet. */
    expect(stepCompletedDispatches.filter((id) => id === 'call_a')).toEqual([]);

    await resumeGraph(graph, interrupted, [{ type: 'approve' }], config);

    /** After resume: the denied tool dispatches exactly once (deferred
     * flush on the resume re-execution); the approved tool dispatches
     * once via the normal execution path. */
    expect(stepCompletedDispatches.filter((id) => id === 'call_a')).toEqual([
      'call_a',
    ]);
    expect(stepCompletedDispatches.filter((id) => id === 'call_b')).toEqual([
      'call_b',
    ]);
  });

  it('enforces allowedDecisions on resume — host-submitted decision outside the allowlist is rejected', async () => {
    const dispatchedToolNames: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedToolNames.push(c.name);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    /** Hook restricts to approve/reject only — edit/respond are
     * forbidden. Even if a buggy or hostile host UI submits an
     * `edit`, the SDK must fail closed instead of mutating the args
     * and running the tool. */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'allowed-enforce' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** Submit `edit` — outside the advertised allowlist. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'edit', updatedInput: { command: 'malicious' } }],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    /** Tool was blocked; arg-mutation never reached the host. */
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'not in allowedDecisions'
    );
    expect(String(toolMessages[0].content)).toContain('approve');
    expect(String(toolMessages[0].content)).toContain('reject');
    expect(dispatchedToolNames).toEqual([]);
  });

  it('enforces allowedDecisions on resume — approved decision passes through when in the allowlist', async () => {
    const dispatchedArgs: Array<Record<string, unknown>> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedArgs.push(c.args);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'allowed-pass' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** Submit `approve` — explicitly in the allowlist. */
    await resumeGraph(graph, interrupted, [{ type: 'approve' }], config);

    expect(dispatchedArgs).toEqual([{ command: 'original' }]);
  });

  it('getInterrupt<T>() returns the captured payload typed as the host-asserted shape', async () => {
    /**
     * Custom graph node raises an interrupt with a payload shape the
     * SDK doesn't know about. `run.getInterrupt<MyCustomPayload>()`
     * returns the payload typed as the host's assertion — the SDK
     * doesn't validate, it just transports.
     */
    interface MyCustomPayload {
      type: 'custom_review';
      diff: string;
      reviewerHints: string[];
    }

    const langgraph = await import('@langchain/langgraph');

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('clarifier', () => {
        langgraph.interrupt({
          type: 'custom_review',
          diff: '+ added line',
          reviewerHints: ['check formatting'],
        } satisfies MyCustomPayload);
        return { messages: [] };
      })
      .addEdge(START, 'clarifier')
      .addEdge('clarifier', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'custom-interrupt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'custom-interrupt-thread' },
        version: 'v2',
      }
    );

    const interrupt = run.getInterrupt<MyCustomPayload>();
    expect(interrupt).toBeDefined();
    expect(interrupt!.payload.type).toBe('custom_review');
    expect(interrupt!.payload.diff).toBe('+ added line');
    expect(interrupt!.payload.reviewerHints).toEqual(['check formatting']);
  });

  it('isToolApprovalInterrupt / isAskUserQuestionInterrupt narrow safely from `unknown` (defensive)', async () => {
    const { isToolApprovalInterrupt, isAskUserQuestionInterrupt } =
      await import('@/types/hitl');

    /** The guards must accept arbitrary runtime values without throwing,
     * since hosts can pass anything from custom interrupts. */
    expect(isToolApprovalInterrupt(null as unknown)).toBe(false);
    expect(isToolApprovalInterrupt(undefined as unknown)).toBe(false);
    expect(isToolApprovalInterrupt('string' as unknown)).toBe(false);
    expect(isToolApprovalInterrupt(42 as unknown)).toBe(false);
    expect(isToolApprovalInterrupt({} as unknown)).toBe(false);
    expect(isToolApprovalInterrupt({ type: 'something_else' } as unknown)).toBe(
      false
    );
    expect(
      isToolApprovalInterrupt({
        type: 'tool_approval',
        action_requests: [],
        review_configs: [],
      } as unknown)
    ).toBe(true);

    expect(isAskUserQuestionInterrupt(null as unknown)).toBe(false);
    expect(
      isAskUserQuestionInterrupt({ type: 'tool_approval' } as unknown)
    ).toBe(false);
    expect(
      isAskUserQuestionInterrupt({
        type: 'ask_user_question',
        question: { question: 'why' },
      } as unknown)
    ).toBe(true);
  });

  it('hook returning ask + updatedInput rewrites args BEFORE the interrupt and BEFORE host execution', async () => {
    const dispatchedArgs: Array<Record<string, unknown>> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedArgs.push(c.args);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    /**
     * Hook returns BOTH a sanitization rewrite AND `ask`. Real-world
     * pattern: one matcher redacts secrets in the args, another
     * matcher requires human approval. Both signals must apply.
     */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review redacted args',
          updatedInput: { command: 'redacted-command' },
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original-secret' } },
    ]);
    const config = { configurable: { thread_id: 'ask-with-update' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }
    /** The interrupt payload surfaces the REWRITTEN args to the
     * reviewer, not the original. Without the fix, the reviewer
     * would see the secret. */
    expect(payload.action_requests[0].arguments).toEqual({
      command: 'redacted-command',
    });

    await resumeGraph(graph, interrupted, [{ type: 'approve' }], config);

    /** And the host execution dispatches the rewritten args, not
     * the original. Without the fix, the policy redaction would be
     * silently dropped after approval. */
    expect(dispatchedArgs).toEqual([{ command: 'redacted-command' }]);
  });

  it.each([
    {
      label: 'rejection',
      decision: { type: 'reject', reason: 'not approved' } as const,
      expectedDispatches: 0,
    },
    {
      label: 'approval',
      decision: { type: 'approve' } as const,
      expectedDispatches: 1,
    },
  ])(
    'Run.resume preserves a reviewed $label when a reusable hook later loosens ask to allow',
    async ({ label, decision, expectedDispatches }) => {
      let dispatchCalls = 0;
      const hostConfigurables: Array<Record<string, unknown> | undefined> = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          dispatchCalls += 1;
          const request = data as {
            toolCalls: t.ToolCallRequest[];
            configurable?: Record<string, unknown>;
            resolve: (results: t.ToolExecuteResult[]) => void;
          };
          hostConfigurables.push(request.configurable);
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              content: 'ran',
              status: 'success' as const,
            }))
          );
        });

      let hookCalls = 0;
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (): Promise<PreToolUseHookOutput> => {
            hookCalls += 1;
            return hookCalls === 1
              ? { decision: 'ask', reason: 'review this command' }
              : { decision: 'allow' };
          },
        ],
      });

      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'a',
        toolCallStepIds: new Map([['call_1', 'step_1']]),
        hookRegistry: registry,
        humanInTheLoop: { enabled: true },
      });
      const graph = buildHITLGraph(node, [
        { id: 'call_1', name: 'echo', args: { command: 'reviewed' } },
      ]);

      const { Run } = await import('@/run');
      const run = await Run.create<t.IState>({
        runId: `reviewed-decision-survives-policy-replay-${label}`,
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: providers.OPENAI,
              clientOptions: {
                modelName: 'gpt-4o-mini',
                apiKey: 'test-key',
              },
              instructions: 'noop',
              maxContextTokens: 8000,
            },
          ],
        },
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
      run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
      const config = {
        configurable: { thread_id: `reviewed-decision-thread-${label}` },
        version: 'v2' as const,
      };

      await run.processStream({ messages: [] }, config);
      expect(run.getInterrupt()?.payload).toMatchObject({
        type: 'tool_approval',
        action_requests: [
          {
            tool_call_id: 'call_1',
            name: 'echo',
            arguments: { command: 'reviewed' },
          },
        ],
      });

      await run.resume([decision], config);

      expect(hookCalls).toBe(2);
      expect(dispatchCalls).toBe(expectedDispatches);
      expect(
        hostConfigurables.every(
          (configurable) =>
            configurable?.__librechat_tool_approval_review == null
        )
      ).toBe(true);
    }
  );

  it('fails closed for a direct sibling missing from restored review evidence', async () => {
    const executed: string[] = [];
    const tools = ['first_direct', 'second_direct'].map(
      (name) =>
        tool(
          async (): Promise<string> => {
            executed.push(name);
            return 'ran';
          },
          {
            name,
            description: `${name} test tool`,
            schema: z.object({ command: z.string() }),
          }
        ) as unknown as StructuredToolInterface
    );
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review direct command',
        }),
      ],
    });
    const node = new ToolNode({
      tools,
      eventDrivenMode: true,
      directToolNames: new Set(tools.map((entry) => entry.name)),
      agentId: 'a',
      toolCallStepIds: new Map([
        ['call_1', 'step_1'],
        ['call_2', 'step_2'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'first_direct', args: { command: 'first' } },
      { id: 'call_2', name: 'second_direct', args: { command: 'second' } },
    ]);
    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'parallel-reviewed-directs',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: { thread_id: 'parallel-reviewed-directs' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    const interrupt = run.getInterrupt();
    const reviewedName =
      interrupt?.payload.type === 'tool_approval'
        ? interrupt.payload.action_requests[0]?.name
        : undefined;
    expect(reviewedName).toBeDefined();
    (node as unknown as { hookRegistry?: HookRegistry }).hookRegistry =
      undefined;

    await run.resume([{ type: 'approve' }], config);

    expect(executed).toEqual([reviewedName]);
  });

  it('waits for a started sibling before checkpointing an approval pause', async () => {
    let release: () => void = () => { throw new Error('not initialized'); };
    const barrier = new Promise<void>((resolve) => { release = resolve; });
    let executions = 0;
    const slow = tool(async () => {
      await barrier;
      executions += 1;
      return 'completed';
    }, { name: 'slow', description: 'controlled side effect', schema: z.object({}) });
    const ask = tool(async () => 'approved', { name: 'ask', description: 'approval', schema: z.object({}) });
    const registry = new HookRegistry();
    registry.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'ask' ? 'ask' : 'allow' })] });
    const node = new ToolNode({ tools: [ask, slow], directToolNames: new Set(['ask', 'slow']),
      hookRegistry: registry, humanInTheLoop: { enabled: true } });
    const graph = buildHITLGraph(node, [
      { id: 'ask_call', name: 'ask', args: {} }, { id: 'slow_call', name: 'slow', args: {} },
    ]);
    let paused = false;
    const pending = graph.invoke({ messages: [] }, { configurable: { thread_id: 'settle-before-pause' } })
      .then((result) => { paused = true; return result; });
    await flushAsyncWork();
    expect(paused).toBe(false);
    expect(executions).toBe(0);
    release();
    const interrupted = await pending;
    expect(executions).toBe(1);
    expect(isInterrupted(interrupted)).toBe(true);
    if (!isInterrupted<unknown>(interrupted)) {
      throw new Error('Expected approval pause');
    }
    const state = getToolBatchReplayState(interrupted.__interrupt__[0].value);
    expect(state?.records).toHaveLength(1);
  });

  it.each([
    { fresh: false, mutate: false, append: false }, { fresh: true, mutate: false, append: false }, { fresh: true, mutate: true, append: false },
    { fresh: true, mutate: false, append: true },
  ])('reuses a settled direct sibling when a later event tool interrupts (fresh=$fresh, mutate=$mutate, append=$append)', async ({ fresh, mutate, append }) => {
    let directExecutions = 0;
    const completedCallIds: string[] = [];
    const dispatchedEventNames: string[] = [];
    const directTool = tool(
      async (): Promise<string> => {
        directExecutions += 1;
        return 'direct-result';
      },
      {
        name: 'local_direct',
        description: 'local direct test tool',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const completed = data as {
            result?: { tool_call?: { id?: string } };
          };
          const id = completed.result?.tool_call?.id;
          if (id != null) {
            completedCallIds.push(id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (results: t.ToolExecuteResult[]) => void;
        };
        dispatchedEventNames.push(
          ...request.toolCalls.map((call) => call.name)
        );
        request.resolve(
          request.toolCalls.map((call) => ({
            toolCallId: call.id,
            content: 'event-result',
            status: 'success' as const,
          }))
        );
      });
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> =>
          input.toolName === 'remote_event'
            ? { decision: 'ask', reason: 'review event tool' }
            : { decision: 'allow' },
      ],
    });
    const createNode = () => new ToolNode({
      tools: [directTool, createSchemaStub('remote_event')],
      toolMap: new Map([
        ['local_direct', directTool],
        ['remote_event', createSchemaStub('remote_event')],
      ]),
      eventDrivenMode: true,
      directToolNames: new Set(['local_direct']),
      agentId: 'a',
      toolCallStepIds: new Map([
        ['call_direct', 'step_direct'],
        ['call_event', 'step_event'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const checkpointer = new MemorySaver();
    const calls = [
      {
        id: 'call_direct',
        name: 'local_direct',
        args: { command: 'local' },
      },
      {
        id: 'call_event',
        name: 'remote_event',
        args: { command: 'remote' },
      },
    ];
    const graph = buildHITLGraph(createNode(), calls, checkpointer);
    const config = { configurable: { thread_id: 'mixed-direct-event-replay' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(directExecutions).toBe(1);
    expect(completedCallIds.filter((id) => id === 'call_direct')).toHaveLength(
      1
    );

    let resumed: { messages: BaseMessage[] };
    if (fresh) {
      const restoredGraph = buildHITLGraph(createNode(), calls, checkpointer);
      const { Run } = await import('@/run');
      const restoredRun = await Run.create<t.IState>({
        runId: 'a-different-process-run-id',
        graphConfig: {
          type: 'standard',
          agents: [{
            agentId: 'a', provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop', maxContextTokens: 8000,
          }],
          compileOptions: { checkpointer },
        },
        humanInTheLoop: { enabled: true },
      });
      restoredRun.graphRunnable = restoredGraph as unknown as t.CompiledStateWorkflow;
      if (mutate) {
        const checkpoint = await checkpointer.getTuple(config);
        const messages = checkpoint?.checkpoint.channel_values.messages as BaseMessage[];
        const original = messages.find((message) => message.getType() === 'ai');
        await expect(restoredRun.resume([{ type: 'approve' }], { ...config, version: 'v2' }, undefined, {
          update: { messages: [new AIMessage({
            id: original?.id, content: '', tool_calls: calls.map((call) => call.id === 'call_direct' ?
              { ...call, args: { command: 'different operation' } } : call),
          })] },
        })).rejects.toThrow('Tool batch proposal changed before approval replay');
        expect(directExecutions).toBe(1);
        expect(dispatchedEventNames).toEqual([]);
        return;
      }
      await restoredRun.resume([{ type: 'approve' }], { ...config, version: 'v2' }, undefined,
        append ? { update: { messages: [new HumanMessage('Additional instructions during approval')] } } : undefined);
      await restoredRun.resume([{ type: 'approve' }], { ...config, version: 'v2' });
      const state = await restoredGraph.getState?.(config);
      resumed = (state as { values: { messages: BaseMessage[] } }).values;
    } else {
      resumed = await resumeGraph(graph, interrupted, [{ type: 'approve' }], config) as { messages: BaseMessage[] };
    }

    expect(directExecutions).toBe(1);
    expect(completedCallIds.filter((id) => id === 'call_direct')).toHaveLength(
      1
    );
    expect(dispatchedEventNames).toEqual(['remote_event']);
    const directResult = resumed.messages.find(
      (message): message is ToolMessage =>
        message._getType() === 'tool' &&
        (message as ToolMessage).tool_call_id === 'call_direct'
    );
    expect(directResult?.status).not.toBe('error');
    expect(directResult?.content).toBe('direct-result');
  });

  it('executes the exact resolved direct arguments that were reviewed', async () => {
    const executedCommands: string[] = [];
    const directTool = tool(
      async ({ command }): Promise<string> => {
        executedCommands.push(command);
        return 'ran';
      },
      {
        name: 'reference_direct',
        description: 'reference test tool',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;
    const registry = makeHookRegistry('ask', 'review resolved command');
    const node = new ToolNode({
      tools: [directTool],
      eventDrivenMode: true,
      directToolNames: new Set(['reference_direct']),
      agentId: 'a',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
      toolOutputReferences: { enabled: true },
    });
    const referenceRegistry = (
      node as unknown as {
        toolOutputRegistry?: ToolOutputReferenceRegistry;
      }
    ).toolOutputRegistry;
    referenceRegistry?.set(
      'reviewed-reference-direct',
      'tool0turn0',
      'resolved-value'
    );
    const graph = buildHITLGraph(node, [
      {
        id: 'call_1',
        name: 'reference_direct',
        args: { command: 'echo {{tool0turn0}}' },
      },
    ]);
    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'reviewed-reference-direct',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
      toolOutputReferences: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: {
        thread_id: 'reviewed-reference-direct',
        run_id: 'reviewed-reference-direct',
      },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    const interrupt = run.getInterrupt();
    const reviewedArguments =
      interrupt?.payload.type === 'tool_approval'
        ? interrupt.payload.action_requests[0]?.arguments
        : undefined;
    expect(reviewedArguments).toEqual({
      command: 'echo resolved-value',
    });
    (
      node as unknown as {
        toolOutputRegistry?: ToolOutputReferenceRegistry;
      }
    ).toolOutputRegistry = new ToolOutputReferenceRegistry();

    await run.resume([{ type: 'approve' }], config);

    expect(executedCommands).toEqual(['echo resolved-value']);
  });

  it('replays a direct hook rewrite from the original arguments', async () => {
    const hookInputs: string[] = [];
    const executedCommands: string[] = [];
    const directTool = tool(
      async ({ command }): Promise<string> => {
        executedCommands.push(command);
        return 'ran';
      },
      {
        name: 'rewritten_direct',
        description: 'rewrite replay test tool',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          const command = String(input.toolInput.command);
          hookInputs.push(command);
          return {
            decision: 'ask',
            reason: 'review rewritten command',
            updatedInput: { command: `safe:${command}` },
          };
        },
      ],
    });
    const node = new ToolNode({
      tools: [directTool],
      eventDrivenMode: true,
      directToolNames: new Set(['rewritten_direct']),
      agentId: 'a',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      {
        id: 'call_1',
        name: 'rewritten_direct',
        args: { command: 'original' },
      },
    ]);
    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'direct-rewrite-replay',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: {
              modelName: 'gpt-4o-mini',
              apiKey: 'test-key',
            },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: { thread_id: 'direct-rewrite-replay' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    await run.resume([{ type: 'approve' }], config);

    expect(hookInputs).toEqual(['original', 'original']);
    expect(executedCommands).toEqual(['safe:original']);
  });

  it('does not let a host mutation change the reviewed direct proposal', async () => {
    const executedCommands: string[] = [];
    const directTool = tool(
      async ({ command }): Promise<string> => {
        executedCommands.push(command);
        return 'ran';
      },
      {
        name: 'immutable_direct',
        description: 'immutable proposal test tool',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;
    const registry = makeHookRegistry('ask', 'review immutable command');
    const node = new ToolNode({
      tools: [directTool],
      eventDrivenMode: true,
      directToolNames: new Set(['immutable_direct']),
      agentId: 'a',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      {
        id: 'call_1',
        name: 'immutable_direct',
        args: { command: 'reviewed' },
      },
    ]);
    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'immutable-direct-proposal',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: {
              modelName: 'gpt-4o-mini',
              apiKey: 'test-key',
            },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: { thread_id: 'immutable-direct-proposal' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    const exposed = run.getInterrupt();
    if (exposed?.payload.type !== 'tool_approval') {
      throw new Error('expected tool approval');
    }
    exposed.payload.action_requests[0].arguments.command = 'mutated';
    await run.resume([{ type: 'approve' }], config);

    expect(executedCommands).toEqual(['reviewed']);
  });

  it('Run.resume fails closed when a reusable hook changes the reviewed arguments', async () => {
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (results: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((call) => ({
            toolCallId: call.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    let hookCalls = 0;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          hookCalls += 1;
          return {
            decision: 'ask',
            reason: 'review this command',
            updatedInput: {
              command: hookCalls === 1 ? 'reviewed' : 'changed-after-review',
            },
          };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'a',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'reviewed-argument-integrity',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: {
              modelName: 'gpt-4o-mini',
              apiKey: 'test-key',
            },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: { thread_id: 'reviewed-argument-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    await run.resume([{ type: 'approve' }], config);

    expect(hookCalls).toBe(2);
    expect(dispatchCalls).toBe(0);
  });

  it('Run.resume preserves the reviewed decision for a direct tool when policy replay loosens', async () => {
    const executedCommands: string[] = [];
    const directTool = tool(
      async ({ command }): Promise<string> => {
        executedCommands.push(command);
        return 'ran';
      },
      {
        name: 'direct_echo',
        description: 'direct test tool',
        schema: z.object({ command: z.string() }),
      }
    ) as unknown as StructuredToolInterface;

    let hookCalls = 0;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          hookCalls += 1;
          return hookCalls === 1
            ? { decision: 'ask', reason: 'review direct command' }
            : { decision: 'allow' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [directTool],
      eventDrivenMode: true,
      directToolNames: new Set(['direct_echo']),
      agentId: 'a',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });
    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'direct_echo', args: { command: 'reviewed' } },
    ]);

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'reviewed-direct-rejection',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: {
              modelName: 'gpt-4o-mini',
              apiKey: 'test-key',
            },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
    const config = {
      configurable: { thread_id: 'reviewed-direct-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, config);
    await run.resume([{ type: 'reject', reason: 'not approved' }], config);

    expect(hookCalls).toBe(2);
    expect(executedCommands).toEqual([]);
  });

  it.each([
    { label: 'event-dispatched', direct: false },
    { label: 'direct', direct: true },
  ])(
    'Run.resume honors a reviewed rejection for a $label tool after its hook registry is removed',
    async ({ direct }) => {
      let executions = 0;
      const echoTool = tool(
        async (): Promise<string> => {
          executions += 1;
          return 'ran';
        },
        {
          name: 'echo',
          description: 'test tool',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;
      if (!direct) {
        jest
          .spyOn(events, 'safeDispatchCustomEvent')
          .mockImplementation(async (event, data) => {
            if (event !== 'on_tool_execute') {
              return;
            }
            executions += 1;
            const request = data as {
              toolCalls: t.ToolCallRequest[];
              resolve: (results: t.ToolExecuteResult[]) => void;
            };
            request.resolve(
              request.toolCalls.map((call) => ({
                toolCallId: call.id,
                content: 'ran',
                status: 'success' as const,
              }))
            );
          });
      }

      let hookCalls = 0;
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (): Promise<PreToolUseHookOutput> => {
            hookCalls += 1;
            return { decision: 'ask', reason: 'review this command' };
          },
        ],
      });
      const node = new ToolNode({
        tools: [echoTool],
        eventDrivenMode: true,
        ...(direct ? { directToolNames: new Set(['echo']) } : {}),
        agentId: 'a',
        toolCallStepIds: new Map([['call_1', 'step_1']]),
        hookRegistry: registry,
        humanInTheLoop: { enabled: true },
      });
      const graph = buildHITLGraph(node, [
        { id: 'call_1', name: 'echo', args: { command: 'reviewed' } },
      ]);
      const { Run } = await import('@/run');
      const run = await Run.create<t.IState>({
        runId: `removed-hook-registry-${direct ? 'direct' : 'event'}`,
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: providers.OPENAI,
              clientOptions: {
                modelName: 'gpt-4o-mini',
                apiKey: 'test-key',
              },
              instructions: 'noop',
              maxContextTokens: 8000,
            },
          ],
        },
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
      run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
      const config = {
        configurable: {
          thread_id: `removed-hook-registry-${direct ? 'direct' : 'event'}`,
        },
        version: 'v2' as const,
      };

      await run.processStream({ messages: [] }, config);
      expect(run.getInterrupt()?.payload).toMatchObject({
        type: 'tool_approval',
      });
      (node as unknown as { hookRegistry?: HookRegistry }).hookRegistry =
        undefined;

      await run.resume([{ type: 'reject', reason: 'not approved' }], config);

      expect(hookCalls).toBe(1);
      expect(executions).toBe(0);
    }
  );

  it.each([
    { direct: true, changedOwner: false, replaceIds: false }, { direct: false, changedOwner: false, replaceIds: false },
    { direct: true, changedOwner: true, replaceIds: false }, { direct: false, changedOwner: true, replaceIds: false },
    { direct: true, changedOwner: true, replaceIds: true }, { direct: false, changedOwner: true, replaceIds: true },
  ])(
    'restores a rewritten approval with fresh instances (direct=$direct, changedOwner=$changedOwner, replaceIds=$replaceIds)',
    async ({ direct, changedOwner, replaceIds }) => {
      const checkpointer = new MemorySaver();
      const executed: string[] = [];
      const echo = tool(async ({ command }) => {
        executed.push(command);
        return command;
      }, {
        name: 'echo',
        description: 'echo a reviewed command',
        schema: z.object({ command: z.string() }),
      });
      if (!direct) {
        jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const request = data as {
            toolCalls: t.ToolCallRequest[];
            resolve: (results: t.ToolExecuteResult[]) => void;
          };
          request.resolve(request.toolCalls.map((call) => {
            const command = String(call.args.command);
            executed.push(command);
            return { toolCallId: call.id, content: command, status: 'success' };
          }));
        });
      }
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [async () => ({ decision: 'ask', updatedInput: { command: 'reviewed' } })],
      });
      const { Run } = await import('@/run');
      const create = async (hooks?: HookRegistry, agentId = 'a') => {
        const node = new ToolNode({
          tools: [echo],
          eventDrivenMode: true,
          ...(direct ? { directToolNames: new Set(['echo']) } : {}),
          agentId,
          toolCallStepIds: new Map([['call_1', 'step_1']]),
          hookRegistry: hooks,
          humanInTheLoop: { enabled: true },
        });
        const run = await Run.create<t.IState>({
          runId: `fresh-approval-${direct}`,
          graphConfig: {
            type: 'standard',
            agents: [{
              agentId: 'a', provider: providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'noop', maxContextTokens: 8000,
            }],
            compileOptions: { checkpointer },
          },
          hooks,
          humanInTheLoop: { enabled: true },
        });
        run.graphRunnable = buildHITLGraph(node, [
          { id: 'call_1', name: 'echo', args: { command: 'original' } },
        ], checkpointer) as unknown as t.CompiledStateWorkflow;
        return run;
      };
      const config = { configurable: { thread_id: `fresh-approval-${direct}` }, version: 'v2' as const };
      const original = await create(registry);
      await original.processStream({ messages: [] }, config);
      expect(original.getInterrupt()?.payload).toMatchObject({
        action_requests: [{ arguments: { command: 'reviewed' } }],
      });
      expect(executed).toEqual([]);
      const restored = await create(undefined, changedOwner ? 'different-agent' : 'a');
      if (changedOwner) {
        const tuple = await checkpointer.getTuple(config);
        const messages = tuple?.checkpoint.channel_values.messages as BaseMessage[];
        const assistant = messages.find((message) => message.getType() === 'ai');
        await expect(restored.resume([{ type: 'reject' }], config, undefined, replaceIds ? {
          update: { messages: [new AIMessage({ id: assistant?.id, content: '', tool_calls: [
            { id: 'replacement', name: 'echo', args: { command: 'unreviewed' } },
          ] })] },
        } : undefined)).rejects.toThrow('Tool approval execution owner changed');
        expect(executed).toEqual([]);
        return;
      }
      await restored.resume([{ type: 'approve' }], config);
      expect(executed).toEqual(['reviewed']);
    }
  );

  it.each([false, true])('preserves ID-less successes beside a rejected approval (mixed=%s)', async (mixed) => {
    let executions = 0;
    const work = tool(async () => { executions += 1; return `completed-${executions}`; }, {
      name: 'work', description: 'work', schema: z.object({}),
    });
    const ask = tool(async () => 'approved', { name: 'ask', description: 'ask', schema: z.object({}) });
    const checkpointer = new MemorySaver();
    const registry = new HookRegistry();
    registry.register('PreToolUse', { hooks: [async (input) => ({ decision: input.toolName === 'ask' ? 'ask' : 'allow' })] });
    const { Run } = await import('@/run');
    const makeRun = async (hooks?: HookRegistry) => {
      const run = await Run.create<t.IState>({
        runId: `idless-${mixed}`,
        graphConfig: { type: 'standard', agents: [{
          agentId: 'a', provider: providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'noop', maxContextTokens: 8000,
        }], compileOptions: { checkpointer } },
        humanInTheLoop: { enabled: true }, hooks,
      });
      const graph = buildHITLGraph(new ToolNode({
        tools: [work, ask], agentId: 'a', eventDrivenMode: mixed,
        directToolNames: new Set(['work']), hookRegistry: hooks,
        toolCallStepIds: new Map([['ask', 'ask-step']]),
        humanInTheLoop: { enabled: true },
      }), [{ name: 'work', args: {} }, { name: 'work', args: {} }, { id: 'ask', name: 'ask', args: {} }], checkpointer);
      run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;
      return { run, graph };
    };
    const config = { configurable: { thread_id: `idless-${mixed}` }, version: 'v2' as const };
    const original = await makeRun(registry);
    await original.run.processStream({ messages: [] }, config);
    expect(original.run.getInterrupt()?.payload).toMatchObject({ type: 'tool_approval' });
    expect(executions).toBe(2);
    const fresh = await makeRun();
    await fresh.run.resume([{ type: 'reject' }], config);
    const result = await fresh.graph.getState?.(config);
    expect(executions).toBe(2);
    expect(JSON.stringify(result)).toContain('completed-1');
    expect(JSON.stringify(result)).toContain('completed-2');
  });

  it('forwards child review evidence without blocking an unrelated parent sibling', async () => {
    const receivedConfigurables: Array<Record<string, unknown> | undefined> =
      [];
    const subagentTool = tool(
      async (_input, config): Promise<string> => {
        receivedConfigurables.push(config.configurable);
        return 'child result';
      },
      {
        name: Constants.SUBAGENT,
        description: 'execute a child agent',
        schema: z.object({
          description: z.string(),
          subagent_type: z.string(),
        }),
      }
    ) as unknown as StructuredToolInterface & ReplayableSubagentTool;
    subagentTool[SUBAGENT_REPLAY_CONTROLLER] = {
      getSettledOutput: async () => undefined,
      persistSettledOutput: async () => undefined,
    };
    let siblingExecutions = 0;
    const sibling = tool(async () => {
      siblingExecutions += 1;
      return 'parent result';
    }, { name: 'parent_sibling', description: 'parent sibling', schema: z.object({}) });
    const node = new ToolNode({
      tools: [subagentTool, sibling],
      directToolNames: new Set([Constants.SUBAGENT, 'parent_sibling']),
      interruptingToolNames: new Set([Constants.SUBAGENT]),
    });
    const graph = buildHITLGraph(node, [
      {
        id: 'parent_call',
        name: Constants.SUBAGENT,
        args: { description: 'inspect logs', subagent_type: 'researcher' },
      },
      { id: 'parent_sibling_call', name: 'parent_sibling', args: {} },
    ]);
    const reviewEvidence = createToolApprovalReviewEvidence('interrupt_1', {
      type: 'tool_approval',
      action_requests: [
        {
          tool_call_id: 'child_call',
          name: 'bash',
          arguments: { command: 'git status' },
        },
      ],
      review_configs: [
        {
          tool_call_id: 'child_call',
          action_name: 'bash',
          allowed_decisions: ['approve', 'reject'],
        },
      ],
    });
    expect(reviewEvidence).toBeDefined();

    await graph.invoke(
      { messages: [] },
      {
        configurable: {
          thread_id: 'nested-review-evidence',
          [TOOL_APPROVAL_REVIEW_CONFIG_KEY]: reviewEvidence,
        },
      }
    );

    expect(receivedConfigurables).toHaveLength(1);
    expect(siblingExecutions).toBe(1);
    expect(receivedConfigurables[0]?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY]).toBe(
      reviewEvidence
    );
  });

  it('captures interrupt even when payload is null (custom node calling interrupt(null))', async () => {
    const langgraph = await import('@langchain/langgraph');

    let stopFired = false;
    const registry = new HookRegistry();
    registry.register('Stop', {
      hooks: [
        async (): Promise<Record<string, never>> => {
          stopFired = true;
          return {};
        },
      ],
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('pauser', () => {
        /** Custom node pauses without payload — valid use case (the
         * pause itself is the signal; no metadata needed). */
        langgraph.interrupt(null);
        return { messages: [] };
      })
      .addEdge(START, 'pauser')
      .addEdge('pauser', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'null-payload-interrupt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'null-payload-thread' },
        version: 'v2',
      }
    );

    /** Run was paused, NOT completed — getInterrupt returns a result
     * (with the null payload preserved) and the Stop hook does not
     * fire. Without the fix, both inversions held. */
    const interrupt = run.getInterrupt<unknown>();
    expect(interrupt).toBeDefined();
    expect(interrupt!.payload).toBeNull();
    expect(stopFired).toBe(false);
  });

  it('halt signal raised by run A does not bleed into a concurrent run B sharing the same registry', async () => {
    /**
     * One registry, two runs. RunStart hook for run A raises
     * preventContinuation; run B has no halt signal. Without
     * per-session scoping, run B's stream-loop poll would see A's
     * signal and silently terminate. With scoping, each run reads
     * only its own halt entry.
     */
    const registry = new HookRegistry();
    let runStartFires = 0;
    registry.register('RunStart', {
      hooks: [
        async (input): Promise<RunStartHookOutput> => {
          runStartFires += 1;
          /** Halt only run A, not run B. */
          if (input.runId === 'run-a') {
            return {
              preventContinuation: true,
              stopReason: 'A halted',
            };
          }
          return {};
        },
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    /** No-op graph so we never hit the real model. */
    const makeNoopGraph = (): t.CompiledStateWorkflow => {
      const builder = new StateGraph(MessagesAnnotation)
        .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
        .addEdge(START, 'noop')
        .addEdge('noop', END);
      return builder.compile() as unknown as t.CompiledStateWorkflow;
    };

    const makeRun = async (
      runId: string
    ): Promise<Awaited<ReturnType<typeof Run.create<t.IState>>>> => {
      const r = await Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'noop',
              maxContextTokens: 8000,
            },
          ],
        },
        hooks: registry,
        humanInTheLoop: { enabled: false },
      });
      r.graphRunnable = makeNoopGraph();
      return r;
    };

    const runA = await makeRun('run-a');
    const runB = await makeRun('run-b');

    /** Run A — its preventContinuation lands in the per-session halt
     * map under key `'run-a'` and triggers a pre-stream early
     * return. Note that the early-return path also clears its own
     * halt signal in the same step, so run B can never observe it
     * even momentarily. */
    await runA.processStream(
      { messages: [new HM('a')] },
      { configurable: { thread_id: 'thread-a' }, version: 'v2' }
    );
    expect(runA.getHaltReason()).toBe('A halted');

    /** Run B's signal must be undefined — A's halt is scoped to A's
     * session id, and was cleared in A's pre-stream finally path. */
    expect(registry.getHaltSignal('run-b')).toBeUndefined();
    expect(registry.getHaltSignal('run-a')).toBeUndefined();

    /** Run B — RunStart returns no halt, so processStream proceeds
     * past the pre-stream gate, executes the no-op graph, and
     * completes without halt. */
    runStartFires = 0;
    await runB.processStream(
      { messages: [new HM('b')] },
      { configurable: { thread_id: 'thread-b' }, version: 'v2' }
    );
    expect(runStartFires).toBe(1);
    expect(runB.getHaltReason()).toBeUndefined();
  });

  it('review_configs entries carry tool_call_id so duplicate-tool batches map unambiguously', async () => {
    mockEventDispatch([]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    /** Same tool name called twice in one batch — by-position
     * mapping breaks down for hosts that reorder; tool_call_id
     * lets the UI map review_configs → action_requests directly. */
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_first', 'step_first'],
        ['call_second', 'step_second'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_first', name: 'echo', args: { command: 'a' } },
      { id: 'call_second', name: 'echo', args: { command: 'b' } },
    ]);
    const config = { configurable: { thread_id: 'duplicate-tool' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }

    /** Each review_config carries its own tool_call_id matching the
     * action_request at the same index. UI can build a Map keyed by
     * tool_call_id rather than relying on positional order. */
    expect(payload.review_configs).toEqual([
      {
        action_name: 'echo',
        tool_call_id: 'call_first',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
      {
        action_name: 'echo',
        tool_call_id: 'call_second',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
    ]);
    /** And the action_requests carry the same ids — pairing is
     * always derivable from id even when names collide. */
    expect(payload.action_requests.map((r) => r.tool_call_id)).toEqual([
      'call_first',
      'call_second',
    ]);
  });

  it('malformed edit decision (missing updatedInput) is blocked, not approved with garbage args', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-malformed' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** `{ type: 'edit' }` with no updatedInput — same trust-boundary
     * issue as malformed respond. Must fail closed, NOT pass undefined
     * into applyInputOverride and approve a tool with garbage args. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'edit' } as unknown as t.ToolApprovalDecision],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing object updatedInput'
    );
    expect(String(toolMessages[0].content)).toContain('<missing>');
    expect(dispatchCount).toBe(0);
  });

  it('malformed edit decision (non-object updatedInput) is blocked', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-nonobject' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** `updatedInput: 'string'` — wire deserializer didn't enforce
     * object shape; SDK must reject. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [
        {
          type: 'edit',
          updatedInput: 'not-an-object' as unknown as Record<string, unknown>,
        },
      ],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing object updatedInput'
    );
    expect(String(toolMessages[0].content)).toContain('string');
    expect(dispatchCount).toBe(0);
  });

  it('malformed edit decision (array updatedInput) is blocked — arrays are objects but not plain records', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-array' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [
        {
          type: 'edit',
          updatedInput: [1, 2, 3] as unknown as Record<string, unknown>,
        },
      ],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('array');
  });

  it('malformed respond decision (missing responseText) is blocked, not crashed', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-malformed' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** Submit a `respond` decision with NO responseText — wire shape
     * the SDK can't honor. Must fail closed (blockEntry path), NOT
     * crash truncateToolResultContent on `undefined.length`. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'respond' } as unknown as t.ToolApprovalDecision],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing string responseText'
    );
    expect(String(toolMessages[0].content)).toContain('<missing>');
    /** Tool was never dispatched — fail-closed worked. */
    expect(dispatchCount).toBe(0);
  });

  it('malformed respond decision (non-string responseText) is blocked, not crashed', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-nonstring' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** `responseText: 42` — wire deserializer didn't enforce string;
     * SDK must reject without crashing. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [
        {
          type: 'respond',
          responseText: 42 as unknown as string,
        },
      ],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing string responseText'
    );
    expect(String(toolMessages[0].content)).toContain('number');
  });

  it('respond decision truncates oversized text the same way real tool output is truncated', async () => {
    mockEventDispatch([]);

    /** Build a ToolNode with a tiny `maxToolResultChars` so the
     * truncation kicks in for a 200-char response. Without the fix,
     * the full string would land in the ToolMessage and PostToolBatch
     * entry — bypassing the model context budget. */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [async (): Promise<PreToolUseHookOutput> => ({ decision: 'ask' })],
    });
    let captured: PostToolBatchEntry | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          captured = (input as PostToolBatchHookInput).entries[0];
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
      maxToolResultChars: 50,
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-truncate' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** 200-char response — well over the 50-char cap. */
    const oversized = 'A'.repeat(200);
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'respond', responseText: oversized }],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    /** The ToolMessage content is truncated; not the raw 200 chars. */
    const content = String(toolMessages[0].content);
    expect(content.length).toBeLessThan(oversized.length);
    /** And the PostToolBatch entry sees the SAME truncated value
     * — batch hooks observe what the model will actually see. */
    expect(captured).toBeDefined();
    expect(typeof captured!.toolOutput).toBe('string');
    expect(captured!.toolOutput).toBe(content);
  });

  it('hook returning both ask + preventContinuation halts cleanly and clears session hooks', async () => {
    mockEventDispatch([]);

    const registry = new HookRegistry();
    /** Session-scoped policy hook returns BOTH `ask` (which would
     * raise an interrupt) AND `preventContinuation: true` (which
     * raises a halt signal). The halt wins — no resume is expected,
     * sessions must clear. */
    const runId = 'ask-and-halt';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          preventContinuation: true,
          stopReason: 'policy halted ask',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'ask-and-halt-thread' },
        version: 'v2',
      }
    );

    /** Both signals landed: interrupt was captured AND halt fired. */
    expect(run.getInterrupt()).toBeDefined();
    expect(run.getHaltReason()).toBe('policy halted ask');
    /** Session hooks MUST be cleared — no resume is expected on a
     * halted run, even one that also captured an interrupt. */
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('preserves Graph sidecars across HITL interrupt + resume so tool completions keep their step ids', async () => {
    /**
     * Regression test for the cleanup-vs-resume bug: previously
     * `processStream` always called `Graph.clearHeavyState()` in its
     * `finally` block AND `Graph.resetValues()` on entry, even when
     * pausing on a HITL interrupt. That wiped `toolCallStepIds`,
     * `_toolOutputRegistry`, and `sessions` between pause and resume,
     * so the resumed `ToolNode` could no longer find the original
     * step id and dispatched `ON_RUN_STEP_COMPLETED` with an empty id
     * — the host's stream consumer would then drop the result.
     *
     * The fix is two gated cleanups:
     *   - `clearHeavyState` skipped when `_interrupt != null && _haltedReason == null && !streamThrew`
     *   - `resetValues` skipped when entering processStream via `Command` (resume)
     *
     * To exercise the SDK Graph's actual sidecar state (not a private
     * test ToolNode), this test wires the custom ToolNode to share
     * the SDK Graph's `toolCallStepIds` Map by reference. After the
     * interrupt fires AND after the resume completes, the
     * pre-populated entry must still be present.
     */
    const dispatchedStepIds: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as { result?: { id?: string } };
          if (payload.result?.id != null) {
            dispatchedStepIds.push(payload.result.id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'sidecar-preserve',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });

    /** Wire the test ToolNode to share the SDK Graph's
     * `toolCallStepIds` Map by reference — this is how the real
     * StandardGraph builds its inner ToolNode at Graph.ts:587. */
    const toolNode = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'a',
      toolCallStepIds: run.Graph!.toolCallStepIds,
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    /** The agent node simulates `attemptInvoke`'s sidecar-population
     * step: in a real run, the model invocation creates a run step
     * and writes its id into `toolCallStepIds` before tools dispatch.
     * Doing it here means the entry lands AFTER `processStream`'s
     * `resetValues` (which fires once on entry) and BEFORE the
     * ToolNode's hook + interrupt — exactly mirroring the production
     * timing the cleanup gate has to preserve. */
    const builder = new StateGraph(MessagesAnnotation)
      .addNode('agent', (): MessagesUpdate => {
        run.Graph!.toolCallStepIds.set('call_1', 'step_real_id');
        return {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        };
      })
      .addNode('tools', toolNode)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'sidecar-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);

    /** After interrupt: sidecar entry MUST still be present. Without
     * the fix, `clearHeavyState` in the `finally` block would have
     * wiped this map. */
    expect(run.getInterrupt()).toBeDefined();
    expect(run.Graph!.toolCallStepIds.has('call_1')).toBe(true);
    expect(run.Graph!.toolCallStepIds.get('call_1')).toBe('step_real_id');

    /** Resume: without the resetValues gate, this would also wipe
     * the map at the START of the second processStream invocation. */
    await run.resume([{ type: 'approve' }], callerConfig);

    /** After resume completes naturally: dispatch fired with the real
     * step id (not an empty string from a wiped map). Without either
     * fix, `dispatchedStepIds` would contain `''`. */
    expect(dispatchedStepIds).toContain('step_real_id');
    expect(dispatchedStepIds).not.toContain('');
    /** And clearHeavyState DID fire on the natural-completion side
     * — sidecar map is now empty after the resume settled. */
    expect(run.Graph!.toolCallStepIds.size).toBe(0);
  });

  it('clears Graph sidecars on natural completion when no interrupt was raised', async () => {
    /** Negative case: when no interrupt fires, `clearHeavyState`
     * MUST run as before. This pins the gate so a future change
     * doesn't accidentally preserve sidecars on natural completion
     * (memory leak across runs). */
    mockEventDispatch([]);

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'sidecar-clear-natural',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: false },
    });

    /** No-op graph — runs to completion without an interrupt. */
    const builder = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END);
    const graph = builder.compile();
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    /** Stash an entry so we can verify clearHeavyState wiped it. */
    run.Graph!.toolCallStepIds.set('stale_call', 'stale_step');

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'sidecar-clear-thread' },
        version: 'v2',
      }
    );

    /** No interrupt → clearHeavyState ran → sidecar wiped. */
    expect(run.getInterrupt()).toBeUndefined();
    expect(run.Graph!.toolCallStepIds.size).toBe(0);
  });

  it('clears session hooks when the stream throws AFTER an interrupt is captured (stale interrupt)', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const registry = new HookRegistry();
    const runId = 'stream-error-after-interrupt';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'session policy',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    /**
     * Holder for forward-referencing the run inside the sentinel
     * handler closure. The handler is constructed before `Run.create`
     * runs (it's passed into `customHandlers`) but needs to read
     * `run.getInterrupt()` at firing time.
     */
    const holder: {
      run: Awaited<ReturnType<typeof Run.create<t.IState>>> | undefined;
    } = { run: undefined };

    /**
     * Handler keyed to a chain-stream event that throws ONLY after the
     * interrupt has been captured. The stream loop captures the
     * interrupt on the chunk that carries `__interrupt__`, then
     * dispatches to handlers in the same iteration — so the throw
     * exits the loop with `_interrupt != null`. Without the
     * `streamThrew` guard, the `finally` block would preserve session
     * hooks on this stale interrupt.
     */
    const sentinelHandler = {
      handle: async (): Promise<void> => {
        if (holder.run?.getInterrupt() != null) {
          throw new Error('post-interrupt handler failure');
        }
      },
    };

    holder.run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
      customHandlers: {
        [GraphEvents.CHAIN_STREAM]: sentinelHandler,
        [GraphEvents.CHAIN_END]: sentinelHandler,
      },
    });
    holder.run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'stale-interrupt-thread' },
      version: 'v2' as const,
    };

    await expect(
      holder.run.processStream({ messages: [] }, callerConfig)
    ).rejects.toThrow('post-interrupt handler failure');

    /** Interrupt WAS captured on the run instance, but because the
     * stream subsequently threw, session hooks must be cleared so the
     * next run on this registry isn't poisoned by stale state. */
    expect(holder.run.getInterrupt()).toBeDefined();
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('mixed deny/ask/allow batch: deny short-circuits, allow runs immediately, ask interrupts; resume completes the asked tool', async () => {
    const dispatchedToolNames: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedToolNames.push(c.name);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    /**
     * Per-tool policy hook: tool_a denied, tool_b asks, tool_c allowed.
     * The hook is registered without a pattern so it fires once per
     * tool call and dispatches by tool name.
     */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (input.toolName === 'tool_a') {
            return { decision: 'deny', reason: 'policy:a' };
          }
          if (input.toolName === 'tool_b') {
            return { decision: 'ask', reason: 'policy:b-needs-review' };
          }
          return { decision: 'allow' };
        },
      ],
    });
    /**
     * Listen on PostToolBatch to verify the batch entry shape after
     * resume reflects the final outcomes (deny + run + run), not
     * stale state from the first pass.
     */
    const batchSnapshots: PostToolBatchEntry[][] = [];
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchSnapshots.push(
            (input as PostToolBatchHookInput).entries.map((e) => ({ ...e }))
          );
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [
        createSchemaStub('tool_a'),
        createSchemaStub('tool_b'),
        createSchemaStub('tool_c'),
      ],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_a', 'step_a'],
        ['call_b', 'step_b'],
        ['call_c', 'step_c'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_a', name: 'tool_a', args: { command: 'a' } },
      { id: 'call_b', name: 'tool_b', args: { command: 'b' } },
      { id: 'call_c', name: 'tool_c', args: { command: 'c' } },
    ]);
    const config = { configurable: { thread_id: 'mixed-thread' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    /** Only tool_b appears in the interrupt — deny short-circuited
     * locally, allow was queued for dispatch but never reached it
     * because `interrupt()` threw inside the same node first. LangGraph
     * rolls back the entire node's effects on throw, so no host event
     * fires for any tool until after resume. This is the safe
     * semantic: partial execution while a human is being asked would
     * leak side effects ahead of approval. */
    expect(payload.action_requests).toHaveLength(1);
    expect(payload.action_requests[0].tool_call_id).toBe('call_b');
    expect(dispatchedToolNames).toEqual([]);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'approve' }],
      config
    )) as { messages: BaseMessage[] };

    /**
     * After resume, all three tools have ToolMessages: tool_a blocked
     * (deny), tool_b ran (host approved), tool_c ran (allow). The
     * ToolNode re-executed from scratch, so both tool_b and tool_c
     * dispatch in this pass.
     */
    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(3);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_a')!.status).toBe('error');
    expect(String(byId.get('call_a')!.content)).toContain('policy:a');
    expect(byId.get('call_b')!.status).not.toBe('error');
    expect(byId.get('call_b')!.content).toBe('ran:tool_b');
    expect(byId.get('call_c')!.status).not.toBe('error');
    expect(byId.get('call_c')!.content).toBe('ran:tool_c');
    /** Both approved tools dispatched on resume; tool_a (deny) never did. */
    expect(new Set(dispatchedToolNames)).toEqual(new Set(['tool_b', 'tool_c']));
    expect(dispatchedToolNames).not.toContain('tool_a');

    /**
     * PostToolBatch is dispatched at the bottom of `dispatchToolEvents`,
     * after tool execution. On the FIRST pass `interrupt()` throws
     * before reaching that line, so PostToolBatch does NOT fire for
     * the interrupted pass. Only the resume pass yields a snapshot —
     * carrying all three entries with their final outcomes (tool_a
     * blocked by deny, tool_b approved + ran, tool_c approved + ran).
     */
    expect(batchSnapshots).toHaveLength(1);
    const finalSnapshot = batchSnapshots[0];
    /**
     * Order assertion: entries must match the original toolCalls
     * sequence (`call_a`, `call_b`, `call_c`) regardless of when each
     * outcome was recorded — `call_a` was denied synchronously in the
     * hook loop, `call_b` was approved through the resume branch,
     * `call_c` was approved+executed via the host event path. Hooks
     * correlating outcomes by position (per the API doc) depend on
     * this stability.
     */
    expect(finalSnapshot.map((e) => e.toolUseId)).toEqual([
      'call_a',
      'call_b',
      'call_c',
    ]);
    const byCallId = new Map(finalSnapshot.map((e) => [e.toolUseId, e]));
    expect(byCallId.size).toBe(3);
    expect(byCallId.get('call_a')!.status).toBe('error');
    expect(byCallId.get('call_a')!.error).toContain('policy:a');
    expect(byCallId.get('call_b')!.status).toBe('success');
    expect(byCallId.get('call_b')!.toolOutput).toBe('ran:tool_b');
    expect(byCallId.get('call_c')!.status).toBe('success');
    expect(byCallId.get('call_c')!.toolOutput).toBe('ran:tool_c');
  });

  it('mixed respond + reject in the same resume: dispatches once each, batch entries in toolCalls order', async () => {
    const stepCompletedDispatches: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as {
            result?: { tool_call?: { id?: string } };
          };
          const id = payload.result?.tool_call?.id;
          if (id != null) {
            stepCompletedDispatches.push(id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const registry = new HookRegistry();
    /** Both tools `ask`; the resume picks `respond` for one and
     * `reject` for the other. Exercises the timing interaction
     * between respond's immediate dispatch and reject's deferred
     * flush in the same resume pass. */
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });
    const batchSnapshots: PostToolBatchEntry[][] = [];
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchSnapshots.push(
            (input as PostToolBatchHookInput).entries.map((e) => ({ ...e }))
          );
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [
        createSchemaStub('respond_tool'),
        createSchemaStub('reject_tool'),
      ],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_respond', 'step_respond'],
        ['call_reject', 'step_reject'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_respond', name: 'respond_tool', args: { command: 'r' } },
      { id: 'call_reject', name: 'reject_tool', args: { command: 'j' } },
    ]);
    const config = { configurable: { thread_id: 'mixed-respond-reject' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    /** First pass: interrupt fires before either dispatch path runs. */
    expect(stepCompletedDispatches).toEqual([]);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [
        { type: 'respond', responseText: 'fake answer' },
        { type: 'reject', reason: 'no thanks' },
      ],
      config
    )) as { messages: BaseMessage[] };

    /** Each tool dispatched ON_RUN_STEP_COMPLETED exactly once on
     * resume — respond via its immediate path, reject via the
     * deferred flush. */
    expect(
      stepCompletedDispatches.filter((id) => id === 'call_respond')
    ).toEqual(['call_respond']);
    expect(
      stepCompletedDispatches.filter((id) => id === 'call_reject')
    ).toEqual(['call_reject']);

    /** PostToolBatch fires once on the resume pass, with entries in
     * the original toolCalls order (respond first, reject second)
     * regardless of which dispatch path landed first into the Map. */
    expect(batchSnapshots).toHaveLength(1);
    expect(batchSnapshots[0].map((e) => e.toolUseId)).toEqual([
      'call_respond',
      'call_reject',
    ]);
    expect(batchSnapshots[0][0].status).toBe('success');
    expect(batchSnapshots[0][0].toolOutput).toBe('fake answer');
    expect(batchSnapshots[0][1].status).toBe('error');
    expect(String(batchSnapshots[0][1].error)).toContain('no thanks');

    /** ToolMessage state matches: success with response text, error with reason. */
    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(2);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_respond')!.status).not.toBe('error');
    expect(byId.get('call_respond')!.content).toBe('fake answer');
    expect(byId.get('call_reject')!.status).toBe('error');
    expect(String(byId.get('call_reject')!.content)).toContain('no thanks');
  });

  it('PostToolBatch entries preserve toolCalls order even when first call is denied and second is approved', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    /**
     * Two different orderings to verify the asserted order really
     * tracks the input — not just incidental ordering from one path
     * landing first.
     */
    const cases: Array<{
      thread: string;
      input: Array<{ id: string; name: string; args: Record<string, unknown> }>;
      expected: string[];
    }> = [
      {
        thread: 'order-deny-first',
        input: [
          { id: 'call_first', name: 'denied_tool', args: { command: 'a' } },
          { id: 'call_second', name: 'allowed_tool', args: { command: 'b' } },
        ],
        expected: ['call_first', 'call_second'],
      },
      {
        thread: 'order-approve-first',
        input: [
          { id: 'call_first', name: 'allowed_tool', args: { command: 'a' } },
          { id: 'call_second', name: 'denied_tool', args: { command: 'b' } },
        ],
        expected: ['call_first', 'call_second'],
      },
    ];

    for (const { thread, input, expected } of cases) {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (hookInput): Promise<PreToolUseHookOutput> => {
            if (hookInput.toolName === 'denied_tool') {
              return { decision: 'deny', reason: 'no' };
            }
            return { decision: 'allow' };
          },
        ],
      });
      const captured: PostToolBatchEntry[] = [];
      registry.register('PostToolBatch', {
        hooks: [
          async (i): Promise<PostToolBatchHookOutput> => {
            captured.push(...(i as PostToolBatchHookInput).entries);
            return {};
          },
        ],
      });

      const node = new ToolNode({
        tools: [
          createSchemaStub('denied_tool'),
          createSchemaStub('allowed_tool'),
        ],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map(input.map((c) => [c.id, `step_${c.id}`])),
        hookRegistry: registry,
        humanInTheLoop: { enabled: false },
      });

      const graph = buildHITLGraph(node, input);
      await graph.invoke(
        { messages: [] },
        { configurable: { thread_id: thread } }
      );

      expect(captured.map((e) => e.toolUseId)).toEqual(expected);
    }
  });

  it('fails closed when the host resume payload carries an unknown decision type', async () => {
    /** Spy MUST be reachable inside Promise.resolve handlers — must not run after mock is restored. */
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'sensitive' } },
    ]);
    const config = { configurable: { thread_id: 'unknown-decision' } };

    const interrupted = await graph.invoke({ messages: [] }, config);

    /** Host sends a typo'd / malformed decision. Must NOT silently approve. */
    const resumed = (await resumeGraph(
      graph,
      interrupted,
      [{ type: 'aproved' as 'approve' }],
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'Unknown approval decision type'
    );
    /** Tool was never dispatched — fail-closed worked. */
    expect(dispatchCalls).toBe(0);
  });

  it('PostToolBatch entry sees the PostToolUse-rewritten output, not the original', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'raw-secret-1234', status: 'success' },
    ]);

    const registry = new HookRegistry();
    /** PostToolUse redacts the output before the model sees it. */
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          updatedOutput: 'raw-secret-[REDACTED]',
        }),
      ],
    });
    let batchEntries: PostToolBatchEntry[] | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchEntries = (input as PostToolBatchHookInput).entries;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'fetch' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-rewrite' } }
    );

    expect(batchEntries).toBeDefined();
    expect(batchEntries).toHaveLength(1);
    /** Batch hook sees the redacted value, not the raw secret. */
    expect(batchEntries![0].toolOutput).toBe('raw-secret-[REDACTED]');
    expect(batchEntries![0].toolOutput).not.toContain('raw-secret-1234');
  });

  it('PostToolUseFailure additionalContext is injected for the next model turn', async () => {
    /** Force the host event dispatch to return an error so the failure path runs. */
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          {
            toolCallId: 'call_1',
            content: '',
            status: 'error',
            errorMessage: 'network timeout',
          },
        ]);
      });

    const registry = new HookRegistry();
    registry.register('PostToolUseFailure', {
      hooks: [
        async (): Promise<PostToolUseFailureHookOutput> => ({
          additionalContext:
            'Tool failed — suggest the user retry with a smaller batch size',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'fetch' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'failure-ctx' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain(
      'suggest the user retry with a smaller batch size'
    );
  });
});

describe('AskUserQuestion — interrupt + resume', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('a node calling askUserQuestion() raises an ask_user_question interrupt and resumes with the answer', async () => {
    const { askUserQuestion } = await import('@/hitl');

    let resumedAnswer: string | undefined;

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('clarifier', () => {
        const resolution = askUserQuestion({
          question: 'Which environment?',
          options: [
            { label: 'Staging', value: 'staging' },
            { label: 'Production', value: 'production' },
          ],
        });
        resumedAnswer = resolution.answer;
        return { messages: [] };
      })
      .addEdge(START, 'clarifier')
      .addEdge('clarifier', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const config = { configurable: { thread_id: 'ask-q-thread' } };

    const interrupted = (await graph.invoke({ messages: [] }, config)) as {
      __interrupt__?: Array<{ id?: string; value?: t.HumanInterruptPayload }>;
    };
    expect(interrupted.__interrupt__).toBeDefined();
    const payload = interrupted.__interrupt__![0].value!;
    if (payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question');
    }
    expect(payload.question.question).toBe('Which environment?');
    expect(payload.question.options).toHaveLength(2);

    const resolution: t.AskUserQuestionResolution = { answer: 'production' };
    await resumeGraph(
      graph as unknown as CompiledMessagesGraph,
      interrupted,
      resolution,
      config
    );

    expect(resumedAnswer).toBe('production');
  });

  it('carries multiSelect through the interrupt payload and resumes with the joined option values', async () => {
    const { askUserQuestion } = await import('@/hitl');

    let resumedAnswer: string | undefined;

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('clarifier', () => {
        const resolution = askUserQuestion({
          question: 'Which environments?',
          options: [
            { label: 'Staging', value: 'staging' },
            { label: 'Production', value: 'production' },
          ],
          multiSelect: true,
        });
        resumedAnswer = resolution.answer;
        return { messages: [] };
      })
      .addEdge(START, 'clarifier')
      .addEdge('clarifier', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const config = { configurable: { thread_id: 'ask-q-multi-thread' } };

    const interrupted = (await graph.invoke({ messages: [] }, config)) as {
      __interrupt__?: Array<{ id?: string; value?: t.HumanInterruptPayload }>;
    };
    const payload = interrupted.__interrupt__![0].value!;
    if (payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question');
    }
    expect(payload.question.multiSelect).toBe(true);
    expect(payload.question.options).toHaveLength(2);

    // Host joins the selected option values with ", ".
    const resolution: t.AskUserQuestionResolution = {
      answer: 'staging, production',
    };
    await resumeGraph(
      graph as unknown as CompiledMessagesGraph,
      interrupted,
      resolution,
      config
    );

    expect(resumedAnswer).toBe('staging, production');
  });

  it('a DIRECT tool in event-driven mode can raise ask_user_question from its body and resume with the answer as its ToolMessage', async () => {
    /**
     * The production host shape (e.g. LibreChat's `AgentInputs.graphTools`
     * plumb): the run is event-driven (other tools are schema-only
     * definitions dispatched to the host), but an interrupt-capable tool is
     * supplied as a real instance and marked direct so it executes inside
     * the Pregel task frame. Event dispatch must never see the call — a
     * host-side handler runs outside the graph, where `interrupt()` throws.
     */
    const { askUserQuestion } = await import('@/hitl');

    const dispatchSpy = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {});

    let bodyRuns = 0;
    const askTool = tool(
      async (input: { question: string }) => {
        bodyRuns += 1;
        const resolution = askUserQuestion(input);
        return resolution.answer;
      },
      {
        name: 'ask_user_question',
        description: 'Ask the user a clarifying question.',
        schema: z.object({ question: z.string() }),
      }
    ) as unknown as StructuredToolInterface;

    const node = new ToolNode({
      tools: [createSchemaStub('echo'), askTool],
      toolMap: new Map([
        ['echo', createSchemaStub('echo')],
        ['ask_user_question', askTool],
      ]),
      eventDrivenMode: true,
      agentId: 'agent-ask-direct',
      toolCallStepIds: new Map([['call_ask_1', 'step_call_ask_1']]),
      directToolNames: new Set(['ask_user_question']),
    });

    const graph = buildHITLGraph(node, [
      {
        id: 'call_ask_1',
        name: 'ask_user_question',
        args: { question: 'Which environment?' },
      },
    ]);
    const config = { configurable: { thread_id: 'thread-ask-direct' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(interrupted)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question payload');
    }
    expect(payload.question.question).toBe('Which environment?');
    expect(bodyRuns).toBe(1);

    /** The interrupt came from the direct path — never dispatched to the host. */
    const toolExecuteDispatches = dispatchSpy.mock.calls.filter(
      ([event]) => event === 'on_tool_execute'
    );
    expect(toolExecuteDispatches).toHaveLength(0);

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      { answer: 'staging' } satisfies t.AskUserQuestionResolution,
      config
    )) as MessagesUpdate;

    expect(bodyRuns).toBe(2); // body re-runs from the top on the resume pass
    const toolMessage = resumed.messages.find(
      (m): m is ToolMessage =>
        m._getType() === 'tool' &&
        (m as ToolMessage).tool_call_id === 'call_ask_1'
    );
    expect(toolMessage).toBeDefined();
    expect(String(toolMessage!.content)).toBe('staging');
  });

  it('askUserQuestion surfaces the calling tool_call_id on the interrupt payload when the body supplies it', async () => {
    /**
     * LangChain stamps the full ToolCall onto the config a `tool(fn, …)`
     * body receives, so the body can attribute its interrupt to the exact
     * call. Hosts use `payload.tool_call_id` to stamp the question/answer
     * onto the right content part when a model emits several ask calls in
     * one turn — positional guessing mislabels the cards.
     */
    const askTool = tool(
      async (input: { question: string }, config) => {
        const resolution = askUserQuestion(input, {
          toolCallId: config.toolCall?.id,
        });
        return resolution.answer;
      },
      {
        name: 'ask_user_question',
        description: 'Ask the user a clarifying question.',
        schema: z.object({ question: z.string() }),
      }
    ) as unknown as StructuredToolInterface;

    const node = new ToolNode({ tools: [askTool] });
    const graph = buildHITLGraph(node, [
      {
        id: 'call_ask_id_1',
        name: 'ask_user_question',
        args: { question: 'Which region?' },
      },
    ]);
    const config = { configurable: { thread_id: 'thread-ask-call-id' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question payload');
    }
    expect(payload.tool_call_id).toBe('call_ask_id_1');
    expect(payload.question.question).toBe('Which region?');

    const resumed = (await resumeGraph(
      graph,
      interrupted,
      { answer: 'us-east' } satisfies t.AskUserQuestionResolution,
      config
    )) as MessagesUpdate;
    const toolMessage = resumed.messages.find(
      (m): m is ToolMessage =>
        m._getType() === 'tool' &&
        (m as ToolMessage).tool_call_id === 'call_ask_id_1'
    );
    expect(String(toolMessage!.content)).toBe('us-east');
  });

  it('isAskUserQuestionInterrupt narrows the payload union correctly', async () => {
    const { isAskUserQuestionInterrupt, isToolApprovalInterrupt } =
      await import('@/types/hitl');

    const askPayload: t.HumanInterruptPayload = {
      type: 'ask_user_question',
      question: { question: 'why?' },
    };
    const approvalPayload: t.HumanInterruptPayload = {
      type: 'tool_approval',
      action_requests: [],
      review_configs: [],
    };

    expect(isAskUserQuestionInterrupt(askPayload)).toBe(true);
    expect(isAskUserQuestionInterrupt(approvalPayload)).toBe(false);
    expect(isToolApprovalInterrupt(approvalPayload)).toBe(true);
    expect(isToolApprovalInterrupt(askPayload)).toBe(false);
  });
});

import { createHash } from 'node:crypto';
import { describe, it, expect, beforeEach } from '@jest/globals';
import { GraphInterrupt, MemorySaver } from '@langchain/langgraph';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AgentInputs,
  GraphSubagentConfig,
  ResolvedSubagentConfig,
  ResolvedSingleAgentSubagentConfig,
  StandardGraphInput,
  SubagentTaskStartRequest,
  SubagentTaskStartResult,
  SubagentUpdateEvent,
  SubagentUsageEvent,
  ToolSessionMap,
  ToolExecuteBatchRequest,
  ToolExecuteResult,
} from '@/types';
import type { StandardGraph } from '@/graphs/Graph';
import {
  SubagentExecutor,
  InMemorySubagentTaskStore,
  filterGraphSubagentResult,
  filterSubagentResult,
  resolveSubagentConfigs,
  buildChildInputs,
  summarizeEvent,
} from '../subagent';
import { sanitizeForwardedSubagentUpdateData } from '../subagent/SubagentExecutor';
import { SUBAGENT_PARENT_BATCH_CONFIG_KEY } from '../subagent/SubagentReplay';
import { Constants, Providers, GraphEvents, StepTypes } from '@/common';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { stableStringify } from '../eagerEventExecution';
import { AgentContext } from '@/agents/AgentContext';
import { HookRegistry } from '@/hooks/HookRegistry';
import { HandlerRegistry } from '@/events';
import { executeHooks } from '@/hooks';

jest.setTimeout(15000);

const makeChildInputs = (agentId = 'child-agent'): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: 'You are a helper agent.',
  maxContextTokens: 8000,
});

const makeConfig = (
  type = 'researcher',
  overrides: Partial<ResolvedSingleAgentSubagentConfig> = {}
): ResolvedSingleAgentSubagentConfig => ({
  type,
  name: 'Test Researcher',
  description: 'Researches things',
  agentInputs: makeChildInputs(),
  ...overrides,
});

describe('filterSubagentResult', () => {
  it('treats a result member with no new AI message as textless completion', () => {
    expect(
      filterGraphSubagentResult(
        {
          messages: [new AIMessage('worker output')],
          subagentResult: { agentId: 'result' },
        },
        'result'
      )
    ).toBe('Task completed');
  });

  it('extracts text from last AIMessage string content', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new AIMessage('Here is the result'),
    ];
    expect(filterSubagentResult(messages)).toBe('Here is the result');
  });

  it('extracts text blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'text', text: 'First part.' },
          { type: 'text', text: 'Second part.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('First part.\nSecond part.');
  });

  it('prefers final text_delta blocks over earlier AI text', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search.' },
          { type: 'tool_use', id: 'call_1', name: 'search', input: {} },
        ],
      }),
      new ToolMessage({ content: 'result', tool_call_id: 'call_1' }),
      new AIMessage({
        content: [
          { type: 'text_delta', index: 0, text: 'Streamed ' },
          { type: 'text_delta', index: 0, text: 'result.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Streamed result.');
  });

  it('keeps annotation-only text blocks within a text_delta sequence', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'text_delta', index: 0, text: 'Cited ' },
          { type: 'text', index: 0, citations: [{ url: 'source' }] },
          { type: 'text_delta', index: 0, text: 'answer.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Cited answer.');
  });

  it('separates text_delta blocks with different indexes', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'text_delta', index: 0, text: 'First.' },
          { type: 'text_delta', index: 1, text: 'Second.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('First.\nSecond.');
  });

  it('strips tool_use blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'tool_use', id: 'call_1', name: 'search', input: {} },
          { type: 'text', text: 'Final answer.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Final answer.');
  });

  it('strips thinking blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'thinking', thinking: 'Let me think...' },
          { type: 'text', text: 'The result.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('The result.');
  });

  it('returns "Task completed" when no text blocks remain', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'tool_use', id: 'call_1', name: 'do_thing', input: {} },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('returns "Task completed" for empty string content', () => {
    const messages: BaseMessage[] = [new AIMessage('')];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('returns "Task completed" when no messages', () => {
    expect(filterSubagentResult([])).toBe('Task completed');
  });

  it('returns "Task completed" when no AIMessage found', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new ToolMessage({ content: 'result', tool_call_id: 'x' }),
    ];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('uses last AIMessage, not first', () => {
    const messages: BaseMessage[] = [
      new AIMessage('First response'),
      new ToolMessage({ content: 'tool output', tool_call_id: 'x' }),
      new AIMessage('Final response'),
    ];
    expect(filterSubagentResult(messages)).toBe('Final response');
  });

  it('salvages text from an earlier AIMessage when the last has only tool_use', () => {
    /**
     * Scenario: subagent hit maxTurns mid-tool-call. The last AIMessage is
     * pure tool_use with no text. Partial progress from an earlier turn
     * should still be returned instead of "Task completed".
     */
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search.' },
          { type: 'tool_use', id: 'c1', name: 'search', input: {} },
        ],
      }),
      new ToolMessage({ content: 'Paris.', tool_call_id: 'c1' }),
      new AIMessage({
        content: [{ type: 'tool_use', id: 'c2', name: 'search', input: {} }],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Let me search.');
  });

  it('salvages from earlier AIMessage when last has empty string content', () => {
    const messages: BaseMessage[] = [
      new AIMessage('Partial answer.'),
      new ToolMessage({ content: 'tool out', tool_call_id: 'x' }),
      new AIMessage(''),
    ];
    expect(filterSubagentResult(messages)).toBe('Partial answer.');
  });
});

describe('resolveSubagentConfigs', () => {
  const parentInputs: AgentInputs = {
    agentId: 'parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o', apiKey: 'test' },
    instructions: 'You are a parent agent.',
    maxContextTokens: 16000,
  };

  it('passes through configs with explicit agentInputs', () => {
    const config = makeConfig();
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([config], parentContext);
    expect(resolved).toHaveLength(1);
    const resolvedConfig = resolved[0];
    expect(resolvedConfig.agentInputs.agentId).toBe('child-agent');
  });

  it('resolves self-spawn from parent _sourceInputs', () => {
    const selfConfig = {
      type: 'self',
      name: 'Self Spawn',
      description: 'Context isolation only',
      self: true,
    };
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([selfConfig], parentContext);
    expect(resolved).toHaveLength(1);
    const resolvedConfig = resolved[0];
    expect(resolvedConfig.agentInputs.provider).toBe(Providers.OPENAI);
    expect(resolvedConfig.agentInputs.instructions).toBe(
      'You are a parent agent.'
    );
  });

  it('filters out configs with self=true when _sourceInputs is missing', () => {
    const selfConfig = {
      type: 'self',
      name: 'Self Spawn',
      description: 'Context isolation only',
      self: true,
    };
    const parentContext = new AgentContext({
      agentId: 'bare',
      provider: Providers.OPENAI,
      instructionTokens: 0,
    });
    const resolved = resolveSubagentConfigs([selfConfig], parentContext);
    expect(resolved).toHaveLength(0);
  });

  it('filters out configs without agentInputs and self=false', () => {
    const badConfig = {
      type: 'broken',
      name: 'Broken',
      description: 'Missing inputs',
    };
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([badConfig], parentContext);
    expect(resolved).toHaveLength(0);
  });

  it('throws on duplicate subagent types', () => {
    const parentContext = AgentContext.fromConfig(parentInputs);
    const dup1 = makeConfig('researcher');
    const dup2 = makeConfig('researcher');
    expect(() => resolveSubagentConfigs([dup1, dup2], parentContext)).toThrow(
      /Duplicate subagent type "researcher"/
    );
  });
});

describe('buildChildInputs', () => {
  const parentAgentInputs: AgentInputs = {
    agentId: 'parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test' },
    instructions: 'parent',
    maxContextTokens: 8000,
    subagentConfigs: [{ type: 'researcher', name: 'R', description: 'd' }],
    maxSubagentDepth: 3,
  };

  it('strips subagentConfigs and maxSubagentDepth when allowNested is false', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.subagentConfigs).toBeUndefined();
    expect(result.maxSubagentDepth).toBeUndefined();
  });

  it('decrements maxSubagentDepth when allowNested is true', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
      allowNested: true,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.maxSubagentDepth).toBe(2);
    expect(result.subagentConfigs).toEqual(parentAgentInputs.subagentConfigs);
  });

  it('clamps decremented depth to 0 (never negative)', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
      allowNested: true,
    };
    const result = buildChildInputs(config, 'child', 0);
    expect(result.maxSubagentDepth).toBe(0);
  });

  it('uses the parent depth at the 1-to-0 boundary even if child inputs request more', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      allowNested: true,
      agentInputs: { ...parentAgentInputs, maxSubagentDepth: 99 },
    };

    const child = buildChildInputs(config, 'child', 1);

    expect(child.maxSubagentDepth).toBe(0);
    expect(child.subagentConfigs).toEqual(parentAgentInputs.subagentConfigs);
    expect(config.agentInputs.maxSubagentDepth).toBe(99);
  });

  it('always strips toolDefinitions (forces traditional mode)', () => {
    const inputsWithToolDefs: AgentInputs = {
      ...parentAgentInputs,
      toolDefinitions: [{ name: 't', description: 'x' }],
    };
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: inputsWithToolDefs,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.toolDefinitions).toBeUndefined();
  });

  it('scrubs INHERITED graphTools on self-spawn (parent-spread config) but keeps an explicit child config’s own', () => {
    const askLikeTool = { name: 'ask_user_question' } as unknown as NonNullable<
      AgentInputs['graphTools']
    >[number];
    const inputsWithGraphTools: AgentInputs = {
      ...parentAgentInputs,
      graphTools: [askLikeTool],
    };

    /**
     * Self-spawn: `resolveSubagentConfigs` fills `agentInputs` as a shallow
     * spread of the parent's `_sourceInputs`, so the parent-scoped direct
     * tool would leak into the child implicitly — it must be scrubbed.
     */
    const selfConfig: ResolvedSubagentConfig = {
      type: 'self',
      name: 'Self',
      description: 'd',
      self: true,
      agentInputs: { ...inputsWithGraphTools },
    };
    expect(
      buildChildInputs(selfConfig, 'child-self', 3).graphTools
    ).toBeUndefined();

    /**
     * Explicit child config: a host that deliberately attaches its own
     * in-process direct tools to a child keeps them (Codex #289 P2). With
     * HITL enabled, interrupt-capable child tools use the shared checkpointer.
     */
    const explicitConfig: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: inputsWithGraphTools,
    };
    expect(
      buildChildInputs(explicitConfig, 'child-explicit', 3).graphTools
    ).toEqual([askLikeTool]);
    expect(inputsWithGraphTools.graphTools).toHaveLength(1); // parent untouched
  });

  it('strips parent-run-scoped initialSummary and discoveredTools from child inputs', () => {
    /**
     * Codex P1: a child inheriting `initialSummary` or `discoveredTools` from
     * the parent's shallow-spread AgentInputs leaks unrelated conversation
     * context / prior tool-search state into an isolated subagent run,
     * defeating the context-isolation contract. Both fields must be cleared.
     */
    const inputsWithRunContext: AgentInputs = {
      ...parentAgentInputs,
      initialSummary: { text: 'prior conversation summary', tokenCount: 42 },
      discoveredTools: ['prior_tool_a', 'prior_tool_b'],
    };
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: inputsWithRunContext,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.initialSummary).toBeUndefined();
    expect(result.discoveredTools).toBeUndefined();
  });

  it('scrubs a parent compaction index on self-spawn but keeps an explicit child index', () => {
    const compactionSemanticIndex: NonNullable<
      AgentInputs['compactionSemanticIndex']
    > = [
      {
        type: 'activity_phase',
        sourceMessageId: 'message-parent',
        sourceContentIndex: 2,
        revision: 1,
        status: 'committed',
        text: 'Parent-only activity',
      },
    ];
    const inputsWithIndex: AgentInputs = {
      ...parentAgentInputs,
      compactionSemanticIndex,
    };

    expect(
      buildChildInputs(
        {
          type: 'self',
          name: 'Self',
          description: 'd',
          self: true,
          agentInputs: { ...inputsWithIndex },
        },
        'child-self',
        3
      ).compactionSemanticIndex
    ).toBeUndefined();
    expect(
      buildChildInputs(
        {
          type: 'researcher',
          name: 'R',
          description: 'd',
          agentInputs: inputsWithIndex,
        },
        'child-explicit',
        3
      ).compactionSemanticIndex
    ).toBe(compactionSemanticIndex);
  });

  it('overrides agentId with the passed childAgentId', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
    };
    const result = buildChildInputs(config, 'my-child', 3);
    expect(result.agentId).toBe('my-child');
  });
});

describe('SubagentExecutor', () => {
  const config = makeConfig();

  class ContinuationTaskStore extends InMemorySubagentTaskStore {
    readonly supportsThreadContinuation = true;
    readonly inputs: string[] = [];
    readonly fingerprints: string[] = [];
    readonly lineage: Array<{
      parentRunId: string;
      parentAgentId?: string;
      parentToolCallId: string;
      subagentKind: SubagentTaskStartRequest['subagentKind'];
    }> = [];

    start(request: SubagentTaskStartRequest): SubagentTaskStartResult {
      this.inputs.push(request.input);
      this.fingerprints.push(request.requestFingerprint ?? '');
      this.lineage.push({
        parentRunId: request.parentRunId,
        ...(request.parentAgentId == null
          ? {}
          : { parentAgentId: request.parentAgentId }),
        parentToolCallId: request.parentToolCallId,
        subagentKind: request.subagentKind,
      });
      const threadId = request.threadId ?? 'new-child-thread';
      return super.start({
        ...request,
        threadId,
        run: (runtime) =>
          request.run(runtime, [new HumanMessage('Saved child turn.')]),
      });
    }
  }

  /**
   * Build a stub `createChildGraph` factory that returns a minimal
   * `StandardGraph`-shaped object whose `createWorkflow().invoke()`
   * resolves to `invokeResult`. Avoids `jest.spyOn(StandardGraph)` so
   * that SubagentExecutor does not need a runtime dep on the graphs
   * module (circular-dep-safe).
   */
  function makeStubGraphFactory(
    invokeResult: { messages: BaseMessage[] },
    clearSpy?: jest.Mock
  ): { factory: () => StandardGraph; clearHeavyState: jest.Mock } {
    const mockClear = clearSpy ?? jest.fn();
    const factory = (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockResolvedValue(invokeResult),
        }),
        clearHeavyState: mockClear,
      }) as unknown as StandardGraph;
    return { factory, clearHeavyState: mockClear };
  }

  function makeThrowingGraphFactory(error: Error): () => StandardGraph {
    return (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockRejectedValue(error),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;
  }

  /** No-op factory for tests that never reach child graph construction. */
  function makeNoopGraphFactory(): () => StandardGraph {
    return (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockResolvedValue({ messages: [] }),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;
  }

  function createExecutor(
    overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
  ): SubagentExecutor {
    return new SubagentExecutor({
      configs: new Map([[config.type, config]]),
      parentRunId: 'test-run',
      parentAgentId: 'parent-agent',
      createChildGraph: makeNoopGraphFactory(),
      ...overrides,
    });
  }

  async function waitForTask(
    store: InMemorySubagentTaskStore,
    taskId: string,
    predicate: (status: string) => boolean
  ): Promise<void> {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      const task = store.get('owner:conversation', taskId);
      if (task != null && predicate(task.status)) {
        return;
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    throw new Error(`Timed out waiting for background task ${taskId}.`);
  }

  it('preserves the legacy background HITL rejection when requested', () => {
    const store = new InMemorySubagentTaskStore();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      humanInTheLoop: { enabled: true, backgroundPausePolicy: 'reject' },
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Blocked by policy.',
        subagentType: 'researcher',
        parentToolCallId: 'call_reject',
      })
    ) as { status: string; message: string };

    expect(response.status).toBe('rejected');
    expect(response.message).toContain(
      'does not support human-in-the-loop pauses'
    );
    expect(store.list('owner:conversation')).toEqual([]);
  });

  it('runs HITL background tasks under deny, omits question tools, and reports blocked calls', async () => {
    const store = new InMemorySubagentTaskStore();
    const questionTool = { name: 'ask_user_question' } as NonNullable<
      AgentInputs['graphTools']
    >[number];
    const normalTool = { name: 'search' } as NonNullable<
      AgentInputs['graphTools']
    >[number];
    const inputs: AgentInputs = {
      ...makeChildInputs(),
      tools: [questionTool, normalTool],
      toolMap: new Map([
        ['ask_user_question', questionTool],
        ['search', normalTool],
      ]),
      toolRegistry: new Map([
        ['ask_user_question', { name: 'ask_user_question' }],
        ['search', { name: 'search' }],
      ]),
      graphTools: [questionTool, normalTool],
      toolDefinitions: [{ name: 'ask_user_question' }, { name: 'search' }],
    };
    const observedInputs: StandardGraphInput[] = [];
    const observedGraphs: StandardGraph[] = [];
    const { factory } = makeStubGraphFactory({
      messages: [
        new ToolMessage({
          name: 'delete_file',
          tool_call_id: 'call_delete',
          status: 'error',
          content:
            'Blocked: Approval required for "delete_file"; unavailable in a background subagent. Ask the parent to run it in the foreground.',
        }),
        new AIMessage('Could not finish that operation.'),
      ],
    });
    const handlers = new HandlerRegistry();
    handlers.register(GraphEvents.ON_TOOL_EXECUTE, {
      handle: (): void => undefined,
    });
    const executor = createExecutor({
      configs: new Map([
        ['researcher', makeConfig('researcher', { agentInputs: inputs })],
      ]),
      humanInTheLoop: { enabled: true },
      taskConfig: { store, scopeId: 'owner:conversation' },
      parentHandlerRegistry: handlers,
      createChildGraph: (input): StandardGraph => {
        observedInputs.push(input);
        const graph = factory();
        graph.eagerEventToolExecution = { enabled: true };
        observedGraphs.push(graph);
        return graph;
      },
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Try the task.',
        subagentType: 'researcher',
        parentToolCallId: 'call_deny',
      })
    ) as { background_task_id: string; status: string };
    expect(response.status).toBe('running');
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(observedGraphs[0].humanInTheLoop).toEqual({
      enabled: false,
      backgroundPausePolicy: 'deny',
    });
    expect(observedGraphs[0].eagerEventToolExecution).toBeUndefined();
    const child = observedInputs[0].agents[0];
    expect(
      child.tools?.map((tool) => ('name' in tool ? tool.name : ''))
    ).toEqual(['search']);
    expect(child.graphTools?.map((tool) => tool.name)).toEqual(['search']);
    expect(child.toolDefinitions?.map((tool) => tool.name)).toEqual(['search']);
    expect([...child.toolMap!.keys()]).toEqual(['search']);
    expect([...child.toolRegistry!.keys()]).toEqual(['search']);
    expect(inputs.graphTools).toHaveLength(2);
    expect(inputs.toolMap).toHaveProperty('size', 2);
    expect(inputs.toolRegistry).toHaveProperty('size', 2);
    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({
      status: 'completed',
      result: expect.stringContaining(
        'Background approval denied for: delete_file'
      ),
    });
  });

  it('fails closed if a detached child unexpectedly raises an interrupt', async () => {
    const store = new InMemorySubagentTaskStore();
    const executor = createExecutor({
      humanInTheLoop: { enabled: true },
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: makeThrowingGraphFactory(
        new GraphInterrupt([
          {
            id: 'unexpected-question',
            value: {
              type: 'ask_user_question',
              question: { question: 'Proceed?' },
            },
          },
        ])
      ),
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Unexpected pause.',
        subagentType: 'researcher',
        parentToolCallId: 'call_unexpected',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'error'
    );
    expect(
      store.get('owner:conversation', response.background_task_id)?.error
    ).toContain('cannot pause for human input');
  });

  it('keeps a detached child and its thread continuation at depth zero', async () => {
    const store = new ContinuationTaskStore();
    const childConfig = makeConfig('researcher', {
      allowNested: true,
      agentInputs: {
        ...makeChildInputs(),
        maxSubagentDepth: 99,
        subagentConfigs: [
          {
            type: 'grandchild',
            name: 'Grandchild',
            description: 'Must not be spawned.',
            agentInputs: makeChildInputs('grandchild'),
          },
        ],
      },
    });
    const observedInputs: StandardGraphInput[] = [];
    const { factory } = makeStubGraphFactory({
      messages: [new AIMessage('child finished')],
    });
    const executor = createExecutor({
      configs: new Map([[childConfig.type, childConfig]]),
      maxDepth: 1,
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (input): StandardGraph => {
        observedInputs.push(input);
        return factory();
      },
    });

    const first = JSON.parse(
      executor.executeInBackground({
        description: 'First task.',
        subagentType: 'researcher',
        parentToolCallId: 'call_first',
      })
    ) as { background_task_id: string; subagent_thread_id: string };
    await waitForTask(
      store,
      first.background_task_id,
      (status) => status === 'completed'
    );

    const continued = JSON.parse(
      executor.executeInBackground({
        description: 'Continue the task.',
        subagentType: 'researcher',
        subagentThreadId: first.subagent_thread_id,
        parentToolCallId: 'call_continued',
      })
    ) as { background_task_id: string; subagent_thread_id: string };
    await waitForTask(
      store,
      continued.background_task_id,
      (status) => status === 'completed'
    );

    expect(continued.subagent_thread_id).toBe(first.subagent_thread_id);
    expect(observedInputs).toHaveLength(2);
    expect(observedInputs.map((input) => input.agents[0])).toEqual([
      expect.objectContaining({
        maxSubagentDepth: 0,
        subagentConfigs: childConfig.agentInputs.subagentConfigs,
      }),
      expect.objectContaining({
        maxSubagentDepth: 0,
        subagentConfigs: childConfig.agentInputs.subagentConfigs,
      }),
    ]);
    expect(
      observedInputs.map((input) => input.subagentExecutionContext?.depth)
    ).toEqual([1, 1]);
  });

  it('rejects background execution when the nesting budget is exhausted', () => {
    const store = new ContinuationTaskStore();
    const executor = createExecutor({
      maxDepth: 0,
      taskConfig: { store, scopeId: 'owner:conversation' },
    });

    const result = JSON.parse(
      executor.executeInBackground({
        description: 'Do not spawn.',
        subagentType: 'researcher',
        parentToolCallId: 'call_blocked',
      })
    ) as { status: string; message: string };

    expect(result).toMatchObject({
      status: 'rejected',
      message: 'Maximum subagent nesting depth exceeded.',
    });
    expect(store.list('owner:conversation')).toEqual([]);
  });

  it('runs a detached child past parent cleanup and exposes its result once', async () => {
    const store = new InMemorySubagentTaskStore();
    let finish = (_value: { messages: BaseMessage[] }): void => undefined;
    const invocation = new Promise<{ messages: BaseMessage[] }>((resolve) => {
      finish = resolve;
    });
    const invoke = jest.fn(() => invocation);
    const clearHeavyState = jest.fn();
    let detachedGraph: StandardGraph | undefined;
    const createDetachedChildGraphFactory = jest.fn(() => (): StandardGraph => {
      detachedGraph = {
        createWorkflow: () => ({ invoke }),
        clearHeavyState,
      } as unknown as StandardGraph;
      return detachedGraph;
    });
    const fallbackFactory = jest.fn(
      (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState,
        }) as unknown as StandardGraph
    );
    const hookRegistry = new HookRegistry();
    hookRegistry.registerSession('test-run', 'PreToolUse', {
      hooks: [async (): Promise<Record<string, never>> => ({})],
    });
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: fallbackFactory,
      createDetachedChildGraphFactory,
      hookRegistry,
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Research independently.',
        subagentType: 'researcher',
        parentToolCallId: 'call_background',
      })
    ) as { background_task_id: string; status: string };

    expect(response.status).toBe('running');
    expect(createDetachedChildGraphFactory).toHaveBeenCalledTimes(1);
    const taskHookSessionId = `test-run:subagent-task:${response.background_task_id}`;
    await waitForTask(
      store,
      response.background_task_id,
      () => invoke.mock.calls.length > 0
    );
    const taskHookRegistry = detachedGraph?.hookRegistry;
    expect(taskHookRegistry).toBeDefined();
    expect(taskHookRegistry).not.toBe(hookRegistry);
    expect(taskHookRegistry?.hasHookFor('PreToolUse', taskHookSessionId)).toBe(
      true
    );
    hookRegistry.clearSession('test-run');
    expect(taskHookRegistry?.hasHookFor('PreToolUse', taskHookSessionId)).toBe(
      true
    );
    executor.clearHeavyState();
    finish({ messages: [new AIMessage('detached result')] });
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'completed', result: 'detached result' });
    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'claimed' });
    expect(fallbackFactory).not.toHaveBeenCalled();
    expect(clearHeavyState).toHaveBeenCalled();
    expect(hookRegistry.hasHookFor('PreToolUse', 'test-run')).toBe(false);
  });

  it('retries detached result delivery without rerunning the child', async () => {
    const store = new InMemorySubagentTaskStore();
    const invoke = jest.fn().mockResolvedValue({
      messages: [new AIMessage('detached result')],
    });
    let deliveryAttempts = 0;
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      subagentContext: {
        prepare: async () => ({}),
        complete: async (_input, result) => {
          deliveryAttempts += 1;
          if (deliveryAttempts === 1) {
            throw new Error('Transient delivery failure');
          }
          return { content: `${result.content} delivered` };
        },
      },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Research independently.',
        subagentType: 'researcher',
        parentToolCallId: 'call_delivery_retry',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(
      store.get('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'completed' });
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(deliveryAttempts).toBe(2);
    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({
      status: 'completed',
      result: 'detached result delivered',
    });
  });

  it('bounds detached result delivery retries', async () => {
    const store = new InMemorySubagentTaskStore();
    const invoke = jest.fn().mockResolvedValue({
      messages: [new AIMessage('detached result')],
    });
    const complete = jest.fn(async () => {
      throw new Error('Persistent delivery failure');
    });
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      subagentContext: { prepare: async () => ({}), complete },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Research independently.',
        subagentType: 'researcher',
        parentToolCallId: 'call_delivery_failure',
      })
    ) as { background_task_id: string };
    await new Promise<void>((resolve) => setTimeout(resolve, 150));

    expect(invoke).toHaveBeenCalledTimes(1);
    expect(complete).toHaveBeenCalledTimes(3);
    expect(
      store.get('owner:conversation', response.background_task_id)
    ).toMatchObject({
      status: 'error',
      error: 'Subagent result delivery failed.',
    });
  });

  it('releases detached state when result delivery hangs past task timeout', async () => {
    const store = new InMemorySubagentTaskStore({ taskTimeoutMs: 20 });
    const clearHeavyState = jest.fn();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      subagentContext: {
        prepare: async () => ({}),
        complete: () => new Promise(() => undefined),
      },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('detached result')],
            }),
          }),
          clearHeavyState,
        }) as unknown as StandardGraph,
    });

    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Research independently.',
        subagentType: 'researcher',
        parentToolCallId: 'call_delivery_timeout',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'error'
    );
    await new Promise<void>((resolve) => setTimeout(resolve, 0));

    expect(clearHeavyState).toHaveBeenCalled();
    expect(
      store.get('owner:conversation', response.background_task_id)
    ).toMatchObject({
      status: 'error',
      error: 'Detached subagent task timed out.',
    });
  });

  it('replaces the ambient parent run config for detached child execution', async () => {
    const store = new InMemorySubagentTaskStore();
    const parentController = new AbortController();
    let observedSignal: AbortSignal | undefined;
    let finish = (_value: { messages: BaseMessage[] }): void => undefined;
    const invocation = new Promise<{ messages: BaseMessage[] }>((resolve) => {
      finish = resolve;
    });
    const invoke = jest.fn(async () => {
      observedSignal = AsyncLocalStorageProviderSingleton.getRunnableConfig()
        ?.signal as AbortSignal | undefined;
      return invocation;
    });
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    const response = await AsyncLocalStorageProviderSingleton.runWithConfig(
      { signal: parentController.signal },
      async () =>
        JSON.parse(
          executor.executeInBackground({
            description: 'Outlive the parent turn.',
            subagentType: 'researcher',
            parentToolCallId: 'call_detached_signal',
          })
        ) as { background_task_id: string }
    );
    await waitForTask(
      store,
      response.background_task_id,
      () => invoke.mock.calls.length > 0
    );

    parentController.abort();
    expect(observedSignal).toBeDefined();
    expect(observedSignal).not.toBe(parentController.signal);
    expect(observedSignal?.aborted).toBe(false);

    finish({ messages: [new AIMessage('detached result')] });
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );
  });

  it('keeps detached siblings alive after their parent aborts', async () => {
    const store = new InMemorySubagentTaskStore();
    const parentController = new AbortController();
    const started = new Set<string>();
    const createInvocation = (): {
      promise: Promise<{ messages: BaseMessage[] }>;
      finish: (value: { messages: BaseMessage[] }) => void;
      signal?: AbortSignal;
    } => {
      let finish = (_value: { messages: BaseMessage[] }): void => undefined;
      const promise = new Promise<{ messages: BaseMessage[] }>((resolve) => {
        finish = resolve;
      });
      return { promise, finish };
    };
    const invocationByCall = new Map([
      ['call_detached_first', createInvocation()],
      ['call_detached_second', createInvocation()],
    ]);
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (input): StandardGraph => {
        const parentToolCallId =
          input.subagentExecutionContext?.ancestry.at(-1)?.parentToolCallId;
        if (parentToolCallId == null) {
          throw new Error('Detached child invocation was not attributed.');
        }
        const invocation = invocationByCall.get(parentToolCallId);
        if (invocation == null) {
          throw new Error('Detached child invocation was not attributed.');
        }
        const attributedCallId = parentToolCallId;
        const invoke = jest.fn(async () => {
          invocation.signal =
            AsyncLocalStorageProviderSingleton.getRunnableConfig()?.signal as
              | AbortSignal
              | undefined;
          if (invocation.signal == null) {
            throw new Error(
              'Detached child did not receive a runnable signal.'
            );
          }
          started.add(attributedCallId);
          return invocation.promise;
        });
        return {
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    const responses = await AsyncLocalStorageProviderSingleton.runWithConfig(
      { signal: parentController.signal },
      async () =>
        ['call_detached_first', 'call_detached_second'].map(
          (parentToolCallId) =>
            JSON.parse(
              executor.executeInBackground({
                description: 'Finish independently after the parent.',
                subagentType: 'researcher',
                parentToolCallId,
              })
            ) as { background_task_id: string }
        )
    );
    await Promise.all(
      responses.map((response) =>
        waitForTask(
          store,
          response.background_task_id,
          () => started.size === 2
        )
      )
    );

    parentController.abort();
    const firstInvocation = invocationByCall.get('call_detached_first');
    const secondInvocation = invocationByCall.get('call_detached_second');
    if (firstInvocation == null || secondInvocation == null) {
      throw new Error('Detached child invocation fixture is incomplete.');
    }
    if (firstInvocation.signal == null || secondInvocation.signal == null) {
      throw new Error('Detached child runnable signal was not captured.');
    }
    const observedSignals = [firstInvocation.signal, secondInvocation.signal];
    expect(observedSignals).toHaveLength(2);
    expect(observedSignals).not.toContain(parentController.signal);
    expect(new Set(observedSignals).size).toBe(2);
    expect(observedSignals.every((signal) => signal.aborted === false)).toBe(
      true
    );

    firstInvocation.finish({
      messages: [new AIMessage('first detached result')],
    });
    await waitForTask(
      store,
      responses[0].background_task_id,
      (status) => status === 'completed'
    );
    expect(secondInvocation.signal.aborted).toBe(false);
    expect(
      store.get('owner:conversation', responses[1].background_task_id)
    ).toMatchObject({
      status: 'running',
    });
    secondInvocation.finish({
      messages: [new AIMessage('second detached result')],
    });
    await waitForTask(
      store,
      responses[1].background_task_id,
      (status) => status === 'completed'
    );

    expect(
      store.claim('owner:conversation', responses[0].background_task_id)
    ).toMatchObject({
      status: 'completed',
      result: 'first detached result',
    });
    expect(
      store.claim('owner:conversation', responses[1].background_task_id)
    ).toMatchObject({
      status: 'completed',
      result: 'second detached result',
    });
  });

  it('continues the same detached child for a queued parent message', async () => {
    const store = new InMemorySubagentTaskStore();
    let finishFirst = (_value: { messages: BaseMessage[] }): void => undefined;
    const firstInvocation = new Promise<{ messages: BaseMessage[] }>(
      (resolve) => {
        finishFirst = resolve;
      }
    );
    const invoke = jest
      .fn()
      .mockImplementationOnce(() => firstInvocation)
      .mockResolvedValueOnce({ messages: [new AIMessage('second answer')] });
    const resetValues = jest.fn();
    let childInput: StandardGraphInput | undefined;
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (input): StandardGraph => {
        childInput = input;
        return {
          createWorkflow: () => ({ invoke }),
          resetValues,
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Start the analysis.',
        subagentType: 'researcher',
        parentToolCallId: 'call_queued',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      () => invoke.mock.calls.length > 0
    );

    expect(
      store.control('owner:conversation', response.background_task_id, {
        action: 'queue',
        message: 'Now compare both sources.',
      })
    ).toMatchObject({ status: 'accepted' });
    finishFirst({ messages: [new AIMessage('first answer')] });
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(invoke).toHaveBeenCalledTimes(2);
    const continuation = invoke.mock.calls[1][0] as {
      messages: BaseMessage[];
    };
    expect(continuation.messages.at(-1)).toBeInstanceOf(HumanMessage);
    expect(continuation.messages.at(-1)?.content).toBe(
      'Now compare both sources.'
    );
    expect(resetValues).toHaveBeenCalledWith(true);
    expect(childInput?.preemption).toBeDefined();
    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'completed', result: 'second answer' });
  });

  it('starts a fresh execution from a host-restored child thread', async () => {
    const store = new ContinuationTaskStore();
    const invoke = jest.fn().mockResolvedValue({
      messages: [new AIMessage('continued answer')],
    });
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Continue with the new evidence.',
        subagentType: 'researcher',
        subagentThreadId: 'child-thread',
        parentToolCallId: 'call_continuation',
      })
    ) as {
      background_task_id: string;
      subagent_thread_id: string;
    };

    expect(response.subagent_thread_id).toBe('child-thread');
    expect(store.inputs).toEqual(['Continue with the new evidence.']);
    expect(store.lineage).toEqual([
      {
        parentRunId: 'test-run',
        parentAgentId: 'parent-agent',
        parentToolCallId: 'call_continuation',
        subagentKind: 'agent',
      },
    ]);
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );
    const childInput = invoke.mock.calls[0][0] as { messages: BaseMessage[] };
    expect(childInput.messages.map((message) => message.content)).toEqual([
      'Saved child turn.',
      'Continue with the new evidence.',
    ]);
  });

  it('preserves the legacy fingerprint when no child thread is selected', () => {
    const store = new ContinuationTaskStore();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
    });

    executor.executeInBackground({
      description: 'Legacy detached task.',
      subagentType: 'researcher',
      parentToolCallId: 'call_legacy_fingerprint',
    });

    expect(store.fingerprints).toEqual([
      createHash('sha256')
        .update(
          stableStringify({
            description: 'Legacy detached task.',
            subagentType: 'researcher',
          })
        )
        .digest('hex'),
    ]);
  });

  it('drains interrupt and steer controls through child-safe hook boundaries', async () => {
    const store = new InMemorySubagentTaskStore();
    let taskHookRegistry: HookRegistry | undefined;
    let childGraph: StandardGraph | undefined;
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let childInput: StandardGraphInput | undefined;
    const invoke = jest.fn(
      async (
        _input: unknown,
        runnableConfig: { configurable?: Record<string, unknown> }
      ): Promise<{ messages: BaseMessage[] }> => {
        await gate;
        taskHookRegistry = childGraph?.hookRegistry;
        const runId = runnableConfig.configurable?.run_id as string;
        expect(childInput?.preemption?.shouldPreempt()).toBe(true);
        if (taskHookRegistry == null) {
          throw new Error('Expected a task-local hook registry.');
        }
        const interrupted = await executeHooks({
          registry: taskHookRegistry,
          input: {
            hook_event_name: 'PreemptBoundary',
            runId,
            agentId: 'child-agent',
            executingAgentId: 'child-agent',
            sealCount: 1,
          },
          sessionId: runId,
        });
        const steered = await executeHooks({
          registry: taskHookRegistry,
          input: {
            hook_event_name: 'PostToolBatch',
            runId,
            agentId: 'child-agent',
            executingAgentId: 'child-agent',
            entries: [],
          },
          sessionId: runId,
        });
        const messages = [
          ...interrupted.injectedMessages,
          ...steered.injectedMessages,
        ];
        return {
          messages: [
            new AIMessage(messages.map((message) => message.content).join('|')),
          ],
        };
      }
    );
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (input): StandardGraph => {
        childInput = input;
        childGraph = {
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
        return childGraph;
      },
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Work until redirected.',
        subagentType: 'researcher',
        parentToolCallId: 'call_controlled',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      () => invoke.mock.calls.length > 0
    );

    store.control('owner:conversation', response.background_task_id, {
      action: 'interrupt',
      message: 'Summarize immediately.',
    });
    store.control('owner:conversation', response.background_task_id, {
      action: 'steer',
      message: 'Include the primary source.',
    });
    release();
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({
      status: 'completed',
      result: 'Summarize immediately.|Include the primary source.',
    });
  });

  it('marks ordinary detached child failures as task errors', async () => {
    const store = new InMemorySubagentTaskStore();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: makeThrowingGraphFactory(new Error('provider failed')),
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Try the provider.',
        subagentType: 'researcher',
        parentToolCallId: 'call_failure',
      })
    ) as { background_task_id: string };

    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'error'
    );

    expect(
      store.claim('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'error', error: 'provider failed' });
  });

  it('reserves recursion headroom for every detached preemption seal', async () => {
    const store = new InMemorySubagentTaskStore();
    const invoke = jest.fn().mockResolvedValue({
      messages: [new AIMessage('done')],
    });
    const executor = createExecutor({
      configs: new Map([
        [config.type, makeConfig(config.type, { maxTurns: 2 })],
      ]),
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Finish after interrupts.',
        subagentType: 'researcher',
        parentToolCallId: 'call_recursion_headroom',
      })
    ) as { background_task_id: string };

    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'completed'
    );

    expect(invoke.mock.calls[0][1]).toMatchObject({
      recursionLimit: 2 * 3 + 32,
    });
  });

  it('cancels a detached child through its dedicated abort signal', async () => {
    const store = new InMemorySubagentTaskStore();
    const invoke = jest.fn(
      (_input: unknown, runnableConfig: { signal?: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          runnableConfig.signal?.addEventListener(
            'abort',
            () => reject(runnableConfig.signal?.reason),
            { once: true }
          );
        })
    );
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const response = JSON.parse(
      executor.executeInBackground({
        description: 'Wait until cancelled.',
        subagentType: 'researcher',
        parentToolCallId: 'call_cancelled',
      })
    ) as { background_task_id: string };
    await waitForTask(
      store,
      response.background_task_id,
      () => invoke.mock.calls.length > 0
    );

    expect(
      store.control('owner:conversation', response.background_task_id, {
        action: 'cancel',
      })
    ).toMatchObject({ status: 'cancelled' });
    await waitForTask(
      store,
      response.background_task_id,
      (status) => status === 'cancelled'
    );
    expect(
      store.get('owner:conversation', response.background_task_id)
    ).toMatchObject({ status: 'cancelled' });
  });

  it('rejects changed arguments replayed under the same parent tool call', async () => {
    const store = new InMemorySubagentTaskStore();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
    });
    const first = JSON.parse(
      executor.executeInBackground({
        description: 'Original task.',
        subagentType: 'researcher',
        parentToolCallId: 'call_conflict',
      })
    ) as { background_task_id: string };
    const conflict = JSON.parse(
      executor.executeInBackground({
        description: 'Changed task.',
        subagentType: 'researcher',
        parentToolCallId: 'call_conflict',
      })
    ) as { status: string; message: string };

    expect(conflict).toMatchObject({
      status: 'rejected',
      message:
        'The same parent tool call ID was already used with different background subagent arguments.',
    });
    store.control('owner:conversation', first.background_task_id, {
      action: 'cancel',
    });
  });

  it('rejects a different child thread under the same parent tool call', () => {
    const store = new ContinuationTaskStore();
    const executor = createExecutor({
      taskConfig: { store, scopeId: 'owner:conversation' },
    });
    const first = JSON.parse(
      executor.executeInBackground({
        description: 'Continue the child.',
        subagentType: 'researcher',
        subagentThreadId: 'child-a',
        parentToolCallId: 'call_thread_conflict',
      })
    ) as { background_task_id: string };
    const conflict = JSON.parse(
      executor.executeInBackground({
        description: 'Continue the child.',
        subagentType: 'researcher',
        subagentThreadId: 'child-b',
        parentToolCallId: 'call_thread_conflict',
      })
    ) as { status: string; message: string };

    expect(conflict).toMatchObject({
      status: 'rejected',
      message:
        'The same parent tool call ID was already used with different background subagent arguments.',
    });
    store.control('owner:conversation', first.background_task_id, {
      action: 'cancel',
    });
  });

  it('fails closed when detached execution is unavailable or not attributable', () => {
    const disabled = JSON.parse(
      createExecutor().executeInBackground({
        description: 'Do not run.',
        subagentType: 'researcher',
        parentToolCallId: 'call_disabled',
      })
    ) as { status: string; message: string };
    const store = new InMemorySubagentTaskStore();
    const unattributed = JSON.parse(
      createExecutor({
        taskConfig: { store, scopeId: 'owner:conversation' },
      }).executeInBackground({
        description: 'Do not run.',
        subagentType: 'researcher',
      })
    ) as { status: string; message: string };

    expect(disabled).toMatchObject({
      status: 'rejected',
      message: 'Background subagent execution is not enabled for this run.',
    });
    expect(unattributed).toMatchObject({
      status: 'rejected',
      message: 'Background subagent execution requires a parent tool call ID.',
    });
    expect(store.list('owner:conversation')).toEqual([]);
  });

  it('fails closed when the host cannot restore child threads', () => {
    const store = new InMemorySubagentTaskStore();
    const result = JSON.parse(
      createExecutor({
        taskConfig: { store, scopeId: 'owner:conversation' },
      }).executeInBackground({
        description: 'Continue this child.',
        subagentType: 'researcher',
        subagentThreadId: 'child-thread',
        parentToolCallId: 'call_unsupported_continuation',
      })
    ) as { status: string; message: string };

    expect(result).toMatchObject({
      status: 'rejected',
      message: 'Child-thread continuation is not enabled by this host.',
    });
    expect(store.list('owner:conversation')).toEqual([]);
  });

  it('reports an unavailable child thread without misclassifying it as capacity', () => {
    class RejectingContinuationStore extends ContinuationTaskStore {
      override start(
        request: SubagentTaskStartRequest
      ): SubagentTaskStartResult {
        if (request.threadId === 'missing-child') {
          return { accepted: false, reason: 'thread_unavailable' };
        }
        return super.start(request);
      }
    }
    const store = new RejectingContinuationStore();
    const result = JSON.parse(
      createExecutor({
        taskConfig: { store, scopeId: 'owner:conversation' },
      }).executeInBackground({
        description: 'Continue this child.',
        subagentType: 'researcher',
        subagentThreadId: 'missing-child',
        parentToolCallId: 'call_missing_continuation',
      })
    ) as { status: string; message: string };

    expect(result).toMatchObject({
      status: 'rejected',
      message:
        'The requested subagent thread is unavailable in this parent scope. Start a new subagent thread or choose one created by this parent for the same subagent type.',
    });
    expect(store.list('owner:conversation')).toEqual([]);
  });

  it('cancels a continuation task when its host omits the required thread ID', async () => {
    class InvalidContinuationStore extends ContinuationTaskStore {
      override start(
        request: SubagentTaskStartRequest
      ): SubagentTaskStartResult {
        const started = super.start(request);
        return started.accepted
          ? { ...started, task: { ...started.task, threadId: undefined } }
          : started;
      }
    }
    const store = new InvalidContinuationStore();
    const result = JSON.parse(
      createExecutor({
        taskConfig: { store, scopeId: 'owner:conversation' },
      }).executeInBackground({
        description: 'Start this child.',
        subagentType: 'researcher',
        parentToolCallId: 'call_missing_host_thread',
      })
    ) as { status: string; message: string };

    expect(result).toMatchObject({
      status: 'rejected',
      message:
        'The host accepted the subagent task without assigning its required thread ID.',
    });
    await Promise.resolve();
    expect(store.list('owner:conversation')).toEqual([
      expect.objectContaining({ status: 'cancelled' }),
    ]);
  });

  it('returns error for unknown subagent type', async () => {
    const executor = createExecutor();
    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'nonexistent',
    });
    expect(result.content).toContain('Unknown subagent type');
    expect(result.content).toContain('nonexistent');
    expect(result.content).toContain('researcher');
    expect(result.messages).toEqual([]);
  });

  it('returns error when maxDepth is 0 (nesting budget exhausted)', async () => {
    const executor = createExecutor({ maxDepth: 0 });
    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });
    expect(result.content).toContain('Maximum subagent nesting depth');
    expect(result.messages).toEqual([]);
  });

  it('fails graph HITL before factories, hooks, events, or usage can run', async () => {
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'graph-team',
      name: 'Graph Team',
      description: 'Runs a graph.',
      agents: [makeChildInputs('graph-member')],
      edges: [],
      entryAgentId: 'graph-member',
      resultAgentId: 'graph-member',
    };
    const createChildGraph = jest.fn(makeNoopGraphFactory());
    const createChildGraphByKind = jest.fn(
      (): StandardGraph => makeNoopGraphFactory()()
    );
    const hook = jest.fn(async (): Promise<Record<string, never>> => ({}));
    const hookRegistry = new HookRegistry();
    hookRegistry.register('SubagentStart', { hooks: [hook] });
    const update = jest.fn();
    const handlerRegistry = new HandlerRegistry();
    handlerRegistry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
      handle: update,
    });
    const usageSink = jest.fn();
    const executor = createExecutor({
      configs: new Map([[graphConfig.type, graphConfig]]),
      humanInTheLoop: { enabled: true },
      createChildGraph,
      createChildGraphByKind,
      hookRegistry,
      parentHandlerRegistry: handlerRegistry,
      usageSink,
    });

    const result = await executor.execute({
      description: 'Do not run.',
      subagentType: graphConfig.type,
    });

    expect(result.content).toBe(
      'Error: Human-in-the-loop execution is not yet supported for graph subagents.'
    );
    expect(createChildGraph).not.toHaveBeenCalled();
    expect(createChildGraphByKind).not.toHaveBeenCalled();
    expect(hook).not.toHaveBeenCalled();
    expect(update).not.toHaveBeenCalled();
    expect(usageSink).not.toHaveBeenCalled();
  });

  it('fails closed when resumable execution has no parent tool call ID', async () => {
    const createChildGraph = jest.fn(makeNoopGraphFactory());
    const executor = createExecutor({
      humanInTheLoop: { enabled: true },
      createChildGraph,
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
      threadId: 'durable-thread',
    });

    expect(result.content).toContain('requires a parent tool call ID');
    expect(result.messages).toEqual([]);
    expect(createChildGraph).not.toHaveBeenCalled();
  });

  it('persists terminal replay outcomes without an active child workflow', async () => {
    const checkpointer = new MemorySaver();
    const hookRegistry = new HookRegistry();
    hookRegistry.registerSession('original-run', 'PreToolUse', {
      hooks: [async (): Promise<Record<string, never>> => ({})],
    });
    const initialExecutor = createExecutor({
      checkpointer,
      hookRegistry,
      humanInTheLoop: { enabled: true },
      parentRunId: 'original-run',
    });
    const call = {
      id: 'call_terminal',
      name: Constants.SUBAGENT,
      args: {
        description: 'blocked before execution',
        subagent_type: 'researcher',
      },
      type: 'tool_call' as const,
    };
    const originalConfig = {
      configurable: {
        thread_id: 'durable-parent',
        checkpoint_id: 'parent-fork',
        run_id: 'original-run',
      },
    };
    const settled = {
      output: new ToolMessage({
        content: 'Blocked: policy',
        status: 'error' as const,
        name: Constants.SUBAGENT,
        tool_call_id: call.id,
      }),
      additionalContexts: ['persisted context'],
      resolvedArgs: { description: 'rewritten' },
    };

    await initialExecutor.persistSettledToolOutput(
      call,
      originalConfig,
      settled
    );

    const rebuiltExecutor = createExecutor({
      checkpointer,
      hookRegistry,
      humanInTheLoop: { enabled: true },
      parentRunId: 'rebuilt-run',
    });
    const restored = await rebuiltExecutor.getSettledToolOutput(call, {
      configurable: {
        ...originalConfig.configurable,
        run_id: 'rebuilt-run',
      },
    });

    expect(restored?.output.content).toBe('Blocked: policy');
    expect(restored?.additionalContexts).toEqual(['persisted context']);
    expect(restored?.resolvedArgs).toEqual({ description: 'rewritten' });
    expect(hookRegistry.hasHookFor('PreToolUse', 'original-run')).toBe(true);
    expect(hookRegistry.hasHookFor('PreToolUse', 'rebuilt-run')).toBe(true);
  });

  it('persists an ordinary child error after the active workflow is released', async () => {
    const checkpointer = new MemorySaver();
    const invoke = jest.fn().mockRejectedValue(new Error('child failed'));
    const executor = createExecutor({
      checkpointer,
      humanInTheLoop: { enabled: true },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({
            getState: jest.fn().mockResolvedValue({
              values: {},
              next: ['child-agent'],
              tasks: [],
            }),
            invoke,
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const call = {
      id: 'call_error',
      name: Constants.SUBAGENT,
      args: { description: 'fail', subagent_type: 'researcher' },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-parent',
        checkpoint_id: 'parent-fork',
        run_id: 'original-run',
      },
    };
    const result = await executor.execute({
      description: 'fail',
      subagentType: 'researcher',
      threadId: 'durable-parent',
      parentToolCallId: call.id,
      parentConfigurable: runnableConfig.configurable,
    });
    expect(result.content).toBe('Subagent error: child failed');

    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: result.content,
        name: Constants.SUBAGENT,
        tool_call_id: call.id,
      }),
      additionalContexts: [],
    });

    const restored = await createExecutor({
      checkpointer,
      humanInTheLoop: { enabled: true },
    }).getSettledToolOutput(call, runnableConfig);
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(restored?.output.content).toBe('Subagent error: child failed');
  });

  it('continues a recovered in-progress child without reseeding its task', async () => {
    const invoke = jest.fn().mockResolvedValue({
      messages: [new AIMessage('continued')],
    });
    const executor = createExecutor({
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({
            getState: jest.fn().mockResolvedValue({
              values: {},
              next: ['child-agent'],
              tasks: [],
            }),
            invoke,
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    const result = await executor.execute({
      description: 'Do not duplicate this task',
      subagentType: 'researcher',
      threadId: 'durable-parent',
      parentToolCallId: 'call_in_progress',
    });

    expect(result.content).toBe('continued');
    expect(invoke).toHaveBeenCalledWith(null, expect.any(Object));
  });

  it('preserves an existing nested approval scope', async () => {
    const nestedScope = {
      run_id: 'grandchild-run',
      agent_id: 'grandchild-agent',
      subagent_type: 'grandchild',
      parent_tool_call_id: 'call_grandchild',
    };
    const nestedInterrupt = new GraphInterrupt([
      {
        id: 'nested-interrupt',
        value: {
          type: 'tool_approval',
          action_requests: [],
          review_configs: [],
          subagent: nestedScope,
        },
      },
    ]);
    const executor = createExecutor({
      humanInTheLoop: { enabled: true },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: () => ({
            getState: jest.fn().mockResolvedValue({
              values: {},
              next: ['agent'],
              tasks: [],
            }),
            invoke: jest.fn().mockRejectedValue(nestedInterrupt),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    let received: GraphInterrupt | undefined;
    try {
      await executor.execute({
        description: 'Delegate again',
        subagentType: 'researcher',
        threadId: 'durable-thread',
        parentToolCallId: 'call_child',
      });
    } catch (error) {
      if (error instanceof GraphInterrupt) {
        received = error;
      }
    }

    expect(received?.interrupts[0].value).toMatchObject({
      subagent: nestedScope,
    });
  });

  it('rethrows a child stream-limit trip instead of converting it to a tool result', async () => {
    const executor = createExecutor({
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockRejectedValue(
              new StreamLimitExceededError({
                kind: 'tool_call_args',
                limit: 10,
                observed: 11,
                toolName: 'db_query',
              })
            ),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    await expect(
      executor.execute({
        description: 'Do something',
        subagentType: 'researcher',
      })
    ).rejects.toBeInstanceOf(StreamLimitExceededError);
  });

  it('closes child steps as \'failed\' when the child trips its own stream limit', async () => {
    /**
     * The closure sweep runs AFTER the catch block self-aborts the child
     * breaker, so deriving the status from the live signal relabels the
     * child's own limit failure as an intentional stop — while the parent
     * stamps 'failed' for the same incident. The status must come from the
     * pre-error snapshot.
     */
    const closeUnfinishedRunSteps = jest.fn();
    const executor = createExecutor({
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockRejectedValue(
              new StreamLimitExceededError({
                kind: 'tool_call_args',
                limit: 10,
                observed: 11,
                toolName: 'db_query',
              })
            ),
          }),
          closeUnfinishedRunSteps,
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    await expect(
      executor.execute({
        description: 'Do something',
        subagentType: 'researcher',
      })
    ).rejects.toBeInstanceOf(StreamLimitExceededError);
    expect(closeUnfinishedRunSteps).toHaveBeenCalledWith(
      'failed',
      expect.any(Number)
    );
  });

  it('closes child steps as \'cancelled\' when the abort preceded the error', async () => {
    /** The counterpart bound: a child that dies BECAUSE it was already
     * aborted (caller abort, sibling breaker trip) is an intentional stop,
     * and must keep reading as 'cancelled' under the snapshot. */
    const closeUnfinishedRunSteps = jest.fn();
    const batchBreaker = new AbortController();
    let releaseChild: (() => void) | undefined;
    const executor = createExecutor({
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(
              () =>
                new Promise((_resolve, reject) => {
                  releaseChild = (): void =>
                    reject(new Error('aborted mid-flight'));
                })
            ),
          }),
          closeUnfinishedRunSteps,
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const pending = executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
      breaker: batchBreaker,
    });
    while (releaseChild == null) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    batchBreaker.abort();
    releaseChild();
    const result = await pending;
    expect(result.content).toContain('Subagent error');
    expect(closeUnfinishedRunSteps).toHaveBeenCalledWith(
      'cancelled',
      expect.any(Number)
    );
  });

  it('binds the child to the batch-captured controller, not the live scope', async () => {
    const batchBreaker = new AbortController();
    const liveBreaker = new AbortController();
    const executor = createExecutor({
      breakerScope: { controller: () => liveBreaker },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockRejectedValue(
              new StreamLimitExceededError({
                kind: 'tool_call_args',
                limit: 10,
                observed: 11,
                toolName: 'db_query',
              })
            ),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    /** `breaker` is the controller the parent TOOL BATCH captured before
     * PreToolUse hooks; a reset during those hooks replaced the live
     * scope, and the child must not revive on — or trip — the new run's
     * controller. */
    await expect(
      executor.execute({
        description: 'Do something',
        subagentType: 'researcher',
        breaker: batchBreaker,
      })
    ).rejects.toBeInstanceOf(StreamLimitExceededError);
    expect(batchBreaker.signal.aborted).toBe(true);
    expect(liveBreaker.signal.aborted).toBe(false);
  });

  it('trips the breaker controller captured at execution start, not the current one', async () => {
    const oldRun = new AbortController();
    const newRun = new AbortController();
    let current = oldRun;
    let releaseChild: (() => void) | undefined;
    const executor = createExecutor({
      breakerScope: { controller: () => current },
      createChildGraph: (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(
              () =>
                new Promise((_resolve, reject) => {
                  releaseChild = (): void =>
                    reject(
                      new StreamLimitExceededError({
                        kind: 'tool_call_args',
                        limit: 10,
                        observed: 11,
                        toolName: 'db_query',
                      })
                    );
                })
            ),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    const pending = executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });
    while (releaseChild == null) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    /** The failed run's reset installs a fresh controller while the child
     * is still settling; the straggler's trip must land on the controller
     * captured when its execution started. */
    current = newRun;
    releaseChild();
    await expect(pending).rejects.toBeInstanceOf(StreamLimitExceededError);
    expect(oldRun.signal.aborted).toBe(true);
    expect(newRun.signal.aborted).toBe(false);
  });

  it('threads run-level streamLimits into every child graph input', async () => {
    const childGraphInputs: StandardGraphInput[] = [];
    const noopFactory = makeNoopGraphFactory();
    const executor = createExecutor({
      streamLimits: { maxToolCallArgBytes: 1234, maxDeltaEventsPerTurn: 9 },
      createChildGraph: (input: StandardGraphInput): StandardGraph => {
        childGraphInputs.push(input);
        return noopFactory();
      },
    });
    await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });
    expect(childGraphInputs).toHaveLength(1);
    expect(childGraphInputs[0].streamLimits).toEqual({
      maxToolCallArgBytes: 1234,
      maxDeltaEventsPerTurn: 9,
    });
  });

  it('executes child graph and returns filtered content', async () => {
    const { factory, clearHeavyState } = makeStubGraphFactory({
      messages: [
        new HumanMessage('research this topic'),
        new AIMessage('Here is my research summary.'),
      ],
    });
    const executor = createExecutor({ createChildGraph: factory });

    const result = await executor.execute({
      description: 'Research this topic',
      subagentType: 'researcher',
    });

    expect(result.content).toBe('Here is my research summary.');
    expect(result.messages).toHaveLength(2);
    expect(clearHeavyState).toHaveBeenCalled();
  });

  it('passes parent Langfuse config to the child graph', async () => {
    const langfuse = {
      enabled: true,
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.test',
      toolOutputTracing: { enabled: false },
    };
    let observedLangfuse: typeof langfuse | undefined;
    const executor = createExecutor({
      langfuse,
      createChildGraph: (input): StandardGraph => {
        observedLangfuse = input.langfuse as typeof langfuse;
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('child done')],
            }),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    await executor.execute({
      description: 'Research this topic',
      subagentType: 'researcher',
    });

    expect(observedLangfuse).toBe(langfuse);
  });

  describe('usage sink', () => {
    type CapturedCallbackHandler = {
      handleChatModelStart?: (
        llm: unknown,
        messages: unknown,
        runId: string,
        parentRunId?: string,
        extraParams?: Record<string, unknown>,
        tags?: string[],
        metadata?: Record<string, unknown>
      ) => unknown;
      handleLLMEnd?: (output: unknown, runId: string) => unknown;
      handleLLMError?: (err: unknown, runId: string) => unknown;
    };
    type CapturedInvokeOptions = { callbacks?: CapturedCallbackHandler[] };

    /**
     * Stub factory that records the `StandardGraphInput` the executor
     * builds and the options passed to `workflow.invoke`, so tests can
     * drive the attached usage-capture callback directly (the stubbed
     * invoke never makes real model calls, so callbacks would otherwise
     * never fire).
     */
    function makeCapturingGraphFactory(driveDuringInvoke?: {
      drive: (handler: CapturedCallbackHandler) => void | Promise<void>;
    }): {
      factory: (input: StandardGraphInput) => StandardGraph;
      getInput: () => StandardGraphInput | undefined;
      getInvokeOptions: () => CapturedInvokeOptions | undefined;
    } {
      let capturedInput: StandardGraphInput | undefined;
      let capturedOptions: CapturedInvokeOptions | undefined;
      const factory = (input: StandardGraphInput): StandardGraph => {
        capturedInput = input;
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest
              .fn()
              .mockImplementation(
                async (_input: unknown, options: CapturedInvokeOptions) => {
                  capturedOptions = options;
                  const usageHandler = options.callbacks?.find(
                    (cb) => cb.handleLLMEnd != null
                  );
                  if (driveDuringInvoke && usageHandler) {
                    await driveDuringInvoke.drive(usageHandler);
                  }
                  return { messages: [new AIMessage('child done')] };
                }
              ),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      };
      return {
        factory,
        getInput: () => capturedInput,
        getInvokeOptions: () => capturedOptions,
      };
    }

    const makeChoice = (
      usage: Record<string, number> | undefined
    ): unknown => ({
      text: 'ok',
      message: new AIMessage({
        content: 'ok',
        ...(usage
          ? {
            usage_metadata: usage as unknown as AIMessage['usage_metadata'],
          }
          : {}),
      }),
    });

    const makeLLMEndOutput = (
      usage: Record<string, number> | undefined
    ): unknown => ({
      generations: [[makeChoice(usage)]],
    });

    it('forwards a wrapped sink into the child graph input that rewrites runId to the root run', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory, getInput } = makeCapturingGraphFactory();
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      const forwarded = getInput()?.subagentUsageSink;
      expect(typeof forwarded).toBe('function');
      /**
       * Simulate a NESTED child's emission: its executor stamps `runId`
       * with its own parent (an intermediate `*_sub_*` id). The wrapper
       * must rewrite it to THIS executor's parent run so the host always
       * sees root-run attribution, while the emitting child's identity
       * (`subagentRunId`) is preserved.
       */
      forwarded?.({
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
        model: 'gpt-4o-mini',
        provider: Providers.OPENAI,
        subagentType: 'nested-grandchild',
        subagentRunId: 'test-run_sub_a_sub_b',
        subagentAgentId: 'grandchild',
        runId: 'test-run_sub_a',
      });

      expect(events).toHaveLength(1);
      expect(events[0].runId).toBe('test-run');
      expect(events[0].subagentRunId).toBe('test-run_sub_a_sub_b');
      expect(events[0].subagentType).toBe('nested-grandchild');
    });

    it('does not attach a capture callback when no sink is provided', async () => {
      const { factory, getInvokeOptions } = makeCapturingGraphFactory();
      const executor = createExecutor({ createChildGraph: factory });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(getInvokeOptions()?.callbacks).toEqual([]);
    });

    it('emits tagged usage events with per-call ls_model_name', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleChatModelStart?.(
            {},
            [[]],
            'call-1',
            undefined,
            undefined,
            undefined,
            { ls_model_name: 'gpt-4o-mini-2024-07-18' }
          );
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 11,
              output_tokens: 7,
              total_tokens: 18,
            }),
            'call-1'
          );
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events).toHaveLength(1);
      const event = events[0];
      expect(event.usage).toEqual({
        input_tokens: 11,
        output_tokens: 7,
        total_tokens: 18,
      });
      expect(event.model).toBe('gpt-4o-mini-2024-07-18');
      expect(event.provider).toBe(Providers.OPENAI);
      expect(event.subagentType).toBe('researcher');
      expect(event.subagentAgentId).toBe('child-agent');
      expect(event.subagentRunId).toContain('test-run_sub_');
      expect(event.runId).toBe('test-run');
    });

    it('falls back to the configured model when a call has no ls_model_name', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 3,
              output_tokens: 2,
              total_tokens: 5,
            }),
            'call-1'
          );
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events).toHaveLength(1);
      /** `makeChildInputs` configures `clientOptions.modelName`. */
      expect(events[0].model).toBe('gpt-4o-mini');
    });

    it('emits one event per generation group when a call has multiple completions (n > 1)', async () => {
      const usage = { input_tokens: 10, output_tokens: 4, total_tokens: 14 };
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          /**
           * One provider request with two choices — both carry the same
           * request-level usage. Emitting per choice would double-bill.
           */
          await handler.handleLLMEnd?.(
            { generations: [[makeChoice(usage), makeChoice(usage)]] },
            'call-1'
          );
          /** Batched prompts: two groups = two requests = two events. */
          await handler.handleLLMEnd?.(
            { generations: [[makeChoice(usage)], [makeChoice(usage)]] },
            'call-2'
          );
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events).toHaveLength(3);
    });

    it('prefers INVOKED_PROVIDER/INVOKED_MODEL metadata for fallback-served calls', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          /**
           * Mirror a fallback-served call: `attemptInvoke` stamps the
           * serving provider, `tryFallbackProviders` stamps the fallback's
           * configured model, and the provider reports no `ls_model_name`.
           */
          await handler.handleChatModelStart?.(
            {},
            [[]],
            'call-1',
            undefined,
            undefined,
            undefined,
            {
              [Constants.INVOKED_PROVIDER]: Providers.ANTHROPIC,
              [Constants.INVOKED_MODEL]: 'claude-fallback-1',
            }
          );
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 5,
              output_tokens: 3,
              total_tokens: 8,
            }),
            'call-1'
          );
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events).toHaveLength(1);
      /** Not the configured primary (openAI / gpt-4o-mini). */
      expect(events[0].provider).toBe(Providers.ANTHROPIC);
      expect(events[0].model).toBe('claude-fallback-1');
    });

    it('prefers provider-reported ls_model_name over INVOKED_MODEL', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleChatModelStart?.(
            {},
            [[]],
            'call-1',
            undefined,
            undefined,
            undefined,
            {
              ls_model_name: 'claude-fallback-1-20260101',
              [Constants.INVOKED_PROVIDER]: Providers.ANTHROPIC,
              [Constants.INVOKED_MODEL]: 'claude-fallback-1',
            }
          );
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 5,
              output_tokens: 3,
              total_tokens: 8,
            }),
            'call-1'
          );
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events[0].model).toBe('claude-fallback-1-20260101');
    });

    it('skips model calls that report no usage_metadata', async () => {
      const events: SubagentUsageEvent[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleLLMEnd?.(makeLLMEndOutput(undefined), 'call-1');
        },
      });
      const executor = createExecutor({
        usageSink: (event) => {
          events.push(event);
        },
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(events).toEqual([]);
    });

    it('swallows sink errors without breaking the child run', async () => {
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 1,
              output_tokens: 1,
              total_tokens: 2,
            }),
            'call-1'
          );
        },
      });
      const executor = createExecutor({
        usageSink: () => {
          throw new Error('host sink exploded');
        },
        createChildGraph: factory,
      });

      const result = await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(result.content).toBe('child done');
    });

    it('awaits async sinks and swallows their rejections', async () => {
      const settled: string[] = [];
      const { factory } = makeCapturingGraphFactory({
        drive: async (handler) => {
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 1,
              output_tokens: 1,
              total_tokens: 2,
            }),
            'call-1'
          );
          await handler.handleLLMEnd?.(
            makeLLMEndOutput({
              input_tokens: 2,
              output_tokens: 2,
              total_tokens: 4,
            }),
            'call-2'
          );
          /**
           * Both sink dispatches must have settled by the time
           * `handleLLMEnd` resolves — a dropped promise would leave
           * `recorded` missing here and surface the second call's
           * rejection as unhandled.
           */
          settled.push('drive-done');
        },
      });
      const executor = createExecutor({
        usageSink: async (event) => {
          await new Promise((resolve) => setTimeout(resolve, 5));
          if (event.usage.input_tokens === 2) {
            throw new Error('async host sink rejected');
          }
          settled.push('recorded');
        },
        createChildGraph: factory,
      });

      const result = await executor.execute({
        description: 'Research this topic',
        subagentType: 'researcher',
      });

      expect(result.content).toBe('child done');
      expect(settled).toEqual(['recorded', 'drive-done']);
    });
  });

  it('returns error message when child graph throws', async () => {
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(
        new Error('Graph recursion limit reached')
      ),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    expect(result.content).toContain('Subagent error');
    expect(result.content).toContain('Graph recursion limit reached');
    expect(result.messages).toEqual([]);
  });

  it('truncates long error messages to 200 chars', async () => {
    const longMessage = 'x'.repeat(500);
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(new Error(longMessage)),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    /**
     * Expected composition: "Subagent error: " (16) + 200 truncated chars + "..." (3) = 219.
     * Assert the exact envelope to catch regressions in the truncation constant.
     */
    const MAX_TRUNCATED_LENGTH = 'Subagent error: '.length + 200 + '...'.length;
    expect(result.content.length).toBe(MAX_TRUNCATED_LENGTH);
    expect(result.content.startsWith('Subagent error: ')).toBe(true);
    expect(result.content.endsWith('...')).toBe(true);
  });

  it('does not truncate short error messages', async () => {
    const shortMessage = 'brief error detail';
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(new Error(shortMessage)),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    expect(result.content).toBe(`Subagent error: ${shortMessage}`);
    expect(result.content.endsWith('...')).toBe(false);
  });

  it('builds child with decremented maxSubagentDepth when allowNested=true', async () => {
    const nestedConfig: ResolvedSubagentConfig = {
      type: 'nested',
      name: 'Nested',
      description: 'allows nesting',
      allowNested: true,
      agentInputs: {
        ...makeChildInputs('nested-child'),
        subagentConfigs: [
          {
            type: 'nested',
            name: 'Nested',
            description: 'allows nesting',
            allowNested: true,
          },
        ],
        maxSubagentDepth: 3,
      },
    };

    let observedChildInputs: AgentInputs | undefined;
    const executor = new SubagentExecutor({
      configs: new Map([[nestedConfig.type, nestedConfig]]),
      parentRunId: 'test-run',
      parentAgentId: 'parent',
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedChildInputs = input.agents[0];
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('nested done')],
            }),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    await executor.execute({
      description: 'nested task',
      subagentType: 'nested',
    });

    expect(observedChildInputs).toBeDefined();
    expect(observedChildInputs!.maxSubagentDepth).toBe(2);
    expect(observedChildInputs!.subagentConfigs).toBeDefined();
  });

  it('strips subagentConfigs from child when allowNested is not set', async () => {
    let observedChildInputs: AgentInputs | undefined;
    const executor = createExecutor({
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedChildInputs = input.agents[0];
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('done')],
            }),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    await executor.execute({
      description: 'task',
      subagentType: 'researcher',
    });

    expect(observedChildInputs).toBeDefined();
    expect(observedChildInputs!.subagentConfigs).toBeUndefined();
    expect(observedChildInputs!.maxSubagentDepth).toBeUndefined();
  });

  it('seeds only the selected single-agent child before its first invocation', async () => {
    const selectedSession = {
      session_id: 'selected-storage',
      files: [
        {
          id: 'selected-file',
          name: 'selected.txt',
          storage_session_id: 'selected-storage',
        },
      ],
      lastUpdated: 1,
    };
    const siblingSession = {
      session_id: 'sibling-storage',
      files: [
        {
          id: 'sibling-file',
          name: 'sibling.txt',
          storage_session_id: 'sibling-storage',
        },
      ],
      lastUpdated: 1,
    };
    const selected = makeConfig('selected', {
      agentInputs: {
        ...makeChildInputs('selected-agent'),
        initialSessions: new Map([[Constants.EXECUTE_CODE, selectedSession]]),
      },
    });
    const sibling = makeConfig('sibling', {
      agentInputs: {
        ...makeChildInputs('sibling-agent'),
        initialSessions: new Map([[Constants.EXECUTE_CODE, siblingSession]]),
      },
    });
    const childSessions: ToolSessionMap = new Map();
    const invoke = jest.fn(async () => {
      expect(childSessions.get(Constants.EXECUTE_CODE)).toEqual(
        selectedSession
      );
      expect(childSessions.get(Constants.EXECUTE_CODE)).not.toBe(
        selectedSession
      );
      return { messages: [new AIMessage('done')] };
    });
    const executor = createExecutor({
      configs: new Map([
        [selected.type, selected],
        [sibling.type, sibling],
      ]),
      createChildGraph: (): StandardGraph =>
        ({
          sessions: childSessions,
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    await executor.execute({
      description: 'Use the selected file.',
      subagentType: selected.type,
    });

    expect(invoke).toHaveBeenCalledTimes(1);
    expect(
      childSessions
        .get(Constants.EXECUTE_CODE)
        ?.files?.some((file) => file.id === 'sibling-file')
    ).toBe(false);
  });

  it('remaps a selected child legacy code seed into its agent partition', async () => {
    const codeSessionKey = 'execute_code:stateful:user-1';
    const seededSession = {
      session_id: 'selected-storage',
      files: [
        {
          id: 'selected-file',
          name: 'selected.txt',
          storage_session_id: 'selected-storage',
        },
      ],
      lastUpdated: 1,
    };
    const selected = makeConfig('selected-partition', {
      agentInputs: {
        ...makeChildInputs('selected-agent'),
        codeSessionKey,
        initialSessions: new Map([[Constants.EXECUTE_CODE, seededSession]]),
      },
    });
    const childSessions: ToolSessionMap = new Map();
    const invoke = jest.fn(async () => {
      expect(childSessions.has(Constants.EXECUTE_CODE)).toBe(false);
      expect(childSessions.get(codeSessionKey)).toEqual(seededSession);
      expect(childSessions.get(codeSessionKey)).not.toBe(seededSession);
      return { messages: [new AIMessage('done')] };
    });
    const executor = createExecutor({
      configs: new Map([[selected.type, selected]]),
      createChildGraph: (): StandardGraph =>
        ({
          sessions: childSessions,
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });

    await executor.execute({
      description: 'Use the selected file.',
      subagentType: selected.type,
    });

    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('combines member session seeds inside a selected graph child', async () => {
    const makeSession = (
      storageSessionId: string,
      includeFileSession = true
    ) => ({
      session_id: storageSessionId,
      files: [
        {
          id: 'shared-file-id',
          name: `${storageSessionId}.txt`,
          ...(includeFileSession
            ? { storage_session_id: storageSessionId }
            : {}),
        },
      ],
      lastUpdated: 1,
    });
    const first = {
      ...makeChildInputs('first'),
      initialSessions: new Map([
        [Constants.EXECUTE_CODE, makeSession('first-storage')],
      ]),
    };
    const second = {
      ...makeChildInputs('second'),
      initialSessions: new Map([
        [Constants.EXECUTE_CODE, makeSession('second-storage', false)],
      ]),
    };
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'graph-team',
      name: 'Graph Team',
      description: 'Runs a graph.',
      agents: [first, second],
      edges: [{ from: 'first', to: 'second', edgeType: 'direct' }],
      entryAgentId: 'first',
      resultAgentId: 'second',
    };
    const childSessions: ToolSessionMap = new Map();
    const invoke = jest.fn(async () => {
      expect(
        childSessions
          .get(Constants.EXECUTE_CODE)
          ?.files?.map((file) => [file.id, file.storage_session_id])
      ).toEqual([
        ['shared-file-id', 'first-storage'],
        ['shared-file-id', 'second-storage'],
      ]);
      return {
        messages: [new AIMessage('done')],
        subagentResult: { agentId: 'second' },
      };
    });
    const childGraph = {
      sessions: childSessions,
      createWorkflow: () => ({ invoke }),
      clearHeavyState: jest.fn(),
    } as unknown as StandardGraph;
    const executor = createExecutor({
      configs: new Map([[graphConfig.type, graphConfig]]),
      createChildGraphByKind: () => childGraph,
    });

    await executor.execute({
      description: 'Use both files.',
      subagentType: graphConfig.type,
    });

    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('preserves the child session across an in-memory HITL resume', async () => {
    const seededSession = {
      session_id: 'seed-storage',
      files: [
        {
          id: 'seed-file',
          name: 'seed.txt',
          storage_session_id: 'seed-storage',
        },
      ],
      lastUpdated: 1,
    };
    const hitlConfig = makeConfig('hitl-child', {
      agentInputs: {
        ...makeChildInputs('hitl-child'),
        initialSessions: new Map([[Constants.EXECUTE_CODE, seededSession]]),
      },
    });
    const childSessions: ToolSessionMap = new Map();
    const childInterrupt = new GraphInterrupt([
      {
        id: 'approval-1',
        value: {
          type: 'tool_approval',
          action_requests: [],
          review_configs: [],
        },
      },
    ]);
    const invoke = jest
      .fn()
      .mockImplementationOnce(async () => {
        const session = childSessions.get(Constants.EXECUTE_CODE)!;
        childSessions.set(Constants.EXECUTE_CODE, {
          ...session,
          session_id: 'live-execution',
          files: [
            ...(session.files ?? []),
            {
              id: 'generated-file',
              name: 'generated.txt',
              storage_session_id: 'live-execution',
            },
          ],
        });
        throw childInterrupt;
      })
      .mockImplementationOnce(async () => {
        expect(childSessions.get(Constants.EXECUTE_CODE)).toMatchObject({
          session_id: 'live-execution',
          files: [{ id: 'seed-file' }, { id: 'generated-file' }],
        });
        return { messages: [new AIMessage('resumed')] };
      });
    const executor = createExecutor({
      configs: new Map([[hitlConfig.type, hitlConfig]]),
      humanInTheLoop: { enabled: true },
      checkpointer: new MemorySaver(),
      createChildGraph: (): StandardGraph =>
        ({
          sessions: childSessions,
          createWorkflow: () => ({
            getState: jest.fn().mockResolvedValue({
              values: {},
              next: [],
              tasks: [],
            }),
            invoke,
          }),
          getChildCheckpointThreadIds: jest.fn(() => []),
          createSubagentResumeState: jest.fn(() => ({
            toolCallSteps: [],
            toolSessions: [],
            toolNodes: [],
            eagerToolUsage: [],
            eagerToolSuppressions: [],
          })),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const params = {
      description: 'Pause and resume.',
      subagentType: hitlConfig.type,
      threadId: 'durable-parent',
      parentToolCallId: 'call-hitl',
    };

    await expect(executor.execute(params)).rejects.toBeInstanceOf(
      GraphInterrupt
    );
    const result = await executor.execute({
      ...params,
      parentConfigurable: {
        __pregel_resume_map: { 'approval-1': [] },
      },
    });

    expect(result.content).toBe('resumed');
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  describe('parentConfigurable inheritance', () => {
    type CapturingGraphFactory = {
      factory: () => StandardGraph;
      getInvokeConfig: () => Record<string, unknown> | undefined;
    };

    /**
     * Build a stub factory that captures the second argument to
     * `workflow.invoke()` (the runnable config) so tests can assert on
     * the `configurable` we forwarded to the child graph.
     */
    function makeCapturingGraphFactory(): CapturingGraphFactory {
      let capturedConfig: Record<string, unknown> | undefined;
      const factory = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest
              .fn()
              .mockImplementation(
                async (
                  _input: unknown,
                  config: Record<string, unknown>
                ): Promise<{ messages: BaseMessage[] }> => {
                  capturedConfig = config;
                  return { messages: [new AIMessage('done')] };
                }
              ),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;
      return { factory, getInvokeConfig: () => capturedConfig };
    }

    it('forwards parentConfigurable into the child workflow.invoke configurable', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({ createChildGraph: factory });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        parentConfigurable: {
          requestBody: { messageId: 'msg-123', conversationId: 'conv-456' },
          user: { id: 'user_abc' },
          user_id: 'user_abc',
          userMCPAuthMap: { 'mcp-github': { token: 'abc' } },
        },
      });

      const invokeConfig = getInvokeConfig();
      expect(invokeConfig).toBeDefined();
      const configurable = invokeConfig!.configurable as Record<
        string,
        unknown
      >;
      expect(configurable.requestBody).toEqual({
        messageId: 'msg-123',
        conversationId: 'conv-456',
      });
      expect(configurable.user).toEqual({ id: 'user_abc' });
      expect(configurable.user_id).toBe('user_abc');
      expect(configurable.userMCPAuthMap).toEqual({
        'mcp-github': { token: 'abc' },
      });
    });

    it('inherits parent thread_id when supplied (subagent is part of same conversation)', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({
        createChildGraph: factory,
        parentRunId: 'parent-run-xyz',
      });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        parentConfigurable: { thread_id: 'parent-thread-conv-abc' },
      });

      const configurable = getInvokeConfig()!.configurable as Record<
        string,
        unknown
      >;
      expect(configurable.thread_id).toBe('parent-thread-conv-abc');
    });

    it('falls back to childRunId for thread_id when parent did not supply one', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({
        createChildGraph: factory,
        parentRunId: 'parent-run-xyz',
      });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        parentConfigurable: { user_id: 'user_abc' },
      });

      const configurable = getInvokeConfig()!.configurable as Record<
        string,
        unknown
      >;
      expect(configurable.thread_id as string).toMatch(/^parent-run-xyz_sub_/);
      expect(configurable.user_id).toBe('user_abc');
    });

    it('forwards run-identity fields verbatim into the child invoke configurable', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({ createChildGraph: factory });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        parentConfigurable: {
          run_id: 'parent-run-id',
          parent_run_id: 'grandparent-run-id',
          requestBody: { messageId: 'msg-1' },
        },
      });

      const configurable = getInvokeConfig()!.configurable as Record<
        string,
        unknown
      >;
      // The SDK forwards these fields as part of its inheritance contract.
      // NOTE: the LangGraph runtime overwrites `configurable.run_id` at
      // actual child-invoke time (verified empirically); this unit test
      // only asserts what the SDK forwards into `workflow.invoke` — not
      // what tools downstream observe. `parent_run_id` and other
      // host-set keys do survive the runtime pass-through.
      expect(configurable.run_id).toBe('parent-run-id');
      expect(configurable.parent_run_id).toBe('grandparent-run-id');
      expect(configurable.requestBody).toEqual({ messageId: 'msg-1' });
    });

    it('strips LangGraph runtime fields from child workflow.invoke configurable', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({ createChildGraph: factory });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        parentConfigurable: {
          __pregel_abort_signals: { externalAbortSignal: 'parent-signal' },
          __pregel_call: (): void => undefined,
          __pregel_scratchpad: { currentTaskInput: 'large-payload' },
          checkpoint_id: 'parent-checkpoint-id',
          checkpoint_map: { parent: 'checkpoint' },
          checkpoint_ns: 'parent-checkpoint-ns',
          [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'assistant-batch',
          requestBody: { messageId: 'msg-1' },
          thread_id: 'parent-thread',
          user: { id: 'user_abc' },
        },
      });

      const configurable = getInvokeConfig()!.configurable as Record<
        string,
        unknown
      >;
      expect(configurable.__pregel_abort_signals).toBeUndefined();
      expect(configurable.__pregel_call).toBeUndefined();
      expect(configurable.__pregel_scratchpad).toBeUndefined();
      expect(configurable.checkpoint_id).toBeUndefined();
      expect(configurable.checkpoint_map).toBeUndefined();
      expect(configurable.checkpoint_ns).toBeUndefined();
      expect(configurable[SUBAGENT_PARENT_BATCH_CONFIG_KEY]).toBeUndefined();
      expect(configurable.requestBody).toEqual({ messageId: 'msg-1' });
      expect(configurable.thread_id).toBe('parent-thread');
      expect(configurable.user).toEqual({ id: 'user_abc' });
    });

    it('isolates child checkpoints created from different parent forks', async () => {
      const invokedConfigs: Record<string, unknown>[] = [];
      const invoke = jest
        .fn()
        .mockImplementation(
          async (
            _input: unknown,
            config: Record<string, unknown>
          ): Promise<{ messages: BaseMessage[] }> => {
            invokedConfigs.push(config);
            return { messages: [new AIMessage('done')] };
          }
        );
      const executor = createExecutor({
        humanInTheLoop: { enabled: true },
        createChildGraph: (): StandardGraph =>
          ({
            createWorkflow: () => ({
              getState: jest.fn().mockResolvedValue({
                values: {},
                next: [],
                tasks: [],
              }),
              invoke,
            }),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph,
      });
      const common = {
        description: 'task',
        subagentType: 'researcher',
        threadId: 'parent-thread',
        parentToolCallId: 'call_shared',
      };

      await executor.execute({
        ...common,
        parentConfigurable: {
          thread_id: 'parent-thread',
          checkpoint_id: 'fork-a',
        },
      });
      await executor.execute({
        ...common,
        parentConfigurable: {
          thread_id: 'parent-thread',
          checkpoint_id: 'fork-b',
        },
      });

      expect(invoke).toHaveBeenCalledTimes(2);
      const childThreadIds = invokedConfigs.map(
        (config) =>
          (config.configurable as Record<string, unknown>).thread_id as string
      );
      expect(childThreadIds[0]).toMatch(/^subagent:/);
      expect(childThreadIds[1]).toMatch(/^subagent:/);
      expect(childThreadIds[0]).not.toBe(childThreadIds[1]);
    });

    it('isolates reused tool-call IDs across assistant batches', async () => {
      const invokedThreadIds: string[] = [];
      const executor = createExecutor({
        humanInTheLoop: { enabled: true },
        createChildGraph: (): StandardGraph =>
          ({
            createWorkflow: () => ({
              getState: jest.fn().mockResolvedValue({
                values: {},
                next: [],
                tasks: [],
              }),
              invoke: jest
                .fn()
                .mockImplementation(
                  async (
                    _input: unknown,
                    invokeConfig: Record<string, unknown>
                  ): Promise<{ messages: BaseMessage[] }> => {
                    invokedThreadIds.push(
                      (invokeConfig.configurable as Record<string, unknown>)
                        .thread_id as string
                    );
                    return { messages: [new AIMessage('done')] };
                  }
                ),
            }),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph,
      });
      const common = {
        description: 'task',
        subagentType: 'researcher',
        threadId: 'parent-thread',
        parentToolCallId: 'call_reused',
      };

      await executor.execute({
        ...common,
        parentConfigurable: {
          thread_id: 'parent-thread',
          [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'assistant-batch-1',
        },
      });
      await executor.execute({
        ...common,
        parentConfigurable: {
          thread_id: 'parent-thread',
          [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'assistant-batch-2',
        },
      });

      expect(invokedThreadIds).toHaveLength(2);
      expect(invokedThreadIds[0]).not.toBe(invokedThreadIds[1]);
    });

    it('keeps ambiguous parent agent and tool-call components collision-free', async () => {
      const invokedThreadIds: string[] = [];
      const createChildGraph = (): StandardGraph =>
        ({
          createWorkflow: () => ({
            getState: jest.fn().mockResolvedValue({
              values: {},
              next: [],
              tasks: [],
            }),
            invoke: jest
              .fn()
              .mockImplementation(
                async (
                  _input: unknown,
                  invokeConfig: Record<string, unknown>
                ): Promise<{ messages: BaseMessage[] }> => {
                  invokedThreadIds.push(
                    (invokeConfig.configurable as Record<string, unknown>)
                      .thread_id as string
                  );
                  return { messages: [new AIMessage('done')] };
                }
              ),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;
      const common = {
        description: 'task',
        subagentType: 'researcher',
        threadId: 'parent',
        parentConfigurable: { thread_id: 'parent' },
      };

      await createExecutor({
        checkpointer: new MemorySaver(),
        humanInTheLoop: { enabled: true },
        parentAgentId: 'a_b',
        createChildGraph,
      }).execute({ ...common, parentToolCallId: 'c' });
      await createExecutor({
        checkpointer: new MemorySaver(),
        humanInTheLoop: { enabled: true },
        parentAgentId: 'a',
        createChildGraph,
      }).execute({ ...common, parentToolCallId: 'b_c' });

      expect(invokedThreadIds).toHaveLength(2);
      expect(invokedThreadIds[0]).not.toBe(invokedThreadIds[1]);
    });

    it('retains owned child checkpoint threads after heavy-state cleanup', async () => {
      const executor = createExecutor({
        checkpointer: new MemorySaver(),
        humanInTheLoop: { enabled: true },
        createChildGraph: (): StandardGraph =>
          ({
            createWorkflow: () => ({
              getState: jest.fn().mockResolvedValue({
                values: {},
                next: [],
                tasks: [],
              }),
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('done')],
              }),
            }),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph,
      });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
        threadId: 'parent',
        parentToolCallId: 'call_owned',
      });
      const beforeCleanup = executor.getChildCheckpointThreadIds();
      executor.clearHeavyState();

      expect(beforeCleanup).toHaveLength(1);
      expect(executor.getChildCheckpointThreadIds()).toEqual(beforeCleanup);

      executor.resetCheckpointThreadIds();
      expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    });

    it('includes checkpoint threads from active descendant graphs', async () => {
      const grandchildThreadId = 'subagent:grandchild';
      const executor = createExecutor({
        checkpointer: new MemorySaver(),
        humanInTheLoop: { enabled: true },
        createChildGraph: (): StandardGraph =>
          ({
            createWorkflow: () => ({
              getState: jest.fn().mockResolvedValue({
                values: {},
                next: ['child-agent'],
                tasks: [],
              }),
              invoke: jest.fn().mockRejectedValue(
                new GraphInterrupt([
                  {
                    id: 'grandchild-interrupt',
                    value: {
                      type: 'tool_approval',
                      action_requests: [],
                      review_configs: [],
                    },
                  },
                ])
              ),
            }),
            getChildCheckpointThreadIds: jest.fn(() => [grandchildThreadId]),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph,
      });

      await expect(
        executor.execute({
          description: 'nested task',
          subagentType: 'researcher',
          threadId: 'parent',
          parentToolCallId: 'call_child',
        })
      ).rejects.toBeInstanceOf(GraphInterrupt);

      expect(executor.getChildCheckpointThreadIds()).toEqual(
        expect.arrayContaining([grandchildThreadId])
      );
    });

    it('does not require parentConfigurable (back-compat with hosts that omit it)', async () => {
      const { factory, getInvokeConfig } = makeCapturingGraphFactory();
      const executor = createExecutor({ createChildGraph: factory });

      await executor.execute({
        description: 'task',
        subagentType: 'researcher',
      });

      const configurable = getInvokeConfig()!.configurable as Record<
        string,
        unknown
      >;
      expect(Object.keys(configurable)).toEqual([
        'executionContext',
        'thread_id',
      ]);
      expect(configurable.executionContext).toMatchObject({
        rootRunId: 'test-run',
        depth: 1,
      });
    });
  });

  describe('hooks', () => {
    let capturedStart: unknown;
    let capturedStop: unknown;

    beforeEach(() => {
      capturedStart = undefined;
      capturedStop = undefined;
    });

    it('fires SubagentStart before execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStart', {
        hooks: [
          async (input): Promise<Record<string, never>> => {
            capturedStart = input;
            return {};
          },
        ],
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        hookRegistry: registry,
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      expect(capturedStart).toBeDefined();
      const input = capturedStart as Record<string, unknown>;
      expect(input.hook_event_name).toBe('SubagentStart');
      expect(input.parentAgentId).toBe('parent-agent');
      expect(input.agentType).toBe('researcher');
    });

    it('fires SubagentStop after execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStop', {
        hooks: [
          async (input): Promise<Record<string, never>> => {
            capturedStop = input;
            return {};
          },
        ],
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        hookRegistry: registry,
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      expect(capturedStop).toBeDefined();
      const input = capturedStop as Record<string, unknown>;
      expect(input.hook_event_name).toBe('SubagentStop');
      expect(input.agentType).toBe('researcher');
    });

    it('finishes missing child lifecycle after recovering a completed checkpoint', async () => {
      const stopHook = jest.fn().mockResolvedValue({});
      const hookRegistry = new HookRegistry();
      hookRegistry.register('SubagentStop', { hooks: [stopHook] });
      const updates: SubagentUpdateEvent[] = [];
      const handlerRegistry = new HandlerRegistry();
      handlerRegistry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          updates.push(data as SubagentUpdateEvent);
        },
      });
      const invoke = jest.fn();
      const updateState = jest.fn().mockResolvedValue({});
      const executor = createExecutor({
        humanInTheLoop: { enabled: true },
        hookRegistry,
        parentHandlerRegistry: handlerRegistry,
        createChildGraph: (): StandardGraph =>
          ({
            createWorkflow: () => ({
              getState: jest.fn().mockResolvedValue({
                values: { messages: [new AIMessage('persisted result')] },
                next: [],
                tasks: [],
              }),
              invoke,
              updateState,
            }),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph,
      });

      const result = await executor.execute({
        description: 'recover me',
        subagentType: 'researcher',
        threadId: 'durable-thread',
        parentToolCallId: 'call_recovered',
      });

      expect(result.content).toBe('persisted result');
      expect(invoke).not.toHaveBeenCalled();
      expect(stopHook).toHaveBeenCalledTimes(1);
      expect(updates.filter((update) => update.phase === 'stop')).toHaveLength(
        1
      );
      expect(updateState).toHaveBeenCalledTimes(1);
      expect(updateState.mock.calls[0][1]).toMatchObject({
        messages: [
          {
            additional_kwargs: {
              __librechat_subagent_checkpoint: {
                version: 1,
                parentToolCallId: 'call_recovered',
                lifecycleComplete: true,
              },
            },
          },
        ],
      });
    });

    it('SubagentStart deny blocks execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStart', {
        hooks: [
          async (): Promise<{ decision: 'deny'; reason: string }> => ({
            decision: 'deny',
            reason: 'Not authorized',
          }),
        ],
      });

      const executor = createExecutor({ hookRegistry: registry });
      const result = await executor.execute({
        description: 'Blocked task',
        subagentType: 'researcher',
      });

      expect(result.content).toBe('Blocked: Not authorized');
      expect(result.messages).toEqual([]);
    });
  });

  describe('event forwarding', () => {
    it('emits start/stop ON_SUBAGENT_UPDATE envelopes when parentHandlerRegistry is provided', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      const phases = events.map((e) => (e as { phase: string }).phase);
      expect(phases[0]).toBe('start');
      expect(phases[phases.length - 1]).toBe('stop');
    });

    it('attributes graph updates to validated payload and step-local member identities', async () => {
      const updates: SubagentUpdateEvent[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          updates.push(data as SubagentUpdateEvent);
        },
      });
      const graphConfig: GraphSubagentConfig = {
        kind: 'graph',
        type: 'graph-team',
        name: 'Graph Team',
        description: 'Runs a graph.',
        agents: [
          makeChildInputs('entry'),
          makeChildInputs('worker'),
          makeChildInputs('result'),
        ],
        edges: [
          { from: 'entry', to: 'worker', edgeType: 'direct' },
          { from: 'worker', to: 'result', edgeType: 'direct' },
        ],
        entryAgentId: 'entry',
        resultAgentId: 'result',
      };
      const graphFactory = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              const opts = options as { callbacks?: unknown[] };
              const forwarder = (opts.callbacks ?? [])[0] as {
                handleCustomEvent?: (
                  eventName: string,
                  data: unknown,
                  runId: string,
                  tags?: string[],
                  metadata?: Record<string, unknown>
                ) => Promise<void> | void;
              };
              const emit = async (
                eventName: string,
                data: unknown,
                agentId: string
              ): Promise<void> => {
                await forwarder.handleCustomEvent?.(
                  eventName,
                  data,
                  'child-run',
                  [],
                  { agentId }
                );
              };

              await emit(
                GraphEvents.ON_RUN_STEP,
                {
                  id: 'step_entry',
                  type: StepTypes.MESSAGE_CREATION,
                  agentId: 'entry',
                },
                'parent-agent'
              );
              await emit(
                GraphEvents.ON_MESSAGE_DELTA,
                { id: 'step_entry', delta: { content: 'entry' } },
                'parent-agent'
              );
              await emit(
                GraphEvents.ON_REASONING_DELTA,
                { id: 'step_entry', delta: { content: [] } },
                'parent-agent'
              );
              await emit(
                GraphEvents.ON_RUN_STEP,
                {
                  id: 'step_worker',
                  type: StepTypes.MESSAGE_CREATION,
                  agentId: 'worker',
                },
                'entry'
              );
              await emit(
                GraphEvents.ON_RUN_STEP,
                {
                  id: 'step_spoofed',
                  type: StepTypes.MESSAGE_CREATION,
                  agentId: 'unknown-member',
                },
                'worker'
              );
              await emit(
                GraphEvents.ON_RUN_STEP_COMPLETED,
                { result: { id: 'step_result', agentId: 'result' } },
                'parent-agent'
              );
              await emit(
                GraphEvents.ON_MESSAGE_DELTA,
                { id: 'step_metadata', delta: { content: 'worker' } },
                'worker'
              );
              return {
                messages: [new AIMessage('ok')],
                subagentResult: { agentId: 'result' },
              };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;
      const executor = createExecutor({
        configs: new Map([[graphConfig.type, graphConfig]]),
        createChildGraphByKind: graphFactory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Run the graph.',
        subagentType: graphConfig.type,
      });

      const activity = updates.filter(
        (update) => update.phase !== 'start' && update.phase !== 'stop'
      );
      expect(
        activity.map(({ phase, memberAgentId }) => ({
          phase,
          memberAgentId,
        }))
      ).toEqual([
        { phase: 'run_step', memberAgentId: 'entry' },
        { phase: 'message_delta', memberAgentId: 'entry' },
        { phase: 'reasoning_delta', memberAgentId: 'entry' },
        { phase: 'run_step', memberAgentId: 'worker' },
        { phase: 'run_step', memberAgentId: undefined },
        { phase: 'run_step_completed', memberAgentId: 'result' },
        { phase: 'message_delta', memberAgentId: 'worker' },
      ]);
      expect(updates[0].memberAgentId).toBeUndefined();
      expect(updates[updates.length - 1].memberAgentId).toBeUndefined();
    });

    it('keeps toolDefinitions on child when registry has ON_TOOL_EXECUTE handler', async () => {
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_TOOL_EXECUTE, {
        handle: (): void => {},
      });
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        parentHandlerRegistry: registry,
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toHaveLength(1);
      expect(observedChildInputs?.toolDefinitions?.[0]?.name).toBe('web');
    });

    it('strips toolDefinitions when registry is present but ON_TOOL_EXECUTE handler is absent', async () => {
      const registry = new HandlerRegistry();
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        parentHandlerRegistry: registry,
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toBeUndefined();
    });

    it('forwards parentToolCallId from execute params to SubagentUpdateEvent envelopes', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_abc123',
      });

      expect(events.length).toBeGreaterThan(0);
      for (const e of events) {
        expect((e as { parentToolCallId?: string }).parentToolCallId).toBe(
          'call_abc123'
        );
      }
    });

    it('still strips toolDefinitions when no parentHandlerRegistry is provided (legacy isolation)', async () => {
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toBeUndefined();
    });

    it('accepts parentHandlerRegistry as a lazy getter', async () => {
      const lazyHolder: { registry?: InstanceType<typeof HandlerRegistry> } =
        {};
      const events: unknown[] = [];
      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: () => lazyHolder.registry,
      });

      lazyHolder.registry = new HandlerRegistry();
      lazyHolder.registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      expect(events.length).toBeGreaterThan(0);
      expect((events[0] as { phase: string }).phase).toBe('start');
    });

    it('routes child ON_TOOL_EXECUTE dispatches through the parent registry', async () => {
      /**
       * Drives the forwarder callback the executor installs on the child's
       * `workflow.invoke({ callbacks: [forwarder] })`. We capture that
       * callback when the child workflow runs, then synthesize the same
       * `handleCustomEvent` call that a real `ToolNode` would make when
       * the child LLM emits a tool_call. If the forwarder routes correctly,
       * the parent's `ON_TOOL_EXECUTE` handler receives the batch and
       * resolves the promise with our canned results.
       */

      const parentToolHandler = jest.fn(
        async (_event: string, rawData: unknown): Promise<void> => {
          const req = rawData as {
            toolCalls: Array<{ id: string; name: string }>;
            resolve: (results: unknown[]) => void;
          };
          req.resolve(
            req.toolCalls.map((tc) => ({
              toolCallId: tc.id,
              status: 'success',
              content: `ran ${tc.name}`,
            }))
          );
        }
      );

      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_TOOL_EXECUTE, {
        handle: parentToolHandler,
      });

      let capturedInvokeOptions: unknown;
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              capturedInvokeOptions = options;
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_parent_123',
      });

      const opts = capturedInvokeOptions as
        | { callbacks?: unknown[] }
        | undefined;
      expect(opts?.callbacks).toBeDefined();
      const forwarder = (opts?.callbacks ?? [])[0] as {
        handleCustomEvent?: (
          eventName: string,
          data: unknown,
          runId: string,
          tags?: string[],
          metadata?: Record<string, unknown>
        ) => Promise<void> | void;
      };
      expect(typeof forwarder.handleCustomEvent).toBe('function');

      /** Simulate the child's ToolNode emitting a real batch request. */
      const resolvePromise = new Promise<
        Array<{ toolCallId: string; status: string; content: string }>
      >((resolve, reject) => {
        const batchRequest = {
          toolCalls: [{ id: 'call_child_xyz', name: 'calculator', args: {} }],
          agentId: 'researcher',
          resolve,
          reject,
        };
        forwarder.handleCustomEvent?.(
          GraphEvents.ON_TOOL_EXECUTE,
          batchRequest,
          'child-run-id'
        );
      });

      const results = await resolvePromise;
      expect(parentToolHandler).toHaveBeenCalledTimes(1);
      expect(results).toEqual([
        {
          toolCallId: 'call_child_xyz',
          status: 'success',
          content: 'ran calculator',
        },
      ]);
    });

    it('sanitizes ON_TOOL_EXECUTE before wrapping it in ON_SUBAGENT_UPDATE', async () => {
      const toolRequests: ToolExecuteBatchRequest[] = [];
      const subagentUpdates: SubagentUpdateEvent[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_TOOL_EXECUTE, {
        handle: (_event, rawData): void => {
          const request = rawData as ToolExecuteBatchRequest;
          toolRequests.push(request);
          const results: ToolExecuteResult[] = request.toolCalls.map(
            (call) => ({
              toolCallId: call.id,
              status: 'success',
              content: `ran ${call.name}`,
            })
          );
          request.resolve(results);
        },
      });
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, rawData): void => {
          subagentUpdates.push(rawData as SubagentUpdateEvent);
        },
      });

      let capturedInvokeOptions: unknown;
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              capturedInvokeOptions = options;
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_parent_123',
      });

      const opts = capturedInvokeOptions as { callbacks?: unknown[] };
      const forwarder = (opts.callbacks ?? [])[0] as {
        handleCustomEvent?: (
          eventName: string,
          data: unknown
        ) => Promise<void> | void;
      };

      const batchRequest: ToolExecuteBatchRequest = {
        toolCalls: [
          {
            id: 'call_child_xyz',
            name: 'calculator',
            args: { expression: '21 * 2' },
            stepId: 'step_secret',
            turn: 7,
          },
        ],
        agentId: 'researcher',
        userId: 'user_secret',
        configurable: {
          user: {
            federatedTokens: {
              access_token: 'access-secret',
              id_token: 'id-secret',
              refresh_token: 'refresh-secret',
            },
          },
          requestBody: { currentTaskInput: 'sensitive task input' },
        },
        metadata: {
          access_token: 'metadata-secret',
        },
        resolve: jest.fn(),
        reject: jest.fn(),
      };

      await forwarder.handleCustomEvent?.(
        GraphEvents.ON_TOOL_EXECUTE,
        batchRequest
      );

      expect(toolRequests).toHaveLength(1);
      expect(toolRequests[0].configurable).toBe(batchRequest.configurable);
      expect(toolRequests[0].metadata).toBe(batchRequest.metadata);

      const toolUpdate = subagentUpdates.find(
        (update) =>
          update.phase === 'run_step' && update.label === 'Calling calculator'
      );
      expect(toolUpdate?.data).toEqual({
        agentId: 'researcher',
        toolCalls: [
          {
            id: 'call_child_xyz',
            name: 'calculator',
            args: { expression: '21 * 2' },
          },
        ],
      });
      const serializedUpdate = JSON.stringify(toolUpdate);
      expect(serializedUpdate).not.toContain('configurable');
      expect(serializedUpdate).not.toContain('metadata');
      expect(serializedUpdate).not.toContain('access-secret');
      expect(serializedUpdate).not.toContain('id-secret');
      expect(serializedUpdate).not.toContain('refresh-secret');
      expect(serializedUpdate).not.toContain('metadata-secret');
      expect(serializedUpdate).not.toContain('sensitive task input');
      expect(serializedUpdate).not.toContain('step_secret');
      expect(serializedUpdate).not.toContain('user_secret');
    });

    it('drains observational updates before stop without parallel handler publishes', async () => {
      const phases: SubagentUpdateEvent['phase'][] = [];
      let activePublishes = 0;
      let maxActivePublishes = 0;
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: async (_event, rawData): Promise<void> => {
          const update = rawData as SubagentUpdateEvent;
          activePublishes += 1;
          maxActivePublishes = Math.max(maxActivePublishes, activePublishes);
          if (update.phase === 'message_delta') {
            await new Promise((resolve) => setTimeout(resolve, 1));
          }
          phases.push(update.phase);
          activePublishes -= 1;
        },
      });

      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              const opts = options as { callbacks?: unknown[] };
              const forwarder = (opts.callbacks ?? [])[0] as {
                handleCustomEvent?: (
                  eventName: string,
                  data: unknown
                ) => Promise<void> | void;
              };
              for (let index = 0; index < 5; index++) {
                await forwarder.handleCustomEvent?.(
                  GraphEvents.ON_MESSAGE_DELTA,
                  {
                    id: `msg_${index}`,
                    delta: { content: [{ type: 'text', text: `${index}` }] },
                  }
                );
              }
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      expect(maxActivePublishes).toBe(1);
      expect(phases[0]).toBe('start');
      expect(phases.slice(1, 6)).toEqual([
        'message_delta',
        'message_delta',
        'message_delta',
        'message_delta',
        'message_delta',
      ]);
      expect(phases[phases.length - 1]).toBe('stop');
    });

    it('does not let a stalled observational update handler block child completion', async () => {
      jest.useFakeTimers();
      try {
        const registry = new HandlerRegistry();
        const updateHandler = jest.fn(
          (): Promise<void> => new Promise<void>(() => {})
        );
        registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
          handle: updateHandler,
        });
        const { factory } = makeStubGraphFactory({
          messages: [new AIMessage('ok')],
        });
        const executor = createExecutor({
          createChildGraph: factory,
          parentHandlerRegistry: registry,
        });

        const execution = executor.execute({
          description: 'Task',
          subagentType: 'researcher',
        });
        const outcome = Promise.race([
          execution.then(({ content }) => content),
          new Promise<string>((resolve) =>
            setTimeout(() => resolve('stalled'), 20_000)
          ),
        ]);

        await jest.advanceTimersByTimeAsync(20_000);

        await expect(outcome).resolves.toBe('ok');
        expect(updateHandler).toHaveBeenCalledTimes(2);
      } finally {
        jest.useRealTimers();
      }
    });

    it('does not let a stalled queued update block the terminal stop envelope', async () => {
      jest.useFakeTimers();
      try {
        const phases: SubagentUpdateEvent['phase'][] = [];
        const registry = new HandlerRegistry();
        registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
          handle: (_event, rawData): void | Promise<void> => {
            const update = rawData as SubagentUpdateEvent;
            phases.push(update.phase);
            if (update.phase === 'run_step') {
              return new Promise<void>(() => {});
            }
          },
        });
        const factory: () => StandardGraph = (): StandardGraph =>
          ({
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockImplementation(async (_state, options) => {
                const opts = options as { callbacks?: unknown[] };
                const forwarder = (opts.callbacks ?? [])[0] as {
                  handleCustomEvent?: (
                    eventName: string,
                    data: unknown
                  ) => Promise<void> | void;
                };
                for (let index = 0; index < 80; index++) {
                  await forwarder.handleCustomEvent?.(GraphEvents.ON_RUN_STEP, {
                    id: `step_${index}`,
                    type: StepTypes.MESSAGE_CREATION,
                    agentId: 'researcher',
                    index,
                  });
                }
                return { messages: [new AIMessage('ok')] };
              }),
            }),
            clearHeavyState: jest.fn(),
          }) as unknown as StandardGraph;
        const executor = createExecutor({
          createChildGraph: factory,
          parentHandlerRegistry: registry,
        });

        const execution = executor.execute({
          description: 'Task',
          subagentType: 'researcher',
        });
        const outcome = Promise.race([
          execution.then(({ content }) => content),
          new Promise<string>((resolve) =>
            setTimeout(() => resolve('stalled'), 20_000)
          ),
        ]);

        await jest.advanceTimersByTimeAsync(20_000);

        await expect(outcome).resolves.toBe('ok');
        expect(phases).toEqual(['start', 'run_step', 'stop']);
      } finally {
        jest.useRealTimers();
      }
    });

    it('allowlists forwarded run step payloads before wrapping them in ON_SUBAGENT_UPDATE', async () => {
      const subagentUpdates: SubagentUpdateEvent[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, rawData): void => {
          subagentUpdates.push(rawData as SubagentUpdateEvent);
        },
      });

      const output = 'tool output that should stay visible';
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              const opts = options as { callbacks?: unknown[] };
              const forwarder = (opts.callbacks ?? [])[0] as {
                handleCustomEvent?: (
                  eventName: string,
                  data: unknown
                ) => Promise<void> | void;
              };
              await forwarder.handleCustomEvent?.(GraphEvents.ON_RUN_STEP, {
                id: 'step_1',
                type: StepTypes.TOOL_CALLS,
                agentId: 'researcher',
                index: 0,
                stepDetails: {
                  type: StepTypes.TOOL_CALLS,
                  tool_calls: [
                    {
                      id: 'call_1',
                      name: 'calculator',
                      args: { expression: '21 * 2' },
                      futureSecret: 'nested-step-secret',
                    },
                  ],
                  futureSecret: 'step-details-secret',
                },
                configurable: { access_token: 'access-secret' },
                metadata: { refresh_token: 'refresh-secret' },
                futureSecret: 'top-level-step-secret',
              });
              await forwarder.handleCustomEvent?.(
                GraphEvents.ON_RUN_STEP_COMPLETED,
                {
                  result: {
                    id: 'step_1',
                    index: 0,
                    type: 'tool_call',
                    tool_call: {
                      id: 'call_1',
                      name: 'calculator',
                      args: '{}',
                      output,
                      progress: 1,
                      futureSecret: 'nested-completed-secret',
                    },
                    futureSecret: 'completed-result-secret',
                  },
                  configurable: { access_token: 'access-secret' },
                  metadata: { refresh_token: 'refresh-secret' },
                  futureSecret: 'top-level-completed-secret',
                }
              );
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      const runStep = subagentUpdates.find(
        (update) => update.phase === 'run_step'
      );
      const completedStep = subagentUpdates.find(
        (update) => update.phase === 'run_step_completed'
      );
      expect(runStep?.data).toEqual({
        id: 'step_1',
        type: StepTypes.TOOL_CALLS,
        agentId: 'researcher',
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '21 * 2' },
            },
          ],
        },
      });
      expect(completedStep?.data).toEqual({
        result: {
          id: 'step_1',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_1',
            name: 'calculator',
            args: '{}',
            output,
            progress: 1,
          },
        },
      });
      const serialized = JSON.stringify([runStep, completedStep]);
      expect(serialized).toContain(output);
      expect(serialized).not.toContain('futureSecret');
      expect(serialized).not.toContain('access-secret');
      expect(serialized).not.toContain('refresh-secret');
      expect(serialized).not.toContain('top-level-step-secret');
      expect(serialized).not.toContain('nested-step-secret');
      expect(serialized).not.toContain('top-level-completed-secret');
      expect(serialized).not.toContain('nested-completed-secret');
    });

    it('bounds queued updates under overload while preserving lifecycle envelopes', async () => {
      const phases: SubagentUpdateEvent['phase'][] = [];
      const completedIds: string[] = [];
      let releaseFirstCompleted!: () => void;
      const firstCompletedBlocked = new Promise<void>((resolve) => {
        releaseFirstCompleted = resolve;
      });
      let markAllEmitted!: () => void;
      const allEmitted = new Promise<void>((resolve) => {
        markAllEmitted = resolve;
      });
      let blockedFirstCompleted = false;
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: async (_event, rawData): Promise<void> => {
          const update = rawData as SubagentUpdateEvent;
          phases.push(update.phase);
          if (update.phase === 'run_step_completed') {
            const data = update.data as { result?: { id?: string } };
            if (data.result?.id != null) {
              completedIds.push(data.result.id);
            }
            if (!blockedFirstCompleted) {
              blockedFirstCompleted = true;
              await firstCompletedBlocked;
            }
          }
        },
      });

      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              const opts = options as { callbacks?: unknown[] };
              const forwarder = (opts.callbacks ?? [])[0] as {
                handleCustomEvent?: (
                  eventName: string,
                  data: unknown
                ) => Promise<void> | void;
              };
              for (let index = 0; index < 80; index++) {
                await forwarder.handleCustomEvent?.(
                  GraphEvents.ON_RUN_STEP_COMPLETED,
                  {
                    result: {
                      id: `step_${index}`,
                      index,
                      type: 'tool_call',
                      tool_call: {
                        id: `call_${index}`,
                        name: 'calculator',
                        args: '{}',
                        output: `${index}`,
                        progress: 1,
                      },
                    },
                  }
                );
              }
              markAllEmitted();
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      const execution = executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });
      await allEmitted;
      releaseFirstCompleted();
      await execution;

      expect(completedIds).toHaveLength(65);
      expect(completedIds[0]).toBe('step_0');
      expect(completedIds).not.toContain('step_1');
      expect(completedIds).toContain('step_16');
      expect(completedIds[completedIds.length - 1]).toBe('step_79');
      expect(phases[0]).toBe('start');
      expect(phases[phases.length - 1]).toBe('stop');
    });

    it('does NOT forward ON_TOOL_EXECUTE when the parent registry has no handler (safe fallback)', async () => {
      /**
       * The executor strips `toolDefinitions` when the parent registry has
       * no `ON_TOOL_EXECUTE` handler (see the companion strip-on-no-handler
       * test). Defence-in-depth: if the LLM somehow still dispatches a tool
       * call, the forwarder must not silently consume it without resolving;
       * reject would be better than hang. This test confirms no handler
       * is invoked on the parent side so it's clear a forwarded request
       * would need separate treatment.
       */

      const registry = new HandlerRegistry();
      /** Only ON_SUBAGENT_UPDATE registered — no ON_TOOL_EXECUTE. */
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, { handle: jest.fn() });

      let capturedInvokeOptions: unknown;
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              capturedInvokeOptions = options;
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      const opts = capturedInvokeOptions as { callbacks?: unknown[] };
      const forwarder = (opts.callbacks ?? [])[0] as {
        handleCustomEvent?: (
          eventName: string,
          data: unknown
        ) => Promise<void> | void;
      };

      let resolved = false;
      const batchRequest = {
        toolCalls: [{ id: 'call_x', name: 'calculator', args: {} }],
        agentId: 'researcher',
        resolve: (): void => {
          resolved = true;
        },
        reject: (): void => {},
      };
      await forwarder.handleCustomEvent?.(
        GraphEvents.ON_TOOL_EXECUTE,
        batchRequest
      );

      /** No handler exists → nothing resolves the promise. This is the
       *  state that justifies the `keepToolDefinitions` gate: without the
       *  gate we'd deadlock here. The gate ensures the LLM never sees
       *  tools in the first place, making this scenario unreachable in
       *  practice — the test just documents the fallback. */
      expect(resolved).toBe(false);
    });

    it('emits an `error` phase envelope when the child graph throws', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const executor = createExecutor({
        createChildGraph: makeThrowingGraphFactory(
          new Error('recursion limit')
        ),
        parentHandlerRegistry: registry,
      });

      const result = await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_err',
      });

      expect(result.content).toContain('Subagent error: recursion limit');
      const phases = events.map((e) => (e as { phase: string }).phase);
      expect(phases).toContain('start');
      expect(phases).toContain('error');
      const errEvent = events.find(
        (e) => (e as { phase: string }).phase === 'error'
      ) as { data?: { message?: string }; parentToolCallId?: string };
      expect(errEvent.data?.message).toContain('recursion limit');
      expect(errEvent.parentToolCallId).toBe('call_err');
    });
  });
});

describe('summarizeEvent', () => {
  it('labels a run step tool_calls stepDetails by tool name', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'calculator', id: 'c1' }],
      },
    });
    expect(label).toBe('Using tool: calculator');
  });

  it('joins multiple tool names on a single run step', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'web' }, { name: 'calculator' }],
      },
    });
    expect(label).toBe('Using tool: web, calculator');
  });

  it('falls back to "Planning tool call" when tool_calls is empty', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: { type: 'tool_calls', tool_calls: [] },
    });
    expect(label).toBe('Planning tool call');
  });

  it('labels message_creation steps as "Thinking…"', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: { type: 'message_creation' },
    });
    expect(label).toBe('Thinking…');
  });

  it('labels ON_TOOL_EXECUTE with the batch of tool names', () => {
    const label = summarizeEvent(GraphEvents.ON_TOOL_EXECUTE, {
      toolCalls: [{ name: 'web' }, { name: 'calculator' }],
    });
    expect(label).toBe('Calling web, calculator');
  });

  it('falls back to a generic "Calling tool" when toolCalls is empty', () => {
    const label = summarizeEvent(GraphEvents.ON_TOOL_EXECUTE, {
      toolCalls: [],
    });
    expect(label).toBe('Calling tool');
  });

  it('labels completed run steps by completed tool name', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP_COMPLETED, {
      result: { type: 'tool_call', tool_call: { name: 'calculator' } },
    });
    expect(label).toBe('Tool calculator complete');
  });

  it('labels completed steps without a tool name as "Step complete"', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP_COMPLETED, {
      result: { type: 'message_creation' },
    });
    expect(label).toBe('Step complete');
  });

  it('labels ON_MESSAGE_DELTA as "Streaming…"', () => {
    expect(summarizeEvent(GraphEvents.ON_MESSAGE_DELTA, {})).toBe('Streaming…');
  });

  it('falls back to top-level `step.type` when `stepDetails` is absent', () => {
    /**
     * Covers the `step.stepDetails?.type ?? step.type ?? 'step'` chain
     * when the payload uses the top-level form (no `stepDetails` wrapper).
     * Exercises the second clause of the fallback so future changes to
     * the resolution order fail fast.
     */
    expect(
      summarizeEvent(GraphEvents.ON_RUN_STEP, { type: 'tool_calls' })
    ).toBe('Planning tool call');
    expect(
      summarizeEvent(GraphEvents.ON_RUN_STEP, { type: 'message_creation' })
    ).toBe('Thinking…');
  });

  it('falls back to "Step: step" when neither `stepDetails.type` nor `step.type` is present', () => {
    /** Exercises the final `?? 'step'` default plus the generic
     *  `Step: <detailType>` branch when a run step arrives with an
     *  unrecognized shape. */
    expect(summarizeEvent(GraphEvents.ON_RUN_STEP, {})).toBe('Step: step');
  });

  it('returns the event name for unknown events', () => {
    expect(summarizeEvent('on_unknown_event', {})).toBe('on_unknown_event');
  });
});

describe('sanitizeForwardedSubagentUpdateData', () => {
  it('preserves bounded assistant phase metadata on message-creation steps', () => {
    const sanitized = sanitizeForwardedSubagentUpdateData(
      GraphEvents.ON_RUN_STEP,
      {
        id: 'step_1',
        stepDetails: {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: {
            message_id: 'message_1',
            content_type: 'text',
            phase: 'commentary',
            futureSecret: 'nested-secret',
          },
        },
      }
    );

    expect(sanitized).toEqual({
      id: 'step_1',
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: {
          message_id: 'message_1',
          content_type: 'text',
          phase: 'commentary',
        },
      },
    });

    expect(
      sanitizeForwardedSubagentUpdateData(GraphEvents.ON_RUN_STEP, {
        stepDetails: {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: {
            message_id: 'message_2',
            content_type: 'image',
            phase: 'internal',
          },
        },
      })
    ).toEqual({
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'message_2' },
      },
    });
  });

  it('uses an allowlist for run step payloads', () => {
    const sanitized = sanitizeForwardedSubagentUpdateData(
      GraphEvents.ON_RUN_STEP,
      {
        id: 'step_1',
        type: StepTypes.TOOL_CALLS,
        agentId: 'researcher',
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '21 * 2' },
              futureSecret: 'nested-secret',
            },
          ],
          futureSecret: 'details-secret',
        },
        configurable: { access_token: 'access-secret' },
        metadata: { refresh_token: 'refresh-secret' },
        futureSecret: 'top-level-secret',
      }
    );

    expect(sanitized).toEqual({
      id: 'step_1',
      type: StepTypes.TOOL_CALLS,
      agentId: 'researcher',
      index: 0,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { expression: '21 * 2' },
          },
        ],
      },
    });
    const serialized = JSON.stringify(sanitized);
    expect(serialized).not.toContain('futureSecret');
    expect(serialized).not.toContain('top-level-secret');
    expect(serialized).not.toContain('details-secret');
    expect(serialized).not.toContain('nested-secret');
    expect(serialized).not.toContain('access-secret');
    expect(serialized).not.toContain('refresh-secret');
  });

  it('forwards run step lifecycle stamps and closed events via allowlists', () => {
    const sanitizedStep = sanitizeForwardedSubagentUpdateData(
      GraphEvents.ON_RUN_STEP,
      {
        id: 'step_1',
        type: StepTypes.MESSAGE_CREATION,
        index: 0,
        created_at: 1_000,
        status: 'in_progress',
        stepDetails: {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: { message_id: 'message_1' },
        },
      }
    ) as { created_at?: number; status?: string };
    expect(sanitizedStep.created_at).toBe(1_000);
    expect(sanitizedStep.status).toBe('in_progress');

    const sanitizedClosed = sanitizeForwardedSubagentUpdateData(
      GraphEvents.ON_RUN_STEP_CLOSED,
      {
        id: 'step_1',
        index: 0,
        type: StepTypes.MESSAGE_CREATION,
        status: 'completed',
        created_at: 1_000,
        closed_at: 2_000,
        runId: 'run_1',
        agentId: 'researcher',
        futureSecret: 'top-level-secret',
      }
    );
    expect(sanitizedClosed).toEqual({
      id: 'step_1',
      index: 0,
      type: StepTypes.MESSAGE_CREATION,
      status: 'completed',
      created_at: 1_000,
      closed_at: 2_000,
      runId: 'run_1',
      agentId: 'researcher',
    });
    expect(JSON.stringify(sanitizedClosed)).not.toContain('top-level-secret');
  });

  it('keeps completed tool output while stripping operational fields', () => {
    const output = 'x'.repeat(10_000);
    const sanitized = sanitizeForwardedSubagentUpdateData(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: 'step_1',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_1',
            name: 'list_tables_mcp_ClickHouse',
            args: '{}',
            output,
            progress: 1,
            futureSecret: 'nested-secret',
          },
          futureSecret: 'result-secret',
        },
        configurable: {
          user: {
            federatedTokens: {
              access_token: 'access-secret',
            },
          },
        },
        metadata: {
          refresh_token: 'refresh-secret',
        },
        futureSecret: 'top-level-secret',
      }
    );

    expect(sanitized).toEqual({
      result: {
        id: 'step_1',
        index: 0,
        type: 'tool_call',
        tool_call: {
          id: 'call_1',
          name: 'list_tables_mcp_ClickHouse',
          args: '{}',
          output,
          progress: 1,
        },
      },
    });
    const serialized = JSON.stringify(sanitized);
    expect(serialized).toContain(output);
    expect(serialized).not.toContain('futureSecret');
    expect(serialized).not.toContain('top-level-secret');
    expect(serialized).not.toContain('result-secret');
    expect(serialized).not.toContain('nested-secret');
    expect(serialized).not.toContain('configurable');
    expect(serialized).not.toContain('metadata');
    expect(serialized).not.toContain('access-secret');
    expect(serialized).not.toContain('refresh-secret');
  });
});

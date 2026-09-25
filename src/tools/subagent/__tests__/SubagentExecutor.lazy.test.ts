import { describe, expect, it, jest } from '@jest/globals';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { GraphInterrupt, MemorySaver } from '@langchain/langgraph';
import type { Checkpoint, CheckpointMetadata } from '@langchain/langgraph';
import type {
  AgentInputs,
  ExecutableSubagentConfigEntry,
  GraphSubagentConfig,
  LazySingleAgentSubagentConfig,
  MultiAgentGraphState,
  ResolvedSubagentConfig,
  SubagentResolveConfigurable,
  StandardGraphInput,
  SubagentResolveContext,
} from '@/types';
import type {
  SubagentResumeExecution,
  SubagentResumeManifest,
  SettledSubagentToolOutput,
} from '@/tools/subagent/SubagentReplay';
import type {
  PreparedSubagentExecutionIdentity,
  SubagentExecutionIdentity,
} from '@/tools/subagent/SubagentExecutionRegistry';
import type { GraphFactoryRequest } from '@/graphs/graphFactory';
import type { StandardGraph } from '@/graphs/Graph';
import {
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY,
  SUBAGENT_RESUME_MANIFEST_CONFIG_KEY,
} from '@/tools/subagent/SubagentReplay';
import {
  getSubagentApprovalExecutionScope,
  SubagentExecutionRegistry,
} from '@/tools/subagent/SubagentExecutionRegistry';
import {
  RUN_BREAKER_SCOPE_CONFIG_KEY,
  StreamLimitExceededError,
} from '@/llm/streamLimits';
import { normalizeSubagentConfigEntries } from '@/tools/subagent/childGraphConfig';
import {
  HookRegistry,
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
} from '@/hooks';
import { SubagentExecutor } from '@/tools/subagent/SubagentExecutor';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const makeAgent = (agentId = 'child-agent'): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
  maxContextTokens: 8000,
});

const makeLazyConfig = (
  type: string,
  resolveAgentInputs: LazySingleAgentSubagentConfig['resolveAgentInputs'],
  overrides: Partial<LazySingleAgentSubagentConfig> = {}
): LazySingleAgentSubagentConfig => ({
  type,
  name: `${type} worker`,
  description: `Handles ${type} tasks.`,
  configId: `${type}@v1`,
  resolveAgentInputs,
  ...overrides,
});

const makeConfigMap = (
  ...configs: ExecutableSubagentConfigEntry[]
): Map<string, ExecutableSubagentConfigEntry> =>
  new Map(configs.map((config) => [config.type, config]));

const makeGraph = (
  result: MultiAgentGraphState = {
    messages: [new AIMessage('Task completed')],
  }
): StandardGraph =>
  ({
    createWorkflow: () => ({
      invoke: jest.fn(async (): Promise<MultiAgentGraphState> => result),
    }),
    clearHeavyState: jest.fn(),
  }) as unknown as StandardGraph;

const makeRecoveredGraph = (): StandardGraph =>
  ({
    createWorkflow: () => ({
      getState: jest.fn(async () => ({
        values: { messages: [new AIMessage('persisted result')] },
        next: [],
        tasks: [],
      })),
      invoke: jest.fn(),
      updateState: jest.fn(async () => ({})),
    }),
    restoreSubagentResumeState: jest.fn(),
    createSubagentResumeState: jest.fn(() => ({
      toolCallSteps: [],
      toolSessions: [],
      toolNodes: [],
      eagerToolUsage: [],
      eagerToolSuppressions: [],
    })),
    clearHeavyState: jest.fn(),
  }) as unknown as StandardGraph;

const makeResumeExecution = (
  parentToolCallId: string,
  configId: string,
  checkpointThreadId = 'resume-source-thread',
  subagentType = 'researcher'
): SubagentResumeExecution => ({
  parentToolCallId,
  childRunId: 'original-child-run',
  subagentType,
  configId,
  approvalExecutionScope: 'original-approval-scope',
  checkpoints: [
    {
      threadId: checkpointThreadId,
      checkpointId: 'resume-checkpoint',
      checkpointNs: '',
    },
  ],
  graphState: {
    toolCallSteps: [],
    toolSessions: [],
    toolNodes: [],
    eagerToolUsage: [],
    eagerToolSuppressions: [],
  },
  approvalReplays: [],
});

const putCheckpoint = async (
  checkpointer: MemorySaver,
  threadId: string,
  checkpointId: string
): Promise<void> => {
  const checkpoint: Checkpoint = {
    v: 4,
    id: checkpointId,
    ts: new Date().toISOString(),
    channel_values: {},
    channel_versions: {},
    versions_seen: {},
  };
  const metadata: CheckpointMetadata = {
    source: 'loop',
    step: 0,
    parents: {},
  };
  await checkpointer.put(
    { configurable: { thread_id: threadId, checkpoint_ns: '' } },
    checkpoint,
    metadata
  );
};

const createExecutor = (
  configs: ExecutableSubagentConfigEntry[],
  overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
): SubagentExecutor =>
  new SubagentExecutor({
    configs: makeConfigMap(...configs),
    parentRunId: 'parent-run',
    parentAgentId: 'parent-agent',
    createChildGraph: () => makeGraph(),
    ...overrides,
  });

describe('SubagentExecutor lazy selected-subagent resolution', () => {
  it('resolves only the selected descriptor', async () => {
    const researcherResolver = jest.fn(async () =>
      makeAgent('lazy-researcher')
    );
    const coderResolver = jest.fn(async () => makeAgent('lazy-coder'));
    const researcher = makeLazyConfig('researcher', researcherResolver);
    const coder = makeLazyConfig('coder', coderResolver);
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([researcher, coder], {
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Research this.',
      subagentType: researcher.type,
      parentToolCallId: 'call_research',
    });

    expect(researcherResolver).toHaveBeenCalledTimes(1);
    expect(coderResolver).not.toHaveBeenCalled();
    expect(observedInput?.agents[0].agentId).toBe('lazy-researcher');
  });

  it('redacts a selected resolver failure without affecting a healthy sibling', async () => {
    const failingConfig = makeLazyConfig('failing', async () => {
      throw new Error('oauth-token=private-selected-resolver-secret');
    });
    const healthyResolver = jest.fn(async () => makeAgent('healthy-child'));
    const healthyConfig = makeLazyConfig('healthy', healthyResolver);
    const executor = createExecutor([failingConfig, healthyConfig]);

    const failed = await executor.execute({
      description: 'Fail this selected child.',
      subagentType: failingConfig.type,
      parentToolCallId: 'call_failing',
    });
    const succeeded = await executor.execute({
      description: 'Run the healthy child.',
      subagentType: healthyConfig.type,
      parentToolCallId: 'call_healthy',
    });

    expect(failed.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(failed.content).not.toContain('oauth-token');
    expect(failed.content).not.toContain('private-selected-resolver-secret');
    expect(succeeded.content).toBe('Task completed');
    expect(healthyResolver).toHaveBeenCalledTimes(1);
  });

  it('releases lazy resolution when SubagentStart denies', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const hookRegistry = new HookRegistry();
    hookRegistry.register('SubagentStart', {
      hooks: [
        async (): Promise<{ decision: 'deny'; reason: string }> => ({
          decision: 'deny',
          reason: 'blocked by policy',
        }),
      ],
    });
    const executor = createExecutor([config], { hookRegistry });
    const params = {
      description: 'Block this selected child.',
      subagentType: config.type,
      parentToolCallId: 'call_blocked',
    };

    const first = await executor.execute(params);
    expect(first.content).toBe('Blocked: blocked by policy');

    const retry = await executor.execute(params);

    expect(retry.content).toBe('Blocked: blocked by policy');
    expect(resolver).toHaveBeenCalledTimes(2);
  });

  it('coalesces concurrent execution and releases resolved inputs after completion', async () => {
    let finishResolution = (_inputs: AgentInputs): void => undefined;
    const resolverResult = new Promise<AgentInputs>((resolve) => {
      finishResolution = resolve;
    });
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const resolver = jest.fn(async () => {
      markResolutionStarted();
      return resolverResult;
    });
    const config = makeLazyConfig('researcher', resolver);
    const invoke = jest.fn(
      async (): Promise<MultiAgentGraphState> => ({
        messages: [new AIMessage('Task completed')],
      })
    );
    const executor = createExecutor([config], {
      parentRunId: 'coalesced-run',
      createChildGraph: () =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const params = {
      description: 'Run the same selected execution.',
      subagentType: config.type,
      parentToolCallId: 'call_same',
    };

    const first = executor.execute(params);
    const duplicate = executor.execute(params);
    await resolutionStarted;
    expect(resolver).toHaveBeenCalledTimes(1);

    finishResolution(makeAgent('coalesced-child'));
    await Promise.all([first, duplicate]);
    expect(invoke).toHaveBeenCalledTimes(1);
    await executor.execute({
      ...params,
      parentToolCallId: 'call_fresh',
    });

    expect(resolver).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenCalledTimes(2);
  });

  it('rejects a different invocation before sharing an in-flight result', async () => {
    let finish = (_result: MultiAgentGraphState): void => undefined;
    const invocation = new Promise<MultiAgentGraphState>((resolve) => {
      finish = resolve;
    });
    const researcherResolver = jest.fn(async () => makeAgent('researcher'));
    const coderResolver = jest.fn(async () => makeAgent('coder'));
    const researcher = makeLazyConfig('researcher', researcherResolver);
    const coder = makeLazyConfig('coder', coderResolver);
    const invoke = jest.fn(() => invocation);
    const executor = createExecutor([researcher, coder], {
      createChildGraph: () =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const common = { parentToolCallId: 'call_immutable_invocation' };

    const first = executor.execute({
      ...common,
      description: 'Run the original request.',
      subagentType: researcher.type,
    });
    const changedType = executor.execute({
      ...common,
      description: 'Run the original request.',
      subagentType: coder.type,
    });
    const changedDescription = executor.execute({
      ...common,
      description: 'Run a changed request.',
      subagentType: researcher.type,
    });

    await expect(changedType).resolves.toMatchObject({
      content:
        'Subagent error: Subagent configuration changed since this execution was paused.',
    });
    await expect(changedDescription).resolves.toMatchObject({
      content:
        'Subagent error: Subagent invocation changed for this execution.',
    });
    expect(coderResolver).not.toHaveBeenCalled();
    finish({ messages: [new AIMessage('original result')] });
    await expect(first).resolves.toMatchObject({ content: 'original result' });
    expect(researcherResolver).toHaveBeenCalledTimes(1);
    expect(invoke).toHaveBeenCalledTimes(1);
  });

  it('does not execute a child after its lazy resolution was invalidated', async () => {
    let finishFirst = (_inputs: AgentInputs): void => undefined;
    let finishSecond = (_inputs: AgentInputs): void => undefined;
    const firstResult = new Promise<AgentInputs>((resolve) => {
      finishFirst = resolve;
    });
    const secondResult = new Promise<AgentInputs>((resolve) => {
      finishSecond = resolve;
    });
    let markFirstStarted = (): void => undefined;
    let markSecondStarted = (): void => undefined;
    const firstStarted = new Promise<void>((resolve) => {
      markFirstStarted = resolve;
    });
    const secondStarted = new Promise<void>((resolve) => {
      markSecondStarted = resolve;
    });
    let resolverInvocation = 0;
    const config = makeLazyConfig('researcher', () => {
      resolverInvocation += 1;
      if (resolverInvocation === 1) {
        markFirstStarted();
        return firstResult;
      }
      markSecondStarted();
      return secondResult;
    });
    const observedAgentIds: string[] = [];
    const createChildGraph = jest.fn((input: StandardGraphInput) => {
      observedAgentIds.push(input.agents[0].agentId);
      return makeGraph();
    });
    const executor = createExecutor([config], { createChildGraph });
    const params = {
      description: 'Run the selected child.',
      subagentType: config.type,
      parentToolCallId: 'call_cleanup_retry',
    };

    const staleExecution = executor.execute(params);
    await firstStarted;
    executor.clearHeavyState();
    const currentExecution = executor.execute(params);
    await secondStarted;

    finishFirst(makeAgent('stale-child'));
    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(createChildGraph).not.toHaveBeenCalled();

    finishSecond(makeAgent('current-child'));
    await expect(currentExecution).resolves.toMatchObject({
      content: 'Task completed',
    });
    expect(createChildGraph).toHaveBeenCalledTimes(1);
    expect(observedAgentIds).toEqual(['current-child']);
  });

  it('rolls back a stale checkpoint fork before committing its identity', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const checkpointer = new MemorySaver();
    const hookRegistry = new HookRegistry();
    const parentToolCallId = 'call_stale_fork';
    const resumeAttemptId = 'attempt-stale';
    const approvalExecutionScope = getSubagentApprovalExecutionScope(
      'original-child-run',
      resumeAttemptId
    );
    const approvalKey = {
      executionScope: approvalExecutionScope,
      agentId: 'child-agent',
      toolUseId: 'call_child_tool',
    };
    const resumeExecution = {
      ...makeResumeExecution(parentToolCallId, config.configId),
      approvalReplays: [
        {
          key: {
            ...approvalKey,
            executionScope: 'original-approval-scope',
          },
          result: {
            decision: 'ask' as const,
            reason: 'review child tool',
            additionalContexts: [],
            injectedMessages: [],
            errors: [],
          },
        },
      ],
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: resumeAttemptId,
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
        version: 1 as const,
        executions: [resumeExecution],
      },
    };
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    let releaseFork = (): void => undefined;
    let markForkStarted = (): void => undefined;
    const forkGate = new Promise<void>((resolve) => {
      releaseFork = resolve;
    });
    const forkStarted = new Promise<void>((resolve) => {
      markForkStarted = resolve;
    });
    const executor = createExecutor([config], {
      checkpointer,
      hookRegistry,
      humanInTheLoop: { enabled: true },
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: (
        sources: SubagentResumeExecution['checkpoints'],
        targetThreadId: string
      ) => Promise<void>;
    };
    jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockImplementation(async (_sources, targetThreadId) => {
        markForkStarted();
        await forkGate;
        await putCheckpoint(
          checkpointer,
          targetThreadId,
          'stale-child-checkpoint'
        );
      });
    const restoreApprovals = jest.spyOn(
      hookRegistry,
      'restorePendingToolApprovals'
    );
    const deleteThread = jest.spyOn(checkpointer, 'deleteThread');

    const staleExecution = executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await forkStarted;
    executor.clearHeavyState();
    releaseFork();

    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(deleteThread).toHaveBeenCalledWith(address.branchChildThreadId);
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeUndefined();
    expect(restoreApprovals).not.toHaveBeenCalled();
    expect(
      hookRegistry.getPendingToolApproval(approvalExecutionScope, approvalKey)
    ).toBeUndefined();
    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]), {
        configurable: parentConfigurable,
      })
    ).resolves.toBeUndefined();
    expect(resolver).not.toHaveBeenCalled();
  });

  it('retries a same-address execution only after an old fork rolls back', async () => {
    const resolver = jest.fn(async () => {
      throw new Error('stop after identity commit');
    });
    const config = makeLazyConfig('researcher', resolver);
    const checkpointer = new MemorySaver();
    const hookRegistry = new HookRegistry();
    const parentToolCallId = 'call_fork_owner_race';
    const resumeAttemptId = 'attempt-shared';
    const approvalExecutionScope = getSubagentApprovalExecutionScope(
      'original-child-run',
      resumeAttemptId
    );
    const approvalKey = {
      executionScope: approvalExecutionScope,
      agentId: 'child-agent',
      toolUseId: 'call_child_tool',
    };
    const approval = {
      decision: 'ask' as const,
      reason: 'review child tool',
      additionalContexts: [],
      injectedMessages: [],
      errors: [],
    };
    const resumeExecution = {
      ...makeResumeExecution(parentToolCallId, config.configId),
      approvalReplays: [
        {
          key: {
            ...approvalKey,
            executionScope: 'original-approval-scope',
          },
          result: approval,
        },
      ],
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: resumeAttemptId,
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
        version: 1 as const,
        executions: [resumeExecution],
      },
    };
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    let releaseOldFork = (): void => undefined;
    let markOldForkStarted = (): void => undefined;
    const oldForkGate = new Promise<void>((resolve) => {
      releaseOldFork = resolve;
    });
    const oldForkStarted = new Promise<void>((resolve) => {
      markOldForkStarted = resolve;
    });
    const executor = createExecutor([config], {
      checkpointer,
      hookRegistry,
      humanInTheLoop: { enabled: true },
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: (
        sources: SubagentResumeExecution['checkpoints'],
        targetThreadId: string
      ) => Promise<void>;
    };
    let forkInvocation = 0;
    jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockImplementation(async (_sources, targetThreadId) => {
        forkInvocation += 1;
        if (forkInvocation === 1) {
          markOldForkStarted();
          await oldForkGate;
          await putCheckpoint(
            checkpointer,
            targetThreadId,
            '00000000-0000-0000-0000-000000000001'
          );
          return;
        }
        await putCheckpoint(
          checkpointer,
          targetThreadId,
          '00000000-0000-0000-0000-000000000002'
        );
      });
    const deleteThread = jest.spyOn(checkpointer, 'deleteThread');
    const params = {
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    };

    const staleExecution = executor.execute(params);
    await oldForkStarted;
    executor.clearHeavyState();
    await expect(executor.execute(params)).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(
      hookRegistry.getPendingToolApproval(approvalExecutionScope, approvalKey)
    ).toBeUndefined();
    releaseOldFork();

    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(deleteThread).toHaveBeenCalledTimes(1);
    expect(deleteThread).toHaveBeenCalledWith(address.branchChildThreadId);
    await expect(executor.execute(params)).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeDefined();
    expect(
      hookRegistry.getPendingToolApproval(approvalExecutionScope, approvalKey)
    ).toEqual(approval);
    expect(executor.getChildCheckpointThreadIds()).toContain(
      address.branchChildThreadId
    );
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]), {
        configurable: parentConfigurable,
      })
    ).resolves.toBeDefined();
    expect(forkInvocation).toBe(2);
  });

  it('does not adopt a partial branch while its producer is pending', async () => {
    const resolver = jest.fn(async () => {
      throw new Error('stop after adopted identity commit');
    });
    const config = makeLazyConfig('researcher', resolver);
    const checkpointer = new MemorySaver();
    const parentToolCallId = 'call_partial_fork_adoption';
    const resumeAttemptId = 'attempt-adopted';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: resumeAttemptId,
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
        version: 1 as const,
        executions: [resumeExecution],
      },
    };
    const {
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: _resumeManifest,
      ...retryConfigurable
    } = parentConfigurable;
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    let releaseOldFork = (): void => undefined;
    let markPartialForkWritten = (): void => undefined;
    const oldForkGate = new Promise<void>((resolve) => {
      releaseOldFork = resolve;
    });
    const partialForkWritten = new Promise<void>((resolve) => {
      markPartialForkWritten = resolve;
    });
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: (
        sources: SubagentResumeExecution['checkpoints'],
        targetThreadId: string
      ) => Promise<void>;
    };
    const forkCheckpointSnapshot = jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockImplementation(async (_sources, targetThreadId) => {
        await putCheckpoint(
          checkpointer,
          targetThreadId,
          '00000000-0000-0000-0000-000000000001'
        );
        markPartialForkWritten();
        await oldForkGate;
      });
    const deleteThread = jest.spyOn(checkpointer, 'deleteThread');
    const staleExecution = executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await partialForkWritten;
    executor.clearHeavyState();

    await expect(
      executor.execute({
        description: 'Adopt the partially forked child.',
        subagentType: config.type,
        threadId: retryConfigurable.thread_id,
        parentToolCallId,
        parentConfigurable: retryConfigurable,
      })
    ).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(forkCheckpointSnapshot).toHaveBeenCalledTimes(1);
    releaseOldFork();

    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(deleteThread).toHaveBeenCalledTimes(1);
    expect(deleteThread).toHaveBeenCalledWith(address.branchChildThreadId);
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeUndefined();
    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]), {
        configurable: retryConfigurable,
      })
    ).resolves.toBeUndefined();
    expect(resolver).not.toHaveBeenCalled();
  });

  it('fences retries until a late checkpoint writer and cleanup settle', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const checkpointer = new MemorySaver();
    const parentToolCallId = 'call_partial_fork_failed_adoption';
    const resumeAttemptId = 'attempt-failed-adoption';
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: resumeAttemptId,
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
        version: 1 as const,
        executions: [makeResumeExecution(parentToolCallId, config.configId)],
      },
    };
    const {
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: _resumeManifest,
      ...retryConfigurable
    } = parentConfigurable;
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    let releaseLateWrite = (): void => undefined;
    let markLateWriteFinished = (): void => undefined;
    let releaseOldFork = (): void => undefined;
    let markPartialForkWritten = (): void => undefined;
    const lateWriteGate = new Promise<void>((resolve) => {
      releaseLateWrite = resolve;
    });
    const lateWriteFinished = new Promise<void>((resolve) => {
      markLateWriteFinished = resolve;
    });
    const oldForkGate = new Promise<void>((resolve) => {
      releaseOldFork = resolve;
    });
    const partialForkWritten = new Promise<void>((resolve) => {
      markPartialForkWritten = resolve;
    });
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
      createChildGraph: makeRecoveredGraph,
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: (
        sources: SubagentResumeExecution['checkpoints'],
        targetThreadId: string
      ) => Promise<void>;
    };
    let forkInvocation = 0;
    const forkCheckpointSnapshot = jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockImplementation(async (_sources, targetThreadId) => {
        forkInvocation += 1;
        if (forkInvocation > 1) {
          await putCheckpoint(
            checkpointer,
            targetThreadId,
            '00000000-0000-0000-0000-000000000002'
          );
          return;
        }
        await putCheckpoint(
          checkpointer,
          targetThreadId,
          '00000000-0000-0000-0000-000000000001'
        );
        markPartialForkWritten();
        await lateWriteGate;
        await putCheckpoint(
          checkpointer,
          targetThreadId,
          '00000000-0000-0000-0000-000000000003'
        );
        markLateWriteFinished();
        await oldForkGate;
      });
    let releaseDelete = (): void => undefined;
    let markDeleteStarted = (): void => undefined;
    const deleteGate = new Promise<void>((resolve) => {
      releaseDelete = resolve;
    });
    const deleteStarted = new Promise<void>((resolve) => {
      markDeleteStarted = resolve;
    });
    const deleteCheckpointThread = checkpointer.deleteThread.bind(checkpointer);
    const deleteThread = jest
      .spyOn(checkpointer, 'deleteThread')
      .mockImplementation(async (threadId) => {
        markDeleteStarted();
        await deleteGate;
        await deleteCheckpointThread(threadId);
      });
    const staleExecution = executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await partialForkWritten;
    executor.clearHeavyState();

    const retryParams = {
      description: 'Adopt the partially forked child.',
      subagentType: config.type,
      threadId: retryConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable: retryConfigurable,
    };
    await expect(executor.execute(retryParams)).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(deleteThread).not.toHaveBeenCalled();
    releaseLateWrite();
    await lateWriteFinished;
    await expect(executor.execute(retryParams)).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(deleteThread).not.toHaveBeenCalled();
    releaseOldFork();
    await deleteStarted;
    await expect(executor.execute(retryParams)).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    releaseDelete();
    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeUndefined();

    await expect(
      executor.execute({
        ...retryParams,
        parentConfigurable,
      })
    ).resolves.toMatchObject({ content: 'persisted result' });

    expect(deleteThread).toHaveBeenCalledTimes(1);
    expect(deleteThread).toHaveBeenCalledWith(address.branchChildThreadId);
    expect(forkCheckpointSnapshot).toHaveBeenCalledTimes(2);
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeDefined();
    expect(executor.getChildCheckpointThreadIds()).toContain(
      address.branchChildThreadId
    );
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]), {
        configurable: parentConfigurable,
      })
    ).resolves.toBeDefined();
    expect(resolver).toHaveBeenCalledTimes(1);
  });

  it('preserves a preexisting branch when its adopter is invalidated', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const checkpointer = new MemorySaver();
    const parentToolCallId = 'call_preexisting_failed_adoption';
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-preexisting-adoption',
    };
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    await putCheckpoint(
      checkpointer,
      address.branchChildThreadId,
      '00000000-0000-0000-0000-000000000001'
    );
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const identityTarget = executor as unknown as {
      prepareIdentity(
        execution: object,
        identity: SubagentExecutionIdentity
      ): PreparedSubagentExecutionIdentity;
    };
    const prepareIdentity = identityTarget.prepareIdentity.bind(identityTarget);
    jest
      .spyOn(identityTarget, 'prepareIdentity')
      .mockImplementation((execution, identity) => {
        const preparation = prepareIdentity(execution, identity);
        queueMicrotask(() => executor.clearHeavyState());
        return preparation;
      });
    const deleteThread = jest.spyOn(checkpointer, 'deleteThread');

    await expect(
      executor.execute({
        description: 'Adopt the existing child branch.',
        subagentType: config.type,
        threadId: parentConfigurable.thread_id,
        parentToolCallId,
        parentConfigurable,
      })
    ).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });

    expect(deleteThread).not.toHaveBeenCalled();
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeDefined();
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]), {
        configurable: parentConfigurable,
      })
    ).resolves.toBeUndefined();
    expect(resolver).not.toHaveBeenCalled();
  });

  it.each([
    'resume manifest fork',
    'branch adoption',
    'missing base',
    'missing base with explicit attempt',
    'base reuse',
    'base fork',
  ] as const)('leases the exact HITL identity for %s', async (path) => {
    const config = makeLazyConfig('researcher', async () => {
      throw new Error('stop after identity commit');
    });
    const checkpointer = new MemorySaver();
    const parentToolCallId = `call_identity_${path.replaceAll(' ', '_')}`;
    const explicitResumeAttempt =
      path === 'resume manifest fork' ||
      path === 'branch adoption' ||
      path === 'missing base with explicit attempt' ||
      path === 'base fork';
    const parentConfigurable: Record<string, unknown> = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      ...(explicitResumeAttempt
        ? { [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-exact' }
        : {}),
    };
    if (path === 'resume manifest fork') {
      parentConfigurable[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY] = {
        version: 1,
        executions: [makeResumeExecution(parentToolCallId, config.configId)],
      };
      await putCheckpoint(
        checkpointer,
        'resume-source-thread',
        'resume-checkpoint'
      );
    }
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable,
    }).address;
    if (path === 'branch adoption') {
      await putCheckpoint(
        checkpointer,
        address.branchChildThreadId,
        '00000000-0000-0000-0000-000000000001'
      );
    }
    if (path === 'base reuse' || path === 'base fork') {
      await putCheckpoint(
        checkpointer,
        address.baseChildThreadId,
        '00000000-0000-0000-0000-000000000001'
      );
    }
    const beginPreparation = jest.spyOn(
      SubagentExecutionRegistry.prototype,
      'beginIdentityPreparation'
    );
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const expectedChildRunId =
      path === 'resume manifest fork'
        ? 'original-child-run'
        : address.currentChildRunId;
    const expectedChildThreadId =
      path === 'missing base' || path === 'base reuse'
        ? address.baseChildThreadId
        : address.branchChildThreadId;

    try {
      await expect(
        executor.execute({
          description: 'Resolve the selected child identity.',
          subagentType: config.type,
          threadId: 'durable-thread',
          parentToolCallId,
          parentConfigurable,
        })
      ).resolves.toMatchObject({
        content: 'Subagent error: Unable to initialize the selected subagent.',
      });
      expect(beginPreparation).toHaveBeenCalledTimes(1);
      expect(beginPreparation.mock.calls[0][0].address.key).toBe(address.key);
      expect(beginPreparation.mock.calls[0][1]).toEqual({
        childRunId: expectedChildRunId,
        childThreadId: expectedChildThreadId,
        approvalExecutionScope: getSubagentApprovalExecutionScope(
          expectedChildRunId,
          address.resumeAttemptId
        ),
      });
    } finally {
      beginPreparation.mockRestore();
    }
  });

  it('preserves a preexisting branch when a later fork fails', async () => {
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const checkpointer = new MemorySaver();
    const parentToolCallId = 'call_preexisting_fork_failure';
    const resumeAttemptId = 'attempt-existing';
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'parent-batch',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: resumeAttemptId,
      [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
        version: 1 as const,
        executions: [makeResumeExecution(parentToolCallId, config.configId)],
      },
    };
    const address = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    }).open({
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    }).address;
    await putCheckpoint(
      checkpointer,
      address.branchChildThreadId,
      '00000000-0000-0000-0000-000000000001'
    );
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
    };
    const forkError = new Error('fork failed after finding target');
    jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockRejectedValue(forkError);
    const deleteThread = jest.spyOn(checkpointer, 'deleteThread');

    await expect(
      executor.execute({
        description: 'Resume the selected child.',
        subagentType: config.type,
        threadId: parentConfigurable.thread_id,
        parentToolCallId,
        parentConfigurable,
      })
    ).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });

    expect(deleteThread).not.toHaveBeenCalled();
    await expect(
      checkpointer.getTuple({
        configurable: { thread_id: address.branchChildThreadId },
      })
    ).resolves.toBeDefined();
  });

  it('rejects a missing durable tool call ID before opening an execution record', async () => {
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const openExecution = jest.spyOn(
      SubagentExecutionRegistry.prototype,
      'open'
    );
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Do not register this child.',
      subagentType: config.type,
    });

    expect(result.content).toBe(
      'Error: Resumable subagent execution requires a parent tool call ID.'
    );
    expect(openExecution).not.toHaveBeenCalled();
    openExecution.mockRestore();
  });

  it('preserves a workflow construction error and allows a retry', async () => {
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const clearFailedGraph = jest.fn();
    let graphCount = 0;
    const executor = createExecutor([config], {
      createChildGraph: (): StandardGraph => {
        graphCount += 1;
        if (graphCount > 1) {
          return makeGraph();
        }
        return {
          createWorkflow: () => {
            throw new Error('workflow setup failed');
          },
          clearHeavyState: clearFailedGraph,
        } as unknown as StandardGraph;
      },
    });
    const params = {
      description: 'Retry workflow construction.',
      subagentType: config.type,
      parentToolCallId: 'call_workflow_retry',
    };

    const failed = await executor.execute(params);
    const retried = await executor.execute(params);

    expect(failed.content).toBe('Subagent error: workflow setup failed');
    expect(retried.content).toBe('Task completed');
    expect(clearFailedGraph).toHaveBeenCalledTimes(1);
  });

  it('keeps concurrent checkpoint forks as distinct child executions', async () => {
    const resolverContexts: SubagentResolveContext[] = [];
    let releaseInvocations = (): void => undefined;
    const invocationGate = new Promise<void>((resolve) => {
      releaseInvocations = resolve;
    });
    let invocationCount = 0;
    let markInvocationsStarted = (): void => undefined;
    const invocationsStarted = new Promise<void>((resolve) => {
      markInvocationsStarted = resolve;
    });
    const invoke = jest.fn(async (): Promise<MultiAgentGraphState> => {
      invocationCount += 1;
      const current = invocationCount;
      if (invocationCount === 2) {
        markInvocationsStarted();
      }
      await invocationGate;
      return { messages: [new AIMessage(`result-${current}`)] };
    });
    const config = makeLazyConfig('researcher', async (context) => {
      resolverContexts.push(context);
      return makeAgent();
    });
    const executor = createExecutor([config], {
      createChildGraph: () =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const common = {
      description: 'Run this selected execution.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId: 'call_reused',
    };

    const first = executor.execute({
      ...common,
      parentConfigurable: { checkpoint_id: 'fork-a' },
    });
    const second = executor.execute({
      ...common,
      parentConfigurable: { checkpoint_id: 'fork-b' },
    });
    await invocationsStarted;
    const callsBeforeRelease = invoke.mock.calls.length;
    releaseInvocations();
    const [firstResult, secondResult] = await Promise.all([first, second]);

    expect(callsBeforeRelease).toBe(2);
    expect(resolverContexts).toHaveLength(2);
    expect(
      new Set(resolverContexts.map((context) => context.executionId)).size
    ).toBe(2);
    expect(new Set([firstResult.content, secondResult.content])).toEqual(
      new Set(['result-1', 'result-2'])
    );
  });

  it('scopes resume manifests to concurrent forks sharing a tool call ID', async () => {
    const checkpointer = new MemorySaver();
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_shared_forks';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Run the forked child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const forkAConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'fork-a',
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
      },
    };
    const forkBConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'fork-b',
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-b',
      },
    };
    const forkCConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'fork-a',
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-c',
      },
    };
    const makeSettledOutput = (content: string): SettledSubagentToolOutput => ({
      output: new ToolMessage({
        content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: call.args,
    });

    await Promise.all([
      executor.persistSettledToolOutput(
        call,
        forkAConfig,
        makeSettledOutput('fork-a result')
      ),
      executor.persistSettledToolOutput(
        call,
        forkBConfig,
        makeSettledOutput('fork-b result')
      ),
    ]);

    const parentToolCallIds = new Set([parentToolCallId]);
    const [manifestA, manifestB] = await Promise.all([
      executor.getResumeManifest(parentToolCallIds, forkAConfig),
      executor.getResumeManifest(parentToolCallIds, forkBConfig),
    ]);
    if (manifestA == null) {
      throw new Error('Expected fork A resume state.');
    }
    const forkCResumeConfig = {
      configurable: {
        ...forkCConfig.configurable,
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: manifestA,
      },
    };
    await executor.persistSettledToolOutput(
      call,
      forkCResumeConfig,
      makeSettledOutput('fork-c result')
    );
    const manifestC = await executor.getResumeManifest(
      parentToolCallIds,
      forkCResumeConfig
    );
    if (manifestB == null || manifestC == null) {
      throw new Error('Expected fork B and C resume state.');
    }

    await expect(
      executor.getResumeManifest(parentToolCallIds)
    ).resolves.toBeUndefined();
    expect(manifestA.executions).toHaveLength(1);
    expect(manifestB.executions).toHaveLength(1);
    expect(manifestC.executions).toHaveLength(1);
    expect(manifestA.executions[0].childRunId).toBe(
      manifestC.executions[0].childRunId
    );
    expect(manifestA.executions[0].childRunId).not.toBe(
      manifestB.executions[0].childRunId
    );
    expect(
      new Set(
        [manifestA, manifestB, manifestC].map(
          (manifest) => manifest.executions[0].checkpoints[0].threadId
        )
      ).size
    ).toBe(3);

    const replay = async (
      manifest: SubagentResumeManifest | undefined,
      parentConfigurable: Record<string, unknown>
    ): Promise<SettledSubagentToolOutput | undefined> =>
      createExecutor([config], {
        checkpointer,
        humanInTheLoop: { enabled: true },
      }).getSettledToolOutput(call, {
        configurable: {
          ...parentConfigurable,
          [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: manifest,
        },
      });
    const [replayedA, replayedB, replayedC] = await Promise.all([
      replay(manifestA, forkAConfig.configurable),
      replay(manifestB, forkBConfig.configurable),
      replay(manifestC, forkCConfig.configurable),
    ]);
    expect(replayedA?.output.content).toBe('fork-a result');
    expect(replayedB?.output.content).toBe('fork-b result');
    expect(replayedC?.output.content).toBe('fork-c result');
  });

  it('isolates concurrent resume attempts without a manifest', async () => {
    const checkpointer = new MemorySaver();
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_shared_attempts';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Run the attempted child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const baseConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'shared-fork',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
    };
    const attemptA = {
      configurable: {
        ...baseConfigurable,
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
      },
    };
    const attemptB = {
      configurable: {
        ...baseConfigurable,
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-b',
      },
    };
    const makeSettled = (content: string): SettledSubagentToolOutput => ({
      output: new ToolMessage({
        content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: call.args,
    });

    await Promise.all([
      executor.persistSettledToolOutput(
        call,
        attemptA,
        makeSettled('attempt-a result')
      ),
      executor.persistSettledToolOutput(
        call,
        attemptB,
        makeSettled('attempt-b result')
      ),
    ]);
    const [manifestA, manifestB] = await Promise.all([
      executor.getResumeManifest(new Set([parentToolCallId]), attemptA),
      executor.getResumeManifest(new Set([parentToolCallId]), attemptB),
    ]);

    expect(manifestA?.executions).toHaveLength(1);
    expect(manifestB?.executions).toHaveLength(1);
    expect(manifestA?.executions[0].childRunId).toBe(
      manifestB?.executions[0].childRunId
    );
    expect(manifestA?.executions[0].checkpoints[0].threadId).not.toBe(
      manifestB?.executions[0].checkpoints[0].threadId
    );
  });

  it('forks a legacy base checkpoint into each concurrent resume attempt', async () => {
    const checkpointer = new MemorySaver();
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const parentToolCallId = 'call_legacy_attempts';
    const baseConfigurable = {
      checkpoint_id: 'shared-fork',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
    };
    const addressRegistry = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    });
    const legacyAddress = addressRegistry.open({
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        ...baseConfigurable,
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
      },
    }).address;
    await putCheckpoint(
      checkpointer,
      legacyAddress.baseChildThreadId,
      '00000000-0000-0000-0000-000000000001'
    );
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Resume the legacy child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const attemptConfig = (attempt: string) => ({
      configurable: {
        thread_id: 'durable-thread',
        ...baseConfigurable,
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: attempt,
      },
    });
    const attemptA = attemptConfig('attempt-a');
    const attemptB = attemptConfig('attempt-b');
    const makeSettled = (content: string): SettledSubagentToolOutput => ({
      output: new ToolMessage({
        content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: call.args,
    });

    await Promise.all([
      executor.persistSettledToolOutput(
        call,
        attemptA,
        makeSettled('attempt-a result')
      ),
      executor.persistSettledToolOutput(
        call,
        attemptB,
        makeSettled('attempt-b result')
      ),
    ]);
    const [manifestA, manifestB] = await Promise.all([
      executor.getResumeManifest(new Set([parentToolCallId]), attemptA),
      executor.getResumeManifest(new Set([parentToolCallId]), attemptB),
    ]);
    const threadA = manifestA?.executions[0].checkpoints[0].threadId;
    const threadB = manifestB?.executions[0].checkpoints[0].threadId;

    expect(threadA).toBe(
      addressRegistry.open({
        threadId: 'durable-thread',
        parentToolCallId,
        parentConfigurable: attemptA.configurable,
      }).address.branchChildThreadId
    );
    expect(threadB).toBe(
      addressRegistry.open({
        threadId: 'durable-thread',
        parentToolCallId,
        parentConfigurable: attemptB.configurable,
      }).address.branchChildThreadId
    );
    expect(threadA).not.toBe(threadB);
    expect(threadA).not.toBe(legacyAddress.baseChildThreadId);
    expect(threadB).not.toBe(legacyAddress.baseChildThreadId);
  });

  it('retires superseded graphs across sequential resume attempts', async () => {
    const checkpointer = new MemorySaver();
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const parentToolCallId = 'call_sequential_attempts';
    const addressRegistry = new SubagentExecutionRegistry({
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      durable: true,
    });
    const baseConfigurable = {
      thread_id: 'durable-thread',
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'shared-batch',
    };
    const attemptConfig = (
      attempt: string,
      checkpointId: string,
      sourceThreadId?: string
    ): Record<string, unknown> => {
      const configurable: Record<string, unknown> = {
        ...baseConfigurable,
        checkpoint_id: checkpointId,
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: attempt,
      };
      if (sourceThreadId != null) {
        configurable[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY] = {
          version: 1,
          executions: [
            makeResumeExecution(
              parentToolCallId,
              config.configId,
              sourceThreadId
            ),
          ],
        };
      }
      return configurable;
    };
    const configA = attemptConfig('attempt-a', 'fork-a');
    const addressA = addressRegistry.open({
      threadId: baseConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable: configA,
    }).address;
    const configB = attemptConfig(
      'attempt-b',
      'fork-b',
      addressA.branchChildThreadId
    );
    const addressB = addressRegistry.open({
      threadId: baseConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable: configB,
    }).address;
    const configC = attemptConfig(
      'attempt-c',
      'fork-c',
      addressB.branchChildThreadId
    );
    const hookRegistry = new HookRegistry();
    const scopeA = getSubagentApprovalExecutionScope(
      addressA.currentChildRunId,
      'attempt-a'
    );
    const scopeB = getSubagentApprovalExecutionScope(
      'original-child-run',
      'attempt-b'
    );
    for (const scope of [scopeA, scopeB]) {
      hookRegistry.registerSession(scope, 'PreToolUse', {
        hooks: [async (): Promise<Record<string, never>> => ({})],
      });
    }
    const clearHeavyState: Array<jest.MockedFunction<() => void>> = [];
    let graphIndex = 0;
    const executor = createExecutor([config], {
      checkpointer,
      hookRegistry,
      humanInTheLoop: { enabled: true },
      createChildGraph: (): StandardGraph => {
        graphIndex += 1;
        const currentGraphIndex = graphIndex;
        const clear = jest.fn();
        clearHeavyState.push(clear);
        return {
          createWorkflow: () => ({
            getState: jest.fn(async () => ({
              values: {},
              next: [],
              tasks: [],
            })),
            invoke: jest.fn(async (): Promise<MultiAgentGraphState> => {
              throw new GraphInterrupt([
                {
                  id: `interrupt-${currentGraphIndex}`,
                  value: 'pause',
                },
              ]);
            }),
            updateState: jest.fn(async () => ({})),
          }),
          createSubagentResumeState: jest.fn(() => ({
            toolCallSteps: [],
            toolSessions: [],
            toolNodes: [],
            eagerToolUsage: [],
            eagerToolSuppressions: [],
          })),
          restoreSubagentResumeState: jest.fn(),
          clearHeavyState: clear,
        } as unknown as StandardGraph;
      },
    });
    const checkpointTarget = executor as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
      getLatestCheckpointSnapshot: (
        threadId: string
      ) => Promise<SubagentResumeExecution['checkpoints']>;
    };
    jest
      .spyOn(checkpointTarget, 'forkCheckpointSnapshot')
      .mockResolvedValue(undefined);
    jest
      .spyOn(checkpointTarget, 'getLatestCheckpointSnapshot')
      .mockImplementation(async (threadId) => [
        {
          threadId,
          checkpointId: '00000000-0000-0000-0000-000000000001',
          checkpointNs: '',
        },
      ]);
    const runAttempt = (parentConfigurable: Record<string, unknown>) =>
      executor.execute({
        description: 'Resume the interrupted child.',
        subagentType: config.type,
        threadId: baseConfigurable.thread_id,
        parentToolCallId,
        parentConfigurable,
      });

    await expect(runAttempt(configA)).rejects.toBeInstanceOf(GraphInterrupt);
    expect(clearHeavyState.map((clear) => clear.mock.calls.length)).toEqual([
      0,
    ]);
    await expect(runAttempt(configB)).rejects.toBeInstanceOf(GraphInterrupt);
    expect(clearHeavyState.map((clear) => clear.mock.calls.length)).toEqual([
      1, 0,
    ]);
    expect(hookRegistry.hasHookFor('PreToolUse', scopeA)).toBe(false);
    expect(hookRegistry.hasHookFor('PreToolUse', scopeB)).toBe(true);
    await expect(runAttempt(configC)).rejects.toBeInstanceOf(GraphInterrupt);
    expect(clearHeavyState.map((clear) => clear.mock.calls.length)).toEqual([
      1, 1, 0,
    ]);
    expect(hookRegistry.hasHookFor('PreToolUse', scopeB)).toBe(false);
  });

  it('coalesces duplicate settlement and replays its retained output in-process', async () => {
    const checkpointer = new MemorySaver();
    const config = makeLazyConfig('researcher', async () => makeAgent());
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_duplicate_settlement';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Persist this result once.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'shared-fork',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
      },
    };
    const settled: SettledSubagentToolOutput = {
      output: new ToolMessage({
        content: 'settled once',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: call.args,
    };
    const persistCheckpoint = jest.spyOn(checkpointer, 'put');

    await Promise.all([
      executor.persistSettledToolOutput(call, runnableConfig, settled),
      executor.persistSettledToolOutput(call, runnableConfig, settled),
    ]);

    expect(persistCheckpoint).toHaveBeenCalledTimes(1);
    const readCheckpoint = jest.spyOn(checkpointer, 'getTuple');
    readCheckpoint.mockClear();
    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toMatchObject({
      output: { content: 'settled once' },
      resolvedArgs: call.args,
    });
    expect(readCheckpoint).not.toHaveBeenCalled();
  });

  it('rejects mismatched settled args before they can win an execution race', async () => {
    let releaseResolver = (): void => undefined;
    const resolverGate = new Promise<void>((resolve) => {
      releaseResolver = resolve;
    });
    const config = makeLazyConfig('researcher', async () => {
      await resolverGate;
      return makeAgent();
    });
    const checkpointer = new MemorySaver();
    const updateState = jest.fn(async () => ({}));
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
      createChildGraph: () =>
        ({
          createWorkflow: () => ({
            getState: jest.fn(async () => ({
              values: { messages: [new AIMessage('original result')] },
              next: [],
              tasks: [],
            })),
            invoke: jest.fn(),
            updateState,
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const parentToolCallId = 'call_settlement_race';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Run the original request.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'shared-fork',
      [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
    };
    const runnableConfig = { configurable: parentConfigurable };
    const execution = executor.execute({
      description: call.args.description,
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    const mismatched: SettledSubagentToolOutput = {
      output: new ToolMessage({
        content: 'mismatched result',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        ...call.args,
        description: 'Run a changed request.',
      },
    };

    await executor.persistSettledToolOutput(call, runnableConfig, mismatched);
    expect(updateState).not.toHaveBeenCalled();
    releaseResolver();
    await expect(execution).resolves.toMatchObject({
      content: 'original result',
    });
    const accepted: SettledSubagentToolOutput = {
      output: new ToolMessage({
        content: 'original result',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: call.args,
    };
    await executor.persistSettledToolOutput(call, runnableConfig, accepted);

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toMatchObject({ output: { content: 'original result' } });
  });

  it('sanitizes private runtime state before calling the resolver', async () => {
    let configurable: Readonly<SubagentResolveConfigurable> | undefined;
    const config = makeLazyConfig('researcher', async (context) => {
      configurable = context.configurable;
      return makeAgent();
    });
    const executor = createExecutor([config]);

    await executor.execute({
      description: 'Resolve safely.',
      subagentType: config.type,
      parentToolCallId: 'call_sanitized',
      parentConfigurable: {
        requestBody: { messageId: 'allowed-message' },
        user: {
          id: 'allowed-user',
          role: 'member',
          tenantId: 'tenant-1',
          privateToken: 'principal-secret',
        },
        user_id: 'allowed-user',
        userMCPAuthMap: { private: { token: 'mcp-secret' } },
        __pregel_scratchpad: { secret: 'pregel-secret' },
        checkpoint_id: 'checkpoint-secret',
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'batch-secret',
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: { secret: 'resume-secret' },
        [TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY]: 'approval-secret',
        [RUN_BREAKER_SCOPE_CONFIG_KEY]: { secret: 'breaker-secret' },
      },
    });

    expect(configurable).toEqual({
      requestBody: { messageId: 'allowed-message' },
      user: {
        id: 'allowed-user',
        role: 'member',
        tenantId: 'tenant-1',
      },
      user_id: 'allowed-user',
    });
    expect(JSON.stringify(configurable)).not.toContain('secret');
  });

  it('uses a per-call abort signal while awaiting a non-cooperative resolver', async () => {
    const callAbort = new AbortController();
    let observedContext: SubagentResolveContext | undefined;
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const config = makeLazyConfig('researcher', async (context) => {
      observedContext = context;
      markResolutionStarted();
      return new Promise<AgentInputs>(() => undefined);
    });
    const executor = createExecutor([config]);

    const execution = executor.execute({
      description: 'Cancel this selection.',
      subagentType: config.type,
      parentToolCallId: 'call_cancel',
      signal: callAbort.signal,
    });
    await resolutionStarted;
    callAbort.abort(new Error('private cancellation reason'));
    const result = await execution;

    expect(observedContext?.signal.aborted).toBe(true);
    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(result.content).not.toContain('private cancellation reason');
  });

  it('rethrows a stream-limit trip while awaiting lazy resolution', async () => {
    const breaker = new AbortController();
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const config = makeLazyConfig('researcher', async () => {
      markResolutionStarted();
      return new Promise<AgentInputs>(() => undefined);
    });
    const executor = createExecutor([config]);
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });

    const execution = executor.execute({
      description: 'Stop this selection.',
      subagentType: config.type,
      parentToolCallId: 'call_limit',
      breaker,
    });
    await resolutionStarted;
    breaker.abort(trip);

    await expect(execution).rejects.toBe(trip);
  });

  it('prefers eager agent inputs without calling an attached resolver', async () => {
    const resolveAgentInputs = jest.fn(async () => makeAgent('unexpected'));
    const eagerInputs = makeAgent('eager-child');
    const eagerConfig: ResolvedSubagentConfig = {
      type: 'eager',
      name: 'Eager Worker',
      description: 'Already has complete inputs.',
      configId: 'eager@v1',
      agentInputs: eagerInputs,
      resolveAgentInputs,
    };
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([eagerConfig], {
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Use eager inputs.',
      subagentType: eagerConfig.type,
      parentToolCallId: 'call_eager',
    });

    expect(resolveAgentInputs).not.toHaveBeenCalled();
    expect(observedInput?.agents[0].agentId).toBe(eagerInputs.agentId);
  });

  it('accepts a legacy type-less resume for a versioned eager config', async () => {
    const configId = 'eager@v1';
    const eagerConfig: ResolvedSubagentConfig = {
      type: 'eager',
      name: 'Eager Worker',
      description: 'Already has complete inputs.',
      configId,
      agentInputs: makeAgent('eager-child'),
    };
    const parentToolCallId = 'call_legacy_eager';
    const resumeExecution = makeResumeExecution(parentToolCallId, configId);
    delete resumeExecution.subagentType;
    resumeExecution.graphState.fadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    resumeExecution.graphState.fadingTiers = {
      'eager-child': resumeExecution.graphState.fadingTier,
    };
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([eagerConfig], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
      createChildGraph: (input) => {
        observedInput = input;
        return makeRecoveredGraph();
      },
    });
    const forkTarget = executor as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
    };
    jest
      .spyOn(forkTarget, 'forkCheckpointSnapshot')
      .mockResolvedValue(undefined);

    const result = await executor.execute({
      description: 'Resume eager inputs.',
      subagentType: eagerConfig.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(result.content).toBe('persisted result');
    expect(observedInput?.fadingTier).toEqual(
      resumeExecution.graphState.fadingTier
    );
    expect(observedInput?.fadingTiers).toEqual(
      resumeExecution.graphState.fadingTiers
    );
  });

  it('preserves nested lazy descriptors and decrements the child depth', async () => {
    const nestedDescriptor = makeLazyConfig('grandchild', async () =>
      makeAgent('grandchild')
    );
    const config = makeLazyConfig(
      'nested',
      async () => ({
        ...makeAgent('nested-child'),
        subagentConfigs: [nestedDescriptor],
        maxSubagentDepth: 3,
      }),
      { allowNested: true }
    );
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([config], {
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Delegate recursively.',
      subagentType: config.type,
      parentToolCallId: 'call_nested',
    });

    expect(observedInput?.agents[0].maxSubagentDepth).toBe(2);
    expect(observedInput?.agents[0].subagentConfigs).toEqual([
      nestedDescriptor,
    ]);
  });

  it('keeps resolver identity stable after executor reconstruction', async () => {
    const contexts: SubagentResolveContext[] = [];
    const config = makeLazyConfig('researcher', async (context) => {
      contexts.push(context);
      return makeAgent();
    });
    const makeRebuiltExecutor = (): SubagentExecutor =>
      createExecutor([config], {
        parentRunId: 'rebuilt-parent-run',
        humanInTheLoop: { enabled: true },
        createChildGraph: makeRecoveredGraph,
      });
    const params = {
      description: 'Recover the deterministic child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId: 'call_rebuild',
    };

    await makeRebuiltExecutor().execute(params);
    await makeRebuiltExecutor().execute(params);

    expect(contexts).toHaveLength(2);
    expect(contexts[0].executionId).toBe(contexts[1].executionId);
    expect(contexts[0].descriptor).toEqual(contexts[1].descriptor);
    expect(contexts[0].descriptor.configId).toBe('researcher@v1');
    expect(contexts[0].threadId).toBe('durable-thread');
    expect(contexts[0].parentToolCallId).toBe('call_rebuild');
  });

  it('keeps the persisted executionId when resuming a lazy child', async () => {
    let resolverContext: SubagentResolveContext | undefined;
    const config = makeLazyConfig('researcher', async (context) => {
      resolverContext = context;
      return makeAgent();
    });
    const parentToolCallId = 'call_persisted_identity';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    resumeExecution.childRunId = 'persisted-execution-id';
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
      createChildGraph: makeRecoveredGraph,
    });
    const forkTarget = executor as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
    };
    jest
      .spyOn(forkTarget, 'forkCheckpointSnapshot')
      .mockResolvedValue(undefined);

    await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(resolverContext?.executionId).toBe('persisted-execution-id');
  });

  it('rejects a changed configId before restoring or recording resume state', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver, {
      configId: 'researcher@v2',
    });
    const parentRunId = 'rebuilt-parent-run';
    const parentAgentId = 'parent-agent';
    const threadId = 'durable-thread';
    const parentToolCallId = 'call_rebuild';
    const branchThreadId = `subagent:${Buffer.from(
      JSON.stringify([
        threadId,
        'root',
        parentAgentId,
        parentToolCallId,
        'batch',
        parentRunId,
      ])
    ).toString('base64url')}`;
    const executor = createExecutor([config], {
      parentRunId,
      parentAgentId,
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the original child.',
      subagentType: config.type,
      threadId,
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [
            {
              parentToolCallId,
              childRunId: 'original-child-run',
              configId: 'researcher@v1',
              approvalExecutionScope: 'original-approval-scope',
              checkpoints: [
                {
                  threadId: branchThreadId,
                  checkpointId: 'original-checkpoint',
                  checkpointNs: '',
                },
              ],
              graphState: {
                toolCallSteps: [],
                toolSessions: [],
                toolNodes: [],
                eagerToolUsage: [],
                eagerToolSuppressions: [],
              },
              approvalReplays: [],
            },
          ],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(result.content).not.toContain('researcher@v1');
    expect(result.content).not.toContain('researcher@v2');
    expect(resolver).not.toHaveBeenCalled();
    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]))
    ).resolves.toBeUndefined();
  });

  it('rejects a versionless resume manifest for a lazy config', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const parentToolCallId = 'call_versionless_resume';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    delete resumeExecution.configId;
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(resolver).not.toHaveBeenCalled();
    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]))
    ).resolves.toBeUndefined();
  });

  it('rejects a lazy resume without manifest or checkpoint type evidence', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const parentToolCallId = 'call_unbound_lazy_resume';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    delete resumeExecution.subagentType;
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(resolver).not.toHaveBeenCalled();
  });

  it('rejects a resume manifest bound to another type with the same configId', async () => {
    const resolver = jest.fn(async () => makeAgent('coder'));
    const config = makeLazyConfig('coder', resolver, {
      configId: 'shared@v1',
    });
    const parentToolCallId = 'call_cross_type_resume';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      'shared@v1',
      'resume-source-thread',
      'researcher'
    );
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(resolver).not.toHaveBeenCalled();
    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]))
    ).resolves.toBeUndefined();
  });

  it('rejects replay lifecycle revision mismatches before restoring state', async () => {
    const config = makeLazyConfig('researcher', async () => makeAgent(), {
      configId: 'researcher@v2',
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_lifecycle_mismatch';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Resume the selected child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1 as const,
          executions: [makeResumeExecution(parentToolCallId, 'researcher@v1')],
        },
      },
    };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Do not persist this mismatch.',
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
    });

    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]))
    ).resolves.toBeUndefined();
  });

  it('rejects replay lifecycle type mismatches with the same configId', async () => {
    const config = makeLazyConfig('coder', async () => makeAgent('coder'), {
      configId: 'shared@v1',
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_lifecycle_type_mismatch';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Resume the selected child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1 as const,
          executions: [
            makeResumeExecution(
              parentToolCallId,
              'shared@v1',
              'resume-source-thread',
              'researcher'
            ),
          ],
        },
      },
    };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Do not persist this mismatch.',
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the coder.',
        subagent_type: config.type,
      },
    });

    expect(executor.getChildCheckpointThreadIds()).toEqual([]);
    await expect(
      executor.getResumeManifest(new Set([parentToolCallId]))
    ).resolves.toBeUndefined();
  });

  it('retains configId in regenerated manifests after resolver failure', async () => {
    const config = makeLazyConfig('researcher', async () => {
      throw new Error('transient resolver failure');
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_resolver_retry';
    const branchThreadId = `subagent:${Buffer.from(
      JSON.stringify([
        'durable-thread',
        'root',
        'parent-agent',
        parentToolCallId,
        'batch',
        'parent-run',
      ])
    ).toString('base64url')}`;
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId,
      branchThreadId
    );
    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });
    const checkpointTarget = executor as unknown as {
      getLatestCheckpointSnapshot: (
        threadId: string
      ) => Promise<SubagentResumeExecution['checkpoints']>;
    };
    jest
      .spyOn(checkpointTarget, 'getLatestCheckpointSnapshot')
      .mockResolvedValue([
        {
          threadId: 'restored-child-thread',
          checkpointId: 'restored-checkpoint',
          checkpointNs: '',
        },
      ]);
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBe(config.configId);
  });

  it('retains configId through fresh lifecycle persistence after resolver failure', async () => {
    const config = makeLazyConfig('researcher', async () => {
      throw new Error('transient resolver failure');
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_fresh_lifecycle_retry';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the selected child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    const result = await executor.execute({
      description: 'Start the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: result.content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBe(config.configId);
  });

  it('persists a bound default description when lifecycle args omit it', async () => {
    const checkpointer = new MemorySaver();
    const resolver = jest.fn(async () => {
      throw new Error('transient resolver failure');
    });
    const config = makeLazyConfig('researcher', resolver);
    const source = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_defaulted_description';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Original description before PreToolUse.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };
    const effectiveDescription = 'No task description provided';
    const result = await source.execute({
      description: effectiveDescription,
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    const settled: SettledSubagentToolOutput = {
      output: new ToolMessage({
        content: result.content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: '   ',
        subagent_type: config.type,
      },
    };
    const persistCheckpoint = jest.spyOn(checkpointer, 'put');

    await Promise.all([
      source.persistSettledToolOutput(call, runnableConfig, settled),
      source.persistSettledToolOutput(call, runnableConfig, settled),
    ]);
    const manifest = await source.getResumeManifest(
      new Set([parentToolCallId])
    );
    if (manifest == null) {
      throw new Error('Expected the defaulted invocation to persist.');
    }
    expect(persistCheckpoint).toHaveBeenCalledTimes(1);
    const rebuilt = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const replayed = await rebuilt.getSettledToolOutput(call, {
      configurable: {
        ...parentConfigurable,
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: manifest,
      },
    });

    expect(replayed?.output.content).toBe(result.content);
    expect(resolver).toHaveBeenCalledTimes(1);
  });

  it('replays a terminal hook result whose description was normalized away', async () => {
    const checkpointer = new MemorySaver();
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const source = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_terminal_default_description';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: '   ',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };
    const terminalHook = jest.fn(
      async (): Promise<SettledSubagentToolOutput> => ({
        output: new ToolMessage({
          content: 'Blocked before execution.',
          name: call.name,
          status: 'error',
          tool_call_id: parentToolCallId,
        }),
        additionalContexts: [],
        resolvedArgs: { subagent_type: config.type },
      })
    );
    const runLifecycle = async (
      executor: SubagentExecutor,
      currentConfig: typeof runnableConfig
    ): Promise<SettledSubagentToolOutput> => {
      const replayed = await executor.getSettledToolOutput(call, currentConfig);
      if (replayed != null) {
        return replayed;
      }
      const settled = await terminalHook();
      await executor.persistSettledToolOutput(call, currentConfig, settled);
      return settled;
    };

    await expect(runLifecycle(source, runnableConfig)).resolves.toMatchObject({
      output: { content: 'Blocked before execution.' },
    });
    const manifest = await source.getResumeManifest(
      new Set([parentToolCallId])
    );
    if (manifest == null) {
      throw new Error('Expected the terminal hook result to persist.');
    }
    const rebuilt = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const replayConfig = {
      configurable: {
        ...parentConfigurable,
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: manifest,
      },
    };
    const replayed = await runLifecycle(rebuilt, replayConfig);

    expect(replayed.output.content).toBe('Blocked before execution.');
    expect(replayed.resolvedArgs).toEqual({
      subagent_type: config.type,
    });
    expect(terminalHook).toHaveBeenCalledTimes(1);
    expect(resolver).not.toHaveBeenCalled();
  });

  it('persists the configId for the effective rewritten subagent type', async () => {
    const researcher = makeLazyConfig('researcher', async () =>
      makeAgent('researcher')
    );
    const coder = makeLazyConfig('coder', async () => {
      throw new Error('transient coder resolver failure');
    });
    const executor = createExecutor([researcher, coder], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_rewritten_type';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the original child.',
        subagent_type: researcher.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    const result = await executor.execute({
      description: 'Run the rewritten child.',
      subagentType: coder.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: result.content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the rewritten child.',
        subagent_type: coder.type,
      },
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].subagentType).toBe(coder.type);
    expect(manifest?.executions[0].configId).toBe(coder.configId);
  });

  it('replays a persisted effective type before rerunning lifecycle hooks', async () => {
    const checkpointer = new MemorySaver();
    const researcher = makeLazyConfig('researcher', async () =>
      makeAgent('researcher')
    );
    const coder = makeLazyConfig('coder', async () => makeAgent('coder'));
    const source = createExecutor([researcher, coder], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_effective_replay';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the original child.',
        subagent_type: researcher.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };

    await expect(
      source.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await source.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Coder completed.',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the rewritten child.',
        subagent_type: coder.type,
      },
    });
    const manifest = await source.getResumeManifest(
      new Set([parentToolCallId])
    );
    expect(manifest?.executions[0].subagentType).toBe(coder.type);
    if (manifest == null) {
      throw new Error('Expected a persisted replay manifest.');
    }
    const mismatchedManifest = {
      ...manifest,
      executions: [
        {
          ...manifest.executions[0],
          subagentType: researcher.type,
        },
      ],
    };
    const rejected = createExecutor([researcher, coder], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const rejectedForkTarget = rejected as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
    };
    const forkCheckpointSnapshot = jest
      .spyOn(rejectedForkTarget, 'forkCheckpointSnapshot')
      .mockResolvedValue(undefined);
    await expect(
      rejected.getSettledToolOutput(call, {
        configurable: {
          ...parentConfigurable,
          [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: mismatchedManifest,
        },
      })
    ).resolves.toBeUndefined();
    expect(forkCheckpointSnapshot).not.toHaveBeenCalled();
    expect(rejected.getChildCheckpointThreadIds()).toEqual([]);

    delete manifest.executions[0].subagentType;

    const rebuilt = createExecutor([researcher, coder], {
      checkpointer,
      humanInTheLoop: { enabled: true },
    });
    const replayed = await rebuilt.getSettledToolOutput(call, {
      configurable: {
        ...parentConfigurable,
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: manifest,
      },
    });

    expect(replayed?.output.content).toBe('Coder completed.');
    expect(replayed?.resolvedArgs).toMatchObject({
      subagent_type: coder.type,
    });
  });

  it('does not retain the original configId after rewriting to an eager config', async () => {
    const original = makeLazyConfig('researcher', async () =>
      makeAgent('researcher')
    );
    const eager: ResolvedSubagentConfig = {
      type: 'eager',
      name: 'Eager worker',
      description: 'Runs from eager inputs.',
      agentInputs: makeAgent('eager-child'),
    };
    const executor = createExecutor([original, eager], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_rewritten_eager';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the original child.',
        subagent_type: original.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'parent-checkpoint',
        run_id: 'parent-run',
      },
    };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Eager child completed.',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the eager child.',
        subagent_type: eager.type,
      },
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].subagentType).toBe(eager.type);
    expect(manifest?.executions[0].configId).toBeUndefined();
  });

  it('uses the standard factory for lazy agents and the polymorphic factory for graphs', async () => {
    const lazyConfig = makeLazyConfig('researcher', async () =>
      makeAgent('lazy-researcher')
    );
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'research-team',
      name: 'Research Team',
      description: 'Runs a graph child.',
      agents: [makeAgent('result')],
      edges: [],
      entryAgentId: 'result',
      resultAgentId: 'result',
    };
    const standardInputs: StandardGraphInput[] = [];
    const graphRequests: GraphFactoryRequest[] = [];
    const createChildGraph = jest.fn((input: StandardGraphInput) => {
      standardInputs.push(input);
      return makeGraph({ messages: [new AIMessage('lazy result')] });
    });
    const createChildGraphByKind = jest.fn((request: GraphFactoryRequest) => {
      graphRequests.push(request);
      return makeGraph({
        messages: [new AIMessage('graph result')],
        subagentResult: {
          agentId: 'result',
          message: new AIMessage('graph result'),
        },
      });
    });
    const executor = createExecutor([lazyConfig, graphConfig], {
      createChildGraph,
      createChildGraphByKind,
    });

    const lazyResult = await executor.execute({
      description: 'Run one agent.',
      subagentType: lazyConfig.type,
      parentToolCallId: 'call_lazy',
    });
    const graphResult = await executor.execute({
      description: 'Run the team.',
      subagentType: graphConfig.type,
      parentToolCallId: 'call_graph',
    });

    expect(lazyResult.content).toBe('lazy result');
    expect(graphResult.content).toBe('graph result');
    expect(createChildGraph).toHaveBeenCalledTimes(1);
    expect(standardInputs[0].agents[0].agentId).toBe('lazy-researcher');
    expect(createChildGraphByKind).toHaveBeenCalledTimes(1);
    expect(graphRequests[0]).toMatchObject({
      kind: 'multi-agent',
      input: {
        agents: [{ agentId: 'result' }],
        resultAgentId: 'result',
      },
    });
  });

  it('rejects graph lazy fields during normalization before resolver or factory invocation', async () => {
    const resolveAgentInputs = jest.fn(async () => makeAgent('unexpected'));
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'invalid-graph',
      name: 'Invalid Graph',
      description: 'Illegally carries lazy single-agent fields.',
      agents: [makeAgent('result')],
      edges: [],
      entryAgentId: 'result',
      resultAgentId: 'result',
    };
    Reflect.set(graphConfig, 'configId', 'invalid-graph@v1');
    Reflect.set(graphConfig, 'resolveAgentInputs', resolveAgentInputs);
    const createChildGraph = jest.fn((_input: StandardGraphInput) =>
      makeGraph()
    );
    const createChildGraphByKind = jest.fn((_request: GraphFactoryRequest) =>
      makeGraph()
    );
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));

    const initializeAndExecute = async (): Promise<void> => {
      const configs = normalizeSubagentConfigEntries(
        [graphConfig],
        parentContext
      );
      const executor = createExecutor(configs, {
        createChildGraph,
        createChildGraphByKind,
      });
      await executor.execute({
        description: 'This must never execute.',
        subagentType: graphConfig.type,
        parentToolCallId: 'call_invalid_graph',
      });
    };

    await expect(initializeAndExecute()).rejects.toThrow(
      /lazy fields configId\/resolveAgentInputs/
    );
    expect(resolveAgentInputs).not.toHaveBeenCalled();
    expect(createChildGraph).not.toHaveBeenCalled();
    expect(createChildGraphByKind).not.toHaveBeenCalled();
  });
});

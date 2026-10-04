import { MemorySaver, GraphInterrupt } from '@langchain/langgraph';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { describe, expect, it, jest, afterEach } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  LazySingleAgentSubagentConfig,
  StandardGraphInput,
} from '@/types';
import type { SubagentResolutionFailureHandler } from '../diagnostics';
import type { SubagentExecutorOptions } from '../SubagentExecutor';
import type { SubagentResumeExecution } from '../SubagentReplay';
import {
  getSubagentResolutionFailureMessage,
  SubagentResolutionError,
} from '../diagnostics';
import { SubagentExecutionRegistry } from '../SubagentExecutionRegistry';
import { InMemorySubagentTaskStore } from '../InMemorySubagentTaskStore';
import { SUBAGENT_RESUME_MANIFEST_CONFIG_KEY } from '../SubagentReplay';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { SubagentExecutor } from '../SubagentExecutor';
import { createGraph } from '@/graphs';
import { Providers } from '@/common';
import { Run } from '@/run';

const secret = 'api_key=private-secret\n    at private-stack (private.ts:42)';
const config: LazySingleAgentSubagentConfig = {
  type: 'reviewer',
  name: 'Reviewer',
  description: 'Review code.',
  configId: 'reviewer@v1',
  resolveAgentInputs: async () => {
    throw new Error(secret);
  },
};
const params = {
  subagentType: 'reviewer',
  description: 'Review this.',
  parentToolCallId: 'call-review',
  threadId: 'conversation-thread',
};

function executor(
  onResolutionFailure?: SubagentResolutionFailureHandler,
  overrides: Partial<SubagentExecutorOptions> = {}
) {
  const createChildGraph = jest.fn((input: StandardGraphInput) =>
    createGraph({ kind: 'standard', input })
  );
  return {
    createChildGraph,
    instance: new SubagentExecutor({
      configs: new Map([[config.type, config]]),
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      createChildGraph,
      onResolutionFailure,
      ...overrides,
    }),
  };
}

afterEach(() => {
  jest.restoreAllMocks();
});

describe('SubagentExecutor startup failure delivery', () => {
  it.each([
    'workspace_unavailable',
    'agent_unavailable',
    'model_unavailable',
    'configuration_changed',
    'unknown',
  ] as const)(
    'returns the host-mapped %s cause before child execution',
    async (cause) => {
      const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
      const hook = jest.fn<SubagentResolutionFailureHandler>(() => cause);
      const { instance, createChildGraph } = executor(hook);
      const result = await instance.execute(params);

      expect(result).toEqual({
        content: getSubagentResolutionFailureMessage(cause),
        error: getSubagentResolutionFailureMessage(cause),
        messages: [],
        resolutionFailure: { phase: 'config', cause },
      });
      expect(hook).toHaveBeenCalledTimes(1);
      expect(hook.mock.calls[0][0]).toMatchObject({
        phase: 'config',
        subagentType: 'reviewer',
        aborted: false,
        type: 'Error',
        parentRunId: 'parent-run',
        parentAgentId: 'parent-agent',
        parentToolCallId: 'call-review',
        threadId: 'conversation-thread',
        childRunId: expect.any(String),
        childThreadId: expect.any(String),
      });
      expect(hook.mock.calls[0][1]).toBeInstanceOf(Error);
      expect(JSON.stringify(result)).not.toContain('private');
      expect(createChildGraph).not.toHaveBeenCalled();
      expect(warn).not.toHaveBeenCalled();
      instance.clearHeavyState();
    }
  );

  it('reports identity failures without calling the config resolver', async () => {
    class OfflineSaver extends MemorySaver {
      override getTuple(
        _config: RunnableConfig
      ): ReturnType<MemorySaver['getTuple']> {
        return Promise.reject(new Error(secret));
      }
    }
    const hook = jest.fn<SubagentResolutionFailureHandler>(
      () => 'configuration_changed'
    );
    const resolver = jest.fn(config.resolveAgentInputs);
    const instance = new SubagentExecutor({
      configs: new Map([
        [config.type, { ...config, resolveAgentInputs: resolver }],
      ]),
      parentRunId: 'parent-run',
      createChildGraph: (input) => createGraph({ kind: 'standard', input }),
      checkpointer: new OfflineSaver(),
      humanInTheLoop: { enabled: true },
      onResolutionFailure: hook,
    });
    const result = await instance.execute(params);

    expect(hook.mock.calls[0][0]).toMatchObject({
      phase: 'identity',
      aborted: false,
      type: 'Error',
    });
    expect(result.resolutionFailure).toEqual({
      phase: 'identity',
      cause: 'configuration_changed',
    });
    expect(resolver).not.toHaveBeenCalled();
    expect(JSON.stringify(result)).not.toContain('private');
    instance.clearHeavyState();
  });

  it.each([
    ['manifest', 'missing checkpoint'],
    ['manifest', 'source read'],
    ['manifest', 'target write'],
    ['manifest', 'config resolver'],
    ['manifest', 'aborted write'],
    ['manifest', 'invalidated write'],
    ['base checkpoint', 'source read'],
    ['base checkpoint', 'source list'],
    ['base replay', 'source list'],
    ['base checkpoint', 'target write'],
    ['base checkpoint', 'config resolver'],
    ['base checkpoint', 'aborted write'],
    ['base checkpoint', 'invalidated write'],
  ] as const)(
    'correlates %s child startup failures during %s',
    async (reconstruction, failureStage) => {
      let sourceThreadId = 'persisted-child-thread';
      const checkpointId = '00000000-0000-0000-0000-000000000001';
      const controller = new AbortController();
      class ForkFailureSaver extends MemorySaver {
        override getTuple(
          request: RunnableConfig
        ): ReturnType<MemorySaver['getTuple']> {
          if (
            failureStage === 'source read' &&
            request.configurable?.thread_id === sourceThreadId &&
            request.configurable.checkpoint_id === checkpointId
          ) {
            return Promise.reject(new Error(secret));
          }
          return super.getTuple(request);
        }

        override async *list(
          ...args: Parameters<MemorySaver['list']>
        ): ReturnType<MemorySaver['list']> {
          if (
            failureStage === 'source list' &&
            args[0].configurable?.thread_id === sourceThreadId
          ) {
            throw new Error(secret);
          }
          yield* super.list(...args);
        }

        override put(
          ...args: Parameters<MemorySaver['put']>
        ): ReturnType<MemorySaver['put']> {
          if (
            args[0].configurable?.thread_id !== sourceThreadId &&
            (failureStage === 'target write' ||
              failureStage === 'aborted write' ||
              failureStage === 'invalidated write')
          ) {
            if (failureStage === 'aborted write')
              controller.abort(new Error(secret));
            if (failureStage === 'invalidated write')
              instance.clearHeavyState();
            return Promise.reject(new Error(secret));
          }
          return super.put(...args);
        }
      }
      const checkpointer = new ForkFailureSaver();
      const resumeExecution: SubagentResumeExecution = {
        parentToolCallId: params.parentToolCallId,
        childRunId: 'persisted-child-run',
        subagentType: config.type,
        configId: config.configId,
        approvalExecutionScope: 'persisted-child-run',
        checkpoints: [
          { threadId: sourceThreadId, checkpointId, checkpointNs: '' },
        ],
        graphState: {
          toolCallSteps: [],
          toolSessions: [],
          toolNodes: [],
          eagerToolUsage: [],
          eagerToolSuppressions: [],
        },
        approvalReplays: [],
      };
      const parentConfigurable = {
        thread_id: params.threadId,
        checkpoint_id: 'parent-checkpoint',
        ...(reconstruction === 'manifest'
          ? {
            [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
              version: 1,
              executions: [resumeExecution],
            },
          }
          : {}),
      };
      const { address } = new SubagentExecutionRegistry({
        parentRunId: 'rebuilt-parent-run',
        parentAgentId: 'parent-agent',
        durable: true,
      }).open({ ...params, parentConfigurable });
      if (reconstruction !== 'manifest')
        sourceThreadId = address.baseChildThreadId;
      if (failureStage !== 'missing checkpoint') {
        await checkpointer.put(
          { configurable: { thread_id: sourceThreadId, checkpoint_ns: '' } },
          {
            v: 4,
            id: checkpointId,
            ts: new Date().toISOString(),
            channel_values:
              reconstruction === 'manifest'
                ? {}
                : {
                  messages: [
                    new HumanMessage({
                      content: 'Paused child',
                      additional_kwargs: {
                        __librechat_subagent_run_id:
                            resumeExecution.childRunId,
                      },
                    }),
                  ],
                },
            channel_versions: {},
            versions_seen: {},
          },
          { source: 'loop', step: 0, parents: {} }
        );
      }
      const hook = jest.fn<SubagentResolutionFailureHandler>(
        () => 'configuration_changed'
      );
      const resolver = jest.fn(config.resolveAgentInputs);
      const createChildGraph = jest.fn((input: StandardGraphInput) =>
        createGraph({ kind: 'standard', input })
      );
      const instance = new SubagentExecutor({
        configs: new Map([
          [config.type, { ...config, resolveAgentInputs: resolver }],
        ]),
        parentRunId: 'rebuilt-parent-run',
        parentAgentId: 'parent-agent',
        checkpointer,
        humanInTheLoop: { enabled: true },
        createChildGraph,
        onResolutionFailure: hook,
      });
      try {
        const result =
          reconstruction === 'base replay'
            ? (
              await instance.getSettledToolOutput(
                {
                  id: params.parentToolCallId,
                  name: 'subagent',
                  args: { subagent_type: config.type },
                },
                {
                  configurable: parentConfigurable,
                  signal: controller.signal,
                }
              )
            )?.output
            : await instance.execute({
              ...params,
              parentConfigurable,
              signal: controller.signal,
            });

        expect(hook).toHaveBeenCalledTimes(1);
        expect(hook.mock.calls[0][0]).toMatchObject({
          phase: failureStage === 'config resolver' ? 'config' : 'identity',
          aborted: failureStage === 'aborted write',
          parentRunId: 'rebuilt-parent-run',
          parentAgentId: 'parent-agent',
          threadId: params.threadId,
          parentToolCallId: params.parentToolCallId,
          childRunId: resumeExecution.childRunId,
          childThreadId: address.branchChildThreadId,
        });
        expect(address.currentChildRunId).not.toBe(resumeExecution.childRunId);
        expect(address.branchChildThreadId).not.toBe(address.baseChildThreadId);
        expect(resolver).toHaveBeenCalledTimes(
          failureStage === 'config resolver' ? 1 : 0
        );
        expect(createChildGraph).not.toHaveBeenCalled();
        if (result instanceof ToolMessage) {
          expect(result).toMatchObject({
            status: 'error',
            content: getSubagentResolutionFailureMessage(
              'configuration_changed'
            ),
          });
        } else {
          expect(result?.resolutionFailure?.cause).toBe(
            'configuration_changed'
          );
        }
        expect(JSON.stringify(result)).not.toContain('private');
      } finally {
        instance.clearHeavyState();
      }
    }
  );

  it('reports an aborted resolver safely through the host sink', async () => {
    const controller = new AbortController();
    const hook = jest.fn<SubagentResolutionFailureHandler>();
    const instance = new SubagentExecutor({
      configs: new Map([
        [
          config.type,
          {
            ...config,
            resolveAgentInputs: async () => {
              controller.abort(new Error(secret));
              throw new Error(secret);
            },
          },
        ],
      ]),
      parentRunId: 'parent-run',
      createChildGraph: (input) => createGraph({ kind: 'standard', input }),
      onResolutionFailure: hook,
    });
    const result = await instance.execute({
      ...params,
      signal: controller.signal,
    });

    expect(hook.mock.calls[0][0]).toMatchObject({
      phase: 'config',
      aborted: true,
    });
    expect(JSON.stringify(result)).not.toContain('private');
    instance.clearHeavyState();
  });

  it('forwards the hook to detached executors and preserves a typed safe error', async () => {
    const store = new InMemorySubagentTaskStore();
    const start = store.start.bind(store);
    let failure: Error | undefined;
    const settled = new Promise<void>((resolve) => {
      jest.spyOn(store, 'start').mockImplementation((request) =>
        start({
          ...request,
          run: async (...args) => {
            try {
              return await request.run(...args);
            } catch (error) {
              if (error instanceof Error) failure = error;
              throw error;
            } finally {
              resolve();
            }
          },
        })
      );
    });
    const hook = jest.fn<SubagentResolutionFailureHandler>(
      () => 'workspace_unavailable'
    );
    const instance = new SubagentExecutor({
      configs: new Map([[config.type, config]]),
      parentRunId: 'parent-run',
      parentAgentId: 'parent-agent',
      createChildGraph: (input) => createGraph({ kind: 'standard', input }),
      onResolutionFailure: hook,
      taskConfig: { store, scopeId: 'conversation-scope' },
    });
    const response = JSON.parse(instance.executeInBackground(params)) as {
      background_task_id: string;
    };
    await settled;
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(hook.mock.calls[0][0]).toMatchObject({
      phase: 'config',
      taskId: response.background_task_id,
      parentRunId: 'parent-run',
    });
    expect(failure).toBeInstanceOf(SubagentResolutionError);
    expect(failure).toMatchObject({
      phase: 'config',
      resolutionCause: 'workspace_unavailable',
    });
    expect(
      store.claim('conversation-scope', response.background_task_id)
    ).toMatchObject({
      status: 'error',
      error: getSubagentResolutionFailureMessage('workspace_unavailable'),
    });
    expect(failure?.stack).not.toContain('private');
    instance.clearHeavyState();
  });

  it.each([
    ['standard', 'workspace_unavailable'],
    ['standard', 'agent_unavailable'],
    ['standard', 'model_unavailable'],
    ['standard', 'configuration_changed'],
    ['standard', 'unknown'],
    ['multi-agent', 'workspace_unavailable'],
  ] as const)(
    'reports %s graph preflight failures with safe %s causes',
    async (kind, cause) => {
      class OfflineChildSaver extends MemorySaver {
        override getTuple(
          request: RunnableConfig
        ): ReturnType<MemorySaver['getTuple']> {
          const threadId = request.configurable?.thread_id;
          if (
            typeof threadId === 'string' &&
            threadId.startsWith('subagent:')
          ) {
            return Promise.reject(new Error(secret));
          }
          return super.getTuple(request);
        }
      }
      const checkpointer = new OfflineChildSaver();
      const hook = jest.fn<SubagentResolutionFailureHandler>(() => cause);
      const execute = jest.spyOn(SubagentExecutor.prototype, 'execute');
      const resolver = jest.fn(config.resolveAgentInputs);
      const parent = {
        agentId: 'parent',
        provider: Providers.OPENAI,
        clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'unused' },
        maxContextTokens: 8000,
        subagentConfigs: [{ ...config, resolveAgentInputs: resolver }],
      };
      const run = await Run.create({
        runId: 'parent-run',
        humanInTheLoop: { enabled: true },
        onSubagentResolutionFailure: hook,
        graphConfig:
          kind === 'standard'
            ? {
              type: 'standard',
              agents: [parent],
              compileOptions: { checkpointer },
            }
            : {
              type: 'multi-agent',
              agents: [parent],
              edges: [],
              entryAgentId: 'parent',
              compileOptions: { checkpointer },
            },
      });
      const graph = run.Graph;
      if (graph == null) throw new Error('Missing graph');
      graph.overrideTestModel(['Starting reviewer', 'Finished'], 0, [
        {
          name: 'subagent',
          id: 'spawn-reviewer',
          type: 'tool_call',
          args: { subagent_type: 'reviewer', description: 'Review code.' },
        },
      ]);
      try {
        const state = await graph
          .createWorkflow()
          .invoke(
            { messages: [new HumanMessage('Review this code.')] },
            { configurable: { thread_id: 'conversation-thread' } }
          );
        const output = state.messages.find(
          (message) => message instanceof ToolMessage
        );
        expect(output).toMatchObject({
          content: getSubagentResolutionFailureMessage(cause),
          status: 'error',
        });
        expect(hook).toHaveBeenCalledTimes(1);
        expect(hook.mock.calls[0][0]).toMatchObject({
          phase: 'identity',
          subagentType: 'reviewer',
          aborted: false,
          parentRunId: 'parent-run',
          parentAgentId: 'parent',
          parentToolCallId: 'spawn-reviewer',
          threadId: 'conversation-thread',
          childRunId: expect.any(String),
          childThreadId: expect.any(String),
        });
        expect(execute).not.toHaveBeenCalled();
        expect(resolver).not.toHaveBeenCalled();
        expect(JSON.stringify(state.messages)).not.toContain('private');
        expect(JSON.stringify(state.messages)).not.toContain('api_key');
      } finally {
        graph.clearHeavyState();
      }
    }
  );

  it.each([
    ['string', secret],
    [
      'UndescribableError',
      new Proxy(
        {},
        {
          getPrototypeOf: () => {
            throw new Error(secret);
          },
        }
      ),
    ],
  ] as const)(
    'redacts a %s replay-preparation rejection',
    async (type, reason) => {
      class RejectingSaver extends MemorySaver {
        override getTuple(
          _request: RunnableConfig
        ): ReturnType<MemorySaver['getTuple']> {
          return Promise.reject(reason);
        }
      }
      const hook = jest.fn<SubagentResolutionFailureHandler>();
      const { instance } = executor(hook, {
        checkpointer: new RejectingSaver(),
        humanInTheLoop: { enabled: true },
      });
      try {
        const result = await instance.getSettledToolOutput(
          {
            id: params.parentToolCallId,
            name: 'subagent',
            args: { subagent_type: config.type },
          },
          { configurable: { thread_id: params.threadId } }
        );
        expect(result?.output).toMatchObject({
          content: getSubagentResolutionFailureMessage('unknown'),
          status: 'error',
        });
        expect(hook).toHaveBeenCalledTimes(1);
        expect(hook.mock.calls[0][0]).toMatchObject({
          phase: 'identity',
          type,
        });
        expect(JSON.stringify(result)).not.toContain('private');
      } finally {
        instance.clearHeavyState();
      }
    }
  );

  it.each([
    new GraphInterrupt([]),
    new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
    }),
  ])('preserves replay framework control flow', async (reason) => {
    class RejectingSaver extends MemorySaver {
      override getTuple(
        _request: RunnableConfig
      ): ReturnType<MemorySaver['getTuple']> {
        return Promise.reject(reason);
      }
    }
    const hook = jest.fn<SubagentResolutionFailureHandler>();
    const { instance } = executor(hook, {
      checkpointer: new RejectingSaver(),
      humanInTheLoop: { enabled: true },
    });
    try {
      await expect(
        instance.getSettledToolOutput(
          {
            id: params.parentToolCallId,
            name: 'subagent',
            args: { subagent_type: config.type },
          },
          { configurable: { thread_id: params.threadId } }
        )
      ).rejects.toBe(reason);
      expect(hook).not.toHaveBeenCalled();
    } finally {
      instance.clearHeavyState();
    }
  });

  it('reports aborted replay preparation while retaining safe cancellation text', async () => {
    const controller = new AbortController();
    class AbortingSaver extends MemorySaver {
      override getTuple(
        _request: RunnableConfig
      ): ReturnType<MemorySaver['getTuple']> {
        controller.abort(new Error(secret));
        return Promise.reject(new Error(secret));
      }
    }
    const hook = jest.fn<SubagentResolutionFailureHandler>(
      () => 'configuration_changed'
    );
    const { instance } = executor(hook, {
      checkpointer: new AbortingSaver(),
      humanInTheLoop: { enabled: true },
    });
    try {
      await expect(
        instance.getSettledToolOutput(
          {
            id: params.parentToolCallId,
            name: 'subagent',
            args: { subagent_type: config.type },
          },
          {
            configurable: { thread_id: params.threadId },
            signal: controller.signal,
          }
        )
      ).rejects.toMatchObject({
        name: 'SubagentResolutionError',
        message: getSubagentResolutionFailureMessage('configuration_changed'),
      });
      expect(hook).toHaveBeenCalledTimes(1);
      expect(hook.mock.calls[0][0]).toMatchObject({
        phase: 'identity',
        aborted: true,
      });
    } finally {
      instance.clearHeavyState();
    }
  });

  it('reports identity failure during settlement without exposing the original rejection', async () => {
    class OfflineSaver extends MemorySaver {
      override getTuple(
        _request: RunnableConfig
      ): ReturnType<MemorySaver['getTuple']> {
        return Promise.reject(new Error(secret));
      }
    }
    const hook = jest.fn<SubagentResolutionFailureHandler>(
      () => 'workspace_unavailable'
    );
    const { instance } = executor(hook, {
      checkpointer: new OfflineSaver(),
      humanInTheLoop: { enabled: true },
    });
    try {
      await expect(
        instance.persistSettledToolOutput(
          {
            id: params.parentToolCallId,
            name: 'subagent',
            args: { subagent_type: config.type },
          },
          { configurable: { thread_id: params.threadId } },
          {
            output: new ToolMessage({
              content: 'Safe result',
              name: 'subagent',
              tool_call_id: params.parentToolCallId,
            }),
            additionalContexts: [],
          }
        )
      ).rejects.toMatchObject({
        name: 'SubagentResolutionError',
        message: getSubagentResolutionFailureMessage('workspace_unavailable'),
        resolutionCause: 'workspace_unavailable',
      });
      expect(hook).toHaveBeenCalledTimes(1);
      expect(hook.mock.calls[0][0]).toMatchObject({
        phase: 'identity',
        parentToolCallId: params.parentToolCallId,
        threadId: params.threadId,
      });
    } finally {
      instance.clearHeavyState();
    }
  });

  it.each(['standard', 'multi-agent'] as const)(
    'wires the public Run callback into %s graphs',
    async (kind) => {
      const hook = jest.fn<SubagentResolutionFailureHandler>(
        () => 'agent_unavailable'
      );
      const parent = {
        agentId: 'parent',
        provider: Providers.OPENAI,
        clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'unused' },
        maxContextTokens: 8000,
        subagentConfigs: [config],
      };
      const run = await Run.create({
        runId: 'parent-run',
        onSubagentResolutionFailure: hook,
        graphConfig:
          kind === 'standard'
            ? { type: 'standard', agents: [parent] }
            : {
              type: 'multi-agent',
              agents: [parent],
              edges: [],
              entryAgentId: 'parent',
            },
      });
      const graph = run.Graph;
      expect(graph).toBeDefined();
      if (graph == null) throw new Error('Missing graph');
      graph.overrideTestModel(['Starting reviewer', 'Finished'], 0, [
        {
          name: 'subagent',
          id: 'spawn-reviewer',
          type: 'tool_call',
          args: { subagent_type: 'reviewer', description: 'Review code.' },
        },
      ]);
      await graph
        .createWorkflow()
        .invoke({ messages: [new HumanMessage('Review this code.')] });

      expect(hook).toHaveBeenCalledWith(
        expect.objectContaining({ phase: 'config', parentRunId: 'parent-run' }),
        expect.any(Error)
      );
      graph.clearHeavyState();
    }
  );

  it('forwards the callback in nested child graph inputs', async () => {
    const hook = jest.fn<SubagentResolutionFailureHandler>(
      () => 'model_unavailable'
    );
    let childInput: StandardGraphInput | undefined;
    const instance = new SubagentExecutor({
      configs: new Map([
        [
          config.type,
          {
            ...config,
            agentInputs: {
              agentId: 'child',
              provider: Providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'unused' },
              maxContextTokens: 8000,
            },
            resolveAgentInputs: undefined,
          },
        ],
      ]),
      parentRunId: 'parent-run',
      onResolutionFailure: hook,
      createChildGraph: (input) => {
        childInput = input;
        const graph = createGraph({ kind: 'standard', input });
        graph.overrideTestModel(['Finished']);
        return graph;
      },
    });
    await instance.execute(params);

    expect(childInput?.onSubagentResolutionFailure).toBe(hook);
    instance.clearHeavyState();
  });
});

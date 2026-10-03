import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest, afterEach } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  LazySingleAgentSubagentConfig,
  StandardGraphInput,
} from '@/types';
import type { SubagentResolutionFailureHandler } from '../diagnostics';
import {
  getSubagentResolutionFailureMessage,
  SubagentResolutionError,
} from '../diagnostics';
import { InMemorySubagentTaskStore } from '../InMemorySubagentTaskStore';
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

function executor(onResolutionFailure?: SubagentResolutionFailureHandler) {
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

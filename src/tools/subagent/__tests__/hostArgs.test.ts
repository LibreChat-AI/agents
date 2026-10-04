import { MemorySaver } from '@langchain/langgraph';
import { AIMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type {
  AgentInputs,
  ExecutableSubagentConfigEntry,
  LazySingleAgentSubagentConfig,
  MultiAgentGraphState,
  SubagentConfig,
  SubagentResolveContext,
} from '@/types';
import type { SubagentResolutionFailureHandler } from '@/tools/subagent/diagnostics';
import type { SubagentResumeExecution } from '@/tools/subagent/SubagentReplay';
import type { StandardGraph } from '@/graphs/Graph';
import {
  getSubagentHostArgsDigest,
  pickSubagentHostArgInput,
  resolveSubagentHostArgs,
  SUBAGENT_HOST_ARG_LIMITS,
} from '@/tools/subagent/hostArgs';
import { InMemorySubagentTaskStore } from '@/tools/subagent/InMemorySubagentTaskStore';
import { SUBAGENT_RESUME_MANIFEST_CONFIG_KEY } from '@/tools/subagent/SubagentReplay';
import { normalizeSubagentConfigEntries } from '@/tools/subagent/childGraphConfig';
import { SubagentHostArgumentError } from '@/tools/subagent/diagnostics';
import { SubagentExecutor } from '@/tools/subagent/SubagentExecutor';
import { buildSubagentToolParams } from '@/tools/SubagentTool';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const makeAgent = (agentId = 'child-agent'): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
  maxContextTokens: 8000,
});

const MACHINE = {
  description: 'Machine the subagent runs on.',
  enum: ['laptop', 'buildbox'],
};

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

const makeGraph = (
  result: MultiAgentGraphState = { messages: [new AIMessage('Task completed')] }
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

const createExecutor = (
  configs: ExecutableSubagentConfigEntry[],
  overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
): SubagentExecutor =>
  new SubagentExecutor({
    configs: new Map(configs.map((config) => [config.type, config])),
    parentRunId: 'parent-run',
    parentAgentId: 'parent-agent',
    createChildGraph: () => makeGraph(),
    ...overrides,
  });

const makeResumeExecution = (
  parentToolCallId: string,
  hostArgsDigest?: string
): SubagentResumeExecution => ({
  parentToolCallId,
  childRunId: 'persisted-execution-id',
  subagentType: 'reviewer',
  configId: 'reviewer@v1',
  ...(hostArgsDigest == null ? {} : { hostArgsDigest }),
  approvalExecutionScope: 'original-approval-scope',
  checkpoints: [
    {
      threadId: 'resume-source-thread',
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

describe('subagent host argument declarations', () => {
  const plainConfigs: SubagentConfig[] = [
    {
      type: 'researcher',
      name: 'Researcher',
      description: 'Finds facts.',
      agentInputs: makeAgent('researcher'),
    },
  ];

  it('leaves the tool schema and description unchanged when nothing is declared', () => {
    const params = buildSubagentToolParams(plainConfigs, { background: true });

    expect(Object.keys(params.schema.properties ?? {})).toEqual([
      'intent',
      'description',
      'subagent_type',
      'run_in_background',
    ]);
    expect(params.description).not.toContain('OPTIONAL ARGUMENTS');
    expect(params.description).toContain(
      '- "researcher" (Researcher): Finds facts.'
    );
    expect(params.description).not.toContain('[optional');
  });

  it('unions enum values across subagents and lists each type’s own choices', () => {
    const reviewer = makeLazyConfig('reviewer', async () => makeAgent(), {
      hostArgs: {
        machine: MACHINE,
        workspace: { description: 'Workspace to open.', enum: ['agents'] },
      },
    });
    const coder = makeLazyConfig('coder', async () => makeAgent(), {
      hostArgs: {
        machine: {
          description: 'Ignored second description.',
          enum: ['buildbox', 'gpu'],
        },
        worktree: {
          description: 'Linked worktree to use.',
          maxLength: 80,
        },
      },
    });

    const params = buildSubagentToolParams([...plainConfigs, reviewer, coder]);

    expect(params.schema.properties?.machine).toEqual({
      type: 'string',
      description: 'Machine the subagent runs on.',
      enum: ['laptop', 'buildbox', 'gpu'],
    });
    expect(params.schema.properties?.workspace).toEqual({
      type: 'string',
      description: 'Workspace to open.',
      enum: ['agents'],
    });
    expect(params.schema.properties?.worktree).toEqual({
      type: 'string',
      description: 'Linked worktree to use.',
      maxLength: 80,
    });
    expect(params.schema.properties?.worktree).not.toHaveProperty('pattern');
    expect(params.schema.required).toEqual(['description', 'subagent_type']);
    expect(params.description).toContain('OPTIONAL ARGUMENTS');
    expect(params.description).toContain(
      '- "reviewer" (reviewer worker): Handles reviewer tasks. [optional machine: laptop | buildbox; workspace: agents]'
    );
    expect(params.description).toContain(
      '- "coder" (coder worker): Handles coder tasks. [optional machine: buildbox | gpu; worktree: text]'
    );
    expect(params.description).toContain(
      '- "researcher" (Researcher): Finds facts.\n'
    );
  });

  it.each([
    ['a reserved name', { subagent_type: MACHINE }, 'built-in'],
    ['an invalid name', { 'Machine-Name': MACHINE }, 'must match'],
    [
      'an empty enum',
      { machine: { description: 'Machine.', enum: [] } },
      'enum must list',
    ],
    [
      'duplicate enum values',
      { machine: { description: 'Machine.', enum: ['a', 'a'] } },
      'unique',
    ],
    [
      'too many enum values',
      {
        machine: {
          description: 'Machine.',
          enum: Array.from(
            { length: SUBAGENT_HOST_ARG_LIMITS.enumValues + 1 },
            (_, index) => `m${index}`
          ),
        },
      },
      'enum must list',
    ],
    [
      'a control character',
      { machine: { description: 'Machine.', enum: ['a\nb'] } },
      'control characters',
    ],
    [
      'enum combined with maxLength',
      { machine: { description: 'Machine.', enum: ['a'], maxLength: 4 } },
      'cannot combine',
    ],
    [
      'a pattern, which would run model input through a host regex',
      { worktree: { description: 'Worktree.', pattern: '(a+)+$' } },
      'cannot declare a pattern',
    ],
    [
      'an oversized maxLength',
      {
        worktree: {
          description: 'Worktree.',
          maxLength: SUBAGENT_HOST_ARG_LIMITS.valueLength + 1,
        },
      },
      'maxLength',
    ],
    ['a missing description', { machine: { enum: ['a'] } }, 'description'],
  ])('rejects %s', (_label, hostArgs, message) => {
    const config = makeLazyConfig('reviewer', async () => makeAgent(), {
      hostArgs: hostArgs as LazySingleAgentSubagentConfig['hostArgs'],
    });
    expect(() => buildSubagentToolParams([config])).toThrow(message);
  });

  it('rejects more declarations than the per-subagent bound', () => {
    const hostArgs = Object.fromEntries(
      Array.from(
        { length: SUBAGENT_HOST_ARG_LIMITS.argsPerSubagent + 1 },
        (_, index) => [`arg_${index}`, { description: 'Arg.', enum: ['x'] }]
      )
    );
    const config = makeLazyConfig('reviewer', async () => makeAgent(), {
      hostArgs,
    });
    expect(() => buildSubagentToolParams([config])).toThrow(
      `more than ${SUBAGENT_HOST_ARG_LIMITS.argsPerSubagent}`
    );
  });

  it('accepts host arguments only on lazy single-agent configs', () => {
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));
    const eager: SubagentConfig = {
      type: 'eager',
      name: 'Eager',
      description: 'Eager child.',
      agentInputs: makeAgent('eager'),
      hostArgs: { machine: MACHINE },
    };
    const graph = {
      kind: 'graph',
      type: 'team',
      name: 'Team',
      description: 'Graph child.',
      agents: [makeAgent('member')],
      edges: [],
      entryAgentId: 'member',
      resultAgentId: 'member',
      hostArgs: { machine: MACHINE },
    } as unknown as SubagentConfig;
    const lazy = makeLazyConfig('lazy', async () => makeAgent(), {
      hostArgs: { machine: MACHINE },
    });

    expect(() =>
      normalizeSubagentConfigEntries([eager], parentContext)
    ).toThrow('only a lazy resolveAgentInputs config');
    expect(() =>
      normalizeSubagentConfigEntries([graph], parentContext)
    ).toThrow('configId/resolveAgentInputs/hostArgs');
    expect(normalizeSubagentConfigEntries([lazy], parentContext)).toEqual([
      lazy,
    ]);
  });
});

describe('subagent host argument values', () => {
  const config = makeLazyConfig('reviewer', async () => makeAgent(), {
    hostArgs: {
      machine: MACHINE,
      worktree: {
        description: 'Worktree.',
        maxLength: 40,
      },
    },
  });
  const names = new Set(['machine', 'worktree', 'workspace']);

  it('treats missing, null, and blank values as omitted and rejects non-strings', () => {
    expect(pickSubagentHostArgInput({ description: 'x' }, names)).toEqual({
      ok: true,
    });
    expect(
      pickSubagentHostArgInput({ machine: null, worktree: '  ' }, names)
    ).toEqual({ ok: true });
    expect(pickSubagentHostArgInput({ machine: 3 }, names)).toEqual({
      ok: false,
      message: 'Error: "machine" must be a string.',
    });
    expect(
      pickSubagentHostArgInput({ machine: 'laptop', other: 'x' }, names)
    ).toEqual({ ok: true, hostArgs: { machine: 'laptop' } });
  });

  it('accepts declared values and returns them frozen', () => {
    const result = resolveSubagentHostArgs(config, {
      worktree: '.worktrees/fix-1',
      machine: 'buildbox',
    });
    expect(result).toEqual({
      ok: true,
      hostArgs: { machine: 'buildbox', worktree: '.worktrees/fix-1' },
    });
    expect(result.ok && Object.isFrozen(result.hostArgs)).toBe(true);
  });

  it('rejects values outside the selected subagent’s declaration', () => {
    expect(resolveSubagentHostArgs(config, { machine: 'gpu' })).toEqual({
      ok: false,
      message:
        'Error: "machine" for subagent "reviewer" must be one of: laptop, buildbox. Omit "machine" to let the host choose.',
    });
    expect(
      resolveSubagentHostArgs(config, { worktree: '.worktrees/a\nb' })
    ).toMatchObject({
      ok: false,
      message: expect.stringContaining(
        'must be at most 40 characters without control characters'
      ),
    });
    expect(
      resolveSubagentHostArgs(config, {
        worktree: `.worktrees/${'a'.repeat(40)}`,
      })
    ).toMatchObject({ ok: false });
    expect(resolveSubagentHostArgs(config, { workspace: 'agents' })).toEqual({
      ok: false,
      message:
        'Error: Subagent "reviewer" does not accept "workspace". Omit it for this subagent type.',
    });
  });

  it('digests values independently of key order and never for an empty set', () => {
    const first = getSubagentHostArgsDigest({ a: '1', b: '2' });
    expect(first).toMatch(/^[a-f0-9]{64}$/);
    expect(getSubagentHostArgsDigest({ b: '2', a: '1' })).toBe(first);
    expect(getSubagentHostArgsDigest({ a: '1', b: '3' })).not.toBe(first);
    expect(getSubagentHostArgsDigest({})).toBeUndefined();
    expect(getSubagentHostArgsDigest(undefined)).toBeUndefined();
  });
});

describe('SubagentExecutor host arguments', () => {
  it('passes validated values to the resolver and omits the key when none were passed', async () => {
    const contexts: SubagentResolveContext[] = [];
    const config = makeLazyConfig(
      'reviewer',
      async (context) => {
        contexts.push(context);
        return makeAgent();
      },
      { hostArgs: { machine: MACHINE } }
    );
    const executor = createExecutor([config]);

    await executor.execute({
      description: 'Review on the build box.',
      subagentType: 'reviewer',
      parentToolCallId: 'call_with_args',
      hostArgs: { machine: 'buildbox' },
    });
    await executor.execute({
      description: 'Review wherever the host decides.',
      subagentType: 'reviewer',
      parentToolCallId: 'call_without_args',
    });

    expect(contexts[0].hostArgs).toEqual({ machine: 'buildbox' });
    expect(Object.isFrozen(contexts[0].hostArgs)).toBe(true);
    expect('hostArgs' in contexts[1]).toBe(false);
  });

  it('rejects an invalid value before opening an execution or resolving', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('reviewer', resolver, {
      hostArgs: { machine: MACHINE },
    });
    const sibling = makeLazyConfig('coder', async () => makeAgent());
    const executor = createExecutor([config, sibling]);

    const invalid = await executor.execute({
      description: 'Review on an unknown machine.',
      subagentType: 'reviewer',
      parentToolCallId: 'call_invalid',
      hostArgs: { machine: 'gpu' },
    });
    const undeclared = await executor.execute({
      description: 'Code on a machine.',
      subagentType: 'coder',
      parentToolCallId: 'call_undeclared',
      hostArgs: { machine: 'laptop' },
    });

    expect(invalid.content).toBe(
      'Error: "machine" for subagent "reviewer" must be one of: laptop, buildbox. Omit "machine" to let the host choose.'
    );
    expect(undeclared.content).toBe(
      'Error: Subagent "coder" does not accept "machine". Omit it for this subagent type.'
    );
    expect(resolver).not.toHaveBeenCalled();
    await expect(
      executor.getResumeManifest(new Set(['call_invalid', 'call_undeclared']))
    ).resolves.toBeUndefined();
  });

  it('binds host arguments to the execution so a changed call cannot share it', async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const resolver = jest.fn(async (): Promise<AgentInputs> => {
      await gate;
      return makeAgent();
    });
    const config = makeLazyConfig('reviewer', resolver, {
      hostArgs: { machine: MACHINE },
    });
    const executor = createExecutor([config]);
    const params = {
      description: 'Review the change.',
      subagentType: 'reviewer',
      parentToolCallId: 'call_bound',
    };

    const first = executor.execute({
      ...params,
      hostArgs: { machine: 'laptop' },
    });
    const changed = await executor.execute({
      ...params,
      hostArgs: { machine: 'buildbox' },
    });
    release();

    expect(changed.content).toBe(
      'Subagent error: Subagent invocation changed for this execution.'
    );
    await expect(first).resolves.toMatchObject({ content: 'Task completed' });
    expect(resolver).toHaveBeenCalledTimes(1);
  });

  it('records a value-free digest in the resume manifest and replays it deterministically', async () => {
    const contexts: SubagentResolveContext[] = [];
    const config = makeLazyConfig(
      'reviewer',
      async (context) => {
        contexts.push(context);
        return makeAgent();
      },
      { hostArgs: { machine: MACHINE } }
    );
    const parentToolCallId = 'call_resume_same';
    const digest = getSubagentHostArgsDigest({ machine: 'buildbox' });
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
      description: 'Resume the review.',
      subagentType: 'reviewer',
      threadId: 'durable-thread',
      parentToolCallId,
      hostArgs: { machine: 'buildbox' },
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [makeResumeExecution(parentToolCallId, digest)],
        },
      },
    });

    expect(contexts).toHaveLength(1);
    expect(contexts[0].executionId).toBe('persisted-execution-id');
    expect(contexts[0].hostArgs).toEqual({ machine: 'buildbox' });
  });

  it('writes the digest, not the values, into a regenerated manifest', async () => {
    const config = makeLazyConfig('reviewer', async () => makeAgent(), {
      hostArgs: { machine: MACHINE },
    });
    const checkpointer = new MemorySaver();
    const executor = createExecutor([config], {
      checkpointer,
      humanInTheLoop: { enabled: true },
      createChildGraph: makeRecoveredGraph,
    });
    const target = executor as unknown as {
      getLatestCheckpointSnapshot: () => Promise<
        Array<{ threadId: string; checkpointId: string; checkpointNs: string }>
      >;
    };
    jest
      .spyOn(target, 'getLatestCheckpointSnapshot')
      .mockResolvedValue([
        { threadId: 'child-thread', checkpointId: 'cp-1', checkpointNs: '' },
      ]);

    await executor.execute({
      description: 'Review on the laptop.',
      subagentType: 'reviewer',
      threadId: 'durable-thread',
      parentToolCallId: 'call_manifest',
      hostArgs: { machine: 'laptop' },
    });
    const manifest = await executor.getResumeManifest(
      new Set(['call_manifest'])
    );

    expect(manifest?.executions[0].hostArgsDigest).toBe(
      getSubagentHostArgsDigest({ machine: 'laptop' })
    );
    expect(JSON.stringify(manifest)).not.toContain('laptop');
  });

  it.each([
    ['different values', { machine: 'laptop' }],
    ['omitted values', undefined],
  ])(
    'rejects a resume whose host arguments changed (%s)',
    async (_label, hostArgs) => {
      const resolver = jest.fn(async () => makeAgent());
      const config = makeLazyConfig('reviewer', resolver, {
        hostArgs: { machine: MACHINE },
      });
      const parentToolCallId = 'call_resume_changed';
      const executor = createExecutor([config], {
        checkpointer: new MemorySaver(),
        humanInTheLoop: { enabled: true },
        createChildGraph: makeRecoveredGraph,
      });

      const result = await executor.execute({
        description: 'Resume the review.',
        subagentType: 'reviewer',
        threadId: 'durable-thread',
        parentToolCallId,
        ...(hostArgs == null ? {} : { hostArgs }),
        parentConfigurable: {
          [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
            version: 1,
            executions: [
              makeResumeExecution(
                parentToolCallId,
                getSubagentHostArgsDigest({ machine: 'buildbox' })
              ),
            ],
          },
        },
      });

      expect(result.content).toBe(
        'Subagent error: Subagent invocation changed for this execution.'
      );
      expect(resolver).not.toHaveBeenCalled();
    }
  );

  it('ignores a malformed manifest digest as an invalid manifest', async () => {
    const resolver = jest.fn(async (_context: SubagentResolveContext) =>
      makeAgent()
    );
    const config = makeLazyConfig('reviewer', resolver, {
      hostArgs: { machine: MACHINE },
    });
    const parentToolCallId = 'call_bad_digest';
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    await executor.execute({
      description: 'Run fresh.',
      subagentType: 'reviewer',
      threadId: 'durable-thread',
      parentToolCallId,
      hostArgs: { machine: 'laptop' },
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [makeResumeExecution(parentToolCallId, 'laptop')],
        },
      },
    });

    expect(resolver).toHaveBeenCalledTimes(1);
    expect(resolver.mock.calls[0][0].executionId).not.toBe(
      'persisted-execution-id'
    );
  });

  it('turns a resolver host-argument refusal into a fixed message naming the argument', async () => {
    const onResolutionFailure = jest.fn<SubagentResolutionFailureHandler>(
      () => 'unknown'
    );
    const config = makeLazyConfig(
      'reviewer',
      async () => {
        throw new SubagentHostArgumentError('machine', 'unavailable');
      },
      { hostArgs: { machine: MACHINE } }
    );
    const denied = makeLazyConfig(
      'coder',
      async () => {
        throw new SubagentHostArgumentError('machine', 'not_allowed');
      },
      { hostArgs: { machine: MACHINE } }
    );
    const executor = createExecutor([config, denied], { onResolutionFailure });

    const unavailable = await executor.execute({
      description: 'Review on the build box.',
      subagentType: 'reviewer',
      parentToolCallId: 'call_offline',
      hostArgs: { machine: 'buildbox' },
    });
    const notAllowed = await executor.execute({
      description: 'Code on the laptop.',
      subagentType: 'coder',
      parentToolCallId: 'call_denied',
      hostArgs: { machine: 'laptop' },
    });

    expect(unavailable.content).toBe(
      'Subagent error: The requested "machine" is unavailable right now. Omit "machine" to let the host choose, or pass another listed value.'
    );
    expect(notAllowed.content).toBe(
      'Subagent error: The requested "machine" is not allowed for this subagent. Omit "machine" to let the host choose, or pass another listed value.'
    );
    expect(unavailable.resolutionFailure).toEqual({
      phase: 'config',
      cause: 'host_argument_rejected',
      hostArgument: { argument: 'machine', rejection: 'unavailable' },
    });
    expect(onResolutionFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        cause: 'host_argument_rejected',
        type: 'SubagentHostArgumentError',
      }),
      expect.any(SubagentHostArgumentError)
    );
    expect(unavailable.content).not.toContain('buildbox');
  });

  describe('background execution', () => {
    const scopeId = 'owner:conversation';

    async function waitForTask(
      store: InMemorySubagentTaskStore,
      taskId: string
    ): Promise<string | undefined> {
      for (let attempt = 0; attempt < 50; attempt += 1) {
        const task = store.get(scopeId, taskId);
        if (task != null && task.status !== 'running') {
          return task.status;
        }
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
      throw new Error(`Timed out waiting for background task ${taskId}.`);
    }

    it('validates, fingerprints, and forwards host arguments to the detached child', async () => {
      const contexts: SubagentResolveContext[] = [];
      const config = makeLazyConfig(
        'reviewer',
        async (context) => {
          contexts.push(context);
          return makeAgent();
        },
        { hostArgs: { machine: MACHINE } }
      );
      const store = new InMemorySubagentTaskStore();
      const executor = createExecutor([config], {
        taskConfig: { store, scopeId },
      });
      const params = {
        description: 'Review in the background.',
        subagentType: 'reviewer',
        parentToolCallId: 'call_background',
      };

      const rejected = JSON.parse(
        executor.executeInBackground({
          ...params,
          hostArgs: { machine: 'gpu' },
        })
      ) as { status: string; message: string };
      const started = JSON.parse(
        executor.executeInBackground({
          ...params,
          hostArgs: { machine: 'buildbox' },
        })
      ) as { background_task_id: string };
      const conflict = JSON.parse(
        executor.executeInBackground({
          ...params,
          hostArgs: { machine: 'laptop' },
        })
      ) as { status: string; message: string };
      const reused = JSON.parse(
        executor.executeInBackground({
          ...params,
          hostArgs: { machine: 'buildbox' },
        })
      ) as { background_task_id: string; message: string };

      expect(rejected.status).toBe('rejected');
      expect(rejected.message).toContain('must be one of: laptop, buildbox');
      expect(conflict).toMatchObject({
        status: 'rejected',
        message: expect.stringContaining(
          'different background subagent arguments'
        ),
      });
      expect(reused.background_task_id).toBe(started.background_task_id);
      await expect(
        waitForTask(store, started.background_task_id)
      ).resolves.toBe('completed');
      expect(contexts.map((context) => context.hostArgs)).toEqual([
        { machine: 'buildbox' },
      ]);
    });

    it('delivers a detached host-argument refusal with its specific message', async () => {
      const config = makeLazyConfig(
        'reviewer',
        async () => {
          throw new SubagentHostArgumentError('machine', 'unavailable');
        },
        { hostArgs: { machine: MACHINE } }
      );
      const store = new InMemorySubagentTaskStore();
      const executor = createExecutor([config], {
        taskConfig: { store, scopeId },
        onResolutionFailure: () => undefined,
      });

      const started = JSON.parse(
        executor.executeInBackground({
          description: 'Review in the background.',
          subagentType: 'reviewer',
          parentToolCallId: 'call_background_refused',
          hostArgs: { machine: 'buildbox' },
        })
      ) as { background_task_id: string };

      await expect(
        waitForTask(store, started.background_task_id)
      ).resolves.toBe('error');
      expect(store.claim(scopeId, started.background_task_id)).toMatchObject({
        status: 'error',
        error: expect.stringContaining(
          'The requested "machine" is unavailable right now.'
        ),
      });
    });
  });
});

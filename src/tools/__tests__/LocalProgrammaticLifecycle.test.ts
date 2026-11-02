import { z } from 'zod';
import { spawn } from 'child_process';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ChildProcess } from 'child_process';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  ProgrammaticCache,
  GenericTool,
  LocalExecutionConfig,
  LocalSpawn,
} from '@/types';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from '../local/LocalProgrammaticToolCalling';
import { MemoryRunLedger } from '@/verification/ledger';
import { registerLedgerHooks } from '@/verification/ledgerHooks';
import { HookRegistry } from '@/hooks';
import { createLocalBashExecutionTool } from '../local/LocalExecutionTools';
import { withAwaitedCancellation } from '../awaitedCancellation';

const runtimes = ['python', 'bash', 'bash-via-code'] as const;
type Runtime = (typeof runtimes)[number];

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function invoke(
  runtime: Runtime,
  nested: GenericTool,
  registry: HookRegistry,
  options: {
    signal?: AbortSignal;
    timeout?: number;
    code?: string;
    local?: LocalExecutionConfig;
    config?: Pick<RunnableConfig, 'callbacks' | 'configurable' | 'metadata'>;
  } = {}
): Promise<string> {
  const runner =
    runtime === 'bash'
      ? createLocalBashProgrammaticToolCallingTool({
        sandbox: { enabled: false },
        ...options.local,
      })
      : createLocalProgrammaticToolCallingTool({
        sandbox: { enabled: false },
        ...options.local,
      });
  const context: ProgrammaticCache = {
    toolMap: new Map([[nested.name, nested]]),
    toolDefs: [{ name: nested.name, description: 'Nested test tool' }],
    hookContext: {
      registry,
      runId: 'ptc-test',
      threadId: 'thread',
      executingAgentId: 'owner',
    },
  };
  return runner.invoke(
    {
      lang: runtime === 'python' ? 'py' : 'bash',
      code:
        options.code ??
        (runtime === 'python'
          ? 'print(await nested(value="original"))'
          : 'nested \'{"value":"original"}\''),
      timeout: options.timeout ?? 5000,
    },
    {
      ...options.config,
      signal: options.signal,
      toolCall: { ...context, name: runner.name, args: {} },
    }
  );
}

describe.each(runtimes)('local PTC lifecycle (%s)', (runtime) => {
  it('records completion and applies rewritten input and output through the real bridge', async () => {
    const registry = new HookRegistry();
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const received: string[] = [];
    const nested = tool(
      async ({ value }) => {
        received.push(value);
        return value;
      },
      {
        name: 'nested',
        description: 'Nested test tool',
        schema: z.object({ value: z.string() }),
      }
    );
    registry.register('PreToolUse', {
      hooks: [() => ({ updatedInput: { value: 'rewritten' } })],
    });
    registry.register('PostToolUse', {
      hooks: [
        (input) => {
          expect(input).toMatchObject({
            runId: 'ptc-test',
            threadId: 'thread',
            executingAgentId: 'owner',
            toolInput: { value: 'rewritten' },
            toolOutput: 'rewritten',
          });
          return { updatedOutput: 'filtered' };
        },
      ],
    });
    const output = await invoke(runtime, nested, registry);
    expect(received).toEqual(['rewritten']);
    expect(output).toContain('filtered');
    const rows = ledger.rows();
    expect(rows.map((row) => row.kind)).toEqual(['tool_call', 'tool_result']);
    expect(rows[1].metadata?.toolUseId).toBe(rows[0].metadata?.toolUseId);
  });

  it.each(['deny', 'ask'] as const)(
    'records %s as a policy denial without executing',
    async (decision) => {
      const registry = new HookRegistry();
      const ledger = new MemoryRunLedger();
      registerLedgerHooks(registry, ledger);
      const execute = jest.fn(async () => 'unexpected');
      const nested = tool(execute, {
        name: 'nested',
        description: 'Nested test tool',
        schema: z.object({ value: z.string() }),
      });
      registry.register('PreToolUse', {
        hooks: [
          () => ({
            decision,
            reason: 'blocked by policy',
            updatedInput: { value: 'sanitized' },
          }),
        ],
      });
      const denials: string[] = [];
      registry.register('PermissionDenied', {
        hooks: [
          (input) => {
            denials.push(String(input.toolInput.value));
            return {};
          },
        ],
      });
      await expect(invoke(runtime, nested, registry)).rejects.toThrow(
        'blocked by policy'
      );
      expect(execute).not.toHaveBeenCalled();
      expect(denials).toEqual(['sanitized']);
      expect(ledger.rows().map((row) => row.kind)).toEqual([
        'tool_call',
        'permission_denied',
      ]);
    }
  );

  it('records nested exceptions once as failures', async () => {
    const registry = new HookRegistry();
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const nested = tool(
      async () => {
        throw new Error('nested failed');
      },
      {
        name: 'nested',
        description: 'Nested test tool',
        schema: z.object({ value: z.string() }),
      }
    );
    await expect(invoke(runtime, nested, registry)).rejects.toThrow();
    expect(ledger.rows().map((row) => row.kind)).toEqual([
      'tool_call',
      'tool_failure',
    ]);
  });

  it('allows a post hook to replace output with null', async () => {
    const registry = new HookRegistry();
    const nested = tool(async () => 'private result', {
      name: 'nested',
      description: 'Nested test tool',
      schema: z.object({ value: z.string() }),
    });
    registry.register('PostToolUse', {
      hooks: [() => ({ updatedOutput: null })],
    });
    const output = await invoke(runtime, nested, registry);
    expect(output).toContain(runtime === 'python' ? 'None' : 'null');
    expect(output).not.toContain('private result');
  });

  it('records argument validation failures through the same lifecycle', async () => {
    const registry = new HookRegistry();
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const execute = jest.fn(async () => 'unexpected');
    const nested = tool(execute, {
      name: 'nested',
      description: 'Strict tool',
      schema: z.object({ value: z.string() }),
    });
    const code =
      runtime === 'python'
        ? 'await nested(value=42)'
        : 'nested \'{"value":42}\'';
    await expect(invoke(runtime, nested, registry, { code })).rejects.toThrow();
    expect(execute).not.toHaveBeenCalled();
    expect(ledger.rows().map((row) => row.kind)).toEqual([
      'tool_call',
      'tool_failure',
    ]);
  });

  it.each(['abort', 'timeout'] as const)(
    'terminates owned nested process trees on %s before settling',
    async (mode) => {
      const children: ChildProcess[] = [];
      const started = deferred();
      const controller = new AbortController();
      const launcher: LocalSpawn = (command, args, options) => {
        const child = spawn(command, args, { ...options, stdio: 'pipe' });
        children.push(child);
        child.stdout.on('data', (chunk: Buffer) => {
          if (chunk.toString().includes('nested-ready')) started.resolve();
        });
        return child;
      };
      const registry = new HookRegistry();
      const ledger = new MemoryRunLedger();
      registerLedgerHooks(registry, ledger);
      const nested = createLocalBashExecutionTool({
        config: {
          spawn: launcher,
          sandbox: { enabled: false },
          timeoutMs: 10000,
        },
      });
      const command = 'printf "nested-ready\\n"; sleep "$1"';
      const code =
        runtime === 'python'
          ? `await bash_tool(command=${JSON.stringify(command)}, args=["30"])`
          : `bash_tool '${JSON.stringify({ command, args: ['30'] })}'`;
      const running = invoke(runtime, nested, registry, {
        code,
        timeout: mode === 'timeout' ? 1000 : 5000,
        signal: controller.signal,
        local: { spawn: launcher },
      });
      const outcome = running.then(
        () => false,
        () => true
      );
      try {
        await started.promise;
        if (mode === 'abort') controller.abort(new Error('cancelled'));
        expect(await outcome).toBe(true);
        expect(children.length).toBeGreaterThanOrEqual(2);
        expect(
          children.every(
            (child) => child.exitCode !== null || child.signalCode !== null
          )
        ).toBe(true);
        expect(ledger.rows().map((row) => row.kind)).toEqual([
          'tool_call',
          'tool_failure',
        ]);
      } finally {
        controller.abort();
        await outcome;
      }
    }
  );

  it('drains unowned nested tools before returning from cancellation', async () => {
    const registry = new HookRegistry();
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const started = deferred();
    const finish = deferred();
    const controller = new AbortController();
    let settled = false;
    let completed = false;
    let nestedSignal: AbortSignal | undefined;
    const nested = tool(
      async (_input, config) => {
        nestedSignal = config.signal;
        started.resolve();
        await finish.promise;
        completed = true;
        return 'finished';
      },
      {
        name: 'nested',
        description: 'Unowned tool',
        schema: z.object({ value: z.string() }),
      }
    );
    const running = invoke(runtime, nested, registry, {
      signal: controller.signal,
    }).then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      }
    );
    try {
      await started.promise;
      controller.abort(new Error('cancelled'));
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(nestedSignal).toBeUndefined();
      expect(settled).toBe(false);
    } finally {
      finish.resolve();
      await running;
    }
    expect(completed).toBe(true);
    expect(ledger.rows().map((row) => row.kind)).toEqual([
      'tool_call',
      'tool_result',
    ]);
  });
});

it('preserves nested callback parentage, metadata and configurable context', async () => {
  const starts: { id: string; parent?: string }[] = [];
  let nestedConfig: RunnableConfig | undefined;
  const nested = tool(
    async (_input, config) => {
      nestedConfig = config;
      return 'done';
    },
    {
      name: 'nested',
      description: 'Tracing fixture',
      schema: z.object({ value: z.string() }),
    }
  );
  await invoke('bash', nested, new HookRegistry(), {
    config: {
      configurable: { run_id: 'run', tenant: 'tenant' },
      metadata: { agentId: 'owner' },
      callbacks: [
        {
          name: 'callback-fixture',
          handleToolStart: (_tool, _input, id, parent) => {
            starts.push({ id, parent });
          },
        },
      ],
    },
  });
  expect(starts).toHaveLength(2);
  expect(starts[1].parent).toBe(starts[0].id);
  expect(nestedConfig?.configurable).toMatchObject({
    run_id: 'run',
    tenant: 'tenant',
  });
  expect(nestedConfig?.metadata).toMatchObject({
    agentId: 'owner',
    run_tools_with_bash: true,
  });
  expect(nestedConfig?.signal).toBeUndefined();
});

it('records successful completion even if cancellation races with its result', async () => {
  const controller = new AbortController();
  const registry = new HookRegistry();
  const ledger = new MemoryRunLedger();
  registerLedgerHooks(registry, ledger);
  const nested = withAwaitedCancellation(
    new DynamicStructuredTool({
      name: 'nested',
      description: 'Completed tool',
      schema: z.object({ value: z.string() }),
      func: async (): Promise<string> => {
        controller.abort();
        return 'completed';
      },
    })
  );
  await expect(
    invoke('bash', nested, registry, { signal: controller.signal })
  ).rejects.toThrow();
  expect(ledger.rows().map((row) => row.kind)).toEqual([
    'tool_call',
    'tool_result',
  ]);
});

it('cancels a disconnected bridge request and drains its cleanup', async () => {
  const started = deferred();
  const cancelled = deferred();
  const cleanup = deferred();
  let signal: AbortSignal | undefined;
  let completed = false;
  let settled = false;
  const nested = withAwaitedCancellation(
    new DynamicStructuredTool({
      name: 'nested',
      description: 'Cooperative tool',
      schema: z.object({ value: z.string() }),
      func: async (_input, _manager, config): Promise<string> => {
        signal = config?.signal;
        signal?.addEventListener('abort', cancelled.resolve, { once: true });
        started.resolve();
        await cancelled.promise;
        await cleanup.promise;
        completed = true;
        signal?.throwIfAborted();
        return 'unexpected';
      },
    })
  );
  const registry = new HookRegistry();
  const ledger = new MemoryRunLedger();
  registerLedgerHooks(registry, ledger);
  const code =
    'curl() { command curl --max-time 0.5 "$@"; }\nnested \'{"value":"original"}\'';
  const running = invoke('bash', nested, registry, { code }).then(
    () => {
      settled = true;
    },
    () => {
      settled = true;
    }
  );
  try {
    await started.promise;
    await cancelled.promise;
    expect(signal?.aborted).toBe(true);
    expect(settled).toBe(false);
  } finally {
    cleanup.resolve();
    cancelled.resolve();
    await running;
  }
  expect(completed).toBe(true);
  expect(ledger.rows().map((row) => row.kind)).toEqual([
    'tool_call',
    'tool_failure',
  ]);
});

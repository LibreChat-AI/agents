import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  createCloudflareExecutionWorld,
  createCloudflareWorkspaceFS,
  createCloudflareLocalExecutionConfig,
  execWithClientTimeout,
  executeCloudflareBash,
  executeCloudflareCode,
} from '../cloudflare/CloudflareSandboxExecutionEngine';
import {
  createCloudflareBashProgrammaticToolCallingTool,
  createCloudflareProgrammaticToolCallingTool,
} from '../cloudflare/CloudflareProgrammaticToolCalling';
import {
  runPostEditSyntaxCheck,
  _resetSyntaxCheckProbeCacheForTests,
} from '../local/syntaxCheck';
import { createCloudflareBridgeRuntime } from '../cloudflare/CloudflareBridgeRuntime';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';
import { isWorkspaceClientTimeoutError } from '../local/workspaceFS';
import { spawnLocalProcess } from '../local/LocalExecutionEngine';
import { Constants } from '@/common';

function sseResponse(events: string): Response {
  return new Response(events, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function sseExit(exitCode = 0): Response {
  return sseResponse(`event: exit\ndata: {"exit_code":${exitCode}}\n\n`);
}

function bodyText(body: BodyInit | null | undefined): string {
  if (typeof body === 'string') {
    return body;
  }
  if (body == null) {
    return '';
  }
  if (body instanceof Uint8Array) {
    return Buffer.from(body).toString('utf8');
  }
  return String(body);
}

function createRuntime(
  overrides: Partial<t.CloudflareSandboxRuntime> = {}
): t.CloudflareSandboxRuntime {
  return {
    exec: async () => ({
      exitCode: 0,
      stdout: '',
      stderr: '',
    }),
    readFile: async () => '',
    writeFile: async () => ({ ok: true }),
    mkdir: async () => ({ ok: true }),
    listFiles: async () => [],
    deleteFile: async () => ({ ok: true }),
    ...overrides,
  };
}

it('reuses one execution world for repeated Cloudflare tool bindings', () => {
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: createRuntime(),
  };

  const firstWorld = createCloudflareExecutionWorld(config);
  const secondWorld = createCloudflareExecutionWorld(config);
  const localConfig = createCloudflareLocalExecutionConfig(config);

  expect(secondWorld).toBe(firstWorld);
  expect(localConfig.exec).toBe(firstWorld);
  expect(firstWorld.sandboxed).toBe(true);
  expect(Object.isFrozen(firstWorld.fs)).toBe(true);
  expect(Object.isFrozen(firstWorld)).toBe(true);
});

it('rebuilds the world when captured Cloudflare settings change', async () => {
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: createRuntime(),
    workspaceRoot: '/first',
    timeoutMs: 100,
  };

  const firstWorld = createCloudflareExecutionWorld(config);
  config.workspaceRoot = '/second';
  const secondWorld = createCloudflareExecutionWorld(config);
  config.timeoutMs = 200;
  const thirdWorld = createCloudflareExecutionWorld(config);

  expect(secondWorld).not.toBe(firstWorld);
  expect(thirdWorld).not.toBe(secondWorld);
  await expect(secondWorld.fs.realpath('/second/file.ts')).resolves.toBe(
    '/second/file.ts'
  );
  await expect(firstWorld.fs.realpath('/second/file.ts')).rejects.toThrow(
    'outside the Cloudflare sandbox workspace'
  );
});

it('preserves a prewarmed factory when creating the first world', async () => {
  let factoryCalls = 0;
  const runtime = createRuntime({ readFile: async () => 'ok' });
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: async () => {
      factoryCalls++;
      return runtime;
    },
  };

  await createCloudflareWorkspaceFS(config).readFile('/workspace/first.txt');
  const world = createCloudflareExecutionWorld(config);
  await world.fs.readFile('/workspace/second.txt');

  expect(factoryCalls).toBe(1);
});

it('does not let a stale factory rejection evict its replacement', async () => {
  let rejectFirst: ((error: Error) => void) | undefined;
  const firstFactory = new Promise<t.CloudflareSandboxRuntime>(
    (_resolve, reject) => {
      rejectFirst = reject;
    }
  );
  let replacementCalls = 0;
  const replacement = createRuntime({ readFile: async () => 'replacement' });
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: () => firstFactory,
  };

  const firstWorld = createCloudflareExecutionWorld(config);
  const staleRead = firstWorld.fs.readFile('/workspace/stale.txt');
  await Promise.resolve();
  config.sandbox = async () => {
    replacementCalls++;
    return replacement;
  };
  const replacementWorld = createCloudflareExecutionWorld(config);
  await expect(
    replacementWorld.fs.readFile('/workspace/replacement.txt', 'utf8')
  ).resolves.toBe('replacement');
  rejectFirst?.(new Error('stale factory failed'));
  await expect(staleRead).rejects.toThrow('stale factory failed');
  await expect(
    replacementWorld.fs.readFile('/workspace/reused.txt', 'utf8')
  ).resolves.toBe('replacement');

  expect(replacementCalls).toBe(1);
});

it('keeps remote capability probes warm across Cloudflare tool bindings', async () => {
  _resetSyntaxCheckProbeCacheForTests();
  let execCalls = 0;
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: createRuntime({
      exec: async () => {
        execCalls++;
        return { exitCode: 0, stdout: '', stderr: '' };
      },
    }),
  };

  const firstBinding = createCloudflareLocalExecutionConfig(config);
  const secondBinding = createCloudflareLocalExecutionConfig(config);
  await runPostEditSyntaxCheck('/workspace/first.js', firstBinding);
  await runPostEditSyntaxCheck('/workspace/second.js', secondBinding);

  expect(execCalls).toBe(3);
});

it('retries a transient capability-probe failure in a stable world', async () => {
  _resetSyntaxCheckProbeCacheForTests();
  let execCalls = 0;
  const config: t.CloudflareSandboxExecutionConfig = {
    sandbox: createRuntime({
      exec: async () => {
        execCalls++;
        if (execCalls === 1) {
          throw new Error('transient sandbox RPC failure');
        }
        if (execCalls === 2) {
          return { exitCode: 0, stdout: 'v22', stderr: '' };
        }
        return { exitCode: 1, stdout: '', stderr: 'syntax error' };
      },
    }),
  };

  const first = await runPostEditSyntaxCheck(
    '/workspace/first.js',
    createCloudflareLocalExecutionConfig(config)
  );
  const second = await runPostEditSyntaxCheck(
    '/workspace/second.js',
    createCloudflareLocalExecutionConfig(config)
  );

  expect(first).toEqual({ ok: true });
  expect(second).toEqual({
    ok: false,
    checker: 'node --check',
    output: 'syntax error',
  });
  expect(execCalls).toBe(3);
});

it('reprobes an expired negative capability in a stateful world', async () => {
  jest.useFakeTimers();
  try {
    jest.setSystemTime(0);
    let execCalls = 0;
    const config: t.CloudflareSandboxExecutionConfig = {
      sandbox: createRuntime({
        exec: async () => {
          execCalls++;
          if (execCalls === 1) {
            return { exitCode: 127, stdout: '', stderr: 'node not found' };
          }
          if (execCalls === 2) {
            return { exitCode: 0, stdout: 'v22', stderr: '' };
          }
          return { exitCode: 1, stdout: '', stderr: 'syntax error' };
        },
      }),
    };

    const first = await runPostEditSyntaxCheck(
      '/workspace/first.js',
      createCloudflareLocalExecutionConfig(config)
    );
    jest.setSystemTime(6000);
    const second = await runPostEditSyntaxCheck(
      '/workspace/second.js',
      createCloudflareLocalExecutionConfig(config)
    );

    expect(first).toEqual({ ok: true });
    expect(second).toEqual({
      ok: false,
      checker: 'node --check',
      output: 'syntax error',
    });
    expect(execCalls).toBe(3);
  } finally {
    jest.useRealTimers();
  }
});

it('reports the invoked runner name for Cloudflare caller-policy failures', async () => {
  const programmatic = createCloudflareProgrammaticToolCallingTool({
    sandbox: createRuntime(),
  });
  const toolCall = {
    id: 'cloudflare-ptc',
    name: Constants.PROGRAMMATIC_TOOL_CALLING,
    type: 'tool_call' as const,
    args: {},
    programmaticToolName: Constants.PROGRAMMATIC_TOOL_CALLING,
    disallowedToolDefs: [{ name: 'direct_only_tool' }],
  } satisfies ToolCall & Partial<t.ProgrammaticCache>;

  await expect(
    programmatic.invoke(
      {
        code: 'direct_only_tool \'{}\'',
        tool_manifest: ['direct_only_tool'],
      },
      { toolCall }
    )
  ).rejects.toThrow(
    'Tool "direct_only_tool" cannot be called from "run_tools_with_code"'
  );
});

describe('Cloudflare sandbox execution backend', () => {
  it('normalizes trailing workspace slashes before clamping paths', async () => {
    const readPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/workspace/',
      sandbox: createRuntime({
        readFile: async (filePath) => {
          readPaths.push(filePath);
          return 'ok';
        },
      }),
    });

    await expect(fs.readFile('file.txt', 'utf8')).resolves.toBe('ok');
    expect(readPaths).toEqual(['/workspace/file.txt']);
  });

  it('allows root workspace paths', async () => {
    const readPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/',
      sandbox: createRuntime({
        readFile: async (filePath) => {
          readPaths.push(filePath);
          return 'ok';
        },
      }),
    });

    await expect(fs.readFile('tmp/file.txt', 'utf8')).resolves.toBe('ok');
    expect(readPaths).toEqual(['/tmp/file.txt']);
  });

  it('stats the workspace root without listing its parent directory', async () => {
    const listPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/workspace/',
      sandbox: createRuntime({
        listFiles: async (filePath) => {
          listPaths.push(filePath);
          return [{ name: 'src', type: 'directory' }];
        },
      }),
    });

    const stats = await fs.stat('.');

    expect(stats.isDirectory()).toBe(true);
    expect(listPaths).toEqual(['/workspace']);
  });

  it('does not pass AbortSignal to Cloudflare spawn exec options', async () => {
    let resolveExecCalled!: () => void;
    const execCalled = new Promise<void>((resolve) => {
      resolveExecCalled = resolve;
    });
    let receivedOptions: t.CloudflareSandboxExecOptions | undefined;
    const sandbox = createRuntime({
      exec: (_command, options) => {
        receivedOptions = options;
        resolveExecCalled();
        return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      timeoutMs: 50,
      workspaceRoot: '/workspace',
    });

    const resultPromise = spawnLocalProcess(
      'bash',
      ['-lc', 'sleep 10'],
      config
    );
    await execCalled;
    const result = await resultPromise;

    expect(receivedOptions).not.toHaveProperty('signal');
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
  });

  it('passes AbortSignal to signal-aware runtimes and aborts it on kill', async () => {
    let resolveExecCalled!: () => void;
    const execCalled = new Promise<void>((resolve) => {
      resolveExecCalled = resolve;
    });
    let receivedSignal: AbortSignal | undefined;
    let abortEvents = 0;
    const sandbox = createRuntime({
      supportsExecSignal: true,
      exec: (_command, options) => {
        receivedSignal = options?.signal;
        receivedSignal?.addEventListener('abort', () => {
          abortEvents += 1;
        });
        resolveExecCalled();
        return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      timeoutMs: 50,
      workspaceRoot: '/workspace',
    });

    const resultPromise = spawnLocalProcess(
      'bash',
      ['-lc', 'sleep 10'],
      config
    );
    await execCalled;
    const result = await resultPromise;

    expect(receivedSignal).toBeDefined();
    expect(receivedSignal?.aborted).toBe(true);
    expect(abortEvents).toBe(1);
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
  });

  it('does not start remote exec when killed before async sandbox resolution finishes', async () => {
    let execCalls = 0;
    let resolveSandbox!: (runtime: t.CloudflareSandboxRuntime) => void;
    const sandboxPromise = new Promise<t.CloudflareSandboxRuntime>(
      (resolve) => {
        resolveSandbox = resolve;
      }
    );
    const config = createCloudflareLocalExecutionConfig({
      sandbox: () => sandboxPromise,
      timeoutMs: 10,
      workspaceRoot: '/workspace',
    });

    const result = await spawnLocalProcess('bash', ['-lc', 'sleep 10'], config);
    resolveSandbox(
      createRuntime({
        exec: async () => {
          execCalls += 1;
          return { exitCode: 0, stdout: '', stderr: '' };
        },
      })
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
    expect(execCalls).toBe(0);
  });

  it('memoizes sandbox factory results per config object', async () => {
    let calls = 0;
    const runtime = createRuntime({
      readFile: async () => 'ok',
      writeFile: async () => ({ ok: true }),
    });
    const config = {
      workspaceRoot: '/workspace',
      sandbox: async (): Promise<t.CloudflareSandboxRuntime> => {
        calls += 1;
        return runtime;
      },
    };
    const fs = createCloudflareWorkspaceFS(config);

    await fs.readFile('a.txt', 'utf8');
    await fs.writeFile('b.txt', 'ok', 'utf8');

    expect(calls).toBe(1);
  });

  it('wraps direct bash commands with an in-sandbox timeout', async () => {
    let execCommand = '';
    let execTimeout: number | undefined;
    let calls = 0;
    const sandbox = createRuntime({
      exec: async (command, options) => {
        calls += 1;
        execCommand = command;
        execTimeout = options?.timeout;
        return {
          exitCode: calls === 1 ? 0 : 124,
          stdout: 'ok',
          stderr: '',
        };
      },
    });

    const result = await executeCloudflareBash('echo ok', {
      sandbox,
      workspaceRoot: '/workspace',
      timeoutMs: 1500,
    });

    expect(execCommand).toContain('timeout -k 2s 2s bash -lc');
    expect(execTimeout).toBe(6500);
    expect(result.timedOut).toBe(true);
  });

  it('rejects with a client-side timeout when sandbox exec stalls (no native cancellation)', async () => {
    // The native Cloudflare Sandbox DO exec() is uncancellable (ExecOptions has no
    // signal) and its own `timeout` is not enforced when the container/RPC stalls,
    // while the in-sandbox `timeout(1)` wrapper only bounds a *running* command.
    // Without a client-side race a stalled exec hangs until the host's run-level
    // abort, burning the whole budget on one tool call (issue #251).
    jest.useFakeTimers();
    try {
      let mainExecCalls = 0;
      const sandbox = createRuntime({
        exec: (command) => {
          // Cleanup (`rm -rf`) resolves immediately; the real command stalls,
          // simulating an unresponsive / cold container exec that never returns.
          if (command.startsWith('rm -rf')) {
            return Promise.resolve({ exitCode: 0, stdout: '', stderr: '' });
          }
          mainExecCalls += 1;
          return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
        },
      });

      const promise = executeCloudflareCode(
        { lang: 'py', code: 'print("slow")' },
        { sandbox, workspaceRoot: '/workspace', timeoutMs: 1000 }
      );
      const assertion = expect(promise).rejects.toThrow(/client-side timeout/);

      // Client backstop = outerTimeoutMs(1000) + 5000 = 11000ms; advance past it.
      await jest.advanceTimersByTimeAsync(11500);
      await assertion;
      expect(mainExecCalls).toBe(1);
    } finally {
      jest.useRealTimers();
    }
  });

  it('aborts signal-aware execs when the client timeout fires', async () => {
    // For signal-aware transports (e.g. the HTTP bridge), a client timeout should
    // actually cancel the underlying exec, not just abandon it.
    jest.useFakeTimers();
    try {
      let mainSignal: AbortSignal | undefined;
      const sandbox = createRuntime({
        supportsExecSignal: true,
        exec: (command, options) => {
          if (command.startsWith('rm -rf')) {
            return Promise.resolve({ exitCode: 0, stdout: '', stderr: '' });
          }
          mainSignal = options?.signal;
          return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
        },
      });

      const promise = executeCloudflareCode(
        { lang: 'py', code: 'print("slow")' },
        { sandbox, workspaceRoot: '/workspace', timeoutMs: 1000 }
      );
      const assertion = expect(promise).rejects.toThrow(/client-side timeout/);
      await jest.advanceTimersByTimeAsync(11500);
      await assertion;

      expect(mainSignal).toBeDefined();
      expect(mainSignal?.aborted).toBe(true);
    } finally {
      jest.useRealTimers();
    }
  });

  it('composes a caller abort signal with the timeout instead of clobbering it', async () => {
    let received: AbortSignal | undefined;
    const sandbox = createRuntime({
      supportsExecSignal: true,
      exec: (_command, options) => {
        received = options?.signal;
        return new Promise<t.CloudflareSandboxExecResult>(
          (_resolve, reject) => {
            options?.signal?.addEventListener(
              'abort',
              () => reject(new Error('aborted')),
              { once: true }
            );
          }
        );
      },
    });

    const caller = new AbortController();
    const settled = execWithClientTimeout(
      sandbox,
      'echo hi',
      { signal: caller.signal },
      60000,
      'test'
    ).catch((e) => e as Error);

    await Promise.resolve();
    expect(received).toBeDefined();
    // The exec gets a composed signal, not the caller's directly.
    expect(received).not.toBe(caller.signal);
    expect(received?.aborted).toBe(false);

    // A caller cancellation must reach the exec (not wait for the client timeout).
    caller.abort();
    await settled;
    expect(received?.aborted).toBe(true);
  });

  it('strips a caller signal for native runtimes that cannot consume it', async () => {
    let received: t.CloudflareSandboxExecOptions | undefined;
    const sandbox = createRuntime({
      // no supportsExecSignal -> native DO, which cannot clone/consume a signal
      exec: async (_command, options) => {
        received = options;
        return { exitCode: 0, stdout: 'ok', stderr: '' };
      },
    });
    const caller = new AbortController();

    await execWithClientTimeout(
      sandbox,
      'echo hi',
      { cwd: '/workspace', signal: caller.signal },
      60000,
      'test'
    );

    expect(received).toBeDefined();
    expect(received).not.toHaveProperty('signal');
  });

  it('rejects with a client-side timeout when sandbox readFile stalls', async () => {
    // The native-DO file-IO RPCs (readFile/writeFile/listFiles/...) have the same
    // stall hazard exec() does: no signal, no enforced timeout. A cold/unresponsive
    // container otherwise hangs the host await until the run-level abort, burning
    // the whole budget on one read (observed live: a `read_file` stalled ~552s).
    jest.useFakeTimers();
    try {
      let readCalls = 0;
      const fs = createCloudflareWorkspaceFS({
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
        sandbox: createRuntime({
          readFile: () => {
            readCalls += 1;
            return new Promise<string>(() => undefined);
          },
        }),
      });

      const error = (fs.readFile as (p: string) => Promise<unknown>)(
        '/workspace/a.txt'
      ).catch((e: unknown) => e);
      // Client backstop = clientFsTimeoutMs(1000) = 6000ms; advance past it.
      await jest.advanceTimersByTimeAsync(6500);
      const settled = await error;
      // Must be the DISTINGUISHABLE timeout error so ENOENT-only callers rethrow
      // it instead of mistaking a stalled read for a missing file.
      expect(isWorkspaceClientTimeoutError(settled)).toBe(true);
      expect((settled as Error).message).toMatch(/client-side timeout/);
      expect(readCalls).toBe(1);
    } finally {
      jest.useRealTimers();
    }
  });

  it('keeps the backstop active while draining a streamed file read', async () => {
    // sandbox.readFile resolves to { content: ReadableStream } whose stream never
    // ends. The race must cover the drain (normalizeReadFileContent), not just the
    // initial RPC, or read_file/open/stat still hang to the run-level abort.
    jest.useFakeTimers();
    try {
      const fs = createCloudflareWorkspaceFS({
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
        sandbox: createRuntime({
          readFile: async () => ({
            content: new ReadableStream<Uint8Array>({
              // start() never enqueues or closes -> the drain stalls forever.
              start() {},
            }),
          }),
        }),
      });

      const error = (fs.readFile as (p: string) => Promise<unknown>)(
        '/workspace/a.txt'
      ).catch((e: unknown) => e);
      await jest.advanceTimersByTimeAsync(6500);
      const settled = await error;
      expect(isWorkspaceClientTimeoutError(settled)).toBe(true);
    } finally {
      jest.useRealTimers();
    }
  });

  it('rethrows a stat directory-probe timeout instead of falling through to readFile', async () => {
    // findChildInfo returns nothing -> the directory probe (listFiles) runs; if it
    // STALLS it must surface, not fall through to the readFile branch (which would
    // burn a SECOND full backstop, ~2x the timeout, before the caller sees it).
    jest.useFakeTimers();
    try {
      let readFileCalls = 0;
      const fs = createCloudflareWorkspaceFS({
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
        sandbox: createRuntime({
          listFiles: (dir) =>
            dir === '/workspace/probe-me'
              ? new Promise(() => undefined) // the probe stalls
              : Promise.resolve([]), // findChildInfo's parent listing returns fast
          readFile: () => {
            readFileCalls += 1;
            return Promise.resolve('');
          },
        }),
      });

      const error = fs.stat('/workspace/probe-me').catch((e: unknown) => e);
      await jest.advanceTimersByTimeAsync(6500);
      const settled = await error;
      expect(isWorkspaceClientTimeoutError(settled)).toBe(true);
      // Must NOT have fallen through to the readFile probe.
      expect(readFileCalls).toBe(0);
    } finally {
      jest.useRealTimers();
    }
  });

  it('bounds execute_code temp-dir setup RPCs (mkdir/writeFile) that stall', async () => {
    jest.useFakeTimers();
    try {
      const sandbox = createRuntime({
        // exec would resolve fine; the stall is in the pre-exec mkdir setup.
        mkdir: () => new Promise<{ ok: true }>(() => undefined),
      });

      const promise = executeCloudflareCode(
        { lang: 'py', code: 'print("hi")' },
        { sandbox, workspaceRoot: '/workspace', timeoutMs: 1000 }
      );
      const assertion = expect(promise).rejects.toThrow(/client-side timeout/);
      await jest.advanceTimersByTimeAsync(6500);
      await assertion;
    } finally {
      jest.useRealTimers();
    }
  });

  it('still cleans up the temp dir when execute_code setup (writeFile) times out', async () => {
    // The setup RPCs are inside the try, so a stalled writeFile still triggers
    // the finally cleanup — otherwise the late (uncancellable) write leaves an
    // orphaned .lc-exec/<uuid> dir behind on every cold-container failure.
    jest.useFakeTimers();
    try {
      const execCommands: string[] = [];
      const sandbox = createRuntime({
        mkdir: async () => ({ ok: true }),
        writeFile: () => new Promise<{ ok: true }>(() => undefined), // stalls
        exec: async (command) => {
          execCommands.push(command);
          return { exitCode: 0, stdout: '', stderr: '' };
        },
      });

      const promise = executeCloudflareCode(
        { lang: 'py', code: 'print("hi")' },
        { sandbox, workspaceRoot: '/workspace', timeoutMs: 1000 }
      ).catch((e: unknown) => e);
      await jest.advanceTimersByTimeAsync(6500);
      const settled = await promise;
      expect(isWorkspaceClientTimeoutError(settled)).toBe(true);
      // Cleanup must have been issued despite the setup failure.
      expect(execCommands.some((c) => c.startsWith('rm -rf'))).toBe(true);
    } finally {
      jest.useRealTimers();
    }
  });

  it('rejects with a client-side timeout when sandbox listFiles stalls', async () => {
    jest.useFakeTimers();
    try {
      const fs = createCloudflareWorkspaceFS({
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
        sandbox: createRuntime({
          listFiles: () =>
            new Promise<t.CloudflareSandboxFileInfo[]>(() => undefined),
        }),
      });

      const promise = (fs.readdir as (p: string) => Promise<unknown>)(
        '/workspace/sub'
      );
      const assertion = expect(promise).rejects.toThrow(/client-side timeout/);
      await jest.advanceTimersByTimeAsync(6500);
      await assertion;
    } finally {
      jest.useRealTimers();
    }
  });

  it('does not time out a native FS RPC that returns in time', async () => {
    jest.useFakeTimers();
    try {
      const fs = createCloudflareWorkspaceFS({
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
        sandbox: createRuntime({ readFile: async () => 'contents' }),
      });

      const result = await (
        fs.readFile as (p: string, e: 'utf8') => Promise<string>
      )('/workspace/a.txt', 'utf8');
      expect(result).toBe('contents');
      // The backstop timer must have been cleared, not left dangling.
      expect(jest.getTimerCount()).toBe(0);
    } finally {
      jest.useRealTimers();
    }
  });

  it('passes call-specific timeouts to the Cloudflare spawn wrapper', async () => {
    let execCommand = '';
    let execTimeout: number | undefined;
    const sandbox = createRuntime({
      exec: async (command, options) => {
        execCommand = command;
        execTimeout = options?.timeout;
        return {
          exitCode: 0,
          stdout: 'ok',
          stderr: '',
        };
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      workspaceRoot: '/workspace',
      timeoutMs: 1000,
    });

    await expect(
      spawnLocalProcess('bash', ['-lc', 'echo ok'], {
        ...config,
        timeoutMs: 120000,
      })
    ).resolves.toMatchObject({ exitCode: 0, timedOut: false });

    expect(execCommand).toContain('timeout -k 2s 120s bash -lc');
    expect(execTimeout).toBe(125000);
  });

  it('marks Cloudflare code execution timeouts', async () => {
    const sandbox = createRuntime({
      exec: async (command) => ({
        exitCode: command.startsWith('rm -rf') ? 0 : 124,
        stdout: '',
        stderr: '',
      }),
    });

    const result = await executeCloudflareCode(
      { lang: 'py', code: 'print("slow")' },
      {
        sandbox,
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
      }
    );

    expect(result.timedOut).toBe(true);
  });

  it('forwards only explicit Cloudflare env vars to sandbox exec', async () => {
    let execEnv: Record<string, string | undefined> | undefined;
    const sandbox = createRuntime({
      exec: async (_command, options) => {
        execEnv = options?.env;
        return {
          exitCode: 0,
          stdout: 'ok',
          stderr: '',
        };
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      workspaceRoot: '/workspace',
      env: { SAFE_FOR_SANDBOX: 'yes' },
    });

    await spawnLocalProcess('bash', ['-lc', 'echo ok'], config);

    expect(execEnv).toEqual({ SAFE_FOR_SANDBOX: 'yes' });
  });

  it('injects read-only and workspace guards into Python programmatic tools', async () => {
    let source = '';
    const sandbox = createRuntime({
      writeFile: async (_path, content) => {
        source = String(content);
        return { ok: true };
      },
      exec: async () => ({
        exitCode: 0,
        stdout: 'done',
        stderr: '',
      }),
    });
    const programmatic = createCloudflareProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
      readOnly: true,
      shell: '/bin/sh',
    });

    await programmatic.invoke({
      code: 'await write_file("x.txt", "blocked")',
      lang: 'py',
    });

    expect(source).toContain('READ_ONLY = True');
    expect(source).toContain('SHELL = "/bin/sh"');
    expect(source).toContain('[SHELL, "-lc", command, "--"]');
    expect(source).toContain('_assert_writable("write_file")');
    expect(source).toContain('if _is_within_workspace(resolved):');
    expect(source).toContain('_validate_bash_command(command, args=args)');
  });

  it('clamps programmatic timeouts before sandbox execution', async () => {
    const timeouts: Array<number | undefined> = [];
    const sandbox = createRuntime({
      writeFile: async () => ({ ok: true }),
      exec: async (_command, options) => {
        timeouts.push(options?.timeout);
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const programmatic = createCloudflareProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
    });

    await programmatic.invoke({
      code: 'print("ok")',
      lang: 'py',
      timeout: 300000,
    });

    expect(timeouts[0]).toBe(305000);
  });

  it('injects bash validation into bash programmatic tools', async () => {
    let execCommand = '';
    const sandbox = createRuntime({
      exec: async (command) => {
        execCommand = command;
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const programmatic = createCloudflareBashProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
      shell: '/bin/sh',
    });

    await programmatic.invoke({
      code: 'printf "%s\\n" "ok"',
    });

    expect(execCommand).toContain('const ALLOW_DANGEROUS_COMMANDS = false;');
    expect(execCommand).toContain('const SHELL = "/bin/sh";');
    expect(execCommand).toContain('cp.spawn(SHELL, ["-lc", command, "--"');
    expect(execCommand).toContain(
      'function validateBashCommand(command, args)'
    );
    expect(execCommand).toContain('validateBashCommand(command, args);');
  });

  it('uses root-safe path containment in programmatic helpers', async () => {
    let pythonSource = '';
    const pythonSandbox = createRuntime({
      writeFile: async (_path, content) => {
        pythonSource = String(content);
        return { ok: true };
      },
      exec: async () => ({
        exitCode: 0,
        stdout: 'done',
        stderr: '',
      }),
    });
    const pythonProgrammatic = createCloudflareProgrammaticToolCallingTool({
      sandbox: pythonSandbox,
      workspaceRoot: '/',
    });

    await pythonProgrammatic.invoke({
      code: 'print("ok")',
      lang: 'py',
    });

    expect(pythonSource).toContain('WORKSPACE = "/"');
    expect(pythonSource).toContain(
      'return os.path.commonpath([root, resolved]) == root'
    );

    let bashCommand = '';
    const bashSandbox = createRuntime({
      exec: async (command) => {
        bashCommand = command;
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const bashProgrammatic = createCloudflareBashProgrammaticToolCallingTool({
      sandbox: bashSandbox,
      workspaceRoot: '/',
    });

    await bashProgrammatic.invoke({
      code: 'printf "%s\\n" "ok"',
    });

    expect(bashCommand).toContain('const WORKSPACE = "/";');
    expect(bashCommand).toContain(
      'const relative = path.relative(root, resolved);'
    );
    expect(bashCommand).toContain(
      'relative.startsWith("..") || path.isAbsolute(relative)'
    );
  });

  it('enforces Cloudflare codingToolNames as an allowlist', () => {
    const tools = resolveLocalToolsForBinding({
      tools: [
        { name: Constants.EXECUTE_CODE } as t.GenericTool,
        { name: Constants.BASH_TOOL } as t.GenericTool,
      ],
      toolExecution: {
        engine: 'cloudflare-sandbox',
        cloudflare: {
          sandbox: createRuntime(),
          codingToolNames: [Constants.BASH_TOOL],
        },
      },
    }) as t.GenericTool[];
    const names = tools.map((toolDef) => toolDef.name);

    expect(names).toContain(Constants.BASH_TOOL);
    expect(names).not.toContain(Constants.EXECUTE_CODE);
  });
});

describe('Cloudflare bridge runtime', () => {
  it.each(['utf8', 'base64', 'stream'] as const)(
    'preserves %s file contents in the HTTP body',
    async (encoding) => {
      const bytes = Buffer.from('héllo\0');
      const fetchImpl: typeof fetch = async (input, init) => {
        const request = new Request(input, init);
        expect(request.method).toBe('PUT');
        expect(request.headers.get('Content-Type')).toBe(
          'application/octet-stream'
        );
        expect(Buffer.from(await request.arrayBuffer())).toEqual(bytes);
        return Response.json({ ok: true });
      };
      const runtime = createCloudflareBridgeRuntime({
        baseURL: 'https://bridge.example',
        sandboxId: 'abc',
        fetch: fetchImpl,
      });
      const content =
        encoding === 'stream'
          ? new ReadableStream<Uint8Array>({
            start(controller) {
              const chunk = new Uint8Array(
                new SharedArrayBuffer(bytes.length)
              );
              chunk.set(bytes);
              controller.enqueue(chunk.subarray(0, 2));
              controller.enqueue(chunk.subarray(2));
              controller.close();
            },
          })
          : bytes.toString(encoding);

      await expect(
        runtime.writeFile('hello.txt', content, { encoding })
      ).resolves.toEqual({ ok: true });
    }
  );

  it('preserves caller-provided sandbox ids', async () => {
    const urls: string[] = [];
    const fetchImpl: typeof fetch = async (input) => {
      urls.push(input.toString());
      if (input.toString().endsWith('/exec')) {
        return sseExit();
      }
      throw new Error(`Unexpected URL: ${input.toString()}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'user-123',
      fetch: fetchImpl,
    });

    await expect(runtime.getSandboxId()).resolves.toBe('user-123');
    await runtime.exec('true');

    expect(urls).toEqual(['https://bridge.example/v1/sandbox/user-123/exec']);
  });

  it('retries sandbox creation after a transient create failure', async () => {
    let createCalls = 0;
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/sandbox')) {
        createCalls += 1;
        if (createCalls === 1) {
          return new Response('try again', { status: 503 });
        }
        return Response.json({ id: 'retryid' });
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      fetch: fetchImpl,
    });

    await expect(runtime.getSandboxId()).rejects.toThrow('503');
    await expect(runtime.getSandboxId()).resolves.toBe('retryid');
    expect(createCalls).toBe(2);
  });

  it('fails exec when the SSE stream ends before an exit event', async () => {
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const stdout = Buffer.from('partial').toString('base64');
        return sseResponse(`event: stdout\ndata: ${stdout}\n\n`);
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await expect(runtime.exec('echo partial')).rejects.toThrow(
      'closed before an exit event'
    );
  });

  it('prunes hidden directory trees when includeHidden is disabled', async () => {
    let command = '';
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const body = JSON.parse(bodyText(init?.body)) as { argv?: string[] };
        command = body.argv?.[2] ?? '';
        return sseExit();
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await runtime.listFiles('/workspace', {
      recursive: true,
      includeHidden: false,
    });

    expect(command).toContain('-prune');
    expect(command).toContain('\\( -name \'.*\' -prune \\) -o');
  });

  it('fails bridge listFiles for non-directory targets', async () => {
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const stderr = Buffer.from('not a directory').toString('base64');
        return sseResponse(
          `event: stderr\ndata: ${stderr}\n\nevent: exit\ndata: {"exit_code":20}\n\n`
        );
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await expect(runtime.listFiles('/workspace/file.txt')).rejects.toThrow(
      'not a directory'
    );
  });
});

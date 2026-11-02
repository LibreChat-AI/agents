// src/verification/__tests__/codeApiVerdict.test.ts
import { codeApiVerdictRunner, commandPasses } from '..';
import { createHash, generateKeyPairSync, sign } from 'crypto';
import type { CodeApiVerdictRunnerOptions, VerdictPublicKey } from '..';
import { stableStringify } from '@/tools/eagerEventExecution';

const sha256 = (text: string): string =>
  createHash('sha256').update(text, 'utf8').digest('hex');

/** A stand-in Code API that signs verdicts in the server's token format. */
function fakeCodeApi(options: {
  exitCode?: number;
  trust?: 'paired' | 'local';
  /** Rewrites the response after signing, to simulate tampering in transit. */
  tamper?: (body: {
    result: Record<string, unknown>;
    verdict: Record<string, unknown>;
  }) => void;
  replayNonce?: string;
  resultSignal?: string;
  truncated?: boolean;
  alterRequest?: (request: Record<string, unknown>) => void;
  alterClaims?: (claims: Record<string, unknown>) => void;
}): { fetch: typeof fetch; key: VerdictPublicKey; calls: string[] } {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const key = {
    kid: 'k1',
    x: (publicKey.export({ format: 'jwk' }) as { x: string }).x,
  };
  const calls: string[] = [];
  const fakeFetch = (async (url: string, init?: RequestInit) => {
    calls.push(url);
    if (url.endsWith('/workspace-tools/verdict-keys')) {
      return new Response(JSON.stringify({ keys: [key] }), { status: 200 });
    }
    const request = JSON.parse(String(init?.body)) as {
      workspaceId: string;
      command: string;
      verdict: { verifierId: string; nonce: string };
    };
    const result = {
      protocolVersion: 1,
      operation: 'execute_command',
      workspaceId: request.workspaceId,
      signal: options.resultSignal,
      exitCode: options.exitCode ?? 0,
      timedOut: false,
      stdout: 'ok\n',
      stderr: '',
      truncated: options.truncated ?? false,
    };
    const { verdict: _verdict, ...callerRequest } = request;
    options.alterRequest?.(callerRequest);
    const claims = {
      v: 1,
      typ: 'codeapi.verdict',
      kid: 'k1',
      trust: options.trust ?? 'paired',
      workspace_id: request.workspaceId,
      request_sha256: sha256(stableStringify(callerRequest)),
      dispatched_request_sha256: sha256('server normalized request'),
      workspace_instance_id: null,
      dispatched_workspace_instance_id: null,
      cwd: null,
      worktree: null,
      environment_action: null,
      verifier_id: request.verdict.verifierId,
      nonce: options.replayNonce ?? request.verdict.nonce,
      command_sha256: sha256(request.command),
      exit_code: result.exitCode,
      signal: result.signal ?? null,
      truncated: result.truncated,
      timed_out: false,
      stdout_sha256: sha256(result.stdout),
      stderr_sha256: sha256(result.stderr),
    };
    options.alterClaims?.(claims);
    const payload = Buffer.from(JSON.stringify(claims)).toString('base64url');
    const token = `${payload}.${sign(null, Buffer.from(payload), privateKey).toString('base64url')}`;
    const body = {
      result: { ...result },
      verdict: { token, keyId: 'k1', trust: claims.trust } as Record<
        string,
        unknown
      >,
    };
    options.tamper?.(body);
    return new Response(JSON.stringify(body), { status: 200 });
  }) as unknown as typeof fetch;
  return { fetch: fakeFetch, key, calls };
}

const ctx = {
  runId: 'r',
  requirement: { mark: 'tests.pass', verifier: 'tests_pass' },
  params: {},
  finalText: null,
  signal: new AbortController().signal,
};

const check = (runner: ReturnType<typeof codeApiVerdictRunner>) =>
  commandPasses({ id: 'tests_pass', command: 'npm test', runner }).run(ctx);

describe('codeApiVerdictRunner', () => {
  it.each([
    ['workspaceId', 'other-workspace'],
    ['workspaceInstanceId', 'other-instance'],
    ['cwd', 'other-path'],
    ['worktree', 'other-lane'],
    ['environmentAction', { name: 'build', fingerprint: 'changed' }],
    ['timeoutMs', 1],
    ['maxOutputBytes', 1],
  ])('rejects a signed result for a changed request field: %s', async (field, value) => {
    const api = fakeCodeApi({ alterRequest: (request) => { request[field] = value; } });
    const runner = codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    });
    await expect(check(runner)).rejects.toThrow('request');
  });

  it('fails closed when a server omits request binding', async () => {
    const api = fakeCodeApi({ alterClaims: (claims) => { delete claims.request_sha256; } });
    await expect(check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }))).rejects.toThrow('request');
  });

  it.each(['signal', 'truncated'])('rejects %s changed after signing', async (field) => {
    const api = fakeCodeApi({ tamper: (body) => { body.result[field] = field === 'signal' ? 'SIGTERM' : true; } });
    await expect(check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }))).rejects.toThrow(field);
  });

  it('rejects unknown trust values even when signed by a pinned key', async () => {
    const api = fakeCodeApi({ alterClaims: (claims) => { claims.trust = 'other'; } });
    await expect(check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }))).rejects.toThrow('trust');
  });

  it('does not pass a command with a signed termination signal and exit zero', async () => {
    const api = fakeCodeApi({ resultSignal: 'SIGTERM' });
    const verdict = await check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }));
    expect(verdict.passed).toBe(false);
    expect(verdict.details).toMatchObject({ signal: 'SIGTERM' });
  });

  it('records signed output truncation in the retained artifact', async () => {
    const api = fakeCodeApi({ truncated: true });
    const verdict = await check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }));
    expect(JSON.parse(verdict.artifact!.content)).toMatchObject({ complete: false });
  });

  it('submits the exact configured target and accepts a distinct dispatched digest', async () => {
    const api = fakeCodeApi({});
    const target = {
      workspaceId: 'ws-1', workspaceInstanceId: 'conversation-1', cwd: 'src',
      worktree: 'feature', environmentAction: { fingerprint: 'env-1', name: 'tests' },
      maxOutputBytes: 4096,
    };
    const fetcher = jest.fn(api.fetch);
    const options: CodeApiVerdictRunnerOptions = {
      baseUrl: 'http://api/v1', ...target, keys: [api.key], fetch: fetcher,
    };
    const runner = codeApiVerdictRunner(options);
    options.workspaceId = 'changed-after-construction';
    options.environmentAction!.fingerprint = 'changed-after-construction';
    const verdict = await check(runner);
    expect(verdict.passed).toBe(true);
    expect(JSON.parse(String(fetcher.mock.calls[0][1]!.body))).toMatchObject({
      ...target, environmentAction: { fingerprint: 'env-1', name: 'tests' },
      command: 'npm test', timeoutMs: 300000,
    });
  });

  it.each(['exit_code', 'signal', 'timed_out', 'truncated'])('rejects missing signed execution field: %s', async (field) => {
    const api = fakeCodeApi({ alterClaims: (claims) => { delete claims[field]; } });
    await expect(check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }))).rejects.toThrow('execution claims');
  });

  it.each(['exitCode', 'timedOut', 'truncated', 'stdout', 'stderr', 'workspaceId', 'protocolVersion', 'operation'])('rejects missing response field: %s', async (field) => {
    const api = fakeCodeApi({ tamper: (body) => { delete body.result[field]; } });
    await expect(check(codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    }))).rejects.toThrow('invalid execution result');
  });

  it('isolates cancellation during concurrent key lookups and caches a completed lookup', async () => {
    const api = fakeCodeApi({});
    const cancelled = new AbortController();
    const started = Promise.withResolvers<void>();
    let keyCalls = 0;
    const runner = codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1',
      fetch: async (input, init) => {
        if (String(input).endsWith('/verdict-keys')) {
          keyCalls += 1;
          if (init?.signal === cancelled.signal) {
            started.resolve();
            return new Promise<Response>((_resolve, reject) => {
              init.signal!.addEventListener('abort', () => reject(new Error('key lookup cancelled')), { once: true });
            });
          }
        }
        return api.fetch(input, init);
      },
    });
    const first = runner('npm test', { timeoutMs: 1000, verifierId: 'tests', signal: cancelled.signal });
    const rejected = expect(first).rejects.toThrow('cancelled');
    await started.promise;
    const second = check(runner);
    cancelled.abort();
    await rejected;
    expect((await second).passed).toBe(true);
    expect((await check(runner)).passed).toBe(true);
    expect(keyCalls).toBe(2);
  });

  it('does not dispatch a command for an already aborted caller', async () => {
    const api = fakeCodeApi({});
    const runner = codeApiVerdictRunner({
      baseUrl: 'http://api/v1', workspaceId: 'ws-1', keys: [api.key], fetch: api.fetch,
    });
    await expect(runner('npm test', { timeoutMs: 1000, verifierId: 'tests', signal: AbortSignal.abort() })).rejects.toThrow();
    expect(api.calls).toHaveLength(0);
  });

  it('accepts a signed verdict for its own nonce and keeps the attestation', async () => {
    const api = fakeCodeApi({});
    const runner = codeApiVerdictRunner({
      baseUrl: 'http://api/v1/',
      workspaceId: 'ws-1',
      keys: [api.key],
      fetch: api.fetch,
    });
    const verdict = await check(runner);
    expect(verdict.passed).toBe(true);
    expect(verdict.attestation).toMatchObject({
      issuer: 'http://api/v1',
      keyId: 'k1',
      trust: 'paired',
    });
    expect(api.calls).toEqual(['http://api/v1/workspace-tools/verify']);
  });

  it('reports a failing command as a failed check, still attested', async () => {
    const api = fakeCodeApi({ exitCode: 1 });
    const verdict = await check(
      codeApiVerdictRunner({
        baseUrl: 'http://api/v1',
        workspaceId: 'ws-1',
        keys: [api.key],
        fetch: api.fetch,
      })
    );
    expect(verdict.passed).toBe(false);
    expect(verdict.attestation?.keyId).toBe('k1');
  });

  it('rejects a result altered after signing, a replayed nonce, and an unknown key', async () => {
    const tampered = fakeCodeApi({
      exitCode: 1,
      tamper: (b) => {
        b.result.exitCode = 0;
      },
    });
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [tampered.key],
          fetch: tampered.fetch,
        })
      )
    ).rejects.toThrow('exit code');
    const swapped = fakeCodeApi({
      tamper: (b) => {
        b.result.stdout = 'all good\n';
      },
    });
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [swapped.key],
          fetch: swapped.fetch,
        })
      )
    ).rejects.toThrow('stdout');
    const replay = fakeCodeApi({ replayNonce: 'x'.repeat(32) });
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [replay.key],
          fetch: replay.fetch,
        })
      )
    ).rejects.toThrow('nonce');
    const other = fakeCodeApi({});
    const api = fakeCodeApi({});
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [{ ...other.key, kid: 'k1' }],
          fetch: api.fetch,
        })
      )
    ).rejects.toThrow('does not verify');
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [{ kid: 'k2', x: api.key.x }],
          fetch: api.fetch,
        })
      )
    ).rejects.toThrow('unknown key');
  });

  it('refuses LOCAL_MODE verdicts unless the host accepts them', async () => {
    const api = fakeCodeApi({ trust: 'local' });
    await expect(
      check(
        codeApiVerdictRunner({
          baseUrl: 'http://api/v1',
          workspaceId: 'ws-1',
          keys: [api.key],
          fetch: api.fetch,
        })
      )
    ).rejects.toThrow('LOCAL_MODE');
    const accepted = await check(
      codeApiVerdictRunner({
        baseUrl: 'http://api/v1',
        workspaceId: 'ws-1',
        keys: [api.key],
        fetch: api.fetch,
        acceptLocal: true,
      })
    );
    expect(accepted).toMatchObject({
      passed: true,
      attestation: { trust: 'local' },
    });
  });

  it('fetches the published keys once when none are pinned', async () => {
    const api = fakeCodeApi({});
    const runner = codeApiVerdictRunner({
      baseUrl: 'http://api/v1',
      workspaceId: 'ws-1',
      fetch: api.fetch,
    });
    await check(runner);
    await check(runner);
    expect(api.calls.filter((u) => u.endsWith('/verdict-keys'))).toHaveLength(
      1
    );
  });
});

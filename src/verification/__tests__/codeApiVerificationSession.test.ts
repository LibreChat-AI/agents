import { AIMessage } from '@langchain/core/messages';
import { createHash, generateKeyPairSync, sign } from 'crypto';
import type { StopFinalizeHookInput } from '@/hooks/types';
import type { CodeApiVerifyRequest } from '../codeApiVerdict';
import type { CodeApiVerificationRecovery, CodeApiVerificationSessionOptions } from '..';
import { createCodeApiVerificationSession, createCompletionGate, commandPasses } from '..';
import { codeApiRequestHash } from '../codeApiVerdict';

const sha256 = (text: string): string => createHash('sha256').update(text).digest('hex');

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function server() {
  const { privateKey, publicKey } = generateKeyPairSync('ed25519');
  const key = { kid: 'test-key', x: publicKey.export({ format: 'jwk' }).x! };
  const saved = new Map<string, CodeApiVerifyRequest>();
  const submitted = deferred<string>();
  const cancelled = deferred<void>();
  const calls: string[] = [];
  const settings = {
    state: 'completed', executionState: 'settled', cancelRequested: false,
    durable: 1, proof: true, claimProof: true, dropSubmission: false,
    missing: false, badBinding: false, local: false,
    onPost: (): void => undefined,
    cancelState: 'cancelled', cancelExecution: 'settled',
  };
  const status = (id: string): object => {
    const request = saved.get(id)!;
    const result = {
      protocolVersion: 1, operation: 'execute_command', workspaceId: request.workspaceId,
      exitCode: 0, timedOut: false, truncated: false, stdout: 'checks passed', stderr: '',
      ...(settings.proof ? { processSettlement: 'process_tree_exited' } : {}),
    };
    const claims = {
      v: 1, typ: 'codeapi.verdict', kid: key.kid, trust: settings.local ? 'local' : 'paired',
      workspace_id: request.workspaceId, request_sha256: codeApiRequestHash(request),
      dispatched_request_sha256: sha256('normalized dispatch'),
      verifier_id: request.verdict.verifierId, nonce: request.verdict.nonce,
      command_sha256: sha256(request.command), exit_code: 0, signal: null,
      truncated: false, timed_out: false,
      stdout_sha256: sha256(result.stdout), stderr_sha256: sha256(result.stderr),
      process_settlement: settings.claimProof ? 'process_tree_exited' : null,
    };
    const payload = Buffer.from(JSON.stringify(claims)).toString('base64url');
    const token = `${payload}.${sign(null, Buffer.from(payload), privateKey).toString('base64url')}`;
    return {
      requestId: id, workspaceId: request.workspaceId,
      workspaceInstanceId: 'principal-mapped-instance',
      state: settings.state, executionState: settings.executionState, cancelRequested: settings.cancelRequested,
      verification: {
        requestSha256: settings.badBinding ? 'wrong' : codeApiRequestHash(request),
        verifierId: request.verdict.verifierId, nonce: request.verdict.nonce,
      },
      ...(settings.state === 'completed' ? { result, verdict: { token, keyId: key.kid, trust: claims.trust } } : {}),
    };
  };
  const json = (body: object, code = 200): Response => new Response(JSON.stringify(body), { status: code });
  const doFetch: typeof fetch = async (input, init): Promise<Response> => {
    const url = input instanceof Request ? input.url : String(input);
    const method = init?.method ?? 'GET';
    calls.push(`${method} ${url}`);
    if (url.endsWith('/capabilities')) return json({ durableWorkspaceVerification: settings.durable });
    if (url.endsWith('/verdict-keys')) return json({ keys: [key] });
    if (url.endsWith('/verify/requests')) {
      const request = JSON.parse(String(init?.body)) as CodeApiVerifyRequest;
      const id = new Headers(init?.headers).get('X-LibreChat-Workspace-Request-Id')!;
      saved.set(id, request);
      submitted.resolve(id);
      settings.onPost();
      if (settings.dropSubmission) throw new TypeError('connection lost after submission');
      return json(status(id), 202);
    }
    const id = url.split('/').at(-1)!;
    if (settings.missing || !saved.has(id)) return json({ error: 'not found' }, 404);
    if (method === 'DELETE') {
      settings.state = settings.cancelState;
      settings.executionState = settings.cancelExecution;
      settings.cancelRequested = true;
      cancelled.resolve();
    }
    return json(status(id));
  };
  return { settings, fetch: doFetch, key, saved, calls, submitted, cancelled };
}

function setup(overrides: Partial<CodeApiVerificationSessionOptions> = {}) {
  const api = server();
  const records: CodeApiVerificationRecovery[] = [];
  const options: CodeApiVerificationSessionOptions = {
    baseUrl: 'http://api/v1', workspaceId: 'workspace', workspaceInstanceId: 'caller-instance',
    keys: [api.key], fetch: api.fetch, pollIntervalMs: 1,
    persist: async (record): Promise<void> => { records.push(record); },
    ...overrides,
  };
  return { api, records, options, session: createCodeApiVerificationSession(options) };
}

const runOptions = (signal = new AbortController().signal) => ({ signal, timeoutMs: 1000, verifierId: 'tests' });
const finalize: StopFinalizeHookInput = {
  hook_event_name: 'StopFinalize', runId: 'run', messages: [new AIMessage('done')],
  stopHookActive: false, continuationCount: 0, continuationBudgetRemaining: 3,
  continuationPlanned: false, continuationPrevented: false,
};

describe('durable Code API verification', () => {
  it('persists before submitting, binds mapped targets, and validates signed process evidence', async () => {
    const saved = deferred<void>();
    const entered = deferred<void>();
    const fixture = setup({ persist: async (): Promise<void> => { entered.resolve(); await saved.promise; } });
    const result = fixture.session.runner('npm test', runOptions());
    await entered.promise;
    expect(fixture.api.calls.filter((call) => call.startsWith('POST'))).toHaveLength(0);
    saved.resolve();
    await expect(result).resolves.toMatchObject({ exitCode: 0, processSettlement: 'process_tree_exited' });
    expect(fixture.session.pending()).toEqual([]);
    expect(fixture.api.calls.filter((call) => call.startsWith('POST'))).toHaveLength(1);
  });

  it('does not dispatch when durable recording fails', async () => {
    const { session, api } = setup({ persist: async (): Promise<void> => { throw new Error('journal unavailable'); } });
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('journal unavailable');
    expect(api.saved.size).toBe(0);
    expect(session.pending()).toEqual([]);
  });

  it('does not submit to a server without durable verification', async () => {
    const { session, api, records } = setup();
    api.settings.durable = 0;
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('does not support');
    expect(records).toHaveLength(0);
    expect(api.saved.size).toBe(0);
  });

  it('rejects a settlement field added after signing without leaving a live operation', async () => {
    const { session, api } = setup();
    api.settings.claimProof = false;
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('process settlement');
    expect(session.pending()).toEqual([]);
  });

  it('rejects local trust by default after confirmed termination', async () => {
    const { session, api } = setup();
    api.settings.local = true;
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('LOCAL_MODE');
    expect(session.pending()).toEqual([]);
  });

  it('cancels with a fresh signal and waits for an explicit worker acknowledgement', async () => {
    const { session, api } = setup();
    api.settings.state = 'admitted';
    api.settings.executionState = 'unknown';
    const controller = new AbortController();
    api.settings.onPost = (): void => controller.abort();
    const result = session.runner('npm test', runOptions(controller.signal));
    await expect(result).rejects.toThrow('cancelled');
    expect(api.calls.some((call) => call.startsWith('DELETE'))).toBe(true);
    expect(session.pending()).toEqual([]);
  });

  it('retains ownership after a lost submission response and reconciles without resubmitting', async () => {
    const { session, api } = setup();
    api.settings.dropSubmission = true;
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    await session.reconcile(id, new AbortController().signal);
    await expect(result).resolves.toMatchObject({ exitCode: 0 });
    expect(api.calls.filter((call) => call.startsWith('POST'))).toHaveLength(1);
    expect(session.pending()).toEqual([]);
  });

  it('still observes cancellation after the polling transport has failed', async () => {
    const { session, api } = setup();
    api.settings.dropSubmission = true;
    const controller = new AbortController();
    const result = session.runner('npm test', runOptions(controller.signal));
    await api.submitted.promise;
    await Promise.resolve();
    controller.abort();
    await expect(result).rejects.toThrow('cancelled');
    expect(session.pending()).toEqual([]);
  });

  it('never equates a missing request, expired assignment or terminal status with process settlement', async () => {
    const { session, api } = setup();
    api.settings.dropSubmission = true;
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    api.settings.missing = true;
    await expect(session.reconcile(id, new AbortController().signal)).rejects.toThrow('HTTP 404');
    expect(session.pending()).toHaveLength(1);
    api.settings.missing = false;
    api.settings.state = 'failed';
    api.settings.executionState = 'unknown';
    await expect(session.reconcile(id, new AbortController().signal)).resolves.toMatchObject({ executionState: 'unknown' });
    expect(session.pending()).toHaveLength(1);
    api.settings.executionState = 'settled';
    const failure = expect(result).rejects.toThrow('failed');
    await session.reconcile(id, new AbortController().signal);
    await failure;
    expect(session.pending()).toEqual([]);
  });

  it('requires status binding even for a claimed pre-admission rejection', async () => {
    const { session, api } = setup();
    api.settings.dropSubmission = true;
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    api.settings.state = 'failed';
    api.settings.executionState = 'not_started';
    api.settings.badBinding = true;
    await expect(session.reconcile(id, new AbortController().signal)).rejects.toThrow('unbound');
    expect(session.pending()).toHaveLength(1);
    api.settings.badBinding = false;
    const rejected = expect(result).rejects.toThrow('not_started');
    await session.reconcile(id, new AbortController().signal);
    await rejected;
    expect(session.pending()).toEqual([]);
  });

  it('keeps completion blocked and release forbidden until cancellation actually settles', async () => {
    const { session, api } = setup();
    api.settings.state = 'admitted';
    api.settings.executionState = 'unknown';
    api.settings.cancelState = 'failed';
    api.settings.cancelExecution = 'unknown';
    const gate = createCompletionGate({
      requirements: [{ mark: 'tests.pass', verifier: 'tests' }],
      verifiers: [commandPasses({ id: 'tests', command: 'npm test', runner: session.runner })],
    });
    const controller = new AbortController();
    const evaluation = gate.hook(finalize, controller.signal);
    const id = await api.submitted.promise;
    controller.abort();
    await api.cancelled.promise;
    await evaluation;
    expect(gate.getCompletion('run')).toMatchObject({ verified: false });
    expect(gate.getCompletion('run').pendingOperations).not.toHaveLength(0);
    expect(() => gate.releaseRun('run')).toThrow('active work');
    api.settings.state = 'cancelled';
    api.settings.executionState = 'settled';
    await session.reconcile(id, new AbortController().signal);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(session.pending()).toEqual([]);
    expect(gate.getCompletion('run')).toMatchObject({ verified: false, pendingOperations: [] });
    expect(() => gate.releaseRun('run')).not.toThrow();
  });

  it('recovers the persisted identity without a second POST', async () => {
    const { session, api, records, options } = setup();
    api.settings.dropSubmission = true;
    const original = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    const restored = createCodeApiVerificationSession(options);
    await expect(restored.recover(records[0], new AbortController().signal)).resolves.toMatchObject({ exitCode: 0 });
    await session.reconcile(id, new AbortController().signal);
    await original;
    expect(api.calls.filter((call) => call.startsWith('POST'))).toHaveLength(1);
    expect(restored.pending()).toEqual([]);
  });

  it('refuses recovery under a different target', async () => {
    const { session, api, records, options } = setup();
    api.settings.dropSubmission = true;
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    const changed = createCodeApiVerificationSession({ ...options, cwd: 'elsewhere' });
    await expect(changed.recover(records[0], new AbortController().signal)).rejects.toThrow('configured verification target');
    expect(changed.pending()).toEqual([]);
    await session.reconcile(id, new AbortController().signal);
    await result;
  });

  it('does not let supplied headers replace the persisted request ID', async () => {
    const { session, api, records } = setup({ headers: { 'x-librechat-workspace-request-id': 'wrong-id' } });
    await session.runner('npm test', runOptions());
    expect([...api.saved.keys()]).toEqual([records[0].requestId]);
  });

  it('keeps a completed legacy-worker command unresolved until it has process evidence', async () => {
    const { session, api } = setup();
    api.settings.executionState = 'unknown';
    api.settings.proof = false;
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    await session.reconcile(id, new AbortController().signal);
    expect(session.pending()).toHaveLength(1);
    api.settings.executionState = 'settled';
    api.settings.proof = true;
    await session.reconcile(id, new AbortController().signal);
    await expect(result).resolves.toMatchObject({ processSettlement: 'process_tree_exited' });
  });

  it.each(['caller', 'explicit'])('rejects %s cancellation during verdict-key lookup', async (kind) => {
    const { api, options } = setup();
    const entered = deferred<void>();
    const keys = deferred<Response>();
    const session = createCodeApiVerificationSession({
      ...options, keys: undefined,
      fetch: async (url, init): Promise<Response> => {
        if (String(url).endsWith('/verdict-keys')) { entered.resolve(); return keys.promise; }
        return api.fetch(url, init);
      },
    });
    const controller = new AbortController();
    const result = session.runner('npm test', runOptions(controller.signal));
    const rejected = expect(result).rejects.toThrow(/cancel|abort/i);
    await entered.promise;
    if (kind === 'caller') controller.abort();
    else await session.cancel(session.pending()[0].requestId, new AbortController().signal);
    keys.resolve(new Response(JSON.stringify({ keys: [api.key] })));
    await rejected;
    expect(session.pending()).toEqual([]);
  });

  it('rejects verification within its deadline when key lookup ignores cancellation after confirmed closure', async () => {
    const { api, options } = setup();
    const session = createCodeApiVerificationSession({
      ...options, keys: undefined, requestTimeoutMs: 20,
      fetch: async (url, init): Promise<Response> => String(url).endsWith('/verdict-keys')
        ? new Promise<Response>(() => undefined) : api.fetch(url, init),
    });
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('deadline');
    expect(session.pending()).toEqual([]);
  });

  it('bounds submission despite an uncooperative transport and ignores its late response', async () => {
    const { api, options } = setup();
    const response = deferred<Response>();
    let submission!: Response;
    const session = createCodeApiVerificationSession({
      ...options, requestTimeoutMs: 20,
      fetch: async (url, init): Promise<Response> => {
        const received = await api.fetch(url, init);
        if (String(url).endsWith('/verify/requests')) { submission = received; return response.promise; }
        return received;
      },
    });
    const result = session.runner('npm test', runOptions());
    const id = await api.submitted.promise;
    await new Promise<void>((resolve) => setTimeout(resolve, 35));
    expect(session.pending()).toMatchObject([{ state: 'unresolved', executionState: 'unknown' }]);
    response.resolve(submission);
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(session.pending()).toHaveLength(1);
    await session.reconcile(id, new AbortController().signal);
    await expect(result).resolves.toMatchObject({ exitCode: 0 });
    expect(api.calls.filter((call) => call.startsWith('POST'))).toHaveLength(1);
  });

  it('stops monitoring and requests cancellation when the server stays queued', async () => {
    const { api, options } = setup();
    api.settings.state = 'queued';
    api.settings.executionState = 'unknown';
    api.settings.cancelExecution = 'not_started';
    const session = createCodeApiVerificationSession({ ...options, monitorTimeoutMs: 20 });
    await expect(session.runner('npm test', runOptions())).rejects.toThrow('not_started');
    expect(api.calls.some((call) => call.startsWith('DELETE'))).toBe(true);
    expect(session.pending()).toEqual([]);
  });
});

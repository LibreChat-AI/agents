import { createHash, createPublicKey, randomBytes, verify } from 'crypto';
import type { CommandOutcome, CommandRunner } from './verifiers';
import { stableStringify } from '@/tools/eagerEventExecution';

/** Ed25519 public key as published by `GET /workspace-tools/verdict-keys`. */
export type VerdictPublicKey = { kid: string; x: string };

type VerdictClaims = {
  v: number;
  typ: string;
  kid: string;
  trust: 'paired' | 'local';
  workspace_id: string;
  request_sha256: string;
  dispatched_request_sha256: string;
  verifier_id: string;
  nonce: string;
  command_sha256: string;
  exit_code: number | null;
  signal: string | null;
  truncated: boolean;
  timed_out: boolean;
  stdout_sha256: string;
  stderr_sha256: string;
  process_settlement?: 'process_tree_exited' | null;
};

export type CodeApiVerifyResponse = {
  result?: {
    protocolVersion?: number;
    operation?: string;
    workspaceId?: string;
    exitCode?: number | null;
    signal?: string;
    truncated?: boolean;
    timedOut?: boolean;
    stdout?: string;
    stderr?: string;
    processSettlement?: string;
  };
  verdict?: { token?: string; keyId?: string; trust?: string };
};

export type CodeApiVerifyRequest = {
  protocolVersion: 1;
  operation: 'execute_command';
  workspaceId: string;
  workspaceInstanceId?: string;
  worktree?: string;
  cwd?: string;
  maxOutputBytes?: number;
  environmentAction?: { name: string; fingerprint: string };
  command: string;
  timeoutMs: number;
  verdict: { verifierId: string; nonce: string };
};

export type CodeApiVerdictRunnerOptions = {
  /** Code API base, e.g. `http://localhost:3112/v1`. */
  baseUrl: string;
  /** Attached BYOM workspace the check runs in. */
  workspaceId: string;
  /** The same host-selected execution instance, lane and cwd as the agent's tools. */
  workspaceInstanceId?: string;
  worktree?: string;
  cwd?: string;
  environmentAction?: { name: string; fingerprint: string };
  maxOutputBytes?: number;
  /**
   * Pinned verdict keys. When omitted the runner fetches the server's
   * published keys once, which trusts that server on first use.
   */
  keys?: VerdictPublicKey[];
  /** Accept verdicts from a LOCAL_MODE Code API, where every caller is one principal (default false). */
  acceptLocal?: boolean;
  headers?: Record<string, string> | ((signal: AbortSignal) => Promise<Record<string, string>>);
  fetch?: typeof fetch;
};

type CodeApiVerdictClient = {
  base: string;
  headers: (signal: AbortSignal) => Promise<Record<string, string>>;
  createRequest: (command: string, timeoutMs: number, verifierId: string) => CodeApiVerifyRequest;
  validate: (body: CodeApiVerifyResponse | null, request: CodeApiVerifyRequest, signal: AbortSignal) => Promise<CommandOutcome>;
  fetch: typeof fetch;
};

const sha256 = (text: string): string =>
  createHash('sha256').update(text, 'utf8').digest('hex');

export function codeApiRequestHash(request: CodeApiVerifyRequest): string {
  const { verdict: _verdict, ...execution } = request;
  return sha256(stableStringify(JSON.parse(JSON.stringify(execution))));
}

function decodeVerdict(token: string, keys: VerdictPublicKey[]): VerdictClaims {
  const parts = token.split('.');
  if (parts.length !== 2 || parts.some((p) => !/^[A-Za-z0-9_-]+$/.test(p)))
    throw new Error('malformed verdict');
  const [payload, signature] = parts;
  const claims = JSON.parse(
    Buffer.from(payload, 'base64url').toString('utf8')
  ) as (Omit<VerdictClaims, 'trust' | 'process_settlement'> & {
    trust: string;
    process_settlement?: string | null;
  }) | null;
  if (claims === null || typeof claims !== 'object' || typeof claims.kid !== 'string')
    throw new Error('malformed verdict claims');
  const key = keys.find((k) => k.kid === claims.kid);
  if (key === undefined)
    throw new Error(`verdict signed by unknown key ${claims.kid}`);
  const publicKey = createPublicKey({
    key: { kty: 'OKP', crv: 'Ed25519', x: key.x },
    format: 'jwk',
  });
  if (
    !verify(
      null,
      Buffer.from(payload, 'utf8'),
      publicKey,
      Buffer.from(signature, 'base64url')
    )
  ) {
    throw new Error('verdict signature does not verify');
  }
  if (claims.v !== 1 || claims.typ !== 'codeapi.verdict')
    throw new Error('not a workspace verdict');
  if (claims.trust !== 'paired' && claims.trust !== 'local')
    throw new Error('invalid verdict trust');
  if (
    typeof claims.request_sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(claims.request_sha256) ||
    typeof claims.dispatched_request_sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(claims.dispatched_request_sha256)
  )
    throw new Error('verdict lacks valid request binding; upgrade the Code API');
  if (
    (claims.exit_code !== null && !Number.isSafeInteger(claims.exit_code)) ||
    (claims.signal !== null && typeof claims.signal !== 'string') ||
    typeof claims.timed_out !== 'boolean' ||
    typeof claims.truncated !== 'boolean' ||
    (claims.process_settlement != null && claims.process_settlement !== 'process_tree_exited')
  )
    throw new Error('malformed verdict execution claims');
  return { ...claims, trust: claims.trust, process_settlement: claims.process_settlement };
}

/** Shared request binding and signature validation for both verification transports. */
export function createCodeApiVerdictClient(
  options: CodeApiVerdictRunnerOptions
): CodeApiVerdictClient {
  const doFetch = options.fetch ?? fetch;
  const base = options.baseUrl.replace(/\/+$/, '');
  const acceptLocal = options.acceptLocal === true;
  const configuredHeaders = typeof options.headers === 'function' ? options.headers : { ...options.headers };
  const target = {
    workspaceId: options.workspaceId,
    workspaceInstanceId: options.workspaceInstanceId,
    worktree: options.worktree,
    cwd: options.cwd,
    maxOutputBytes: options.maxOutputBytes,
    environmentAction:
      options.environmentAction === undefined ? undefined : { ...options.environmentAction },
  };
  let keys = options.keys?.map((key) => ({ ...key }));
  const headers = async (signal: AbortSignal): Promise<Record<string, string>> =>
    typeof configuredHeaders === 'function'
      ? configuredHeaders(signal)
      : { ...configuredHeaders };

  const publishedKeys = async (signal: AbortSignal): Promise<VerdictPublicKey[]> => {
    if (keys !== undefined) return keys;
    const requestHeaders = await headers(signal);
    signal.throwIfAborted();
    const response = await doFetch(`${base}/workspace-tools/verdict-keys`, {
      headers: requestHeaders,
      signal,
    });
    if (!response.ok)
      throw new Error(`verdict keys unavailable: HTTP ${response.status}`);
    const body = (await response.json()) as { keys?: (VerdictPublicKey | null)[] } | null;
    if (!Array.isArray(body?.keys) || body.keys.some((key) =>
      key === null || typeof key.kid !== 'string' || typeof key.x !== 'string'
    ))
      throw new Error('Code API returned invalid verdict keys');
    signal.throwIfAborted();
    // Cache only completed lookups; one caller's cancellation cannot reject another's fetch.
    keys = body.keys as VerdictPublicKey[];
    return keys;
  };

  const createRequest = (
    command: string,
    timeoutMs: number,
    verifierId: string
  ): CodeApiVerifyRequest => {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0)
      throw new Error('verification timeoutMs must be a positive integer');
    return {
      protocolVersion: 1,
      operation: 'execute_command',
      ...target,
      command,
      timeoutMs,
      verdict: { verifierId, nonce: randomBytes(24).toString('base64url') },
    };
  };

  const validate = async (
    body: CodeApiVerifyResponse | null,
    request: CodeApiVerifyRequest,
    signal: AbortSignal
  ): Promise<CommandOutcome> => {
    const result = body?.result;
    const token = body?.verdict?.token;
    if (result == null || typeof token !== 'string')
      throw new Error('Code API returned no signed verdict');
    if (
      result.protocolVersion !== 1 || result.operation !== 'execute_command' ||
      result.workspaceId !== target.workspaceId ||
      (result.exitCode !== null && !Number.isSafeInteger(result.exitCode)) ||
      (result.signal !== undefined && typeof result.signal !== 'string') ||
      (result.processSettlement !== undefined && result.processSettlement !== 'process_tree_exited') ||
      typeof result.timedOut !== 'boolean' || typeof result.truncated !== 'boolean' ||
      typeof result.stdout !== 'string' || typeof result.stderr !== 'string'
    )
      throw new Error('Code API returned an invalid execution result');
    const claims = decodeVerdict(token, await publishedKeys(signal));
    signal.throwIfAborted();
    const { stdout, stderr } = result;
    const mismatches = [
      claims.nonce !== request.verdict.nonce && 'nonce',
      claims.verifier_id !== request.verdict.verifierId && 'verifier',
      claims.workspace_id !== target.workspaceId && 'workspace',
      claims.request_sha256 !== codeApiRequestHash(request) && 'request',
      claims.command_sha256 !== sha256(request.command) && 'command',
      claims.exit_code !== result.exitCode && 'exit code',
      claims.signal !== (result.signal ?? null) && 'signal',
      claims.truncated !== result.truncated && 'truncated',
      claims.timed_out !== result.timedOut && 'timeout flag',
      claims.stdout_sha256 !== sha256(stdout) && 'stdout',
      claims.stderr_sha256 !== sha256(stderr) && 'stderr',
      (claims.process_settlement ?? null) !== (result.processSettlement ?? null) && 'process settlement',
    ].filter((m): m is string => typeof m === 'string');
    if (mismatches.length > 0)
      throw new Error(
        `verdict does not match the result: ${mismatches.join(', ')}`
      );
    if (claims.trust === 'local' && !acceptLocal) {
      throw new Error(
        'verdict came from a LOCAL_MODE Code API; pass acceptLocal to accept it'
      );
    }
    return {
      exitCode: claims.exit_code,
      timedOut: claims.timed_out,
      ...(claims.signal === null ? {} : { signal: claims.signal }),
      outputComplete: !claims.truncated,
      ...(claims.process_settlement == null ? {} : { processSettlement: claims.process_settlement }),
      stdout,
      stderr,
      attestation: {
        issuer: base,
        keyId: claims.kid,
        token,
        trust: claims.trust,
      },
    };
  };
  return { base, headers, createRequest, validate, fetch: doFetch };
}

/**
 * Validates a signed result for the exact submitted request. This transport
 * does not recover remote work after a disconnected request or establish a
 * dirty-workspace revision. Use the durable session to retain remote ownership.
 */
export function codeApiVerdictRunner(options: CodeApiVerdictRunnerOptions): CommandRunner {
  const client = createCodeApiVerdictClient(options);
  return async (command, { timeoutMs, signal, verifierId }): Promise<CommandOutcome> => {
    signal.throwIfAborted();
    const request = client.createRequest(command, timeoutMs, verifierId);
    const response = await client.fetch(`${client.base}/workspace-tools/verify`, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', ...(await client.headers(signal)) },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`Code API verify failed: HTTP ${response.status}`);
    return client.validate((await response.json()) as CodeApiVerifyResponse | null, request, signal);
  };
}

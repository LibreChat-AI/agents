import { randomBytes } from 'crypto';
import { setTimeout as delay } from 'timers/promises';
import type {
  CodeApiVerifyRequest,
  CodeApiVerifyResponse,
  CodeApiVerdictRunnerOptions,
} from './codeApiVerdict';
import type { CommandOutcome, CommandRunner } from './verifiers';
import { codeApiRequestHash, createCodeApiVerdictClient } from './codeApiVerdict';

/** Persist before dispatch. Contains the check command but never authentication headers. */
export type CodeApiVerificationRecovery = {
  schemaVersion: 1;
  baseUrl: string;
  requestId: string;
  request: CodeApiVerifyRequest;
};

export type CodeApiVerificationProgress = {
  requestId: string;
  state: 'submitting' | 'queued' | 'admitted' | 'completed' | 'failed' | 'cancelled' | 'unresolved';
  executionState: 'not_started' | 'settled' | 'unknown';
  cancelRequested: boolean;
  error?: string;
};

export type CodeApiVerificationSessionOptions = CodeApiVerdictRunnerOptions & {
  /** Host-owned durable storage, scoped to the run and authenticated principal. */
  persist: (record: CodeApiVerificationRecovery, signal: AbortSignal) => Promise<void>;
  /** Bound each HTTP request, including credential lookup and response decoding. */
  requestTimeoutMs?: number;
  /** How long to reconcile cancellation before leaving an unresolved operation for the host. */
  cancellationTimeoutMs?: number;
  /** Stops active monitoring and requests cancellation after this finite horizon. */
  monitorTimeoutMs?: number;
  pollIntervalMs?: number;
};

export type CodeApiVerificationSession = {
  /** This promise settles only after remote termination or proof that nothing started. */
  runner: CommandRunner;
  pending(): CodeApiVerificationProgress[];
  reconcile(requestId: string, signal: AbortSignal): Promise<CodeApiVerificationProgress>;
  cancel(requestId: string, signal: AbortSignal): Promise<CodeApiVerificationProgress>;
  /** Recover only from host storage. Never resubmits the command. */
  recover(record: CodeApiVerificationRecovery, signal: AbortSignal): Promise<CommandOutcome>;
};

type RemoteStatus = CodeApiVerifyResponse & {
  requestId?: string;
  workspaceId?: string;
  state?: string;
  executionState?: string;
  cancelRequested?: boolean;
  verification?: { requestSha256?: string; verifierId?: string; nonce?: string };
};

type Operation = {
  record: CodeApiVerificationRecovery;
  progress: CodeApiVerificationProgress;
  result: Promise<CommandOutcome>;
  resolve: (outcome: CommandOutcome) => void;
  reject: (error: Error) => void;
  finalizing: boolean;
  dispatched: boolean;
  removeAbort: () => void;
  signal: AbortSignal;
  cancellation?: Promise<void>;
};

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 2_147_483_647)
    throw new Error(`Invalid ${label}`);
  return value;
}

function cancelled(operation: Operation): boolean {
  return operation.progress.cancelRequested || operation.signal.aborted;
}

function supportedRecovery(record: { schemaVersion: number }): boolean {
  return record.schemaVersion === 1;
}

/**
 * Durable verification transport. A lost response, expired assignment or abort
 * leaves the execution promise pending, so CompletionGate retains its unsettled
 * check. The host can reconcile with the saved request ID without replaying it.
 * This supplies process ownership, not a workspace revision or receipt reuse.
 */
export function createCodeApiVerificationSession(
  options: CodeApiVerificationSessionOptions
): CodeApiVerificationSession {
  const client = createCodeApiVerdictClient(options);
  const persist = options.persist;
  const requestTimeout = positiveInteger(options.requestTimeoutMs ?? 10_000, 'requestTimeoutMs');
  const cancellationTimeout = positiveInteger(options.cancellationTimeoutMs ?? 10_000, 'cancellationTimeoutMs');
  const interval = positiveInteger(options.pollIntervalMs ?? 500, 'pollIntervalMs');
  const monitorTimeout = options.monitorTimeoutMs === undefined
    ? undefined : positiveInteger(options.monitorTimeoutMs, 'monitorTimeoutMs');
  const operations = new Map<string, Operation>();

  const bounded = async <T>(
    action: (signal: AbortSignal) => Promise<T>,
    parent?: AbortSignal,
    timeout = requestTimeout
  ): Promise<T> => {
    const deadline = new AbortController();
    const timer = setTimeout(() => deadline.abort(new Error('Code API request deadline exceeded')), timeout);
    const signal = parent === undefined ? deadline.signal : AbortSignal.any([parent, deadline.signal]);
    let abort: (() => void) | undefined;
    try {
      signal.throwIfAborted();
      const aborted = new Promise<never>((_resolve, reject) => {
        abort = (): void => reject(signal.reason);
        signal.addEventListener('abort', abort, { once: true });
      });
      return await Promise.race([action(signal), aborted]);
    } finally {
      clearTimeout(timer);
      if (abort !== undefined) signal.removeEventListener('abort', abort);
    }
  };

  const requestJson = async (
    path: string,
    signal: AbortSignal,
    method = 'GET',
    operation?: Operation
  ): Promise<RemoteStatus> => {
    const record = operation?.record;
    const headers = new Headers(await client.headers(signal));
    if (record !== undefined) {
      headers.set('Content-Type', 'application/json');
      headers.set('X-LibreChat-Workspace-Request-Id', record.requestId);
    }
    signal.throwIfAborted();
    if (operation !== undefined) operation.dispatched = true;
    const response = await client.fetch(`${client.base}/workspace-tools/${path}`, {
      method,
      headers,
      signal,
      ...(record === undefined ? {} : { body: JSON.stringify(record.request) }),
    });
    if (!response.ok) throw new Error(`Code API verification request failed: HTTP ${response.status}`);
    const status = (await response.json()) as RemoteStatus;
    signal.throwIfAborted();
    return status;
  };

  const lookup = (id: string): Operation => {
    const operation = operations.get(id);
    if (operation === undefined) throw new Error('No pending verification with that request ID');
    return operation;
  };
  const rememberFailure = (operation: Operation): void => {
    if (!operations.has(operation.record.requestId) || operation.finalizing) return;
    operation.progress.state = 'unresolved';
    operation.progress.error = 'Remote execution is unresolved; reconcile the saved request ID. Do not replay it.';
  };

  const applyStatus = async (
    operation: Operation,
    status: RemoteStatus | null,
    signal: AbortSignal
  ): Promise<CodeApiVerificationProgress> => {
    signal.throwIfAborted();
    if (operation.finalizing || !operations.has(operation.record.requestId)) return { ...operation.progress };
    const { requestId, request } = operation.record;
    if (
      status === null || typeof status !== 'object' ||
      status.requestId !== requestId || status.workspaceId !== request.workspaceId ||
      status.verification?.requestSha256 !== codeApiRequestHash(request) ||
      status.verification.verifierId !== request.verdict.verifierId ||
      status.verification.nonce !== request.verdict.nonce ||
      typeof status.cancelRequested !== 'boolean' ||
      !['queued', 'admitted', 'completed', 'failed', 'cancelled'].includes(status.state ?? '') ||
      !['not_started', 'settled', 'unknown'].includes(status.executionState ?? '')
    ) throw new Error('Code API returned an unbound verification status');
    const terminal = status.state === 'completed' || status.state === 'failed' || status.state === 'cancelled';
    if (
      (!terminal && status.executionState !== 'unknown') ||
      (status.state === 'completed' && status.executionState === 'not_started') ||
      (status.state === 'completed' && status.executionState === 'settled' &&
        status.result?.processSettlement !== 'process_tree_exited')
    ) throw new Error('Code API returned inconsistent process settlement');
    operation.progress = {
      requestId,
      state: status.state as CodeApiVerificationProgress['state'],
      executionState: status.executionState as CodeApiVerificationProgress['executionState'],
      cancelRequested: operation.progress.cancelRequested || status.cancelRequested,
    };
    if (!terminal || status.executionState === 'unknown') return { ...operation.progress };
    operation.finalizing = true;
    try {
      // Authenticated status owns process lifetime; the signature separately owns check admission.
      if (operation.progress.cancelRequested || status.state !== 'completed')
        throw new Error(`Remote verification ${status.state}; execution ${status.executionState}`);
      const outcome = await bounded((child) => client.validate(status, request, child), signal);
      if (cancelled(operation))
        throw new Error('Remote verification cancelled after execution settled');
      operation.resolve(outcome);
    } catch (error) {
      if (error instanceof Error) operation.reject(error);
      else operation.reject(new Error(cancelled(operation)
        ? 'Remote verification cancelled after execution settled'
        : 'Remote verification evidence is invalid'));
    } finally {
      operation.removeAbort();
      operations.delete(requestId);
    }
    return { ...operation.progress };
  };

  const reconcile = async (operation: Operation, signal: AbortSignal, cancel = false): Promise<CodeApiVerificationProgress> => {
    if (cancel) operation.progress.cancelRequested = true;
    try {
      return await bounded(async (child) => applyStatus(operation, await requestJson(
        `requests/${operation.record.requestId}`, child, cancel ? 'DELETE' : 'GET'
      ), child), signal);
    } catch (error) {
      rememberFailure(operation);
      throw error;
    }
  };

  const drainCancellation = async (operation: Operation): Promise<void> => {
    operation.progress.cancelRequested = true;
    try {
      await bounded(async (signal) => {
        await reconcile(operation, signal, true);
        while (operations.has(operation.record.requestId)) {
          await delay(interval, undefined, { signal });
          await reconcile(operation, signal);
        }
      }, undefined, cancellationTimeout);
    } catch {
      rememberFailure(operation);
    }
  };

  const cancelAndDrain = (operation: Operation): Promise<void> => {
    operation.cancellation ??= drainCancellation(operation);
    return operation.cancellation;
  };

  const watch = async (operation: Operation, parent: AbortSignal, submit: boolean): Promise<void> => {
    const deadline = new AbortController();
    const timer = setTimeout(() => deadline.abort(), monitorTimeout ?? Math.min(
      operation.record.request.timeoutMs + 300_000 + cancellationTimeout, 2_147_483_647
    ));
    timer.unref();
    const signal = AbortSignal.any([parent, deadline.signal]);
    try {
      if (submit) {
        await bounded(async (child) => applyStatus(operation, await requestJson(
          'verify/requests', child, 'POST', operation
        ), child), signal);
      } else await reconcile(operation, signal);
      while (operations.has(operation.record.requestId)) {
        if (['completed', 'failed', 'cancelled'].includes(operation.progress.state)) return;
        await delay(interval, undefined, { signal });
        await reconcile(operation, signal);
      }
    } catch (error) {
      if (!operation.dispatched) {
        operation.removeAbort();
        operations.delete(operation.record.requestId);
        operation.reject(error instanceof Error ? error : new Error('Verification was not dispatched'));
        return;
      }
      rememberFailure(operation);
      if (signal.aborted && !operation.finalizing && operations.has(operation.record.requestId)) await cancelAndDrain(operation);
    } finally {
      clearTimeout(timer);
    }
  };

  const register = (record: CodeApiVerificationRecovery, signal: AbortSignal, recovering = false): Operation => {
    if (operations.has(record.requestId)) throw new Error('Verification request ID already has an owner');
    let resolve!: Operation['resolve'];
    let reject!: Operation['reject'];
    const result = new Promise<CommandOutcome>((done, fail) => { resolve = done; reject = fail; });
    const operation: Operation = {
      record,
      progress: { requestId: record.requestId, state: 'submitting', executionState: 'unknown', cancelRequested: false },
      result, resolve, reject, finalizing: false, dispatched: recovering, removeAbort: (): void => undefined, signal,
    };
    const abort = (): void => {
      if (!operation.dispatched || operation.finalizing) return;
      void cancelAndDrain(operation);
    };
    signal.addEventListener('abort', abort, { once: true });
    operation.removeAbort = (): void => signal.removeEventListener('abort', abort);
    operations.set(record.requestId, operation);
    return operation;
  };

  return {
    runner: async (command, { timeoutMs, signal, verifierId }): Promise<CommandOutcome> => {
      signal.throwIfAborted();
      await bounded(async (child) => {
        const headers = await client.headers(child);
        child.throwIfAborted();
        const response = await client.fetch(`${client.base}/workspace-tools/capabilities`, {
          headers, signal: child,
        });
        if (!response.ok) throw new Error(`Code API capabilities unavailable: HTTP ${response.status}`);
        const capabilities = (await response.json()) as { durableWorkspaceVerification?: number } | null;
        if (capabilities?.durableWorkspaceVerification !== 1)
          throw new Error('Code API does not support durable verification');
      }, signal);
      const record: CodeApiVerificationRecovery = {
        schemaVersion: 1,
        baseUrl: client.base,
        requestId: randomBytes(24).toString('base64url'),
        request: client.createRequest(command, timeoutMs, verifierId),
      };
      await persist(structuredClone(record), signal);
      signal.throwIfAborted();
      const operation = register(record, signal);
      void watch(operation, signal, true);
      return operation.result;
    },
    pending: (): CodeApiVerificationProgress[] => [...operations.values()].map((operation) => ({ ...operation.progress })),
    reconcile: (id, signal): Promise<CodeApiVerificationProgress> => reconcile(lookup(id), signal),
    cancel: (id, signal): Promise<CodeApiVerificationProgress> => reconcile(lookup(id), signal, true),
    recover: async (saved, signal): Promise<CommandOutcome> => {
      signal.throwIfAborted();
      const record = structuredClone(saved);
      const expected = client.createRequest(record.request.command, record.request.timeoutMs, record.request.verdict.verifierId);
      if (
        !supportedRecovery(record) || record.baseUrl !== client.base ||
        !/^[A-Za-z0-9_-]{16,128}$/.test(record.requestId) ||
        !/^[A-Za-z0-9_-]{16,128}$/.test(record.request.verdict.nonce) ||
        codeApiRequestHash(record.request) !== codeApiRequestHash(expected)
      ) throw new Error('Recovery record does not match the configured verification target');
      const operation = register(record, signal, true);
      void watch(operation, signal, false);
      return operation.result;
    },
  };
}

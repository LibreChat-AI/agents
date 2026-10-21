/**
 * Node-only `@librechat/agents/decisions/node` entry: credential helpers that
 * need `node:child_process`. The `./decisions` entry stays platform-neutral.
 */
import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import type { Readable } from 'node:stream';
import type { DecisionFailure, DecisionTokenMinter } from './types';
import { DecisionError } from './types';
import { redact } from './redact';

/** Fifty minutes: under the ClickHouse gateway's hourly token. */
const DEFAULT_TTL_MS = 50 * 60_000;
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_KILL_GRACE_MS = 2_000;
const MAX_OUTPUT_BYTES = 1024 * 1024;
const STDERR_LIMIT = 200;
/** Off Windows a shell reports a command killed by signal N as exit status 128 + N; Linux numbers signals up to 64. */
const SIGNAL_EXIT_MIN = 129;
const SIGNAL_EXIT_MAX = 128 + 64;
const KILL_SIGNAL = 'SIGTERM';
const FORCE_SIGNAL = 'SIGKILL';
const GROUP_KILL = process.platform !== 'win32';

export const COMMAND_TOKEN_PROVIDER = 'command';

export interface CommandTokenOptions {
  /**
   * How long a minted token is reused without running the command again,
   * counted from when the command finished; `0` mints on every call. Leave
   * headroom under the issuer's lifetime for the command's own runtime.
   */
  ttlMs?: number;
  /** Bounds one run of the command. */
  timeoutMs?: number;
  /**
   * How long a killed run gets to exit on `SIGTERM` before it is sent
   * `SIGKILL`, so a command that traps or ignores `SIGTERM` cannot outlive
   * its run; `0` escalates at once. Defaults to two seconds.
   */
  killGraceMs?: number;
}

interface SharedRun {
  token: Promise<string>;
  /** Callers still awaiting `token`; the run is killed when the last one aborts. */
  waiters: number;
  abandon: AbortController;
}

function abandonedError(): DecisionError {
  return new DecisionError('aborted', 'token command was abandoned', {
    provider: COMMAND_TOKEN_PROVIDER,
  });
}

function validMs(value: number, allowZero: boolean): boolean {
  return (
    Number.isSafeInteger(value) &&
    (allowZero ? value >= 0 : value > 0) &&
    value <= 24 * 3_600_000
  );
}

function lastLine(text: string): string {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '');
  return lines.length > 0 ? lines[lines.length - 1] : '';
}

/** Why a run ended without a token. */
type RunEnd =
  | { kind: 'exit'; code: number }
  | { kind: 'signal'; signal: string }
  | { kind: 'shellSignal'; code: number }
  | { kind: 'timeout' | 'overflow' | 'spawn' | 'abandoned' };

/**
 * Only a run that finished and exited non-zero is a refusal. A run killed at
 * `timeoutMs` is a `timeout`; a spawn error, an output overflow, or a signal
 * from outside is `network`, including one that killed a command the shell
 * forked, which the surviving shell reports as exit status 129–192. None of
 * these says anything about the credential, so none may trip a breaker.
 */
function classify(end: RunEnd): { failure: DecisionFailure; reason: string } {
  if (end.kind === 'exit') {
    return {
      failure: 'unauthorized',
      reason: `token command exited with code ${end.code}`,
    };
  }
  if (end.kind === 'timeout') {
    return {
      failure: 'timeout',
      reason: `token command timed out (${KILL_SIGNAL})`,
    };
  }
  if (end.kind === 'shellSignal') {
    return {
      failure: 'network',
      reason: `token command exited with code ${end.code} (killed by signal ${end.code - 128})`,
    };
  }
  if (end.kind === 'signal') {
    return {
      failure: 'network',
      reason: `token command was killed (${end.signal})`,
    };
  }
  if (end.kind === 'abandoned') {
    return { failure: 'aborted', reason: 'token command was abandoned' };
  }
  return { failure: 'network', reason: 'token command failed' };
}

/**
 * Off Windows the shell leads its own process group, so a kill reaches the
 * command it forked (dash does not `exec` it) and anything that command
 * started, not only the shell. On Windows it reaches only the shell.
 */
function killRun(child: ChildProcess, signal: NodeJS.Signals): void {
  if (child.pid === undefined) {
    return;
  }
  try {
    if (GROUP_KILL) {
      process.kill(-child.pid, signal);
    } else {
      child.kill(signal);
    }
  } catch {
    // Already gone.
  }
}

/** Whether anything a kill would reach is still running: off Windows, any member of the run's process group. */
function runAlive(child: ChildProcess): boolean {
  if (child.pid === undefined) {
    return false;
  }
  if (!GROUP_KILL) {
    return child.exitCode === null && child.signalCode === null;
  }
  try {
    process.kill(-child.pid, 0);
    return true;
  } catch (error) {
    return !(
      error instanceof Error &&
      'code' in error &&
      error.code === 'ESRCH'
    );
  }
}

/** Collects a stream up to the output limit; `onOverflow` fires once past it. */
function collect(
  stream: Readable | null,
  onOverflow: () => void
): () => string {
  const chunks: Buffer[] = [];
  let bytes = 0;
  stream?.on('data', (chunk: Buffer) => {
    if (bytes > MAX_OUTPUT_BYTES) {
      return;
    }
    bytes += chunk.length;
    chunks.push(chunk);
    if (bytes > MAX_OUTPUT_BYTES) {
      onOverflow();
    }
  });
  stream?.on('error', () => {});
  return () => Buffer.concat(chunks).toString('utf8');
}

/**
 * The command text and every word of it whatever its length, so neither a
 * shell that echoes the line nor a program that quotes an argument back can
 * leak an inline secret; a short word may garble the excerpt instead.
 */
function commandSecrets(command: string): string[] {
  return [command, ...command.split(/[\s'"`=]+/)].filter(
    (word) => word.trim() !== ''
  );
}

/** How a run that closed with `code` and `signal` ended; off Windows, an exit status in the shell's signal range is a kill, not a refusal. */
function closedEnd(code: number | null, signal: string | null): RunEnd {
  if (code === null) {
    return { kind: 'signal', signal: signal ?? 'unknown' };
  }
  if (GROUP_KILL && code >= SIGNAL_EXIT_MIN && code <= SIGNAL_EXIT_MAX) {
    return { kind: 'shellSignal', code };
  }
  return { kind: 'exit', code };
}

/** Names how the command failed; never its text, stdout, or `error.message`, which echoes both. */
function describeFailure(
  reason: string,
  stderr: string,
  secrets: ReadonlyArray<string | null>
): string {
  const detail = redact(stderr, secrets).replace(/\s+/g, ' ').trim();
  return detail === '' ? reason : `${reason}: ${detail.slice(0, STDERR_LIMIT)}`;
}

/**
 * A credential minted by a shell command that prints a short-lived bearer
 * token (for example `dataplanectl chai auth token`); the last non-empty line
 * of stdout is the token, and the command reads no stdin. The token is reused
 * for `ttlMs` (default 50 minutes) after the run that minted it finishes,
 * concurrent calls share one run of the command, and a `refresh` (the
 * transport asks after a 401) runs it again unless the token it `refused` has
 * already been replaced, in which case the newer token (or the run minting
 * it) is shared instead.
 *
 * A caller whose `signal` aborts rejects at once with `aborted`; the run is
 * killed when every caller sharing it has aborted, and the next call starts a
 * fresh one. A run that exits non-zero or prints nothing rejects with an
 * `unauthorized` `DecisionError`; a run killed at `timeoutMs` rejects with
 * `timeout`, and a spawn failure, output overflow, or kill from outside with
 * `network`, so a slow command or one that cannot start never trips a
 * breaker. Off Windows an exit status of 129–192 is how a shell reports a
 * command it forked being killed by a signal (137 for `SIGKILL`), so it is a
 * kill, not a refusal; a pipeline such as `cmd | tail -1` hides a killed
 * `cmd` behind a clean exit with no output, which reads as `unauthorized`,
 * so keep the token command last in a pipeline. A kill sends
 * `SIGTERM`, then `SIGKILL` once `killGraceMs` passes with the run still
 * alive, so a killed run overlaps the next one for at most that grace. Off
 * Windows a kill reaches the command's whole process group; on Windows it
 * ends only the shell, not the command the shell launched.
 *
 * The message carries the exit status and a redacted stderr excerpt, never
 * a line of stdout, the command or any word of it (split at whitespace,
 * quotes, and `=`), or a minted token, whatever their length, including a
 * refused token whose refresh failed.
 */
export function commandToken(
  command: string,
  options: CommandTokenOptions = {}
): DecisionTokenMinter {
  const ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const killGraceMs = options.killGraceMs ?? DEFAULT_KILL_GRACE_MS;
  if (typeof command !== 'string' || command.trim() === '') {
    throw new DecisionError('bad_request', 'token command is empty', {
      provider: COMMAND_TOKEN_PROVIDER,
    });
  }
  if (
    !validMs(ttlMs, true) ||
    !validMs(timeoutMs, false) ||
    !validMs(killGraceMs, true)
  ) {
    throw new DecisionError('bad_request', 'invalid token command timing', {
      provider: COMMAND_TOKEN_PROVIDER,
    });
  }

  /** The last minted token; kept after a refusal so a failed refresh still redacts it. */
  let cached: string | null = null;
  let expiresAt = 0;
  let pending: SharedRun | null = null;
  const echoed = commandSecrets(command);

  const run = (abandoned: AbortSignal): Promise<string> =>
    new Promise<string>((resolve, reject) => {
      let child: ChildProcess;
      try {
        child = spawn(command, {
          shell: true,
          detached: GROUP_KILL,
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'pipe'],
        });
      } catch {
        /** Node's synchronous spawn errors quote the command, so none of the message survives. */
        reject(
          new DecisionError('network', 'token command failed', {
            provider: COMMAND_TOKEN_PROVIDER,
          })
        );
        return;
      }
      let settled = false;
      /** Armed by the first kill; outlives the shell, since a forked command may ignore the `SIGTERM` that ended it. */
      let escalation: NodeJS.Timeout | undefined;
      const stop = (end: RunEnd): void => {
        if (escalation === undefined) {
          killRun(child, KILL_SIGNAL);
          escalation = setTimeout(
            () => killRun(child, FORCE_SIGNAL),
            killGraceMs
          );
        }
        child.stdout?.destroy();
        child.stderr?.destroy();
        fail(end);
      };
      const stdout = collect(child.stdout, () => stop({ kind: 'overflow' }));
      const stderr = collect(child.stderr, () => stop({ kind: 'overflow' }));
      const onAbandon = (): void => stop({ kind: 'abandoned' });
      const timer = setTimeout(() => stop({ kind: 'timeout' }), timeoutMs);
      abandoned.addEventListener('abort', onAbandon, { once: true });

      const finish = (): boolean => {
        if (settled) {
          return false;
        }
        settled = true;
        clearTimeout(timer);
        abandoned.removeEventListener('abort', onAbandon);
        return true;
      };
      function fail(end: RunEnd): void {
        if (!finish()) {
          return;
        }
        /**
         * Any stdout line may be the token, since a successful run accepts
         * one of any length, and the cached token is a known credential; all
         * are listed whatever their length, at the cost of a garbled excerpt.
         */
        const secrets = [...echoed, ...stdout().split(/\r?\n/), cached];
        const { failure, reason } = classify(end);
        reject(
          new DecisionError(
            failure,
            describeFailure(reason, stderr(), secrets),
            { provider: COMMAND_TOKEN_PROVIDER }
          )
        );
      }

      child.once('error', () => fail({ kind: 'spawn' }));
      child.once('exit', () => {
        if (escalation !== undefined && !runAlive(child)) {
          clearTimeout(escalation);
        }
      });
      child.once('close', (code: number | null, signal: string | null) => {
        if (code !== 0) {
          fail(closedEnd(code, signal));
          return;
        }
        if (!finish()) {
          return;
        }
        const token = lastLine(stdout());
        if (token === '') {
          reject(
            new DecisionError(
              'unauthorized',
              'token command printed no token',
              {
                provider: COMMAND_TOKEN_PROVIDER,
              }
            )
          );
          return;
        }
        if (!abandoned.aborted) {
          cached = token;
          /** From completion, so a slow command cannot hand out a token that is already expired. */
          expiresAt = Date.now() + ttlMs;
        }
        resolve(token);
      });
    });

  const start = (): SharedRun => {
    const abandon = new AbortController();
    const shared: SharedRun = {
      token: run(abandon.signal),
      waiters: 0,
      abandon,
    };
    const settle = (): void => {
      if (pending === shared) {
        pending = null;
      }
    };
    shared.token.then(settle, settle);
    return shared;
  };

  /** One caller's view of a shared run; the run is killed only when its last caller leaves. */
  const join = (shared: SharedRun, signal?: AbortSignal): Promise<string> =>
    new Promise<string>((resolve, reject) => {
      shared.waiters += 1;
      const onAbort = (): void => {
        shared.waiters -= 1;
        if (shared.waiters === 0) {
          if (pending === shared) {
            pending = null;
          }
          shared.abandon.abort();
        }
        reject(abandonedError());
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      shared.token.then(
        (token) => {
          signal?.removeEventListener('abort', onAbort);
          resolve(token);
        },
        (error: unknown) => {
          signal?.removeEventListener('abort', onAbort);
          reject(error);
        }
      );
    });

  return ({ refresh, refused, signal }): Promise<string> => {
    if (signal?.aborted === true) {
      return Promise.reject(abandonedError());
    }
    /** A refusal of an older token says nothing about the one cached since. */
    if (refresh && (refused === undefined || refused === cached)) {
      /** The cached token was refused; never hand it out again. */
      expiresAt = 0;
    }
    if (pending === null && cached !== null && Date.now() < expiresAt) {
      return Promise.resolve(cached);
    }
    pending ??= start();
    return join(pending, signal);
  };
}

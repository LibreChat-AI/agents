/**
 * Node-only `@librechat/agents/decisions/node` entry: credential helpers that
 * need `node:child_process`. The `./decisions` entry stays platform-neutral.
 */
import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import type { Readable } from 'node:stream';
import type { DecisionFailure, DecisionTokenMinter } from './types';
import type { DecisionBreaker } from './breaker';
import { DecisionError } from './types';
import { redactExcerpt } from './redact';

/** Fifty minutes: under the ClickHouse gateway's hourly token. */
const DEFAULT_TTL_MS = 50 * 60_000;
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_KILL_GRACE_MS = 2_000;
const MAX_OUTPUT_BYTES = 1024 * 1024;
const STDERR_LIMIT = 200;
/** Off Windows a shell reports a command killed by signal N as exit status 128 + N; Linux numbers signals up to 64. */
const SIGNAL_EXIT_MIN = 129;
const SIGNAL_EXIT_MAX = 128 + 64;
/** How long a run whose shell has exited waits for a process it left holding stdout or stderr to let go. */
const DRAIN_MS = 100;
/** How often a killed run checks whether its process group has emptied, so the `SIGKILL` escalation stops waiting. */
const REAP_POLL_MS = 25;
const MAX_EXIT_CODE = 255;
/** How many recent tokens, cached or not, a failed run's stderr is redacted against. */
const KNOWN_TOKENS = 8;
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
  /**
   * The exit statuses that mean the credential was refused. A run that exits
   * with any other non-zero status is a `network` failure, which the
   * transport retries and which never trips a breaker, so a token CLI that
   * exits 7 when its auth server is unreachable cannot open the breaker until
   * a manual reset. By default every non-zero status outside 129–192 is a
   * refusal.
   */
  refusalExitCodes?: ReadonlyArray<number>;
  /**
   * The breaker this credential's model is fused with. Neither a run that
   * started before `breaker.reset()` nor a token it minted, cached or still in
   * flight, serves callers that arrive after it, so the operator's fix is
   * tested by a fresh run instead of failing on stale evidence and reopening
   * the breaker.
   */
  breaker?: Pick<DecisionBreaker, 'generation'>;
}

interface SharedRun {
  token: Promise<string>;
  /** Callers still awaiting `token`; the run is killed when the last one aborts. */
  waiters: number;
  abandon: AbortController;
  /** The breaker generation the run started under; `undefined` without a breaker. */
  generation: number | undefined;
  /** Set once a breaker reset stops the run serving new callers; its token is then not cached. */
  detached: boolean;
}

/** Runs that may still have a live process group; the host's `exit` kills each, since the command's own process group does not receive the host's signals. */
const live = new Set<ChildProcess>();

function killLive(): void {
  for (const child of live) {
    killRun(child, FORCE_SIGNAL);
  }
}

function track(child: ChildProcess): void {
  if (child.pid === undefined) {
    return;
  }
  if (live.size === 0) {
    process.on('exit', killLive);
  }
  live.add(child);
}

function untrack(child: ChildProcess): void {
  if (live.delete(child) && live.size === 0) {
    process.off('exit', killLive);
  }
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
 * Only a run that finished and exited non-zero, with a status in `refusals`
 * when that list is given, is a refusal. A run killed at
 * `timeoutMs` is a `timeout`; a spawn error, an output overflow, or a signal
 * from outside is `network`, including one that killed a command the shell
 * forked, which the surviving shell reports as exit status 129–192. None of
 * these says anything about the credential, so none may trip a breaker.
 */
function classify(
  end: RunEnd,
  refusals: ReadonlySet<number> | undefined
): { failure: DecisionFailure; reason: string } {
  if (end.kind === 'exit') {
    return {
      failure:
        refusals === undefined || refusals.has(end.code)
          ? 'unauthorized'
          : 'network',
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
    /** Not `instanceof Error`: a host that runs this module in a `vm` context (Jest does) gets the error from another realm. */
    return !(
      typeof error === 'object' &&
      error !== null &&
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
 * leak an inline secret; a short word may garble the excerpt instead. Words
 * split at quotes, which the shell removes; at braces and commas, which
 * bash's brace expansion splits a word at; and at `;`, `&`, `|`, `(`, `)`,
 * `<`, and `>`, which end an unquoted word without whitespace in a POSIX
 * shell and in cmd.exe alike. Splitting quoted text there too is harmless,
 * since every piece is still redacted.
 */
function commandSecrets(command: string): string[] {
  return [command, ...command.split(/[\s'"`=,{};&|()<>]+/)].filter(
    (word) => word.trim() !== ''
  );
}

/** A double-quoted, single-quoted, or unquoted stretch of a POSIX command; a stray quote reads as unquoted. */
const SHELL_SEGMENT = /"[^"]*"|'[^']*'|[^"']+|["']/g;
const SINGLE_QUOTED = /^'[^']*'$/;

/**
 * Whether the shell may hand the program an argument the command text does
 * not contain, which no word of the text can redact: off Windows, a backslash
 * escape or a `$` or backtick expansion outside single quotes; on Windows,
 * cmd.exe's `%` expansion and `^` escape, or a `\"` a program's argument
 * parser decodes.
 */
function rewritesWords(command: string): boolean {
  if (!GROUP_KILL) {
    return /[%^]|\\"/.test(command);
  }
  for (const [segment] of command.matchAll(SHELL_SEGMENT)) {
    if (!SINGLE_QUOTED.test(segment) && /[\\$`]/.test(segment)) {
      return true;
    }
  }
  return false;
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

/**
 * Names how the command failed; never its text, stdout, or `error.message`,
 * which echoes both. Stderr that is not `scrubbable` is withheld.
 */
function describeFailure(
  reason: string,
  stderr: string,
  secrets: ReadonlyArray<string | null>,
  scrubbable: boolean
): string {
  if (stderr.trim() === '') {
    return reason;
  }
  if (!scrubbable) {
    return `${reason} (stderr withheld: the command uses shell escapes or expansion)`;
  }
  const detail = redactExcerpt(stderr, secrets, STDERR_LIMIT);
  return detail === '' ? reason : `${reason}: ${detail}`;
}

function validExitCodes(codes: ReadonlyArray<number> | undefined): boolean {
  return (
    codes === undefined ||
    (Array.isArray(codes) &&
      codes.every(
        (code) =>
          Number.isSafeInteger(code) && code > 0 && code <= MAX_EXIT_CODE
      ))
  );
}

/**
 * A credential minted by a shell command that prints a short-lived bearer
 * token (for example `dataplanectl chai auth token`); the last non-empty line
 * of stdout is the token, and the command reads no stdin. The token is reused
 * for `ttlMs` (default 50 minutes) after the run that minted it finishes,
 * concurrent calls share one run of the command, and a `refresh` (the
 * transport asks after a 401) runs it again unless the token it `refused` has
 * already been replaced, in which case the newer token (or the run minting
 * it) is shared instead. Given the model's `breaker`, a call made after
 * `breaker.reset()` never joins a run that started before it.
 *
 * A run ends when its shell exits: a process the command leaves behind
 * holding stdout or stderr gets 100 ms to let go, then the run settles from
 * what it printed and that process is left running, unkilled.
 *
 * A caller whose `signal` aborts rejects at once with `aborted`; the run is
 * killed when every caller sharing it has aborted, and the next call starts a
 * fresh one. A run that exits non-zero or prints nothing rejects with an
 * `unauthorized` `DecisionError`, unless `refusalExitCodes` is given and does
 * not list its status, which makes it `network`; a run killed at `timeoutMs`
 * rejects with `timeout`, and a spawn failure, output overflow, or kill from
 * outside with `network`, so a slow command or one that cannot start never
 * trips a breaker. Off Windows an exit status of 129–192 is how a shell
 * reports a command it forked being killed by a signal (137 for `SIGKILL`),
 * so it is a kill, not a refusal; a pipeline such as `cmd | tail -1` hides a
 * killed `cmd` behind a clean exit with no output, which reads as
 * `unauthorized`, so keep the token command last in a pipeline. A kill sends
 * `SIGTERM`, then `SIGKILL` once `killGraceMs` passes with the run still
 * alive, so a killed run overlaps the next one for at most that grace; the
 * escalation is dropped as soon as nothing it would reach is left. Off
 * Windows a kill reaches the command's whole process group; on Windows it
 * ends only the shell, not the command the shell launched.
 *
 * Off Windows the command runs in its own session, with no controlling
 * terminal, so a signal sent to the host's process group (a terminal's
 * Ctrl-C) does not reach it. When the host process exits, each run still
 * going is sent `SIGKILL`; a host killed by a signal it does not handle never
 * reaches its `exit` event, so it leaves such a run going.
 *
 * The message carries the exit status and a redacted stderr excerpt, never
 * a line of stdout, the command or any word of it (split at whitespace,
 * quotes, `=`, braces, commas, and the shell operators `;`, `&`, `|`, `(`,
 * `)`, `<`, and `>`), or any of the last eight tokens it minted
 * or was told were refused, cached or not, whatever their length, including a
 * refused token whose refresh failed. The excerpt reads only the start of
 * stderr, so formatting it stays cheap whatever the command printed. When the
 * shell may pass the program an argument the command text does not contain
 * (off Windows, a backslash escape or a `$` or backtick expansion outside
 * single quotes; on Windows, `%`, `^`, or `\"`), stderr is withheld instead.
 */
export function commandToken(
  command: string,
  options: CommandTokenOptions = {}
): DecisionTokenMinter {
  const ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const killGraceMs = options.killGraceMs ?? DEFAULT_KILL_GRACE_MS;
  const { breaker } = options;
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
  if (!validExitCodes(options.refusalExitCodes)) {
    throw new DecisionError(
      'bad_request',
      'invalid token command refusal exit codes',
      { provider: COMMAND_TOKEN_PROVIDER }
    );
  }
  const refusals =
    options.refusalExitCodes === undefined
      ? undefined
      : new Set(options.refusalExitCodes);

  /** The last minted token; kept after a refusal so a failed refresh still redacts it. */
  let cached: string | null = null;
  let expiresAt = 0;
  /** The breaker generation the run that minted `cached` started under; a reset since then bypasses it. */
  let cachedGeneration: number | undefined;
  let pending: SharedRun | null = null;
  const echoed = commandSecrets(command);
  /** Stderr may quote an argument the shell decoded or expanded, which no word of the text matches. */
  const scrubbable = !rewritesWords(command);
  /**
   * The last tokens minted or refused, cached or not: a run detached by a
   * breaker reset hands out a token it never caches, and the command's own
   * store may still hold it or an older one when a later run fails.
   */
  const known: string[] = [];
  const remember = (token: string): void => {
    if (known.includes(token)) {
      return;
    }
    known.push(token);
    if (known.length > KNOWN_TOKENS) {
      known.shift();
    }
  };

  /**
   * `feedsCache` says whether a token this run mints may still be cached when
   * it lands; `generation` is the breaker generation the run started under.
   */
  const run = (
    abandoned: AbortSignal,
    generation: number | undefined,
    feedsCache: () => boolean
  ): Promise<string> =>
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
      track(child);
      let settled = false;
      let killed = false;
      /** Armed by the first kill; outlives the shell, since a forked command may ignore the `SIGTERM` that ended it. */
      let escalation: NodeJS.Timeout | undefined;
      /** Polls the killed group, so the escalation holds the host only while something it would reach is alive. */
      let reaping: NodeJS.Timeout | undefined;
      /** After the shell exits, bounds the wait for a process it left holding stdout or stderr. */
      let draining: NodeJS.Timeout | undefined;
      const release = (): void => {
        clearTimeout(escalation);
        clearInterval(reaping);
        untrack(child);
      };
      const reap = (): void => {
        if (!runAlive(child)) {
          release();
        }
      };
      const stop = (end: RunEnd): void => {
        if (!killed) {
          killed = true;
          killRun(child, KILL_SIGNAL);
          escalation = setTimeout(() => {
            killRun(child, FORCE_SIGNAL);
            release();
          }, killGraceMs);
          reaping = setInterval(reap, REAP_POLL_MS);
          reaping.unref();
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
        clearTimeout(draining);
        abandoned.removeEventListener('abort', onAbandon);
        return true;
      };
      function fail(end: RunEnd): void {
        if (!finish()) {
          return;
        }
        /**
         * Any stdout line may be the token, since a successful run accepts
         * one of any length, and the cached and recent tokens are known
         * credentials; all are listed whatever their length, at the cost of a
         * garbled excerpt.
         */
        const secrets = [
          ...echoed,
          ...stdout().split(/\r?\n/),
          cached,
          ...known,
        ];
        const { failure, reason } = classify(end, refusals);
        reject(
          new DecisionError(
            failure,
            describeFailure(reason, stderr(), secrets, scrubbable),
            { provider: COMMAND_TOKEN_PROVIDER }
          )
        );
      }
      /** Settles a run that ended on its own, from what it printed. */
      const ended = (code: number | null, signal: string | null): void => {
        if (!killed) {
          untrack(child);
        }
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
        remember(token);
        if (!abandoned.aborted && feedsCache()) {
          cached = token;
          cachedGeneration = generation;
          /** From completion, so a slow command cannot hand out a token that is already expired. */
          expiresAt = Date.now() + ttlMs;
        }
        resolve(token);
      };

      child.once('error', () => fail({ kind: 'spawn' }));
      child.once('exit', (code: number | null, signal: string | null) => {
        if (killed) {
          reap();
          return;
        }
        draining = setTimeout(() => {
          child.stdout?.destroy();
          child.stderr?.destroy();
          ended(code, signal);
        }, DRAIN_MS);
      });
      child.once('close', (code: number | null, signal: string | null) => {
        if (killed) {
          reap();
          return;
        }
        ended(code, signal);
      });
    });

  const start = (): SharedRun => {
    const abandon = new AbortController();
    const generation = breaker?.generation;
    const shared: SharedRun = {
      token: run(abandon.signal, generation, () => !shared.detached),
      waiters: 0,
      abandon,
      generation,
      detached: false,
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
    if (refused !== undefined) {
      remember(refused);
    }
    /** A run from before a breaker reset carries the credential state the reset was meant to clear; its callers keep it, new ones start fresh. */
    if (pending !== null && pending.generation !== breaker?.generation) {
      pending.detached = true;
      pending = null;
    }
    /** A refusal of an older token says nothing about the one cached since. */
    if (refresh && (refused === undefined || refused === cached)) {
      /** The cached token was refused; never hand it out again. */
      expiresAt = 0;
    }
    /** A token minted before a reset is stale evidence too, even when its run settled before anyone arrived to detach it. */
    if (
      pending === null &&
      cached !== null &&
      cachedGeneration === breaker?.generation &&
      Date.now() < expiresAt
    ) {
      return Promise.resolve(cached);
    }
    pending ??= start();
    return join(pending, signal);
  };
}

/**
 * Node-only `@librechat/agents/decisions/node` entry: credential helpers that
 * need `node:child_process`. The `./decisions` entry stays platform-neutral.
 */
import { exec } from 'node:child_process';
import type { ExecException } from 'node:child_process';
import type { DecisionFailure, DecisionTokenMinter } from './types';
import { DecisionError } from './types';
import { redact } from './redact';

/** Fifty minutes: under the ClickHouse gateway's hourly token. */
const DEFAULT_TTL_MS = 50 * 60_000;
const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_OUTPUT_BYTES = 1024 * 1024;
const STDERR_LIMIT = 200;
const MIN_SECRET_LENGTH = 8;

export const COMMAND_TOKEN_PROVIDER = 'command';

export interface CommandTokenOptions {
  /** How long a minted token is reused without running the command again; `0` mints on every call. */
  ttlMs?: number;
  /** Bounds one run of the command. */
  timeoutMs?: number;
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

/**
 * Only a run that finished and exited non-zero is a refusal. A run killed at
 * `timeoutMs` is a `timeout`; a spawn error, an output overflow, or a signal
 * from outside is `network`. None of these says anything about the
 * credential, so none may trip a breaker. `code` is typed as a number, but
 * Node sets a string code on spawn and output-overflow errors.
 */
function classify(error: ExecException): {
  failure: DecisionFailure;
  reason: string;
} {
  const code: unknown = error.code;
  if (typeof code === 'number') {
    return {
      failure: 'unauthorized',
      reason: `token command exited with code ${code}`,
    };
  }
  if (code == null && error.killed === true && error.signal != null) {
    return {
      failure: 'timeout',
      reason: `token command timed out (${error.signal})`,
    };
  }
  if (code == null && error.signal != null) {
    return {
      failure: 'network',
      reason: `token command was killed (${error.signal})`,
    };
  }
  return { failure: 'network', reason: 'token command failed' };
}

/** The command text and each of its words long enough to be a credential, so a shell that echoes the line cannot leak it. */
function commandSecrets(command: string): string[] {
  return [command, ...command.split(/[\s'"`=]+/)].filter(
    (word) => word.trim().length >= MIN_SECRET_LENGTH
  );
}

/** Names how the command failed; never its text, stdout, or `error.message`, which echoes both. */
function describeFailure(
  reason: string,
  stderr: string,
  secrets: string[]
): string {
  const detail = redact(stderr, secrets).replace(/\s+/g, ' ').trim();
  return detail === '' ? reason : `${reason}: ${detail.slice(0, STDERR_LIMIT)}`;
}

/**
 * A credential minted by a shell command that prints a short-lived bearer
 * token (for example `dataplanectl chai auth token`); the last non-empty line
 * of stdout is the token. The token is reused for `ttlMs` (default 50
 * minutes), concurrent calls share one run of the command, and a `refresh`
 * (the transport asks after a 401) runs it again. A run that exits non-zero or
 * prints nothing rejects with an `unauthorized` `DecisionError`; a run killed
 * at `timeoutMs` rejects with `timeout`, and a spawn failure or output
 * overflow with `network`, so a slow command or one that cannot start never
 * trips a breaker.
 * The message carries the exit status and a redacted stderr excerpt, never
 * stdout, the command, or a token.
 */
export function commandToken(
  command: string,
  options: CommandTokenOptions = {}
): DecisionTokenMinter {
  const ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (typeof command !== 'string' || command.trim() === '') {
    throw new DecisionError('bad_request', 'token command is empty', {
      provider: COMMAND_TOKEN_PROVIDER,
    });
  }
  if (!validMs(ttlMs, true) || !validMs(timeoutMs, false)) {
    throw new DecisionError('bad_request', 'invalid token command timing', {
      provider: COMMAND_TOKEN_PROVIDER,
    });
  }

  let cached: string | null = null;
  let expiresAt = 0;
  let pending: Promise<string> | null = null;
  const echoed = commandSecrets(command);

  const run = (): Promise<string> =>
    new Promise<string>((resolve, reject) => {
      const started = Date.now();
      exec(
        command,
        {
          encoding: 'utf8',
          timeout: timeoutMs,
          maxBuffer: MAX_OUTPUT_BYTES,
          windowsHide: true,
        },
        (error, stdout, stderr) => {
          const printed = String(stdout);
          if (error) {
            /** Partial stdout may hold a token; shorter lines are not credentials and would garble the excerpt. */
            const secrets = [
              ...echoed,
              cached ?? '',
              ...printed.split(/\r?\n/),
            ].filter((line) => line.trim().length >= MIN_SECRET_LENGTH);
            const { failure, reason } = classify(error);
            reject(
              new DecisionError(
                failure,
                describeFailure(reason, String(stderr), secrets),
                { provider: COMMAND_TOKEN_PROVIDER }
              )
            );
            return;
          }
          const token = lastLine(printed);
          if (token === '') {
            reject(
              new DecisionError(
                'unauthorized',
                'token command printed no token',
                { provider: COMMAND_TOKEN_PROVIDER }
              )
            );
            return;
          }
          cached = token;
          expiresAt = started + ttlMs;
          resolve(token);
        }
      );
    });

  return ({ refresh }): Promise<string> => {
    if (refresh) {
      /** The cached token was refused; never hand it out again. */
      cached = null;
    }
    if (pending !== null) {
      return pending;
    }
    if (cached !== null && Date.now() < expiresAt) {
      return Promise.resolve(cached);
    }
    const minting = run().finally(() => {
      if (pending === minting) {
        pending = null;
      }
    });
    pending = minting;
    return minting;
  };
}

import { readFile, stat } from 'fs/promises';
import { isAbsolute, relative, resolve, sep } from 'path';
import type {
  VerdictAttestation,
  VerifierDefinition,
  VerifierVerdict,
} from './types';
import type * as t from '@/types';
import {
  executeLocalBash,
  resolveWorkspacePathSafe,
} from '@/tools/local/LocalExecutionEngine';

/** Output tail shown to the model when a check fails. */
const FEEDBACK_CHARS = 2000;

export type CommandOutcome = {
  exitCode: number | null;
  timedOut: boolean;
  signal?: string;
  stdout: string;
  stderr: string;
  /** Omit when the backend cannot establish whether output was capped. */
  outputComplete?: boolean;
  /** Explicit backend acknowledgement, never inferred from HTTP completion. */
  processSettlement?: 'process_tree_exited';
  fullOutputPath?: string;
  attestation?: VerdictAttestation;
};

/** Runs a host-defined check command where the run's work lives. */
export type CommandRunner = (
  command: string,
  options: { timeoutMs: number; signal: AbortSignal; verifierId: string }
) => Promise<CommandOutcome>;

/** Rejects absolute paths and lexical or current symlink escapes before artifact I/O. */
async function insideRoot(root: string, path: unknown): Promise<string | null> {
  if (typeof path !== 'string' || path === '' || isAbsolute(path)) return null;
  const full = resolve(root, path);
  const rel = relative(root, full);
  const escapes = rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel);
  if (rel === '' || escapes) return null;
  try {
    return await resolveWorkspacePathSafe(full, { cwd: root }, 'read');
  } catch {
    return null;
  }
}

/** Passes when `params.path` is a regular file inside `root`. The blank check: "the artifact arrived". */
export function artifactExists(root: string): VerifierDefinition {
  return {
    id: 'artifact_exists',
    timeoutMs: 5_000,
    run: async ({ params }): Promise<VerifierVerdict> => {
      const full = await insideRoot(root, params.path);
      if (full === null)
        return {
          passed: false,
          details: { error: 'path must be relative to the workspace' },
        };
      const info = await stat(full).catch(() => null);
      return info?.isFile() === true
        ? { passed: true, details: { path: params.path } }
        : {
          passed: false,
          details: { path: params.path },
          feedback: `Required file ${String(params.path)} does not exist.`,
        };
    },
  };
}

/** Passes when `params.path` is a file inside `root` that parses as JSON. */
export function jsonFileValid(root: string): VerifierDefinition {
  return {
    id: 'json_file_valid',
    timeoutMs: 5_000,
    run: async ({ params }): Promise<VerifierVerdict> => {
      const full = await insideRoot(root, params.path);
      if (full === null)
        return {
          passed: false,
          details: { error: 'path must be relative to the workspace' },
        };
      try {
        JSON.parse(await readFile(full, 'utf8'));
        return { passed: true, details: { path: params.path } };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          passed: false,
          details: { path: params.path, error: message },
          feedback: `${String(params.path)} is not valid JSON: ${message}`,
        };
      }
    },
  };
}

/**
 * Passes when the host-defined `command` exits 0 within the time limit. The
 * command is part of the definition, never taken from params or the model.
 */
export function commandPasses(options: {
  id: string;
  command: string;
  runner: CommandRunner;
  timeoutMs?: number;
}): VerifierDefinition {
  const { id, command, runner, timeoutMs = 300_000 } = options;
  return {
    id,
    timeoutMs,
    run: async ({ signal, params }): Promise<VerifierVerdict> => {
      const limit =
        typeof params.timeout_ms === 'number'
          ? Math.min(params.timeout_ms, timeoutMs)
          : timeoutMs;
      const outcome = await runner(command, {
        timeoutMs: limit,
        signal,
        verifierId: id,
      });
      const passed =
        outcome.exitCode === 0 && outcome.signal === undefined &&
        !outcome.timedOut && !signal.aborted;
      const output = `${outcome.stdout}${outcome.stderr === '' ? '' : `\n${outcome.stderr}`}`;
      const termination = outcome.signal === undefined
        ? `exited with ${String(outcome.exitCode)}`
        : `terminated by ${outcome.signal}`;
      return {
        passed,
        artifact: {
          label: 'verification.output',
          content: JSON.stringify({
            stdout: outcome.stdout,
            stderr: outcome.stderr,
            complete: outcome.outputComplete ?? null,
            fullOutputPath: outcome.fullOutputPath,
          }),
        },
        details: {
          command,
          exitCode: outcome.exitCode,
          timedOut: outcome.timedOut,
          ...(outcome.signal === undefined ? {} : { signal: outcome.signal }),
        },
        ...(passed
          ? {}
          : {
            feedback: `\`${command}\` ${outcome.timedOut ? 'timed out' : termination}. Output tail:\n${output.slice(-FEEDBACK_CHARS)}`,
          }),
        ...(outcome.attestation !== undefined
          ? { attestation: outcome.attestation }
          : {}),
      };
    },
  };
}

/** Runs check commands through the local execution engine, with its validation and sandbox. */
export function localCommandRunner(
  config: t.LocalExecutionConfig
): CommandRunner {
  return async (command, { timeoutMs, signal }) => {
    const result = await executeLocalBash(command, { ...config, timeoutMs }, signal);
    return {
      exitCode: signal.aborted ? null : result.exitCode,
      timedOut: result.timedOut,
      stdout: result.stdout,
      stderr: result.stderr,
      fullOutputPath: result.fullOutputPath,
    };
  };
}

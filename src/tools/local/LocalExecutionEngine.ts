import { tmpdir } from 'os';
import { spawn } from 'child_process';
import { StringDecoder } from 'string_decoder';
import { createWriteStream } from 'fs';
import { createHash, randomUUID } from 'crypto';
import { isAbsolute, relative, resolve, sep } from 'path';
import { mkdir, realpath, rm, writeFile } from 'fs/promises';
import type { ChildProcess } from 'child_process';
import type { WorkspaceFS } from './workspaceFS';
import type * as t from '@/types';
import { runBashAstChecks, bashAstFindingsToErrors } from './bashAst';
import { nodeWorkspaceFS } from './workspaceFS';

const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_MAX_OUTPUT_CHARS = 200000;
/**
 * Hard cap on total stdout+stderr bytes a child process can stream
 * before we kill its process tree. Independent from `maxOutputChars`
 * (which only affects what the *model* sees) — this is the OOM
 * backstop. Configurable via `local.maxSpawnedBytes`.
 */
const DEFAULT_MAX_SPAWNED_BYTES = 50 * 1024 * 1024;
const DEFAULT_LOCAL_SESSION_ID = 'local';
const DEFAULT_SHELL = process.platform === 'win32' ? 'bash.exe' : 'bash';
const MAX_COMMAND_AVAILABILITY_ENVIRONMENTS = 16;
const NEGATIVE_COMMAND_AVAILABILITY_TTL_MS = 5000;

/** Produces a stable, non-plaintext key for environment-sensitive probes. */
export function commandAvailabilityEnvCacheKey(
  env: NodeJS.ProcessEnv | undefined
): string {
  if (env == null) {
    return '';
  }
  const sorted = Object.keys(env)
    .sort()
    .map((key): [string, string | null] => [key, env[key] ?? null]);
  return createHash('sha256').update(JSON.stringify(sorted)).digest('hex');
}

/** Adds a probe entry while bounding retained environment variants. */
export function setCommandAvailabilityCacheEntry<T>(
  cache: Map<string, T>,
  key: string,
  value: T
): void {
  if (!cache.has(key) && cache.size >= MAX_COMMAND_AVAILABILITY_ENVIRONMENTS) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }
  cache.set(key, value);
}

// `(?:--\s+)?` before each destructive-target alternation: GNU/BSD
// utilities accept `--` as an end-of-options marker, so `rm -rf -- /`
// is identical in effect to `rm -rf /` but pre-fix it slipped past
// the guard because the regex required the path to follow option
// flags directly. Codex P1 #20.
// `DESTRUCTIVE_TARGET` is the canonical "protected location" pattern:
// matches `/`, `~`, `$HOME`, `${HOME}`, `.`, each optionally followed
// by a trailing-slash and/or wildcard glob suffix. The suffix matrix:
//   ''     — `$HOME`              (round 14)
//   '/'    — `$HOME/`             (round 14, Codex P1 [37])
//   '*'    — `$HOME*`             (round 15, Codex P1 [42])
//   '/*'   — `$HOME/*`            (round 15, Codex P1 [42])
//   '.*'   — `$HOME.*`            (round 17, Codex P1 [47])
//   '/.*'  — `$HOME/.*`           (round 17, Codex P1 [47]) — the
//            dot-glob form deletes all dotfiles under the protected
//            root, just as destructive as `/*` but the prior matrix
//            missed it.
// Suffix expression: `(?:\/?\.?\*|\/)?` — one of:
//   `\/?\.?\*` → `*`, `.*`, `/*`, `/.*`
//   `\/`       → `/`
//   (empty)    → bare base
const DESTRUCTIVE_TARGET = '(?:\\/|~|\\$\\{?HOME\\}?|\\.)(?:\\/?\\.?\\*|\\/)?';

const dangerousCommandPatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    `\\brm\\s+(?:-[^\\s]*[rf][^\\s]*\\s+|-[^\\s]*[r][^\\s]*\\s+-[^\\s]*[f][^\\s]*\\s+)(?:--\\s+)?${DESTRUCTIVE_TARGET}\\s*(?:$|[;&|])`
  ),
  /\b(?:mkfs|mkswap|fdisk|parted|diskutil)\b/,
  /\bdd\s+[^;&|]*\bof=\/dev\//,
  new RegExp(
    `\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?${DESTRUCTIVE_TARGET}(?:$|\\s|[;&|])`
  ),
  new RegExp(
    `\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?${DESTRUCTIVE_TARGET}(?:$|\\s|[;&|])`
  ),
  /:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:/,
];

/**
 * Companion patterns that look for destructive targets *inside*
 * matching quote pairs. These are checked against the ORIGINAL
 * command (not the post-quote-strip `normalized` form), because
 * `stripQuotedContent` blanks the contents of quoted spans —
 * which would otherwise let `rm -rf "/"` and friends slip past
 * `dangerousCommandPatterns`.
 *
 * Kept as a separate list so we don't pay false-positive cost on
 * benign uses like `echo "rm -rf /"` (the print case): each pattern
 * here REQUIRES a quote *around the destructive path argument*, not
 * just a quote *somewhere* in the command. `echo "rm -rf /"` has
 * `/` outside of any quote-pair-around-the-path (the quotes wrap
 * the whole `rm -rf /` text), so it doesn't match here either.
 */
// Quoted variant uses the same DESTRUCTIVE_TARGET (which accepts an
// optional trailing slash) so `rm -rf "$HOME/"` and `rm -rf "~/"`
// don't slip past. Codex P1 #37.
const quotedDestructivePatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    `\\brm\\s+(?:-[^\\s]*[rf][^\\s]*\\s+){1,3}(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
  new RegExp(
    `\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
  new RegExp(
    `\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
];

/**
 * Catches destructive operations smuggled inside a nested shell or
 * `eval` call, e.g. `bash -lc "rm -rf $HOME"` — the outer command
 * looks benign (`bash -lc "..."`) and the destructive `rm` lives
 * inside the quoted payload that `stripQuotedContent` blanks out.
 * Comprehensive review (manual finding C) flagged this as a real
 * bypass of the otherwise-correct quote-strip-then-match approach.
 *
 * Run against the ORIGINAL command (quotes intact) so the inside of
 * the nested-shell payload is visible. Conservative: matches only
 * the same operation set as `dangerousCommandPatterns` (rm -rf,
 * chmod -R 777, chown -R) when they appear inside a `<shell> -[l]?c
 * "..."` or `eval "..."` payload.
 */
const NESTED_SHELL_PREFIX = '(?:(?:ba|z|da|k)?sh|eval)\\s+(?:-l?c\\s+)?';
const nestedShellDestructivePatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\brm\\s+-[^\\s"\']*[rf][^\\s"\']*\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
];

const mutatingCommandPattern =
  /\b(?:rm|mv|cp|touch|mkdir|rmdir|ln|truncate|tee|sed\s+-i|perl\s+-pi|python(?:3)?\s+-c|node\s+-e|npm\s+(?:install|ci|update|publish)|pnpm\s+(?:install|update|publish)|yarn\s+(?:install|add|publish)|git\s+(?:add|commit|checkout|switch|reset|clean|rebase|merge|push|pull|stash|tag|branch)|chmod|chown)\b|(?:^|[^<])>\s*[^&]|\bcat\s+[^|;&]*>\s*/;

export type SpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  /**
   * True when the process was force-killed because total streamed bytes
   * exceeded `maxSpawnedBytes`. Distinct from `timedOut`. Without this
   * flag, callers (`bash_tool`, `execute_code`, etc.) would see a
   * SIGKILL'd process with `exitCode: null` and treat it as success
   * (Codex P1 — runaway commands like `yes` or noisy builds silently
   * looked successful even though their output was truncated).
   */
  overflowKilled?: boolean;
  /**
   * Signal name (e.g. `'SIGKILL'`, `'SIGSEGV'`) when the process was
   * terminated by a signal. Distinct from the overflow-kill path:
   * this captures `kill -9 $$` from inside the script, native crashes,
   * OS OOM killer, etc. Without this, signal-killed processes
   * reported `exitCode: null` and looked like clean runs (Codex P2 —
   * generalization of the overflow-kill fix). When present, the
   * exitCode field is also synthesized to `128 + signum` per the
   * POSIX convention so non-null-exit-code consumers see a failure.
   */
  signal?: string;
  /** Path to captured stdout/stderr; a terminated process may have incomplete output. */
  fullOutputPath?: string;
  outputTruncated?: boolean;
  outputRecoveryError?: string;
  /** A custom spawn backend may not share the host's output filesystem. */
  outputOnHost?: boolean;
};

/**
 * POSIX convention: `128 + signum` when a process is killed by a
 * signal. Maps the common signals; unknown ones default to 1 so the
 * caller still sees a non-zero (failed) exit. Only used when Node's
 * `close` event reports `exitCode === null` (true signal kill).
 */
const SIGNAL_TO_EXIT_CODE: Record<string, number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGQUIT: 131,
  SIGILL: 132,
  SIGTRAP: 133,
  SIGABRT: 134,
  SIGBUS: 135,
  SIGFPE: 136,
  SIGKILL: 137,
  SIGUSR1: 138,
  SIGSEGV: 139,
  SIGUSR2: 140,
  SIGPIPE: 141,
  SIGALRM: 142,
  SIGTERM: 143,
};
function exitCodeForSignal(signal: string | null): number {
  if (signal == null) return 1;
  return SIGNAL_TO_EXIT_CODE[signal] ?? 1;
}

type RuntimeCommand = {
  command: string;
  args: string[];
  fileName: string;
  source?: string;
};

type SandboxManagerType = {
  checkDependencies(): { errors: string[] };
  initialize(config: BuiltSandboxRuntimeConfig): Promise<void>;
  reset(): Promise<void>;
  wrapWithSandbox(command: string): Promise<string>;
};

type SandboxRuntimeModule = {
  getDefaultWritePaths(): string[];
  SandboxManager: SandboxManagerType;
};

let sandboxConfigKey: string | undefined;
let sandboxInitialized = false;
let sandboxRuntimePromise: Promise<SandboxRuntimeModule> | undefined;

export type BashValidationResult = {
  valid: boolean;
  errors: string[];
  warnings: string[];
};

function isToolExecutionConfig(
  config: t.ToolExecutionConfig | t.LocalExecutionConfig
): config is t.ToolExecutionConfig {
  return 'engine' in config || 'local' in config;
}

export function resolveLocalExecutionConfig(
  config?: t.ToolExecutionConfig | t.LocalExecutionConfig
): t.LocalExecutionConfig {
  if (config != null && isToolExecutionConfig(config)) {
    return config.local ?? {};
  }
  return config ?? {};
}

export function getLocalCwd(config?: t.LocalExecutionConfig): string {
  return resolve(config?.workspace?.root ?? config?.cwd ?? process.cwd());
}

/**
 * Resolves the effective workspace boundary: a list of absolute roots
 * that file operations are allowed to touch. The first entry is always
 * the canonical root (`getLocalCwd`); subsequent entries come from
 * `workspace.additionalRoots` when provided.
 *
 * Returns plain absolute paths — callers symlink-resolve when they
 * need realpath equality (see `resolveWorkspacePathSafe`).
 */
export function getWorkspaceRoots(config?: t.LocalExecutionConfig): string[] {
  const root = getLocalCwd(config);
  const extras = config?.workspace?.additionalRoots ?? [];
  if (extras.length === 0) return [root];
  const seen = new Set<string>([root]);
  const out: string[] = [root];
  for (const extra of extras) {
    // Relative `additionalRoots` entries are anchored to the
    // workspace root (so monorepo configs like
    // `additionalRoots: ['../shared']` resolve to a sibling of
    // `root` rather than to `process.cwd()/../shared`, which would
    // mean something completely different on a server with a
    // different cwd).
    const abs = isAbsolute(extra) ? resolve(extra) : resolve(root, extra);
    if (!seen.has(abs)) {
      seen.add(abs);
      out.push(abs);
    }
  }
  return out;
}

/** Node-host execution world used when no backend override is configured. */
export const nodeExecutionWorld: t.ExecutionWorld = Object.freeze({
  spawn: spawn as t.LocalSpawn,
  fs: nodeWorkspaceFS,
  sandboxed: false,
});

/** Resolves filesystem and subprocess capabilities as one execution world. */
export function getExecutionWorld(
  config?: t.LocalExecutionConfig
): t.ExecutionWorld {
  if (config?.exec == null && config?.spawn == null) {
    return nodeExecutionWorld;
  }
  return {
    spawn: config.exec?.spawn ?? config.spawn ?? nodeExecutionWorld.spawn,
    fs: config.exec?.fs ?? nodeExecutionWorld.fs,
    sandboxed: config.exec?.sandboxed ?? false,
  };
}

/**
 * Pluggable spawn resolver. Honours `local.exec.spawn` first, falls
 * back to the legacy top-level `local.spawn`, then to Node's
 * `child_process.spawn`. Centralised so engine swapping is one knob.
 */
export function getSpawn(config?: t.LocalExecutionConfig): t.LocalSpawn {
  return getExecutionWorld(config).spawn;
}

/**
 * Pluggable filesystem resolver. Honours `local.exec.fs`, falls back
 * to the Node-host implementation. A future remote engine supplies
 * its own implementation here and inherits every file-touching tool.
 */
export function getWorkspaceFS(config?: t.LocalExecutionConfig): WorkspaceFS {
  return getExecutionWorld(config).fs;
}

/**
 * Resolves the workspace boundary for *write* operations. Honours
 * `workspace.allowWriteOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`, which the path-safety
 * helpers interpret as "skip the write clamp".
 */
export function getWriteRoots(
  config: t.LocalExecutionConfig = {}
): string[] | null {
  // Granular flag wins over the legacy one when explicitly set
  // (true OR false) — otherwise a host tightening access during
  // migration (`allowOutsideWorkspace: true, workspace.
  // allowWriteOutside: false`) would still get the loose behavior
  // because the legacy flag short-circuited the OR. Codex P1 #36.
  const granular = config.workspace?.allowWriteOutside;
  if (granular === true) return null;
  if (granular === false) return getWorkspaceRoots(config);
  if (config.allowOutsideWorkspace === true) return null;
  return getWorkspaceRoots(config);
}

/**
 * Resolves the workspace boundary for *read* operations. Honours
 * `workspace.allowReadOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`.
 */
export function getReadRoots(
  config: t.LocalExecutionConfig = {}
): string[] | null {
  // Same precedence as getWriteRoots: granular flag is authoritative
  // when set, legacy flag is the fallback. Codex P1 #36.
  const granular = config.workspace?.allowReadOutside;
  if (granular === true) return null;
  if (granular === false) return getWorkspaceRoots(config);
  if (config.allowOutsideWorkspace === true) return null;
  return getWorkspaceRoots(config);
}

export function getLocalSessionId(config?: t.LocalExecutionConfig): string {
  const cwd = getLocalCwd(config);
  const digest = createHash('sha1').update(cwd).digest('hex').slice(0, 12);
  return `${DEFAULT_LOCAL_SESSION_ID}:${digest}`;
}

const missingSandboxRuntimeMessage = [
  'Local sandbox is enabled, but @anthropic-ai/sandbox-runtime is not installed.',
  'Install it with `npm install @anthropic-ai/sandbox-runtime`, or disable local sandboxing with `local.sandbox.enabled: false`.',
].join(' ');
const sandboxRuntimePackage = '@anthropic-ai/sandbox-runtime';

/** Lazy-loads the ESM-only sandbox runtime only when sandboxing is enabled. */
function loadSandboxRuntime(): Promise<SandboxRuntimeModule> {
  sandboxRuntimePromise ??= import(
    sandboxRuntimePackage
  ) as Promise<SandboxRuntimeModule>;
  return sandboxRuntimePromise;
}

function shouldUseLocalSandbox(config: t.LocalExecutionConfig): boolean {
  return config.sandbox?.enabled === true;
}

let sandboxOffWarned = false;

function maybeWarnSandboxOff(config: t.LocalExecutionConfig): void {
  if (
    sandboxOffWarned ||
    shouldUseLocalSandbox(config) ||
    getExecutionWorld(config).sandboxed
  ) {
    return;
  }
  sandboxOffWarned = true;
  // eslint-disable-next-line no-console
  console.warn(
    '[@librechat/agents] Local execution engine is running without ' +
      '@anthropic-ai/sandbox-runtime wrapping. The agent has full access to ' +
      'the host filesystem and network. Set toolExecution.local.sandbox.enabled ' +
      '= true to opt into process sandboxing.'
  );
}

/**
 * Test-only reset hook for the sandbox-off warning latch.
 *
 * @internal Not part of the public SDK surface.
 */
export function _resetLocalEngineWarningsForTests(): void {
  sandboxOffWarned = false;
}

export function truncateLocalOutput(
  value: string,
  maxChars = DEFAULT_MAX_OUTPUT_CHARS
): string {
  if (value.length <= maxChars) {
    return value;
  }
  const head = Math.floor(maxChars * 0.6);
  const tail = maxChars - head;
  const omitted = value.length - maxChars;
  return `${value.slice(0, head)}\n\n[... ${omitted} characters truncated ...]\n\n${value.slice(
    value.length - tail
  )}`;
}

function stripQuotedContent(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;

  for (let i = 0; i < command.length; i++) {
    const char = command[i];

    if (escaped) {
      escaped = false;
      output += ' ';
      continue;
    }

    if (char === '\\') {
      escaped = true;
      output += ' ';
      continue;
    }

    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      output += ' ';
      continue;
    }

    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      output += ' ';
      continue;
    }

    if (char === '#') {
      while (i < command.length && command[i] !== '\n') {
        output += ' ';
        i++;
      }
      output += '\n';
      continue;
    }

    output += char;
  }

  return output;
}

export async function validateBashCommand(
  command: string,
  config: t.LocalExecutionConfig = {}
): Promise<BashValidationResult> {
  const errors: string[] = [];
  const warnings: string[] = [];
  const normalized = stripQuotedContent(command);

  if (command.trim() === '') {
    errors.push('Command is empty.');
  }

  if (command.includes('\0')) {
    errors.push('Command contains a NUL byte.');
  }

  if (config.allowDangerousCommands !== true) {
    let blocked = false;
    // Strip-then-match for the bare-form patterns (avoids false
    // positives where the destructive text is buried inside a
    // string the user is just printing).
    for (const pattern of dangerousCommandPatterns) {
      if (pattern.test(normalized)) {
        errors.push('Command matches a destructive command pattern.');
        blocked = true;
        break;
      }
    }
    // Original-form pass for patterns that REQUIRE matching quote
    // pairs around a destructive path. Without this, `rm -rf "/"`
    // and `chmod -R 777 "/"` slip past the strip-then-match pass
    // because their destructive target is inside quotes.
    if (!blocked) {
      for (const pattern of quotedDestructivePatterns) {
        if (pattern.test(command)) {
          errors.push(
            'Command matches a destructive command pattern (quoted target).'
          );
          blocked = true;
          break;
        }
      }
    }
    if (!blocked) {
      for (const pattern of nestedShellDestructivePatterns) {
        if (pattern.test(command)) {
          errors.push(
            'Command matches a destructive command pattern (nested shell payload).'
          );
          break;
        }
      }
    }
  }

  const bashAstMode = config.bashAst ?? 'off';
  if (bashAstMode !== 'off' && config.allowDangerousCommands !== true) {
    const findings = runBashAstChecks(normalized, bashAstMode);
    const split = bashAstFindingsToErrors(findings);
    errors.push(...split.errors);
    warnings.push(...split.warnings);
  }

  if (config.readOnly === true && mutatingCommandPattern.test(normalized)) {
    errors.push(
      'Command appears to mutate files or repository state in read-only local mode.'
    );
  }

  // Use the same shell the actual execution path will use. Hard-coding
  // DEFAULT_SHELL here would reject perfectly valid commands when the
  // host configures `local.shell` to a non-bash binary (or when the
  // runtime doesn't have bash installed at all but does have e.g. zsh).
  const syntaxShell = config.shell ?? DEFAULT_SHELL;
  const syntax = await spawnLocalProcess(
    syntaxShell,
    ['-n', '-c', command],
    {
      ...config,
      timeoutMs: Math.min(config.timeoutMs ?? DEFAULT_TIMEOUT_MS, 5000),
      sandbox: { enabled: false },
    },
    { internal: true }
  ).catch(
    (error: Error): SpawnResult => ({
      stdout: '',
      stderr: error.message,
      exitCode: 1,
      timedOut: false,
    })
  );

  if (syntax.exitCode !== 0) {
    errors.push(
      syntax.stderr.trim() === ''
        ? 'Command failed shell syntax validation.'
        : `Command failed shell syntax validation: ${syntax.stderr.trim()}`
    );
  }

  if (/\bsudo\b/.test(normalized)) {
    warnings.push('Command requests elevated privileges with sudo.');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

async function ensureSandbox(
  config: t.LocalExecutionConfig,
  cwd: string
): Promise<SandboxManagerType | undefined> {
  if (!shouldUseLocalSandbox(config)) {
    return undefined;
  }

  const runtime = await loadSandboxRuntime().catch((error: Error) => {
    throw new Error(`${missingSandboxRuntimeMessage} Cause: ${error.message}`);
  });

  const runtimeConfig = buildSandboxRuntimeConfig(
    config,
    cwd,
    runtime.getDefaultWritePaths
  );
  const nextKey = JSON.stringify(runtimeConfig);

  if (sandboxInitialized && sandboxConfigKey === nextKey) {
    return runtime.SandboxManager;
  }

  const dependencyCheck = runtime.SandboxManager.checkDependencies();
  if (dependencyCheck.errors.length > 0) {
    if (config.sandbox?.failIfUnavailable === true) {
      throw new Error(
        `Local sandbox requested but unavailable: ${dependencyCheck.errors.join('; ')}`
      );
    }
    return undefined;
  }

  if (sandboxInitialized) {
    await runtime.SandboxManager.reset();
  }

  await runtime.SandboxManager.initialize(runtimeConfig);
  sandboxInitialized = true;
  sandboxConfigKey = nextKey;
  return runtime.SandboxManager;
}

/**
 * Loopback addresses the in-process programmatic-tool bridge listens
 * on (`LocalProgrammaticToolCalling.ts` binds 127.0.0.1). Sandboxed
 * code launched by `run_tools_with_code` / `run_tools_with_bash` HTTPs
 * back to that address — without the entries below, the bridge is
 * silently blocked under sandbox.
 */
const BRIDGE_LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1'] as const;

/**
 * Structural shape of the sandbox-runtime config we hand to
 * `SandboxManager.initialize()`. Intentionally NOT typed as the peer
 * `SandboxRuntimeConfig` from `@anthropic-ai/sandbox-runtime`: that
 * package is an OPTIONAL peer dep, and exporting a function whose
 * return type references it would make our generated `.d.ts` import
 * a module the consumer may not have installed (Codex P1 #22 — type-
 * checking would fail with `Cannot find module
 * '@anthropic-ai/sandbox-runtime'` for any host that doesn't enable
 * local sandboxing). The shape here is a structural subset; assignable
 * to the real `SandboxRuntimeConfig` at the one runtime call site.
 *
 * @internal
 */
export interface BuiltSandboxRuntimeConfig {
  network: {
    allowedDomains: string[];
    deniedDomains: string[];
    allowUnixSockets?: string[];
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
    allowMachLookup?: string[];
  };
  filesystem: {
    denyRead: string[];
    allowRead?: string[];
    allowWrite: string[];
    denyWrite: string[];
    allowGitConfig?: boolean;
  };
}

export function buildSandboxRuntimeConfig(
  config: t.LocalExecutionConfig,
  cwd: string,
  getDefaultWritePaths: () => string[]
): BuiltSandboxRuntimeConfig {
  const sandbox = config.sandbox;
  // Seed allowedDomains with loopback so the programmatic-tool bridge
  // works under sandbox. If the host explicitly denied a loopback
  // entry via `deniedDomains`, respect that and skip seeding it.
  const userAllowed = sandbox?.network?.allowedDomains ?? [];
  const denied = new Set(sandbox?.network?.deniedDomains ?? []);
  const seededLoopback = BRIDGE_LOOPBACK_HOSTS.filter(
    (host) => !denied.has(host) && !userAllowed.includes(host)
  );
  const allowedDomains = [...seededLoopback, ...userAllowed];
  // Mirror the file-tools workspace boundary: anything in
  // `additionalRoots` counts as in-workspace, so sandboxed shell/code
  // can write there too. Without this, file_tools can resolve a
  // sibling-root path but `bash`/`execute_code` is denied write
  // access — confusing divergence flagged in Codex P2 #15.
  const workspaceWriteRoots =
    config.workspace?.additionalRoots != null
      ? getWorkspaceRoots(config)
      : [cwd];
  return {
    network: {
      allowedDomains,
      deniedDomains: sandbox?.network?.deniedDomains ?? [],
      ...(sandbox?.network?.allowUnixSockets != null && {
        allowUnixSockets: sandbox.network.allowUnixSockets,
      }),
      ...(sandbox?.network?.allowAllUnixSockets != null && {
        allowAllUnixSockets: sandbox.network.allowAllUnixSockets,
      }),
      ...(sandbox?.network?.allowLocalBinding != null && {
        allowLocalBinding: sandbox.network.allowLocalBinding,
      }),
      ...(sandbox?.network?.allowMachLookup != null && {
        allowMachLookup: sandbox.network.allowMachLookup,
      }),
    },
    filesystem: {
      denyRead: sandbox?.filesystem?.denyRead ?? [],
      allowRead: sandbox?.filesystem?.allowRead,
      allowWrite: sandbox?.filesystem?.allowWrite ?? [
        ...workspaceWriteRoots,
        ...getDefaultWritePaths(),
      ],
      denyWrite: sandbox?.filesystem?.denyWrite ?? [
        '.env',
        '.env.*',
        '.git/config',
        '.git/hooks/**',
      ],
      allowGitConfig: sandbox?.filesystem?.allowGitConfig,
    },
  };
}

/**
 * Internal options for {@link spawnLocalProcess} that we don't want
 * exposed on the public `LocalExecutionConfig` type.
 *
 * @internal
 */
export interface SpawnLocalProcessOptions {
  /** Cancels the owned process tree and waits for its close event. */
  signal?: AbortSignal;
  /**
   * When true, suppress the "sandbox is off" warning AND its latch
   * for this spawn. Use for SDK-internal probes (`bash -n` syntax
   * preflight, `rg --version`, etc.) that intentionally run with
   * the sandbox forced off — the warning is noise for those, and
   * letting the latch flip would hide the warning when a *real*
   * unsandboxed execution happens later in the same process.
   */
  internal?: boolean;
}

export const LOCAL_SPAWN_TIMEOUT_MS = Symbol('librechat.localSpawn.timeoutMs');

export async function spawnLocalProcess(
  command: string,
  args: string[],
  config: t.LocalExecutionConfig = {},
  options?: SpawnLocalProcessOptions
): Promise<SpawnResult> {
  options?.signal?.throwIfAborted();
  const cwd = getLocalCwd(config);
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxOutputChars = config.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS;
  // Streaming caps. Local tools execute arbitrary shell/code, so a noisy
  // command (`yes`, `cat /dev/urandom | base64`, a verbose build) could
  // accumulate gigabytes in memory before hitting the post-close cap.
  // We bound in-memory per-stream and spill the rest to disk; we also
  // hard-kill the child once total streamed bytes pass `maxSpawnedBytes`
  // so a process producing unbounded output gets stopped instead of
  // letting the host OOM.
  const inMemoryCapBytes = maxOutputChars * 2;
  const hardKillBytes = config.maxSpawnedBytes ?? DEFAULT_MAX_SPAWNED_BYTES;
  const sandboxManager = await ensureSandbox(config, cwd);
  // Internal probes (validateBashCommand syntax preflight,
  // isRipgrepAvailable, syntax-check probe cache priming) pass
  // `internal: true` so they don't emit a misleading "sandbox is
  // off" warning AND don't flip `sandboxOffWarned = true`. Without
  // this Codex P2 path: a run with `sandbox.enabled: true` would
  // see a false warning from the syntax preflight, and the latch
  // flip would suppress the warning in a *later* truly-unsandboxed
  // run — exactly the scenario operators need to see.
  if (sandboxManager == null && options?.internal !== true) {
    maybeWarnSandboxOff(config);
  }
  let spawnCommand = command;
  let spawnArgs = args;

  if (sandboxManager != null) {
    const rendered = [command, ...args.map(shellQuote)].join(' ');
    const sandboxed = await sandboxManager.wrapWithSandbox(rendered);
    spawnCommand = config.shell ?? DEFAULT_SHELL;
    spawnArgs = ['-lc', sandboxed];
  }

  const launcher = getSpawn(config);
  options?.signal?.throwIfAborted();
  return new Promise<SpawnResult>((resolveResult, reject) => {
    const spawnOptions: import('child_process').SpawnOptions = {
      cwd,
      detached: process.platform !== 'win32',
      env: { ...process.env, ...(config.env ?? {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
    };
    Object.defineProperty(spawnOptions, LOCAL_SPAWN_TIMEOUT_MS, {
      value: timeoutMs,
    });
    const child = launcher(spawnCommand, spawnArgs, spawnOptions);
    const onAbort = (): void => killProcessTree(child);

    let stdout = '';
    let stderr = '';
    let totalSpawnedBytes = 0;
    let overflowKilled = false;
    let spillStream: import('fs').WriteStream | undefined;
    let spillPath: string | undefined;
    let outputRecoveryError: string | undefined;
    let capturePrefix = '';
    const decoders = { stdout: new StringDecoder('utf8'), stderr: new StringDecoder('utf8') };
    let settled = false;
    let timedOut = false;
    let timeout: NodeJS.Timeout | undefined;

    const ensureSpill = (): void => {
      if (spillStream != null || outputRecoveryError != null) return;
      spillPath = resolve(tmpdir(), `lc-local-output-${randomUUID()}.txt`);
      try {
        spillStream = createWriteStream(spillPath, { flags: 'wx', mode: 0o600 });
        spillStream.on('error', (error) => {
          outputRecoveryError = error.message;
        });
        spillStream.write(capturePrefix);
      } catch (error) {
        outputRecoveryError = error instanceof Error ? error.message : String(error);
      }
      capturePrefix = '';
    };

    const captureText = (text: string, kind: 'stdout' | 'stderr'): void => {
      const current = kind === 'stdout' ? stdout : stderr;
      const totalCap = config.toolResultMaxChars;
      if (current.length + text.length > maxOutputChars ||
        (totalCap != null && totalCap > 0 && stdout.length + stderr.length + text.length > totalCap)) ensureSpill();
      if (spillStream != null && outputRecoveryError == null) spillStream.write(text);
      else if (outputRecoveryError == null) capturePrefix += text;
      if (current.length < inMemoryCapBytes) {
        const retained = text.slice(0, inMemoryCapBytes - current.length);
        if (kind === 'stdout') stdout += retained;
        else stderr += retained;
      }
    };

    const handleChunk = (buf: Buffer, kind: 'stdout' | 'stderr'): void => {
      totalSpawnedBytes += buf.length;
      // hardKillBytes <= 0 means "no cap" per the public config contract
      // (see LocalExecutionConfig.maxSpawnedBytes). Skip the kill check
      // entirely in that case so a single byte doesn't terminate the run.
      if (
        hardKillBytes > 0 &&
        totalSpawnedBytes > hardKillBytes &&
        !overflowKilled
      ) {
        overflowKilled = true;
        killProcessTree(child);
        return;
      }
      if (!overflowKilled) captureText(decoders[kind].write(buf), kind);
    };

    const finish = (result: SpawnResult): void => {
      if (settled) {
        return;
      }
      settled = true;
      options?.signal?.removeEventListener('abort', onAbort);
      if (timeout != null) {
        clearTimeout(timeout);
      }
      let finalized = false;
      const finalize = (): void => {
        if (finalized) return;
        finalized = true;
        const truncated = {
          stdout: truncateLocalOutput(result.stdout, maxOutputChars),
          stderr: truncateLocalOutput(result.stderr, maxOutputChars),
        };
        resolveResult({
          ...result,
          ...truncated,
          outputTruncated: result.stdout.length > maxOutputChars || result.stderr.length > maxOutputChars,
          ...(spillPath != null && outputRecoveryError == null ? { fullOutputPath: spillPath } : {}),
          ...(config.exec?.spawn == null && config.spawn == null ? {} : { outputOnHost: true }),
          ...(outputRecoveryError == null ? {} : { outputRecoveryError }),
        });
      };
      if (spillStream == null || spillStream.closed) {
        finalize();
        return;
      }
      spillStream.once('close', finalize);
      spillStream.end((error?: Error | null) => {
        if (error != null) outputRecoveryError = error.message;
      });
    };

    const fail = (error: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      options?.signal?.removeEventListener('abort', onAbort);
      if (timeout != null) {
        clearTimeout(timeout);
      }
      if (spillStream != null) {
        spillStream.end();
      }
      reject(error);
    };

    if (timeoutMs > 0) {
      timeout = setTimeout(() => {
        timedOut = true;
        killProcessTree(child);
      }, timeoutMs);
    }

    child.stdout.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stdout');
    });

    child.stderr.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stderr');
    });

    child.on('error', fail);

    child.on('close', (exitCode, signal) => {
      captureText(decoders.stdout.end(), 'stdout');
      captureText(decoders.stderr.end(), 'stderr');
      // Synthesize a non-zero exit code whenever the process exited
      // by signal — Node reports `exitCode: null` in that case and
      // the formatter only prints non-null exit codes, so signal
      // kills (overflow guard, `kill -9 $$` from inside the script,
      // native crashes, OS OOM killer, …) would otherwise look like
      // successful runs (Codex P1 + Codex P2). Overflow path keeps
      // its 137 (SIGKILL) for compatibility; other signals map per
      // POSIX `128 + signum`.
      let finalExit: number | null = exitCode;
      if (finalExit == null) {
        if (overflowKilled) {
          finalExit = 137;
        } else if (signal != null) {
          finalExit = exitCodeForSignal(signal);
        }
      }
      finish({
        stdout,
        stderr,
        exitCode: finalExit,
        timedOut,
        ...(overflowKilled ? { overflowKilled: true } : {}),
        ...(signal != null ? { signal } : {}),
      });
    });
    if (options?.signal?.aborted === true) onAbort();
    else options?.signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Result of a command-availability probe and whether its verdict is stable. */
export type CommandAvailabilityProbe = {
  available: boolean;
  cacheable: boolean;
  cacheUntil?: number;
};

async function isDurableCommandLookupError(
  error: object,
  config: t.LocalExecutionConfig
): Promise<boolean> {
  if (!('code' in error) || error.code !== 'ENOENT') {
    return false;
  }
  try {
    const cwd = getLocalCwd(config);
    return (await getWorkspaceFS(config).stat(cwd)).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Probes one executable without turning transient backend failures into
 * permanent capability facts. Exit 126/127 is a definite unusable/missing
 * executable. A native ENOENT is stable only when the current working
 * directory exists; timeouts, transport failures, and ambiguous lookup
 * failures are retried.
 */
export async function probeLocalCommandAvailability(
  command: string,
  args: string[],
  config: t.LocalExecutionConfig
): Promise<CommandAvailabilityProbe> {
  try {
    const result = await spawnLocalProcess(
      command,
      args,
      { ...config, timeoutMs: 5000, sandbox: { enabled: false } },
      { internal: true }
    );
    const available = result.exitCode === 0;
    const durableNegative =
      result.exitCode === 126 || result.exitCode === 127;
    return {
      available,
      cacheable: available || durableNegative,
      ...(durableNegative
        ? { cacheUntil: Date.now() + NEGATIVE_COMMAND_AVAILABILITY_TTL_MS }
        : {}),
    };
  } catch (error) {
    const cacheable =
      typeof error === 'object' &&
      error != null &&
      (await isDurableCommandLookupError(error, config));
    return {
      available: false,
      cacheable,
      ...(cacheable
        ? { cacheUntil: Date.now() + NEGATIVE_COMMAND_AVAILABILITY_TTL_MS }
        : {}),
    };
  }
}

export async function executeLocalBash(
  command: string,
  config: t.LocalExecutionConfig = {},
  signal?: AbortSignal
): Promise<SpawnResult> {
  signal?.throwIfAborted();
  const validation = await validateBashCommand(command, config);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }
  const shell = config.shell ?? DEFAULT_SHELL;
  return spawnLocalProcess(shell, ['-lc', command], config, { signal });
}

/**
 * Variant of `executeLocalBash` that exposes `args` as positional
 * shell parameters (`$1`, `$2`, …). Mirrors what the other runtimes
 * do in `getRuntimeCommand`. Uses the standard `bash -c <code> --
 * arg0 arg1 …` form: the `--` becomes `$0`, then `args[0]` is `$1`
 * and so on. Same AST validation as the no-args path.
 *
 * Used by both the `execute_code`/`lang:'bash'` path AND the
 * `bash_tool` factory so the schema's `args` contract works
 * identically in both surfaces.
 */
/**
 * Matches a single arg that, on its own, references a protected
 * location (`/`, `~`, `$HOME`, `${HOME}`, `.`, with optional trailing
 * slash, wildcard, or dot-glob suffix). Used to spot the
 * `command: 'rm -rf "$1"', args: ['/']` shape where the destructive
 * target is moved into a positional arg to evade the command regex.
 * Codex P1 [45], extended for dot-glob in Codex P1 [47] (mirrors the
 * `DESTRUCTIVE_TARGET` suffix matrix exactly).
 */
const PROTECTED_TARGET_ARG_RE = /^(?:\/|~|\$\{?HOME\}?|\.)(?:\/?\.?\*|\/)?$/;

/**
 * Mutating-op recognizer for the args check. Conservative: only the
 * three operations the destructive-command guard already covers
 * directly (`rm -rf …`, `chmod -R …`, `chown -R …`). Other shell
 * builtins might mutate state (`mv`, `cp` over an existing file,
 * etc.) but the destructive guard doesn't try to catch those today,
 * so we don't widen here either.
 */
const DESTRUCTIVE_OP_IN_COMMAND_RE =
  /\b(?:rm\s+-[^\s]*[rf]|chmod\s+-R|chown\s+-R)\b/;

export async function executeLocalBashWithArgs(
  command: string,
  args: readonly string[],
  config: t.LocalExecutionConfig = {},
  signal?: AbortSignal
): Promise<SpawnResult> {
  signal?.throwIfAborted();
  const validation = await validateBashCommand(command, config);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }
  // Per-arg protected-target check (Codex P1 [45]). The command
  // regex can't see `$1`/`$@` substitutions at runtime — `command:
  // 'rm -rf "$1"', args: ['/']` would expand to `rm -rf '/'` inside
  // bash but the validator only saw `rm -rf "$1"` (no destructive
  // target). Block when (a) the command contains a destructive op
  // AND (b) at least one arg matches the protected-target shape.
  // Skipped when allowDangerousCommands is true (host-opted-in).
  if (
    args.length > 0 &&
    config.allowDangerousCommands !== true &&
    DESTRUCTIVE_OP_IN_COMMAND_RE.test(command)
  ) {
    const offending = args.find((a) => PROTECTED_TARGET_ARG_RE.test(a));
    if (offending !== undefined) {
      throw new Error(
        `Command matches a destructive command pattern (protected target "${offending}" passed via positional arg).`
      );
    }
  }
  const shell = config.shell ?? DEFAULT_SHELL;
  return spawnLocalProcess(shell, ['-lc', command, '--', ...args], config, { signal });
}

export async function executeLocalCode(
  input: {
    lang: string;
    code: string;
    args?: string[];
  },
  config: t.LocalExecutionConfig = {},
  signal?: AbortSignal
): Promise<SpawnResult> {
  signal?.throwIfAborted();
  if (input.lang === 'bash') {
    // Append `args` as positional parameters via the standard
    // `bash -c <code> -- <args...>` form so `$1`, `$2`, … inside
    // `code` resolve correctly. Honours the same args contract the
    // other runtimes (py, js, …) already support.
    if (input.args != null && input.args.length > 0) {
      return executeLocalBashWithArgs(input.code, input.args, config, signal);
    }
    return executeLocalBash(input.code, config, signal);
  }

  const tempDir = resolve(tmpdir(), `lc-local-${randomUUID()}`);
  await mkdir(tempDir, { recursive: true });

  try {
    const runtime = getRuntimeCommand(
      input.lang,
      tempDir,
      input.code,
      input.args,
      config.shell
    );
    if (runtime.source != null) {
      await writeFile(
        resolve(tempDir, runtime.fileName),
        runtime.source,
        'utf8'
      );
    }
    return await spawnLocalProcess(runtime.command, runtime.args, config, { signal });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

function getRuntimeCommand(
  lang: string,
  tempDir: string,
  code: string,
  args: string[] = [],
  // Override for the shell used by compile-style runtimes (`rs`,
  // `c`, `cpp`, `java`, `d`, `f90`). Threads `local.shell` so a host
  // that doesn't have bash (or wants `/bin/sh` / zsh) can still
  // execute these languages — Codex P2 #29: the bare-bash hardcode
  // mirrored the same gap that Codex P1 #6 fixed for the syntax
  // preflight, but had been missed for these runtime invocations.
  shellOverride?: string
): RuntimeCommand {
  const fileFor = (name: string): string => resolve(tempDir, name);
  const shell = shellOverride ?? configShell();

  switch (lang) {
  case 'py':
    return {
      command: 'python3',
      args: [fileFor('main.py'), ...args],
      fileName: 'main.py',
      source: code,
    };
  case 'js':
    return {
      command: 'node',
      args: [fileFor('main.js'), ...args],
      fileName: 'main.js',
      source: code,
    };
  case 'ts':
    return {
      command: 'npx',
      args: ['--no-install', 'tsx', fileFor('main.ts'), ...args],
      fileName: 'main.ts',
      source: code,
    };
  case 'php':
    return {
      command: 'php',
      args: [fileFor('main.php'), ...args],
      fileName: 'main.php',
      source: code,
    };
  case 'go':
    return {
      command: 'go',
      args: ['run', fileFor('main.go'), ...args],
      fileName: 'main.go',
      source: code,
    };
  case 'rs':
    return {
      command: shell,
      args: [
        '-lc',
        `rustc ${shellQuote(fileFor('main.rs'))} -o ${shellQuote(
          fileFor('main-rs')
        )} && ${shellQuote(fileFor('main-rs'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.rs',
      source: code,
    };
  case 'c':
    return {
      command: shell,
      args: [
        '-lc',
        `cc ${shellQuote(fileFor('main.c'))} -o ${shellQuote(
          fileFor('main-c')
        )} && ${shellQuote(fileFor('main-c'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.c',
      source: code,
    };
  case 'cpp':
    return {
      command: shell,
      args: [
        '-lc',
        `c++ ${shellQuote(fileFor('main.cpp'))} -o ${shellQuote(
          fileFor('main-cpp')
        )} && ${shellQuote(fileFor('main-cpp'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.cpp',
      source: code,
    };
  case 'java':
    return {
      command: shell,
      args: [
        '-lc',
        `javac ${shellQuote(fileFor('Main.java'))} && java -cp ${shellQuote(
          tempDir
        )} Main ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'Main.java',
      source: code,
    };
  case 'r':
    return {
      command: 'Rscript',
      args: [fileFor('main.R'), ...args],
      fileName: 'main.R',
      source: code,
    };
  case 'd':
    return {
      command: shell,
      args: [
        '-lc',
        `dmd ${shellQuote(fileFor('main.d'))} -of=${shellQuote(
          fileFor('main-d')
        )} && ${shellQuote(fileFor('main-d'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.d',
      source: code,
    };
  case 'f90':
    return {
      command: shell,
      args: [
        '-lc',
        `gfortran ${shellQuote(fileFor('main.f90'))} -o ${shellQuote(
          fileFor('main-f90')
        )} && ${shellQuote(fileFor('main-f90'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.f90',
      source: code,
    };
  default:
    throw new Error(`Unsupported local runtime: ${lang}`);
  }
}

function configShell(): string {
  return process.platform === 'win32' ? 'bash.exe' : 'bash';
}

/**
 * How long after SIGTERM we wait before escalating to SIGKILL. A
 * cooperative process gets a graceful chance to flush + clean up;
 * a process that ignores or traps SIGTERM (`trap '' TERM`) gets
 * killed unconditionally so timeoutMs / maxSpawnedBytes can't be
 * defeated by a hostile script. Codex P1 #28 — pre-fix the spawn
 * promise would never resolve in that case and the entire tool run
 * would hang past the advertised timeout.
 */
const SIGKILL_ESCALATION_MS = 2000;

function sigterm(child: ChildProcess): void {
  if (child.pid == null) {
    child.kill('SIGTERM');
    return;
  }
  try {
    if (process.platform === 'win32') {
      child.kill('SIGTERM');
      return;
    }
    process.kill(-child.pid, 'SIGTERM');
  } catch {
    child.kill('SIGTERM');
  }
}

function sigkill(child: ChildProcess): void {
  if (child.exitCode != null || child.signalCode != null) return;
  if (child.pid == null) {
    child.kill('SIGKILL');
    return;
  }
  try {
    if (process.platform === 'win32') {
      child.kill('SIGKILL');
      return;
    }
    process.kill(-child.pid, 'SIGKILL');
  } catch {
    try {
      child.kill('SIGKILL');
    } catch {
      /* already dead */
    }
  }
}

function killProcessTree(child: ChildProcess): void {
  sigterm(child);
  // Escalate to SIGKILL if the child is still alive after the grace
  // window. Use unref() so the timer doesn't keep the Node process
  // alive past the parent's natural exit.
  const escalation = setTimeout(() => sigkill(child), SIGKILL_ESCALATION_MS);
  escalation.unref();
  child.once('close', () => clearTimeout(escalation));
}

export function shellQuote(value: string): string {
  if (value === '') {
    return '\'\'';
  }
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, '\'\\\'\'')}'`;
}

export function resolveWorkspacePath(
  filePath: string,
  config: t.LocalExecutionConfig = {},
  intent: 'read' | 'write' = 'write'
): string {
  const cwd = getLocalCwd(config);
  const absolutePath = isAbsolute(filePath)
    ? resolve(filePath)
    : resolve(cwd, filePath);

  const roots =
    intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
  if (roots == null) return absolutePath; // explicit allow-outside

  if (absolutePath === cwd || isInsideAnyRoot(absolutePath, roots)) {
    return absolutePath;
  }
  throw new Error(`Path is outside the local workspace: ${filePath}`);
}

function isInsideAnyRoot(absolutePath: string, roots: string[]): boolean {
  for (const root of roots) {
    if (absolutePath === root) return true;
    const rel = relative(root, absolutePath);
    if (rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel)) return true;
  }
  return false;
}

type RealpathFn = (p: string) => Promise<string>;

async function realpathOrSelf(
  absolutePath: string,
  realpathImpl: RealpathFn = realpath
): Promise<string> {
  try {
    return await realpathImpl(absolutePath);
  } catch {
    return absolutePath;
  }
}

/**
 * Resolves the realpath of `absolutePath`, falling back to the nearest
 * existing ancestor when the target itself does not yet exist (so the
 * containment check still works for `write_file` to a brand-new path).
 *
 * Codex P2 #38: takes the realpath impl as a parameter so callers
 * can route through `WorkspaceFS.realpath` when a custom engine is
 * configured. Pre-fix, host `fs/promises.realpath` would fail on a
 * remote/in-memory FS path and silently fall back to lexical
 * containment, leaving the symlink-escape clamp ineffective on
 * non-default engines.
 */
async function realpathOfPathOrAncestor(
  absolutePath: string,
  realpathImpl: RealpathFn = realpath
): Promise<string> {
  let current = absolutePath;
  let suffix = '';
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  while (true) {
    try {
      const real = await realpathImpl(current);
      return suffix === '' ? real : resolve(real, suffix);
    } catch {
      const parent = resolve(current, '..');
      if (parent === current) {
        return absolutePath;
      }
      const base = current.slice(parent.length + 1);
      suffix = suffix === '' ? base : `${base}/${suffix}`;
      current = parent;
    }
  }
}

/**
 * Resolves a workspace path AND follows any symlinks before checking
 * containment, so a symlink inside the workspace pointing outside is
 * rejected even though the lexical path looks safe. Handles paths that
 * don't yet exist (e.g. write_file targets) by realpath-resolving the
 * nearest existing ancestor and re-attaching the unresolved suffix.
 */
export async function resolveWorkspacePathSafe(
  filePath: string,
  config: t.LocalExecutionConfig = {},
  intent: 'read' | 'write' = 'write'
): Promise<string> {
  const lexical = resolveWorkspacePath(filePath, config, intent);
  const roots =
    intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
  if (roots == null) {
    return lexical;
  }
  // Route realpath through the configured WorkspaceFS so a custom
  // engine (in-memory, remote) gets the same symlink-escape clamp
  // the host-fs path gets. Codex P2 #38: pre-fix the host realpath
  // would fail on a non-default FS path and silently fall back to
  // lexical containment, leaving the clamp ineffective.
  const fsRealpath: RealpathFn = (p) => getWorkspaceFS(config).realpath(p);
  const realRoots = await Promise.all(
    roots.map((r) => realpathOrSelf(r, fsRealpath))
  );
  const realPath = await realpathOfPathOrAncestor(lexical, fsRealpath);
  if (isInsideAnyRoot(realPath, realRoots)) {
    return lexical;
  }
  throw new Error(
    `Path is outside the local workspace (symlink escape): ${filePath}`
  );
}

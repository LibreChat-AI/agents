/**
 * Opt-in `Stop` hook for headless coding agents: when the model tries to
 * finish after changing code without a passing test run since its last
 * change, the Run is kept warm once with an instruction to run the
 * project's existing tests. A run that already verified its last change
 * costs nothing; a run that made no change is never held.
 *
 * Detection is a heuristic over the Run's own messages: file changes are
 * successful `edit_file` / `write_file` calls (plus in-place `sed -i`, `perl -pi` and `patch` shell
 * commands), and a test run is a shell call whose last command starts a
 * common test runner and whose result reports no non-zero `exit_code` or
 * timeout.
 */

import type { BaseMessage } from '@langchain/core/messages';
import type { HookCallback, StopHookOutput } from './types';
import { Constants } from '@/common';

export interface VerifyBeforeStopConfig {
  /** Maximum continuations this hook requests per Run. Defaults to 1. */
  maxContinuations?: number;
  /** Shell tool names whose `command` argument is inspected. */
  shellToolNames?: readonly string[];
  /** Overrides the default test-runner pattern, matched against the last command of a shell line. */
  testCommandPattern?: RegExp;
  /** Overrides the instruction sent back to the model. */
  message?: string;
}

const TEST_RUNNERS =
  'pytest|py\\.test|unittest|tox|nox|jest|vitest|mocha|ava|tap|go test|cargo test|ctest|rspec|phpunit|mvn test|gradle test|make (?:test|check)|(?:npm|pnpm|yarn|bun) (?:run )?test';

/** Common wrappers in front of a runner: `python3 -m`, `npx`, `uv run`, `timeout 60`. */
const RUNNER_WRAPPERS =
  '(?:sudo|env|time|npx|uvx|uv run|poetry run|pipenv run|hatch run|python[\\d.]* -m|timeout(?: -\\S+)* \\S+)';

/**
 * A command segment that starts a test runner, after env assignments and
 * wrappers. Matched against the last segment of a shell command only (see
 * `isTestRun`), so a runner name elsewhere, such as in
 * `rg pytest pyproject.toml`, is not a test run.
 */
export const DEFAULT_TEST_PATTERN = new RegExp(
  `^\\s*(?:\\(\\s*)?(?:\\w+=\\S*\\s+)*(?:${RUNNER_WRAPPERS}\\s+)*(?:\\S*/)?(?:${TEST_RUNNERS})\\b`
);

/** Whether the `&` at `index` belongs to a redirection (`2>&1`, `<&3`, `&>`) rather than a separator. */
function isRedirection(command: string, index: number): boolean {
  const previous = command[index - 1];
  return previous === '>' || previous === '<' || command[index + 1] === '>';
}

/** Runner modes that list, describe or collect tests without running them. */
const NON_RUN_FLAGS =
  /\s(?:--collect-only|--co|--listTests|--list-tests|--list|--no-run|--dry-run|-list|-N|--help|-h|--version|-V|list)(?=\s|$)/;

/**
 * The commands of a shell command line, split at `;`, `&`, `|` and
 * newlines outside quotes (redirections such as `2>&1` do not split).
 */
export function shellSegments(command: string): string[] {
  const segments: string[] = [];
  let quote: string | undefined;
  let start = 0;
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (quote != null) {
      if (char === '\\' && quote === '"') i++;
      else if (char === quote) quote = undefined;
    } else if (char === '\\') {
      i++;
    } else if (char === '\'' || char === '"') {
      quote = char;
    } else if (char === '&' && isRedirection(command, i)) {
      continue;
    } else if (char === ';' || char === '&' || char === '|' || char === '\n') {
      segments.push(command.slice(start, i));
      start = i + 1;
    }
  }
  segments.push(command.slice(start));
  return segments;
}

/** The last command of a shell command line; its exit code is the one the shell reports. */
export function lastShellSegment(command: string): string {
  const segments = shellSegments(command);
  return segments[segments.length - 1];
}

/**
 * Whether a shell command runs tests whose result is the command's exit
 * code: the last segment must start with a runner. `pytest; echo done`,
 * `pytest || true` and `pytest | tail` report another command's exit code,
 * so they do not count; neither do a runner after `||`, which may not run,
 * or listing modes such as `--collect-only`, `--listTests` or `--help`.
 */
export function isTestRun(command: string, pattern: RegExp): boolean {
  if (command === '') return false;
  const segment = lastShellSegment(command);
  const before = command.slice(0, command.length - segment.length).trimEnd();
  if (before.endsWith('||')) return false;
  return pattern.test(segment) && !NON_RUN_FLAGS.test(segment);
}

/** A copy of `pattern` without the `g` and `y` flags, whose `test()` keeps no state between calls. */
export function statelessPattern(pattern: RegExp): RegExp {
  return pattern.global || pattern.sticky
    ? new RegExp(pattern.source, pattern.flags.replace(/[gy]/g, ''))
    : pattern;
}

/** A command segment that starts an in-place edit: `sed -i`, `perl -pi`, `patch`, `git apply`. */
export const SHELL_EDIT_PATTERN =
  /^\s*(?:\(\s*)?(?:\w+=\S*\s+)*(?:sudo\s+)?(?:(?:sed|perl)\s(?:.*\s)?-[a-zA-Z]*i\S*(?=\s|$)|patch\b|git\s+apply\b)/;

/** Dry-run modes of the shell edit commands, which change no file. */
const DRY_RUN_FLAGS = /\s(?:--check|--dry-run)(?=\s|$)/;

/**
 * Whether a shell command edits files in place: some command of the line,
 * outside quotes, starts an edit command that is not a dry run. A heuristic
 * over the command text.
 */
export function isShellEdit(command: string): boolean {
  return shellSegments(command).some(
    (segment) =>
      SHELL_EDIT_PATTERN.test(segment) && !DRY_RUN_FLAGS.test(segment)
  );
}

/** One tool call of an assistant turn, as the verification rules see it. */
export interface CheckedToolCall {
  name: string;
  /** The shell command, or `''` for other tools. */
  command: string;
  succeeded: boolean;
  /** The output the model saw; `undefined` when there is none. */
  output: string | undefined;
}

/** Whether a code change happened, and whether a passing test run followed it. */
export interface VerificationState {
  changed: boolean;
  verified: boolean;
}

function isChange(call: CheckedToolCall): boolean {
  return call.name === Constants.EDIT_FILE || call.name === Constants.WRITE_FILE
    ? call.succeeded
    : isShellEdit(call.command);
}

/**
 * Applies the tool calls of one assistant turn. Those calls run
 * concurrently, so a test run verifies changes from earlier turns, or an
 * edit earlier in its own shell command, but never a sibling call's edit.
 * Updates `state` in place and returns it.
 */
export function applyToolBatch(
  state: VerificationState,
  calls: readonly CheckedToolCall[],
  testPattern: RegExp
): VerificationState {
  const changes = calls.map(isChange);
  if (changes.some(Boolean)) {
    state.changed = true;
    state.verified = false;
  }
  calls.forEach((call, index) => {
    if (!state.changed || !isTestRun(call.command, testPattern)) return;
    if (changes.some((change, other) => change && other !== index)) return;
    state.verified =
      call.succeeded && call.output !== undefined && !failed(call.output);
  });
  return state;
}

const DEFAULT_MESSAGE =
  'Not finished yet: no passing test run follows your last code change. ' +
  'Run the existing tests that cover the files you changed (the whole suite if it is quick). ' +
  'If a test fails or the test command does not start, fix it and run the tests again. ' +
  'Then finish with a short summary.';

type ToolCallLike = { id?: string; name: string; args?: unknown };

function toolCallsOf(message: BaseMessage): ToolCallLike[] {
  const calls = (message as { tool_calls?: ToolCallLike[] }).tool_calls;
  return Array.isArray(calls) ? calls : [];
}

function commandOf(args: unknown): string {
  if (args == null || typeof args !== 'object') return '';
  const command = (args as { command?: unknown }).command;
  return typeof command === 'string' ? command : '';
}

function textOf(message: BaseMessage): string {
  const { content } = message;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part): string => {
      if (typeof part === 'string') return part;
      const text = (part as { text?: unknown }).text;
      return typeof text === 'string' ? text : '';
    })
    .join('');
}

export function failed(output: string): boolean {
  return (
    /(^|\n)exit_code: [1-9]\d*/.test(output) ||
    /(^|\n)timed_out: true/.test(output)
  );
}

/**
 * Returns whether the last code change in `messages` is followed by a
 * passing test run. `undefined` when the messages contain no code change.
 */
export function verifiedSinceLastChange(
  messages: readonly BaseMessage[],
  config: VerifyBeforeStopConfig = {}
): boolean | undefined {
  const shellNames = new Set(config.shellToolNames ?? [Constants.BASH_TOOL]);
  const testPattern = statelessPattern(
    config.testCommandPattern ?? DEFAULT_TEST_PATTERN
  );
  /** Tool output text by call id; `undefined` for an errored tool result. */
  const outputs = new Map<string, string | undefined>();
  const erroredIds = new Set<string>();
  for (const message of messages) {
    const id = (message as { tool_call_id?: unknown }).tool_call_id;
    if (typeof id !== 'string') continue;
    const errored = (message as { status?: unknown }).status === 'error';
    if (errored) erroredIds.add(id);
    outputs.set(id, errored ? undefined : textOf(message));
  }
  const state: VerificationState = { changed: false, verified: false };
  for (const message of messages) {
    const calls = toolCallsOf(message).map(
      (call): CheckedToolCall => ({
        name: call.name,
        command: shellNames.has(call.name) ? commandOf(call.args) : '',
        succeeded: call.id === undefined || !erroredIds.has(call.id),
        output: call.id === undefined ? undefined : outputs.get(call.id),
      })
    );
    if (calls.length > 0) applyToolBatch(state, calls, testPattern);
  }
  return state.changed ? state.verified : undefined;
}

export function createVerifyBeforeStopHook(
  config: VerifyBeforeStopConfig = {}
): HookCallback<'Stop'> {
  const maxContinuations = Math.max(0, config.maxContinuations ?? 1);
  return (input): StopHookOutput => {
    if (input.continuationCount >= maxContinuations) return {};
    if (verifiedSinceLastChange(input.messages, config) !== false) return {};
    return {
      decision: 'block',
      reason: 'unverified change',
      additionalContext: config.message ?? DEFAULT_MESSAGE,
    };
  };
}

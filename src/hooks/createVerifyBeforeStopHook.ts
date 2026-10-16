/**
 * Opt-in `Stop` hook for headless coding agents: when the model tries to
 * finish after changing code without a passing test run since its last
 * change, the Run is kept warm once with an instruction to run the
 * project's existing tests. A run that already verified its last change
 * costs nothing; a run that made no change is never held.
 *
 * Detection reads the Run's own messages: file changes are `edit_file` /
 * `write_file` calls (plus in-place `sed -i`, `perl -pi` and `patch` shell
 * commands), and a test run is a shell call whose command matches a common
 * test runner and whose result reports no non-zero `exit_code` or timeout.
 */

import type { BaseMessage } from '@langchain/core/messages';
import type { HookCallback, StopHookOutput } from './types';
import { Constants } from '@/common';

export interface VerifyBeforeStopConfig {
  /** Maximum continuations this hook requests per Run. Defaults to 1. */
  maxContinuations?: number;
  /** Shell tool names whose `command` argument is inspected. */
  shellToolNames?: readonly string[];
  /** Overrides the default test-runner pattern. */
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
 * A test runner invoked as a command: at the start of a command segment,
 * after env assignments and wrappers. A runner name elsewhere in a command,
 * such as `rg pytest pyproject.toml`, is not a test run.
 */
export const DEFAULT_TEST_PATTERN = new RegExp(
  `(?:^|[;&|(]|\\b(?:then|do)\\b)\\s*(?:\\w+=\\S*\\s+)*(?:${RUNNER_WRAPPERS}\\s+)*(?:\\S*/)?(?:${TEST_RUNNERS})\\b`
);

/** A copy of `pattern` without the `g` and `y` flags, whose `test()` keeps no state between calls. */
export function statelessPattern(pattern: RegExp): RegExp {
  return pattern.global || pattern.sticky
    ? new RegExp(pattern.source, pattern.flags.replace(/[gy]/g, ''))
    : pattern;
}

export const SHELL_EDIT_PATTERN = /\b(sed -i|perl -p?i|patch -p\d|git apply)\b/;

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
  for (const message of messages) {
    const id = (message as { tool_call_id?: unknown }).tool_call_id;
    if (typeof id !== 'string') continue;
    const errored = (message as { status?: unknown }).status === 'error';
    outputs.set(id, errored ? undefined : textOf(message));
  }
  let changed = false;
  let verified = false;
  for (const message of messages) {
    for (const call of toolCallsOf(message)) {
      const command = shellNames.has(call.name) ? commandOf(call.args) : '';
      if (
        call.name === Constants.EDIT_FILE ||
        call.name === Constants.WRITE_FILE ||
        SHELL_EDIT_PATTERN.test(command)
      ) {
        changed = true;
        verified = false;
        continue;
      }
      if (changed && testPattern.test(command)) {
        const output = call.id === undefined ? undefined : outputs.get(call.id);
        verified = output !== undefined && !failed(output);
      }
    }
  }
  return changed ? verified : undefined;
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

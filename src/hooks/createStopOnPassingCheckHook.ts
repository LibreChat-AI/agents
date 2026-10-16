import type { HookCallback, PostToolBatchHookOutput } from './types';
import {
  DEFAULT_TEST_PATTERN,
  isShellEdit,
  failed,
  isTestRun,
  statelessPattern,
} from './createVerifyBeforeStopHook';
import { Constants } from '@/common';

/**
 * Opt-in `PostToolBatch` hook for headless coding agents: ends the Run as
 * soon as a tool batch closes with a passing test run that follows the last
 * code change, so the model does not spend a further turn only to summarize.
 * A batch that changes code after its passing test, a failing test, or a run
 * that has made no change never stops early. The hook keeps per-Run state, so
 * register a fresh instance for every Run.
 */
export interface StopOnPassingCheckConfig {
  /** Shell tool names whose `command` argument is inspected. */
  shellToolNames?: readonly string[];
  /** Overrides the default test-runner pattern, matched against the last command of a shell line. */
  testCommandPattern?: RegExp;
}

function commandOf(input: Record<string, unknown>): string {
  return typeof input.command === 'string' ? input.command : '';
}

/** Text of a tool output as the model sees it (content-and-artifact tuples keep the content). */
export function outputText(output: unknown): string {
  if (typeof output === 'string') return output;
  if (Array.isArray(output) && typeof output[0] === 'string') return output[0];
  if (output != null && typeof output === 'object') {
    const content = (output as { content?: unknown }).content;
    if (typeof content === 'string') return content;
  }
  return '';
}

export function createStopOnPassingCheckHook(
  config: StopOnPassingCheckConfig = {}
): HookCallback<'PostToolBatch'> {
  const shellNames = new Set(config.shellToolNames ?? [Constants.BASH_TOOL]);
  const testPattern = statelessPattern(
    config.testCommandPattern ?? DEFAULT_TEST_PATTERN
  );
  let changed = false;
  return (input): PostToolBatchHookOutput => {
    let verified = false;
    for (const entry of input.entries) {
      const command = shellNames.has(entry.toolName)
        ? commandOf(entry.toolInput)
        : '';
      if (
        entry.toolName === Constants.EDIT_FILE ||
        entry.toolName === Constants.WRITE_FILE
      ) {
        if (entry.status === 'success') {
          changed = true;
          verified = false;
        }
        continue;
      }
      if (isShellEdit(command)) {
        changed = true;
        verified = false;
      }
      if (changed && isTestRun(command, testPattern)) {
        verified =
          entry.status === 'success' && !failed(outputText(entry.toolOutput));
      }
    }
    return changed && verified
      ? {
        preventContinuation: true,
        stopReason: 'passing test run after the last code change',
      }
      : {};
  };
}

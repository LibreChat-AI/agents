import type { HookCallback, PostToolBatchHookOutput } from './types';
import type { VerificationState } from './createVerifyBeforeStopHook';
import {
  DEFAULT_TEST_PATTERN,
  applyToolBatch,
  statelessPattern,
} from './createVerifyBeforeStopHook';
import { Constants } from '@/common';

/**
 * Opt-in `PostToolBatch` hook for headless coding agents: ends the Run as
 * soon as a tool batch closes with a passing test run that follows the last
 * code change, so the model does not spend a further turn only to summarize.
 * Calls in one batch run concurrently, so a test only verifies changes from
 * earlier batches (or an edit earlier in its own shell command).
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
  const state: VerificationState = { changed: false, verified: false };
  return (input): PostToolBatchHookOutput => {
    state.verified = false;
    const { changed, verified } = applyToolBatch(
      state,
      input.entries.map((entry) => ({
        name: entry.toolName,
        command: shellNames.has(entry.toolName)
          ? commandOf(entry.toolInput)
          : '',
        succeeded: entry.status === 'success',
        output: outputText(entry.toolOutput),
      })),
      testPattern
    );
    return changed && verified
      ? {
        preventContinuation: true,
        stopReason: 'passing test run after the last code change',
      }
      : {};
  };
}

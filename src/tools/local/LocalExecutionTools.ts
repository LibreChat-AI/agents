import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  executeLocalBash,
  executeLocalBashWithArgs,
  executeLocalCode,
  getLocalCwd,
  getLocalSessionId,
} from './LocalExecutionEngine';
import {
  BashExecutionToolName,
  BashExecutionToolSchema,
  BashToolOutputReferencesGuide,
} from '@/tools/BashExecutor';
import { capHeadTail } from './toolResultCaps';
import {
  CodeExecutionToolName,
  CodeExecutionToolSchema,
} from '@/tools/CodeExecutor';
import { Constants } from '@/common';
import { createAwaitedTool } from '../awaitedCancellation';

const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

export const LocalCodeExecutionToolDescription = `
Runs code on the local machine in the configured working directory. Unlike the remote Code API sandbox, this tool can see the local repository, installed runtimes, environment variables, and filesystem available to the host process.

Usage:
- The remote sandbox API remains the default; this description applies only when local execution mode is enabled.
- Local commands can use the Anthropic sandbox runtime when local.sandbox.enabled=true and @anthropic-ai/sandbox-runtime is installed.
- Commands execute in the local working directory and may modify local files.
- Input code is already displayed to the user, so do not repeat it unless asked.
- Output is not displayed unless you print it explicitly.
`.trim();

export const LocalBashExecutionToolDescription = `
Runs bash commands on the local machine in the configured working directory. Unlike the remote Code API sandbox, this tool can see the local repository, installed tools, environment variables, and filesystem available to the host process.

Usage:
- The remote sandbox API remains the default; this description applies only when local execution mode is enabled.
- Local commands can use the Anthropic sandbox runtime when local.sandbox.enabled=true and @anthropic-ai/sandbox-runtime is installed.
- Commands execute in the local working directory and may modify local files.
- Output is not displayed unless you print it explicitly.
- Prefer project-native commands and inspect files before changing them.
`.trim();

export const LocalBashExecutionToolSchema = {
  ...BashExecutionToolSchema,
  properties: {
    ...BashExecutionToolSchema.properties,
    command: {
      type: 'string',
      description: 'Bash command or script in the configured working directory. Files persist between executions; shell variables do not. Print the results needed for the task.',
    },
  },
} as const;

export function formatLocalOutput(
  result: {
    stdout: string;
    stderr: string;
    exitCode: number | null;
    timedOut: boolean;
    overflowKilled?: boolean;
    signal?: string;
    fullOutputPath?: string;
    outputTruncated?: boolean;
    outputRecoveryError?: string;
    outputOnHost?: boolean;
  },
  cwd: string
): string {
  let formatted = '';
  if (result.stdout !== '') {
    formatted += `stdout:\n${result.stdout}\n`;
  } else {
    formatted += emptyOutputMessage;
  }

  if (result.stderr !== '') {
    formatted += `stderr:\n${result.stderr}\n`;
  }

  if (result.exitCode != null && result.exitCode !== 0) {
    formatted += `exit_code: ${result.exitCode}\n`;
  }

  if (result.timedOut) {
    formatted += 'timed_out: true\n';
  }

  if (result.overflowKilled === true) {
    // Surface the force-kill explicitly so the model treats this as a
    // failure rather than misreading "exit_code: 137 + truncated
    // stdout" as a normal completion. (Codex P1 — pre-fix the close
    // handler swallowed the overflow flag and exitCode was null on
    // signal-killed processes.)
    formatted +=
      'killed: true (output exceeded local.maxSpawnedBytes; process tree was terminated)\n';
  } else if (result.signal != null) {
    // Generic signal kill: `kill -9 $$` from inside the script,
    // native crash, OS OOM killer, etc. Codex P2 generalization of
    // the overflow case.
    formatted += `killed: true (signal=${result.signal})\n`;
  }

  if (result.outputTruncated === true) formatted += 'output_truncated: true\n';
  if (result.overflowKilled === true || result.timedOut || result.signal != null) {
    formatted += 'capture_incomplete: true (execution was terminated)\n';
  }
  if (result.fullOutputPath != null) {
    const location = result.outputOnHost === true
      ? 'host capture; retrieval requires the host filesystem, which the custom execution backend may not share'
      : 'combined captured output; use local bash to inspect a bounded range';
    formatted += `full_output_path: ${result.fullOutputPath} (${location})\n`;
  }
  if (result.outputRecoveryError != null) {
    formatted += `output_recovery_failed: ${result.outputRecoveryError}\n`;
  }

  formatted += `working_directory: ${cwd}`;
  return formatted.trim();
}

export function createLocalCodeExecutionTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return createAwaitedTool({
    func: async (
      rawInput,
      _manager,
      invocation
    ): Promise<[string, t.CodeExecutionArtifact]> => {
      const input = rawInput as {
        lang: string;
        code: string;
        args?: string[];
      };
      const cwd = getLocalCwd(config);
      const result = await executeLocalCode(input, config, invocation?.signal);
      invocation?.signal?.throwIfAborted();
      return [
        formatLocalOutput(capLocalOutput(result, config), cwd),
        {
          session_id: getLocalSessionId(config),
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    name: CodeExecutionToolName,
    description: LocalCodeExecutionToolDescription,
    schema: CodeExecutionToolSchema,
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
  });
}

/**
 * Applies `toolResultMaxChars` to stdout and stderr before formatting, so
 * the status lines (`exit_code`, `timed_out`, `killed`) are always kept.
 * stderr gets at most half of the budget.
 */
export function capLocalOutput<T extends { stdout: string; stderr: string; outputTruncated?: boolean }>(
  result: T,
  config: t.LocalExecutionConfig
): T {
  const maxChars = config.toolResultMaxChars;
  if (maxChars == null || maxChars <= 0 || result.stdout.length + result.stderr.length <= maxChars) return result;
  const stderrBudget = Math.min(result.stderr.length, Math.floor(maxChars / 2));
  return {
    ...result,
    outputTruncated: result.outputTruncated === true || result.stdout.length + result.stderr.length > maxChars,
    stdout: capHeadTail(result.stdout, maxChars - stderrBudget),
    stderr: capHeadTail(result.stderr, Math.max(stderrBudget, 1)),
  };
}

export function createLocalBashExecutionTool(options?: {
  config?: t.LocalExecutionConfig;
  enableToolOutputReferences?: boolean;
}): DynamicStructuredTool {
  const config = options?.config ?? {};
  return createAwaitedTool({
    func: async (
      rawInput,
      _manager,
      invocation
    ): Promise<[string, t.CodeExecutionArtifact]> => {
      const input = rawInput as { command: string; args?: string[] };
      const cwd = getLocalCwd(config);
      // Use the standard `bash -lc <command> -- <args...>` form so
      // `$1`, `$2`, … resolve correctly inside `command`. The
      // previous implementation appended args literally to the
      // command string (`${command} ${args.join(' ')}`), which
      // doesn't populate positional parameters and silently broke
      // shell snippets like `command: 'echo "$1"'`.
      const result =
        input.args != null && input.args.length > 0
          ? await executeLocalBashWithArgs(
            input.command,
            input.args,
            config,
            invocation?.signal
          )
          : await executeLocalBash(input.command, config, invocation?.signal);
      invocation?.signal?.throwIfAborted();
      return [
        formatLocalOutput(capLocalOutput(result, config), cwd),
        {
          session_id: getLocalSessionId(config),
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    name: BashExecutionToolName,
    description:
      options?.enableToolOutputReferences === true
        ? `${LocalBashExecutionToolDescription}\n\n${BashToolOutputReferencesGuide}`
        : LocalBashExecutionToolDescription,
    schema: LocalBashExecutionToolSchema,
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
  });
}

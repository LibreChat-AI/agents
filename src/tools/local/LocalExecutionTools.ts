import { tool } from '@langchain/core/tools';
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

function formatLocalOutput(
  result: {
    stdout: string;
    stderr: string;
    exitCode: number | null;
    timedOut: boolean;
    overflowKilled?: boolean;
    signal?: string;
    fullOutputPath?: string;
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

  if (result.fullOutputPath != null) {
    formatted += `full_output_path: ${result.fullOutputPath} (output exceeded the configured cap; use bash to inspect — the file holds the complete stdout/stderr)\n`;
  }

  formatted += `working_directory: ${cwd}`;
  return formatted.trim();
}

export function createLocalCodeExecutionTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        lang: string;
        code: string;
        args?: string[];
      };
      const cwd = getLocalCwd(config);
      const result = await executeLocalCode(input, config);
      return [
        formatLocalOutput(result, cwd),
        {
          session_id: getLocalSessionId(config),
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    {
      name: CodeExecutionToolName,
      description: LocalCodeExecutionToolDescription,
      schema: CodeExecutionToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalBashExecutionTool(options?: {
  config?: t.LocalExecutionConfig;
  enableToolOutputReferences?: boolean;
}): DynamicStructuredTool {
  const config = options?.config ?? {};
  return tool(
    async (rawInput) => {
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
          ? await executeLocalBashWithArgs(input.command, input.args, config)
          : await executeLocalBash(input.command, config);
      const formatted = formatLocalOutput(result, cwd);
      return [
        config.toolResultMaxChars != null && config.toolResultMaxChars > 0
          ? capHeadTail(formatted, config.toolResultMaxChars)
          : formatted,
        {
          session_id: getLocalSessionId(config),
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    {
      name: BashExecutionToolName,
      description:
        options?.enableToolOutputReferences === true
          ? `${LocalBashExecutionToolDescription}\n\n${BashToolOutputReferencesGuide}`
          : LocalBashExecutionToolDescription,
      schema: BashExecutionToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

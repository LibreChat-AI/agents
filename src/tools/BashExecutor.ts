import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  BASH_SHELL_GUIDANCE,
  CODE_ARTIFACT_PATH_GUIDANCE,
  appendFailedExecutionFileReminder,
  appendTmpScratchReminder,
  appendCodeSessionFileSummary,
  addCodeApiExecutionProfileHeader,
  emptyOutputMessage,
  buildCodeApiHttpErrorMessage,
  buildCodeApiExecutionErrorMessage,
  CodeApiRequestError,
  buildCodeApiEndpoint,
  getCodeBaseURL,
  normalizeCodeApiRequestError,
  resolveCodeApiAuthHeaders,
  selectRuntimeSessionHint,
} from './CodeExecutor';
import {
  appendArtifactDeliveryWarning,
  normalizeArtifactDeliveryFailure,
} from '@/tools/ArtifactDelivery';
import {
  appendArtifactTruncationWarning,
  normalizeArtifactTruncation,
} from '@/tools/ArtifactTruncation';
import { appendExecutionArtifactFileSummary } from '@/tools/CodeSessionFileSummary';
import { resolveAttachedWorkspaceInstanceId } from '@/tools/workspaceIdentity';
import { prepareBashProgrammaticCode } from './BashProgrammaticToolCalling';
import { logCodeApiDiagnostic } from '@/tools/diagnostics';
import { makeRequest } from './ProgrammaticToolCalling';
import { resolveFetchProxyAgent } from '@/utils/proxy';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { Constants } from '@/common';

config();

export const BashExecutionToolSchema = {
  type: 'object',
  properties: {
    intent: { ...INTENT_PROPERTY },
    command: {
      type: 'string',
      description: `The bash command or script to execute.
- The environment is stateless; variables and state don't persist between executions.
- Prior /mnt/data files are available and can be modified in place.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- Input code **IS ALREADY** displayed to the user, so **DO NOT** repeat it in your response unless asked.
- Output code **IS NOT** displayed to the user, so **DO** write all desired output explicitly.
- IMPORTANT: You MUST explicitly print/output ALL results you want the user to see.
- Use \`echo\`, \`printf\`, or \`cat\` for all outputs.`,
    },
    args: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Additional arguments to execute the command with. This should only be used if the input command requires additional arguments to run.',
    },
  },
  required: ['command'],
} as const;

export const BashExecutionToolDescription = `
Runs bash commands and returns stdout/stderr output from a stateless execution environment, similar to running scripts in a command-line interface. Each execution is isolated and independent.

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- NEVER use this tool to execute malicious commands.
`.trim();

/**
 * Bash statefulness is filesystem-tier and scoped to `/mnt/data`. The machine
 * is warm across calls, but each call runs in a fresh sandbox (new process
 * tree + private /tmp), so background processes are reaped when the call ends
 * and anything written outside /mnt/data is discarded. The note must not
 * promise otherwise: a model told background processes survive will start a
 * server in one call and assume it is listening in the next.
 */
export const STATEFUL_BASH_NOTE =
  'Session state: commands in this conversation run on the same warm machine, so files written to /mnt/data persist between calls. Each call runs in a fresh, isolated sandbox: shell variables, the working directory, /tmp, and background processes do NOT survive after the call returns — a process started in one call is terminated when that call ends. Only /mnt/data is durable (the machine itself may also be reset at any time).';

export const StatefulBashExecutionToolDescription = `
Runs bash commands and returns stdout/stderr output. Commands in this conversation share one warm machine with a persistent /mnt/data, but each command runs in its own isolated sandbox (not a persistent shell session).

${STATEFUL_BASH_NOTE}

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- NEVER use this tool to execute malicious commands.
`.trim();

const AttachedWorkspaceBashExecutionToolDescription = `
Runs bash commands in the selected persistent project through an isolated sandbox process.

Usage:
- Project file changes persist between calls; shell variables, background processes, and execution-private temporary files do not.
- Injected files and generated artifacts use \${LIBRECHAT_CODE_DATA_DIR:-/mnt/data}; write durable files to the project root.
- Generated artifacts are automatically delivered; **DO NOT** provide download links.
- ${BASH_SHELL_GUIDANCE}
- NEVER use this tool to execute malicious commands.
`.trim();

/**
 * Supplemental prompt documenting the tool-output reference feature.
 *
 * Hosts should append this (separated by a blank line) to the base
 * {@link BashExecutionToolDescription} only when
 * `RunConfig.toolOutputReferences.enabled` is `true`. When the feature
 * is disabled, including this text would tell the LLM to emit
 * `{{tool0turn0}}` placeholders that pass through unsubstituted and
 * leak into the shell.
 */
export const BashToolOutputReferencesGuide = `
Referencing previous tool outputs:
- Every successful tool result is tagged with a reference key of the form \`tool<idx>turn<turn>\` (e.g., \`tool0turn0\`). The key appears either as a \`[ref: tool0turn0]\` prefix line or, when the output is a JSON object, as a \`_ref\` field on the object.
- To pipe a previous tool output into this tool, embed the placeholder \`{{tool<idx>turn<turn>}}\` literally anywhere in the \`command\` string (or any string arg). It will be substituted with the stored output verbatim before the command runs.
- The substituted value is the original output string (no \`[ref: …]\` prefix, no \`_ref\` key), so it is safe to pipe directly into \`jq\`, \`grep\`, \`awk\`, etc.
- Example (simple ASCII output): \`echo '{{tool0turn0}}' | jq '.foo'\` takes the full output of the first tool from the first turn and pipes it into jq.
- For payloads that may contain quotes, parentheses, backticks, or arbitrary bytes (random/binary data, JSON with embedded quotes, multi-line strings), prefer a quoted-delimiter heredoc over \`echo '…'\`. The heredoc body is not interpreted by the shell, so substituted payloads pass through unchanged.
- Heredoc example: \`wc -c << 'EOF'\\n{{tool0turn0}}\\nEOF\` (the quotes around \`'EOF'\` disable interpolation inside the body).
- Unknown reference keys are left in place and surfaced as \`[unresolved refs: …]\` after the output.
`.trim();

/**
 * Composes the bash tool description, optionally appending the
 * tool-output references guide. Hosts that enable
 * `RunConfig.toolOutputReferences` should pass `enableToolOutputReferences: true`
 * when registering the tool so the LLM learns the `{{…}}` syntax it
 * will actually be able to use.
 */
export function buildBashExecutionToolDescription(options?: {
  enableToolOutputReferences?: boolean;
  statefulSessions?: boolean;
  attachedWorkspace?: boolean;
}): string {
  let base = BashExecutionToolDescription;
  if (options?.attachedWorkspace === true) {
    base = AttachedWorkspaceBashExecutionToolDescription;
  } else if (options?.statefulSessions === true) {
    base = StatefulBashExecutionToolDescription;
  }
  if (options?.enableToolOutputReferences === true) {
    return `${base}\n\n${BashToolOutputReferencesGuide}`;
  }
  return base;
}

const STATELESS_BASH_PARAM_NOTE =
  'The environment is stateless; variables and state don\'t persist between executions.';
const STATEFUL_BASH_PARAM_NOTE =
  'Files written to /mnt/data persist between calls on the same warm machine. Each call runs in a fresh sandbox: shell variables, cwd, /tmp, and background processes do NOT survive the call. Only /mnt/data is durable.';
const ATTACHED_BASH_PARAM_NOTE =
  'Commands start in the selected persistent project. Project file changes persist, but shell variables, background processes, and execution-private temporary files do not.';
const ATTACHED_BASH_ARTIFACT_PATH_GUIDANCE =
  'Injected files and generated artifacts use `${LIBRECHAT_CODE_DATA_DIR:-/mnt/data}` for this execution only. Write anything needed later into the selected project.';
const ATTACHED_BASH_TMP_REMINDER =
  'Note: /tmp files are same-call scratch only and were not persisted; write files needed later into the selected project.';

export function buildBashExecutionToolSchema(opts?: {
  statefulSessions?: boolean;
  attachedWorkspace?: boolean;
}): typeof BashExecutionToolSchema {
  let note = STATELESS_BASH_PARAM_NOTE;
  if (opts?.attachedWorkspace === true) {
    note = ATTACHED_BASH_PARAM_NOTE;
  } else if (opts?.statefulSessions === true) {
    note = STATEFUL_BASH_PARAM_NOTE;
  }
  let commandDescription =
    BashExecutionToolSchema.properties.command.description.replace(
      STATELESS_BASH_PARAM_NOTE,
      note
    );
  if (opts?.attachedWorkspace === true) {
    commandDescription = commandDescription
      .replace('- Prior /mnt/data files are available and can be modified in place.\n', '')
      .replace(CODE_ARTIFACT_PATH_GUIDANCE, ATTACHED_BASH_ARTIFACT_PATH_GUIDANCE);
  }
  return {
    ...BashExecutionToolSchema,
    properties: {
      ...BashExecutionToolSchema.properties,
      command: {
        ...BashExecutionToolSchema.properties.command,
        description: commandDescription,
      },
    },
  } as typeof BashExecutionToolSchema;
}

export const BashExecutionToolName = Constants.BASH_TOOL;

function quoteBashArgument(value: string): string {
  return `'${value.replace(/'/g, '\'"\'"\'')}'`;
}

function commandWithArguments(command: string, args: string[] | undefined): string {
  if (args == null || args.length === 0) return command;
  return `bash -c ${quoteBashArgument(command)} -- ${args.map(quoteBashArgument).join(' ')}`;
}

/**
 * Default bash tool definition using the base description.
 *
 * When `RunConfig.toolOutputReferences.enabled` is `true`, build a
 * reference-aware description with
 * {@link buildBashExecutionToolDescription}
 * (`{ enableToolOutputReferences: true }`) and construct a custom
 * definition using it — using this constant as-is leaves the LLM
 * unaware of the `{{tool<i>turn<n>}}` syntax.
 */
export const BashExecutionToolDefinition = {
  name: BashExecutionToolName,
  description: BashExecutionToolDescription,
  schema: BashExecutionToolSchema,
} as const;

function createBashExecutionTool(
  params: t.BashExecutionToolParams | null = {}
): DynamicStructuredTool {
  const workspaceId = params?.workspaceId?.trim();
  const hasWorkspace = workspaceId != null && workspaceId !== '';
  const workspaceInstanceId = resolveAttachedWorkspaceInstanceId(
    params?.workspaceInstanceId,
    hasWorkspace
  );
  if (
    hasWorkspace &&
    !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(workspaceId)
  ) {
    throw new Error('Invalid attached workspace identifier');
  }
  const execEndpoint = buildCodeApiEndpoint(
    params?.baseUrl ?? getCodeBaseURL(),
    hasWorkspace ? 'exec/programmatic' : 'exec'
  );

  return tool(
    async (rawInput, config) => {
      /* `statefulSessions` drives the prompt and gates runtime affinity hints;
       * keep the flag itself out of the wire body. */
      const {
        authHeaders,
        baseUrl: _baseUrl,
        executionProfile,
        runtimeSessionHint,
        statefulSessions,
        workspaceId: _workspaceId,
        workspaceInstanceId: _workspaceInstanceId,
        ...executionParams
      } = params ?? {};
      void _baseUrl;
      void _workspaceId;
      void _workspaceInstanceId;
      /* Drop any model-supplied `runtime_session_hint` from the raw args: the
       * hint must only come from ToolNode's injected `_runtime_session_hint`
       * (below), never from the tool call itself. */
      /* `intent` is a UI display label — never part of the wire body. */
      const {
        command: rawCommand,
        intent: _ignoredIntent,
        runtime_session_hint: _ignoredModelHint,
        workspace_instance_id: _ignoredModelWorkspaceInstanceId,
        args,
        ...rest
      } = rawInput as {
        command: string;
        intent?: unknown;
        runtime_session_hint?: unknown;
        workspace_instance_id?: unknown;
        args?: string[];
      };
      void _ignoredModelHint;
      void _ignoredModelWorkspaceInstanceId;
      void _ignoredIntent;
      const command = hasWorkspace
        ? commandWithArguments(rawCommand, args)
        : rawCommand;
      const { session_id, _injected_files, _runtime_session_hint } =
        (config.toolCall ?? {}) as {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };

      const postData: Record<string, unknown> = {
        lang: 'bash',
        code: hasWorkspace ? prepareBashProgrammaticCode(command) : command,
        ...(hasWorkspace ? { tools: [] } : {}),
        ...(!hasWorkspace && args != null ? { args } : {}),
        ...rest,
        ...executionParams,
        ...(workspaceInstanceId != null && workspaceInstanceId !== ''
          ? { workspace_instance_id: workspaceInstanceId }
          : {}),
      };

      const effectiveRuntimeSessionHint = selectRuntimeSessionHint(
        runtimeSessionHint,
        _runtime_session_hint
      );
      if (
        statefulSessions === true &&
        executionProfile !== 'default' &&
        typeof effectiveRuntimeSessionHint === 'string' &&
        effectiveRuntimeSessionHint !== ''
      ) {
        postData.runtime_session_hint = effectiveRuntimeSessionHint;
      }

      /* See `CodeExecutor.ts` for the rationale — `/files/<session_id>`
       * HTTP fallback was removed because codeapi's sessionAuth requires
       * kind/id query params unavailable at this point. */
      if (_injected_files && _injected_files.length > 0) {
        postData.files = _injected_files;
      } else if (
        session_id != null &&
        session_id.length > 0 &&
        !Array.isArray(postData.files)
      ) {
        logCodeApiDiagnostic(
          'BashExecutor',
          'debug',
          'session carried no injected files; exec will run without input files',
          { files: 'none' }
        );
      }

      try {
        const resolvedAuthHeaders =
          await resolveCodeApiAuthHeaders(authHeaders);
        let result: Partial<t.ExecuteResult> & t.ProgrammaticExecutionResponse;
        if (hasWorkspace) {
          result = await makeRequest(
            execEndpoint,
            postData,
            undefined,
            {
              ...resolvedAuthHeaders,
              'X-LibreChat-Code-Workspace-ID': workspaceId,
            },
            executionProfile,
            config.signal
          );
        } else {
          const fetchOptions: RequestInit = {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'User-Agent': 'LibreChat/1.0',
              ...addCodeApiExecutionProfileHeader(
                resolvedAuthHeaders,
                executionProfile
              ),
            },
            body: JSON.stringify(postData),
            signal: config.signal,
          };

          const proxyAgent = resolveFetchProxyAgent(execEndpoint);
          if (proxyAgent != null) {
            fetchOptions.agent = proxyAgent;
          }
          const response = await fetch(execEndpoint, fetchOptions);
          if (!response.ok) {
            throw new CodeApiRequestError(
              await buildCodeApiHttpErrorMessage('POST', execEndpoint, response, {
                profile: executionProfile,
              })
            );
          }
          result = (await response.json()) as Partial<t.ExecuteResult> &
            t.ProgrammaticExecutionResponse;
        }
        if (hasWorkspace && result.status !== 'completed') {
          throw new CodeApiRequestError(
            buildCodeApiExecutionErrorMessage(
              result as t.ProgrammaticExecutionResponse
            )
          );
        }
        let formattedOutput = '';
        if (typeof result.stdout === 'string' && result.stdout.length > 0) {
          formattedOutput += `stdout:\n${result.stdout}\n`;
        } else {
          formattedOutput += emptyOutputMessage;
        }
        if (typeof result.stderr === 'string' && result.stderr.length > 0) {
          formattedOutput += `stderr:\n${result.stderr}\n`;
        }

        const outputWithReminder = appendTmpScratchReminder(
          formattedOutput,
          command,
          hasWorkspace ? ATTACHED_BASH_TMP_REMINDER : undefined
        );
        const artifactDelivery = normalizeArtifactDeliveryFailure(
          result.artifact_delivery
        );
        const outputWithDeliveryWarning = appendArtifactDeliveryWarning(
          outputWithReminder,
          artifactDelivery
        );
        const artifactTruncation = normalizeArtifactTruncation(
          result.artifact_truncation
        );
        const outputWithWarnings = appendArtifactTruncationWarning(
          outputWithDeliveryWarning,
          artifactTruncation
        );
        const hasFiles = result.files != null && result.files.length > 0;
        const deletionEcho =
          result.deleted_files != null
            ? { deleted_files: result.deleted_files }
            : {};
        const runtimeEcho =
          result.runtime_session_id != null
            ? {
              runtime_session_id: result.runtime_session_id,
              runtime_status: result.runtime_status,
            }
            : {};
        return [
          hasWorkspace
            ? appendExecutionArtifactFileSummary(
              outputWithWarnings,
              result.files
            )
            : appendCodeSessionFileSummary(
              outputWithWarnings,
              result.files
            ),
          (hasFiles
            ? {
              session_id: result.session_id,
              files: result.files,
              ...(artifactDelivery != null
                ? { artifact_delivery: artifactDelivery }
                : {}),
              ...(artifactTruncation != null
                ? { artifact_truncation: artifactTruncation }
                : {}),
              ...deletionEcho,
              ...runtimeEcho,
            }
            : {
              session_id: result.session_id,
              ...(artifactDelivery != null
                ? { artifact_delivery: artifactDelivery }
                : {}),
              ...(artifactTruncation != null
                ? { artifact_truncation: artifactTruncation }
                : {}),
              ...deletionEcho,
              ...runtimeEcho,
            }) satisfies t.CodeExecutionArtifact,
        ];
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          normalizeCodeApiRequestError(error).message,
          command
        );
        throw new CodeApiRequestError(
          `Execution error:\n\n${messageWithReminder}`
        );
      }
    },
    {
      name: BashExecutionToolName,
      description: buildBashExecutionToolDescription({
        statefulSessions: params?.statefulSessions,
        attachedWorkspace: hasWorkspace,
      }),
      schema: buildBashExecutionToolSchema({
        statefulSessions: params?.statefulSessions,
        attachedWorkspace: hasWorkspace,
      }),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export { createBashExecutionTool };

import { nanoid } from 'nanoid';
import { createHash } from 'crypto';
import { ToolCall } from '@langchain/core/messages/tool';
import { RunnableLambda } from '@langchain/core/runnables';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  isAIMessage,
  isBaseMessage,
} from '@langchain/core/messages';
import {
  END,
  Send,
  Command,
  GraphInterrupt,
  isCommand,
  interrupt,
  isGraphInterrupt,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type {
  ToolRuntime,
  StructuredToolInterface,
} from '@langchain/core/tools';
import type { LangGraphRunnableConfig } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  ToolOutputResolveView,
  ToolOutputReferenceState,
  PreResolvedArgsMap,
  ResolvedArgsByCallId,
  ResolveResult,
  ResolveOptions,
} from '@/tools/toolOutputReferences';
import type {
  ReplayableSubagentTool,
  SubagentResumeManifest,
  SubagentToolNodeResumeState,
} from '@/tools/subagent/SubagentReplay';
import type {
  HookRegistry,
  AggregatedHookResult,
  PostToolBatchEntry,
  ToolApprovalReplayKey,
} from '@/hooks';
import type { SettledToolBatchResult } from '@/tools/toolBatchReplay';
import type { PreparedSubagents } from '@/tools/preparedSubagents';
import type { RunBreakerScope } from '@/llm/streamLimits';
import type * as t from '@/types';
import {
  TOOL_BATCH_REPLAY_KEY,
  getToolBatchReplayOwner,
  getToolBatchReplayScope,
  attachToolBatchReplayState,
  restoreToolBatchReplayState,
  getToolBatchReplayState,
  clearStaleToolApprovalConfig,
  isChildToolReplayOwner,
  getToolReplayResumeStatus,
} from '@/tools/toolBatchReplay';
import {
  type CallerCapabilityProjection,
  createCallerCapabilityProjectionSnapshot,
  isToolDefinitionActive,
  isProgrammaticControlTool,
  mergeCallerCapabilityDefinitions,
  resolveCallerCapabilityProjection,
} from '@/tools/CallerCapabilities';
import {
  cloneToolMessageWithContent,
  compactToolContent,
  hasComputerCallOutputMarker,
  isComputerCallOutputContent,
  isComputerCallOutputMessage,
  serializeStructuredValueBounded,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  getReviewedToolApproval,
  getToolApprovalReviewEvidence,
  toolApprovalPayloadMatches,
  toolApprovalProposalMatches,
  TOOL_APPROVAL_REVIEW_CONFIG_KEY,
} from '@/hitl/approvalReview';
import {
  buildToolExecutionRequestPlan,
  coerceArgsForSchema,
  coerceRecordArgs,
  resolveRuntimeSessionHint,
  recordArgsEqual,
  stableStringify,
} from '@/tools/eagerEventExecution';
import {
  attachSubagentResumeManifest,
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_REPLAY_CONTROLLER,
} from '@/tools/subagent/SubagentReplay';
import {
  INTENT_ARG,
  readOutcomeFields,
  resolveToolOutcome,
  isIntentLabelProperty,
  outcomeFieldsFromResult,
} from '@/tools/intentArg';
import {
  resolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  buildReferenceKey,
  ToolOutputReferenceRegistry,
} from '@/tools/toolOutputReferences';
import {
  calculateMaxToolResultChars,
  truncateToolResultContent,
} from '@/utils/truncation';
import {
  StreamLimitExceededError,
  RUN_BREAKER_SCOPE_CONFIG_KEY,
} from '@/llm/streamLimits';
import {
  resolveLocalToolRegistry,
  resolveLocalExecutionTools,
} from '@/tools/local';
import { stripCodeSessionFileSummary } from '@/tools/CodeSessionFileSummary';
import { Constants, GraphEvents, CODE_EXECUTION_TOOLS } from '@/common';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { attachRunStepResumeState } from '@/tools/runStepResume';

function stripToolApprovalReviewConfig(
  configurable: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  if (
    configurable == null ||
    (!(TOOL_APPROVAL_REVIEW_CONFIG_KEY in configurable) &&
      !(TOOL_BATCH_REPLAY_KEY in configurable))
  ) {
    return configurable;
  }
  const {
    [TOOL_APPROVAL_REVIEW_CONFIG_KEY]: _review,
    [TOOL_BATCH_REPLAY_KEY]: _replay,
    ...rest
  } = configurable;
  return rest;
}

/** Host-facing requests must not carry SDK-private execution state — hosts
 * spread `configurable` into their own run configs. */
function stripHostExecutionConfig(
  configurable: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  const withoutReview = stripToolApprovalReviewConfig(configurable);
  if (
    withoutReview == null ||
    !(RUN_BREAKER_SCOPE_CONFIG_KEY in withoutReview)
  ) {
    return withoutReview;
  }
  const { [RUN_BREAKER_SCOPE_CONFIG_KEY]: _scope, ...rest } = withoutReview;
  return rest;
}
import { convertInjectedMessages } from '@/messages/injected';
import { stampSyntheticProviderMessage } from '@/messages/provenance';
import { safeDispatchCustomEvent } from '@/utils/events';
import { RunnableCallable, composeAbortSignals } from '@/utils';
import {
  executeHooks,
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
} from '@/hooks';

function createToolApprovalReplayKey(
  config: RunnableConfig,
  agentId: string,
  toolUseId: string
): ToolApprovalReplayKey {
  const configuredScope =
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY];
  const threadId = config.configurable?.thread_id;
  let executionScope = '';
  if (typeof configuredScope === 'string') {
    executionScope = configuredScope;
  } else if (typeof threadId === 'string') {
    executionScope = threadId;
  }
  return {
    executionScope,
    agentId,
    toolUseId,
  };
}

function getToolApprovalReplaySessionId(
  config: RunnableConfig,
  hookSessionId: string
): string {
  const configuredScope =
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY];
  return typeof configuredScope === 'string' && configuredScope.length > 0
    ? configuredScope
    : hookSessionId;
}

/**
 * Per-call batch context for `runTool`. Bundles every optional
 * batch-scoped value the method needs so the signature stays at
 * three positional parameters even as new context fields are added.
 */
type RunToolBatchContext<T = unknown> = {
  /** Position of this call within the parent ToolNode batch. */
  batchIndex?: number;
  /** Batch turn shared across every call in the batch. */
  turn?: number;
  /** Registry partition scope (run id or anonymous batch id). */
  batchScopeId?: string;
  /** Batch-local sink for post-substitution args. */
  resolvedArgsByCallId?: ResolvedArgsByCallId;
  /** File identities actually injected into each direct code call. */
  codeSessionBaselineByCallId?: Map<string, ReadonlyMap<string, string>>;
  /**
   * Frozen pre-batch view of the tool-output registry. When supplied,
   * `runTool` resolves `{{tool…turn…}}` placeholders against this
   * snapshot instead of the live registry, so a slow `PreToolUse`
   * hook on one direct call cannot cause a sibling's just-registered
   * output to leak into this call's args mid-batch (Codex P1 #18 —
   * `Promise.all`-induced ordering would otherwise be observable).
   */
  preBatchSnapshot?: ToolOutputResolveView;
  /**
   * Pre-incremented per-tool usage counter. Set by
   * `runDirectToolWithLifecycleHooks` so PreToolUse hooks observe
   * the same `turn` the tool will actually execute under (Codex P2
   * #27 — without this, parallel direct calls of the same tool in
   * one Promise.all batch all read `turn=N` for the hook but
   * actually executed as `turn=N`, `N+1`, `N+2`). When supplied,
   * `runTool` skips its own counter increment.
   */
  usageCount?: number;
  /**
   * Per-batch sink for `additionalContext` strings returned by
   * direct-path PreToolUse / PostToolUse / PostToolUseFailure hooks.
   * The caller in `run()` materializes the accumulated strings as a
   * `HumanMessage` appended to outputs so the next model turn sees
   * them — same shape as the event-driven path's `injected[]`.
   * Codex P2 #39: pre-fix the direct path called `executeHooks` and
   * discarded `additionalContexts`, silently breaking the hook API
   * contract for hosts relying on it for policy / recovery guidance.
   */
  additionalContextsSink?: string[];
  /** Stable identity of the assistant tool-call batch across HITL replay. */
  replayBatchKey?: string;
  /** Positions in the original proposal, before completed calls are filtered. */
  replayCallPositions?: ReadonlyMap<ToolCall, number>;
  /** Tool-call identities owned by this ToolNode invocation. */
  approvalScopeCallIds?: ReadonlySet<string>;
  /**
   * Graph state the ToolNode was invoked with, threaded from `run()`
   * so `tool.invoke` can forward it as langgraph 1.4's `runtime.state`
   * (the deprecation-free replacement for `getCurrentTaskInput()`,
   * which relies on `node:async_hooks` and is browser-incompatible).
   */
  runInput?: T;
  /** Batch-local error-completion ownership (see {@link ToolErrorOwnership}). */
  errorOwnership?: ToolErrorOwnership;
};

function withSubagentReplayBatch(
  config: RunnableConfig,
  replayBatchKey: string | undefined
): RunnableConfig {
  if (replayBatchKey == null) {
    return config;
  }
  return {
    ...config,
    configurable: {
      ...config.configurable,
      [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: replayBatchKey,
    },
  };
}

type SettledDirectToolResult = SettledToolBatchResult;

/** Positional identity is safe within the immutable full-proposal batch key. */
function directReplayKey(call: ToolCall, batchIndex: number): string {
  return call.id == null || call.id === ''
    ? JSON.stringify(['position', batchIndex])
    : JSON.stringify(['id', call.id]);
}

/**
 * Batch-local record of who owns each failed call's completion event.
 *
 * Kept per invocation rather than on the instance: tool-call ids are
 * provider-scoped (and synthetic ids can repeat), so concurrent `run()`s on
 * one ToolNode would otherwise share and cross-consume these markers — one
 * invocation's thrown-error marker suppressing another's only completion,
 * or leaving a stale marker behind when an interrupt aborts a batch before
 * the output loop reads it.
 *
 * - `handlerOwned`: the errorHandler ran and dispatched (or threw — a throw
 *   is not proof it didn't dispatch). The output loop must skip these.
 * - `undispatched`: the handler explicitly reported it could NOT dispatch,
 *   so the output loop owns the completion instead.
 *
 * A call in NEITHER set returned an error `ToolMessage` without ever
 * entering the catch path, so the output loop owns it too.
 */
export type ToolErrorOwnership = {
  handlerOwned: Set<string>;
  undispatched: Set<string>;
};

export function createToolErrorOwnership(): ToolErrorOwnership {
  return { handlerOwned: new Set(), undispatched: new Set() };
}

type BoundedToolOutput = {
  content: string;
  registryContent: string;
};

/**
 * Produces the provider preview and, when requested, the exact registry prefix
 * in one bounded traversal. String outputs already exist in memory when a tool
 * returns them; structured outputs must never be fully JSON-materialized before
 * either limit applies.
 */
function serializeToolOutputWithinLimits(
  output: unknown,
  maxToolResultChars: number,
  registryPrefixChars = 0
): BoundedToolOutput {
  if (typeof output === 'string') {
    return {
      content: truncateToolResultContent(output, maxToolResultChars),
      registryContent: registryPrefixChars > 0 ? output : '',
    };
  }

  const serialized = serializeStructuredValueBounded(
    output,
    maxToolResultChars,
    registryPrefixChars
  );
  return {
    content: serialized.content,
    registryContent: serialized.prefix,
  };
}

const TOOL_NODE_RUN_NAME = 'tool_batch';
const NANOID_URL_ALPHABET =
  '_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
const RUNTIME_HANDOFF_GROUP_OFFSET = 2 ** 48;

/**
 * Per-batch context for `dispatchToolEvents` / `executeViaEvent`.
 * Mirrors {@link RunToolBatchContext} for the event-driven path,
 * with bulk indices and the snapshot/pre-resolved-args carriers
 * used in the mixed direct+event flow.
 */
type DispatchBatchContext = {
  /** Per-call batch indices, parallel to the `toolCalls` array. */
  batchIndices?: number[];
  /** Batch turn shared across every call in the batch. */
  turn?: number;
  /** Registry partition scope (run id or anonymous batch id). */
  batchScopeId?: string;
  /**
   * Pre-resolved args keyed by `toolCallId`. Populated by the mixed
   * path so event calls don't re-resolve against a registry that
   * already contains same-turn direct outputs.
   */
  preResolvedArgs?: PreResolvedArgsMap;
  /**
   * Frozen pre-batch registry view used to re-resolve placeholders
   * a `PreToolUse` hook injects via `updatedInput` — preserves the
   * same-turn isolation guarantee for hook-rewritten args.
   */
  preBatchSnapshot?: ToolOutputResolveView;
};

/**
 * Helper to check if a value is a Send object
 */
function isSend(value: unknown): value is Send {
  return value instanceof Send;
}

function isHandoffToolName(name: string): boolean {
  return name.startsWith(Constants.LC_TRANSFER_TO_);
}

/**
 * Encodes 48 random bits from the persisted batch key into a safe integer.
 * The high offset keeps runtime groups disjoint from low, structural group IDs.
 */
function getRuntimeHandoffGroupId(batch: string): number {
  let value = 0;
  for (const char of batch.slice(0, 8)) {
    const digit = NANOID_URL_ALPHABET.indexOf(char);
    value = value * 64 + Math.max(digit, 0);
  }
  return RUNTIME_HANDOFF_GROUP_OFFSET + value;
}

function findHandoffMessage(
  messages: BaseMessage[],
  destination: string
): ToolMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.getType() !== 'tool') {
      continue;
    }
    const toolMessage = message as ToolMessage;
    const isStandardHandoff =
      toolMessage.name === `${Constants.LC_TRANSFER_TO_}${destination}`;
    const isConditionalHandoff =
      toolMessage.name === 'conditional_transfer' &&
      toolMessage.additional_kwargs.handoff_destination === destination;
    if (isStandardHandoff || isConditionalHandoff) {
      return toolMessage;
    }
  }
  return undefined;
}

/**
 * Format a fail-closed diagnostic for malformed approval-decision
 * fields. Hosts deserialize resume payloads from untyped JSON, so
 * `responseText` and `updatedInput` can land here as anything; the
 * blocking ToolMessage carries this string so the host can debug the
 * exact wire shape that was rejected.
 */
function describeOfferedShape(value: unknown): string {
  if (value === undefined) {
    return '<missing>';
  }
  if (value === null) {
    return 'null';
  }
  if (Array.isArray(value)) {
    return 'array';
  }
  return typeof value;
}

type AssistantBatch = {
  message: AIMessage;
};

function findAssistantBatch(
  messages: BaseMessage[]
): AssistantBatch | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (isAIMessage(message)) {
      return { message };
    }
  }
  return undefined;
}

function getAssistantBatchReplayKey(
  batch: AssistantBatch,
  threadId: string | undefined
): string {
  return JSON.stringify([
    threadId ?? null,
    batch.message.id == null || batch.message.id === ''
      ? null
      : batch.message.id,
    createHash('sha256')
      .update(stableStringify(batch.message.tool_calls ?? []))
      .digest('hex'),
  ]);
}

/**
 * Per-entry record collected during PreToolUse hook handling for tool
 * calls that need human approval. Carries everything
 * `buildToolApprovalInterruptPayload` needs to assemble the interrupt
 * payload, plus the per-tool decision allowlist if the hook supplied
 * one. Defined at module scope so the payload-builder helper can be
 * extracted out of `dispatchToolEvents` without leaking the locally-
 * inferred shape.
 */
type AskEntry = {
  entry: {
    call: ToolCall;
    args: Record<string, unknown>;
    stepId: string;
  };
  reason?: string;
  allowedDecisions?: ReadonlyArray<'approve' | 'reject' | 'edit' | 'respond'>;
};

/**
 * Build the `tool_approval` interrupt payload from the set of pending
 * `ask`-decision entries collected during PreToolUse hook handling.
 * Pure function — doesn't touch ToolNode state — so it lives at module
 * scope. The interrupt itself is raised by the caller (which still
 * needs `interrupt()` plus the AsyncLocalStorage anchoring shim).
 */
function buildToolApprovalInterruptPayload(
  askEntries: ReadonlyArray<AskEntry>,
  hookSessionId?: string
): t.ToolApprovalInterruptPayload {
  return {
    type: 'tool_approval',
    ...(hookSessionId == null || hookSessionId === ''
      ? {}
      : { hook_session_id: hookSessionId }),
    action_requests: askEntries.map(({ entry, reason }) => {
      const request: t.ToolApprovalRequest = {
        tool_call_id: entry.call.id!,
        name: entry.call.name,
        arguments: entry.args,
      };
      if (reason != null) {
        request.description = reason;
      }
      return request;
    }),
    review_configs: askEntries.map(({ entry, allowedDecisions }) => ({
      action_name: entry.call.name,
      tool_call_id: entry.call.id!,
      allowed_decisions: (allowedDecisions ?? [
        'approve',
        'reject',
        'edit',
        'respond',
      ]) as t.ToolApprovalDecisionType[],
    })),
  };
}

/**
 * Build a `tool_call_id → ToolApprovalDecision` map from the host's
 * resume value. Hosts may return decisions either as an array (one per
 * action_request, in order) or as a record keyed by `tool_call_id`. Any
 * unrecognized shape (or a decision missing for a given call id) is
 * treated as "no decision" by callers — typically rejected so the run
 * doesn't silently invoke a tool the human never approved.
 */
function normalizeApprovalDecisions(
  callIds: string[],
  resumeValue: t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap | undefined
): Map<string, t.ToolApprovalDecision> {
  const map = new Map<string, t.ToolApprovalDecision>();
  if (resumeValue == null) {
    return map;
  }
  if (Array.isArray(resumeValue)) {
    const limit = Math.min(callIds.length, resumeValue.length);
    for (let i = 0; i < limit; i++) {
      map.set(callIds[i], resumeValue[i]);
    }
    return map;
  }
  if (typeof resumeValue === 'object') {
    for (const callId of callIds) {
      const decision = (resumeValue as Partial<t.ToolApprovalDecisionMap>)[
        callId
      ];
      if (decision !== undefined) {
        map.set(callId, decision);
      }
    }
  }
  return map;
}

/**
 * Merges code execution session context into the sessions map.
 *
 * The codeapi worker reports two distinct ids on a code-execution result:
 *  - `artifact.session_id` (the `sessionId` arg here) is the EXEC session
 *    — the sandbox VM that ran the code. It's transient and torn down
 *    post-execution; subsequent calls cannot reuse it as a sandbox.
 *  - `file.storage_session_id` on each `artifact.files[i]` is the STORAGE
 *    session — the file-server bucket prefix where the artifact actually
 *    lives and is served from.
 *
 * Per-file `storage_session_id` is preserved (not overwritten with the
 * exec id) because `_injected_files` are looked up against the
 * file-server's storage path on subsequent tool calls. Stomping the
 * storage id with the exec id silently 404s every follow-up tool call
 * within the same run — `cat /mnt/data/foo.txt` reports "No such file
 * or directory" because the worker can't mount a file at a path the
 * storage doesn't know about. Fall back to the exec id only when the
 * per-file id is absent (e.g. inline `content` files have no persistent
 * storage location).
 */
/**
 * Builds a `CodeEnvFile` ref from an arbitrary `FileRef`-like input,
 * narrowing onto the discriminated union: `kind: 'skill'` requires
 * `version`, other kinds forbid it.
 *
 * Defaults `kind` to `'user'` when unset — most ad-hoc files are
 * user-private; shared resources (skills/agents) populate their kind
 * upstream. A skill ref missing `version` falls back to `'user'` so
 * the upstream contract bug surfaces as a degraded sessionKey rather
 * than a runtime crash; primeSkillFiles is the only writer, and it
 * always sets `version` — see LC packages/api/src/agents/skillFiles.ts.
 *
 * `resource_id` carries the entity-that-owns-this-file's-session
 * identity (skill `_id` etc.); falls back to `id` (the storage
 * file_id) for inputs that haven't been updated to send the field
 * explicitly. The fallback degrades sessionKey resolution on the
 * codeapi side for shared kinds (it'll match the storage nanoid
 * against a skill _id and 403) — but won't crash, so an unmigrated
 * client still produces a diagnosable error instead of a stack
 * trace.
 */
function toInjectedFileRef(
  file: {
    id: string;
    resource_id?: string;
    name: string;
    storage_session_id?: string;
    kind?: t.CodeEnvKind;
    version?: number;
  },
  execSessionId: string
): t.CodeEnvFile {
  const base = {
    id: file.id,
    resource_id: file.resource_id ?? file.id,
    name: file.name,
    /* Inline `content` files have no persistent storage location;
     * fall back to the execution session id for those entries. */
    storage_session_id: file.storage_session_id ?? execSessionId,
  };
  const kind = file.kind ?? 'user';
  if (kind === 'skill' && file.version != null) {
    return { ...base, kind: 'skill', version: file.version };
  }
  if (kind === 'agent') {
    return { ...base, kind: 'agent' };
  }
  return { ...base, kind: 'user' };
}

/* Stable file identity = `(storage_session_id, id)`. Same name in
 * different storage sessions are distinct files. */
function fileIdentityKey(
  file: {
    storage_session_id?: string;
    id: string;
  },
  fallbackSessionId = ''
): string {
  return `${file.storage_session_id ?? fallbackSessionId}\0${file.id}`;
}

function codeSessionIdentityByName(
  context:
    | t.ToolCallRequest['codeSessionContext']
    | t.CodeSessionContext
    | undefined
): ReadonlyMap<string, string> {
  return new Map(
    (context?.files ?? []).map((file) => [
      file.name,
      fileIdentityKey(file, context?.session_id),
    ])
  );
}

function storageSessionIdFromIdentity(
  identity: string | undefined,
  fileId: string
): string | undefined {
  const suffix = `\0${fileId}`;
  return identity != null && identity.endsWith(suffix)
    ? identity.slice(0, identity.length - suffix.length)
    : undefined;
}

function updateCodeSession(
  sessions: t.ToolSessionMap,
  sessionKey: string,
  execSessionId: string,
  files: t.FileRefs | undefined,
  deletedFiles: readonly string[] | undefined,
  baselineIdentityByName: ReadonlyMap<string, string> | undefined
): void {
  const newFiles = files ?? [];
  const existingSession = sessions.get(sessionKey) as
    | t.CodeSessionContext
    | undefined;
  const existingSessionId = existingSession?.session_id ?? '';
  const existingFiles = (existingSession?.files ?? []).map((file) =>
    file.storage_session_id == null
      ? { ...file, storage_session_id: existingSessionId }
      : file
  );
  const deletedIdentityByName = new Map<string, string>();
  for (const name of deletedFiles ?? []) {
    const identity = baselineIdentityByName?.get(name);
    if (identity != null) {
      deletedIdentityByName.set(name, identity);
    }
  }

  /* Absence from `files` is not a deletion signal: older Code API versions
   * and truncated responses can omit otherwise-live refs. Remove only paths
   * explicitly attested by `deleted_files`, and only while the current ref
   * still has the identity injected into that request. */
  if (newFiles.length === 0) {
    sessions.set(sessionKey, {
      session_id: execSessionId,
      files: existingFiles.filter(
        (file) =>
          deletedIdentityByName.get(file.name) !==
          fileIdentityKey(file, existingSessionId)
      ),
      lastUpdated: Date.now(),
    });
    return;
  }

  /* Worker echoes lack ownership identity (kind/resource_id/version) —
   * sandbox doesn't re-attest; that's signed at upload. Merge by
   * (storage_session_id, id) so prior identity survives the echo. */
  const filesWithSession: t.FileRefs = [];
  const newFileNames = new Set<string>();
  const incomingByIdentity = new Map<string, number>();
  const existingIdentities = new Set(
    existingFiles.map((file) => fileIdentityKey(file, existingSessionId))
  );
  for (const file of newFiles) {
    const baselineIdentity = baselineIdentityByName?.get(file.name);
    const withSession = {
      ...file,
      storage_session_id:
        file.storage_session_id ??
        storageSessionIdFromIdentity(baselineIdentity, file.id) ??
        execSessionId,
    };
    const identity = fileIdentityKey(withSession);
    const wasTrackedByRequest = baselineIdentity === identity;
    /* An inherited echo is evidence that a request observed an existing ref,
     * not authority to resurrect one removed by a concurrent sibling. Fresh
     * and modified outputs remain eligible to replace by path below. */
    if (
      file.inherited === true &&
      wasTrackedByRequest &&
      (!existingIdentities.has(identity) ||
        deletedIdentityByName.get(file.name) === identity)
    ) {
      continue;
    }
    incomingByIdentity.set(identity, filesWithSession.length);
    newFileNames.add(withSession.name);
    filesWithSession.push(withSession);
  }

  const filteredExisting: t.FileRefs = [];
  for (const e of existingFiles) {
    const idx = incomingByIdentity.get(fileIdentityKey(e, existingSessionId));
    if (idx !== undefined) {
      filesWithSession[idx] = { ...e, ...filesWithSession[idx] };
    }
    const explicitlyDeleted =
      deletedIdentityByName.get(e.name) ===
      fileIdentityKey(e, existingSessionId);
    if (!newFileNames.has(e.name) && !explicitlyDeleted) {
      filteredExisting.push(e);
    }
  }

  sessions.set(sessionKey, {
    session_id: execSessionId,
    files: [...filteredExisting, ...filesWithSession],
    lastUpdated: Date.now(),
  });
}

/** Retains host-provisioned inputs even when execution returns no usable artifact. */
function retainCodeSessionInputs(
  sessions: t.ToolSessionMap,
  sessionKey: string,
  context: t.ToolCallRequest['codeSessionContext']
): void {
  if (
    context?.session_id == null ||
    context.session_id === '' ||
    context.files == null
  )
    return;
  const existing = sessions.get(sessionKey) as t.CodeSessionContext | undefined;
  const existingFiles = existing?.files ?? [];
  const files = [...existingFiles];
  const indexByIdentity = new Map<string, number>();
  const indexByName = new Map<string, number>();
  for (let i = 0; i < files.length; i++) {
    indexByIdentity.set(fileIdentityKey(files[i]), i);
    indexByName.set(files[i].name, i);
  }
  let changed = existing?.session_id !== context.session_id;
  for (const file of context.files) {
    if (!file.id || !file.name || !file.storage_session_id) continue;
    const identity = fileIdentityKey(file);
    if (indexByIdentity.has(identity)) continue;
    const replacementIndex = indexByName.get(file.name);
    if (replacementIndex !== undefined) {
      indexByIdentity.delete(fileIdentityKey(files[replacementIndex]));
      files[replacementIndex] = { ...file };
      indexByIdentity.set(identity, replacementIndex);
      changed = true;
      continue;
    }
    indexByIdentity.set(identity, files.length);
    indexByName.set(file.name, files.length);
    files.push({ ...file });
    changed = true;
  }
  if (!changed) return;
  sessions.set(sessionKey, {
    session_id: context.session_id,
    files,
    lastUpdated: Date.now(),
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export class ToolNode<T = any> extends RunnableCallable<T, T> {
  private toolMap: Map<string, StructuredToolInterface | RunnableToolLike>;
  private loadRuntimeTools?: t.ToolRefGenerator;
  handleToolErrors = true;
  trace = false;
  private runLangfuse?: t.LangfuseConfig;
  private agentLangfuse?: t.LangfuseConfig;
  toolCallStepIds?: Map<string, string>;
  errorHandler?: t.ToolNodeConstructorParams['errorHandler'];
  /**
   * Fallback error-completion ownership for calls that reach `runTool`
   * outside a batch context (direct `runTool` use in tests / embedders).
   * Batch-scoped ownership is threaded via `RunToolBatchContext` instead —
   * see {@link ToolErrorOwnership} for why per-invocation scoping matters.
   */
  private looseErrorOwnership: ToolErrorOwnership = createToolErrorOwnership();
  private toolUsageCount: Map<string, number>;
  /** Maps toolCallId → turn captured in runTool, used by handleRunToolCompletions */
  private toolCallTurns: Map<string, number> = new Map();
  /**
   * `call.id → turn` map dedicated to the direct-path lifecycle so the
   * turn assigned on first entry is REUSED on LangGraph resume.
   * Distinct from `toolCallTurns` (which is cleared at the start of
   * every `run()` to keep per-batch event-dispatch metadata fresh) —
   * the direct path needs stability across re-entries triggered by
   * `interrupt()` resumes (Codex P2 #30). Cleared with the rest of
   * the per-Run state in `clearHeavyState`-equivalent flushes when
   * the Run ends.
   */
  private directPathTurns: Map<string, number> = new Map();
  /** Terminal direct results that must survive a LangGraph replay of the
   * containing ToolNode. Includes the sidecar data the fresh batch needs for
   * hook-context injection and completion-event deduplication. */
  private settledDirectResultsByBatch = new Map<
    string,
    Map<string, SettledDirectToolResult>
  >();
  /** Tool registry for filtering (lazy computation of programmatic maps) */
  private toolRegistry?: t.LCToolRegistry;
  /** Schema-only definitions used when event mode has no runtime registry. */
  private toolDefinitions?: t.LCToolRegistry;
  /** Tool-map entries created or replaced by the local execution resolver. */
  private localImplementationNames = new Set<string>();
  /** Reads deferred-tool discovery state from the owning agent context. */
  private getDiscoveredToolNames?: () => readonly string[];
  /** Reference to Graph's sessions map for automatic session injection */
  private sessions?: t.ToolSessionMap;
  /** Partition within `sessions` owned by this agent's execution profile. */
  private codeSessionKey: string;
  /** When true, dispatches ON_TOOL_EXECUTE events instead of invoking tools directly */
  private eventDrivenMode: boolean = false;
  /** Opt-in stream-layer prestart config for event-driven tools. */
  private eagerEventToolExecution?: t.EagerEventToolExecutionConfig;
  /** Host tools that write to the code sandbox and share its exec session. */
  private codeSessionToolNames?: ReadonlySet<string>;
  /** Shared per-run prestarted tool registry populated by ChatModelStreamHandler. */
  private eagerEventToolExecutions?: Map<string, t.EagerEventToolExecution>;
  /** Shared per-run per-tool turn counter used by eager and normal event dispatch. */
  private eagerEventToolUsageCount?: Map<string, number>;
  /**
   * Shared per-run eager prestart circuit breaker. Tool names added here
   * (when a prestarted execution's args mismatch the final request) are no
   * longer prestarted by the stream handler for the rest of the run.
   */
  private eagerEventToolSuppressions?: Set<string>;
  /** Agent ID for event-driven mode */
  private agentId?: string;
  /**
   * ID of the agent that owns this tool node, whenever the graph knows it
   * (including top-level agents in a multi-agent graph). Surfaced to hooks as
   * `executingAgentId` so they can attribute a tool batch to a specific agent
   * even where `agentId` (the subagent-scope marker) is undefined.
   */
  private executingAgentId?: string;
  private readonly executionContext?: t.ToolNodeOptions['executionContext'];
  private executingAgentName?: string;
  private rootAgentId?: string;
  private rootAgentName?: string;
  /** Tool names that bypass event dispatch and execute directly (e.g., graph-managed handoff tools) */
  private directToolNames?: Set<string>;
  /**
   * Tool names whose in-process body may raise a LangGraph `interrupt()`
   * mid-execution (e.g. `ask_user_question`). Used only to REORDER within
   * the direct group: a tool named here that is *already* direct (a real
   * in-process graphTool — the only kind whose body can reach
   * `interrupt()`) is scheduled ahead of its non-interrupting direct
   * siblings, so a mid-body interrupt unwinds the ToolNode before a
   * non-idempotent sibling executes and LangGraph's resume-time batch
   * re-execution can't double it. This set is deliberately NOT folded
   * into direct classification: a name that resolves to a schema-only
   * event stub (an inherited `toolDefinition` with no executable
   * instance) stays event-dispatched — forcing it direct would invoke
   * the stub, which throws. See {@link t.ToolNodeOptions.interruptingToolNames}.
   */
  private interruptingToolNames?: Set<string>;
  /**
   * File checkpointer extracted from the local coding tool bundle when
   * `toolExecution.local.fileCheckpointing === true`. Exposed via
   * {@link getFileCheckpointer}. Undefined when checkpointing is off
   * or the local coding suite isn't bound to this node.
   */
  private fileCheckpointer?: t.LocalFileCheckpointer;
  /** Maximum characters allowed in a single tool result before truncation. */
  private maxToolResultChars: number;
  /** Hook registry for PreToolUse/PostToolUse lifecycle hooks */
  private hookRegistry?: HookRegistry;
  /**
   * Run-scoped HITL config. When `enabled`, `ask` decisions from
   * PreToolUse hooks raise a LangGraph `interrupt()` instead of being
   * treated as fail-closed denies.
   */
  private humanInTheLoop?: t.HumanInTheLoopConfig;
  /**
   * Registry of tool outputs keyed by `tool<idx>turn<turn>`.
   *
   * Populated only when `toolOutputReferences.enabled` is true. The
   * registry owns the run-scoped state (turn counter, last-seen runId,
   * warn-once memo, stored outputs), so sharing a single instance
   * across multiple ToolNodes in a run lets cross-agent `{{…}}`
   * references resolve — which is why multi-agent graphs pass the
   * *same* instance to every ToolNode they compile rather than each
   * ToolNode building its own.
   */
  private toolOutputRegistry?: ToolOutputReferenceRegistry;
  /** Run-scoped selection for swapping remote code tools to local executors. */
  private toolExecution?: t.ToolExecutionConfig;
  /** Owning graph's run-scoped breaker signal, composed into each batch's config. */
  private getBreakerSignal?: () => AbortSignal | undefined;
  /** Owning graph's immutable run scope; captured once per batch. */
  private getRunScope?: () => RunBreakerScope;
  private preparedSubagents?: PreparedSubagents;
  /**
   * Monotonic counter used to mint a unique scope id for anonymous
   * batches (ones invoked without a `run_id` in
   * `config.configurable`). Each such batch gets its own registry
   * partition so concurrent anonymous invocations can't delete each
   * other's in-flight state.
   */
  private anonBatchCounter: number = 0;
  private handoffRouting?: t.ToolNodeOptions['handoffRouting'];

  constructor({
    handoffRouting,
    tools,
    toolMap,
    name,
    tags,
    trace,
    runLangfuse,
    agentLangfuse,
    errorHandler,
    toolCallStepIds,
    handleToolErrors,
    loadRuntimeTools,
    toolRegistry,
    toolDefinitions,
    getDiscoveredToolNames,
    sessions,
    codeSessionKey,
    eventDrivenMode,
    eagerEventToolExecution,
    eagerEventToolExecutions,
    eagerEventToolUsageCount,
    eagerEventToolSuppressions,
    agentId,
    executingAgentId,
    executionContext,
    executingAgentName,
    rootAgentId,
    rootAgentName,
    directToolNames,
    interruptingToolNames,
    codeSessionToolNames,
    maxContextTokens,
    maxToolResultChars,
    hookRegistry,
    humanInTheLoop,
    toolOutputReferences,
    toolOutputRegistry,
    toolExecution,
    fileCheckpointer,
    getBreakerSignal,
    getRunScope,
    preparedSubagents,
    restoreRunStepResumeState,
    createRunStepResumeState,
    onToolCallsClaimed,
  }: t.ToolNodeConstructorParams) {
    super({
      name: name ?? TOOL_NODE_RUN_NAME,
      tags,
      func: async (input, config) => {
        const messages = Array.isArray(input)
          ? (input as BaseMessage[])
          : (input as { messages?: BaseMessage[] }).messages;
        const assistantBatch =
          messages == null ? undefined : findAssistantBatch(messages);
        const activeReplayKey =
          assistantBatch == null || getToolBatchReplayScope(config) == null
            ? undefined
            : getAssistantBatchReplayKey(
              assistantBatch,
              getToolBatchReplayScope(config)
            );
        const replayOwner = getToolBatchReplayOwner(
          config,
          this.executingAgentId ?? this.agentId ?? ''
        );
        let reviewEvidence = getToolApprovalReviewEvidence(config);
        const replayState = getToolBatchReplayState(config.configurable);
        const ownsParentBatch =
          replayState?.records.some(
            (record) =>
              record.owner === replayOwner && record.batch === activeReplayKey
          ) === true;
        let isChildReplay = false;
        if (ownsParentBatch) {
          const calls = this.isSendInput(input)
            ? [input.lg_tool_call]
            : (assistantBatch?.message.tool_calls ?? []);
          const trustedParentCallIds = new Set(
            calls
              .filter(
                (call) =>
                  call.name === Constants.SUBAGENT &&
                  (
                    this.toolMap.get(call.name) as
                      | ReplayableSubagentTool
                      | undefined
                  )?.[SUBAGENT_REPLAY_CONTROLLER] != null
              )
              .map((call) => call.id ?? '')
          );
          isChildReplay =
            reviewEvidence?.owner != null
              ? isChildToolReplayOwner(
                config,
                reviewEvidence.owner,
                trustedParentCallIds
              )
              : replayState.records.some((record) =>
                isChildToolReplayOwner(
                  config,
                  record.owner,
                  trustedParentCallIds
                )
              );
        }
        const hasApprovalAuthorization =
          reviewEvidence?.owner != null || replayState?.approvalOwner != null;
        const replayResumeStatus =
          hasApprovalAuthorization || replayState != null
            ? getToolReplayResumeStatus(
              config,
              reviewEvidence?.interruptId ?? replayState?.interruptId,
              isChildReplay
            )
            : undefined;
        if (replayResumeStatus === 'unverifiable' && hasApprovalAuthorization) {
          throw new Error(
            'Cannot verify the active tool approval resume — failing closed'
          );
        }
        if (replayResumeStatus === 'stale') {
          const configurable = { ...config.configurable };
          clearStaleToolApprovalConfig(
            configurable,
            replayOwner,
            activeReplayKey
          );
          config = {
            ...config,
            configurable,
          };
          reviewEvidence = undefined;
        }
        if (
          reviewEvidence?.owner != null &&
          reviewEvidence.owner !== replayOwner &&
          !isChildReplay
        ) {
          throw new Error(
            'Tool approval execution owner changed before resume — failing closed'
          );
        }
        const referenceReplay: { state?: ToolOutputReferenceState } = {};
        const restoredBatches = await restoreToolBatchReplayState(
          config,
          replayOwner
        );
        if (
          restoredBatches.length > 0 &&
          !restoredBatches.some(({ batch }) => batch === activeReplayKey)
        ) {
          throw new Error(
            'Tool batch proposal changed before approval replay — failing closed'
          );
        }
        if (
          activeReplayKey != null &&
          getToolBatchReplayState(config.configurable) != null
        ) {
          this.settledDirectResultsByBatch.delete(activeReplayKey);
        }
        for (const {
          batch,
          results,
          turnState,
          referenceState,
        } of restoredBatches) {
          if (batch !== activeReplayKey) {
            continue;
          }
          this.settledDirectResultsByBatch.set(batch, new Map(results));
          if (turnState != null) {
            this.restoreSubagentResumeState(turnState);
          }
          referenceReplay.state = referenceState;
        }
        const state = input as T & Pick<t.BaseGraphState, 'runStepState'>;
        restoreRunStepResumeState?.(state.runStepState, config);
        // Freeze replay authority before observers can yield to sibling tasks.
        if (onToolCallsClaimed != null && assistantBatch?.message.id != null) {
          await onToolCallsClaimed(assistantBatch.message.id, config);
        }
        let result: T;
        try {
          result = await this.run(input, config, referenceReplay);
        } catch (error) {
          if (!isGraphInterrupt(error)) {
            throw error;
          }
          const resumeState = createRunStepResumeState?.();
          const activeResults =
            activeReplayKey == null
              ? undefined
              : this.settledDirectResultsByBatch.get(activeReplayKey);
          const activeBatches =
            activeReplayKey == null
              ? new Map<string, Map<string, SettledDirectToolResult>>()
              : new Map([
                [
                  activeReplayKey,
                  activeResults ?? new Map<string, SettledDirectToolResult>(),
                ],
              ]);
          const turnState = this.createSubagentResumeState();
          const activeCallIds = new Set(
            assistantBatch?.message.tool_calls?.map((call) => call.id)
          );
          turnState.directPathTurns = turnState.directPathTurns.filter(
            ({ toolCallId }) => activeCallIds.has(toolCallId)
          );
          throw new GraphInterrupt(
            await Promise.all(
              error.interrupts.map(async (pendingInterrupt) => {
                const value = await attachToolBatchReplayState(
                  pendingInterrupt.value,
                  replayOwner,
                  activeBatches,
                  turnState,
                  referenceReplay.state
                );
                return {
                  ...pendingInterrupt,
                  value:
                    resumeState == null
                      ? value
                      : attachRunStepResumeState(value, resumeState),
                };
              })
            )
          );
        }
        if (createRunStepResumeState == null) {
          return result;
        }
        const runStepState = createRunStepResumeState();
        if (Array.isArray(result)) {
          return [...result, { runStepState }] as T;
        }
        return {
          ...result,
          runStepState,
        };
      },
    });
    this.trace = trace ?? this.trace;
    this.runLangfuse = runLangfuse;
    this.agentLangfuse = agentLangfuse;
    this.toolMap = toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
    this.handoffRouting = handoffRouting;
    this.toolCallStepIds = toolCallStepIds;
    this.handleToolErrors = handleToolErrors ?? this.handleToolErrors;
    this.loadRuntimeTools = loadRuntimeTools;
    this.errorHandler = errorHandler;
    this.toolUsageCount = new Map<string, number>();
    this.toolRegistry = resolveLocalToolRegistry({
      toolRegistry,
      toolExecution,
    });
    this.toolDefinitions = toolDefinitions;
    this.getDiscoveredToolNames = getDiscoveredToolNames;
    this.sessions = sessions;
    this.codeSessionKey = codeSessionKey ?? Constants.EXECUTE_CODE;
    this.eventDrivenMode = eventDrivenMode ?? false;
    this.eagerEventToolExecution = eagerEventToolExecution;
    this.eagerEventToolExecutions = eagerEventToolExecutions;
    this.eagerEventToolUsageCount = eagerEventToolUsageCount;
    this.eagerEventToolSuppressions = eagerEventToolSuppressions;
    this.agentId = agentId;
    // Default to agentId so callers constructing ToolNode directly (who pass the
    // existing agentId option) still get attribution without knowing the new option.
    this.executingAgentId = executingAgentId ?? agentId;
    this.executionContext = executionContext;
    this.executingAgentName = executingAgentName;
    this.rootAgentId = rootAgentId;
    this.rootAgentName = rootAgentName;
    this.directToolNames = directToolNames;
    this.interruptingToolNames =
      interruptingToolNames != null && interruptingToolNames.size > 0
        ? interruptingToolNames
        : undefined;
    this.codeSessionToolNames =
      codeSessionToolNames != null && codeSessionToolNames.length > 0
        ? new Set(codeSessionToolNames)
        : undefined;
    this.maxToolResultChars =
      maxToolResultChars ?? calculateMaxToolResultChars(maxContextTokens);
    this.hookRegistry = hookRegistry;
    this.humanInTheLoop = humanInTheLoop;
    this.toolExecution = toolExecution;
    this.getBreakerSignal = getBreakerSignal;
    this.getRunScope = getRunScope;
    this.preparedSubagents = preparedSubagents;
    // Caller-provided checkpointer wins. Graphs use this to share a
    // single per-Run instance across every ToolNode they compile so
    // `Run.rewindFiles()` reaches the same snapshot store regardless
    // of which agent's tool batch ran. Falls through to the bundle's
    // auto-created one when undefined (direct ToolNode construction).
    this.fileCheckpointer = fileCheckpointer;
    this.applyToolExecutionOverrides();
    /**
     * Precedence: an explicitly passed `toolOutputRegistry` instance
     * wins over a config object so a host (`Graph`) can share one
     * registry across many ToolNodes. When only the config is
     * provided (direct ToolNode usage), build a local registry so
     * the feature still works without graph-level plumbing. Registry
     * caps are intentionally decoupled from `maxToolResultChars`:
     * the registry stores the raw untruncated output so a later
     * `{{…}}` substitution pipes the full payload into the next
     * tool, even when the LLM saw a truncated preview.
     */
    if (toolOutputRegistry != null) {
      this.toolOutputRegistry = toolOutputRegistry;
    } else if (toolOutputReferences?.enabled === true) {
      this.toolOutputRegistry = new ToolOutputReferenceRegistry({
        maxOutputSize: toolOutputReferences.maxOutputSize,
        maxTotalSize: toolOutputReferences.maxTotalSize,
      });
    }
  }

  override async invoke(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: any,
    options?: Partial<RunnableConfig>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<any> {
    return this.withToolScope(options, (scoped) => super.invoke(input, scoped));
  }

  private withToolScope<R>(
    options: Partial<RunnableConfig> | undefined,
    invoke: (config: Partial<RunnableConfig> | undefined) => Promise<R>
  ): Promise<R> {
    // Explicit agent identity for tool callbacks: node-name parsing is
    // ambiguous when agent ids themselves embed node prefixes, so the
    // handler prefers this metadata (see `isForeignScope`).
    const scopedOptions =
      this.executingAgentId == null
        ? options
        : {
          ...options,
          configurable: {
            ...options?.configurable,
            executionContext: this.executionContext,
          },
          metadata: {
            ...options?.metadata,
            executionContext: this.executionContext,
            agentId: this.executingAgentId,
            activeAgentId: this.executingAgentId,
            ...(this.executingAgentName == null
              ? {}
              : { activeAgentName: this.executingAgentName }),
            ...(this.rootAgentId == null
              ? {}
              : { rootAgentId: this.rootAgentId }),
            ...(this.rootAgentName == null
              ? {}
              : { rootAgentName: this.rootAgentName }),
          },
        };
    return withLangfuseRuntimeScope(
      resolveLangfuseRuntimeScope({
        runLangfuse: this.runLangfuse,
        langfuseOverlay: this.agentLangfuse,
        // Run identity is inherited from the ambient stream scope (tool
        // supersteps execute on the owning run's chain); the agent identity
        // must be stamped here so a concurrent sibling agent's queued
        // callback cannot adopt this agent's overlay (see
        // `LangfuseRuntimeContext.agentId`).
        agentId: this.executingAgentId,
      }),
      () => invoke(scopedOptions)
    );
  }

  /**
   * Returns the run-scoped tool output registry, or `undefined` when
   * the feature is disabled.
   *
   * @internal Exposed for test observation only. Host code should rely
   * on `{{tool<i>turn<n>}}` substitution at tool-invocation time and
   * not mutate the registry directly.
   */
  public _unsafeGetToolOutputRegistry():
    | ToolOutputReferenceRegistry
    | undefined {
    return this.toolOutputRegistry;
  }

  /**
   * Replaces known remote Code API tools with local-process tools when
   * `RunConfig.toolExecution.engine === 'local'`. In event-driven mode those
   * names are also marked direct so the SDK executes them locally instead of
   * dispatching the batch to a host-side remote sandbox handler. When the
   * local coding suite is enabled, this also injects file/search/edit tools.
   */
  private applyToolExecutionOverrides(): void {
    const resolved = resolveLocalExecutionTools({
      toolMap: this.toolMap,
      toolExecution: this.toolExecution,
      toolRegistry: this.toolRegistry,
      fileCheckpointer: this.fileCheckpointer,
    });

    this.toolMap = resolved.toolMap;
    this.localImplementationNames = resolved.localImplementationNames;
    if (resolved.fileCheckpointer != null) {
      this.fileCheckpointer = resolved.fileCheckpointer;
    }
    if (resolved.directToolNames.size === 0) {
      return;
    }

    this.directToolNames = new Set([
      ...(this.directToolNames ?? new Set<string>()),
      ...resolved.directToolNames,
    ]);
  }

  /**
   * Returns the per-Run file checkpointer when
   * `toolExecution.local.fileCheckpointing === true`. Hosts call
   * `rewind()` on the returned object to restore captured pre-write
   * file contents — the standard "undo a tool batch" pattern. Returns
   * undefined when checkpointing is disabled or the local coding suite
   * isn't bound. Manual review (finding E): without this getter, the
   * config flag was a silent no-op outside of direct
   * `createLocalCodingToolBundle()` use.
   */
  getFileCheckpointer(): t.LocalFileCheckpointer | undefined {
    return this.fileCheckpointer;
  }

  private *getRegisteredHandoffNames(): IterableIterator<string> {
    if (this.directToolNames != null) {
      for (const toolName of this.directToolNames) {
        yield toolName;
      }
    }

    for (const toolName of this.toolMap.keys()) {
      if (this.directToolNames?.has(toolName) === true) {
        continue;
      }
      yield toolName;
    }
  }

  private hasRegisteredHandoffTool(): boolean {
    for (const toolName of this.getRegisteredHandoffNames()) {
      if (isHandoffToolName(toolName)) {
        return true;
      }
    }
    return false;
  }

  private getHandoffToolNameSuggestion(callName: string): string | undefined {
    if (!isHandoffToolName(callName)) {
      return undefined;
    }

    let suggestion: string | undefined;
    for (const toolName of this.getRegisteredHandoffNames()) {
      if (
        !isHandoffToolName(toolName) ||
        toolName.length >= callName.length ||
        !callName.startsWith(toolName)
      ) {
        continue;
      }
      if (suggestion == null || toolName.length > suggestion.length) {
        suggestion = toolName;
      }
    }
    return suggestion;
  }

  private shouldHandleUnknownHandoffLocally(
    callName: string,
    hasRegisteredHandoffTool?: boolean
  ): boolean {
    if (!isHandoffToolName(callName) || this.toolMap.has(callName)) {
      return false;
    }
    return hasRegisteredHandoffTool ?? this.hasRegisteredHandoffTool();
  }

  private getUnknownToolErrorMessage(callName: string): string {
    const suggestion = this.getHandoffToolNameSuggestion(callName);
    if (suggestion == null) {
      return `Tool "${callName}" not found.`;
    }
    return (
      `Tool "${callName}" not found. Did you mean "${suggestion}"? ` +
      'Handoff tool names must match exactly.'
    );
  }

  /**
   * Flush per-Run direct replay state. Called by the Graph at end-of-Run via
   * `clearHeavyState`. The state intentionally survives `run()` re-entry so
   * interrupt + resume keeps both original turn slots and terminal sibling
   * outputs, but it would otherwise grow linearly and could collide across
   * Runs if a provider reused call IDs. Hosts can also call this directly if
   * they reuse a ToolNode across batches outside of a Graph.
   */
  clearDirectPathTurns(): void {
    this.directPathTurns.clear();
    this.settledDirectResultsByBatch.clear();
  }

  /** Returns the live caller projection used by direct and event execution. */
  private getCallerCapabilityProjection(): CallerCapabilityProjection {
    const discoveredToolNames = new Set(this.getDiscoveredToolNames?.() ?? []);
    return resolveCallerCapabilityProjection(
      mergeCallerCapabilityDefinitions(
        this.toolDefinitions?.values(),
        this.toolRegistry?.values()
      ),
      (toolDef) => isToolDefinitionActive(toolDef, discoveredToolNames)
    );
  }

  private getToolParameterSchema(
    toolName: string
  ): t.JsonSchemaType | undefined {
    return (
      this.toolDefinitions?.get(toolName)?.parameters ??
      this.toolRegistry?.get(toolName)?.parameters
    );
  }

  private coerceEventToolArgs(
    toolName: string,
    args: unknown
  ): Record<string, unknown> {
    const recordArgs = coerceRecordArgs(args);
    return coerceArgsForSchema(
      recordArgs ?? (args as Record<string, unknown>),
      this.getToolParameterSchema(toolName)
    );
  }

  /** Serializes the live caller projection for event-driven hosts. */
  private getCallerCapabilityProjectionSnapshot(): t.CallerCapabilityProjectionSnapshot {
    return createCallerCapabilityProjectionSnapshot(
      this.getCallerCapabilityProjection()
    );
  }

  /** Returns active tools projected by their effective caller capabilities. */
  private getProgrammaticTools(): t.ProgrammaticCache {
    const toolMap: t.ToolMap = new Map();
    const toolDefs: t.LCTool[] = [];
    const discoveredToolNames = new Set(this.getDiscoveredToolNames?.() ?? []);
    const executableCapabilities = resolveCallerCapabilityProjection(
      this.toolRegistry?.values() ?? [],
      (toolDef) => isToolDefinitionActive(toolDef, discoveredToolNames)
    );
    for (const toolDef of executableCapabilities.codeExecutionTools) {
      if (
        this.eventDrivenMode &&
        this.directToolNames?.has(toolDef.name) !== true &&
        !this.localImplementationNames.has(toolDef.name)
      ) {
        continue;
      }
      const tool = this.toolMap.get(toolDef.name);
      if (tool != null) {
        toolMap.set(toolDef.name, tool);
        toolDefs.push(toolDef);
      }
    }
    const activeCapabilities = this.getCallerCapabilityProjection();

    return {
      toolMap,
      toolDefs,
      disallowedToolDefs: activeCapabilities.directOnlyTools
        .filter((toolDef) => !isProgrammaticControlTool(toolDef.name))
        .map((toolDef) => ({ name: toolDef.name })),
    };
  }

  /**
   * Returns a snapshot of the current tool usage counts.
   * @returns A ReadonlyMap where keys are tool names and values are their usage counts.
   */
  public getToolUsageCounts(): ReadonlyMap<string, number> {
    return new Map(this.toolUsageCount); // Return a copy
  }

  createSubagentResumeState(): SubagentToolNodeResumeState {
    return {
      stateKey: JSON.stringify([
        this.executingAgentId ?? '',
        this.agentId ?? '',
        this.name,
      ]),
      toolUsageCounts: [...this.toolUsageCount].map(([toolName, count]) => ({
        toolName,
        count,
      })),
      directPathTurns: [...this.directPathTurns].map(([toolCallId, turn]) => ({
        toolCallId,
        turn,
      })),
    };
  }

  restoreSubagentResumeState(state: SubagentToolNodeResumeState): void {
    this.toolUsageCount.clear();
    for (const { toolName, count } of state.toolUsageCounts) {
      this.toolUsageCount.set(toolName, count);
    }
    this.directPathTurns.clear();
    for (const { toolCallId, turn } of state.directPathTurns) {
      this.directPathTurns.set(toolCallId, turn);
    }
  }

  private recordToolUsageTurn(
    toolName: string,
    turn: number,
    callId?: string
  ): void {
    this.toolUsageCount.set(
      toolName,
      Math.max(this.toolUsageCount.get(toolName) ?? 0, turn + 1)
    );
    if (callId != null && callId !== '') {
      this.toolCallTurns.set(callId, turn);
    }
  }

  private recordEventToolPlanningTurn(
    toolName: string,
    turn: number,
    callId?: string,
    runId?: string
  ): void {
    this.recordToolUsageTurn(toolName, turn, callId);
    if (this.canConsumeEagerEventExecution(runId)) {
      this.eagerEventToolUsageCount?.set(
        toolName,
        Math.max(this.eagerEventToolUsageCount.get(toolName) ?? 0, turn + 1)
      );
    }
  }

  /**
   * Forward the graph state as langgraph 1.4's `runtime.state` so
   * tools can read it off their second argument instead of the
   * deprecated `getCurrentTaskInput()` (which relies on
   * `node:async_hooks` and is browser-incompatible). Shape mirrors
   * langgraph's prebuilt ToolNode runtime exactly.
   */
  private invokeWithRuntime(
    tool: t.GenericTool,
    invokeParams: Record<string, unknown>,
    call: ToolCall,
    config: RunnableConfig,
    runInput?: RunToolBatchContext<T>['runInput']
  ): Promise<unknown> {
    const lgConfig = config as LangGraphRunnableConfig;
    const isTrustedSubagentHop =
      call.name === Constants.SUBAGENT &&
      (tool as ReplayableSubagentTool)[SUBAGENT_REPLAY_CONTROLLER] != null;
    const toolConfig: RunnableConfig = {
      ...config,
      configurable: isTrustedSubagentHop
        ? config.configurable
        : stripToolApprovalReviewConfig(
            config.configurable as Record<string, unknown> | undefined
        ),
    };
    const runtime: ToolRuntime<T> = {
      ...toolConfig,
      state: runInput as ToolRuntime<T>['state'],
      toolCallId: call.id ?? '',
      config: toolConfig,
      context: lgConfig.context as ToolRuntime<T>['context'],
      store: (lgConfig.store ?? null) as ToolRuntime<T>['store'],
      writer:
        lgConfig.writer ??
        (config.configurable?.writer as ToolRuntime<T>['writer']) ??
        null,
    };
    this.throwIfBreakerTripped(config);
    config.signal?.throwIfAborted();
    return tool.invoke(invokeParams, runtime);
  }

  /** Only Graph's built-in subagent binding is allowed to call this seam. */
  prestartSubagent(
    call: ToolCall,
    attempt: string,
    config: RunnableConfig,
    capacity: number
  ): void {
    const runId = (config.configurable?.run_id as string | undefined) ?? '';
    if (
      call.name !== Constants.SUBAGENT ||
      call.id == null ||
      call.args.run_in_background === true ||
      call.args.subagent_thread_id != null ||
      this.humanInTheLoop?.enabled === true ||
      this.hookRegistry?.hasResultAlteringHooks(runId) === true ||
      this.eagerEventToolExecution?.excludeToolNames?.includes(call.name) ===
        true
    ) {
      return;
    }
    const tool = this.toolMap.get(call.name);
    if (tool == null || this.preparedSubagents == null) {
      return;
    }
    const scope = this.getRunScope?.();
    const capturedConfig = {
      ...config,
      configurable: {
        ...config.configurable,
        [RUN_BREAKER_SCOPE_CONFIG_KEY]: scope,
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: attempt,
      },
    };
    const args = JSON.parse(JSON.stringify(call.args)) as ToolCall['args'];
    const preparedCall = { ...call, args };
    this.preparedSubagents.start(
      attempt,
      this.executingAgentId ?? '',
      preparedCall,
      capacity,
      (signal) => {
        const turn = this.toolUsageCount.get(call.name) ?? 0;
        this.toolUsageCount.set(call.name, turn + 1);
        this.directPathTurns.set(call.id!, turn);
        // An early call is its own dispatch, before the completed batch exists.
        // Give it a real chain parent instead of borrowing the model's callback
        // parent or attempting to reparent already-exported observations later.
        return this.withToolScope(capturedConfig, async (scopedConfig) => {
          let output: unknown;
          let failure: { error: unknown } | undefined;
          const dispatchOptions = {
            ...scopedConfig,
            runName: `tools=${this.executingAgentId ?? ''}`,
          };
          delete dispatchOptions.runId;
          delete dispatchOptions.signal;
          delete dispatchOptions.timeout;
          // Isolate ambient cancellation too: RunnableLambda races its signal,
          // but capacity must remain owned until the actual tool settles.
          await AsyncLocalStorageProviderSingleton.runWithConfig(
            dispatchOptions,
            () =>
              RunnableLambda.from(async (_input, dispatchConfig) => {
                try {
                  output = await this.invokeWithRuntime(
                    tool,
                    {
                      ...preparedCall,
                      type: 'tool_call',
                      turn,
                      stepId: this.toolCallStepIds?.get(call.id!),
                    },
                    preparedCall,
                    {
                      ...dispatchConfig,
                      signal: composeAbortSignals(
                        signal,
                        composeAbortSignals(
                          config.signal,
                          scope?.controller.signal
                        )
                      ),
                    }
                  );
                  // Raw output belongs to the tool observation, whose redaction policy
                  // knows the tool name. A chain's scalar output has no such identity.
                  return { status: 'completed' };
                } catch (error) {
                  failure = { error };
                  return { status: 'failed' };
                }
              }).invoke(
                {
                  messages: [
                    new AIMessage({ content: '', tool_calls: [preparedCall] }),
                  ],
                },
                dispatchOptions
              )
          );
          if (failure != null) {
            throw failure.error;
          }
          return output;
        });
      }
    );
  }
  /**
   * Runs a single tool call with error handling.
   *
   * `batchIndex` is the tool's position within the current ToolNode
   * batch and, together with `this.currentTurn`, forms the key used to
   * register the output for future `{{tool<idx>turn<turn>}}`
   * substitutions. Omit when no registration should occur.
   */
  protected async runTool(
    call: ToolCall,
    config: RunnableConfig,
    batchContext: RunToolBatchContext<T> = {}
  ): Promise<BaseMessage | Command> {
    const {
      batchIndex,
      turn,
      batchScopeId,
      resolvedArgsByCallId,
      preBatchSnapshot,
      runInput,
    } = batchContext;
    const tool = this.toolMap.get(call.name);
    const registry = this.toolOutputRegistry;
    /**
     * Prefer the caller-provided snapshot when present — `run()`
     * captures one synchronously per batch so direct-path placeholder
     * resolution stays isolated from same-turn sibling outputs even
     * when a slow `PreToolUse` hook lets siblings finish first.
     * Falls back to the live registry for callers that didn't thread
     * a snapshot (anonymous direct invokes, legacy paths).
     */
    type ResolveFn = <T>(
      runIdArg: string | undefined,
      args: T
    ) => ResolveResult<T>;
    let resolveFn: ResolveFn | undefined;
    const resolveOptions = {
      substituteIntentKey: this.toolDeclaresBusinessIntent(call.name),
    };
    if (preBatchSnapshot != null) {
      resolveFn = <T>(_runId: string | undefined, args: T): ResolveResult<T> =>
        preBatchSnapshot.resolve(args, resolveOptions);
    } else if (registry != null) {
      resolveFn = <T>(
        runIdArg: string | undefined,
        args: T
      ): ResolveResult<T> => registry.resolve(runIdArg, args, resolveOptions);
    }
    /**
     * Precompute the reference key once per call — captured locally
     * so concurrent `invoke()` calls on the same ToolNode cannot race
     * on a shared turn field.
     */
    const refKey =
      registry != null && batchIndex != null && turn != null
        ? buildReferenceKey(batchIndex, turn)
        : undefined;
    /**
     * Hoisted outside the try so the catch branch can append
     * `[unresolved refs: …]` to error messages — otherwise the LLM
     * only sees a generic error when it references a bad key, losing
     * the self-correction signal this feature is meant to provide.
     */
    let unresolvedRefs: string[] = [];
    /**
     * Use the caller-provided `batchScopeId` when threaded from
     * `run()` (so anonymous batches get their own unique scope).
     * Fall back to the config's `run_id` when runTool is invoked
     * from a context that doesn't thread it — that still preserves
     * the runId-based partitioning for named runs.
     */
    const runId =
      batchScopeId ?? (config.configurable?.run_id as string | undefined);
    try {
      if (tool === undefined) {
        throw new Error(this.getUnknownToolErrorMessage(call.name));
      }
      /**
       * `usageCount` is the per-tool-name invocation index that
       * web-search and other tools observe via `invokeParams.turn`.
       * It is intentionally distinct from the outer `turn` parameter
       * (the batch turn used for ref keys); the latter is captured
       * before the try block when constructing `refKey`.
       *
       * Prefer the value `runDirectToolWithLifecycleHooks` already
       * incremented (Codex P2 #27) — its hook wants the SAME turn
       * the tool will execute under. When called from a path that
       * doesn't pre-increment (event dispatch, the no-hooks
       * shortcut), do the read+increment here.
       */
      const usageCount =
        batchContext.usageCount ??
        this.directPathTurns.get(call.id ?? '') ??
        ((): number => {
          const next = this.toolUsageCount.get(call.name) ?? 0;
          this.toolUsageCount.set(call.name, next + 1);
          if (call.id != null && call.id !== '') {
            this.toolCallTurns.set(call.id, next);
          }
          return next;
        })();
      this.recordToolUsageTurn(call.name, usageCount, call.id);
      if (
        batchContext.replayBatchKey != null &&
        call.id != null &&
        call.id !== ''
      ) {
        this.directPathTurns.set(call.id, usageCount);
      }
      let args = call.args;
      if (resolveFn != null) {
        const { resolved, unresolved } = resolveFn(runId, args);
        args = resolved;
        unresolvedRefs = unresolved;
        /**
         * Expose the post-substitution args to downstream completion
         * events so audit logs / host-side `ON_RUN_STEP_COMPLETED`
         * handlers observe what actually ran, not the `{{…}}`
         * template. Only string/object args are worth recording.
         */
        if (
          resolvedArgsByCallId != null &&
          call.id != null &&
          call.id !== '' &&
          resolved !== call.args &&
          typeof resolved === 'object'
        ) {
          resolvedArgsByCallId.set(
            call.id,
            resolved as Record<string, unknown>
          );
        }
      }
      const stepId = this.toolCallStepIds?.get(call.id!);

      // Build invoke params - LangChain extracts non-schema fields to config.toolCall
      // `turn` here is the per-tool usage count (matches what tools have
      // observed historically via config.toolCall.turn — e.g. web search).
      let invokeParams: Record<string, unknown> = {
        ...call,
        args,
        type: 'tool_call',
        stepId,
        turn: usageCount,
      };

      // Inject runtime data for special tools (becomes available at config.toolCall)
      if (
        call.name === Constants.PROGRAMMATIC_TOOL_CALLING ||
        call.name === Constants.BASH_PROGRAMMATIC_TOOL_CALLING
      ) {
        const { toolMap, toolDefs, disallowedToolDefs } =
          this.getProgrammaticTools();
        invokeParams = {
          ...invokeParams,
          toolMap,
          toolDefs,
          disallowedToolDefs,
          programmaticToolName: call.name,
          // Plumb the hook context into the programmatic-tool path so
          // inner tool calls made via the in-process bridge can run
          // through `PreToolUse` (deny / updatedInput) before reaching
          // the underlying tool. Without this, programmatic tool calls
          // bypass every PreToolUse hook the host registered for the tools
          // they dispatch — including HITL gates on `write_file` / `edit_file`.
          hookContext: {
            registry: this.hookRegistry,
            runId: (config.configurable?.run_id as string | undefined) ?? '',
            threadId: config.configurable?.thread_id as string | undefined,
            agentId: this.agentId,
            executionContext: this.executionContext,
            executingAgentId: this.executingAgentId,
          },
        };
      } else if (call.name === Constants.TOOL_SEARCH) {
        invokeParams = {
          ...invokeParams,
          toolRegistry: this.toolRegistry,
        };
      }

      /**
       * Inject session context for code execution tools when available.
       * Each file uses its own session_id (supporting multi-session file tracking).
       * Both session_id and _injected_files are injected directly to invokeParams
       * (not inside args) so they bypass Zod schema validation and reach config.toolCall.
       *
       * session_id is always injected when available, but concrete file refs
       * still need to travel through `_injected_files`; the legacy
       * `/files/<session_id>` fallback was removed from the executors.
       */
      if (this.participatesInCodeSession(call.name)) {
        const codeSession = this.sessions?.get(this.codeSessionKey) as
          | t.CodeSessionContext
          | undefined;
        const execSessionId = codeSession?.session_id;
        if (batchContext.codeSessionBaselineByCallId != null) {
          batchContext.codeSessionBaselineByCallId.set(
            call.id ?? '',
            codeSessionIdentityByName(codeSession)
          );
        }
        if (execSessionId != null && execSessionId !== '') {
          invokeParams = {
            ...invokeParams,
            session_id: execSessionId,
          };

          if (codeSession?.files != null && codeSession.files.length > 0) {
            invokeParams._injected_files = codeSession.files.map((file) =>
              toInjectedFileRef(file, execSessionId)
            );
          }
        }

        /**
         * Stateful runtime session hint — orthogonal to the transient
         * exec-session above, and injected independently (a first call has a
         * hint but no exec session yet). Explicit host hint wins; otherwise
         * fall back to the conversation's thread_id.
         */
        const runtimeSessionHint = this.resolveRuntimeSessionHint(config);
        if (runtimeSessionHint != null) {
          invokeParams = {
            ...invokeParams,
            _runtime_session_hint: runtimeSessionHint,
          };
        }
      }

      this.throwIfBreakerTripped(config);
      const output = await (this.preparedSubagents?.take(
        this.executingAgentId ?? '',
        { ...call, args }
      ) ?? this.invokeWithRuntime(tool, invokeParams, call, config, runInput));
      if (isCommand(output)) {
        return output;
      }
      if (isBaseMessage(output) && output._getType() === 'tool') {
        const toolMsg = output as ToolMessage;
        if (isComputerCallOutputMessage(toolMsg)) {
          return toolMsg;
        }
        const originalContent = toolMsg.content;
        const compacted = compactToolContent(
          originalContent,
          this.maxToolResultChars
        );
        if (compacted.changed) {
          toolMsg.content = compacted.content;
        }
        const isError = toolMsg.status === 'error';
        if (isError) {
          /**
           * Error ToolMessages bypass registration but still stamp the
           * unresolved-refs hint into `additional_kwargs` so the lazy
           * annotation transform surfaces it to the LLM, letting the
           * model self-correct when its reference key caused the
           * failure. Persisted `content` stays clean.
           */
          if (unresolvedRefs.length > 0) {
            toolMsg.additional_kwargs = {
              ...toolMsg.additional_kwargs,
              _unresolvedRefs: unresolvedRefs,
            };
          }
          return toolMsg;
        }
        if (this.toolOutputRegistry != null || unresolvedRefs.length > 0) {
          if (typeof originalContent === 'string') {
            const rawContent = originalContent;
            const registryContent = stripCodeSessionFileSummary(rawContent);
            const refMeta = this.recordOutputReference(
              runId,
              registryContent,
              refKey,
              unresolvedRefs
            );
            if (refMeta != null) {
              toolMsg.additional_kwargs = {
                ...toolMsg.additional_kwargs,
                ...refMeta,
              };
            }
          } else {
            /**
             * Non-string content (multi-part content blocks — text +
             * image). It is now bounded for the LLM, but cannot register under a
             * reference key because there's no canonical serialized
             * form. Warn once per tool per run when the caller
             * intended to register. The unresolved-refs hint is still
             * stamped as metadata; the lazy transform prepends a text
             * block at request time so the LLM gets the self-correction
             * signal.
             */
            if (unresolvedRefs.length > 0) {
              toolMsg.additional_kwargs = {
                ...toolMsg.additional_kwargs,
                _unresolvedRefs: unresolvedRefs,
              };
            }
            if (
              refKey != null &&
              this.toolOutputRegistry!.claimWarnOnce(runId, call.name)
            ) {
              // eslint-disable-next-line no-console
              console.warn(
                `[ToolNode] Skipping tool output reference for "${call.name}": ` +
                  'ToolMessage content is not a string (further occurrences for this tool in the same run are suppressed).'
              );
            }
          }
        }
        return toolMsg;
      }
      const serialized = serializeToolOutputWithinLimits(
        output,
        this.maxToolResultChars,
        this.toolOutputRegistry != null && refKey != null
          ? this.toolOutputRegistry.perOutputLimit
          : 0
      );
      const refMeta = this.recordOutputReference(
        runId,
        stripCodeSessionFileSummary(serialized.registryContent),
        refKey,
        unresolvedRefs
      );
      return new ToolMessage({
        status: 'success',
        name: tool.name,
        content: serialized.content,
        tool_call_id: call.id!,
        ...(refMeta != null && {
          additional_kwargs: refMeta as Record<string, unknown>,
        }),
      });
    } catch (_e: unknown) {
      const e = _e as Error;
      if (!this.handleToolErrors) {
        throw e;
      }
      if (isGraphInterrupt(e)) {
        throw e;
      }
      /**
       * A stream circuit breaker tripped by a child run (subagent) is a
       * safety abort, not a tool failure to report back to the model. It
       * passes through like an interrupt so the whole run rejects.
       */
      if (
        e instanceof StreamLimitExceededError ||
        e instanceof PreparedSubagentError
      ) {
        throw e;
      }
      if (this.errorHandler) {
        try {
          const dispatched = await this.errorHandler(
            {
              error: e,
              id: call.id!,
              name: call.name,
              input: call.args,
            },
            config.metadata
          );
          const ownership =
            batchContext.errorOwnership ?? this.looseErrorOwnership;
          if (call.id != null && call.id !== '') {
            if (dispatched === false) {
              /**
               * The handler could not dispatch the error completion (typically
               * a resume pass, where a fast-failing tool errors before the
               * step replay registers its run step). Remember the call so the
               * output loop dispatches the completion itself instead of
               * assuming the handler covered it.
               */
              ownership.undispatched.add(call.id);
            } else {
              ownership.handlerOwned.add(call.id);
            }
          }
        } catch (handlerError) {
          // A THROWN handler is not proof the completion wasn't dispatched: the
          // built-in session handler emits `tool.completed` BEFORE invoking a
          // user ON_RUN_STEP_COMPLETED callback, so a throw from that callback
          // has already dispatched. Marking it undispatched would make the
          // fallback loop re-emit a duplicate completion — only an explicit
          // `false` return (handled above) means "nothing dispatched"; a throw
          // is just logged.
          if (call.id != null && call.id !== '') {
            (
              batchContext.errorOwnership ?? this.looseErrorOwnership
            ).handlerOwned.add(call.id);
          }
          // eslint-disable-next-line no-console
          console.error('Error in errorHandler:', {
            toolName: call.name,
            toolCallId: call.id,
            toolArgs: call.args,
            stepId: this.toolCallStepIds?.get(call.id!),
            turn: this.toolUsageCount.get(call.name),
            originalError: {
              message: e.message,
              stack: e.stack ?? undefined,
            },
            handlerError:
              handlerError instanceof Error
                ? {
                  message: handlerError.message,
                  stack: handlerError.stack ?? undefined,
                }
                : {
                  message: String(handlerError),
                  stack: undefined,
                },
          });
        }
      }
      const errorContent = truncateToolResultContent(
        `Error: ${e.message}\n Please fix your mistakes.`,
        this.maxToolResultChars
      );
      const refMeta =
        unresolvedRefs.length > 0
          ? this.recordOutputReference(
            runId,
            errorContent,
            undefined,
            unresolvedRefs
          )
          : undefined;
      return new ToolMessage({
        status: 'error',
        content: errorContent,
        name: call.name,
        tool_call_id: call.id ?? '',
        ...(refMeta != null && {
          additional_kwargs: refMeta as Record<string, unknown>,
        }),
      });
    }
  }

  /**
   * Runs a single in-process tool call with the same lifecycle hooks
   * the event-dispatch path fires (`PreToolUse`, `PermissionDenied`,
   * `PostToolUse`, `PostToolUseFailure`). Used for any tool whose
   * implementation lives in the SDK process — i.e. every entry in
   * `directToolNames` — so host-supplied policy hooks gate
   * direct-invoked tools the same way they gate dispatched ones.
   *
   * Fast path: when the registry has none of the relevant events
   * registered for this run, falls through to `runTool` with zero
   * extra work. The hook list is also checked via
   * `hasHookFor(event, runId)`, which performs the registry's own
   * O(1) shortcut.
   *
   * Hook semantics intentionally mirror `dispatchToolEvents` for the
   * single-call case:
   *   - `PreToolUse` returning `decision: 'deny'` synthesizes an error
   *     `ToolMessage` and fires `PermissionDenied` (observational).
   *   - `PreToolUse` returning `decision: 'ask'`:
   *     • When `humanInTheLoop.enabled === true`: raises a real
   *       `tool_approval` interrupt for this single tool call (the
   *       same payload shape the event path produces). On resume:
   *       `approve` runs the tool, `reject` blocks via
   *       `blockDirectCall`, `respond` returns the host-supplied
   *       `responseText` as a synthetic success ToolMessage,
   *       `edit` re-runs with edited args. LangGraph re-enters
   *       ToolNode.run from the start on resume. Reusable hooks fire
   *       again; a consumed one-shot hook replays its pending approval
   *       result. In both cases `interrupt()` consumes the resume value.
   *     • When HITL is off: collapses to a fail-closed deny (matches
   *       the rest of the SDK's HITL-disabled default). One-time
   *       warning logged so hosts notice the gap.
   *   - `PreToolUse.updatedInput` is applied to the call before
   *     `runTool` runs; placeholder resolution inside `runTool` is
   *     idempotent on already-resolved args.
   *   - `PostToolUse.updatedOutput` replaces the returned
   *     `ToolMessage` content (preserving id/name/status).
   *   - `PostToolUseFailure` fires when `runTool` returns a
   *     `ToolMessage` whose `status === 'error'`. Observational only;
   *     the error message stays the source of truth.
   *
   * `PostToolBatch` aggregation across direct + dispatched outcomes is
   * a separate concern: `dispatchToolEvents` accumulates batch entries
   * locally and fires `PostToolBatch` at the end of its scope. Wiring
   * direct-call entries into that aggregation crosses the two paths'
   * scopes and is left to a follow-up.
   */
  private async runDirectToolWithLifecycleHooks(
    call: ToolCall,
    config: RunnableConfig,
    batchContext: RunToolBatchContext<T> = {}
  ): Promise<BaseMessage | Command> {
    if (
      this.preparedSubagents?.owns(this.executingAgentId ?? '', call) ===
        true &&
      this.hookRegistry?.hasResultAlteringHooks(
        (config.configurable?.run_id as string | undefined) ?? ''
      ) === true
    ) {
      this.preparedSubagents.clear();
      throw new PreparedSubagentError(
        'Tool hooks changed after a subagent started.'
      );
    }
    const replayController = (
      this.toolMap.get(call.name) as ReplayableSubagentTool | undefined
    )?.[SUBAGENT_REPLAY_CONTROLLER];
    const isTrustedSubagentReviewHop =
      call.name === Constants.SUBAGENT && replayController != null;
    const replayConfig =
      replayController == null
        ? config
        : withSubagentReplayBatch(config, batchContext.replayBatchKey);
    const settledOutput = await replayController?.getSettledOutput(
      call,
      replayConfig
    );
    if (settledOutput != null) {
      if (
        batchContext.additionalContextsSink != null &&
        settledOutput.additionalContexts.length > 0
      ) {
        batchContext.additionalContextsSink.push(
          ...settledOutput.additionalContexts
        );
      }
      if (
        batchContext.resolvedArgsByCallId != null &&
        call.id != null &&
        settledOutput.resolvedArgs != null
      ) {
        batchContext.resolvedArgsByCallId.set(
          call.id,
          settledOutput.resolvedArgs
        );
      }
      const refMeta = settledOutput.output.additional_kwargs as
        | t.ToolMessageRefMetadata
        | undefined;
      if (
        this.toolOutputRegistry != null &&
        refMeta?._refKey != null &&
        settledOutput.referenceContent != null
      ) {
        this.toolOutputRegistry.set(
          refMeta._refScope,
          refMeta._refKey,
          settledOutput.referenceContent
        );
      }
      return settledOutput.output;
    }
    const replayAdditionalContexts: string[] = [];
    const persistOutput = async (
      output: ToolMessage,
      terminalArgs?: Record<string, unknown>
    ): Promise<ToolMessage> => {
      const refMeta = output.additional_kwargs as
        | t.ToolMessageRefMetadata
        | undefined;
      if (
        terminalArgs != null &&
        call.id != null &&
        batchContext.resolvedArgsByCallId != null
      ) {
        batchContext.resolvedArgsByCallId.set(call.id, terminalArgs);
      }
      const resolvedArgs =
        terminalArgs ??
        (call.id == null
          ? undefined
          : batchContext.resolvedArgsByCallId?.get(call.id));
      const referenceContent =
        this.toolOutputRegistry == null || refMeta?._refKey == null
          ? undefined
          : this.toolOutputRegistry.get(refMeta._refScope, refMeta._refKey);
      await replayController?.persistSettledOutput(call, replayConfig, {
        output,
        additionalContexts: replayAdditionalContexts,
        ...(resolvedArgs == null ? {} : { resolvedArgs }),
        ...(referenceContent == null ? {} : { referenceContent }),
      });
      return output;
    };
    const runId = (config.configurable?.run_id as string | undefined) ?? '';
    const threadId = config.configurable?.thread_id as string | undefined;
    const hookRegistry = this.hookRegistry;
    const hasPreHook = hookRegistry?.hasHookFor('PreToolUse', runId) === true;
    const approvalReplayKey =
      call.id == null || call.id === ''
        ? undefined
        : createToolApprovalReplayKey(
          config,
          this.executingAgentId ?? this.agentId ?? '',
          call.id
        );
    const approvalReplaySessionId = getToolApprovalReplaySessionId(
      config,
      runId
    );
    const pendingApproval =
      approvalReplayKey == null
        ? undefined
        : hookRegistry?.getPendingToolApproval(
          approvalReplaySessionId,
          approvalReplayKey
        );
    const approvalReviewEvidence = getToolApprovalReviewEvidence(
      config,
      getToolBatchReplayOwner(
        config,
        this.executingAgentId ?? this.agentId ?? ''
      )
    );
    const reviewedApproval = getReviewedToolApproval(
      approvalReviewEvidence?.payload,
      call.id
    );
    const approvalEvidenceAppliesToBatch =
      approvalReviewEvidence?.payload.action_requests.some((request) =>
        batchContext.approvalScopeCallIds == null
          ? request.tool_call_id === call.id
          : batchContext.approvalScopeCallIds.has(request.tool_call_id)
      ) === true;
    const hasPostHook = hookRegistry?.hasHookFor('PostToolUse', runId) === true;
    const hasFailureHook =
      hookRegistry?.hasHookFor('PostToolUseFailure', runId) === true;

    if (
      !hasPreHook &&
      pendingApproval == null &&
      reviewedApproval == null &&
      approvalReviewEvidence == null &&
      !hasPostHook &&
      !hasFailureHook
    ) {
      const output = await this.runTool(call, replayConfig, batchContext);
      return output instanceof ToolMessage ? persistOutput(output) : output;
    }

    const registryRunId =
      batchContext.batchScopeId ??
      (config.configurable?.run_id as string | undefined);
    // Slot reservation, synchronous, before any await:
    //   1. If this call.id already has a recorded turn (from a prior
    //      entry that asked / interrupted), REUSE it. LangGraph
    //      re-runs the entire ToolNode on resume, so the same call
    //      can hit this code multiple times — incrementing on each
    //      pass would push the eventual approved execution to
    //      `turn=N` instead of `turn=0` (Codex P2 #30: the fix from
    //      P2 #27 over-incremented across re-entries).
    //   2. Otherwise reserve the next slot from the counter. Done
    //      synchronously so concurrent same-tool calls in a single
    //      Promise.all batch get distinct turns (the original P2 #27
    //      requirement still holds).
    // Net: turns are stable per call.id across interrupt/resume,
    // unique per call within a batch.
    let usageCount: number;
    // Look in the resume-stable map first; fall back to the
    // per-batch one. (`directPathTurns` is set on first entry and
    // survives `run()`'s clear, so a resume sees the original
    // assignment.)
    const cachedTurn =
      call.id != null && call.id !== ''
        ? (this.directPathTurns.get(call.id) ?? this.toolCallTurns.get(call.id))
        : undefined;
    if (cachedTurn != null) {
      usageCount = cachedTurn;
    } else {
      usageCount = this.toolUsageCount.get(call.name) ?? 0;
      this.toolUsageCount.set(call.name, usageCount + 1);
      if (call.id != null && call.id !== '') {
        this.toolCallTurns.set(call.id, usageCount);
        // Dedicated direct-path map that SURVIVES `run()`'s
        // toolCallTurns.clear() — so a re-entry triggered by
        // LangGraph interrupt resume reuses this slot instead of
        // re-incrementing. Codex P2 #30.
        this.directPathTurns.set(call.id, usageCount);
      }
    }
    const turn = usageCount;
    const stepId = this.toolCallStepIds?.get(call.id ?? '') ?? '';

    // Use the caller-threaded snapshot when available (P1 #18) so the
    // value the PreToolUse hook observes matches the value the
    // (later-awaited) `runTool` will actually run with — both are
    // anchored to the pre-batch registry state.
    const hookResolveOptions = {
      substituteIntentKey: this.toolDeclaresBusinessIntent(call.name),
    };
    const resolveDirectArgs = (
      args: Record<string, unknown>
    ): Record<string, unknown> => {
      if (batchContext.preBatchSnapshot != null) {
        return batchContext.preBatchSnapshot.resolve(args, hookResolveOptions)
          .resolved as Record<string, unknown>;
      }
      if (this.toolOutputRegistry != null) {
        return this.toolOutputRegistry.resolve(
          registryRunId,
          args,
          hookResolveOptions
        ).resolved as Record<string, unknown>;
      }
      return args;
    };
    const originalResolvedArgs = resolveDirectArgs(
      call.args as Record<string, unknown>
    );
    /** Reusable and pending hooks must replay from the original proposal so
     * non-idempotent rewrites are applied exactly once. A rebuilt runtime
     * without that policy instead falls back to the resolved arguments the
     * user actually reviewed. */
    let resolvedArgs =
      hasPreHook || pendingApproval != null
        ? originalResolvedArgs
        : (reviewedApproval?.request.arguments ?? originalResolvedArgs);

    let effectiveCall = call;
    if (
      hasPreHook ||
      pendingApproval != null ||
      reviewedApproval != null ||
      approvalReviewEvidence != null
    ) {
      const preResult: AggregatedHookResult | undefined =
        hookRegistry == null
          ? {
            additionalContexts: [],
            injectedMessages: [],
            errors: [],
          }
          : await executeHooks({
            registry: hookRegistry,
            input: {
              hook_event_name: 'PreToolUse',
              runId,
              threadId,
              agentId: this.agentId,
              executionContext: this.executionContext,
              executingAgentId: this.executingAgentId,
              toolName: call.name,
              toolInput: resolvedArgs,
              toolUseId: call.id ?? '',
              stepId,
              turn,
            },
            sessionId: runId,
            matchQuery: call.name,
            onceReplayKey: approvalReplayKey,
            onceReplaySessionId: approvalReplaySessionId,
          }).catch((): AggregatedHookResult | undefined =>
            approvalReviewEvidence == null
              ? undefined
              : {
                decision: 'deny',
                reason: 'Approval policy could not be evaluated on resume',
                additionalContexts: [],
                injectedMessages: [],
                errors: [],
              }
          );

      if (preResult != null) {
        // Forward any additionalContext strings hooks returned into
        // the per-batch sink so the caller materializes them as a
        // HumanMessage for the next model turn — same shape as the
        // event-driven path's `injected[]`. Codex P2 #39.
        if (preResult.additionalContexts.length > 0) {
          replayAdditionalContexts.push(...preResult.additionalContexts);
          batchContext.additionalContextsSink?.push(
            ...preResult.additionalContexts
          );
        }
        // Apply any input rewrite first — `ask`-with-`updatedInput` is
        // a valid combination (one matcher sanitises args, another asks
        // for approval); the reviewer should see the sanitised args.
        if (preResult.updatedInput != null) {
          resolvedArgs = resolveDirectArgs(
            preResult.updatedInput as Record<string, unknown>
          );
          effectiveCall = {
            ...call,
            args: resolvedArgs,
          };
        } else if (reviewedApproval != null) {
          resolvedArgs = reviewedApproval.request.arguments;
          effectiveCall = {
            ...call,
            args: resolvedArgs,
          };
        }

        if (preResult.decision === 'deny') {
          return persistOutput(
            this.blockDirectCall({
              call,
              resolvedArgs,
              reason: preResult.reason ?? 'Blocked by hook',
              hookRegistry,
              runId,
              threadId,
            }),
            effectiveCall.args as Record<string, unknown>
          );
        }

        /** A sibling absent from restored evidence may have raised a parallel
         * approval that LangGraph did not preserve. A current hook can ask
         * again; otherwise never let that unreviewed sibling execute. */
        if (
          approvalEvidenceAppliesToBatch &&
          reviewedApproval == null &&
          !isTrustedSubagentReviewHop &&
          preResult.decision !== 'ask'
        ) {
          return persistOutput(
            this.blockDirectCall({
              call,
              resolvedArgs,
              reason:
                'Tool call was not included in the restored approval batch — failing closed',
              hookRegistry,
              runId,
              threadId,
            }),
            resolvedArgs
          );
        }

        if (preResult.decision === 'ask' || reviewedApproval != null) {
          if (this.humanInTheLoop?.enabled !== true) {
            // Fail-closed: no HITL UI configured, so we can't actually
            // ask. Logged once via the existing helper.
            const reason = this.resolveAskDecisionForDirectTool(
              preResult.reason,
              call.name
            );
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason,
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }
          const toolCallId = call.id;
          if (
            toolCallId == null ||
            toolCallId === '' ||
            approvalReplayKey == null
          ) {
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason:
                  'Tool approval requires a non-empty tool call ID — failing closed',
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }

          // Raise a single-tool tool_approval interrupt. LangGraph
          // throws on the first execution (host gets the interrupt)
          // and returns the resume value on re-entry. Because direct
          // tools re-enter the entire ToolNode.run on resume. Reusable
          // hooks fire again; a consumed one-shot hook instead replays
          // the pending approval. We anchor `interrupt()` against the
          // node's RunnableConfig the same way `dispatchToolEvents`
          // does. A one-shot hook's pending contribution reconstructs this
          // same ask entry without dispatching the consumed hook again.
          // ToolNode disables LangSmith tracing, so the AsyncLocalStorage
          // frame must be re-established here.
          const allowedDecisions =
            preResult.decision === 'ask'
              ? preResult.allowedDecisions
              : reviewedApproval?.reviewConfig.allowed_decisions;
          const askReason =
            preResult.decision === 'ask'
              ? preResult.reason
              : reviewedApproval?.request.description;
          if (effectiveCall === call) {
            effectiveCall = { ...call, args: resolvedArgs };
          }
          const askEntry: AskEntry = {
            entry: {
              call: effectiveCall,
              args: effectiveCall.args as Record<string, unknown>,
              stepId,
            },
            reason: askReason,
            allowedDecisions,
          };
          const payload = buildToolApprovalInterruptPayload([askEntry], runId);
          const resumeValue = AsyncLocalStorageProviderSingleton.runWithConfig(
            config,
            () =>
              interrupt<
                t.ToolApprovalInterruptPayload,
                t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap
              >(payload)
          );
          hookRegistry?.clearPendingToolApproval(
            approvalReplaySessionId,
            approvalReplayKey
          );
          const currentApproval = getReviewedToolApproval(payload, toolCallId);
          if (
            reviewedApproval != null &&
            (currentApproval == null ||
              !toolApprovalProposalMatches(currentApproval, reviewedApproval))
          ) {
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason:
                  'Reviewed tool proposal changed before resume; retry approval',
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }
          const decisionByCallId = normalizeApprovalDecisions(
            [toolCallId],
            resumeValue
          );
          const decision = decisionByCallId.get(toolCallId) ?? {
            type: 'reject' as const,
            reason: 'No decision provided for tool approval',
          };
          const declaredType = (decision as { type?: unknown }).type;

          if (
            allowedDecisions != null &&
            (typeof declaredType !== 'string' ||
              !allowedDecisions.includes(
                declaredType as t.ToolApprovalDecisionType
              ))
          ) {
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason: `Decision "${typeof declaredType === 'string' ? declaredType : '<missing>'}" not in allowedDecisions [${allowedDecisions.join(', ')}] — failing closed`,
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }

          if (decision.type === 'reject') {
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason: decision.reason ?? askReason ?? 'Rejected by user',
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }

          if (decision.type === 'respond') {
            const responseText = (decision as { responseText?: unknown })
              .responseText;
            if (typeof responseText !== 'string') {
              return persistOutput(
                this.blockDirectCall({
                  call,
                  resolvedArgs,
                  reason:
                    'Approval payload `respond` was missing a string `responseText`',
                  hookRegistry,
                  runId,
                  threadId,
                }),
                effectiveCall.args as Record<string, unknown>
              );
            }
            return persistOutput(
              new ToolMessage({
                status: 'success',
                content: truncateToolResultContent(
                  responseText,
                  this.maxToolResultChars
                ),
                name: call.name,
                tool_call_id: call.id ?? '',
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }

          if (decision.type === 'edit') {
            // Mirror the event-driven path's validation
            // (see `dispatchToolEvents`'s edit branch). The wire
            // field is `updatedInput`, NOT `args` — hosts following
            // the documented `ToolApprovalDecision` shape were
            // silently ignored before, so the tool ran with the
            // original (un-edited) arguments. Fail closed on
            // malformed payloads instead of falling through with
            // undefined args.
            const updatedInput = (decision as { updatedInput?: unknown })
              .updatedInput;
            if (
              updatedInput === null ||
              typeof updatedInput !== 'object' ||
              Array.isArray(updatedInput)
            ) {
              return persistOutput(
                new ToolMessage({
                  status: 'error',
                  content:
                    'Decision "edit" missing object updatedInput — failing closed.',
                  name: call.name,
                  tool_call_id: call.id ?? '',
                }),
                effectiveCall.args as Record<string, unknown>
              );
            }
            effectiveCall = {
              ...call,
              args: updatedInput as Record<string, unknown>,
            };
            // fall through to executing the edited call
          }
          if (declaredType !== 'approve' && declaredType !== 'edit') {
            const unknownType =
              typeof declaredType === 'string' ? declaredType : '<missing>';
            return persistOutput(
              this.blockDirectCall({
                call,
                resolvedArgs,
                reason: `Unknown approval decision type "${unknownType}" — failing closed`,
                hookRegistry,
                runId,
                threadId,
              }),
              effectiveCall.args as Record<string, unknown>
            );
          }
          // 'approve' (or 'edit' after applying edits) → fall through
        }
      }
    }

    /**
     * A hook (`PreToolUse.updatedInput`) or HITL `edit` decision rewrote the
     * args: expose the EFFECTIVE args to downstream completion handling
     * (`handleRunToolCompletions` reads this sink), so the emitted
     * `tool_call.args` — and any intent/outcome label resolved from them —
     * reflect what the tool actually ran with. `runTool`'s own placeholder
     * substitution may overwrite this entry with the post-substitution args,
     * which is strictly more accurate.
     */
    if (
      effectiveCall !== call &&
      batchContext.resolvedArgsByCallId != null &&
      call.id != null &&
      call.id !== ''
    ) {
      batchContext.resolvedArgsByCallId.set(
        call.id,
        effectiveCall.args as Record<string, unknown>
      );
    }

    const output = await this.runTool(effectiveCall, replayConfig, {
      ...batchContext,
      usageCount,
    });

    if (!(output instanceof ToolMessage)) {
      return output;
    }

    if (output.status === 'error' && hasFailureHook) {
      // Await the failure hook (instead of fire-and-forget) so we
      // can capture additionalContexts before returning. The hook is
      // still observational w.r.t. the tool result itself — we don't
      // mutate `output`, just plumb the contexts. Codex P2 #39.
      const failureResult = await executeHooks({
        registry: hookRegistry,
        input: {
          hook_event_name: 'PostToolUseFailure',
          runId,
          threadId,
          agentId: this.agentId,
          executionContext: this.executionContext,
          executingAgentId: this.executingAgentId,
          toolName: call.name,
          toolInput: effectiveCall.args as Record<string, unknown>,
          toolUseId: call.id ?? '',
          error:
            typeof output.content === 'string'
              ? output.content
              : serializeStructuredValueBounded(
                output.content,
                this.maxToolResultChars
              ).content,
          stepId,
          turn,
        },
        sessionId: runId,
        matchQuery: call.name,
      }).catch(() => undefined);
      if (
        failureResult != null &&
        failureResult.additionalContexts.length > 0
      ) {
        replayAdditionalContexts.push(...failureResult.additionalContexts);
        batchContext.additionalContextsSink?.push(
          ...failureResult.additionalContexts
        );
      }
      return persistOutput(output);
    }

    if (output.status !== 'error' && hasPostHook) {
      const postResult = await executeHooks({
        registry: hookRegistry,
        input: {
          hook_event_name: 'PostToolUse',
          runId,
          threadId,
          agentId: this.agentId,
          executionContext: this.executionContext,
          executingAgentId: this.executingAgentId,
          toolName: call.name,
          toolInput: effectiveCall.args as Record<string, unknown>,
          toolOutput: output.content,
          toolUseId: call.id ?? '',
          stepId,
          turn,
        },
        sessionId: runId,
        matchQuery: call.name,
      }).catch(() => undefined);

      // Forward additionalContexts from the PostToolUse hook into
      // the per-batch sink (Codex P2 #39).
      if (postResult != null && postResult.additionalContexts.length > 0) {
        replayAdditionalContexts.push(...postResult.additionalContexts);
        batchContext.additionalContextsSink?.push(
          ...postResult.additionalContexts
        );
      }

      if (postResult?.updatedOutput != null) {
        if (hasComputerCallOutputMarker(output)) {
          if (!isComputerCallOutputContent(postResult.updatedOutput)) {
            throw new Error(
              'PostToolUse updatedOutput for a computer call must be a valid screenshot URL or screenshot content block.'
            );
          }
          return persistOutput(
            cloneToolMessageWithContent(output, postResult.updatedOutput)
          );
        }
        // Keep the tool-output registry in sync with what the model
        // actually sees. Without this, `runTool` already registered
        // the PRE-hook content under `_refKey`, and a later
        // `{{tool<i>turn<n>}}` substitution would deliver the stale
        // pre-hook bytes while the model (and downstream tools)
        // observed the post-hook replacement. Read `_refKey` /
        // `_refScope` straight off the message metadata that
        // `recordOutputReference` stamped — no need to re-derive
        // (and we couldn't, for anonymous-batch synthetic scopes).
        const refMeta = output.additional_kwargs as
          | t.ToolMessageRefMetadata
          | undefined;
        const refKey = refMeta?._refKey;
        const refScope = refMeta?._refScope;
        const replaced = serializeToolOutputWithinLimits(
          postResult.updatedOutput,
          this.maxToolResultChars,
          this.toolOutputRegistry != null && refKey != null
            ? this.toolOutputRegistry.perOutputLimit
            : 0
        );
        if (this.toolOutputRegistry != null && refKey != null) {
          this.toolOutputRegistry.set(
            refScope,
            refKey,
            replaced.registryContent
          );
        }
        return persistOutput(
          cloneToolMessageWithContent(output, replaced.content)
        );
      }
    }

    return persistOutput(output);
  }

  /**
   * `ask` decisions on direct-path tools collapse to fail-closed deny
   * only when `humanInTheLoop.enabled !== true` (i.e. there's no host
   * UI configured to actually prompt the user). Logged once per process
   * so the gap is visible. When HITL IS enabled, `ask` raises a real
   * LangGraph `interrupt()` instead — see `runDirectToolWithLifecycleHooks`.
   */
  private askDirectWarningEmitted = false;
  private resolveAskDecisionForDirectTool(
    reason: string | undefined,
    toolName: string
  ): string {
    if (!this.askDirectWarningEmitted) {
      this.askDirectWarningEmitted = true;
      // eslint-disable-next-line no-console
      console.warn(
        `[ToolNode] PreToolUse returned 'ask' for direct-path tool "${toolName}" but ` +
          'humanInTheLoop is not enabled — failing closed. Set humanInTheLoop.enabled=true ' +
          'to raise a tool_approval interrupt the host can resolve.'
      );
    }
    return reason ?? 'Blocked by hook';
  }

  /**
   * Synthesize a Blocked ToolMessage AND fire `PermissionDenied`
   * (observational) for a direct-path tool call. Centralised so the
   * deny path looks identical whether the block came from `'deny'` or
   * from a fail-closed/`'reject'`/policy-violation path.
   */
  private blockDirectCall(args: {
    call: ToolCall;
    resolvedArgs: Record<string, unknown>;
    reason: string;
    hookRegistry: HookRegistry | undefined;
    runId: string;
    threadId: string | undefined;
  }): ToolMessage {
    const { call, resolvedArgs, reason, hookRegistry, runId, threadId } = args;
    if (hookRegistry?.hasHookFor('PermissionDenied', runId) === true) {
      executeHooks({
        registry: hookRegistry,
        input: {
          hook_event_name: 'PermissionDenied',
          runId,
          threadId,
          agentId: this.agentId,
          executionContext: this.executionContext,
          executingAgentId: this.executingAgentId,
          toolName: call.name,
          toolInput: resolvedArgs,
          toolUseId: call.id ?? '',
          reason,
        },
        sessionId: runId,
        matchQuery: call.name,
      }).catch(() => {
        /* observational */
      });
    }
    return new ToolMessage({
      status: 'error',
      content: `Blocked: ${reason}`,
      name: call.name,
      tool_call_id: call.id ?? '',
    });
  }

  /**
   * Registers the full, raw output under `refKey` (when provided) and
   * builds the per-message ref metadata stamped onto the resulting
   * `ToolMessage.additional_kwargs`. The metadata is read at LLM-
   * request time by `annotateMessagesForLLM` to produce a transient
   * annotated copy of the message — the persisted `content` itself
   * stays clean.
   *
   * @param registryContent  The full, untruncated output to store in
   *   the registry so `{{tool<i>turn<n>}}` substitutions deliver the
   *   complete payload. Ignored when `refKey` is undefined.
   * @param refKey  Precomputed `tool<i>turn<n>` key, or undefined when
   *   the output is not to be registered (errors, disabled feature,
   *   unavailable batch/turn).
   * @param unresolved  Placeholder keys that did not resolve; surfaced
   *   to the LLM lazily so it can self-correct.
   * @returns A `ToolMessageRefMetadata` object when there is anything
   *   to stamp, otherwise `undefined`.
   */
  private recordOutputReference(
    runId: string | undefined,
    registryContent: string,
    refKey: string | undefined,
    unresolved: string[]
  ): t.ToolMessageRefMetadata | undefined {
    if (this.toolOutputRegistry != null && refKey != null) {
      this.toolOutputRegistry.set(runId, refKey, registryContent);
    }
    if (refKey == null && unresolved.length === 0) return undefined;
    const meta: t.ToolMessageRefMetadata = {};
    if (refKey != null) {
      meta._refKey = refKey;
      /**
       * Stamp the registry scope alongside the key so the lazy
       * annotation transform can look up the right bucket. Anonymous
       * invocations get a synthetic per-batch scope (`\0anon-<n>`)
       * that `attemptInvoke` cannot derive from
       * `config.configurable.run_id` — without this, anonymous-run
       * refs would silently fail registry lookup and the LLM would
       * never see `[ref: …]` markers for outputs that were registered.
       */
      if (runId != null) meta._refScope = runId;
    }
    if (unresolved.length > 0) meta._unresolvedRefs = unresolved;
    return meta;
  }

  /**
   * Builds code session context for injection into event-driven tool calls.
   * Mirrors the session injection logic in runTool() for direct execution.
   */
  private getCodeSessionContext(): t.ToolCallRequest['codeSessionContext'] {
    if (!this.sessions) {
      return undefined;
    }

    const codeSession = this.sessions.get(this.codeSessionKey) as
      | t.CodeSessionContext
      | undefined;
    if (!codeSession) {
      return undefined;
    }

    const execSessionId = codeSession.session_id;
    const context: NonNullable<t.ToolCallRequest['codeSessionContext']> = {
      session_id: execSessionId,
    };

    if (codeSession.files && codeSession.files.length > 0) {
      context.files = codeSession.files.map((file) =>
        toInjectedFileRef(file, execSessionId)
      );
    }

    return context;
  }

  /**
   * Extracts code execution session context from tool results and stores in Graph.sessions.
   * Mirrors the session storage logic in handleRunToolCompletions for direct execution.
   */
  /**
   * True when a tool's successful result should fold its returned exec
   * `session_id` into the shared code session: built-in `CODE_EXECUTION_TOOLS`,
   * plus host-declared sandbox-writing tools (`codeSessionToolNames`, e.g.
   * create_file/edit_file). Kept name-scoped rather than a blanket artifact
   * opt-in so only host-declared tools can influence the shared session.
   */
  private participatesInCodeSession(name: string): boolean {
    if (name === '') {
      return false;
    }
    return (
      CODE_EXECUTION_TOOLS.has(name) ||
      this.codeSessionToolNames?.has(name) === true
    );
  }

  /** Delegates to the shared resolver so the direct and event-driven planning
   *  paths derive the runtime session hint identically. */
  private resolveRuntimeSessionHint(
    config: RunnableConfig
  ): string | undefined {
    return resolveRuntimeSessionHint(
      this.toolExecution,
      config.configurable?.thread_id as string | undefined
    );
  }

  private storeCodeSessionFromResults(
    results: t.ToolExecuteResult[],
    requestMap: Map<string, t.ToolCallRequest>,
    baselineByRequestId: ReadonlyMap<
      string,
      ReadonlyMap<string, string>
    > = new Map(),
    executionBaselineByRequestId: ReadonlyMap<
      string,
      ReadonlyMap<string, string>
    > = baselineByRequestId
  ): void {
    if (!this.sessions) {
      return;
    }

    this.retainCodeSessionInputsFromRequests(
      requestMap.values(),
      baselineByRequestId
    );
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const request = requestMap.get(result.toolCallId);
      if (
        request?.name == null ||
        request.name === '' ||
        (!this.participatesInCodeSession(request.name) &&
          request.name !== Constants.SKILL_TOOL)
      ) {
        continue;
      }

      if (result.status !== 'success' || result.artifact == null) {
        continue;
      }

      const artifact = result.artifact as t.CodeExecutionArtifact | undefined;
      const execSessionId = artifact?.session_id;
      if (execSessionId == null || execSessionId === '') {
        continue;
      }

      updateCodeSession(
        this.sessions,
        this.codeSessionKey,
        execSessionId,
        artifact?.files,
        artifact?.deleted_files,
        executionBaselineByRequestId.get(request.id)
      );
    }
  }

  private retainCodeSessionInputsFromRequests(
    requests: Iterable<t.ToolCallRequest>,
    baselineByRequestId: ReadonlyMap<string, ReadonlyMap<string, string>>
  ): void {
    if (!this.sessions) {
      return;
    }
    const refreshedByName = new Map<string, t.CodeEnvFile>();
    let refreshedSessionId: string | undefined;
    let fileSessionId: string | undefined;
    for (const request of requests) {
      if (
        request.name === '' ||
        (!this.participatesInCodeSession(request.name) &&
          request.name !== Constants.SKILL_TOOL &&
          request.name !== Constants.READ_FILE &&
          request.retainCodeSessionInputs !== true)
      ) {
        continue;
      }
      const context = request.codeSessionContext;
      if (context?.session_id == null || context.session_id === '') continue;
      if (context.session_id !== request.codeSessionBaselineId) {
        refreshedSessionId = context.session_id;
      }
      const baselineIdentityByName = baselineByRequestId.get(request.id);
      for (const file of context.files ?? []) {
        if (!file.id || !file.name || !file.storage_session_id) continue;
        if (baselineIdentityByName?.get(file.name) === fileIdentityKey(file)) {
          continue;
        }
        fileSessionId = context.session_id;
        refreshedByName.set(file.name, file);
      }
    }
    const sessionId = refreshedSessionId ?? fileSessionId;
    if (sessionId == null) return;
    retainCodeSessionInputs(this.sessions, this.codeSessionKey, {
      session_id: sessionId,
      files: [...refreshedByName.values()],
    });
  }

  /**
   * Post-processes standard runTool outputs: dispatches ON_RUN_STEP_COMPLETED
   * and stores code session context. Mirrors the completion handling in
   * dispatchToolEvents for the event-driven path.
   *
   * By handling completions here in graph context (rather than in the
   * stream consumer via ToolEndHandler), the race between the stream
   * consumer and graph execution is eliminated.
   *
   * @param resolvedArgsByCallId  Per-batch resolved-args sink populated
   *   by `runTool`. Threaded as a local map (instead of instance state)
   *   so concurrent batches cannot read each other's entries.
   */
  private async handleRunToolCompletions(
    calls: ToolCall[],
    outputs: (BaseMessage | Command)[],
    config: RunnableConfig,
    resolvedArgsByCallId?: ResolvedArgsByCallId,
    codeSessionBaselineByCallId?: ReadonlyMap<
      string,
      ReadonlyMap<string, string>
    >,
    errorOwnership?: ToolErrorOwnership
  ): Promise<void> {
    const ownership = errorOwnership ?? this.looseErrorOwnership;
    for (let i = 0; i < calls.length; i++) {
      const call = calls[i];
      const output = outputs[i];
      const turn = this.toolCallTurns.get(call.id!) ?? 0;

      if (isCommand(output)) {
        continue;
      }

      const toolMessage = output as ToolMessage;
      const toolCallId = call.id ?? '';

      // Skip error ToolMessages only when the errorHandler OWNS the completion —
      // it ran and dispatched (or threw) via handleToolCallErrorStatic, so
      // dispatching again here would double-dispatch. Two cases fall through:
      //  - the handler reported it could NOT dispatch (no run step registered yet
      //    at error time, e.g. a fast-failing tool on a resume pass); by now the
      //    step replay has usually registered the id, so this loop's dispatch is
      //    the only terminal event the client's tool-call part will ever get.
      //  - the tool RETURNED an error ToolMessage rather than throwing, so the
      //    catch path (and the handler with it) never ran at all.
      // Markers are CONSUMED: leaving one set would let a later re-entry (same
      // ToolNode re-executing the batch) take the wrong branch, and the sets
      // would grow unbounded across a long-lived graph's failing calls.
      if (toolMessage.status === 'error' && this.errorHandler != null) {
        if (ownership.handlerOwned.has(toolCallId)) {
          ownership.handlerOwned.delete(toolCallId);
          continue;
        }
        ownership.undispatched.delete(toolCallId);
      }

      if (
        toolMessage.status !== 'error' &&
        this.sessions &&
        this.participatesInCodeSession(call.name)
      ) {
        const artifact = toolMessage.artifact as
          | t.CodeExecutionArtifact
          | undefined;
        const execSessionId = artifact?.session_id;
        if (execSessionId != null && execSessionId !== '') {
          updateCodeSession(
            this.sessions,
            this.codeSessionKey,
            execSessionId,
            artifact?.files,
            artifact?.deleted_files,
            codeSessionBaselineByCallId?.get(toolCallId)
          );
        }
      }

      // Dispatch ON_RUN_STEP_COMPLETED via custom event (same path as dispatchToolEvents)
      const stepId = this.toolCallStepIds?.get(toolCallId) ?? '';
      if (!stepId) {
        continue;
      }

      let contentString: string;
      if (isComputerCallOutputMessage(toolMessage)) {
        contentString = truncateToolResultContent(
          '[Computer screenshot omitted from completion event]',
          this.maxToolResultChars
        );
      } else if (typeof toolMessage.content === 'string') {
        contentString = toolMessage.content;
      } else {
        contentString = serializeStructuredValueBounded(
          toolMessage.content,
          this.maxToolResultChars
        ).content;
      }

      /**
       * Prefer the post-substitution args when a `{{…}}` placeholder
       * was resolved in `runTool`. This keeps
       * `ON_RUN_STEP_COMPLETED.tool_call.args` consistent with what
       * the tool actually received rather than leaking the template.
       */
      const effectiveArgs = resolvedArgsByCallId?.get(toolCallId) ?? call.args;
      /** Authored outcomes apply to failed calls too (“Search failed for…”);
       *  without one, an error call simply stays unlabeled. */
      const outcome = resolveToolOutcome(
        effectiveArgs,
        readOutcomeFields(toolMessage.artifact),
        { isError: toolMessage.status === 'error' }
      );
      const tool_call: t.ProcessedToolCall = {
        args: serializeToolContentBounded(
          (effectiveArgs as unknown) ?? {},
          this.maxToolResultChars
        ),
        name: call.name,
        id: toolCallId,
        output: contentString,
        progress: 1,
        ...(outcome != null && { outcome }),
      };

      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_COMPLETED,
        {
          result: {
            id: stepId,
            index: turn,
            type: 'tool_call' as const,
            tool_call,
            completed_at: Date.now(),
          },
        },
        config
      );
    }
  }

  /**
   * Dispatches tool calls to the host via ON_TOOL_EXECUTE event and returns raw ToolMessages.
   * Core logic for event-driven execution, separated from output shaping.
   *
   * Hook lifecycle (when `hookRegistry` is set):
   * 1. **PreToolUse** fires per call in parallel before dispatch. Denied
   *    calls produce error ToolMessages and fire **PermissionDenied**;
   *    surviving calls proceed with optional `updatedInput`.
   * 2. Surviving calls are dispatched to the host via `ON_TOOL_EXECUTE`.
   * 3. **PostToolUse** / **PostToolUseFailure** fire per result. Post hooks
   *    can replace tool output via `updatedOutput`.
   * 4. Injected messages from results are collected and returned alongside
   *    ToolMessages (appended AFTER to respect provider ordering).
   */
  /** Rethrows a stream-limit trip carried on the batch config's composed
   * signal. Rechecked at each later execution/dispatch stage because a tool
   * that ignores cancellation can complete normally across the trip, and
   * the next stage would otherwise start fresh side effects on a failed
   * run. */
  private throwIfBreakerTripped(config: RunnableConfig): void {
    const signal = config.signal;
    if (
      signal?.aborted === true &&
      (signal.reason instanceof StreamLimitExceededError ||
        signal.reason instanceof PreparedSubagentError)
    ) {
      throw signal.reason;
    }
  }

  private async dispatchToolEvents(
    toolCalls: ToolCall[],
    config: RunnableConfig,
    batchContext: DispatchBatchContext = {}
  ): Promise<{ toolMessages: ToolMessage[]; injected: BaseMessage[] }> {
    this.throwIfBreakerTripped(config);
    const {
      batchIndices,
      turn,
      batchScopeId,
      preResolvedArgs,
      preBatchSnapshot,
    } = batchContext;
    const runId = (config.configurable?.run_id as string | undefined) ?? '';
    /**
     * Registry-facing scope id — prefers the caller-threaded
     * `batchScopeId` so anonymous batches target their own unique
     * bucket and don't step on concurrent anonymous invocations.
     * Hooks and event payloads keep using the empty-string coerced
     * `runId` for backward compat.
     */
    const registryRunId =
      batchScopeId ?? (config.configurable?.run_id as string | undefined);
    const threadId = config.configurable?.thread_id as string | undefined;
    const registry = this.toolOutputRegistry;
    const unresolvedByCallId = new Map<string, string[]>();

    const preToolCalls = toolCalls.map((call, i) => {
      const originalArgs = call.args as Record<string, unknown>;
      let resolvedArgs = originalArgs;
      /**
       * When the caller provided a pre-resolved map (the mixed
       * direct+event path snapshots event args synchronously before
       * awaiting directs so they can't accidentally resolve
       * same-turn direct outputs), use those entries verbatim instead
       * of re-resolving against a registry that may have changed
       * since the batch started.
       */
      const pre = call.id != null ? preResolvedArgs?.get(call.id) : undefined;
      if (pre != null) {
        resolvedArgs = pre.resolved;
        if (pre.unresolved.length > 0 && call.id != null) {
          unresolvedByCallId.set(call.id, pre.unresolved);
        }
      } else if (registry != null) {
        const options = {
          substituteIntentKey: this.toolDeclaresBusinessIntent(call.name),
        };
        const { resolved, unresolved } =
          preBatchSnapshot != null
            ? preBatchSnapshot.resolve(originalArgs, options)
            : registry.resolve(registryRunId, originalArgs, options);
        resolvedArgs = resolved as Record<string, unknown>;
        if (unresolved.length > 0 && call.id != null) {
          unresolvedByCallId.set(call.id, unresolved);
        }
      }
      return {
        call,
        stepId: this.toolCallStepIds?.get(call.id!) ?? '',
        args: this.coerceEventToolArgs(call.name, resolvedArgs),
        batchIndex: batchIndices?.[i],
      };
    });

    const messageByCallId = new Map<string, ToolMessage>();
    const approvedEntries: typeof preToolCalls = [];
    /**
     * Batch-level accumulator for `additionalContext` strings returned
     * by any PreToolUse / PostToolUse / PostToolUseFailure hook in this
     * dispatch. We emit one consolidated `HumanMessage` after all tool
     * results land so the next model turn sees the injected context
     * exactly once, ordered after the ToolMessages.
     */
    const batchAdditionalContexts: string[] = [];
    /**
     * Batch-level outcome record keyed by `tool_call_id`. Captures
     * every tool call's final result (success / error from the host,
     * blocked from HITL deny / reject, substituted from HITL respond)
     * across the three call sites that touch it. We materialize the
     * `PostToolBatch` entry array in `toolCalls` order at dispatch
     * time so hooks correlating outcomes by position see exactly the
     * same sequence the model emitted — independent of when each
     * outcome was recorded (deny entries land synchronously in the
     * hook loop, approved entries land after host execution, respond
     * entries land in the resume branch).
     */
    const postToolBatchEntryByCallId = new Map<string, PostToolBatchEntry>();
    const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
      additionalContexts: [] as string[],
      injectedMessages: [] as t.InjectedMessage[],
      errors: [] as string[],
    });

    const hookRegistry = this.hookRegistry;
    const approvalReplaySessionId = getToolApprovalReplaySessionId(
      config,
      runId
    );
    const approvalReviewEvidence = getToolApprovalReviewEvidence(
      config,
      getToolBatchReplayOwner(
        config,
        this.executingAgentId ?? this.agentId ?? ''
      )
    );
    const hasPendingApproval = preToolCalls.some(
      (entry) =>
        entry.call.id != null &&
        hookRegistry?.getPendingToolApproval(
          approvalReplaySessionId,
          createToolApprovalReplayKey(
            config,
            this.executingAgentId ?? this.agentId ?? '',
            entry.call.id
          )
        ) != null
    );
    if (
      hookRegistry?.hasHookFor('PreToolUse', runId) === true ||
      hasPendingApproval ||
      approvalReviewEvidence != null
    ) {
      /**
       * Pull each call's prestarted eager record BEFORE awaiting the hooks:
       * an async deny must never race the eager host promise into emitting a
       * successful completion (`dispatchEagerToolCompletions`' map-identity
       * check skips deleted records). Allowed entries get their record
       * restored in the decision loop below so consumption still works;
       * denied entries stay deleted. Ask/interrupt configs never have eager
       * records (HITL disables the reservation gate).
       */
      const preemptedEagerRecords = new Map<
        string,
        t.EagerEventToolExecution
      >();
      if (this.eagerEventToolExecutions != null) {
        for (const entry of preToolCalls) {
          const callId = entry.call.id;
          if (callId == null) {
            continue;
          }
          const record = this.eagerEventToolExecutions.get(callId);
          if (record != null) {
            preemptedEagerRecords.set(callId, record);
            this.eagerEventToolExecutions.delete(callId);
          }
        }
      }
      const preResults =
        hookRegistry == null
          ? preToolCalls.map(() => HOOK_FALLBACK)
          : await Promise.all(
            preToolCalls.map((entry) => {
              const toolUseId = entry.call.id;
              const approvalReplayKey =
                  toolUseId == null
                    ? undefined
                    : createToolApprovalReplayKey(
                      config,
                      this.executingAgentId ?? this.agentId ?? '',
                      toolUseId
                    );
              return executeHooks({
                registry: hookRegistry,
                input: {
                  hook_event_name: 'PreToolUse',
                  runId,
                  threadId,
                  agentId: this.agentId,
                  executionContext: this.executionContext,
                  executingAgentId: this.executingAgentId,
                  toolName: entry.call.name,
                  toolInput: entry.args,
                  toolUseId: entry.call.id!,
                  stepId: entry.stepId,
                  turn: this.toolUsageCount.get(entry.call.name) ?? 0,
                },
                sessionId: runId,
                matchQuery: entry.call.name,
                onceReplayKey: approvalReplayKey,
                onceReplaySessionId: approvalReplaySessionId,
              }).catch((): AggregatedHookResult => HOOK_FALLBACK);
            })
          );

      type PendingEntry = (typeof preToolCalls)[number];

      /**
       * Side effects deferred from `blockEntry` until after any pending
       * `interrupt()` resolves. Without deferral, a batch that mixes a
       * `deny` decision with an `ask` decision would dispatch
       * `ON_RUN_STEP_COMPLETED` for the denied tool on the FIRST node
       * execution (before `interrupt()` throws), then dispatch the
       * same event AGAIN on the resume re-execution — hosts would
       * observe two completion events for one logical denial. By
       * queueing the dispatch + PermissionDenied hook here and
       * flushing after the interrupt block, we ensure each side effect
       * fires exactly once: never on the first pass when interrupt
       * throws (the flush is unreachable), once on resume / no-ask
       * passes when control reaches the flush.
       */
      const deferredBlockedSideEffects: Array<{
        callId: string;
        toolName: string;
        args: Record<string, unknown>;
        contentString: string;
        reason: string;
      }> = [];

      const blockEntry = (entry: PendingEntry, reason: string): void => {
        const contentString = `Blocked: ${reason}`;
        messageByCallId.set(
          entry.call.id!,
          new ToolMessage({
            status: 'error',
            content: contentString,
            name: entry.call.name,
            tool_call_id: entry.call.id!,
          })
        );
        postToolBatchEntryByCallId.set(entry.call.id!, {
          toolName: entry.call.name,
          toolInput: entry.args,
          toolUseId: entry.call.id!,
          stepId: entry.stepId,
          /**
           * Records the pre-invocation turn count — the same value the
           * executed path captures before incrementing `toolUsageCount`.
           * For a blocked tool the counter is never incremented (no
           * invocation happened), so this is always the count of prior
           * successful invocations of this tool name in earlier batches.
           * Surfaces in the `PostToolBatch` entry so batch hooks see
           * a uniform shape regardless of outcome.
           */
          turn: this.toolUsageCount.get(entry.call.name) ?? 0,
          status: 'error',
          error: contentString,
        });
        deferredBlockedSideEffects.push({
          callId: entry.call.id!,
          toolName: entry.call.name,
          args: entry.args,
          contentString,
          reason,
        });
        /**
         * A prestarted eager execution for a now-blocked call must neither be
         * consumed nor emit its completion — the run reports this call as
         * blocked. Deleting the record makes `dispatchEagerToolCompletions`'
         * map-identity check skip the pending emission (same pattern as the
         * rejected-results cleanup below).
         */
        this.eagerEventToolExecutions?.delete(entry.call.id!);
      };

      const flushDeferredBlockedSideEffects = async (): Promise<void> => {
        for (const item of deferredBlockedSideEffects) {
          await this.dispatchStepCompleted(
            item.callId,
            item.toolName,
            item.args,
            item.contentString,
            config
          );
          if (hookRegistry?.hasHookFor('PermissionDenied', runId) === true) {
            executeHooks({
              registry: hookRegistry,
              input: {
                hook_event_name: 'PermissionDenied',
                runId,
                threadId,
                agentId: this.agentId,
                executionContext: this.executionContext,
                executingAgentId: this.executingAgentId,
                toolName: item.toolName,
                toolInput: item.args,
                toolUseId: item.callId,
                reason: item.reason,
              },
              sessionId: runId,
              matchQuery: item.toolName,
            }).catch(() => {
              /* PermissionDenied is observational — swallow errors */
            });
          }
        }
        deferredBlockedSideEffects.length = 0;
      };

      /**
       * Apply a hook-supplied or host-supplied input override to a pending
       * entry, re-running the `{{tool<i>turn<n>}}` resolver so any new
       * placeholders introduced by the override are substituted (and any
       * formerly-unresolved refs cleared from the unresolved set).
       *
       * Mixed direct+event batches must use the pre-batch snapshot so a
       * hook-introduced placeholder cannot accidentally resolve to a
       * same-turn direct output that has just registered. Pure event
       * batches don't have a snapshot and resolve against the live
       * registry — safe because no event-side registrations have happened
       * yet.
       */
      const applyInputOverride = (
        entry: PendingEntry,
        nextArgs: Record<string, unknown>
      ): void => {
        if (registry != null) {
          const view: ToolOutputResolveView = preBatchSnapshot ?? {
            resolve: <T>(args: T, options?: ResolveOptions) =>
              registry.resolve(registryRunId, args, options),
          };
          const { resolved, unresolved } = view.resolve(nextArgs, {
            substituteIntentKey: this.toolDeclaresBusinessIntent(
              entry.call.name
            ),
          });
          entry.args = this.coerceEventToolArgs(
            entry.call.name,
            resolved as Record<string, unknown>
          );
          if (entry.call.id != null) {
            if (unresolved.length > 0) {
              unresolvedByCallId.set(entry.call.id, unresolved);
            } else {
              unresolvedByCallId.delete(entry.call.id);
            }
          }
          return;
        }
        entry.args = this.coerceEventToolArgs(entry.call.name, nextArgs);
      };

      const askEntries: Array<{
        entry: PendingEntry;
        reason?: string;
        allowedDecisions?: ReadonlyArray<
          'approve' | 'reject' | 'edit' | 'respond'
        >;
      }> = [];

      for (let i = 0; i < preToolCalls.length; i++) {
        const hookResult = preResults[i];
        const entry = preToolCalls[i];

        for (const ctx of hookResult.additionalContexts) {
          batchAdditionalContexts.push(ctx);
        }

        if (hookResult.decision === 'deny') {
          blockEntry(entry, hookResult.reason ?? 'Blocked by hook');
          continue;
        }

        if (hookResult.decision === 'ask') {
          /**
           * HITL is OFF by default — hosts must explicitly opt in via
           * `humanInTheLoop: { enabled: true }` to engage the
           * `interrupt()` path. When opted out (or omitted), `ask`
           * collapses into the pre-HITL fail-closed path: a blocked
           * tool with an error `ToolMessage`. The default stays
           * conservative until host UIs are ready to render
           * `tool_approval` interrupts; see `HumanInTheLoopConfig`
           * JSDoc for the full rationale and the migration plan.
           */
          if (this.humanInTheLoop?.enabled !== true) {
            blockEntry(entry, hookResult.reason ?? 'Blocked by hook');
            continue;
          }
          /**
           * Apply `updatedInput` BEFORE queuing into `askEntries` —
           * a hook is allowed to return both a sanitization rewrite
           * and an `ask` decision (e.g. one matcher redacts secrets,
           * another matcher requires approval). Without this, the
           * interrupt payload would surface the original args to the
           * reviewer AND the post-approve execution would run with
           * the original args, silently dropping the hook's rewrite.
           */
          if (hookResult.updatedInput != null) {
            applyInputOverride(entry, hookResult.updatedInput);
          } else {
            const reviewed = getReviewedToolApproval(
              approvalReviewEvidence?.payload,
              entry.call.id
            );
            if (reviewed != null) {
              entry.args = this.coerceEventToolArgs(
                entry.call.name,
                reviewed.request.arguments
              );
              if (entry.call.id != null) {
                unresolvedByCallId.delete(entry.call.id);
              }
            }
          }
          askEntries.push({
            entry,
            reason: hookResult.reason,
            allowedDecisions: hookResult.allowedDecisions,
          });
          continue;
        }

        if (hookResult.updatedInput != null) {
          applyInputOverride(entry, hookResult.updatedInput);
        }
        const reviewedApproval = getReviewedToolApproval(
          approvalReviewEvidence?.payload,
          entry.call.id
        );
        if (reviewedApproval != null) {
          if (hookResult.updatedInput == null) {
            entry.args = this.coerceEventToolArgs(
              entry.call.name,
              reviewedApproval.request.arguments
            );
            if (entry.call.id != null) {
              unresolvedByCallId.delete(entry.call.id);
            }
          }
          askEntries.push({
            entry,
            reason: reviewedApproval.request.description,
            allowedDecisions: reviewedApproval.reviewConfig.allowed_decisions,
          });
          continue;
        }
        if (entry.call.id != null) {
          const preempted = preemptedEagerRecords.get(entry.call.id);
          if (preempted != null) {
            this.eagerEventToolExecutions?.set(entry.call.id, preempted);
          }
        }
        approvedEntries.push(entry);
      }

      /**
       * If any entries asked for approval, raise a single LangGraph
       * `interrupt()` carrying every pending request together. The host
       * pauses, gathers human input, and resumes the run with one
       * decision per request. On resume LangGraph re-executes this node
       * from the start; `interrupt()` then returns the resume value
       * instead of throwing. Reusable hooks rebuild their entries, while
       * consumed one-shot hooks replay their pending approval aggregate.
       */
      if (askEntries.length > 0) {
        const payload = buildToolApprovalInterruptPayload(askEntries, runId);

        /**
         * `interrupt()` reads the current `RunnableConfig` from
         * AsyncLocalStorage. ToolNode usually runs with tracing disabled
         * (unless Langfuse explicitly enables it), so the upstream
         * `runWithConfig` frame may not exist. Re-anchor here using the
         * node's own `config` — Pregel hands us a config that already
         * carries every checkpoint/scratchpad key `interrupt()` needs to
         * suspend and resume.
         */
        const resumeValue = AsyncLocalStorageProviderSingleton.runWithConfig(
          config,
          () =>
            interrupt<
              t.ToolApprovalInterruptPayload,
              t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap
            >(payload)
        );

        for (const { entry } of askEntries) {
          hookRegistry?.clearPendingToolApproval(
            approvalReplaySessionId,
            createToolApprovalReplayKey(
              config,
              this.executingAgentId ?? this.agentId ?? '',
              entry.call.id!
            )
          );
        }

        const proposalMatches =
          approvalReviewEvidence == null ||
          toolApprovalPayloadMatches(payload, approvalReviewEvidence.payload);
        if (!proposalMatches) {
          for (const { entry } of askEntries) {
            blockEntry(
              entry,
              'Reviewed tool proposal changed before resume; retry approval'
            );
          }
        }
        const decisionEntries = proposalMatches ? askEntries : [];
        const decisionByCallId = normalizeApprovalDecisions(
          decisionEntries.map(({ entry }) => entry.call.id!),
          resumeValue
        );

        for (const {
          entry,
          reason: askReason,
          allowedDecisions,
        } of decisionEntries) {
          const decision = decisionByCallId.get(entry.call.id!) ?? {
            type: 'reject' as const,
            reason: 'No decision provided for tool approval',
          };
          /**
           * Read `decision.type` through a widened view once: hosts
           * deserialize resume payloads from untyped JSON, so the
           * runtime value can be a typo, the wrong type, or missing
           * entirely. Both the `allowedDecisions` enforcement
           * immediately below and the unknown-type fallthrough at the
           * end of this loop body share this single read so the
           * fail-closed checks compare against the same source.
           */
          const declaredType = (decision as { type?: unknown }).type;

          /**
           * Enforce the per-tool `allowedDecisions` allowlist that the
           * `PreToolUse` hook surfaced in `review_configs`. The host
           * UI is supposed to honor this when collecting the user's
           * decision, but the wire is untrusted: a buggy or hostile
           * host could submit a decision type the policy explicitly
           * forbids (e.g. `'edit'` when the hook restricted to
           * `['approve', 'reject']`), bypassing argument-mutation /
           * response-substitution safeguards. Fail closed when the
           * declared type isn't in the allowlist.
           */
          if (
            allowedDecisions != null &&
            (typeof declaredType !== 'string' ||
              !allowedDecisions.includes(
                declaredType as t.ToolApprovalDecisionType
              ))
          ) {
            const offered =
              typeof declaredType === 'string' ? declaredType : '<missing>';
            blockEntry(
              entry,
              `Decision "${offered}" not in allowedDecisions [${allowedDecisions.join(', ')}] — failing closed`
            );
            continue;
          }

          if (decision.type === 'reject') {
            blockEntry(
              entry,
              decision.reason ?? askReason ?? 'Rejected by user'
            );
            continue;
          }

          /**
           * `respond` short-circuits tool execution: the human supplies
           * the result the model should see in place of running the
           * tool. We emit a successful `ToolMessage` directly and skip
           * dispatch — no host event fires, no real tool side effect
           * occurs. Mirrors LangChain HITL middleware semantics.
           */
          if (decision.type === 'respond') {
            /**
             * Validate the wire shape before touching it: hosts
             * deserialize resume payloads from untyped JSON, so a
             * malformed `{ type: 'respond' }` (no `responseText`) or
             * `{ type: 'respond', responseText: 42 }` would crash
             * `truncateToolResultContent` (which calls
             * `content.length`) and turn a fail-closed approval path
             * into a hard run failure. Route bad shapes through
             * `blockEntry` like any other unusable decision.
             */
            const responseText = (decision as { responseText?: unknown })
              .responseText;
            if (typeof responseText !== 'string') {
              blockEntry(
                entry,
                `Decision "respond" missing string responseText (got ${describeOfferedShape(responseText)}) — failing closed`
              );
              continue;
            }
            /**
             * Truncate the human-supplied text just like the success
             * path does for real tool output. Without this, a user
             * pasting a large document as a manual response bypasses
             * `maxToolResultChars` and can blow past the model's
             * context window. The PostToolBatch entry surfaces the
             * truncated text too so batch hooks see what the model
             * will actually see.
             */
            const truncatedResponse = truncateToolResultContent(
              responseText,
              this.maxToolResultChars
            );
            messageByCallId.set(
              entry.call.id!,
              new ToolMessage({
                status: 'success',
                content: truncatedResponse,
                name: entry.call.name,
                tool_call_id: entry.call.id!,
              })
            );
            postToolBatchEntryByCallId.set(entry.call.id!, {
              toolName: entry.call.name,
              toolInput: entry.args,
              toolUseId: entry.call.id!,
              stepId: entry.stepId,
              turn: this.toolUsageCount.get(entry.call.name) ?? 0,
              status: 'success',
              toolOutput: truncatedResponse,
            });
            /**
             * Safe to dispatch immediately — unlike `blockEntry` which
             * defers, `respond` only executes inside the decision-
             * processing loop, which is reachable only AFTER
             * `interrupt()` has returned (the resume pass). There is
             * no risk of being rolled back by a subsequent throw, so
             * no risk of a duplicate `ON_RUN_STEP_COMPLETED` event.
             */
            await this.dispatchStepCompleted(
              entry.call.id!,
              entry.call.name,
              entry.args,
              truncatedResponse,
              config
            );
            continue;
          }

          if (decision.type === 'edit') {
            /**
             * Validate the wire shape before touching it: hosts
             * deserialize resume payloads from untyped JSON, so a
             * malformed `{ type: 'edit' }` (no `updatedInput`),
             * `{ type: 'edit', updatedInput: 'string' }` (non-object),
             * or `{ type: 'edit', updatedInput: [...] }` (array, not a
             * plain object) would feed garbage into
             * `applyInputOverride` and silently approve a tool with
             * undefined / wrong-shape args. Same trust boundary as
             * the `respond` validation above — fail closed via
             * `blockEntry` with a diagnostic.
             */
            const updatedInput = (decision as { updatedInput?: unknown })
              .updatedInput;
            if (
              updatedInput === null ||
              typeof updatedInput !== 'object' ||
              Array.isArray(updatedInput)
            ) {
              blockEntry(
                entry,
                `Decision "edit" missing object updatedInput (got ${describeOfferedShape(updatedInput)}) — failing closed`
              );
              continue;
            }
            applyInputOverride(entry, updatedInput as Record<string, unknown>);
            approvedEntries.push(entry);
            continue;
          }

          /**
           * Defensive type widening: hosts deserialize resume payloads
           * from untyped JSON, so the `decision.type` value at runtime
           * is whatever string the wire sent — not necessarily one of
           * the four union variants TS knows about. We compare against
           * the literal `'approve'` through the widened `declaredType`
           * captured at the top of this iteration, so a typo or schema
           * drift (`'aproved'`, `null`, `undefined`) hits the fail-
           * closed branch below instead of silently approving the
           * tool. Without this widening, TS narrows the union after
           * the three earlier branches and treats `=== 'approve'` as
           * trivially true.
           */
          if (declaredType === 'approve') {
            approvedEntries.push(entry);
            continue;
          }

          /**
           * Unknown / missing decision type — fail closed. The whole
           * point of an approval gate is that "no decision" or
           * "garbled decision" deny by default.
           */
          const unknownType =
            typeof declaredType === 'string' ? declaredType : '<missing>';
          blockEntry(
            entry,
            `Unknown approval decision type "${unknownType}" — failing closed`
          );
        }
      }

      /**
       * Flush deferred denial side effects exactly once. On the FIRST
       * pass through a batch that contains an `ask`, `interrupt()`
       * threw above and we never reach this line — so no
       * `ON_RUN_STEP_COMPLETED` / `PermissionDenied` events fire
       * for blocked tools yet. On resume the node re-executes from
       * scratch, `blockEntry` re-queues the same entries, and the
       * flush below dispatches them once. For batches without any
       * `ask` (deny-only or empty), the flush still runs here and
       * dispatches in the same relative position as the pre-deferral
       * code did (after hook processing, before tool execution).
       */
      await flushDeferredBlockedSideEffects();
    } else {
      approvedEntries.push(...preToolCalls);
    }

    const injected: BaseMessage[] = [];

    const batchIndexByCallId = new Map<string, number>();

    if (approvedEntries.length > 0) {
      const plan = buildToolExecutionRequestPlan({
        toolCalls: approvedEntries.map((entry) => {
          const codeSessionContext =
            this.participatesInCodeSession(entry.call.name) ||
            entry.call.name === Constants.SKILL_TOOL ||
            entry.call.name === Constants.READ_FILE
              ? this.getCodeSessionContext()
              : undefined;
          const runtimeSessionHint = this.participatesInCodeSession(
            entry.call.name
          )
            ? this.resolveRuntimeSessionHint(config)
            : undefined;
          return {
            id: entry.call.id,
            name: entry.call.name,
            args: entry.args,
            stepId: entry.stepId,
            codeSessionContext,
            runtimeSessionHint,
          };
        }),
        usageCount: this.toolUsageCount,
        invalidArgsBehavior: 'error-result',
        getToolSchema: (toolName) => this.getToolParameterSchema(toolName),
        recordTurn: (toolName, reservedTurn, callId) => {
          this.recordEventToolPlanningTurn(
            toolName,
            reservedTurn,
            callId,
            runId
          );
        },
      });
      if (plan == null) {
        throw new Error('Unable to build event tool execution request plan');
      }
      const requests = plan.requests;

      for (const entry of approvedEntries) {
        if (entry.batchIndex != null && entry.call.id != null) {
          batchIndexByCallId.set(entry.call.id, entry.batchIndex);
        }
      }

      for (const result of plan.rejectedResults) {
        this.eagerEventToolExecutions?.delete(result.toolCallId);
      }

      const requestMap = new Map(plan.allRequests.map((r) => [r.id, r]));
      for (const request of plan.allRequests) {
        request.codeSessionBaselineId = request.codeSessionContext?.session_id;
      }
      const codeSessionBaselineByRequestId = new Map<
        string,
        ReadonlyMap<string, string>
      >(
        plan.allRequests.map((request) => [
          request.id,
          codeSessionIdentityByName(request.codeSessionContext),
        ])
      );
      const eagerExecutions: Array<{
        request: t.ToolCallRequest;
        execution: t.EagerEventToolExecution;
      }> = [];
      const dispatchRequests: t.ToolCallRequest[] = [];

      for (const request of requests) {
        const eagerExecution = this.takeMatchingEagerEventExecution(request);
        if (eagerExecution != null) {
          eagerExecutions.push({ request, execution: eagerExecution });
          if (eagerExecution.codeSessionBaselineByName != null) {
            codeSessionBaselineByRequestId.set(
              request.id,
              eagerExecution.codeSessionBaselineByName
            );
          }
          request.codeSessionBaselineId =
            eagerExecution.request.codeSessionBaselineId;
        } else {
          dispatchRequests.push(request);
        }
      }

      /**
       * Per-call completion fast-path: when the host reports a result
       * through `onResult` before the batch resolves, emit that call's
       * completed run step immediately instead of waiting for the slowest
       * call in the batch. Safe only when nothing can change the result
       * after execution — post-tool hooks may rewrite output and HITL may
       * deny a call, so those configurations keep batch-time emission.
       * Ids are claimed synchronously before the async dispatch and
       * released if the dispatch fails, letting the batch path re-emit.
       */
      const canEmitEarlyCompletions =
        this.hookRegistry?.hasResultAlteringHooks(runId) !== true &&
        this.humanInTheLoop?.enabled !== true;
      /**
       * Snapshot the post-hook gates at the same instant as the early-emission
       * gate: a result-altering hook registered while this batch is in flight
       * applies from the NEXT batch. Evaluating these after results settle
       * would let a late hook rewrite the ToolMessage AFTER an early
       * completion already emitted the pre-hook output — two views of the
       * same call.
       */
      const hasPostHook =
        this.hookRegistry?.hasHookFor('PostToolUse', runId) === true;
      const hasFailureHook =
        this.hookRegistry?.hasHookFor('PostToolUseFailure', runId) === true;
      const earlyCompletionDispatchedIds = new Set<string>();
      const earlyCompletionDispatches: Array<Promise<void>> = [];
      const dispatchRequestById = new Map(
        dispatchRequests.map((request) => [request.id, request])
      );
      const onResult = (result: t.ToolExecuteResult): void => {
        /** Runtime-honest widening: hosts may omit the id despite the
         * field type. */
        const resultToolCallId = result.toolCallId as string | undefined;
        const request =
          resultToolCallId != null
            ? dispatchRequestById.get(resultToolCallId)
            : undefined;
        if (
          request == null ||
          earlyCompletionDispatchedIds.has(result.toolCallId)
        ) {
          return;
        }
        earlyCompletionDispatchedIds.add(result.toolCallId);
        earlyCompletionDispatches.push(
          this.dispatchEarlyToolCompletion(result, request, config).then(
            (dispatched) => {
              if (!dispatched) {
                earlyCompletionDispatchedIds.delete(result.toolCallId);
              }
            },
            () => {
              earlyCompletionDispatchedIds.delete(result.toolCallId);
            }
          )
        );
      };

      /** The approval hooks above are awaited; a sibling's trip during them
       * must stop the approved batch before it reaches a host handler that
       * may never inspect the request signal. */
      if (dispatchRequests.length > 0) {
        this.throwIfBreakerTripped(config);
      }
      const dispatchPromise =
        dispatchRequests.length === 0
          ? Promise.resolve([] as t.ToolExecuteResult[])
          : new Promise<t.ToolExecuteResult[]>((resolve, reject) => {
            let dispatchSettled = false;
            let resultSettled = false;
            let settledResults: t.ToolExecuteResult[] | undefined;

            const maybeResolve = (): void => {
              if (dispatchSettled && resultSettled) {
                resolve(settledResults ?? []);
              }
            };

            const batchRequest: t.ToolExecuteBatchRequest = {
              executionContext: this.executionContext,
              toolCalls: dispatchRequests,
              userId: config.configurable?.user_id as string | undefined,
              // Dispatch attribution, NOT the hook subagent-scope marker:
              // hosts key tool/credential lookup on the owning agent, and
              // the eager path sends `agentContext.agentId` — this must
              // match it at the top level too.
              agentId: this.executingAgentId,
              callerCapabilityProjection:
                  this.getCallerCapabilityProjectionSnapshot(),
              configurable: stripHostExecutionConfig(
                  config.configurable as Record<string, unknown> | undefined
              ),
              metadata: config.metadata as
                  | Record<string, unknown>
                  | undefined,
              signal: config.signal,
              resolve: (results): void => {
                resultSettled = true;
                settledResults = results;
                maybeResolve();
              },
              reject,
              ...(canEmitEarlyCompletions && { onResult }),
            };

            void safeDispatchCustomEvent(
              GraphEvents.ON_TOOL_EXECUTE,
              batchRequest,
              config
            )
              .then(() => {
                dispatchSettled = true;
                maybeResolve();
              })
              .catch(reject);
          });

      const eagerResultsPromise = Promise.all(
        eagerExecutions.map(async ({ request, execution }) => {
          try {
            const results = await this.resolveEagerEventExecution(
              request,
              execution
            );
            return {
              results,
              completionDispatched:
                execution.completionDispatched === true &&
                execution.request.turn === request.turn,
              toolCallId: request.id,
            };
          } finally {
            if (execution.request.codeSessionContext != null) {
              request.codeSessionContext = execution.request.codeSessionContext;
            }
          }
        })
      );

      const [eagerOutcome, dispatchedOutcome] = await Promise.allSettled([
        eagerResultsPromise,
        dispatchPromise,
      ]);
      if (eagerOutcome.status === 'rejected') {
        this.retainCodeSessionInputsFromRequests(
          requestMap.values(),
          codeSessionBaselineByRequestId
        );
        throw eagerOutcome.reason;
      }
      if (dispatchedOutcome.status === 'rejected') {
        this.retainCodeSessionInputsFromRequests(
          requestMap.values(),
          codeSessionBaselineByRequestId
        );
        throw dispatchedOutcome.reason;
      }
      const eagerResults = eagerOutcome.value;
      const dispatchedResults = dispatchedOutcome.value;
      // Settle in-flight early completion dispatches before the batch loop
      // below decides which completions still need emitting.
      await Promise.allSettled(earlyCompletionDispatches);
      const eagerCompletionDispatchedIds = new Set(
        eagerResults
          .filter((result) => result.completionDispatched)
          .map((result) => result.toolCallId)
      );
      const flattenedEagerResults = eagerResults.flatMap(
        (result) => result.results
      );
      const results = [
        ...plan.rejectedResults,
        ...flattenedEagerResults,
        ...dispatchedResults,
      ];
      /* Hosts may lazily refresh codeSessionContext while dispatching. The
       * pre-dispatch snapshot above remains the authority for detecting those
       * additions, while deletion reconciliation must use the identities the
       * execution actually received. */
      const executionBaselineByRequestId = new Map<
        string,
        ReadonlyMap<string, string>
      >(
        plan.allRequests.map((request) => [
          request.id,
          codeSessionIdentityByName(request.codeSessionContext),
        ])
      );

      this.storeCodeSessionFromResults(
        results,
        requestMap,
        codeSessionBaselineByRequestId,
        executionBaselineByRequestId
      );

      for (const result of results) {
        if (result.injectedMessages && result.injectedMessages.length > 0) {
          try {
            injected.push(...convertInjectedMessages(result.injectedMessages));
          } catch (e) {
            // eslint-disable-next-line no-console
            console.warn(
              `[ToolNode] Failed to convert injectedMessages for toolCallId=${result.toolCallId}:`,
              e instanceof Error ? e.message : e
            );
          }
        }
        const request = requestMap.get(result.toolCallId);
        const toolName = request?.name ?? 'unknown';

        let contentString: string;
        let toolMessage: ToolMessage;
        /**
         * Tracks the post-PostToolUse-hook output so the
         * `PostToolBatch` entry below sees the final transformed value
         * even when a hook replaced the original via `updatedOutput`.
         * Lives at the loop-iteration scope so the success branch can
         * mutate it; the error branch leaves it unset (and the batch
         * entry uses `error` instead of `toolOutput` in that case).
         */
        let finalToolOutput: unknown = result.content;

        if (result.status === 'error') {
          contentString = truncateToolResultContent(
            `Error: ${result.errorMessage ?? 'Unknown error'}\n Please fix your mistakes.`,
            this.maxToolResultChars
          );
          /**
           * Error results bypass registration but stamp the
           * unresolved-refs hint into `additional_kwargs` so the lazy
           * annotation transform surfaces it to the LLM at request
           * time, letting the model self-correct when its reference
           * key caused the failure. Persisted `content` stays clean.
           */
          const unresolved = unresolvedByCallId.get(result.toolCallId) ?? [];
          const errorRefMeta =
            unresolved.length > 0
              ? this.recordOutputReference(
                registryRunId,
                contentString,
                undefined,
                unresolved
              )
              : undefined;
          toolMessage = new ToolMessage({
            status: 'error',
            content: contentString,
            name: toolName,
            tool_call_id: result.toolCallId,
            ...(errorRefMeta != null && {
              additional_kwargs: errorRefMeta as Record<string, unknown>,
            }),
          });

          if (hasFailureHook) {
            const failureHookResult = await executeHooks({
              registry: this.hookRegistry!,
              input: {
                hook_event_name: 'PostToolUseFailure',
                runId,
                threadId,
                agentId: this.agentId,
                executionContext: this.executionContext,
                executingAgentId: this.executingAgentId,
                toolName,
                toolInput: request?.args ?? {},
                toolUseId: result.toolCallId,
                error: result.errorMessage ?? 'Unknown error',
                stepId: request?.stepId,
                turn: request?.turn,
              },
              sessionId: runId,
              matchQuery: toolName,
            }).catch((): undefined => undefined);
            /**
             * Collect `additionalContext` from failure hooks too. Without
             * this, recovery guidance returned on tool errors (e.g.
             * "if this tool errors with X, suggest Y to the user") is
             * silently dropped even though the API surface advertises
             * `additionalContext` for this event. PostToolUseFailure
             * remains observational for errors thrown by the hook
             * itself, but a successfully-returned result is honored.
             */
            if (failureHookResult != null) {
              for (const ctx of failureHookResult.additionalContexts) {
                batchAdditionalContexts.push(ctx);
              }
            }
          }
        } else {
          const batchIndex = batchIndexByCallId.get(result.toolCallId);
          const refKey =
            this.toolOutputRegistry != null &&
            batchIndex != null &&
            turn != null
              ? buildReferenceKey(batchIndex, turn)
              : undefined;
          let serialized = serializeToolOutputWithinLimits(
            result.content,
            this.maxToolResultChars,
            this.toolOutputRegistry != null && refKey != null
              ? this.toolOutputRegistry.perOutputLimit
              : 0
          );
          let registryRaw = serialized.registryContent;
          contentString = serialized.content;

          if (hasPostHook) {
            const hookResult = await executeHooks({
              registry: this.hookRegistry!,
              input: {
                hook_event_name: 'PostToolUse',
                runId,
                threadId,
                agentId: this.agentId,
                executionContext: this.executionContext,
                executingAgentId: this.executingAgentId,
                toolName,
                toolInput: request?.args ?? {},
                toolOutput: result.content,
                toolUseId: result.toolCallId,
                stepId: request?.stepId,
                turn: request?.turn,
              },
              sessionId: runId,
              matchQuery: toolName,
            }).catch((): undefined => undefined);
            if (hookResult != null) {
              for (const ctx of hookResult.additionalContexts) {
                batchAdditionalContexts.push(ctx);
              }
            }
            if (hookResult?.updatedOutput != null) {
              serialized = serializeToolOutputWithinLimits(
                hookResult.updatedOutput,
                this.maxToolResultChars,
                this.toolOutputRegistry != null && refKey != null
                  ? this.toolOutputRegistry.perOutputLimit
                  : 0
              );
              registryRaw = serialized.registryContent;
              contentString = serialized.content;
              finalToolOutput = hookResult.updatedOutput;
              /**
               * The hook ACTUALLY rewrote this output: any completion the
               * stream already emitted for the consumed eager result showed
               * the raw content, so un-mark it and let the dispatch below
               * re-emit the corrected version. Hooks that merely observe or
               * add context leave the original emission final — no
               * duplicates.
               */
              eagerCompletionDispatchedIds.delete(result.toolCallId);
            }
          }

          const unresolved = unresolvedByCallId.get(result.toolCallId) ?? [];
          const successRefMeta = this.recordOutputReference(
            registryRunId,
            stripCodeSessionFileSummary(registryRaw),
            refKey,
            unresolved
          );

          toolMessage = new ToolMessage({
            status: 'success',
            name: toolName,
            content: contentString,
            artifact: result.artifact,
            tool_call_id: result.toolCallId,
            ...(successRefMeta != null && {
              additional_kwargs: successRefMeta as Record<string, unknown>,
            }),
          });
        }

        if (
          !eagerCompletionDispatchedIds.has(result.toolCallId) &&
          !earlyCompletionDispatchedIds.has(result.toolCallId)
        ) {
          await this.dispatchStepCompleted(
            result.toolCallId,
            toolName,
            request?.args ?? {},
            contentString,
            config,
            request?.turn,
            resolveToolOutcome(request?.args, outcomeFieldsFromResult(result), {
              isError: result.status === 'error',
            })
          );
        }

        postToolBatchEntryByCallId.set(result.toolCallId, {
          toolName,
          toolInput: request?.args ?? {},
          toolUseId: result.toolCallId,
          stepId: request?.stepId,
          turn: request?.turn,
          status: result.status === 'error' ? 'error' : 'success',
          ...(result.status === 'error'
            ? { error: result.errorMessage ?? 'Unknown error' }
            : { toolOutput: finalToolOutput }),
        });

        messageByCallId.set(result.toolCallId, toolMessage);
      }
    }

    const toolMessages = toolCalls
      .map((call) => messageByCallId.get(call.id!))
      .filter((m): m is ToolMessage => m != null);

    await this.dispatchPostToolBatchAndInjectContext({
      toolCalls,
      entriesByCallId: postToolBatchEntryByCallId,
      batchAdditionalContexts,
      injected,
      runId,
      threadId,
    });

    return { toolMessages, injected };
  }

  /** Run-scoped so another run's session hooks can't flip this run's gate. */
  private canConsumeEagerEventExecution(runId?: string): boolean {
    return (
      this.eventDrivenMode &&
      this.eagerEventToolExecution?.enabled === true &&
      this.hookRegistry?.hasResultAlteringHooks(runId) !== true &&
      this.humanInTheLoop?.enabled !== true
    );
  }

  private takeMatchingEagerEventExecution(
    request: t.ToolCallRequest
  ): t.EagerEventToolExecution | undefined {
    // Static enablement only: a stored record means the reservation-time
    // gates passed and the host already dispatched the execution. Re-checking
    // the dynamic hook gate here could decline consumption after a mid-run
    // registration and send the same call through normal dispatch — executing
    // the tool twice. PostToolUse hooks still process consumed results below.
    if (
      !this.eventDrivenMode ||
      this.eagerEventToolExecution?.enabled !== true
    ) {
      return undefined;
    }

    const execution = this.eagerEventToolExecutions?.get(request.id);
    if (execution == null) {
      return undefined;
    }

    this.eagerEventToolExecutions?.delete(request.id);

    // Only tool identity + canonical args define side-effect identity here.
    // `request.turn` is final-planning metadata; if it drifts between the
    // streamed eager reservation and model-end materialization, consume the
    // same-name/same-args eager result and let the final request drive refs,
    // completion metadata, and PostToolBatch state.
    if (
      execution.toolName !== request.name ||
      !recordArgsEqual(execution.args, request.args)
    ) {
      // Circuit breaker: a prestart/final mismatch means the streamed eager
      // snapshot cannot be trusted for this tool in this run. Without this,
      // the model retries the call, the retry prestarts and diverges the
      // same way, and the run loops to the recursion limit (LibreChat#14371).
      // On an identity mismatch, suppress the eagerly executed name too —
      // otherwise a retry that deterministically streams name A but
      // materializes name B keeps prestarting A (and repeating its side
      // effects) every round.
      this.eagerEventToolSuppressions?.add(request.name);
      this.eagerEventToolSuppressions?.add(execution.toolName);
      if (
        this.participatesInCodeSession(execution.toolName) ||
        execution.toolName === Constants.SKILL_TOOL ||
        execution.toolName === Constants.READ_FILE
      ) {
        request.retainCodeSessionInputs = true;
      }
      // eslint-disable-next-line no-console
      console.warn(
        '[ToolNode] eager prestart args diverged from the final request for ' +
          `tool "${request.name}" (toolCallId=${request.id}); suppressing ` +
          'eager prestart for this tool for the rest of the run'
      );
      const createMismatchOutcome = (): t.EagerEventToolExecutionOutcome => ({
        results: [
          {
            toolCallId: request.id,
            status: 'error',
            content: '',
            errorMessage:
              'Tool call changed after eager execution started; refusing to re-run the tool to avoid duplicate side effects.',
          },
        ],
      });
      return {
        toolCallId: request.id,
        toolName: request.name,
        args: request.args,
        request: execution.request,
        codeSessionBaselineByName: execution.codeSessionBaselineByName,
        promise: execution.promise.then(
          createMismatchOutcome,
          createMismatchOutcome
        ),
      };
    }

    return execution;
  }

  private async resolveEagerEventExecution(
    request: t.ToolCallRequest,
    execution: t.EagerEventToolExecution
  ): Promise<t.ToolExecuteResult[]> {
    const outcome = await execution.promise;
    if (outcome.error != null) {
      throw outcome.error;
    }

    const results = outcome.results.filter(
      (result) => result.toolCallId === request.id
    );
    if (results.length > 0) {
      return results;
    }

    return [
      {
        toolCallId: request.id,
        status: 'error',
        content: '',
        errorMessage:
          'Tool execution completed without a result for this tool call',
      },
    ];
  }

  /**
   * Fires the `PostToolBatch` hook (if registered) and appends the
   * accumulated batch-level `additionalContext` strings to `injected`
   * as a single `HumanMessage`. Entries are materialized in the
   * original `toolCalls` order so hooks correlating outcomes by
   * position (as the type docs promise) see exactly the sequence
   * the model emitted, regardless of when each individual outcome
   * was recorded into the map (deny synchronous, approved
   * post-execution, respond on resume).
   *
   * The PostToolBatch hook's `additionalContexts` flow into the same
   * batch accumulator per-tool hooks already use, so a single
   * batch-level convention message can be injected through one path.
   *
   * Mutates `batchAdditionalContexts` (push from batch hook) and
   * `injected` (push the consolidated HumanMessage). The caller owns
   * those arrays and consumes them right after this returns.
   */
  private async dispatchPostToolBatchAndInjectContext(args: {
    toolCalls: ToolCall[];
    entriesByCallId: Map<string, PostToolBatchEntry>;
    batchAdditionalContexts: string[];
    injected: BaseMessage[];
    runId: string;
    threadId: string | undefined;
  }): Promise<void> {
    const {
      toolCalls,
      entriesByCallId,
      batchAdditionalContexts,
      injected,
      runId,
      threadId,
    } = args;

    const orderedBatchEntries: PostToolBatchEntry[] = [];
    for (const call of toolCalls) {
      const callId = call.id;
      if (callId == null) {
        continue;
      }
      const entry = entriesByCallId.get(callId);
      if (entry != null) {
        orderedBatchEntries.push(entry);
      }
    }
    const hookInjectedMessages: t.InjectedMessage[] = [];
    if (
      this.hookRegistry?.hasHookFor('PostToolBatch', runId) === true &&
      orderedBatchEntries.length > 0
    ) {
      const batchHookResult = await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'PostToolBatch',
          runId,
          threadId,
          agentId: this.agentId,
          executionContext: this.executionContext,
          executingAgentId: this.executingAgentId,
          entries: orderedBatchEntries,
        },
        sessionId: runId,
      }).catch((): undefined => undefined);
      if (batchHookResult != null) {
        for (const ctx of batchHookResult.additionalContexts) {
          batchAdditionalContexts.push(ctx);
        }
        for (const msg of batchHookResult.injectedMessages) {
          hookInjectedMessages.push(msg);
        }
      }
    }

    if (batchAdditionalContexts.length > 0) {
      /**
       * `HumanMessage` carrying a metadata `role: 'system'` marker —
       * see `convertInjectedMessages` for the wider rationale. Anthropic
       * and Google reject mid-conversation `SystemMessage`s, so we use
       * a user-role message and surface the system intent through
       * `additional_kwargs` for hosts inspecting state. The model sees
       * a user message; `role` is metadata only.
       */
      injected.push(
        stampSyntheticProviderMessage(
          new HumanMessage({
            content: batchAdditionalContexts.join('\n\n'),
            additional_kwargs: { role: 'system', source: 'hook' },
          })
        )
      );
    }

    /**
     * Hook-returned `injectedMessages` land AFTER the consolidated context
     * message: one converted `HumanMessage` per entry (role/source kept in
     * `additional_kwargs`), preserving per-message identity so verbatim
     * user speech (e.g. steering) sits closest to the next model call.
     */
    if (hookInjectedMessages.length > 0) {
      try {
        injected.push(...convertInjectedMessages(hookInjectedMessages));
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(
          '[ToolNode] Failed to convert PostToolBatch injectedMessages:',
          e instanceof Error ? e.message : e
        );
      }
    }
  }

  /**
   * Whether `name`'s own schema declares a BUSINESS parameter called
   * `intent` (declared, and not the injected label contract — see
   * `isIntentLabelProperty`). Such a parameter must keep participating in
   * `{{tool…}}` placeholder substitution; only the display label is exempt.
   */
  private toolDeclaresBusinessIntent(name: string): boolean {
    const instance = this.toolMap.get(name) as
      | {
          schema?: {
            shape?: Record<string, unknown>;
            properties?: Record<string, unknown>;
          };
        }
      | undefined;
    const instanceProp =
      instance?.schema?.properties?.[INTENT_ARG] ??
      instance?.schema?.shape?.[INTENT_ARG];
    if (instanceProp != null) {
      return !isIntentLabelProperty(instanceProp);
    }
    const defProp =
      this.toolRegistry?.get(name)?.parameters?.properties?.[INTENT_ARG];
    if (defProp != null) {
      return !isIntentLabelProperty(defProp);
    }
    return false;
  }

  private async dispatchStepCompleted(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>,
    output: string,
    config: RunnableConfig,
    turn?: number,
    outcome?: string
  ): Promise<boolean> {
    const stepId = this.toolCallStepIds?.get(toolCallId) ?? '';
    if (!stepId) {
      // eslint-disable-next-line no-console
      console.warn(
        `[ToolNode] toolCallStepIds missing entry for toolCallId=${toolCallId} (tool=${toolName}). ` +
          'This indicates a race between the stream consumer and graph execution. ' +
          `Map size: ${this.toolCallStepIds?.size ?? 0}`
      );
    }

    const dispatched = await safeDispatchCustomEvent(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: stepId,
          index: turn ?? this.toolUsageCount.get(toolName) ?? 0,
          type: 'tool_call' as const,
          tool_call: {
            args: serializeToolContentBounded(args, this.maxToolResultChars),
            name: toolName,
            id: toolCallId,
            output,
            progress: 1,
            ...(outcome != null && { outcome }),
          } as t.ProcessedToolCall,
          completed_at: Date.now(),
        },
      },
      config
    );
    return dispatched !== false;
  }

  /**
   * Emits the completed run step for a single host-reported result before
   * the batch resolves. Mirrors the batch loop's output formatting exactly;
   * callers gate on the no-hooks/no-HITL configuration, so the raw result
   * content here is also the final content. Returns whether the event was
   * actually dispatched so the caller can fall back to batch-time emission.
   */
  private async dispatchEarlyToolCompletion(
    result: t.ToolExecuteResult,
    request: t.ToolCallRequest,
    config: RunnableConfig
  ): Promise<boolean> {
    const output =
      result.status === 'error'
        ? truncateToolResultContent(
          `Error: ${result.errorMessage ?? 'Unknown error'}\n Please fix your mistakes.`,
          this.maxToolResultChars
        )
        : serializeToolOutputWithinLimits(
          result.content,
          this.maxToolResultChars
        ).content;
    return this.dispatchStepCompleted(
      result.toolCallId,
      request.name,
      request.args,
      output,
      config,
      request.turn,
      resolveToolOutcome(request.args, outcomeFieldsFromResult(result), {
        isError: result.status === 'error',
      })
    );
  }

  /**
   * Execute a group of direct (in-process) tool calls with interrupt-safe
   * ordering, returning outputs aligned 1:1 with `directCalls`.
   *
   * Fast path (the common case): when no call in the group is in
   * `interruptingToolNames`, this is a single `Promise.all` — byte-for-byte
   * the prior behavior, so ordinary batches are unaffected.
   *
   * Interrupt-safe path: when the group contains an interrupting tool (e.g.
   * `ask_user_question`, whose body raises a LangGraph `interrupt()` to
   * collect a human answer), the interrupting calls run as their own awaited
   * group **first**; only after they all settle without interrupting do the
   * remaining (potentially non-idempotent) siblings run. If an interrupting
   * call throws a `GraphInterrupt`, the `await` below rejects and unwinds the
   * whole ToolNode *before* any non-interrupting sibling has started — so a
   * sibling with real side effects (send_email, billing) never executes on
   * the first pass. Terminal interrupting siblings are cached by call id, so
   * LangGraph replay reuses their complete lifecycle output instead of
   * repeating model calls, hooks, or side effects.
   *
   * Without this ordering, a flat `Promise.all` starts every sibling
   * concurrently, so a non-idempotent sibling can complete its side effect
   * before the interrupt unwinds and then run a SECOND time on resume — the
   * duplicate side effect this method exists to prevent. A tool that actually
   * suspends re-enters until it reaches a terminal result; siblings that
   * already settled do not re-enter.
   *
   * `batchIndices[i]` is `directCalls[i]`'s position within the parent
   * ToolNode batch (used for `{{tool<i>turn<n>}}` registration); it is
   * preserved regardless of execution order. `baseContext` carries the
   * batch-scoped fields every call shares; `batchIndex` is filled in
   * per-call here.
   */
  private async runDirectBatchInterruptSafe(
    directCalls: ToolCall[],
    batchIndices: number[],
    config: RunnableConfig,
    baseContext: Omit<RunToolBatchContext<T>, 'batchIndex'>
  ): Promise<(BaseMessage | Command)[]> {
    const settledBatchResults =
      baseContext.replayBatchKey == null
        ? undefined
        : (this.settledDirectResultsByBatch.get(baseContext.replayBatchKey) ??
          new Map<string, SettledDirectToolResult>());
    const cacheResult = (
      call: ToolCall,
      position: number,
      result: SettledDirectToolResult
    ): void => {
      if (baseContext.replayBatchKey == null || settledBatchResults == null) {
        return;
      }
      settledBatchResults.set(
        directReplayKey(
          call,
          baseContext.replayCallPositions?.get(call) ?? batchIndices[position]
        ),
        result
      );
      this.settledDirectResultsByBatch.set(
        baseContext.replayBatchKey,
        settledBatchResults
      );
    };
    const restoreResult = (
      call: ToolCall,
      result: SettledDirectToolResult
    ): SettledDirectToolResult => {
      baseContext.additionalContextsSink?.push(...result.additionalContexts);
      const turn = result.turn;
      if (turn != null) {
        this.recordToolUsageTurn(call.name, turn, call.id);
      }
      const refMeta = isBaseMessage(result.output)
        ? (result.output.additional_kwargs as t.ToolMessageRefMetadata)
        : undefined;
      if (refMeta?._refKey != null && result.referenceContent != null) {
        this.toolOutputRegistry?.set(
          baseContext.batchScopeId,
          refMeta._refKey,
          result.referenceContent
        );
        if (isBaseMessage(result.output)) {
          result = {
            ...result,
            output: new ToolMessage({
              ...(result.output as ToolMessage),
              additional_kwargs: {
                ...result.output.additional_kwargs,
                _refScope: baseContext.batchScopeId,
              },
            }),
          };
        }
      }
      if (
        call.id != null &&
        result.resolvedArgs != null &&
        baseContext.resolvedArgsByCallId != null
      ) {
        baseContext.resolvedArgsByCallId.set(call.id, result.resolvedArgs);
      }
      if (
        result.codeSessionBaseline != null &&
        baseContext.codeSessionBaselineByCallId != null
      ) {
        baseContext.codeSessionBaselineByCallId.set(
          call.id ?? '',
          new Map(result.codeSessionBaseline)
        );
      }
      return result;
    };
    const runOne = async (
      call: ToolCall,
      position: number
    ): Promise<SettledDirectToolResult> => {
      const cachedResult = settledBatchResults?.get(
        directReplayKey(
          call,
          baseContext.replayCallPositions?.get(call) ?? batchIndices[position]
        )
      );
      if (cachedResult != null) {
        if (
          cachedResult.proposal.name !== call.name ||
          !recordArgsEqual(
            cachedResult.proposal.args,
            call.args as Record<string, unknown>
          )
        ) {
          throw new Error(
            'Cannot change an already completed tool call during approval replay'
          );
        }
        return restoreResult(call, cachedResult);
      }
      const additionalContexts: string[] = [];
      const output = await this.runDirectToolWithLifecycleHooks(call, config, {
        ...baseContext,
        batchIndex: batchIndices[position],
        additionalContextsSink: additionalContexts,
      });
      const resolvedArgs =
        call.id == null
          ? undefined
          : baseContext.resolvedArgsByCallId?.get(call.id);
      const codeSessionBaseline = baseContext.codeSessionBaselineByCallId?.get(
        call.id ?? ''
      );
      const result: SettledDirectToolResult = {
        proposal: structuredClone({
          name: call.name,
          args: call.args as Record<string, unknown>,
        }),
        turn:
          call.id == null
            ? undefined
            : (this.toolCallTurns.get(call.id) ??
              this.directPathTurns.get(call.id)),
        output,
        additionalContexts,
        ...(resolvedArgs == null ? {} : { resolvedArgs }),
        ...(codeSessionBaseline == null
          ? {}
          : { codeSessionBaseline: [...codeSessionBaseline.entries()] }),
      };
      const refMeta = isBaseMessage(output)
        ? (output.additional_kwargs as t.ToolMessageRefMetadata)
        : undefined;
      if (refMeta?._refKey != null) {
        result.referenceContent = this.toolOutputRegistry?.get(
          refMeta._refScope,
          refMeta._refKey
        );
      }
      cacheResult(call, position, result);
      return restoreResult(call, result);
    };

    const interrupting = this.interruptingToolNames;
    const hasInterrupting =
      interrupting != null &&
      directCalls.some((call) => interrupting.has(call.name));

    if (!hasInterrupting) {
      const results = await Promise.allSettled(
        directCalls.map((call, i) => runOne(call, i))
      );
      this.throwIfBreakerTripped(config);
      const failed =
        results.find(
          (result) =>
            result.status === 'rejected' && !isGraphInterrupt(result.reason)
        ) ?? results.find((result) => result.status === 'rejected');
      if (failed?.status === 'rejected') {
        throw failed.reason;
      }
      return results.map((result) => {
        if (result.status === 'rejected') {
          throw result.reason;
        }
        return result.value.output;
      });
    }

    const outputs: (BaseMessage | Command)[] = new Array(directCalls.length);
    const interruptingPositions: number[] = [];
    const regularPositions: number[] = [];
    for (let i = 0; i < directCalls.length; i++) {
      if (interrupting.has(directCalls[i].name)) {
        interruptingPositions.push(i);
      } else {
        regularPositions.push(i);
      }
    }

    // Interrupting group first. Wait for every sibling to settle before
    // propagating an interrupt so approved subagents cannot keep running in
    // the background while the parent exposes the next pending approval.
    const interruptingResults = await Promise.allSettled(
      interruptingPositions.map((i) => runOne(directCalls[i], i))
    );
    /** A sibling may have tripped the run-wide breaker while another raised
     * a GraphInterrupt. Safety failures take precedence over approval pauses
     * regardless of input order. */
    this.throwIfBreakerTripped(config);
    let hasInterruptingError = false;
    let interruptingError: unknown;
    for (let i = 0; i < interruptingResults.length; i++) {
      const result = interruptingResults[i];
      if (result.status === 'rejected') {
        if (!hasInterruptingError) {
          hasInterruptingError = true;
          interruptingError = result.reason;
        }
        continue;
      }
      const position = interruptingPositions[i];
      outputs[position] = result.value.output;
    }
    if (hasInterruptingError) {
      if (isGraphInterrupt(interruptingError)) {
        const controllers = new Set<
          NonNullable<ReplayableSubagentTool[typeof SUBAGENT_REPLAY_CONTROLLER]>
        >();
        const parentToolCallIds = new Set<string>();
        for (const call of directCalls) {
          const controller = (
            this.toolMap.get(call.name) as ReplayableSubagentTool | undefined
          )?.[SUBAGENT_REPLAY_CONTROLLER];
          if (controller?.getResumeManifest != null) {
            controllers.add(controller);
          }
          if (call.id != null && call.id !== '') {
            parentToolCallIds.add(call.id);
          }
        }
        const executionsByParentCall = new Map<
          string,
          SubagentResumeManifest['executions'][number]
        >();
        for (const controller of controllers) {
          const manifest = await controller.getResumeManifest?.(
            parentToolCallIds,
            withSubagentReplayBatch(config, baseContext.replayBatchKey)
          );
          if (manifest != null) {
            for (const execution of manifest.executions) {
              executionsByParentCall.set(execution.parentToolCallId, execution);
            }
          }
        }
        const executions = [...executionsByParentCall.values()];
        if (executions.length > 0) {
          const manifest: SubagentResumeManifest = {
            version: 1,
            executions,
          };
          throw new GraphInterrupt(
            interruptingError.interrupts.map((pendingInterrupt) => ({
              ...pendingInterrupt,
              value: attachSubagentResumeManifest(
                pendingInterrupt.value,
                manifest
              ),
            }))
          );
        }
      }
      throw interruptingError;
    }

    /** The breaker can trip while the interrupting group is awaited — e.g.
     * a tool that ignores cancellation and completes normally. Recheck
     * before starting the non-idempotent siblings. */
    this.throwIfBreakerTripped(config);

    // No interrupting call suspended — safe to run the remaining siblings.
    const regularOutputs = await Promise.all(
      regularPositions.map((i) => runOne(directCalls[i], i))
    );
    regularPositions.forEach((i, k) => {
      outputs[i] = regularOutputs[k].output;
    });

    return outputs;
  }

  /**
   * Execute all tool calls via ON_TOOL_EXECUTE event dispatch.
   * Injected messages are placed AFTER ToolMessages to respect provider
   * message ordering (AIMessage tool_calls must be immediately followed
   * by their ToolMessage results).
   *
   * `batchIndices` mirrors `toolCalls` and carries each call's position
   * within the parent batch. `turn` is the per-`run()` batch index
   * captured locally by the caller. Both are threaded so concurrent
   * invocations cannot race on shared mutable state.
   */
  private async executeViaEvent(
    toolCalls: ToolCall[],
    config: RunnableConfig,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: any,
    batchContext: DispatchBatchContext = {}
  ): Promise<T> {
    const { toolMessages, injected } = await this.dispatchToolEvents(
      toolCalls,
      config,
      batchContext
    );
    const outputs: BaseMessage[] = [...toolMessages, ...injected];
    return (Array.isArray(input) ? outputs : { messages: outputs }) as T;
  }

  protected async run(
    input: T,
    config: RunnableConfig,
    referenceReplay: { state?: ToolOutputReferenceState } = {}
  ): Promise<T> {
    /**
     * Breaker read once at batch entry: every downstream path (direct
     * `tool.invoke` runtimes and the ON_TOOL_EXECUTE dispatch) inherits this
     * config, and a graph reset mid-batch must not detach in-flight tools
     * from an already-tripped signal.
     */
    const breakerSignal = this.getBreakerSignal?.();
    const composedSignal = composeAbortSignals(config.signal, breakerSignal);
    /** Immutable batch scope, captured with the signal BEFORE hooks: tools
     * that spawn runs (subagents) bind their child to this controller, so a
     * reset during a PreToolUse hook cannot hand them the new run's one. */
    const batchRunScope = this.getRunScope?.();
    if (composedSignal !== config.signal || batchRunScope != null) {
      config = {
        ...config,
        signal: composedSignal,
        ...(batchRunScope != null && {
          configurable: {
            ...config.configurable,
            [RUN_BREAKER_SCOPE_CONFIG_KEY]: batchRunScope,
          },
        }),
      };
    }
    /** Already-tripped-at-entry: a parallel sibling's breach failed the run
     * before this batch was scheduled. Rethrow before hooks, direct
     * `tool.invoke` calls, or ON_TOOL_EXECUTE dispatch — a tool or host
     * handler that doesn't synchronously inspect an aborted signal would
     * otherwise perform side effects on a failed run. */
    if (
      composedSignal?.aborted === true &&
      (composedSignal.reason instanceof StreamLimitExceededError ||
        composedSignal.reason instanceof PreparedSubagentError)
    ) {
      throw composedSignal.reason;
    }
    this.toolCallTurns.clear();
    /**
     * Per-batch local map for resolved (post-substitution) args.
     * Lives on the stack so concurrent `run()` calls on the same
     * ToolNode cannot read or wipe each other's entries.
     */
    const resolvedArgsByCallId = new Map<string, Record<string, unknown>>();
    const codeSessionBaselineByCallId = new Map<
      string,
      ReadonlyMap<string, string>
    >();
    /** Per-invocation error-completion ownership — see `ToolErrorOwnership`. */
    const errorOwnership = createToolErrorOwnership();
    /**
     * Claim this batch's turn synchronously from the registry (or
     * fall back to 0 when the feature is disabled). The registry is
     * partitioned by scope id so overlapping batches cannot
     * overwrite each other's state even under a shared registry.
     *
     * For anonymous callers (no `run_id` in config), mint a unique
     * per-batch scope id so two concurrent anonymous invocations
     * don't target the same bucket. The scope is threaded down to
     * every subsequent registry call on this batch.
     */
    const incomingRunId = config.configurable?.run_id as string | undefined;
    const batchScopeId = incomingRunId ?? `\0anon-${this.anonBatchCounter++}`;
    const resumedReferenceState = referenceReplay.state;
    const replayInputSnapshot =
      resumedReferenceState == null
        ? undefined
        : this.toolOutputRegistry?.resumeBatch(
          batchScopeId,
          resumedReferenceState
        );
    referenceReplay.state =
      resumedReferenceState ??
      this.toolOutputRegistry?.snapshotState(batchScopeId);
    const turn =
      resumedReferenceState?.turnCounter ??
      this.toolOutputRegistry?.nextTurn(batchScopeId) ??
      0;
    let outputs: (BaseMessage | Command)[];
    let replayBatchKey: string | undefined;
    /** Hoisted from the messages-state branch so the Command tail can carry
     *  the promotion into handoff updates (same-id state copies there would
     *  otherwise overwrite the replacement message). */
    let promotedAiMessage: AIMessage | undefined;
    let invalidCallResults: ToolMessage[] = [];

    if (this.isSendInput(input)) {
      const isLocalTool =
        this.directToolNames?.has(input.lg_tool_call.name) === true ||
        this.shouldHandleUnknownHandoffLocally(input.lg_tool_call.name);
      if (this.eventDrivenMode && !isLocalTool) {
        return this.executeViaEvent([input.lg_tool_call], config, input, {
          batchIndices: [0],
          turn,
          batchScopeId,
          preBatchSnapshot: replayInputSnapshot,
        });
      }
      // Same per-batch sink the message-state branches use so
      // direct-path PreToolUse/PostToolUse/Failure additionalContexts
      // surface here too. Codex P2 [44] — round 14 added the sink to
      // both message-state branches but missed this Send-input
      // branch, so direct tools dispatched via Send (a supported
      // input shape) still silently dropped hook context.
      const directAdditionalContexts: string[] = [];
      // Mirror langgraph's prebuilt ToolNode: the Send-input state is
      // the input minus the `lg_tool_call` envelope key.
      const { lg_tool_call: _sendToolCall, ...sendState } = input;
      const sendMessages = (sendState as { messages?: unknown }).messages;
      const sendAssistantBatch = Array.isArray(sendMessages)
        ? findAssistantBatch(sendMessages as BaseMessage[])
        : undefined;
      const sendReplayBatchKey =
        sendAssistantBatch == null || getToolBatchReplayScope(config) == null
          ? undefined
          : getAssistantBatchReplayKey(
            sendAssistantBatch,
            getToolBatchReplayScope(config)
          );
      const sendOutput = await this.runDirectToolWithLifecycleHooks(
        input.lg_tool_call,
        config,
        {
          batchIndex: 0,
          turn,
          batchScopeId,
          preBatchSnapshot: replayInputSnapshot,
          resolvedArgsByCallId,
          codeSessionBaselineByCallId,
          errorOwnership,
          additionalContextsSink: directAdditionalContexts,
          replayBatchKey: sendReplayBatchKey,
          runInput: sendState as T,
        }
      );
      outputs =
        directAdditionalContexts.length > 0
          ? [
            sendOutput,
            stampSyntheticProviderMessage(
              new HumanMessage({
                content: directAdditionalContexts.join('\n\n'),
                // Match the event-driven path's marker so hosts /
                // model-side annotators treat this as system intent
                // rather than ordinary user text. Codex P2 [46].
                additional_kwargs: { role: 'system', source: 'hook' },
              })
            ),
          ]
          : [sendOutput];
      await this.handleRunToolCompletions(
        [input.lg_tool_call],
        // Pass only the tool output to completion handling; the
        // HumanMessage isn't a tool result.
        [sendOutput],
        config,
        resolvedArgsByCallId,
        codeSessionBaselineByCallId,
        errorOwnership
      );
    } else {
      let messages: BaseMessage[];
      if (Array.isArray(input)) {
        messages = input;
      } else if (this.isMessagesState(input)) {
        messages = input.messages;
      } else {
        throw new Error(
          'ToolNode only accepts BaseMessage[] or { messages: BaseMessage[] } as input.'
        );
      }

      const toolMessageIds: Set<string> = new Set(
        messages
          .filter((msg) => msg._getType() === 'tool')
          .map((msg) => (msg as ToolMessage).tool_call_id)
      );

      const assistantBatch = findAssistantBatch(messages);
      if (assistantBatch == null) {
        throw new Error('ToolNode only accepts AIMessages as input.');
      }
      const aiMessage = assistantBatch.message;
      replayBatchKey =
        getToolBatchReplayScope(config) == null
          ? undefined
          : getAssistantBatchReplayKey(
            assistantBatch,
            getToolBatchReplayScope(config)
          );

      if (this.loadRuntimeTools) {
        const { tools, toolMap } = this.loadRuntimeTools(
          aiMessage.tool_calls ?? []
        );
        this.toolMap =
          toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
        this.applyToolExecutionOverrides();
      }

      const replayCallPositions = new Map(
        (aiMessage.tool_calls ?? []).map((call, i) => [call, i])
      );
      const filteredCalls =
        aiMessage.tool_calls?.filter((call) => {
          /**
           * Filter out:
           * 1. Already processed tool calls (present in toolMessageIds)
           * 2. Server tool calls (e.g., web_search with IDs starting with 'srvtoolu_')
           *    which are executed by the provider's API and don't require invocation
           */
          return (
            (call.id == null ||
              call.id === '' ||
              !toolMessageIds.has(call.id)) &&
            !(
              call.id?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) ??
              false
            )
          );
        }) ?? [];

      /**
       * Synthesize error results for `invalid_tool_calls` — calls whose
       * streamed args never collapsed into a JSON object (`@langchain/core`
       * files them separately with `error: "Malformed args."`, and they never
       * enter `tool_calls`). Their `tool_use` blocks still ride the AI
       * message content the provider receives, so skipping them leaves a
       * `tool_use` with no `tool_result` and the NEXT model call is rejected
       * (Anthropic 400 INVALID_TOOL_RESULTS) — fatal on HITL resume, where
       * the paused AI message is replayed from the checkpoint. Only calls
       * with an id can be paired (and only those produce the 400); the
       * already-processed / server-tool filters mirror `filteredCalls`.
       */
      /**
       * Invalid-call handling only applies when the replacement AI message
       * can actually land: the MESSAGES-STATE input form (the returned
       * messages flow through `messagesStateReducer` and the replacement
       * upserts by id) with an id-bearing AI message. A `BaseMessage[]`
       * caller receives a plain output LIST it appends to its own history —
       * the replacement would duplicate the assistant turn — and an id-less
       * message cannot be upserted. In both cases the synthesized results
       * are skipped TOO: results and promotion are all-or-nothing, or the
       * next provider request would carry an output whose call the
       * converters never emit (the inverted rejection). Such callers keep
       * the untouched status quo.
       */
      const canPromoteInvalidCalls =
        !Array.isArray(input) &&
        typeof aiMessage.id === 'string' &&
        aiMessage.id.length > 0;
      const attributableInvalidCalls = !canPromoteInvalidCalls
        ? []
        : (aiMessage.invalid_tool_calls ?? []).filter(
          (call) =>
            call.id != null &&
              call.id !== '' &&
              !toolMessageIds.has(call.id) &&
              !call.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        );
      invalidCallResults = attributableInvalidCalls.map(
        (call) =>
          new ToolMessage({
            status: 'error',
            content: truncateToolResultContent(
              `Error: ${call.error ?? 'Malformed tool call arguments.'} ` +
                'The tool call input could not be parsed as a JSON object; the tool was not run.\n Please fix your mistakes.',
              this.maxToolResultChars
            ),
            name: normalizeInvalidCallName(call.name),
            tool_call_id: call.id!,
          })
      );

      /**
       * Promote the answered invalid calls into well-formed `tool_calls` on a
       * REPLACEMENT copy of the AI message (`messagesStateReducer` upserts by
       * id). Without this, provider converters that rebuild the call side of
       * the wire from `tool_calls` — OpenAI Completions `tool_calls`, OpenAI
       * Responses `function_call` items, Gemini/Bedrock function-call parts —
       * drop the invalid call while the synthesized result above still
       * references it, inverting the dangling-pair rejection (an output whose
       * call is missing). Promoting at this single seam keeps the call and
       * result sides agreeing for EVERY provider; args become `{}` (the raw
       * string never parsed — the paired error result tells the model why).
       * Skipped when the message has no id: the reducer would append a
       * duplicate instead of replacing, which is worse than the dangle.
       */
      promotedAiMessage =
        attributableInvalidCalls.length > 0
          ? new AIMessage({
            id: aiMessage.id,
            content: sanitizeInvalidToolUseBlocks(
              aiMessage.content,
              attributableInvalidCalls
            ),
            name: aiMessage.name,
            additional_kwargs: aiMessage.additional_kwargs,
            response_metadata: aiMessage.response_metadata,
            usage_metadata: aiMessage.usage_metadata,
            tool_calls: [
              ...(aiMessage.tool_calls ?? []),
              ...attributableInvalidCalls.map((call) => ({
                id: call.id!,
                name: normalizeInvalidCallName(call.name),
                args: {},
                type: 'tool_call' as const,
              })),
            ],
            invalid_tool_calls: (aiMessage.invalid_tool_calls ?? []).filter(
              (call) => !attributableInvalidCalls.includes(call)
            ),
          })
          : undefined;

      if (this.eventDrivenMode && filteredCalls.length > 0) {
        const directToolNames = this.directToolNames;
        const hasRegisteredHandoffTool = this.hasRegisteredHandoffTool();

        const directEntries: Array<{ call: ToolCall; batchIndex: number }> = [];
        const eventEntries: Array<{ call: ToolCall; batchIndex: number }> = [];
        for (let i = 0; i < filteredCalls.length; i++) {
          const call = filteredCalls[i];
          const entry = { call, batchIndex: i };
          if (
            directToolNames?.has(call.name) === true ||
            this.shouldHandleUnknownHandoffLocally(
              call.name,
              hasRegisteredHandoffTool
            )
          ) {
            directEntries.push(entry);
          } else {
            eventEntries.push(entry);
          }
        }

        if (directEntries.length === 0 && invalidCallResults.length === 0) {
          return this.executeViaEvent(filteredCalls, config, input, {
            batchIndices: eventEntries.map((entry) => entry.batchIndex),
            turn,
            batchScopeId,
            preBatchSnapshot: replayInputSnapshot,
          });
        }

        const directCalls = directEntries.map((e) => e.call);
        const directIndices = directEntries.map((e) => e.batchIndex);
        const eventCalls = eventEntries.map((e) => e.call);
        const eventIndices = eventEntries.map((e) => e.batchIndex);
        const approvalScopeCallIds = new Set(
          filteredCalls.flatMap((call) => (call.id == null ? [] : [call.id]))
        );

        /**
         * Snapshot the event calls' args against the *pre-batch*
         * registry state synchronously, before any await runs. The
         * directs are then awaited first (preserving fail-fast
         * semantics — a thrown error in a direct tool, e.g. with
         * `handleToolErrors=false` or a `GraphInterrupt`, aborts
         * before we dispatch any event-driven tools to the host).
         * Because the event args were captured pre-await, they stay
         * isolated from same-turn direct outputs that register
         * during the await.
         */
        const preResolvedEventArgs = new Map<
          string,
          { resolved: Record<string, unknown>; unresolved: string[] }
        >();
        /**
         * Take a frozen snapshot of the registry state before any
         * direct registrations land. The snapshot resolves
         * placeholders against this point-in-time view, so a
         * `PreToolUse` hook later rewriting event args via
         * `updatedInput` can introduce placeholders that resolve
         * cross-batch (against prior runs) without ever picking up
         * same-turn direct outputs.
         */
        const preBatchSnapshot =
          replayInputSnapshot ??
          this.toolOutputRegistry?.snapshot(batchScopeId);
        if (preBatchSnapshot != null) {
          for (const entry of eventEntries) {
            if (entry.call.id != null) {
              const { resolved, unresolved } = preBatchSnapshot.resolve(
                entry.call.args as Record<string, unknown>,
                {
                  substituteIntentKey: this.toolDeclaresBusinessIntent(
                    entry.call.name
                  ),
                }
              );
              preResolvedEventArgs.set(entry.call.id, {
                resolved: resolved as Record<string, unknown>,
                unresolved,
              });
            }
          }
        }

        // Per-batch sink for direct-path hook additionalContexts
        // (Codex P2 #39). Materialized as a HumanMessage at end-of-
        // batch so the next model turn sees the injected context,
        // matching the event path's `injected[]` shape.
        const directAdditionalContexts: string[] = [];
        const directOutputs: (BaseMessage | Command)[] =
          directCalls.length > 0
            ? await this.runDirectBatchInterruptSafe(
              directCalls,
              directIndices,
              config,
              {
                turn,
                batchScopeId,
                resolvedArgsByCallId,
                codeSessionBaselineByCallId,
                errorOwnership,
                preBatchSnapshot,
                additionalContextsSink: directAdditionalContexts,
                replayBatchKey,
                replayCallPositions,
                approvalScopeCallIds,
                runInput: input as T,
              }
            )
            : [];

        const settledDirectResults =
          replayBatchKey == null
            ? undefined
            : this.settledDirectResultsByBatch.get(replayBatchKey);
        const unhandledDirectCalls: ToolCall[] = [];
        const unhandledDirectOutputs: (BaseMessage | Command)[] = [];
        for (let i = 0; i < directCalls.length; i++) {
          const settled = settledDirectResults?.get(
            directReplayKey(
              directCalls[i],
              replayCallPositions.get(directCalls[i])!
            )
          );
          if (settled?.completionHandled === true) {
            continue;
          }
          unhandledDirectCalls.push(directCalls[i]);
          unhandledDirectOutputs.push(directOutputs[i]);
        }

        if (unhandledDirectCalls.length > 0) {
          await this.handleRunToolCompletions(
            unhandledDirectCalls,
            unhandledDirectOutputs,
            config,
            resolvedArgsByCallId,
            codeSessionBaselineByCallId,
            errorOwnership
          );
          for (let i = 0; i < directCalls.length; i++) {
            const settled = settledDirectResults?.get(
              directReplayKey(
                directCalls[i],
                replayCallPositions.get(directCalls[i])!
              )
            );
            if (settled != null) {
              settled.completionHandled = true;
            }
          }
        }

        const eventResult =
          eventCalls.length > 0
            ? await this.dispatchToolEvents(eventCalls, config, {
              batchIndices: eventIndices,
              turn,
              batchScopeId,
              preResolvedArgs: preResolvedEventArgs,
              preBatchSnapshot,
            })
            : {
              toolMessages: [] as ToolMessage[],
              injected: [] as BaseMessage[],
            };

        const directInjected: BaseMessage[] =
          directAdditionalContexts.length > 0
            ? [
              stampSyntheticProviderMessage(
                new HumanMessage({
                  content: directAdditionalContexts.join('\n\n'),
                  // System-role metadata to match the event-driven
                  // path so policy/recovery guidance is treated
                  // consistently regardless of whether the tool ran
                  // direct or dispatched. Codex P2 [46].
                  additional_kwargs: { role: 'system', source: 'hook' },
                })
              ),
            ]
            : [];
        outputs = [
          // Replacement AI message first (reducer upsert-by-id), then results.
          ...(promotedAiMessage != null ? [promotedAiMessage] : []),
          ...directOutputs,
          ...eventResult.toolMessages,
          // Synthesized invalid-call errors sit with the real tool results,
          // before injected context, to keep provider tool-result adjacency.
          ...invalidCallResults,
          ...directInjected,
          ...eventResult.injected,
        ];
      } else {
        // Same per-batch pre-snapshot as the mixed path, applied to
        // the all-direct case so `Promise.all`-induced ordering can't
        // leak a sibling's just-registered output into a sister
        // call's args mid-await (Codex P1 #18).
        const preBatchSnapshot =
          replayInputSnapshot ??
          this.toolOutputRegistry?.snapshot(batchScopeId);
        const directAdditionalContexts: string[] = [];
        const approvalScopeCallIds = new Set(
          filteredCalls.flatMap((call) => (call.id == null ? [] : [call.id]))
        );
        const toolOutputs = await this.runDirectBatchInterruptSafe(
          filteredCalls,
          filteredCalls.map((_call, i) => i),
          config,
          {
            turn,
            batchScopeId,
            resolvedArgsByCallId,
            codeSessionBaselineByCallId,
            errorOwnership,
            preBatchSnapshot,
            additionalContextsSink: directAdditionalContexts,
            replayBatchKey,
            replayCallPositions,
            approvalScopeCallIds,
            runInput: input as T,
          }
        );
        await this.handleRunToolCompletions(
          filteredCalls,
          toolOutputs,
          config,
          resolvedArgsByCallId,
          codeSessionBaselineByCallId,
          errorOwnership
        );
        // Append accumulated additionalContexts as a single
        // HumanMessage so the next model turn sees them. Codex P2 #39.
        const promotedPrefix =
          promotedAiMessage != null ? [promotedAiMessage] : [];
        outputs =
          directAdditionalContexts.length > 0
            ? [
              ...promotedPrefix,
              ...toolOutputs,
              ...invalidCallResults,
              stampSyntheticProviderMessage(
                new HumanMessage({
                  content: directAdditionalContexts.join('\n\n'),
                  // Same system-role marker the event-driven path
                  // uses so direct vs dispatched is invisible to
                  // downstream consumers. Codex P2 [46].
                  additional_kwargs: { role: 'system', source: 'hook' },
                })
              ),
            ]
            : [...promotedPrefix, ...toolOutputs, ...invalidCallResults];
      }

      /**
       * Resolve the streamed tool-call cards for invalid calls, best-effort.
       * Runs AFTER the direct batch settled: on an interrupting first pass
       * this line is unreachable (the node unwound), so interrupt/resume
       * flows emit the completion exactly once — same reasoning as the
       * deferred blocked-call side effects. Skipped when the stream never
       * registered a step for the call (non-streaming providers), where a
       * completion could not be routed to a card anyway.
       */
      for (const result of invalidCallResults) {
        const invalidStepId = this.toolCallStepIds?.get(result.tool_call_id);
        if (invalidStepId == null || invalidStepId === '') {
          continue;
        }
        await this.dispatchStepCompleted(
          result.tool_call_id,
          result.name ?? 'unknown',
          {},
          typeof result.content === 'string' ? result.content : '',
          config
        );
      }
    }

    if (!outputs.some(isCommand)) {
      if (replayBatchKey != null) {
        this.settledDirectResultsByBatch.delete(replayBatchKey);
      }
      return (Array.isArray(input) ? outputs : { messages: outputs }) as T;
    }

    /**
     * Carry the invalid-call promotion into handoff commands. A handoff
     * tool's Command snapshots `update.messages` from the PRE-promotion
     * state (MultiAgentGraph builds a filtered same-id copy of the original
     * AI message), and commands apply after the sibling reducer updates —
     * so the stale copy would overwrite the replacement message, and a
     * Send handoff's child state could omit the synthesized results
     * entirely. Patch each command's same-id AI message with the promotion
     * and append any missing synthesized results.
     */
    if (promotedAiMessage != null) {
      outputs = outputs.map((output) =>
        isCommand(output)
          ? patchCommandUpdateForPromotedInvalidCalls(
            output,
              promotedAiMessage!,
              invalidCallResults
          )
          : output
      );
    }

    const combinedOutputs: (
      | { messages: BaseMessage[] }
      | BaseMessage[]
      | Command
    )[] = [];
    let parentCommand: Command | null = null;

    /**
     * Collect handoff commands (Commands with string goto and Command.PARENT)
     * for potential parallel handoff aggregation
     */
    const handoffCommands: Command[] = [];
    const nonCommandOutputs: BaseMessage[] = [];

    for (const output of outputs) {
      if (isCommand(output)) {
        if (
          output.graph === Command.PARENT &&
          Array.isArray(output.goto) &&
          output.goto.every((send): send is Send => isSend(send))
        ) {
          /** Aggregate Send-based commands */
          if (parentCommand) {
            (parentCommand.goto as Send[]).push(...(output.goto as Send[]));
          } else {
            parentCommand = new Command({
              graph: Command.PARENT,
              goto: output.goto,
            });
          }
        } else if (output.graph === Command.PARENT) {
          /**
           * Handoff Command with destination.
           * Handle both string ('agent') and array (['agent']) formats.
           * Collect for potential parallel aggregation.
           */
          const goto = output.goto;
          const isSingleStringDest = typeof goto === 'string';
          const isSingleArrayDest =
            Array.isArray(goto) &&
            goto.length === 1 &&
            typeof goto[0] === 'string';

          if (isSingleStringDest || isSingleArrayDest) {
            handoffCommands.push(output);
          } else {
            /** Multi-destination or other command - pass through */
            combinedOutputs.push(output);
          }
        } else {
          /** Other commands - pass through */
          combinedOutputs.push(output);
        }
      } else {
        nonCommandOutputs.push(output);
        combinedOutputs.push(
          Array.isArray(input) ? [output] : { messages: [output] }
        );
      }
    }

    /**
     * Handle handoff commands - convert to Send objects for parallel execution
     * when multiple handoffs are requested
     */
    if (handoffCommands.length > 1) {
      /**
       * Multiple parallel handoffs - convert to Send objects.
       * Each Send carries its own state with the appropriate messages.
       * This enables LLM-initiated parallel execution when calling multiple
       * transfer tools simultaneously.
       */

      /** Collect all destinations for sibling tracking */
      const allDestinations = handoffCommands.map((cmd) => {
        const goto = cmd.goto;
        return typeof goto === 'string' ? goto : (goto as string[])[0];
      });
      const parallelBatch = nanoid();
      const parallelGroupId = getRuntimeHandoffGroupId(parallelBatch);

      const sends = handoffCommands.map((cmd, idx) => {
        const destination = allDestinations[idx];
        /** Get siblings (other destinations, not this one) */
        const siblings = allDestinations.filter((d) => d !== destination);

        const update = cmd.update as { messages?: BaseMessage[] } | undefined;
        const handoffMessage = update?.messages
          ? findHandoffMessage(update.messages, destination)
          : undefined;
        if (handoffMessage) {
          handoffMessage.additional_kwargs.handoff_parallel_siblings = siblings;
          handoffMessage.additional_kwargs[Constants.HANDOFF_PARALLEL_BATCH] =
            parallelBatch;
          handoffMessage.additional_kwargs[Constants.HANDOFF_GROUP_ID] =
            parallelGroupId;
        }

        return new Send(destination, cmd.update);
      });

      const parallelCommand = new Command({
        graph: Command.PARENT,
        goto: sends,
      });
      combinedOutputs.push(parallelCommand);
    } else if (handoffCommands.length === 1) {
      /** Single handoff - pass through as-is */
      combinedOutputs.push(handoffCommands[0]);
    }

    if (parentCommand) {
      combinedOutputs.push(parentCommand);
    }

    if (replayBatchKey != null) {
      this.settledDirectResultsByBatch.delete(replayBatchKey);
    }
    if (!Array.isArray(input) && !this.isSendInput(input)) {
      this.handoffRouting?.finalize(
        combinedOutputs.filter(isCommand),
        input as t.BaseGraphState,
        config
      );
    }
    return combinedOutputs as T;
  }

  private isSendInput(input: unknown): input is { lg_tool_call: ToolCall } {
    return (
      typeof input === 'object' && input != null && 'lg_tool_call' in input
    );
  }

  private isMessagesState(
    input: unknown
  ): input is { messages: BaseMessage[] } {
    return (
      typeof input === 'object' &&
      input != null &&
      'messages' in input &&
      Array.isArray((input as { messages: unknown }).messages) &&
      (input as { messages: unknown[] }).messages.every(isBaseMessage)
    );
  }
}

function areToolCallsInvoked(
  message: AIMessage,
  invokedToolIds?: Set<string>
): boolean {
  if (!invokedToolIds || invokedToolIds.size === 0) return false;
  return (
    message.tool_calls?.every(
      (toolCall) => toolCall.id != null && invokedToolIds.has(toolCall.id)
    ) ?? false
  );
}

/**
 * Normalize the `tool_use` content blocks of promoted invalid calls so the
 * replacement AI message is valid on EVERY provider surface, not just
 * `tool_calls`. Anthropic formats an array-content AI message from its blocks
 * verbatim, and a call whose streamed `input_json` never parsed leaves the
 * block's `input` as the raw accumulated STRING — replayed as-is, the API
 * rejects it with `tool_use.input: Input should be an object` before pairing
 * is even checked. Blocks matching a promoted call id get `input: {}`
 * (mirroring the promoted args); everything else passes through untouched.
 * String content (OpenAI-style) is returned as-is.
 */
function sanitizeInvalidToolUseBlocks(
  content: AIMessage['content'],
  promotedCalls: ReadonlyArray<{ id?: string; name?: string }>
): AIMessage['content'] {
  if (!Array.isArray(content)) {
    return content;
  }
  const promotedNamesById = new Map(
    promotedCalls
      .filter((call) => call.id != null)
      .map((call) => [call.id!, normalizeInvalidCallName(call.name)])
  );
  return content.map((block) => {
    if (
      typeof block !== 'object' ||
      (block as { type?: string } | null)?.type !== 'tool_use'
    ) {
      return block;
    }
    const toolUse = block as { id?: string; name?: string; input?: unknown };
    if (toolUse.id == null || !promotedNamesById.has(toolUse.id)) {
      return block;
    }
    const inputIsObject =
      typeof toolUse.input === 'object' &&
      toolUse.input != null &&
      !Array.isArray(toolUse.input);
    /** `name` normalizes with the SAME fallback the promoted `tool_calls`
     *  entry uses — a nameless block would fail provider validation on its
     *  own even with a valid input. */
    const nameIsValid = typeof toolUse.name === 'string' && toolUse.name !== '';
    if (inputIsObject && nameIsValid) {
      return block;
    }
    return {
      ...block,
      ...(inputIsObject ? {} : { input: {} }),
      ...(nameIsValid ? {} : { name: promotedNamesById.get(toolUse.id) }),
    };
  });
}

/**
 * Name fallback for attributable invalid calls, shared by every surface that
 * materializes them (synthesized result, promoted tool_calls entry, sanitized
 * block, handoff patch): `''` is normalized like `undefined` — providers
 * reject nameless calls, so an empty string would defeat the promotion.
 *
 * INVARIANT MAP — a tool call lives in several parallel representations, and
 * any surface that materializes, copies, filters, routes on, or reports one
 * must keep ALL of them agreeing (`tool_calls`, `invalid_tool_calls`,
 * provider content blocks, paired results). The attribution predicate is:
 * id-bearing (non-empty), non-server (`srvtoolu_`), unanswered
 * (`toolMessageIds`), messages-state input, id-bearing AI message. Surfaces
 * that apply it today — extend this list when adding another:
 *   - `run()`'s `canPromoteInvalidCalls` gate + attributable filter
 *   - `toolsCondition`'s invalid-only / server-mix routing branch
 *   - `sanitizeInvalidToolUseBlocks` (block input AND name)
 *   - `patchCommandUpdateForPromotedInvalidCalls` (handoff snapshots)
 *   - `processHandoffReception`'s transfer-block filtering (MultiAgentGraph)
 *   - `findPendingToolCalls` in langfuseTraceShaping (span claims)
 *   - `serializeMessage`/`deserializeMessage` (session round-trip keeps
 *     `invalid_tool_calls` with the content blocks they repair)
 */
function normalizeInvalidCallName(name: string | undefined | null): string {
  return name != null && name !== '' ? name : 'unknown';
}

/**
 * Rewrite a handoff Command's `update.messages` so the invalid-call promotion
 * survives into the child state: the same-id AI message copy (snapshotted
 * pre-promotion by the handoff tool) gets the sanitized content, the promoted
 * `tool_calls` entries for the answered invalid calls, and the leftover
 * `invalid_tool_calls`; synthesized results missing from the update are
 * appended so the child's history keeps every call/result pair. The update
 * copy's own `tool_calls` narrowing (parallel handoffs filter to a single
 * call) is preserved. Commands without a same-id AI message pass through.
 */
function patchCommandUpdateForPromotedInvalidCalls(
  command: Command,
  promoted: AIMessage,
  invalidResults: ToolMessage[]
): Command {
  const update = command.update as { messages?: BaseMessage[] } | undefined;
  const messages = update?.messages;
  if (
    !Array.isArray(messages) ||
    promoted.id == null ||
    invalidResults.length === 0
  ) {
    return command;
  }
  const hasSameIdAiMessage = messages.some(
    (msg) => isAIMessage(msg) && msg.id === promoted.id
  );
  if (!hasSameIdAiMessage) {
    return command;
  }
  const next: BaseMessage[] = messages.map((msg) => {
    if (!isAIMessage(msg) || msg.id !== promoted.id) {
      return msg;
    }
    const existingIds = new Set((msg.tool_calls ?? []).map((call) => call.id));
    const promotedEntries = invalidResults
      .filter((result) => !existingIds.has(result.tool_call_id))
      .map((result) => ({
        id: result.tool_call_id,
        name: normalizeInvalidCallName(result.name),
        args: {},
        type: 'tool_call' as const,
      }));
    return new AIMessage({
      id: msg.id,
      content: promoted.content,
      name: msg.name,
      additional_kwargs: msg.additional_kwargs,
      response_metadata: msg.response_metadata,
      usage_metadata: msg.usage_metadata,
      tool_calls: [...(msg.tool_calls ?? []), ...promotedEntries],
      invalid_tool_calls: promoted.invalid_tool_calls,
    });
  });
  const presentResultIds = new Set(
    next
      .filter((msg): msg is ToolMessage => msg._getType() === 'tool')
      .map((msg) => msg.tool_call_id)
  );
  const missingResults = invalidResults.filter(
    (result) => !presentResultIds.has(result.tool_call_id)
  );
  return new Command({
    graph: command.graph,
    goto: command.goto,
    resume: command.resume,
    update: { ...update, messages: [...next, ...missingResults] },
  });
}

/**
 * Whether the message carries an `invalid_tool_calls` entry ToolNode can pair a
 * synthesized error result with (id-bearing, non-server). Shared by the routing
 * condition below so an invalid-only turn still enters ToolNode — otherwise the
 * malformed `tool_use` block is committed with no `tool_result` and the next
 * model call is rejected by pairing-strict providers.
 */
function hasAttributableInvalidToolCalls(message: AIMessage): boolean {
  return (
    message.invalid_tool_calls?.some(
      (call) =>
        call.id != null &&
        call.id !== '' &&
        !call.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
    ) ?? false
  );
}

export function toolsCondition<T extends string>(
  state: BaseMessage[] | typeof MessagesAnnotation.State,
  toolNode: T,
  invokedToolIds?: Set<string>
): T | typeof END {
  const messages = Array.isArray(state) ? state : state.messages;
  const message = messages[messages.length - 1] as AIMessage | undefined;

  if (
    message &&
    'tool_calls' in message &&
    (message.tool_calls?.length ?? 0) > 0 &&
    !areToolCallsInvoked(message, invokedToolIds)
  ) {
    return toolNode;
  }
  /**
   * The valid calls (if any) did not route above, but ToolNode still owes any
   * malformed calls their synthesized error results. Route when EVERY valid
   * call is provider-server-executed (`srvtoolu_` — ToolNode's batch filter
   * excludes those before execution, so nothing re-runs): that covers both the
   * invalid-only turn and the Anthropic server-call + malformed-client-call
   * mix, where `handleAnthropicSearchResults` marks the server call invoked
   * and the first branch declines. A valid NON-server call that was invoked
   * externally stays conservative (no routing) — ToolNode does not filter on
   * `invokedToolIds`, so entering it would re-execute that call.
   *
   * Mirrors ToolNode's own gating exactly, or the routed turn would no-op and
   * bounce back to the model with the dangle intact: array-state graphs get a
   * plain output list (no reducer upsert — ToolNode skips invalid handling
   * there), and an id-less message cannot take the replacement upsert either.
   */
  if (
    !Array.isArray(state) &&
    message &&
    typeof message.id === 'string' &&
    message.id.length > 0 &&
    hasAttributableInvalidToolCalls(message) &&
    (message.tool_calls ?? []).every(
      (call) =>
        call.id?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) === true
    )
  ) {
    return toolNode;
  }
  return END;
}

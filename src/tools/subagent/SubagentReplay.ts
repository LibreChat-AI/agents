import type { ToolCall, ToolMessage } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolOutputReferenceState } from '@/tools/toolOutputReferences';
import type { ToolApprovalReplaySnapshot } from '@/hooks';
import type {
  FadingTier,
  FadingTiers,
  RunStepResumeState,
  ToolSessionContext,
} from '@/types';
import { isFadingTier, isLegacyFadingTier } from '@/messages/fading';
import {
  attachRunStepResumeState,
  getRunStepResumeState,
  isRunStepResumeState,
  stripRunStepResumeState,
} from '@/tools/runStepResume';

export const SUBAGENT_RESUME_MANIFEST_CONFIG_KEY =
  '__librechat_subagent_resume_manifest';
export const SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY =
  '__librechat_subagent_resume_attempt';
export const SUBAGENT_PARENT_BATCH_CONFIG_KEY =
  '__librechat_subagent_parent_batch';

const SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY =
  '__librechat_subagent_resume_manifest';
const SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY =
  '__librechat_subagent_resume_payload';
const SUBAGENT_RESUME_WRAPPER_VERSION_KEY =
  '__librechat_subagent_resume_wrapper';
const SUBAGENT_RESUME_PRIVATE_PAYLOAD_KEYS = [
  SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY,
  SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY,
  SUBAGENT_RESUME_WRAPPER_VERSION_KEY,
] as const;
const MAX_RESUME_MANIFEST_DEPTH = 32;

export interface SubagentCheckpointReference {
  threadId: string;
  checkpointId: string;
  checkpointNs: string;
}

export interface SubagentToolCallStepReference {
  toolCallId: string;
  stepId: string;
}

export interface SubagentToolSessionReference {
  toolName: string;
  context: ToolSessionContext;
}

export interface SubagentToolNodeResumeState {
  stateKey: string;
  toolUsageCounts: Array<{ toolName: string; count: number }>;
  directPathTurns: Array<{ toolCallId: string; turn: number }>;
}

export interface SubagentEagerToolUsageState {
  agentId: string;
  toolUsageCounts: Array<{ toolName: string; count: number }>;
}

export interface SubagentGraphResumeState {
  toolCallSteps: SubagentToolCallStepReference[];
  toolSessions: SubagentToolSessionReference[];
  toolNodes: SubagentToolNodeResumeState[];
  eagerToolUsage: SubagentEagerToolUsageState[];
  eagerToolSuppressions: string[];
  runStepState?: RunStepResumeState;
  toolOutputReferences?: ToolOutputReferenceState;
  fadingTier?: FadingTier;
  fadingTiers?: FadingTiers;
}

/** Private checkpoint payload linking a parent pause to an exact child state. */
export interface SubagentResumeExecution {
  parentToolCallId: string;
  childRunId: string;
  /** Effective child type bound when this execution first ran. */
  subagentType?: string;
  /** Lazy child revision bound when this execution first resolved. */
  configId?: string;
  approvalExecutionScope: string;
  checkpoints: SubagentCheckpointReference[];
  graphState: SubagentGraphResumeState;
  approvalReplays: ToolApprovalReplaySnapshot[];
  descendant?: SubagentResumeManifest;
}

/** Private checkpoint payload linking a parent pause to every child state. */
export interface SubagentResumeManifest {
  version: 1;
  executions: SubagentResumeExecution[];
}

type PayloadWithSubagentResumeManifest = {
  [SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY]?: unknown;
};

type WrappedSubagentResumePayload = PayloadWithSubagentResumeManifest & {
  [SUBAGENT_RESUME_WRAPPER_VERSION_KEY]: 1;
  [SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY]: unknown;
};

function isString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function isNonnegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function isApprovalReplaySnapshot(
  value: unknown
): value is ToolApprovalReplaySnapshot {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const snapshot = value as Partial<ToolApprovalReplaySnapshot>;
  const key = snapshot.key;
  const result = snapshot.result;
  return (
    key != null &&
    isString(key.executionScope) &&
    typeof key.agentId === 'string' &&
    isString(key.toolUseId) &&
    result != null &&
    result.decision === 'ask' &&
    Array.isArray(result.additionalContexts) &&
    Array.isArray(result.injectedMessages) &&
    Array.isArray(result.errors)
  );
}

function isCheckpointReference(
  value: unknown
): value is SubagentCheckpointReference {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const checkpoint = value as Partial<SubagentCheckpointReference>;
  return (
    isString(checkpoint.threadId) &&
    isString(checkpoint.checkpointId) &&
    typeof checkpoint.checkpointNs === 'string'
  );
}

function isToolCallStepReference(
  value: unknown
): value is SubagentToolCallStepReference {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const reference = value as Partial<SubagentToolCallStepReference>;
  return isString(reference.toolCallId) && isString(reference.stepId);
}

function isToolSessionReference(
  value: unknown
): value is SubagentToolSessionReference {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const reference = value as Partial<SubagentToolSessionReference>;
  const context = reference.context;
  return (
    isString(reference.toolName) &&
    context != null &&
    typeof context === 'object' &&
    isString(context.session_id) &&
    Number.isFinite(context.lastUpdated) &&
    (context.files == null || Array.isArray(context.files))
  );
}

function isToolUsageCountReference(
  value: unknown
): value is SubagentToolNodeResumeState['toolUsageCounts'][number] {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const entry = value as Partial<
    SubagentToolNodeResumeState['toolUsageCounts'][number]
  >;
  return isString(entry.toolName) && isNonnegativeInteger(entry.count);
}

function isDirectPathTurnReference(
  value: unknown
): value is SubagentToolNodeResumeState['directPathTurns'][number] {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const entry = value as Partial<
    SubagentToolNodeResumeState['directPathTurns'][number]
  >;
  return isString(entry.toolCallId) && isNonnegativeInteger(entry.turn);
}

export function isToolNodeResumeState(
  value: unknown
): value is SubagentToolNodeResumeState {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const state = value as Partial<SubagentToolNodeResumeState>;
  if (
    !isString(state.stateKey) ||
    !Array.isArray(state.toolUsageCounts) ||
    !Array.isArray(state.directPathTurns)
  ) {
    return false;
  }
  const validUsageCounts = state.toolUsageCounts.every(
    isToolUsageCountReference
  );
  const validDirectTurns = state.directPathTurns.every(
    isDirectPathTurnReference
  );
  if (!validUsageCounts || !validDirectTurns) {
    return false;
  }
  const usageToolNames = new Set(
    state.toolUsageCounts.map((entry) => entry.toolName)
  );
  const directToolCallIds = new Set(
    state.directPathTurns.map((entry) => entry.toolCallId)
  );
  return (
    usageToolNames.size === state.toolUsageCounts.length &&
    directToolCallIds.size === state.directPathTurns.length
  );
}

function isEagerToolUsageState(
  value: unknown
): value is SubagentEagerToolUsageState {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const state = value as Partial<SubagentEagerToolUsageState>;
  if (
    typeof state.agentId !== 'string' ||
    !Array.isArray(state.toolUsageCounts) ||
    !state.toolUsageCounts.every(isToolUsageCountReference)
  ) {
    return false;
  }
  const toolNames = new Set(
    state.toolUsageCounts.map((entry) => entry.toolName)
  );
  return toolNames.size === state.toolUsageCounts.length;
}

function isToolOutputReferenceEntry(
  value: unknown
): value is ToolOutputReferenceState['entries'][number] {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const entry = value as Partial<ToolOutputReferenceState['entries'][number]>;
  return isString(entry.key) && typeof entry.value === 'string';
}

export function isToolOutputReferenceState(
  value: unknown
): value is ToolOutputReferenceState {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const state = value as Partial<ToolOutputReferenceState>;
  if (
    !Array.isArray(state.entries) ||
    !state.entries.every(isToolOutputReferenceEntry) ||
    !isNonnegativeInteger(state.turnCounter) ||
    !Array.isArray(state.warnedNonStringTools) ||
    !state.warnedNonStringTools.every(isString)
  ) {
    return false;
  }
  const entryKeys = new Set(state.entries.map((entry) => entry.key));
  const warnedTools = new Set(state.warnedNonStringTools);
  return (
    entryKeys.size === state.entries.length &&
    warnedTools.size === state.warnedNonStringTools.length
  );
}

/** A current tier, or a legacy one the run then drops so it re-derives. */
function isStoredFadingTier(value: unknown): boolean {
  return isFadingTier(value) || isLegacyFadingTier(value);
}

function isGraphResumeState(value: unknown): value is SubagentGraphResumeState {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const state = value as Partial<SubagentGraphResumeState>;
  if (
    !Array.isArray(state.toolCallSteps) ||
    !state.toolCallSteps.every(isToolCallStepReference) ||
    !Array.isArray(state.toolSessions) ||
    !state.toolSessions.every(isToolSessionReference) ||
    !Array.isArray(state.toolNodes) ||
    !state.toolNodes.every(isToolNodeResumeState) ||
    !Array.isArray(state.eagerToolUsage) ||
    !state.eagerToolUsage.every(isEagerToolUsageState) ||
    !Array.isArray(state.eagerToolSuppressions) ||
    !state.eagerToolSuppressions.every(isString) ||
    (state.runStepState != null && !isRunStepResumeState(state.runStepState)) ||
    (state.toolOutputReferences != null &&
      !isToolOutputReferenceState(state.toolOutputReferences)) ||
    (state.fadingTier != null && !isStoredFadingTier(state.fadingTier)) ||
    (state.fadingTiers != null &&
      (typeof state.fadingTiers !== 'object' ||
        Array.isArray(state.fadingTiers) ||
        !Object.values(state.fadingTiers).every(isStoredFadingTier)))
  ) {
    return false;
  }
  const toolCallIds = new Set(
    state.toolCallSteps.map((reference) => reference.toolCallId)
  );
  const toolNames = new Set(
    state.toolSessions.map((reference) => reference.toolName)
  );
  const toolNodeKeys = new Set(state.toolNodes.map((node) => node.stateKey));
  const eagerAgentIds = new Set(
    state.eagerToolUsage.map((usage) => usage.agentId)
  );
  const eagerSuppressions = new Set(state.eagerToolSuppressions);
  return (
    toolCallIds.size === state.toolCallSteps.length &&
    toolNames.size === state.toolSessions.length &&
    toolNodeKeys.size === state.toolNodes.length &&
    eagerAgentIds.size === state.eagerToolUsage.length &&
    eagerSuppressions.size === state.eagerToolSuppressions.length
  );
}

function isSubagentResumeExecution(
  value: unknown,
  seen: Set<object>,
  depth: number
): value is SubagentResumeExecution {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const execution = value as Partial<SubagentResumeExecution>;
  const childRunId = execution.childRunId;
  const subagentType = execution.subagentType;
  const configId = execution.configId;
  const approvalExecutionScope = execution.approvalExecutionScope;
  const checkpoints = execution.checkpoints;
  const graphState = execution.graphState;
  const approvalReplays = execution.approvalReplays;
  if (
    !isString(execution.parentToolCallId) ||
    !isString(childRunId) ||
    (subagentType != null && !isString(subagentType)) ||
    (configId != null && !isString(configId)) ||
    !isString(approvalExecutionScope) ||
    !Array.isArray(checkpoints) ||
    checkpoints.length === 0 ||
    !checkpoints.every(isCheckpointReference) ||
    !isGraphResumeState(graphState) ||
    !Array.isArray(approvalReplays)
  ) {
    return false;
  }
  const checkpointThreadId = checkpoints[0].threadId;
  const checkpointNamespaces = new Set(
    checkpoints.map((checkpoint) => checkpoint.checkpointNs)
  );
  const validApprovalReplays = approvalReplays.every(
    (snapshot) =>
      isApprovalReplaySnapshot(snapshot) &&
      snapshot.key.executionScope === approvalExecutionScope
  );
  if (!validApprovalReplays) {
    return false;
  }
  const approvalKeys = new Set(
    approvalReplays.map((snapshot) =>
      JSON.stringify([
        snapshot.key.executionScope,
        snapshot.key.agentId,
        snapshot.key.toolUseId,
      ])
    )
  );
  return (
    checkpoints.every(
      (checkpoint) => checkpoint.threadId === checkpointThreadId
    ) &&
    checkpointNamespaces.size === checkpoints.length &&
    approvalKeys.size === approvalReplays.length &&
    (execution.descendant == null ||
      isSubagentResumeManifest(execution.descendant, seen, depth + 1))
  );
}

function isSubagentResumeManifest(
  value: unknown,
  seen: Set<object> = new Set(),
  depth = 0
): value is SubagentResumeManifest {
  if (
    value == null ||
    typeof value !== 'object' ||
    depth > MAX_RESUME_MANIFEST_DEPTH ||
    seen.has(value)
  ) {
    return false;
  }
  seen.add(value);
  const manifest = value as Partial<SubagentResumeManifest>;
  if (
    manifest.version !== 1 ||
    !Array.isArray(manifest.executions) ||
    manifest.executions.length === 0 ||
    !manifest.executions.every((execution) =>
      isSubagentResumeExecution(execution, seen, depth)
    )
  ) {
    return false;
  }
  const parentToolCallIds = new Set(
    manifest.executions.map((execution) => execution.parentToolCallId)
  );
  return parentToolCallIds.size === manifest.executions.length;
}

export function getSubagentResumeManifest(
  payload: unknown
): SubagentResumeManifest | undefined {
  payload = stripRunStepResumeState(payload);
  if (payload == null || typeof payload !== 'object') {
    return undefined;
  }
  if (
    !Object.prototype.hasOwnProperty.call(
      payload,
      SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY
    )
  ) {
    return undefined;
  }
  const manifest = (payload as PayloadWithSubagentResumeManifest)[
    SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY
  ];
  return isSubagentResumeManifest(manifest) ? manifest : undefined;
}

export function requireValidSubagentResumeManifest(
  payload: unknown
): SubagentResumeManifest | undefined {
  if (
    payload == null ||
    typeof payload !== 'object' ||
    !Object.prototype.hasOwnProperty.call(
      payload,
      SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY
    )
  ) {
    return undefined;
  }
  const manifest = getSubagentResumeManifest(payload);
  if (manifest == null) {
    throw new Error('Invalid subagent resume manifest.');
  }
  return manifest;
}

function canAttachSubagentResumeManifestInline(
  payload: unknown
): payload is Record<string, unknown> {
  if (
    payload == null ||
    typeof payload !== 'object' ||
    Array.isArray(payload) ||
    Object.getPrototypeOf(payload) !== Object.prototype
  ) {
    return false;
  }
  return SUBAGENT_RESUME_PRIVATE_PAYLOAD_KEYS.every(
    (key) => !Object.prototype.hasOwnProperty.call(payload, key)
  );
}

function isInlineSubagentResumePayload(
  payload: unknown
): payload is PayloadWithSubagentResumeManifest & Record<string, unknown> {
  return (
    payload != null &&
    typeof payload === 'object' &&
    !Array.isArray(payload) &&
    Object.getPrototypeOf(payload) === Object.prototype &&
    !Object.prototype.hasOwnProperty.call(
      payload,
      SUBAGENT_RESUME_WRAPPER_VERSION_KEY
    ) &&
    !Object.prototype.hasOwnProperty.call(
      payload,
      SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY
    ) &&
    getSubagentResumeManifest(payload) != null
  );
}

export function attachSubagentResumeManifest(
  payload: unknown,
  manifest: SubagentResumeManifest
): object {
  const runStepState = getRunStepResumeState(payload);
  if (runStepState != null) {
    return attachRunStepResumeState(
      attachSubagentResumeManifest(stripRunStepResumeState(payload), manifest),
      runStepState
    );
  }
  if (
    isWrappedSubagentResumePayload(payload) ||
    isInlineSubagentResumePayload(payload)
  ) {
    return {
      ...payload,
      [SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY]: manifest,
    };
  }
  if (!canAttachSubagentResumeManifestInline(payload)) {
    return {
      [SUBAGENT_RESUME_WRAPPER_VERSION_KEY]: 1,
      [SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY]: payload,
      [SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY]: manifest,
    };
  }
  return {
    ...payload,
    [SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY]: manifest,
  };
}

export function stripSubagentResumeManifest(payload: unknown): unknown {
  payload = stripRunStepResumeState(payload);
  if (isWrappedSubagentResumePayload(payload)) {
    return payload[SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY];
  }
  if (
    payload == null ||
    typeof payload !== 'object' ||
    getSubagentResumeManifest(payload) == null
  ) {
    return payload;
  }
  return Object.fromEntries(
    Object.entries(payload).filter(
      ([key]) => key !== SUBAGENT_RESUME_MANIFEST_PAYLOAD_KEY
    )
  );
}

function isWrappedSubagentResumePayload(
  payload: unknown
): payload is WrappedSubagentResumePayload {
  if (
    payload == null ||
    typeof payload !== 'object' ||
    Array.isArray(payload) ||
    Object.getPrototypeOf(payload) !== Object.prototype ||
    Object.keys(payload).length !== SUBAGENT_RESUME_PRIVATE_PAYLOAD_KEYS.length
  ) {
    return false;
  }
  const wrapper = payload as Partial<WrappedSubagentResumePayload>;
  return (
    wrapper[SUBAGENT_RESUME_WRAPPER_VERSION_KEY] === 1 &&
    Object.prototype.hasOwnProperty.call(
      wrapper,
      SUBAGENT_RESUME_WRAPPED_PAYLOAD_KEY
    ) &&
    getSubagentResumeManifest(wrapper) != null
  );
}

export const SUBAGENT_REPLAY_CONTROLLER = Symbol.for(
  '@librechat/agents/subagent-replay-controller'
);

export type SettledSubagentToolOutput = {
  output: ToolMessage;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  referenceContent?: string;
};

export interface SubagentReplayController {
  getResumeManifest?(
    parentToolCallIds?: ReadonlySet<string>,
    config?: RunnableConfig
  ): Promise<SubagentResumeManifest | undefined>;
  getSettledOutput(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<SettledSubagentToolOutput | undefined>;
  persistSettledOutput(
    call: ToolCall,
    config: RunnableConfig,
    settled: SettledSubagentToolOutput
  ): Promise<void>;
}

export type ReplayableSubagentTool = {
  [SUBAGENT_REPLAY_CONTROLLER]?: SubagentReplayController;
};

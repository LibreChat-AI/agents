import { Command, MemorySaver, isCommand } from '@langchain/langgraph';
import { isBaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { Interrupt } from '@langchain/langgraph';
import type { SubagentToolNodeResumeState } from '@/tools/subagent/SubagentReplay';
import type { ToolApprovalReviewEvidence } from '@/hitl/approvalReview';
import type { ToolOutputReferenceState } from '@/tools/toolOutputReferences';
import {
  isToolNodeResumeState,
  isToolOutputReferenceState,
  stripSubagentResumeManifest,
  requireValidSubagentResumeManifest,
} from '@/tools/subagent/SubagentReplay';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY } from '@/hooks/types';
import {
  TOOL_APPROVAL_REVIEW_CONFIG_KEY,
  getToolApprovalReviewEvidence,
  createToolApprovalReviewEvidence,
} from '@/hitl/approvalReview';
import {
  attachRunStepResumeState,
  getRunStepResumeState,
  stripRunStepResumeState,
} from '@/tools/runStepResume';

export const TOOL_BATCH_REPLAY_KEY = '__librechat_tool_batch_replay';
export const TOOL_REPLAY_CONFIGS_KEY = '__librechat_tool_replay_configs';
const TOOL_BATCH_PAYLOAD_KEY = '__librechat_tool_batch_payload';
const TOOL_BATCH_WRAPPER_KEY = '__librechat_tool_batch_wrapper';

export interface SettledToolBatchResult {
  proposal: { name: string; args: Record<string, unknown> };
  output: BaseMessage | Command;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  /** Code-session identities injected into this completed direct call. */
  codeSessionBaseline?: Array<[string, string]>;
  completionHandled?: boolean;
  referenceContent?: string;
  turn?: number;
}

export interface ToolBatchReplayRecord {
  owner: string;
  batch: string;
  encoding: string;
  data: string;
  turnState?: SubagentToolNodeResumeState;
  referenceState?: ToolOutputReferenceState;
}

export type ToolReplayResumeStatus = 'active' | 'stale' | 'unverifiable';

interface ToolBatchReplayState {
  version: 1;
  approvalOwner?: string;
  /** Populated from the pending interrupt by Run, including question pauses. */
  interruptId?: string;
  records: ToolBatchReplayRecord[];
  wrappedPayload?: boolean;
}

type ToolReplayConfig = {
  [TOOL_BATCH_REPLAY_KEY]?: ToolBatchReplayState;
  [TOOL_APPROVAL_REVIEW_CONFIG_KEY]?: ToolApprovalReviewEvidence;
};

/** Use the same message/Command codec as LangGraph checkpoints. */
const serializer = new MemorySaver().serde;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function bindLegacyReplayOwner(
  owner: string,
  configurable: Record<string, unknown>
): string {
  let identity: unknown;
  try {
    identity = JSON.parse(owner);
  } catch {
    return owner;
  }
  if (!Array.isArray(identity) || identity.length !== 3) {
    return owner;
  }
  return JSON.stringify([
    ...identity,
    typeof configurable.user_id === 'string' ? configurable.user_id : '',
    typeof configurable.thread_id === 'string' ? configurable.thread_id : '',
  ]);
}

function bindLegacyReplayState(
  state: ToolBatchReplayState | undefined,
  configurable: Record<string, unknown>
): ToolBatchReplayState | undefined {
  if (state == null) return undefined;
  return {
    ...state,
    approvalOwner:
      state.approvalOwner == null
        ? undefined
        : bindLegacyReplayOwner(state.approvalOwner, configurable),
    records: state.records.map((record) => ({
      ...record,
      owner: bindLegacyReplayOwner(record.owner, configurable),
    })),
  };
}

/** The only host-to-runtime entry point for checkpoint-owned tool replay state. */
export function restoreToolReplayConfig(
  configurable: Record<string, unknown>,
  interruptId: string | undefined,
  payload: unknown
): void {
  delete configurable[TOOL_BATCH_REPLAY_KEY];
  delete configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
  const state = bindLegacyReplayState(
    getToolBatchReplayState(payload),
    configurable
  );
  const publicPayload = getPublicToolInterruptPayload(payload);
  const review = createToolApprovalReviewEvidence(
    interruptId,
    publicPayload,
    state?.approvalOwner
  );
  if (isToolApprovalInterrupt(publicPayload) &&
    (review == null || (state != null && state.approvalOwner == null))) {
    throw new Error('Invalid tool approval checkpoint');
  }
  if (state != null) {
    configurable[TOOL_BATCH_REPLAY_KEY] = { ...state, interruptId };
  }
  if (review != null) {
    configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY] = review;
  }
}

/** Restore each checkpointed interrupt without sharing member authorization. */
export function restoreToolReplayConfigs(
  configurable: NonNullable<RunnableConfig['configurable']>,
  interrupts: readonly Interrupt[],
  sourceScope: string,
  destinationScope: string,
  destinationThreadId: string,
  resumedInterruptIds: ReadonlySet<string>
): void {
  delete configurable[TOOL_REPLAY_CONFIGS_KEY];
  const configs = new Map<string, ToolReplayConfig>();
  let first: ToolReplayConfig | undefined;
  for (const pending of interrupts) {
    const restored: ToolReplayConfig &
      NonNullable<RunnableConfig['configurable']> = {
        user_id: configurable.user_id,
        thread_id: configurable.thread_id,
      };
    restoreToolReplayConfig(restored, pending.id, pending.value);
    rebindToolBatchReplayScope(
      restored,
      sourceScope,
      destinationScope,
      destinationThreadId
    );
    const replay = restored[TOOL_BATCH_REPLAY_KEY];
    const review = restored[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
    const resumed = pending.id != null && resumedInterruptIds.has(pending.id);
    const evidence: ToolReplayConfig = {
      [TOOL_BATCH_REPLAY_KEY]: resumed || replay == null
        ? replay
        : { ...replay, approvalOwner: undefined },
      [TOOL_APPROVAL_REVIEW_CONFIG_KEY]: resumed ? review : undefined,
    };
    if (resumed) first ??= evidence;
    const owner = review?.owner ?? replay?.approvalOwner;
    const owners = owner == null
      ? new Set(replay?.records.map(record => record.owner))
      : new Set([owner]);
    for (const owner of owners) {
      if (configs.has(owner)) {
        throw new Error('Ambiguous tool replay checkpoint owner');
      }
      configs.set(owner, evidence);
    }
  }
  delete configurable[TOOL_BATCH_REPLAY_KEY];
  delete configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
  Object.assign(configurable, first);
  if (configs.size > 1) configurable[TOOL_REPLAY_CONFIGS_KEY] = configs;
}

/** Select a member's evidence before the unchanged owner/proposal safeguards. */
export function getToolReplayConfig(
  config: RunnableConfig,
  owner: string
): RunnableConfig {
  const candidate = config.configurable?.[TOOL_REPLAY_CONFIGS_KEY];
  if (!(candidate instanceof Map)) return config;
  const evidence = (candidate as ReadonlyMap<string, ToolReplayConfig>).get(owner);
  if (evidence == null) return config;
  const configurable: NonNullable<RunnableConfig['configurable']> = {
    ...config.configurable,
    ...evidence,
  };
  delete configurable[TOOL_REPLAY_CONFIGS_KEY];
  return { ...config, configurable };
}

/**
 * Resume maps identify the checkpointed interrupt, while the task scratchpad
 * identifies the node that can consume it. Config survives into later graph
 * steps, so the presence of replay evidence alone does not imply a resume.
 * A parent replaying a checkpointed child has no local resume value.
 */
export function getToolReplayResumeStatus(
  config: RunnableConfig,
  interruptId: string | undefined,
  isChildReplay: boolean
): ToolReplayResumeStatus {
  const resumeMap: unknown = config.configurable?.__pregel_resume_map;
  const scratchpad: unknown = config.configurable?.__pregel_scratchpad;
  if (!isRecord(scratchpad) || !Array.isArray(scratchpad.resume)) {
    return 'unverifiable';
  }
  /**
   * LangGraph supports both an interrupt-id resume map and a task-local
   * `nullResume` value. The latter is already scoped to the checkpoint task,
   * so it remains a supported, verifiable resume without a map entry.
   */
  if (scratchpad.nullResume !== undefined) {
    return 'active';
  }
  const hasResume = scratchpad.resume.length > 0;
  if (!isRecord(resumeMap)) {
    return hasResume ? 'unverifiable' : 'stale';
  }
  if (interruptId == null) {
    /** Legacy direct graph callers do not restore the ID into replay state. */
    if (!hasResume) return 'stale';
    return Object.keys(resumeMap).length === 1 ? 'active' : 'unverifiable';
  }
  if (!Object.prototype.hasOwnProperty.call(resumeMap, interruptId)) {
    return hasResume ? 'unverifiable' : 'stale';
  }
  if (isChildReplay) {
    return 'active';
  }
  return hasResume ? 'active' : 'stale';
}

export function getToolBatchReplayOwner(
  config: RunnableConfig,
  agentId: string
): string {
  return JSON.stringify([
    getToolBatchReplayScope(config),
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY] == null
      ? (config.configurable?.checkpoint_ns ?? '')
      : '',
    agentId,
    typeof config.configurable?.user_id === 'string'
      ? config.configurable.user_id
      : '',
    typeof config.configurable?.thread_id === 'string'
      ? config.configurable.thread_id
      : '',
  ]);
}

/** Remove stale authorization while retaining only settled results for this batch. */
export function clearStaleToolApprovalConfig(
  configurable: Record<string, unknown>,
  owner: string,
  batch: string | undefined
): void {
  delete configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
  const state = getToolBatchReplayState(configurable);
  const records = batch == null
    ? []
    : (state?.records.filter(
      (record) => record.owner === owner && record.batch === batch
    ) ?? []);
  if (state == null || records.length === 0) {
    delete configurable[TOOL_BATCH_REPLAY_KEY];
    return;
  }
  configurable[TOOL_BATCH_REPLAY_KEY] = {
    version: 1,
    records,
    wrappedPayload: state.wrappedPayload,
  } satisfies ToolBatchReplayState;
}

export function getToolBatchReplayScope(
  config: RunnableConfig
): string | undefined {
  const scope =
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY] ??
    config.configurable?.thread_id;
  return typeof scope === 'string' && scope.length > 0 ? scope : undefined;
}

/** Foreign replay evidence may only transit a checkpoint-proven child path. */
export function isChildToolReplayOwner(
  config: RunnableConfig,
  owner: string,
  trustedParentCallIds: ReadonlySet<string>
): boolean {
  const manifest = requireValidSubagentResumeManifest(config.configurable);
  if (manifest == null || trustedParentCallIds.size === 0) return false;
  const identity: unknown = JSON.parse(owner);
  if (!Array.isArray(identity) ||
    (identity.length !== 3 && identity.length !== 5) ||
    identity[1] !== '') return false;
  const pending = manifest.executions.filter((execution) => trustedParentCallIds.has(execution.parentToolCallId));
  while (pending.length > 0) {
    const execution = pending.pop()!;
    if (execution.approvalExecutionScope === identity[0]) return true;
    pending.push(...(execution.descendant?.executions ?? []));
  }
  return false;
}

/** Rebind only the checkpoint-proven source execution when a child fork is created. */
export function rebindToolBatchReplayScope(
  configurable: Record<string, unknown>,
  sourceScope: string,
  destinationScope: string,
  destinationThreadId?: string
): void {
  const rebind = (key: string): string => {
    const parts: unknown = JSON.parse(key);
    if (!Array.isArray(parts) || parts[0] !== sourceScope) {
      return key;
    }
    const rebound = [destinationScope, ...parts.slice(1)];
    if (rebound.length === 5 && destinationThreadId != null) {
      rebound[4] = destinationThreadId;
    }
    return JSON.stringify(rebound);
  };
  const config = { configurable };
  const review = getToolApprovalReviewEvidence(config);
  if (review?.owner != null) {
    configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY] = {
      ...review,
      owner: rebind(review.owner),
    };
  }
  const state = getToolBatchReplayState(configurable);
  if (state != null) {
    configurable[TOOL_BATCH_REPLAY_KEY] = {
      ...state,
      approvalOwner:
        state.approvalOwner == null ? undefined : rebind(state.approvalOwner),
      records: state.records.map((record) => ({
        ...record,
        owner: rebind(record.owner),
        batch: rebind(record.batch),
      })),
    };
  }
}

/** A fork may be paused again before it consumes its pending interrupt. */
export function rebindToolBatchReplayPayload(
  payload: unknown,
  sourceScope: string,
  destinationScope: string,
  destinationThreadId?: string
): unknown {
  const state = getToolBatchReplayState(payload);
  const value = stripRunStepResumeState(payload);
  if (state == null || !isRecord(value)) {
    return payload;
  }
  const configurable = { [TOOL_BATCH_REPLAY_KEY]: state };
  rebindToolBatchReplayScope(
    configurable,
    sourceScope,
    destinationScope,
    destinationThreadId
  );
  const rebound = {
    ...value,
    [TOOL_BATCH_REPLAY_KEY]: configurable[TOOL_BATCH_REPLAY_KEY],
  };
  const runStepState = getRunStepResumeState(payload);
  return runStepState == null
    ? rebound
    : attachRunStepResumeState(rebound, runStepState);
}

export function getToolBatchReplayState(
  payload: unknown
): ToolBatchReplayState | undefined {
  const value = stripRunStepResumeState(payload);
  if (
    value == null ||
    typeof value !== 'object' ||
    !(TOOL_BATCH_REPLAY_KEY in value)
  ) {
    return undefined;
  }
  const candidate: unknown = value[TOOL_BATCH_REPLAY_KEY];
  if (candidate === undefined) {
    return undefined;
  }
  const state = candidate as Partial<ToolBatchReplayState> | null;
  if (
    state?.version !== 1 ||
    (state.wrappedPayload != null &&
      typeof state.wrappedPayload !== 'boolean') ||
    (state.approvalOwner != null && typeof state.approvalOwner !== 'string') ||
    (state.interruptId != null && typeof state.interruptId !== 'string') ||
    !Array.isArray(state.records) ||
    state.records.some((value: unknown) => {
      if (value == null || typeof value !== 'object') {
        return true;
      }
      const record = value as Partial<ToolBatchReplayRecord>;
      return (
        typeof record.owner !== 'string' ||
        typeof record.batch !== 'string' ||
        typeof record.encoding !== 'string' ||
        typeof record.data !== 'string' ||
        (record.turnState != null &&
          !isToolNodeResumeState(record.turnState)) ||
        (record.referenceState != null &&
          !isToolOutputReferenceState(record.referenceState))
      );
    })
  ) {
    throw new Error('Invalid tool batch replay checkpoint');
  }
  return structuredClone(state as ToolBatchReplayState);
}

export function stripToolBatchReplayState(payload: unknown): unknown {
  if (
    payload == null ||
    typeof payload !== 'object' ||
    !(TOOL_BATCH_REPLAY_KEY in payload)
  ) {
    return payload;
  }
  const state = getToolBatchReplayState(payload);
  const { [TOOL_BATCH_REPLAY_KEY]: _state, ...publicPayload } = payload;
  if (
    state?.wrappedPayload === true &&
    TOOL_BATCH_PAYLOAD_KEY in publicPayload
  ) {
    return publicPayload[TOOL_BATCH_PAYLOAD_KEY];
  }
  return publicPayload;
}

/** Decode outer execution metadata before shape-sensitive child wrappers. */
export function getPublicToolInterruptPayload(payload: unknown): unknown {
  return stripSubagentResumeManifest(
    stripToolBatchReplayState(stripRunStepResumeState(payload))
  );
}

export async function attachToolBatchReplayState(
  payload: unknown,
  owner: string,
  batches: ReadonlyMap<string, ReadonlyMap<string, object>>,
  turnState?: SubagentToolNodeResumeState,
  referenceState?: ToolOutputReferenceState
): Promise<unknown> {
  const publicPayload = stripRunStepResumeState(payload);
  const previous = getToolBatchReplayState(publicPayload);
  const records =
    previous?.records.filter((record) => record.owner !== owner) ?? [];
  for (const [batch, results] of batches) {
    const [encoding, bytes] = await serializer.dumpsTyped([...results]);
    records.push({
      owner,
      batch,
      encoding,
      data: Buffer.from(bytes).toString('base64'),
      ...(turnState == null ? {} : { turnState }),
      ...(referenceState == null ? {} : { referenceState }),
    });
  }
  const approvalOwner =
    previous?.approvalOwner ??
    (isToolApprovalInterrupt(publicPayload) ? owner : undefined);
  if (records.length === 0 && approvalOwner == null) {
    return payload;
  }
  const result = {
    ...(isRecord(publicPayload)
      ? publicPayload
      : {
        [TOOL_BATCH_WRAPPER_KEY]: 1,
        [TOOL_BATCH_PAYLOAD_KEY]: publicPayload,
      }),
    [TOOL_BATCH_REPLAY_KEY]: {
      version: 1,
      approvalOwner,
      records,
      wrappedPayload: previous?.wrappedPayload ?? !isRecord(publicPayload),
    } satisfies ToolBatchReplayState,
  };
  const runStepState = getRunStepResumeState(payload);
  return runStepState == null
    ? result
    : attachRunStepResumeState(result, runStepState);
}

export async function restoreToolBatchReplayState(
  config: RunnableConfig,
  owner: string
): Promise<
  Array<{
    batch: string;
    results: Array<[string, SettledToolBatchResult]>;
    turnState?: SubagentToolNodeResumeState;
    referenceState?: ToolOutputReferenceState;
  }>
> {
  const state = getToolBatchReplayState(config.configurable);
  if (state == null) {
    return [];
  }
  return Promise.all(
    state.records
      .filter((record) => record.owner === owner)
      .map(async (record) => {
        const values: unknown = await serializer.loadsTyped(
          record.encoding,
          Buffer.from(record.data, 'base64')
        );
        if (!Array.isArray(values)) {
          throw new Error('Invalid settled tool batch results');
        }
        const seen = new Set<string>();
        const results = values.map(
          (entry: unknown): [string, SettledToolBatchResult] => {
            if (
              !Array.isArray(entry) ||
              entry.length !== 2 ||
              typeof entry[0] !== 'string' ||
              seen.has(entry[0])
            ) {
              throw new Error('Invalid settled tool batch results');
            }
            const result = entry[1] as Partial<SettledToolBatchResult> | null;
            const codeSessionBaseline: unknown = (
              result as { codeSessionBaseline?: unknown } | null
            )?.codeSessionBaseline;
            const baselineNames = Array.isArray(codeSessionBaseline)
              ? codeSessionBaseline.map((pair) =>
                Array.isArray(pair) ? pair[0] : undefined
              )
              : [];
            if (
              result == null ||
              (!isBaseMessage(result.output) && !isCommand(result.output)) ||
              !Array.isArray(result.additionalContexts) ||
              result.additionalContexts.some(
                (context) => typeof context !== 'string'
              ) ||
              (result.completionHandled != null &&
                typeof result.completionHandled !== 'boolean') ||
              (result.referenceContent != null &&
                typeof result.referenceContent !== 'string') ||
              (result.turn != null &&
                (!Number.isSafeInteger(result.turn) || result.turn < 0)) ||
              !isRecord(result.proposal) ||
              typeof result.proposal.name !== 'string' ||
              result.proposal.name.length === 0 ||
              !isRecord(result.proposal.args) ||
              (result.resolvedArgs != null && !isRecord(result.resolvedArgs)) ||
              (codeSessionBaseline != null &&
                (!Array.isArray(codeSessionBaseline) ||
                  codeSessionBaseline.some(
                    (pair) =>
                      !Array.isArray(pair) ||
                      pair.length !== 2 ||
                      typeof pair[0] !== 'string' ||
                      pair[0].length === 0 ||
                      typeof pair[1] !== 'string' ||
                      pair[1].length === 0
                  ) ||
                  new Set(baselineNames).size !== baselineNames.length))
            ) {
              throw new Error('Invalid settled tool batch results');
            }
            seen.add(entry[0]);
            const output = isCommand(result.output)
              ? new Command({
                update: result.output.update,
                goto: result.output.goto,
                graph: result.output.graph,
                resume: result.output.resume,
              })
              : result.output;
            return [
              entry[0],
              {
                ...result,
                proposal: result.proposal,
                output,
                additionalContexts: result.additionalContexts,
              },
            ];
          }
        );
        return {
          batch: record.batch,
          results,
          turnState: record.turnState,
          referenceState: record.referenceState,
        };
      })
  );
}

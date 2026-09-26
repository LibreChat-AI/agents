import { nanoid } from 'nanoid';
import { createHash } from 'crypto';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import {
  Command,
  END,
  GraphInterrupt,
  INTERRUPT,
  MessagesAnnotation,
  START,
  StateGraph,
  copyCheckpoint,
  isGraphInterrupt,
  isInterrupted,
} from '@langchain/langgraph';
import type {
  Interrupt,
  StateSnapshot,
  CheckpointTuple,
  BaseCheckpointSaver,
} from '@langchain/langgraph';
import type { ChatGeneration, LLMResult } from '@langchain/core/outputs';
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  AgentInputs,
  BaseGraphState,
  CompiledStateWorkflow,
  EventHandler,
  MultiAgentGraphState,
  HumanInTheLoopConfig,
  InjectedMessage,
  MessageDeltaEvent,
  MessageCreationDetails,
  ProcessedToolCall,
  ReasoningDeltaEvent,
  RunStep,
  RunStepDeltaEvent,
  RunStepClosedEvent,
  RunStepStatus,
  StandardGraphInput,
  ExecutableSubagentConfigEntry,
  ResolvedSubagentConfig,
  ResolvedSubagentConfigEntry,
  SubagentExecutionContext,
  SubagentContextAdapter,
  SubagentContextInput,
  SubagentContextResult,
  PreparedSubagentContext,
  SubagentTaskConfig,
  SubagentTaskRuntime,
  SubagentResolveConfigurable,
  SubagentResolveRequestContext,
  SubagentResolveUserContext,
  StepCompleted,
  SubagentUpdateEvent,
  SubagentUpdatePhase,
  SubagentUsageSink,
  ToolApprovalInterruptPayload,
  ToolExecuteBatchRequest,
  ToolCallDelta,
  TokenCounter,
  ToolApprovalDecision,
  ToolApprovalDecisionMap,
} from '@/types';
import type {
  PreparedSubagentExecutionIdentity,
  SubagentDefinitionBinding,
  SubagentExecutionIdentity,
  SubagentExecutionPreparationLease,
  SubagentExecutionRecord,
} from './SubagentExecutionRegistry';
import type {
  SubagentResumeExecution,
  SubagentResumeManifest,
  SubagentCheckpointReference,
  SettledSubagentToolOutput,
} from './SubagentReplay';
import type {
  AggregatedHookResult,
  PostToolBatchHookOutput,
  PreemptBoundaryHookOutput,
  ToolApprovalReplaySnapshot,
} from '@/hooks';
import type { GraphFactory } from '@/graphs/graphFactory';
import type { StandardGraph } from '@/graphs/Graph';
import {
  getSubagentApprovalExecutionScope,
  SubagentDefinitionBindingError,
  SubagentExecutionRegistry,
  SubagentInvocationBindingError,
  SubagentSettlementBindingError,
} from './SubagentExecutionRegistry';
import {
  getSubagentResumeManifest,
  attachSubagentResumeManifest,
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY,
  SUBAGENT_RESUME_MANIFEST_CONFIG_KEY,
} from './SubagentReplay';
import {
  rebindToolBatchReplayScope,
  rebindToolBatchReplayPayload,
  restoreToolReplayConfig,
} from '@/tools/toolBatchReplay';
import {
  executeHooks,
  HookRegistry,
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
} from '@/hooks';
import {
  StreamLimitExceededError,
  RUN_BREAKER_SCOPE_CONFIG_KEY,
} from '@/llm/streamLimits';
import {
  DEFAULT_SUBAGENT_MAX_TURNS,
  SUBAGENT_RECURSION_MULTIPLIER,
} from './runtimeLimits';
import {
  ContentTypes,
  Constants,
  GraphEvents,
  Callback,
  StepTypes,
} from '@/common';
import {
  createChildGraphPlan,
  isGraphSubagentConfig,
} from './childGraphConfig';
import { stripRunStepResumeState } from '@/tools/runStepResume';
import { seedAgentInitialSessions } from '@/utils/toolSessions';
import { stableStringify } from '@/tools/eagerEventExecution';
import { convertInjectedMessages } from '@/messages/injected';
import { resolveClientOptionsModel } from '@/llm/request';
import { composeAbortSignals } from '@/utils/misc';
import { HandlerRegistry } from '@/events';
import { sleep } from '@/utils/run';

export {
  buildChildInputs,
  isGraphSubagentConfig,
  normalizeSubagentConfigs,
  normalizeSubagentConfigEntries,
  resolveSubagentConfigs,
  resolveSubagentConfigEntries,
} from './childGraphConfig';

const ERROR_MESSAGE_MAX_CHARS = 200;
const MAX_QUEUED_SUBAGENT_UPDATES = 64;
const MAX_BACKGROUND_SUBAGENT_SEALS = 32;
const SUBAGENT_UPDATE_HANDLER_TIMEOUT_MS = 5_000;
const TEXT_DELTA_CONTENT_TYPE = `${ContentTypes.TEXT}_delta`;
const SUBAGENT_RESOLUTION_ERROR_MESSAGE =
  'Subagent error: Unable to initialize the selected subagent.';
const SUBAGENT_CONFIG_CHANGED_MESSAGE =
  'Subagent error: Subagent configuration changed since this execution was paused.';
const SUBAGENT_INVOCATION_CHANGED_MESSAGE =
  'Subagent error: Subagent invocation changed for this execution.';

function seedChildGraphSessions(
  childGraph: StandardGraph,
  agents: AgentInputs[]
): void {
  seedAgentInitialSessions(childGraph.sessions, agents);
}

async function awaitWithAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal
): Promise<T> {
  signal.throwIfAborted();
  let onAbort: (() => void) | undefined;
  const aborted = new Promise<never>((_resolve, reject) => {
    onAbort = (): void => {
      reject(
        signal.reason instanceof Error
          ? signal.reason
          : new Error('Detached subagent task cancelled.')
      );
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
  try {
    return await Promise.race([promise, aborted]);
  } finally {
    if (onAbort != null) {
      signal.removeEventListener('abort', onAbort);
    }
  }
}

async function dispatchObservationalSubagentUpdate(
  handler: EventHandler,
  event: SubagentUpdateEvent
): Promise<boolean> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve()
        .then(() => handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event))
        .then(
          () => true,
          () => true
        ),
      new Promise<boolean>((resolve) => {
        timeout = setTimeout(
          () => resolve(false),
          SUBAGENT_UPDATE_HANDLER_TIMEOUT_MS
        );
      }),
    ]);
  } finally {
    if (timeout != null) {
      clearTimeout(timeout);
    }
  }
}

const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
  additionalContexts: [] as string[],
  injectedMessages: [] as InjectedMessage[],
  errors: [] as string[],
});

type SanitizedSubagentToolCall = {
  id: string;
  name: string;
  args?: ToolExecuteBatchRequest['toolCalls'][number]['args'];
};

type SanitizedSubagentToolExecuteData = {
  toolCalls: SanitizedSubagentToolCall[];
  agentId?: string;
};

type SanitizedRunStep = Partial<
  Pick<
    RunStep,
    | 'agentId'
    | 'created_at'
    | 'groupId'
    | 'id'
    | 'index'
    | 'runId'
    | 'status'
    | 'stepIndex'
    | 'summary'
    | 'type'
    | 'usage'
  >
> & {
  stepDetails?: SanitizedStepDetails;
};

type SanitizedRunStepClosed = Partial<RunStepClosedEvent>;

type SanitizedStepDetails =
  | {
      type: StepTypes.MESSAGE_CREATION;
      message_creation?: Partial<MessageCreationDetails['message_creation']>;
    }
  | {
      type: StepTypes.TOOL_CALLS;
      tool_calls?: SanitizedAgentToolCall[];
    };

type SanitizedAgentToolCall = {
  id?: string;
  name?: string;
  args?: string | object;
  type?: string;
  function?: {
    name?: string;
    arguments?: string | object;
  };
};

type SanitizedRunStepDelta = Partial<Pick<RunStepDeltaEvent, 'id'>> & {
  delta?: SanitizedToolCallDelta;
};

type SanitizedToolCallDelta = Partial<
  Pick<ToolCallDelta, 'auth' | 'expires_at' | 'summary' | 'type'>
> & {
  tool_calls?: SanitizedAgentToolCall[];
};

type SanitizedStepCompleted =
  | {
      id?: string;
      index?: number;
      type: 'tool_call';
      tool_call?: SanitizedProcessedToolCall;
      completed_at?: number;
    }
  | {
      type: 'summary';
      summary?: Extract<StepCompleted, { type: 'summary' }>['summary'];
    };

type SanitizedProcessedToolCall = Partial<
  Pick<
    ProcessedToolCall,
    'args' | 'id' | 'name' | 'output' | 'progress' | 'outcome'
  >
>;

type SanitizedRunStepCompleted = {
  result?: SanitizedStepCompleted;
};

type SanitizedMessageDelta = Partial<Pick<MessageDeltaEvent, 'id'>> & {
  delta?: {
    content?: MessageDeltaEvent['delta']['content'];
    tool_call_ids?: MessageDeltaEvent['delta']['tool_call_ids'];
  };
};

type SanitizedReasoningDelta = Partial<Pick<ReasoningDeltaEvent, 'id'>> & {
  delta?: {
    content?: ReasoningDeltaEvent['delta']['content'];
  };
};

type QueuedSubagentUpdate = {
  eventName: string;
  phase: SubagentUpdatePhase;
  data: unknown;
  memberAgentId?: string;
};

type ForwarderCallback = {
  handler: BaseCallbackHandler;
  drain: () => Promise<void>;
};

type StatefulCompiledWorkflow = Omit<CompiledStateWorkflow, 'invoke'> & {
  invoke(
    input: BaseGraphState | Command | null,
    config?: RunnableConfig
  ): Promise<MultiAgentGraphState>;
  getState(config: RunnableConfig): Promise<StateSnapshot>;
  updateState?(
    config: RunnableConfig,
    values: Record<string, unknown>,
    asNode?: string
  ): Promise<RunnableConfig>;
};

type ReplayCheckpointWorkflow = {
  updateState(
    config: RunnableConfig,
    values: { messages: BaseMessage[] },
    asNode: string
  ): Promise<RunnableConfig>;
};

type ActiveChildRun = {
  graph: StandardGraph;
  workflow: StatefulCompiledWorkflow;
  pendingInterrupts: Interrupt[];
  invokeConfig?: RunnableConfig;
  childAgentId: string;
  checkpointNodeId: string;
  childRunId: string;
};

type PersistedToolOutput = {
  content: ToolMessage['content'];
  toolCallId: string;
  id?: string;
  name?: string;
  status?: 'success' | 'error';
  additionalKwargs: ToolMessage['additional_kwargs'];
  responseMetadata: ToolMessage['response_metadata'];
  metadata?: Record<string, unknown>;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  referenceContent?: string;
};

type SubagentCheckpointMarker = {
  version: 1;
  parentToolCallId: string;
  lifecycleComplete: true;
  hookSessionId?: string;
  childRunId?: string;
  settledOutput?: PersistedToolOutput;
};

function isResumeExecutionCompatible(
  resumeExecution: SubagentResumeExecution | undefined,
  subagentType: string | undefined,
  executableConfig: ExecutableSubagentConfigEntry | undefined,
  hasPersistedTypeEvidence = false
): boolean {
  if (resumeExecution == null) {
    return true;
  }
  if (
    subagentType == null ||
    executableConfig == null ||
    resumeExecution.configId !== executableConfig.configId
  ) {
    return false;
  }
  if (resumeExecution.subagentType != null) {
    return resumeExecution.subagentType === subagentType;
  }
  if (hasPersistedTypeEvidence) {
    return true;
  }
  return (
    isGraphSubagentConfig(executableConfig) ||
    executableConfig.agentInputs != null
  );
}

function getSubagentTypeFromArgs(args: unknown): string | undefined {
  if (!isObjectLike(args)) {
    return undefined;
  }
  return asNonEmptyString((args as { subagent_type?: unknown }).subagent_type);
}

function getSubagentDescriptionFromArgs(args: unknown): string | undefined {
  if (!isObjectLike(args)) {
    return undefined;
  }
  const description = (args as { description?: unknown }).description;
  return typeof description === 'string' && description.trim().length > 0
    ? description
    : undefined;
}

function getSubagentType(call: ToolCall): string | undefined {
  return getSubagentTypeFromArgs(call.args);
}

const LANGGRAPH_RUNTIME_CONFIG_PREFIX = '__pregel_';
const LANGGRAPH_RESUME_MAP_CONFIG_KEY = '__pregel_resume_map';
const LANGGRAPH_CHECKPOINT_CONFIG_KEYS = new Set([
  'checkpoint_id',
  'checkpoint_map',
  'checkpoint_ns',
]);
const SUBAGENT_CHECKPOINT_MARKER_KEY = '__librechat_subagent_checkpoint';
const SUBAGENT_HOOK_SESSION_KEY = '__librechat_subagent_hook_session';
const SUBAGENT_RUN_ID_KEY = '__librechat_subagent_run_id';
const SUBAGENT_REPLAY_NODE = 'subagent-replay';
export const DEFAULT_SUBAGENT_DESCRIPTION = 'No task description provided';

function isCheckpointSaver(value: unknown): value is BaseCheckpointSaver {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<BaseCheckpointSaver>;
  return (
    typeof candidate.getTuple === 'function' &&
    typeof candidate.list === 'function' &&
    typeof candidate.put === 'function' &&
    typeof candidate.putWrites === 'function' &&
    typeof candidate.deleteThread === 'function'
  );
}

function isSubagentCheckpointMarker(
  value: unknown
): value is SubagentCheckpointMarker {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const marker = value as Partial<SubagentCheckpointMarker>;
  return (
    marker.version === 1 &&
    marker.lifecycleComplete === true &&
    typeof marker.parentToolCallId === 'string' &&
    (marker.hookSessionId == null ||
      typeof marker.hookSessionId === 'string') &&
    (marker.childRunId == null || typeof marker.childRunId === 'string') &&
    (marker.settledOutput == null ||
      isPersistedToolOutput(marker.settledOutput))
  );
}

function isPersistedToolOutput(value: unknown): value is PersistedToolOutput {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const output = value as {
    content?: unknown;
    toolCallId?: unknown;
    status?: unknown;
    additionalKwargs?: unknown;
    responseMetadata?: unknown;
    additionalContexts?: unknown;
    resolvedArgs?: unknown;
    referenceContent?: unknown;
  };
  const contentIsValid =
    typeof output.content === 'string' || Array.isArray(output.content);
  const statusIsValid =
    output.status == null ||
    output.status === 'success' ||
    output.status === 'error';
  const contextsAreValid =
    Array.isArray(output.additionalContexts) &&
    output.additionalContexts.every((context) => typeof context === 'string');
  const resolvedArgsAreValid =
    output.resolvedArgs == null ||
    (typeof output.resolvedArgs === 'object' &&
      !Array.isArray(output.resolvedArgs));
  return (
    contentIsValid &&
    typeof output.toolCallId === 'string' &&
    output.additionalKwargs != null &&
    typeof output.additionalKwargs === 'object' &&
    output.responseMetadata != null &&
    typeof output.responseMetadata === 'object' &&
    statusIsValid &&
    contextsAreValid &&
    resolvedArgsAreValid &&
    (output.referenceContent == null ||
      typeof output.referenceContent === 'string')
  );
}

function getSubagentCheckpointMarker(
  messages: BaseMessage[],
  parentToolCallId: string
): SubagentCheckpointMarker | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const marker =
      messages[i].additional_kwargs[SUBAGENT_CHECKPOINT_MARKER_KEY];
    if (
      isSubagentCheckpointMarker(marker) &&
      marker.parentToolCallId === parentToolCallId
    ) {
      return marker;
    }
  }
  return undefined;
}

function getSubagentHookSessionId(messages: BaseMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const sessionId = messages[i].additional_kwargs[SUBAGENT_HOOK_SESSION_KEY];
    if (typeof sessionId === 'string' && sessionId.length > 0) {
      return sessionId;
    }
  }
  return undefined;
}

function getSubagentRunId(messages: BaseMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const runId = messages[i].additional_kwargs[SUBAGENT_RUN_ID_KEY];
    if (typeof runId === 'string' && runId.length > 0) {
      return runId;
    }
    const marker =
      messages[i].additional_kwargs[SUBAGENT_CHECKPOINT_MARKER_KEY];
    if (
      isSubagentCheckpointMarker(marker) &&
      marker.childRunId != null &&
      marker.childRunId.length > 0
    ) {
      return marker.childRunId;
    }
  }
  return undefined;
}

function isSubagentCheckpointMarkerMessage(message: BaseMessage): boolean {
  return isSubagentCheckpointMarker(
    message.additional_kwargs[SUBAGENT_CHECKPOINT_MARKER_KEY]
  );
}

function createSubagentCheckpointMarkerMessage(
  marker: SubagentCheckpointMarker
): AIMessage {
  return new AIMessage({
    content: '',
    additional_kwargs: { [SUBAGENT_CHECKPOINT_MARKER_KEY]: marker },
  });
}

function getCheckpointMessages(value: unknown): BaseMessage[] {
  return Array.isArray(value) ? value.filter(BaseMessage.isInstance) : [];
}

function getConfigurableString(
  config: RunnableConfig | undefined,
  key: string
): string | undefined {
  const value = config?.configurable?.[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function getTupleMessages(tuple: CheckpointTuple | undefined): BaseMessage[] {
  return getCheckpointMessages(tuple?.checkpoint.channel_values.messages);
}

function getCheckpointReference(
  tuple: CheckpointTuple | undefined
): SubagentCheckpointReference | undefined {
  const configurable = tuple?.config.configurable;
  const threadId = configurable?.thread_id;
  const checkpointId = configurable?.checkpoint_id;
  const checkpointNs = configurable?.checkpoint_ns ?? '';
  if (
    typeof threadId !== 'string' ||
    threadId.length === 0 ||
    typeof checkpointId !== 'string' ||
    checkpointId.length === 0 ||
    typeof checkpointNs !== 'string'
  ) {
    return undefined;
  }
  return { threadId, checkpointId, checkpointNs };
}

function serializeToolOutput(
  settled: SettledSubagentToolOutput
): PersistedToolOutput {
  const { output } = settled;
  return {
    content: output.content,
    toolCallId: output.tool_call_id,
    ...(output.id == null ? {} : { id: output.id }),
    ...(output.name == null ? {} : { name: output.name }),
    ...(output.status == null ? {} : { status: output.status }),
    additionalKwargs: output.additional_kwargs,
    responseMetadata: output.response_metadata,
    ...(output.metadata == null ? {} : { metadata: output.metadata }),
    additionalContexts: settled.additionalContexts,
    ...(settled.resolvedArgs == null
      ? {}
      : { resolvedArgs: settled.resolvedArgs }),
    ...(settled.referenceContent == null
      ? {}
      : { referenceContent: settled.referenceContent }),
  };
}

function getSettlementFingerprint(output: PersistedToolOutput): string {
  return createHash('sha256').update(stableStringify(output)).digest('hex');
}

function getBackgroundTaskFingerprint(
  description: string,
  subagentType: string,
  threadId?: string
): string {
  return createHash('sha256')
    .update(
      stableStringify({
        description,
        subagentType,
        ...(threadId == null || threadId === '' ? {} : { threadId }),
      })
    )
    .digest('hex');
}

function getBackgroundTaskHookSessionId(
  sourceHookSessionId: string,
  taskId: string
): string {
  return `${sourceHookSessionId}:subagent-task:${taskId}`;
}

function deserializeToolOutput(
  output: PersistedToolOutput
): SettledSubagentToolOutput {
  return {
    output: new ToolMessage({
      content: output.content,
      tool_call_id: output.toolCallId,
      ...(output.id == null ? {} : { id: output.id }),
      ...(output.name == null ? {} : { name: output.name }),
      ...(output.status == null ? {} : { status: output.status }),
      additional_kwargs: output.additionalKwargs,
      response_metadata: output.responseMetadata,
      ...(output.metadata == null ? {} : { metadata: output.metadata }),
    }),
    additionalContexts: output.additionalContexts,
    ...(output.resolvedArgs == null
      ? {}
      : { resolvedArgs: output.resolvedArgs }),
    ...(output.referenceContent == null
      ? {}
      : { referenceContent: output.referenceContent }),
  };
}

function isToolApprovalPayload(
  value: unknown
): value is ToolApprovalInterruptPayload {
  return (
    value != null &&
    typeof value === 'object' &&
    (value as { type?: unknown }).type === 'tool_approval'
  );
}

function addSubagentScope(
  interrupts: Interrupt[],
  scope: NonNullable<ToolApprovalInterruptPayload['subagent']>,
  resumeManifest?: SubagentResumeManifest
): Interrupt[] {
  return interrupts.map((childInterrupt) => {
    let payload = stripRunStepResumeState(childInterrupt.value);
    if (isToolApprovalPayload(payload)) {
      payload = {
        ...payload,
        subagent: payload.subagent ?? scope,
      };
    }
    return {
      ...childInterrupt,
      value:
        resumeManifest == null
          ? payload
          : attachSubagentResumeManifest(payload, resumeManifest),
    };
  });
}

type ToolApprovalResumeValue = ToolApprovalDecision[] | ToolApprovalDecisionMap;

function getChildResumeMap(
  pendingInterrupts: Interrupt[],
  parentConfigurable: Record<string, unknown> | undefined
): Record<string, ToolApprovalResumeValue> | undefined {
  const resumeMap = parentConfigurable?.[LANGGRAPH_RESUME_MAP_CONFIG_KEY];
  if (resumeMap == null || typeof resumeMap !== 'object') {
    return undefined;
  }

  const parentResumeMap = resumeMap as Record<string, ToolApprovalResumeValue>;
  const childResumeMap: Record<string, ToolApprovalResumeValue> = {};
  for (const childInterrupt of pendingInterrupts) {
    const interruptId = childInterrupt.id;
    if (
      typeof interruptId === 'string' &&
      Object.prototype.hasOwnProperty.call(parentResumeMap, interruptId)
    ) {
      childResumeMap[interruptId] = parentResumeMap[interruptId];
    }
  }
  return Object.keys(childResumeMap).length > 0 ? childResumeMap : undefined;
}

function getPersistedInterrupts(snapshot: StateSnapshot): Interrupt[] {
  const interrupts: Interrupt[] = [];
  for (const task of snapshot.tasks) {
    for (const pendingInterrupt of task.interrupts) {
      interrupts.push(pendingInterrupt);
    }
  }
  return interrupts;
}

function getPersistedMessages(
  snapshot: StateSnapshot
): BaseMessage[] | undefined {
  if (snapshot.values == null || typeof snapshot.values !== 'object') {
    return undefined;
  }
  const values = snapshot.values as { messages?: BaseMessage[] };
  if (!Array.isArray(values.messages) || values.messages.length === 0) {
    return undefined;
  }
  const messages = values.messages.filter(
    (message) => !isSubagentCheckpointMarkerMessage(message)
  );
  return messages.length > 0 ? messages : undefined;
}

function createReplayCheckpointWorkflow(
  checkpointer: BaseCheckpointSaver
): ReplayCheckpointWorkflow {
  return new StateGraph(MessagesAnnotation)
    .addNode(SUBAGENT_REPLAY_NODE, (state) => state)
    .addEdge(START, SUBAGENT_REPLAY_NODE)
    .addEdge(SUBAGENT_REPLAY_NODE, END)
    .compile({ checkpointer }) as ReplayCheckpointWorkflow;
}

export type SubagentExecuteParams = {
  description: string;
  subagentType: string;
  /** Saved logical child thread selected by the parent model. */
  subagentThreadId?: string;
  threadId?: string;
  /** Signal attached to this specific parent tool invocation. */
  signal?: AbortSignal;
  /**
   * Breaker controller captured at the parent TOOL BATCH's entry, before
   * PreToolUse hooks. Preferred over the live scope accessor: a graph reset
   * during a hook would otherwise bind this child to the NEW run's
   * controller — reviving it on a fresh signal and letting its trips cancel
   * unrelated work.
   */
  breaker?: AbortController;
  /**
   * Parent-side `tool_call_id` of the `subagent` tool invocation that
   * triggered this execution. Surfaced on {@link SubagentUpdateEvent} so
   * hosts can correlate child updates back to the originating tool call
   * without relying on event ordering heuristics.
   */
  parentToolCallId?: string;
  /**
   * Snapshot of the parent invocation's host `config.configurable` at
   * the spawn-tool call site. Host-set fields (`requestBody`, `user`,
   * `userMCPAuthMap`, etc.) propagate into the child workflow's
   * `configurable` — fixing MCP body-placeholder substitution and
   * per-user lookups for subagent tool calls. LangGraph runtime keys
   * (`__pregel_*`, checkpoint bookkeeping) are intentionally not
   * inherited; the child graph recreates its own runtime config.
   *
   * Inheritance details (verified empirically against LangGraph):
   *   - host-set keys propagate as-is into the child's tool dispatches;
   *   - with nested HITL enabled, `thread_id` is replaced with a stable
   *     child checkpoint id derived from the parent's durable thread id,
   *     checkpoint fork, parent agent id, and spawning tool call id so parent
   *     and child checkpoints cannot collide, sibling parent forks stay
   *     isolated, and reconstruction returns to the same child checkpoint;
   *     parent-scoped hook lookup remains keyed by the inherited `run_id`;
   *   - `parent_run_id` propagates when the host put it on parent's
   *     configurable;
   *   - `run_id` is *overwritten by the LangGraph runtime* at child
   *     invoke time regardless of what we forward — child's tool
   *     dispatches see the child graph's runtime runId in
   *     `configurable.run_id`, not the parent's. Hosts that need
   *     parent-scoped run identity for downstream consumers should
   *     plumb it via a host-defined key (e.g. `requestBody.messageId`),
   *     not `run_id`.
   *
   * A future revision will likely make this inheritance configurable
   * per spawn type — background / async subagents may want isolation
   * rather than sharing parent's host context.
   */
  parentConfigurable?: Record<string, unknown>;
  /** Dedicated hook session used by a detached task. @internal */
  hookSessionId?: string;
  /** Process-local task controls consumed by the child graph. @internal */
  taskRuntime?: SubagentTaskRuntime;
  /** Host-restored child transcript for a fresh continuation run. @internal */
  initialMessages?: BaseMessage[];
};

export type SubagentExecuteResult = SubagentContextResult & {
  /** Tagged internal failure; foreground callers retain the legacy content. */
  error?: string;
  /** Completed child work whose host projection must be retried without re-execution. */
  retryableDelivery?: true;
};

function createSubagentFailure(
  content: string,
  error = content
): SubagentExecuteResult {
  return { content, messages: [], error };
}

/**
 * Factory that constructs a child graph for subagent execution. Injected
 * rather than imported so that `SubagentExecutor` does not have a runtime
 * dependency on `StandardGraph` — this avoids a circular dependency between
 * `src/graphs/Graph.ts` and `src/tools/subagent/` that would otherwise break
 * Rollup's chunking under `preserveModules`.
 */
export type ChildGraphFactory = (input: StandardGraphInput) => StandardGraph;

export type SubagentExecutorOptions = {
  configs: ReadonlyMap<string, ExecutableSubagentConfigEntry>;
  parentSignal?: AbortSignal;
  /** Run-scoped breaker abort shared by every executor of one graph, so a
   * child tripping a stream limit stops subagents running under OTHER
   * parallel agent nodes too. An accessor rather than a captured controller:
   * the graph recreates its controller per run, and each execution must
   * bind to the controller current when it STARTS — signal and trip target
   * together, so a straggler from a failed run can neither revive on nor
   * circuit-break a later run's controller. Absent (tests, minimal hosts),
   * the executor falls back to its own private controller. */
  breakerScope?: {
    controller: () => AbortController;
  };
  hookRegistry?: HookRegistry;
  parentRunId: string;
  parentAgentId?: string;
  /** Root identity, policy session, and lineage inherited by a child graph. */
  executionContext?: SubagentExecutionContext;
  langfuse?: StandardGraphInput['langfuse'];
  tokenCounter?: TokenCounter;
  /**
   * Run-level stream circuit breakers, forwarded into every child graph so
   * a host raising, lowering, or disabling the limits governs subagents too.
   * Child model calls run through `attemptInvoke`'s local stream handler
   * (children have no registered dispatcher), which enforces the child
   * graph's own resolved limits; without this the child would silently
   * revert to the defaults.
   */
  streamLimits?: StandardGraphInput['streamLimits'];
  humanInTheLoop?: HumanInTheLoopConfig;
  /** Shared durable saver used to recover outer tool lifecycle results before
   * parent hooks re-enter after a process rebuild. Narrowed structurally at
   * construction because graph compile options also permit framework flags. */
  checkpointer?: unknown;
  /** Remaining nesting budget. 0 or negative blocks execution. */
  maxDepth?: number;
  /**
   * Factory for constructing the isolated child graph. Callers pass
   * `(input) => new StandardGraph(input)` — injected to break a circular
   * module dependency.
   */
  createChildGraph: ChildGraphFactory;
  /** Preferred polymorphic child constructor. The legacy standard-only
   * factory remains required for source compatibility. */
  createChildGraphByKind?: GraphFactory;
  /**
   * Captures a child-graph factory and its run-scoped host dependencies
   * synchronously, before a detached task can outlive parent cleanup.
   */
  createDetachedChildGraphFactory?: (
    parentHandlerRegistry: HandlerRegistry
  ) => GraphFactory;
  /**
   * Parent's event handler registry. When provided, child-graph events are
   * forwarded through this registry so hosts can:
   *   (a) execute event-driven tools (`ON_TOOL_EXECUTE` routed to parent's handler),
   *   (b) surface child activity to a UI via wrapped {@link GraphEvents.ON_SUBAGENT_UPDATE}.
   * When omitted, the child runs fully isolated (legacy behavior).
   *
   * Can be a direct `HandlerRegistry` or a zero-arg getter — use the getter
   * form when the registry is assigned to the graph AFTER the executor is
   * constructed (the current `Run.create` flow sets `handlerRegistry`
   * post-`createWorkflow`, so `createAgentNode` must capture lazily).
   */
  parentHandlerRegistry?: HandlerRegistry | (() => HandlerRegistry | undefined);
  /**
   * Receives a usage event for every model call the child run makes. The
   * child workflow executes via `invoke()` with a detached callbacks array,
   * so its `on_chat_model_end` events never reach the parent's handler
   * registry — without this sink, child token usage is invisible to the
   * host (unbilled model calls). Forwarded into the child graph's input so
   * nested subagents report through the same sink.
   */
  usageSink?: SubagentUsageSink;
  /** Host-owned process-local task namespace for detached execution. */
  taskConfig?: SubagentTaskConfig;
  subagentContext?: SubagentContextAdapter;
};

type DurableExecutionRecord = SubagentExecutionRecord<
  SubagentExecuteResult,
  ResolvedSubagentConfig,
  ActiveChildRun,
  PersistedToolOutput
>;

export class SubagentExecutor {
  private readonly configs: ReadonlyMap<string, ExecutableSubagentConfigEntry>;
  private readonly parentSignal?: AbortSignal;
  /** Aborted when a child trips a stream circuit breaker: parallel sibling
   * subagents run concurrently on the parent's signal, and rejecting the
   * batch alone would leave their provider requests streaming after the
   * safety abort. One-way by design — a tripped breaker ends the run, so no
   * later child of this executor should start either. Fallback for hosts
   * that do not supply the graph's shared `breakerScope`. */
  private readonly childRunAbort = new AbortController();
  private readonly breakerScope?: SubagentExecutorOptions['breakerScope'];
  private readonly hookRegistry?: HookRegistry;
  private readonly parentRunId: string;
  private readonly parentAgentId?: string;
  private readonly executionContext: SubagentExecutionContext;
  private readonly langfuse?: StandardGraphInput['langfuse'];
  private readonly tokenCounter?: TokenCounter;
  private readonly streamLimits?: StandardGraphInput['streamLimits'];
  private readonly humanInTheLoop?: HumanInTheLoopConfig;
  private readonly checkpointer?: BaseCheckpointSaver;
  private readonly maxDepth: number;
  private readonly createChildGraph: ChildGraphFactory;
  private readonly createChildGraphByKind?: GraphFactory;
  private readonly createDetachedChildGraphFactory?: (
    parentHandlerRegistry: HandlerRegistry
  ) => GraphFactory;
  private readonly usageSink?: SubagentUsageSink;
  private readonly taskConfig?: SubagentTaskConfig;
  private readonly subagentContext?: SubagentContextAdapter;
  private readonly executions: SubagentExecutionRegistry<
    SubagentExecuteResult,
    ResolvedSubagentConfig,
    ActiveChildRun,
    PersistedToolOutput
  >;
  private replayCheckpointWorkflow?: ReplayCheckpointWorkflow;
  private readonly resolveParentHandlerRegistry?: () =>
    | HandlerRegistry
    | undefined;

  constructor(options: SubagentExecutorOptions) {
    this.configs = options.configs;
    this.parentSignal = options.parentSignal;
    this.breakerScope = options.breakerScope;
    this.hookRegistry = options.hookRegistry;
    this.parentRunId = options.parentRunId;
    this.parentAgentId = options.parentAgentId;
    this.executionContext = options.executionContext ?? {
      rootRunId: options.parentRunId,
      hookSessionId: options.parentRunId,
      depth: 0,
      ancestry: [],
    };
    this.langfuse = options.langfuse;
    this.tokenCounter = options.tokenCounter;
    this.streamLimits = options.streamLimits;
    this.humanInTheLoop = options.humanInTheLoop;
    this.checkpointer = isCheckpointSaver(options.checkpointer)
      ? options.checkpointer
      : undefined;
    this.executions = new SubagentExecutionRegistry({
      parentRunId: options.parentRunId,
      parentAgentId: options.parentAgentId,
      durable:
        options.humanInTheLoop?.enabled === true && this.checkpointer != null,
    });
    this.maxDepth = options.maxDepth ?? 1;
    this.createChildGraph = options.createChildGraph;
    this.createChildGraphByKind = options.createChildGraphByKind;
    this.createDetachedChildGraphFactory =
      options.createDetachedChildGraphFactory;
    this.usageSink = options.usageSink;
    this.taskConfig = options.taskConfig;
    this.subagentContext = options.subagentContext;
    const rawRegistry = options.parentHandlerRegistry;
    if (typeof rawRegistry === 'function') {
      this.resolveParentHandlerRegistry = rawRegistry;
    } else if (rawRegistry != null) {
      this.resolveParentHandlerRegistry = (): HandlerRegistry => rawRegistry;
    }
  }

  /** The breaker controller current for this execution — read per spawn
   * because the graph recreates its controller each run. */
  private resolveBreakerController(): AbortController {
    return this.breakerScope?.controller() ?? this.childRunAbort;
  }

  /** One signal that fires on the parent's abort or the breaker abort,
   * collapsed to a single signal when possible (mirrors
   * `composeAbortSignals` in Graph.ts). */
  private composeChildSignal(
    breaker: AbortController,
    executionSignal?: AbortSignal
  ): AbortSignal {
    const parentAndBreaker = composeAbortSignals(
      this.parentSignal,
      breaker.signal
    );
    return (
      composeAbortSignals(parentAndBreaker, executionSignal) ?? breaker.signal
    );
  }

  /** Snapshot of the parent's registry at the moment a subagent is dispatched. */
  private getParentHandlerRegistry(): HandlerRegistry | undefined {
    return this.resolveParentHandlerRegistry?.();
  }

  /**
   * Starts one independently-owned executor behind the configured task store.
   * The parent ToolNode receives the handle synchronously; the detached clone
   * is not registered on the parent graph, so end-of-turn cleanup cannot
   * invalidate or clear a child that intentionally outlives that turn.
   */
  executeInBackground(params: SubagentExecuteParams): string {
    if (this.taskConfig == null) {
      return JSON.stringify({
        status: 'rejected',
        message: 'Background subagent execution is not enabled for this run.',
      });
    }
    const executableConfig = this.configs.get(params.subagentType);
    if (executableConfig == null) {
      return JSON.stringify({
        status: 'rejected',
        message: `Unknown subagent type "${params.subagentType}".`,
      });
    }
    if (this.maxDepth <= 0) {
      return JSON.stringify({
        status: 'rejected',
        message: 'Maximum subagent nesting depth exceeded.',
      });
    }
    if (this.humanInTheLoop?.enabled === true) {
      return JSON.stringify({
        status: 'rejected',
        message:
          'Background subagent execution does not support human-in-the-loop pauses.',
      });
    }
    const parentToolCallId = params.parentToolCallId?.trim();
    if (parentToolCallId == null || parentToolCallId === '') {
      return JSON.stringify({
        status: 'rejected',
        message:
          'Background subagent execution requires a parent tool call ID.',
      });
    }
    const subagentThreadId = params.subagentThreadId?.trim();
    if (
      subagentThreadId != null &&
      subagentThreadId !== '' &&
      this.taskConfig.store.supportsThreadContinuation !== true
    ) {
      return JSON.stringify({
        status: 'rejected',
        tool: Constants.SUBAGENT,
        message: 'Child-thread continuation is not enabled by this host.',
      });
    }
    const detachedHandlers = new HandlerRegistry();
    const sourceHookSessionId =
      asNonEmptyString(params.parentConfigurable?.run_id) ??
      this.executionContext.hookSessionId;
    const taskHookRegistry =
      this.hookRegistry?.forkSession(sourceHookSessionId) ?? new HookRegistry();
    const toolHandler = this.getParentHandlerRegistry()?.getHandler(
      GraphEvents.ON_TOOL_EXECUTE
    );
    if (toolHandler != null) {
      detachedHandlers.register(GraphEvents.ON_TOOL_EXECUTE, toolHandler);
    }
    const detachedGraphFactory =
      this.createDetachedChildGraphFactory?.(detachedHandlers);
    const started = this.taskConfig.store.start({
      scopeId: this.taskConfig.scopeId,
      idempotencyKey: JSON.stringify([
        this.parentRunId,
        this.parentAgentId ?? '',
        parentToolCallId,
      ]),
      parentRunId: this.parentRunId,
      ...(this.parentAgentId == null
        ? {}
        : { parentAgentId: this.parentAgentId }),
      parentToolCallId,
      requestFingerprint: getBackgroundTaskFingerprint(
        params.description,
        params.subagentType,
        subagentThreadId
      ),
      ...(subagentThreadId == null || subagentThreadId === ''
        ? {}
        : { threadId: subagentThreadId }),
      input: params.description,
      subagentKind:
        'kind' in executableConfig && executableConfig.kind === 'graph'
          ? 'graph'
          : 'agent',
      subagentType: params.subagentType,
      run: (runtime, initialMessages) =>
        this.executeDetached(
          {
            ...params,
            ...(initialMessages == null ? {} : { initialMessages }),
          },
          runtime,
          detachedHandlers,
          taskHookRegistry,
          detachedGraphFactory
        ),
    });
    if (!started.accepted) {
      if (started.reason === 'thread_unavailable') {
        return JSON.stringify({
          status: 'rejected',
          tool: Constants.SUBAGENT,
          message:
            'The requested subagent thread is unavailable in this parent scope. Start a new subagent thread or choose one created by this parent for the same subagent type.',
        });
      }
      if (started.reason === 'conflict') {
        return JSON.stringify({
          status: 'rejected',
          tool: Constants.SUBAGENT,
          message:
            'The same parent tool call ID was already used with different background subagent arguments.',
        });
      }
      return JSON.stringify({
        status: 'rejected',
        tool: Constants.SUBAGENT,
        message:
          'Too many background subagent tasks are already running in this scope or process. Poll or cancel an existing task, or run this call in the foreground.',
      });
    }
    const startedThreadId = started.task.threadId?.trim();
    if (
      this.taskConfig.store.supportsThreadContinuation === true &&
      (startedThreadId == null || startedThreadId === '')
    ) {
      this.taskConfig.store.control(
        this.taskConfig.scopeId,
        started.task.taskId,
        { action: 'cancel' }
      );
      return JSON.stringify({
        status: 'rejected',
        tool: Constants.SUBAGENT,
        message:
          'The host accepted the subagent task without assigning its required thread ID.',
      });
    }
    return JSON.stringify({
      background_task_id: started.task.taskId,
      ...(this.taskConfig.store.supportsThreadContinuation === true
        ? { subagent_thread_id: startedThreadId }
        : {}),
      tool: Constants.SUBAGENT,
      subagent_type: params.subagentType,
      status: started.task.status,
      message: `${started.isNew ? 'Started' : 'Reused'} subagent "${params.subagentType}" background task. Poll the host background-task tool with background_task_id "${started.task.taskId}" to check progress or collect its result.`,
    });
  }

  private async executeDetached(
    params: SubagentExecuteParams,
    runtime: SubagentTaskRuntime,
    detachedHandlers: HandlerRegistry,
    taskHookRegistry: HookRegistry,
    detachedGraphFactory?: GraphFactory
  ): Promise<SubagentExecuteResult> {
    const sourceHookSessionId =
      asNonEmptyString(params.parentConfigurable?.run_id) ??
      this.executionContext.hookSessionId;
    const taskHookSessionId = getBackgroundTaskHookSessionId(
      sourceHookSessionId,
      runtime.taskId
    );
    const unregisterHooks: Array<() => void> = [];
    unregisterHooks.push(
      taskHookRegistry.registerSession(taskHookSessionId, 'PostToolBatch', {
        hooks: [
          (): PostToolBatchHookOutput => ({
            injectedMessages: runtime.drain('tool'),
          }),
        ],
      }),
      taskHookRegistry.registerSession(taskHookSessionId, 'PreemptBoundary', {
        hooks: [
          (): PreemptBoundaryHookOutput => ({
            injectedMessages: runtime.drain('preempt'),
          }),
        ],
      })
    );

    detachedHandlers.register(GraphEvents.ON_SUBAGENT_UPDATE, {
      handle: (_event, data): void => {
        runtime.reportProgress(data as SubagentUpdateEvent);
      },
    });

    const detached = new SubagentExecutor({
      configs: this.configs,
      parentSignal: runtime.signal,
      hookRegistry: taskHookRegistry,
      parentHandlerRegistry: detachedHandlers,
      parentRunId: this.parentRunId,
      parentAgentId: this.parentAgentId,
      executionContext: {
        ...this.executionContext,
        hookSessionId: taskHookSessionId,
      },
      langfuse: this.langfuse,
      tokenCounter: this.tokenCounter,
      usageSink: this.usageSink,
      subagentContext: this.subagentContext,
      streamLimits: this.streamLimits,
      maxDepth: this.maxDepth,
      createChildGraph:
        detachedGraphFactory == null
          ? this.createChildGraph
          : (input): StandardGraph =>
            detachedGraphFactory({ kind: 'standard', input }),
      createChildGraphByKind:
        detachedGraphFactory ?? this.createChildGraphByKind,
    });
    try {
      const executeAttempt = async (): Promise<SubagentExecuteResult> => {
        const result = await detached.execute({
          ...params,
          signal: undefined,
          breaker: undefined,
          hookSessionId: taskHookSessionId,
          taskRuntime: runtime,
          parentConfigurable: {
            ...params.parentConfigurable,
            run_id: taskHookSessionId,
          },
        });
        if (runtime.signal.aborted) {
          throw runtime.signal.reason instanceof Error
            ? runtime.signal.reason
            : new Error('Detached subagent task cancelled.');
        }
        return result;
      };
      let deliveryAttempts = 1;
      let result = await executeAttempt();
      while (result.retryableDelivery === true && deliveryAttempts < 3) {
        if (deliveryAttempts > 1) {
          await sleep(
            Math.min(100 * 2 ** Math.min(deliveryAttempts - 2, 6), 5_000)
          );
        }
        deliveryAttempts += 1;
        result = await executeAttempt();
      }
      if (result.error != null) {
        throw new Error(result.error);
      }
      return result;
    } finally {
      detached.clearHeavyState();
      for (const unregister of unregisterHooks) {
        unregister();
      }
      taskHookRegistry.clearSession(taskHookSessionId);
    }
  }

  private bindExecutionDefinition(
    execution: DurableExecutionRecord,
    binding: SubagentDefinitionBinding,
    authority: 'provisional' | 'effective'
  ): boolean {
    try {
      execution.bindDefinition(binding, authority);
      return true;
    } catch (error) {
      if (error instanceof SubagentDefinitionBindingError) {
        return false;
      }
      throw error;
    }
  }

  /** Resolve one lazy descriptor per stable child execution. Concurrent
   * duplicate dispatches share the same in-flight resolution; HITL re-entry
   * on this executor reuses the resolved config until the child settles. */
  private resolveExecutionConfig(
    config: ExecutableSubagentConfigEntry,
    execution: DurableExecutionRecord,
    context: {
      childRunId: string;
      childSignal: AbortSignal;
      threadId?: string;
      parentToolCallId?: string;
      parentConfigurable?: Record<string, unknown>;
    }
  ): Promise<ResolvedSubagentConfigEntry> {
    if (isGraphSubagentConfig(config) || config.agentInputs != null) {
      return Promise.resolve(config as ResolvedSubagentConfigEntry);
    }
    const resolver = config.resolveAgentInputs;
    const configId = config.configId;
    return execution.resolveConfig(
      (): Promise<ResolvedSubagentConfig> =>
        resolveAgentInputsWithSignal(
          (): Promise<AgentInputs> =>
            resolver({
              descriptor: {
                type: config.type,
                name: config.name,
                description: config.description,
                configId,
              },
              executionId: context.childRunId,
              parentRunId: this.parentRunId,
              parentAgentId: this.parentAgentId,
              parentToolCallId: context.parentToolCallId,
              threadId: context.threadId,
              signal: context.childSignal,
              configurable:
                context.parentConfigurable == null
                  ? undefined
                  : sanitizeResolverConfigurable(context.parentConfigurable),
            }),
          context.childSignal
        ).then(
          (agentInputs): ResolvedSubagentConfig => ({
            ...config,
            agentInputs,
          })
        ),
      context.childSignal
    );
  }

  /**
   * Keeps the original child thread as an immutable resume source once a
   * different parent Run reconstructs it. Each rebuilt parent gets a private
   * checkpoint fork, while the persisted child run ID remains stable for
   * activity and usage correlation across the interrupt boundary.
   */
  private async resolveChildExecutionIdentity(
    execution: DurableExecutionRecord
  ): Promise<SubagentExecutionIdentity> {
    return execution.resolveIdentity(async () => {
      const preparation = await this.createChildExecutionIdentity(execution);
      const { identity } = preparation;
      return {
        identity,
        lease: preparation.lease,
        commit: (lease): void => {
          const { resumeExecution } = execution;
          if (resumeExecution != null) {
            this.executions.retireResumeSources(
              execution,
              new Set(
                resumeExecution.checkpoints.map(
                  (checkpoint) => checkpoint.threadId
                )
              ),
              (source) => {
                if (source.activeRun != null) {
                  this.clearChildGraph(source.activeRun.graph);
                }
                if (source.identity != null) {
                  this.hookRegistry?.clearSession(
                    source.identity.approvalExecutionScope
                  );
                }
              }
            );
          }
          preparation.commit?.(lease);
          if (
            this.humanInTheLoop?.enabled === true &&
            this.checkpointer != null
          ) {
            this.executions.rememberCheckpointThread(identity.childThreadId);
          }
        },
        rollback: async (lease): Promise<void> => {
          await preparation.rollback?.(lease);
        },
      };
    });
  }

  private async createChildExecutionIdentity(
    execution: DurableExecutionRecord
  ): Promise<PreparedSubagentExecutionIdentity> {
    const {
      baseChildThreadId,
      branchChildThreadId,
      currentChildRunId,
      explicitResumeAttempt,
      resumeAttemptId,
    } = execution.address;
    if (this.humanInTheLoop?.enabled !== true || this.checkpointer == null) {
      return this.prepareIdentity(execution, {
        childRunId: currentChildRunId,
        childThreadId: baseChildThreadId,
        approvalExecutionScope: currentChildRunId,
      });
    }

    const { resumeExecution } = execution;
    if (resumeExecution != null) {
      const approvalExecutionScope = getSubagentApprovalExecutionScope(
        resumeExecution.childRunId,
        resumeAttemptId
      );
      const identity = {
        childRunId: resumeExecution.childRunId,
        childThreadId: branchChildThreadId,
        approvalExecutionScope,
      };
      const lease = this.executions.beginIdentityPreparation(
        execution,
        identity
      );
      await this.prepareCheckpointFork(
        resumeExecution.checkpoints,
        branchChildThreadId,
        lease,
        {
          source: resumeExecution.approvalExecutionScope,
          target: approvalExecutionScope,
        }
      );
      let previousApprovals: ToolApprovalReplaySnapshot[] = [];
      let approvalsMutated = false;
      return {
        identity,
        lease,
        commit: (currentLease): void => {
          const hookRegistry = this.hookRegistry;
          if (hookRegistry == null) {
            return;
          }
          previousApprovals = hookRegistry.snapshotPendingToolApprovals(
            approvalExecutionScope,
            approvalExecutionScope
          );
          const approvalReplayCreatesScope =
            previousApprovals.length === 0 &&
            resumeExecution.approvalReplays.length > 0;
          const approvalsChanged =
            previousApprovals.length > 0 ||
            resumeExecution.approvalReplays.length > 0;
          if (approvalReplayCreatesScope) {
            currentLease.markApprovalScopeCreated(async (): Promise<void> => {
              hookRegistry.restorePendingToolApprovals(
                approvalExecutionScope,
                approvalExecutionScope,
                previousApprovals
              );
            });
          }
          approvalsMutated = approvalsChanged && !approvalReplayCreatesScope;
          hookRegistry.restorePendingToolApprovals(
            approvalExecutionScope,
            approvalExecutionScope,
            resumeExecution.approvalReplays
          );
        },
        rollback: async (currentLease): Promise<void> => {
          if (
            approvalsMutated &&
            currentLease.ownsApprovalScope &&
            this.hookRegistry != null
          ) {
            this.hookRegistry.restorePendingToolApprovals(
              approvalExecutionScope,
              approvalExecutionScope,
              previousApprovals
            );
          }
        },
      };
    }

    const branchTuple = await this.checkpointer.getTuple({
      configurable: { thread_id: branchChildThreadId },
    });
    if (branchTuple != null) {
      const childRunId =
        getSubagentRunId(getTupleMessages(branchTuple)) ?? currentChildRunId;
      return this.prepareIdentity(execution, {
        childRunId,
        childThreadId: branchChildThreadId,
        approvalExecutionScope: getSubagentApprovalExecutionScope(
          childRunId,
          resumeAttemptId
        ),
      });
    }

    const baseTuple = await this.checkpointer.getTuple({
      configurable: { thread_id: baseChildThreadId },
    });
    if (baseTuple == null) {
      return this.prepareIdentity(execution, {
        childRunId: currentChildRunId,
        childThreadId: explicitResumeAttempt
          ? branchChildThreadId
          : baseChildThreadId,
        approvalExecutionScope: getSubagentApprovalExecutionScope(
          currentChildRunId,
          resumeAttemptId
        ),
      });
    }
    const persistedChildRunId = getSubagentRunId(getTupleMessages(baseTuple));
    if (
      !explicitResumeAttempt &&
      (persistedChildRunId == null || persistedChildRunId === currentChildRunId)
    ) {
      const childRunId = persistedChildRunId ?? currentChildRunId;
      return this.prepareIdentity(execution, {
        childRunId,
        childThreadId: baseChildThreadId,
        approvalExecutionScope: getSubagentApprovalExecutionScope(
          childRunId,
          resumeAttemptId
        ),
      });
    }
    const sourceCheckpoints =
      await this.getLatestCheckpointSnapshot(baseChildThreadId);
    if (sourceCheckpoints.length === 0) {
      throw new Error(
        `Cannot fork subagent checkpoint thread "${baseChildThreadId}" without a checkpoint ID.`
      );
    }
    const childRunId = persistedChildRunId ?? currentChildRunId;
    const identity = {
      childRunId,
      childThreadId: branchChildThreadId,
      approvalExecutionScope: getSubagentApprovalExecutionScope(
        childRunId,
        resumeAttemptId
      ),
    };
    const lease = this.executions.beginIdentityPreparation(execution, identity);
    await this.prepareCheckpointFork(
      sourceCheckpoints,
      branchChildThreadId,
      lease
    );
    return {
      identity,
      lease,
    };
  }

  private prepareIdentity(
    execution: DurableExecutionRecord,
    identity: SubagentExecutionIdentity
  ): PreparedSubagentExecutionIdentity {
    return {
      identity,
      lease: this.executions.beginIdentityPreparation(execution, identity),
    };
  }

  private async prepareCheckpointFork(
    sources: ReadonlyArray<SubagentCheckpointReference>,
    targetThreadId: string,
    lease: SubagentExecutionPreparationLease,
    approvalScope?: { source: string; target: string }
  ): Promise<void> {
    const checkpointer = this.checkpointer;
    try {
      if (
        checkpointer == null ||
        sources.length === 0 ||
        sources.every((source) => source.threadId === targetThreadId)
      ) {
        lease.assertOwned();
        return;
      }
      const targetExists = await this.checkpointThreadExists(targetThreadId);
      lease.assertOwned();
      if (!targetExists) {
        lease.markCheckpointCreated(() =>
          checkpointer.deleteThread(targetThreadId)
        );
      }
      await this.forkCheckpointSnapshot(sources, targetThreadId, approvalScope);
      lease.assertOwned();
    } catch (error) {
      return lease.rollback(error);
    }
  }

  private async checkpointThreadExists(threadId: string): Promise<boolean> {
    if (this.checkpointer == null) {
      return false;
    }
    for await (const _tuple of this.checkpointer.list(
      { configurable: { thread_id: threadId } },
      { limit: 1 }
    )) {
      return true;
    }
    return false;
  }

  /** Captures one exact checkpoint head per namespace for a child thread. */
  private async getLatestCheckpointSnapshot(
    threadId: string
  ): Promise<SubagentCheckpointReference[]> {
    if (this.checkpointer == null) {
      return [];
    }
    const checkpointsByNamespace = new Map<
      string,
      SubagentCheckpointReference
    >();
    for await (const tuple of this.checkpointer.list({
      configurable: { thread_id: threadId },
    })) {
      const checkpoint = getCheckpointReference(tuple);
      const current =
        checkpoint == null
          ? undefined
          : checkpointsByNamespace.get(checkpoint.checkpointNs);
      if (
        checkpoint != null &&
        (current == null ||
          checkpoint.checkpointId.localeCompare(current.checkpointId) > 0)
      ) {
        checkpointsByNamespace.set(checkpoint.checkpointNs, checkpoint);
      }
    }
    return [...checkpointsByNamespace.values()].sort((left, right) =>
      left.checkpointNs.localeCompare(right.checkpointNs)
    );
  }

  /** Copies exact checkpoint lineages, including pending task writes. */
  private async forkCheckpointSnapshot(
    sources: ReadonlyArray<SubagentCheckpointReference>,
    targetThreadId: string,
    approvalScope?: { source: string; target: string }
  ): Promise<void> {
    if (
      this.checkpointer == null ||
      sources.length === 0 ||
      sources.every((source) => source.threadId === targetThreadId)
    ) {
      return;
    }
    for (const source of sources) {
      const tuples: CheckpointTuple[] = [];
      const visited = new Set<string>();
      let tuple = await this.checkpointer.getTuple({
        configurable: {
          thread_id: source.threadId,
          checkpoint_ns: source.checkpointNs,
          checkpoint_id: source.checkpointId,
        },
      });
      if (tuple == null) {
        throw new Error(
          `Subagent checkpoint "${source.checkpointId}" is unavailable.`
        );
      }
      for (;;) {
        const reference = getCheckpointReference(tuple);
        if (reference == null) {
          throw new Error(
            'Subagent checkpoint lineage contains an invalid tuple.'
          );
        }
        if (
          reference.threadId !== source.threadId ||
          reference.checkpointNs !== source.checkpointNs
        ) {
          throw new Error(
            'Subagent checkpoint lineage escapes its source namespace.'
          );
        }
        const lineageKey = JSON.stringify([
          reference.threadId,
          reference.checkpointNs,
          reference.checkpointId,
        ]);
        if (visited.has(lineageKey)) {
          throw new Error('Subagent checkpoint lineage contains a cycle.');
        }
        visited.add(lineageKey);
        tuples.push(tuple);
        if (tuple.parentConfig == null) {
          break;
        }
        tuple = await this.checkpointer.getTuple(tuple.parentConfig);
        if (tuple == null) {
          throw new Error('Subagent checkpoint lineage is incomplete.');
        }
      }
      tuples.reverse();

      for (const lineageTuple of tuples) {
        const checkpointNs =
          getConfigurableString(lineageTuple.config, 'checkpoint_ns') ?? '';
        const parentCheckpointId = getConfigurableString(
          lineageTuple.parentConfig,
          'checkpoint_id'
        );
        const storedConfig = await this.checkpointer.put(
          {
            configurable: {
              thread_id: targetThreadId,
              checkpoint_ns: checkpointNs,
              ...(parentCheckpointId == null
                ? {}
                : { checkpoint_id: parentCheckpointId }),
            },
          },
          copyCheckpoint(lineageTuple.checkpoint),
          lineageTuple.metadata ?? {
            source: 'fork',
            step: -1,
            parents: {},
          },
          lineageTuple.checkpoint.channel_versions
        );
        const writesByTask = new Map<string, Array<[string, unknown]>>();
        for (const [taskId, channel, value] of lineageTuple.pendingWrites ??
          []) {
          let writes = writesByTask.get(taskId);
          if (writes == null) {
            writes = [];
            writesByTask.set(taskId, writes);
          }
          /** Repeated unconsumed forks must persist their current owner. */
          const reboundValue =
            channel === INTERRUPT &&
            approvalScope != null &&
            value != null &&
            typeof value === 'object' &&
            'value' in value
              ? {
                ...value,
                value: rebindToolBatchReplayPayload(
                  value.value,
                  approvalScope.source,
                  approvalScope.target,
                  targetThreadId
                ),
              }
              : value;
          writes.push([channel, reboundValue]);
        }
        for (const [taskId, writes] of writesByTask) {
          await this.checkpointer.putWrites(storedConfig, writes, taskId);
        }
      }
    }
  }

  private async createResumeManifest(
    records: ReadonlyArray<DurableExecutionRecord>
  ): Promise<SubagentResumeManifest | undefined> {
    if (this.checkpointer == null) {
      return undefined;
    }
    const executions: SubagentResumeExecution[] = [];
    for (const record of records) {
      const { identity, binding, activeRun } = record.snapshot;
      if (identity == null) {
        continue;
      }
      const { parentToolCallId } = record.address;
      const checkpoints = await this.getLatestCheckpointSnapshot(
        identity.childThreadId
      );
      if (checkpoints.length === 0) {
        continue;
      }
      const configuredHookSessionId =
        activeRun?.invokeConfig?.configurable?.run_id;
      const hookSessionId =
        typeof configuredHookSessionId === 'string' &&
        configuredHookSessionId.length > 0
          ? configuredHookSessionId
          : this.parentRunId;
      const approvalReplays =
        this.hookRegistry?.snapshotPendingToolApprovals(
          identity.approvalExecutionScope,
          identity.approvalExecutionScope
        ) ?? [];
      const descendant = activeRun?.pendingInterrupts
        .map((pendingInterrupt) =>
          getSubagentResumeManifest(pendingInterrupt.value)
        )
        .find((manifest) => manifest != null);
      const graphState = activeRun?.graph.createSubagentResumeState(
        hookSessionId
      ) ?? {
        toolCallSteps: [],
        toolSessions: [],
        toolNodes: [],
        eagerToolUsage: [],
        eagerToolSuppressions: [],
      };
      executions.push({
        parentToolCallId,
        childRunId: identity.childRunId,
        ...(binding?.subagentType == null
          ? {}
          : { subagentType: binding.subagentType }),
        ...(binding?.configId == null ? {} : { configId: binding.configId }),
        approvalExecutionScope: identity.approvalExecutionScope,
        checkpoints,
        graphState,
        approvalReplays,
        ...(descendant == null ? {} : { descendant }),
      });
    }
    if (executions.length === 0) {
      return undefined;
    }
    return {
      version: 1,
      executions,
    };
  }

  getResumeManifest(
    parentToolCallIds?: ReadonlySet<string>,
    config?: RunnableConfig
  ): Promise<SubagentResumeManifest | undefined> {
    return this.createResumeManifest(
      this.executions.selectForResume({ parentToolCallIds, config })
    );
  }

  getChildCheckpointThreadIds(): string[] {
    return this.executions.getCheckpointThreadIds((activeRun) =>
      this.getGraphChildCheckpointThreadIds(activeRun.graph)
    );
  }

  resetCheckpointThreadIds(): void {
    this.executions.resetCheckpointThreadIds();
  }

  private getGraphChildCheckpointThreadIds(graph: StandardGraph): string[] {
    const checkpointGraph = graph as {
      getChildCheckpointThreadIds?: () => string[];
    };
    return checkpointGraph.getChildCheckpointThreadIds?.() ?? [];
  }

  private clearChildGraph(graph: StandardGraph): void {
    this.executions.rememberCheckpointThread(
      ...this.getGraphChildCheckpointThreadIds(graph)
    );
    graph.clearHeavyState();
  }

  /**
   * Terminal sweep for a child graph's run steps. `Run.processStream` sweeps
   * only the graph it owns, and a child executes through `workflow.invoke()`
   * outside that loop — so without this, a child's last message step (no
   * successor step ever opens to close it) and every step of an aborted child
   * stay `in_progress` forever, leaving hosts with unmatched starts.
   *
   * Must run BEFORE `forwarding.drain()`: closures reach the parent only as
   * child custom events through the forwarder, which stops relaying once
   * drained.
   *
   * Deliberately not called on the interrupt path, mirroring the parent's
   * `isAwaitingResume` guard: an in-process resume continues into these same
   * steps, so closing them would strand the resumed run. A resume that
   * crosses a process boundary is a known gap either way — the child rebuilds
   * from `SubagentGraphResumeState`, which carries tool-call identity but not
   * `contentData`, so pre-interrupt steps are unreachable from the new graph
   * and stay open. That is the child-level instance of the same cross-process
   * orphan the parent graph has, and it needs persisted lifecycle state to
   * fix rather than a sweep here.
   *
   * `at` is the moment execution actually ended, threaded through so
   * observational work between then and the sweep cannot inflate the stamps.
   */
  private async closeChildRunSteps(
    graph: StandardGraph,
    status: Exclude<RunStepStatus, 'in_progress'>,
    at?: number
  ): Promise<void> {
    try {
      await graph.closeUnfinishedRunSteps(status, at);
    } catch (_e) {
      /** A failed sweep must never mask the subagent's own outcome */
    }
  }

  clearHeavyState(): void {
    this.executions.clear((record) => {
      if (record.activeRun != null) {
        this.clearChildGraph(record.activeRun.graph);
      }
      if (record.identity != null) {
        this.hookRegistry?.clearSession(record.identity.approvalExecutionScope);
      }
    });
    this.replayCheckpointWorkflow = undefined;
  }

  private async getResumeCheckpointMarker(
    resumeExecution: SubagentResumeExecution,
    parentToolCallId: string
  ): Promise<
    { marker: SubagentCheckpointMarker; messages: BaseMessage[] } | undefined
  > {
    if (this.checkpointer == null) {
      return undefined;
    }
    let fallback:
      | { marker: SubagentCheckpointMarker; messages: BaseMessage[] }
      | undefined;
    for (const checkpoint of resumeExecution.checkpoints) {
      const tuple = await this.checkpointer.getTuple({
        configurable: {
          thread_id: checkpoint.threadId,
          checkpoint_ns: checkpoint.checkpointNs,
          checkpoint_id: checkpoint.checkpointId,
        },
      });
      const messages = getTupleMessages(tuple);
      const marker = getSubagentCheckpointMarker(messages, parentToolCallId);
      if (marker?.settledOutput != null) {
        return { marker, messages };
      }
      if (marker != null && fallback == null) {
        fallback = { marker, messages };
      }
    }
    return fallback;
  }

  async getSettledToolOutput(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<SettledSubagentToolOutput | undefined> {
    const parentToolCallId = call.id;
    if (
      this.humanInTheLoop?.enabled !== true ||
      this.checkpointer == null ||
      parentToolCallId == null ||
      parentToolCallId === ''
    ) {
      return undefined;
    }
    const parentConfigurable = config.configurable as
      | Record<string, unknown>
      | undefined;
    const threadId = parentConfigurable?.thread_id;
    const execution = this.executions.open({
      threadId: typeof threadId === 'string' ? threadId : undefined,
      parentToolCallId,
      parentConfigurable,
    });
    const { resumeExecution } = execution;
    const inProcessSettledOutput = execution.settledOutput;
    if (inProcessSettledOutput != null) {
      const persistedSubagentType = getSubagentTypeFromArgs(
        inProcessSettledOutput.resolvedArgs
      );
      const subagentType =
        persistedSubagentType ??
        execution.binding?.subagentType ??
        getSubagentType(call);
      const executableConfig =
        subagentType == null ? undefined : this.configs.get(subagentType);
      if (
        !isResumeExecutionCompatible(
          resumeExecution,
          subagentType,
          executableConfig,
          persistedSubagentType != null
        )
      ) {
        return undefined;
      }
      return deserializeToolOutput(inProcessSettledOutput);
    }
    let resumeCheckpoint:
      | { marker: SubagentCheckpointMarker; messages: BaseMessage[] }
      | undefined;
    if (resumeExecution != null) {
      resumeCheckpoint = await this.getResumeCheckpointMarker(
        resumeExecution,
        parentToolCallId
      );
    }
    let marker = resumeCheckpoint?.marker;
    let messages = resumeCheckpoint?.messages ?? [];
    let persistedSubagentType = getSubagentTypeFromArgs(
      marker?.settledOutput?.resolvedArgs
    );
    let subagentType =
      persistedSubagentType ??
      resumeExecution?.subagentType ??
      getSubagentType(call);
    let executableConfig =
      subagentType == null ? undefined : this.configs.get(subagentType);
    if (
      !isResumeExecutionCompatible(
        resumeExecution,
        subagentType,
        executableConfig,
        persistedSubagentType != null
      )
    ) {
      this.executions.remove(execution);
      return undefined;
    }
    const { childThreadId } =
      await this.resolveChildExecutionIdentity(execution);
    if (marker == null) {
      const checkpoint = await this.checkpointer.getTuple({
        configurable: { thread_id: childThreadId },
      });
      messages = getCheckpointMessages(
        checkpoint?.checkpoint.channel_values.messages
      );
      marker = getSubagentCheckpointMarker(messages, parentToolCallId);
      persistedSubagentType = getSubagentTypeFromArgs(
        marker?.settledOutput?.resolvedArgs
      );
      if (persistedSubagentType != null) {
        subagentType = persistedSubagentType;
        executableConfig = this.configs.get(subagentType);
      }
      if (
        !isResumeExecutionCompatible(
          resumeExecution,
          subagentType,
          executableConfig,
          persistedSubagentType != null
        )
      ) {
        this.executions.remove(execution);
        return undefined;
      }
    }
    const configId = executableConfig?.configId;
    const bound = this.bindExecutionDefinition(
      execution,
      {
        ...(subagentType == null ? {} : { subagentType }),
        ...(configId == null ? {} : { configId }),
      },
      persistedSubagentType != null || resumeExecution?.subagentType != null
        ? 'effective'
        : 'provisional'
    );
    if (!bound) {
      return undefined;
    }
    const persistedHookSessionId =
      marker?.hookSessionId ?? getSubagentHookSessionId(messages);
    const currentHookSessionId = parentConfigurable?.run_id;
    if (
      persistedHookSessionId != null &&
      typeof currentHookSessionId === 'string' &&
      currentHookSessionId.length > 0
    ) {
      this.hookRegistry?.copySession(
        persistedHookSessionId,
        currentHookSessionId
      );
    }
    return marker?.settledOutput == null
      ? undefined
      : deserializeToolOutput(marker.settledOutput);
  }

  async persistSettledToolOutput(
    call: ToolCall,
    config: RunnableConfig,
    settled: SettledSubagentToolOutput
  ): Promise<void> {
    const parentToolCallId = call.id;
    const checkpointer = this.checkpointer;
    if (
      this.humanInTheLoop?.enabled !== true ||
      checkpointer == null ||
      parentToolCallId == null ||
      parentToolCallId === ''
    ) {
      return;
    }
    if (settled.output.status === 'error' && config.signal?.aborted === true) {
      return;
    }
    const parentConfigurable = config.configurable as
      | Record<string, unknown>
      | undefined;
    const threadId = parentConfigurable?.thread_id;
    const execution = this.executions.open({
      threadId: typeof threadId === 'string' ? threadId : undefined,
      parentToolCallId,
      parentConfigurable,
    });
    if (
      execution.completedResult != null &&
      settled.output.status === 'error'
    ) {
      return;
    }
    const { resumeExecution } = execution;
    const resolvedSubagentType = getSubagentTypeFromArgs(settled.resolvedArgs);
    const resolvedDescription = getSubagentDescriptionFromArgs(
      settled.resolvedArgs
    );
    const boundInvocation = execution.invocation;
    if (
      boundInvocation != null &&
      resolvedDescription != null &&
      resolvedDescription !== boundInvocation.description
    ) {
      return;
    }
    const description =
      boundInvocation?.description ??
      resolvedDescription ??
      getSubagentDescriptionFromArgs(call.args) ??
      DEFAULT_SUBAGENT_DESCRIPTION;
    const subagentType =
      resolvedSubagentType ??
      execution.binding?.subagentType ??
      getSubagentType(call);
    const executableConfig =
      subagentType == null ? undefined : this.configs.get(subagentType);
    const configId =
      resolvedSubagentType == null
        ? (execution.binding?.configId ??
          executableConfig?.configId ??
          resumeExecution?.configId)
        : executableConfig?.configId;
    if (subagentType == null) {
      return;
    }
    if (
      !isResumeExecutionCompatible(
        resumeExecution,
        subagentType,
        executableConfig
      )
    ) {
      this.executions.remove(execution);
      return;
    }
    const persistedOutput = serializeToolOutput(settled);
    const invocation = boundInvocation ?? {
      description,
      subagentType,
      ...(configId == null ? {} : { configId }),
    };
    try {
      await execution.settle(
        {
          definitionAuthority:
            resolvedSubagentType == null ? 'provisional' : 'effective',
          fingerprint: getSettlementFingerprint(persistedOutput),
          invocation,
          subagentType,
          ...(configId == null ? {} : { configId }),
        },
        persistedOutput,
        async (): Promise<void> => {
          const { childRunId, childThreadId } =
            await this.resolveChildExecutionIdentity(execution);
          const activeChildRun = execution.activeRun;
          if (activeChildRun != null) {
            await this.persistChildCheckpointMarker(
              activeChildRun,
              parentToolCallId,
              persistedOutput
            );
            this.clearChildGraph(activeChildRun.graph);
            return;
          }
          this.replayCheckpointWorkflow ??=
            createReplayCheckpointWorkflow(checkpointer);
          await this.replayCheckpointWorkflow.updateState(
            { configurable: { thread_id: childThreadId } },
            {
              messages: [
                createSubagentCheckpointMarkerMessage({
                  version: 1,
                  parentToolCallId,
                  lifecycleComplete: true,
                  hookSessionId:
                    typeof parentConfigurable?.run_id === 'string'
                      ? parentConfigurable.run_id
                      : this.parentRunId,
                  childRunId,
                  settledOutput: persistedOutput,
                }),
              ],
            },
            SUBAGENT_REPLAY_NODE
          );
        }
      );
    } catch (error) {
      if (
        error instanceof SubagentDefinitionBindingError ||
        error instanceof SubagentInvocationBindingError ||
        error instanceof SubagentSettlementBindingError
      ) {
        return;
      }
      throw error;
    }
  }

  private async persistChildCheckpointMarker(
    activeChildRun: ActiveChildRun,
    parentToolCallId: string,
    settledOutput?: PersistedToolOutput
  ): Promise<void> {
    if (
      this.humanInTheLoop?.enabled !== true ||
      activeChildRun.workflow.updateState == null ||
      activeChildRun.invokeConfig == null
    ) {
      return;
    }
    await activeChildRun.workflow.updateState(
      activeChildRun.invokeConfig,
      {
        messages: [
          createSubagentCheckpointMarkerMessage({
            version: 1,
            parentToolCallId,
            lifecycleComplete: true,
            hookSessionId:
              typeof activeChildRun.invokeConfig.configurable?.run_id ===
              'string'
                ? activeChildRun.invokeConfig.configurable.run_id
                : this.parentRunId,
            childRunId: activeChildRun.childRunId,
            ...(settledOutput == null ? {} : { settledOutput }),
          }),
        ],
      },
      activeChildRun.checkpointNodeId
    );
  }

  execute(params: SubagentExecuteParams): Promise<SubagentExecuteResult> {
    const executableConfig = this.configs.get(params.subagentType);
    if (executableConfig == null) {
      const available = [...this.configs.keys()].join(', ');
      return Promise.resolve(
        createSubagentFailure(
          `Error: Unknown subagent type "${params.subagentType}". Available types: ${available}`
        )
      );
    }
    if (this.maxDepth <= 0) {
      return Promise.resolve(
        createSubagentFailure('Error: Maximum subagent nesting depth exceeded.')
      );
    }
    if (
      isGraphSubagentConfig(executableConfig) &&
      this.humanInTheLoop?.enabled === true
    ) {
      return Promise.resolve(
        createSubagentFailure(
          'Error: Human-in-the-loop execution is not yet supported for graph subagents.'
        )
      );
    }
    if (
      this.humanInTheLoop?.enabled === true &&
      (params.parentToolCallId == null || params.parentToolCallId === '')
    ) {
      return Promise.resolve(
        createSubagentFailure(
          'Error: Resumable subagent execution requires a parent tool call ID.'
        )
      );
    }
    const execution = this.executions.open({
      threadId: params.threadId,
      parentToolCallId: params.parentToolCallId ?? nanoid(8),
      parentConfigurable: params.parentConfigurable,
    });
    try {
      return execution.execute(
        {
          description: params.description,
          subagentType: params.subagentType,
          ...(executableConfig.configId == null
            ? {}
            : { configId: executableConfig.configId }),
        },
        () => this.executeOnce(params, execution, executableConfig)
      );
    } catch (error) {
      if (error instanceof SubagentDefinitionBindingError) {
        return Promise.resolve(
          createSubagentFailure(SUBAGENT_CONFIG_CHANGED_MESSAGE)
        );
      }
      if (error instanceof SubagentInvocationBindingError) {
        return Promise.resolve(
          createSubagentFailure(SUBAGENT_INVOCATION_CHANGED_MESSAGE)
        );
      }
      throw error;
    }
  }

  private async executeOnce(
    params: SubagentExecuteParams,
    execution: DurableExecutionRecord,
    executableConfig: ExecutableSubagentConfigEntry
  ): Promise<SubagentExecuteResult> {
    const { description, subagentType, threadId, parentToolCallId } = params;
    /** Captured ONCE per execution, preferring the controller the parent
     * tool batch captured at ITS entry (before PreToolUse hooks): a failed
     * run's graph reset replaces the live controller, and resolving it here
     * — after the hook awaits — would bind this child to the NEW run's
     * un-aborted controller: reviving old-run work, or worse, tripping the
     * new run's breaker from an old child's stream-limit breach. Signal and
     * trip target both bind to this capture. */
    const childBreaker = params.breaker ?? this.resolveBreakerController();
    const childSignal = this.composeChildSignal(childBreaker, params.signal);

    const { parentToolCallId: executionSuffix } = execution.address;
    const { resumeExecution } = execution;
    if (
      !isResumeExecutionCompatible(
        resumeExecution,
        subagentType,
        executableConfig
      )
    ) {
      this.executions.remove(execution);
      return createSubagentFailure(SUBAGENT_CONFIG_CHANGED_MESSAGE);
    }
    let identity: SubagentExecutionIdentity;
    try {
      identity = await this.resolveChildExecutionIdentity(execution);
      execution.assertUsable(childSignal);
    } catch (error) {
      if (error instanceof StreamLimitExceededError) {
        throw error;
      }
      return createSubagentFailure(SUBAGENT_RESOLUTION_ERROR_MESSAGE);
    }
    const { childRunId, childThreadId, approvalExecutionScope } = identity;
    const bound = this.bindExecutionDefinition(
      execution,
      {
        subagentType,
        ...(executableConfig.configId == null
          ? {}
          : { configId: executableConfig.configId }),
      },
      'effective'
    );
    if (!bound) {
      return createSubagentFailure(SUBAGENT_CONFIG_CHANGED_MESSAGE);
    }
    let config: ResolvedSubagentConfigEntry;
    try {
      config = await this.resolveExecutionConfig(executableConfig, execution, {
        childRunId,
        childSignal,
        threadId,
        parentToolCallId,
        parentConfigurable: params.parentConfigurable,
      });
      execution.assertUsable(childSignal);
    } catch (error) {
      if (error instanceof StreamLimitExceededError) {
        throw error;
      }
      return createSubagentFailure(SUBAGENT_RESOLUTION_ERROR_MESSAGE);
    }

    const parentRegistry = this.getParentHandlerRegistry();
    const forwardingEnabled = parentRegistry != null;
    /**
     * Keep `toolDefinitions` only when the host has actually wired an
     * `ON_TOOL_EXECUTE` handler. `Run` always constructs a `HandlerRegistry`,
     * so treating any registry as "forwarding enabled" would leak
     * `toolDefinitions` into children whose hosts cannot execute them — the
     * child's `ToolNode` batch promise would hang forever with no handler to
     * resolve/reject. Gating on the tool-execute handler preserves the
     * recoverable "no tools" path for registry-but-no-handler configs.
     */
    const hasToolExecuteHandler =
      parentRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) != null;
    const childPlan = createChildGraphPlan({
      config,
      executionSuffix,
      parentAgentId: this.parentAgentId,
      parentMaxDepth: this.maxDepth,
      keepToolDefinitions: hasToolExecuteHandler,
    });
    const childAgentId = childPlan.subjectAgentId;
    const currentHookSessionId =
      asNonEmptyString(params.hookSessionId) ??
      asNonEmptyString(params.parentConfigurable?.run_id) ??
      this.executionContext.hookSessionId;
    const childExecutionContext: SubagentExecutionContext = {
      rootRunId: this.executionContext.rootRunId,
      hookSessionId: currentHookSessionId,
      depth: this.executionContext.depth + 1,
      ancestry: [
        ...this.executionContext.ancestry,
        {
          subagentRunId: childRunId,
          subagentType,
          subagentKind: childPlan.kind,
          subagentAgentId: childAgentId,
          parentRunId: this.parentRunId,
          parentAgentId: this.parentAgentId,
          parentToolCallId,
        },
      ],
    };
    const hostContextInput: SubagentContextInput = {
      executionContext: childExecutionContext,
      parentThreadId: threadId,
      memberAgentIds: childPlan.agents.map((agent) => agent.agentId),
      signal: childSignal,
      resumed: execution.started || resumeExecution != null,
    };
    let preparedContext: PreparedSubagentContext | undefined;
    try {
      preparedContext =
        this.subagentContext?.prepare == null
          ? undefined
          : await awaitWithAbort(
            Promise.resolve(this.subagentContext.prepare(hostContextInput)),
            childSignal
          );
      execution.assertUsable(childSignal);
      for (const [agentId, sessions] of Object.entries(
        preparedContext?.agentSessions ?? {}
      )) {
        const member = childPlan.memberInputs.get(agentId);
        if (member == null) throw new Error('Unknown subagent context member.');
        const activeMember =
          execution.activeRun?.graph.agentContexts.get(agentId);
        if (
          activeMember != null &&
          activeMember.codeSessionKey !== sessions.codeSessionKey
        ) {
          throw new Error(
            'Subagent session partition changed during execution.'
          );
        }
        member.codeSessionKey = sessions.codeSessionKey;
        member.initialSessions = sessions.initialSessions;
      }
    } catch (error) {
      if (childSignal.aborted) {
        throw childSignal.reason instanceof Error ? childSignal.reason : error;
      }
      if (execution.completedResult != null) {
        return {
          ...createSubagentFailure(
            'Subagent context is unavailable or access was denied.'
          ),
          retryableDelivery: true,
        };
      }
      return createSubagentFailure(
        'Subagent context is unavailable or access was denied.'
      );
    }
    const completedChildResult = execution.completedResult;
    if (completedChildResult != null) {
      return this.completeHostContext(hostContextInput, completedChildResult);
    }
    const maxTurns = config.maxTurns ?? DEFAULT_SUBAGENT_MAX_TURNS;
    const memberRecursionLimit = maxTurns * SUBAGENT_RECURSION_MULTIPLIER;
    const recursionLimit =
      memberRecursionLimit *
        (childPlan.kind === 'graph' ? childPlan.agents.length : 1) +
      (params.taskRuntime == null ? 0 : MAX_BACKGROUND_SUBAGENT_SEALS);

    const hostUsageSink = this.usageSink;
    let subagentUsageSink: SubagentUsageSink | undefined;
    if (hostUsageSink != null) {
      subagentUsageSink = (event): void | Promise<void> =>
        hostUsageSink({
          ...event,
          runId: this.executionContext.rootRunId,
        });
    }
    const cachedChildRun = execution.activeRun;
    const childGraphInput: StandardGraphInput = {
      runId: childRunId,
      signal: childSignal,
      agents: childPlan.agents,
      langfuse: this.langfuse,
      tokenCounter: this.tokenCounter,
      streamLimits: this.streamLimits,
      subagentScope: true,
      subagentExecutionContext: childExecutionContext,
      subagentContext: this.subagentContext,
      ...(resumeExecution?.graphState.fadingTier == null
        ? {}
        : { fadingTier: resumeExecution.graphState.fadingTier }),
      ...(resumeExecution?.graphState.fadingTiers == null
        ? {}
        : { fadingTiers: resumeExecution.graphState.fadingTiers }),
      /**
       * Forwarded so the child graph's own `SubagentExecutor` (created in
       * its `createAgentNode` when `allowNested` keeps subagentConfigs)
       * reports nested-child usage through the same host sink. Each nesting
       * level attaches its own capture callback — `workflow.invoke` replaces
       * the inherited callback chain, so a single top-level handler would
       * never see grandchild model calls.
       */
      subagentUsageSink,
      ...(params.taskRuntime == null
        ? {}
        : {
          preemption: {
            shouldPreempt: (): boolean =>
              params.taskRuntime?.shouldPreempt() === true,
            maxSeals: MAX_BACKGROUND_SUBAGENT_SEALS,
          },
        }),
    };
    let childGraph = cachedChildRun?.graph;
    if (childGraph == null && childPlan.kind === 'graph') {
      if (this.createChildGraphByKind == null) {
        return createSubagentFailure(
          'Error: Graph subagent execution requires a polymorphic child graph factory.'
        );
      }
      childGraph = this.createChildGraphByKind({
        kind: 'multi-agent',
        input: {
          ...childGraphInput,
          edges: childPlan.edges,
          resultAgentId: childPlan.resultAgentId,
          memberRecursionLimit,
        },
      });
    }
    childGraph ??= this.createChildGraph(childGraphInput);
    if (params.taskRuntime != null) {
      childGraph.hookRegistry = this.hookRegistry;
    }
    seedChildGraphSessions(childGraph, childPlan.agents);
    let forwarding: ForwarderCallback | undefined;
    if (forwardingEnabled) {
      forwarding = this.createForwarderCallback({
        parentRegistry: parentRegistry!,
        subagentType,
        subagentKind: childPlan.kind,
        subagentAgentId: childAgentId,
        graphMemberInputs:
          childPlan.kind === 'graph' ? childPlan.memberInputs : undefined,
        childRunId,
        parentToolCallId,
        executionContext: childExecutionContext,
      });
    }
    const forwarder = forwarding?.handler;
    let childAlreadyStarted = execution.started;
    let childAlreadyCompleted = execution.completed;

    let result: MultiAgentGraphState | undefined;
    let recoveredComplete = false;
    let recoveredInProgress = false;
    try {
      const workflow = (cachedChildRun?.workflow ??
        childGraph.createWorkflow()) as StatefulCompiledWorkflow;
      const activeChildRun = execution.activate(
        cachedChildRun ?? {
          graph: childGraph,
          workflow,
          pendingInterrupts: [],
          childAgentId,
          checkpointNodeId: childPlan.checkpointNodeId,
          childRunId,
        }
      );
      /**
       * When `parentHandlerRegistry` is provided (forwarding mode), attach a
       * lightweight callback that intercepts the child's `on_custom_event`
       * dispatches and routes them to the parent's registry — either as
       * operational events (ON_TOOL_EXECUTE) or wrapped ON_SUBAGENT_UPDATE
       * envelopes. Native LangChain streaming events (on_chat_model_stream,
       * etc.) still do NOT propagate to the parent's outer streamEvents
       * iterator — the `callbacks` array REPLACES the inherited chain, so
       * parent handlers won't receive child stream chunks and raise "No
       * agent context found" lookups on the parent's agentContexts map.
       *
       * When no registry is provided (legacy isolation), `callbacks: []`
       * fully detaches the child.
       *
       * `runName` gives the child a distinct LangSmith trace root (avoids
       * nested trace pollution).
       */
      const callbackHandlers: BaseCallbackHandler[] = [];
      if (forwarder) {
        callbackHandlers.push(forwarder);
      }
      /**
       * Usage capture rides the same detached callbacks array. Because
       * `callbacks` REPLACES the inherited chain (see above), the host's
       * `CHAT_MODEL_END` handler never observes the child's model calls —
       * this handler is the child-side equivalent of `ModelEndHandler`,
       * reporting per-call usage to the host's sink for billing.
       */
      if (this.usageSink) {
        callbackHandlers.push(
          createUsageCaptureHandler({
            sink: this.usageSink,
            subagentType,
            subagentKind: childPlan.kind,
            subagentRunId: childRunId,
            subagentAgentId: childAgentId,
            parentRunId: this.parentRunId,
            executionContext: childExecutionContext,
            memberInputs: childPlan.memberInputs,
          })
        );
      }
      const callbacks: Callbacks = callbackHandlers;
      /**
       * Inherit the parent's host `configurable` while binding LangGraph's
       * checkpoint identity to a stable child id derived from the durable
       * parent thread and checkpoint fork. The parent thread id cannot be
       * reused here: parent and child share one checkpointer when nested HITL
       * is enabled, and root checkpoint namespaces are normalized by
       * LangGraph, so a shared `thread_id` would collide with the parent.
       *
       * `run_id` still propagates as the parent run id, which is the key used
       * for session-scoped hook lookup. Child hook inputs therefore retain
       * the parent policy scope while their `threadId` truthfully identifies
       * the independently checkpointed child execution.
       */
      const inheritedConfigurable: Record<string, unknown> =
        sanitizeChildConfigurable(params.parentConfigurable);
      const { resumeAttemptId } = execution.address;
      if (cachedChildRun == null && resumeExecution != null) {
        childGraph.restoreSubagentResumeState(
          resumeExecution.graphState,
          currentHookSessionId
        );
      }
      const childConfigurable: Record<string, unknown> = {
        ...inheritedConfigurable,
        ...sanitizePreparedConfigurable(preparedContext?.configurable),
        executionContext: childExecutionContext,
      };
      if (this.humanInTheLoop?.enabled === true) {
        childConfigurable[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY] =
          approvalExecutionScope;
      }
      if (resumeExecution?.descendant != null) {
        childConfigurable[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY] =
          resumeExecution.descendant;
        childConfigurable[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY] = resumeAttemptId;
      }
      childConfigurable.thread_id =
        this.humanInTheLoop?.enabled === true
          ? childThreadId
          : (inheritedConfigurable.thread_id ?? childRunId);
      const childInvokeConfig = {
        recursionLimit,
        signal: childSignal,
        callbacks,
        runName: `subagent:${subagentType}`,
        configurable: childConfigurable,
      };
      activeChildRun.invokeConfig = childInvokeConfig;
      if (cachedChildRun == null && this.humanInTheLoop?.enabled === true) {
        /** Rehydrate child-owned interrupt state when a host rebuilds Run
         * around the same durable checkpointer after a process boundary. */
        const persistedState = await workflow.getState(childInvokeConfig);
        const checkpointMessages = getCheckpointMessages(
          (persistedState.values as { messages?: unknown } | undefined)
            ?.messages
        );
        const persistedHookSessionId =
          getSubagentHookSessionId(checkpointMessages);
        if (persistedHookSessionId != null) {
          this.hookRegistry?.copySession(
            persistedHookSessionId,
            currentHookSessionId
          );
        }
        const persistedInterrupts = getPersistedInterrupts(persistedState);
        if (persistedInterrupts.length > 0) {
          activeChildRun.pendingInterrupts =
            resumeExecution == null
              ? persistedInterrupts
              : persistedInterrupts.map((pending) => ({
                ...pending,
                value: rebindToolBatchReplayPayload(
                  pending.value,
                  resumeExecution.approvalExecutionScope,
                  approvalExecutionScope,
                  childThreadId
                ),
              }));
          execution.markStarted();
          childAlreadyStarted = true;
        } else if (persistedState.next.length > 0) {
          recoveredInProgress = true;
          childAlreadyStarted = true;
          execution.markStarted();
        } else if (persistedState.next.length === 0) {
          const persistedMessages = getPersistedMessages(persistedState);
          if (persistedMessages != null) {
            const marker = getSubagentCheckpointMarker(
              checkpointMessages,
              executionSuffix
            );
            result = { messages: persistedMessages };
            recoveredComplete = true;
            childAlreadyStarted = true;
            childAlreadyCompleted = marker?.lifecycleComplete === true;
            execution.markStarted();
          }
        }
      }
      if (!recoveredComplete) {
        const childResumeMap = getChildResumeMap(
          activeChildRun.pendingInterrupts,
          params.parentConfigurable
        );
        if (
          activeChildRun.pendingInterrupts.length > 0 &&
          childResumeMap == null
        ) {
          throw new GraphInterrupt(activeChildRun.pendingInterrupts);
        }
        let childInput: BaseGraphState | Command | null;
        if (childResumeMap != null) {
          const pending =
            activeChildRun.pendingInterrupts.find(
              (entry) =>
                entry.id != null &&
                Object.prototype.hasOwnProperty.call(childResumeMap, entry.id)
            ) ?? activeChildRun.pendingInterrupts[0];
          restoreToolReplayConfig(childConfigurable, pending.id, pending.value);
          rebindToolBatchReplayScope(
            childConfigurable,
            resumeExecution?.approvalExecutionScope ?? approvalExecutionScope,
            approvalExecutionScope,
            childThreadId
          );
          childInput = new Command({ resume: childResumeMap });
        } else if (recoveredInProgress) {
          childInput = null;
        } else {
          childInput = {
            messages: [
              ...(params.initialMessages ?? []),
              new HumanMessage({
                content: description,
                additional_kwargs: {
                  [SUBAGENT_HOOK_SESSION_KEY]: currentHookSessionId,
                  [SUBAGENT_RUN_ID_KEY]: childRunId,
                },
              }),
              ...(preparedContext?.messages ?? []),
            ],
          };
        }

        if (
          !childAlreadyStarted &&
          this.hookRegistry?.hasHookFor(
            'SubagentStart',
            currentHookSessionId
          ) === true
        ) {
          const hookResult = await executeHooks({
            registry: this.hookRegistry,
            input: {
              hook_event_name: 'SubagentStart',
              executionContext: childExecutionContext,
              runId: currentHookSessionId,
              threadId,
              parentAgentId: this.parentAgentId,
              agentId: childAgentId,
              agentType: subagentType,
              inputs: [new HumanMessage(description)],
            },
            sessionId: currentHookSessionId,
            matchQuery: subagentType,
          }).catch((): AggregatedHookResult => HOOK_FALLBACK);

          if (hookResult.decision === 'deny' || hookResult.decision === 'ask') {
            this.clearChildGraph(childGraph);
            execution.releaseActiveRun();
            this.executions.remove(execution);
            return {
              content: `Blocked: ${hookResult.reason ?? 'Blocked by hook'}`,
              messages: [],
            };
          }
        }
        execution.markStarted();

        if (forwarder && !childAlreadyStarted) {
          await this.emitSubagentUpdate(parentRegistry!, {
            childRunId,
            subagentType,
            subagentKind: childPlan.kind,
            subagentAgentId: childAgentId,
            parentToolCallId,
            executionContext: childExecutionContext,
            phase: 'start',
            label: `Subagent "${subagentType}" started`,
          });
        }

        for (;;) {
          let childResult: MultiAgentGraphState;
          if (
            this.humanInTheLoop?.enabled === true ||
            params.taskRuntime != null
          ) {
            /** Execute as an independent LangGraph root. HITL children need an
             * isolated Pregel namespace, while detached children must replace the
             * parent ToolNode's ambient runnable config so its AbortSignal cannot
             * cancel task-owned work when the parent stream settles. */
            childResult =
              await AsyncLocalStorageProviderSingleton.runWithConfig(
                childInvokeConfig,
                (): Promise<MultiAgentGraphState> =>
                  workflow.invoke(childInput, childInvokeConfig)
              );
          } else {
            childResult = await workflow.invoke(childInput, childInvokeConfig);
          }
          const childInterrupts = isInterrupted(childResult)
            ? childResult[INTERRUPT]
            : undefined;
          if (childInterrupts != null && childInterrupts.length > 0) {
            throw new GraphInterrupt(childInterrupts);
          }
          result = childResult;
          const continuation = params.taskRuntime?.closeTurn();
          if (continuation == null || continuation.closed) {
            break;
          }
          /**
           * A queued control becomes the next user turn inside the same
           * detached task. Reset graph sidecars while carrying the complete
           * child transcript forward, preserving one task/trace identity and
           * avoiding a second subagent lifecycle or duplicate billing path.
           */
          childGraph.resetValues(true);
          childInput = {
            messages: [
              ...childResult.messages,
              ...convertInjectedMessages(continuation.messages),
            ],
          };
        }
      }
    } catch (error) {
      /** Stamped at failure, not after the error-envelope work below. */
      const childTerminalAt = Date.now();
      /**
       * Captured at catch entry, BEFORE the self-abort below flips it. The
       * closure sweep distinguishes "stopped on purpose" from "died of this
       * error" by whether the child was already aborted when the error
       * arrived — reading the signal after `childBreaker.abort(error)` would
       * relabel the child's own stream-limit failure as an intentional stop
       * (`cancelled`), while the parent stamps `failed` for the same
       * incident. A trip that arrived from a parallel sibling has already
       * aborted the composed signal by this point, so it still reads as
       * `cancelled` here.
       */
      const abortedBeforeError = childSignal.aborted;
      if (isGraphInterrupt(error)) {
        const activeChildRun = execution.activeRun;
        if (activeChildRun != null) {
          activeChildRun.pendingInterrupts = error.interrupts;
          execution.markInterrupted();
        }
        const resumeManifest =
          activeChildRun == null || parentToolCallId == null
            ? undefined
            : await this.createResumeManifest([execution]);
        await forwarding?.drain();
        throw new GraphInterrupt(
          addSubagentScope(
            error.interrupts,
            {
              run_id: childRunId,
              agent_id: childAgentId,
              subagent_type: subagentType,
              parent_tool_call_id: parentToolCallId,
            },
            resumeManifest
          )
        );
      }
      /** Aborted before any observational work below: parallel siblings — in
       * this executor and, via the graph-scoped breaker, under other
       * parallel agent nodes — stream on the composed child signal, and
       * awaiting forwarding.drain() first would let them consume provider
       * quota for that entire interval. Trips the ENTRY-captured controller:
       * after a reset, a straggler must break its own dead run, not the
       * current one. */
      if (error instanceof StreamLimitExceededError) {
        childBreaker.abort(error);
      }
      const errorMessage = truncateErrorMessage(error);
      /**
       * `cancelled` vs `failed` mirrors `Run.resolveSweepStatus`: an aborted
       * child was stopped on purpose (caller abort, or a breaker trip from a
       * parallel sibling), anything else died of an unexpected error. Uses
       * the pre-error snapshot, not the live signal — the self-abort above
       * has already tripped it for the child's own limit error.
       */
      await this.closeChildRunSteps(
        childGraph,
        abortedBeforeError ? 'cancelled' : 'failed',
        childTerminalAt
      );
      if (forwarding) {
        await forwarding.drain();
        await this.emitSubagentUpdate(parentRegistry!, {
          childRunId,
          subagentType,
          subagentKind: childPlan.kind,
          subagentAgentId: childAgentId,
          parentToolCallId,
          executionContext: childExecutionContext,
          phase: 'error',
          label: `Subagent "${subagentType}" errored: ${errorMessage}`,
          data: { message: errorMessage },
        });
      }
      this.clearChildGraph(childGraph);
      execution.markFailed();
      /**
       * A tripped stream circuit breaker is a safety abort, not a recoverable
       * subagent failure: converting it into a tool result would let the
       * parent keep generating (or spawn another child) after the limit
       * fired. Rethrown here and passed through ToolNode's error conversion,
       * so the parent run rejects with the child's limit error.
       */
      if (error instanceof StreamLimitExceededError) {
        throw error;
      }
      return createSubagentFailure(
        `Subagent error: ${errorMessage}`,
        errorMessage
      );
    }

    /**
     * Child execution ended here. Captured before the observational work
     * below — an awaited `SubagentStop` hook, then the forwarder drain — so a
     * slow hook cannot inflate the closure stamps with its own latency.
     * Mirrors the parent's `terminalAt` capture in `Run.processStream`.
     */
    const childTerminalAt = Date.now();

    if (result == null) {
      throw new Error('Subagent completed without producing graph state.');
    }
    const filteredContent =
      childPlan.kind === 'graph'
        ? filterGraphSubagentResult(result, childPlan.resultAgentId)
        : filterSubagentResult(result.messages);

    if (
      !childAlreadyCompleted &&
      this.hookRegistry?.hasHookFor('SubagentStop', currentHookSessionId) ===
        true
    ) {
      /**
       * Awaited (not fire-and-forget) for deterministic test synchronization
       * and consistency with PostCompact. The parent is already waiting on the
       * tool result, so the small extra latency is acceptable. Errors are
       * swallowed — SubagentStop is observational.
       */
      await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'SubagentStop',
          executionContext: childExecutionContext,
          runId: currentHookSessionId,
          threadId,
          agentId: childAgentId,
          agentType: subagentType,
          messages: result.messages,
        },
        sessionId: currentHookSessionId,
        matchQuery: subagentType,
      }).catch(() => {
        /* SubagentStop is observational — swallow errors */
      });
    }

    if (!childAlreadyCompleted) {
      await this.closeChildRunSteps(childGraph, 'completed', childTerminalAt);
    }

    if (forwarding && !childAlreadyCompleted) {
      await forwarding.drain();
      await this.emitSubagentUpdate(parentRegistry!, {
        childRunId,
        subagentType,
        subagentKind: childPlan.kind,
        subagentAgentId: childAgentId,
        parentToolCallId,
        executionContext: childExecutionContext,
        phase: 'stop',
        label: `Subagent "${subagentType}" finished`,
      });
    }
    if (!childAlreadyCompleted) {
      const activeChildRun = execution.activeRun;
      if (activeChildRun != null && parentToolCallId != null) {
        await this.persistChildCheckpointMarker(
          activeChildRun,
          parentToolCallId
        );
      }
    }

    this.clearChildGraph(childGraph);

    const completedResult: SubagentExecuteResult = {
      content: filteredContent,
      messages: result.messages,
    };
    execution.markCompleted(completedResult);
    return this.completeHostContext(hostContextInput, completedResult);
  }

  private async completeHostContext(
    input: SubagentContextInput,
    result: SubagentExecuteResult
  ): Promise<SubagentExecuteResult> {
    if (this.subagentContext?.complete == null) return result;
    try {
      input.signal.throwIfAborted();
      const completed = await awaitWithAbort(
        this.subagentContext.complete(input, result),
        input.signal
      );
      input.signal.throwIfAborted();
      return { ...result, content: completed.content };
    } catch {
      return {
        ...createSubagentFailure('Subagent result delivery failed.'),
        retryableDelivery: true,
      };
    }
  }

  /**
   * Emits a single {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope through the
   * parent's handler registry. Silent no-op when no parent registry is set.
   * Errors are swallowed — update events are observational.
   */
  private async emitSubagentUpdate(
    parentRegistry: HandlerRegistry,
    args: {
      childRunId: string;
      subagentType: string;
      subagentKind: 'agent' | 'graph';
      subagentAgentId: string;
      memberAgentId?: string;
      parentToolCallId?: string;
      executionContext: SubagentExecutionContext;
      phase: SubagentUpdatePhase;
      data?: unknown;
      label?: string;
    }
  ): Promise<void> {
    const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
    if (!handler) {
      return;
    }
    const event: SubagentUpdateEvent = {
      runId: args.executionContext.rootRunId,
      parentRunId: this.parentRunId,
      subagentRunId: args.childRunId,
      subagentType: args.subagentType,
      subagentKind: args.subagentKind,
      subagentAgentId: args.subagentAgentId,
      memberAgentId: args.memberAgentId,
      parentAgentId: this.parentAgentId,
      parentToolCallId: args.parentToolCallId,
      depth: args.executionContext.depth,
      ancestry: args.executionContext.ancestry,
      phase: args.phase,
      data: args.data,
      label: args.label,
      timestamp: new Date().toISOString(),
    };
    await dispatchObservationalSubagentUpdate(handler, event);
  }

  /**
   * Builds a BaseCallbackHandler that intercepts the child graph's custom
   * events. Routing rules:
   *   - `ON_TOOL_EXECUTE` → forwarded as-is to the parent's ON_TOOL_EXECUTE
   *     handler (so event-driven tools work identically for child and parent).
   *   - `ON_RUN_STEP` / `ON_RUN_STEP_DELTA` / `ON_RUN_STEP_COMPLETED` /
   *     `ON_MESSAGE_DELTA` / `ON_REASONING_DELTA` → wrapped in a
   *     {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope with a human-readable
   *     label, delivered to the parent's subagent-update handler.
   *   - Everything else → ignored (keeps parent's UI scoped to the events it
   *     cares about; host apps can extend by registering more phases).
   */
  private createForwarderCallback(args: {
    parentRegistry: HandlerRegistry;
    subagentType: string;
    subagentKind: 'agent' | 'graph';
    subagentAgentId: string;
    graphMemberInputs?: ReadonlyMap<string, AgentInputs>;
    childRunId: string;
    parentToolCallId?: string;
    executionContext: SubagentExecutionContext;
  }): ForwarderCallback {
    const {
      parentRegistry,
      subagentType,
      subagentKind,
      subagentAgentId,
      graphMemberInputs,
      childRunId,
      parentToolCallId,
      executionContext,
    } = args;
    const immediateParentRunId = this.parentRunId;
    const parentAgentId = this.parentAgentId;
    const memberAgentIdByStepId = new Map<string, string>();

    const wrap = async (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown,
      memberAgentId?: string
    ): Promise<boolean> => {
      const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
      if (!handler) {
        return true;
      }
      const event: SubagentUpdateEvent = {
        runId: executionContext.rootRunId,
        parentRunId: immediateParentRunId,
        subagentRunId: childRunId,
        subagentType,
        subagentKind,
        subagentAgentId,
        memberAgentId,
        parentAgentId,
        parentToolCallId,
        depth: executionContext.depth,
        ancestry: executionContext.ancestry,
        phase,
        data: sanitizeForwardedSubagentUpdateData(eventName, data),
        label: summarizeEvent(eventName, data),
        timestamp: new Date().toISOString(),
      };
      return dispatchObservationalSubagentUpdate(handler, event);
    };

    const queuedUpdates: QueuedSubagentUpdate[] = [];
    let drainPromise: Promise<void> | undefined;
    let queueOpen = true;

    const enqueue = (update: QueuedSubagentUpdate): void => {
      if (!queueOpen) {
        return;
      }
      if (queuedUpdates.length >= MAX_QUEUED_SUBAGENT_UPDATES) {
        const dropIndex = queuedUpdates.findIndex((queued) =>
          isLowPrioritySubagentUpdatePhase(queued.phase)
        );
        if (dropIndex >= 0) {
          queuedUpdates.splice(dropIndex, 1);
        } else if (isLowPrioritySubagentUpdatePhase(update.phase)) {
          return;
        } else {
          queuedUpdates.shift();
        }
      }
      queuedUpdates.push(update);
    };

    const drain = async (): Promise<void> => {
      if (drainPromise != null) {
        await drainPromise;
        return;
      }
      drainPromise = (async (): Promise<void> => {
        while (queuedUpdates.length > 0) {
          const update = queuedUpdates.shift();
          if (update == null) {
            continue;
          }
          const handlerSettled = await wrap(
            update.eventName,
            update.phase,
            update.data,
            update.memberAgentId
          );
          if (!handlerSettled) {
            queueOpen = false;
            queuedUpdates.length = 0;
          }
        }
      })();
      try {
        await drainPromise;
      } finally {
        drainPromise = undefined;
        if (queuedUpdates.length > 0) {
          await drain();
        }
      }
    };

    const scheduleWrap = (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown,
      memberAgentId?: string
    ): void => {
      enqueue({ eventName, phase, data, memberAgentId });
      void drain();
    };

    const handler = BaseCallbackHandler.fromMethods({
      [Callback.CUSTOM_EVENT]: async (
        eventName: string,
        data: unknown,
        _runId?: string,
        _tags?: string[],
        metadata?: Record<string, unknown>
      ): Promise<void> => {
        const memberAgentId =
          graphMemberInputs == null
            ? (asNonEmptyString(metadata?.agentId) ?? getEventAgentId(data))
            : getGraphEventMemberAgentId({
              data,
              metadata,
              memberInputs: graphMemberInputs,
              memberAgentIdByStepId,
            });
        if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
          const toolHandler = parentRegistry.getHandler(
            GraphEvents.ON_TOOL_EXECUTE
          );
          if (toolHandler) {
            await toolHandler.handle(
              GraphEvents.ON_TOOL_EXECUTE,
              data as ToolExecuteBatchRequest
            );
          }
          /**
           * We also surface a short notice in the subagent-update stream so
           * the UI can show "calling <tool>" for each tool the child spawns.
           */
          scheduleWrap(eventName, 'run_step', data, memberAgentId);
          return;
        }

        if (eventName === GraphEvents.ON_RUN_STEP) {
          scheduleWrap(eventName, 'run_step', data, memberAgentId);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
          scheduleWrap(eventName, 'run_step_delta', data, memberAgentId);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
          scheduleWrap(eventName, 'run_step_completed', data, memberAgentId);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_CLOSED) {
          scheduleWrap(eventName, 'run_step_closed', data, memberAgentId);
          return;
        }
        if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
          scheduleWrap(eventName, 'message_delta', data, memberAgentId);
          return;
        }
        if (eventName === GraphEvents.ON_REASONING_DELTA) {
          scheduleWrap(eventName, 'reasoning_delta', data, memberAgentId);
          return;
        }
      },
    });
    /**
     * `awaitHandlers = true` is required so the child's `ToolNode` actually
     * blocks on the parent's `ON_TOOL_EXECUTE` handler until it resolves
     * the batch request. Observational `ON_SUBAGENT_UPDATE` calls are queued
     * behind a bounded sequential dispatcher so host UI publication cannot
     * backpressure each child emission or run unbounded concurrent publishes.
     * The executor drains this queue before terminal stop/error envelopes to
     * preserve phase ordering.
     */
    handler.awaitHandlers = true;
    return { handler, drain };
  }
}

/**
 * Builds the child-run equivalent of a host `CHAT_MODEL_END` handler: a
 * callback that joins per-call model identity (captured from
 * `ls_model_name` at chat-model start) with the usage metadata reported at
 * LLM end, and emits a {@link SubagentUsageEvent} through the host's sink.
 *
 * Attached to the child `workflow.invoke` callbacks array, so it observes
 * every model call inside the child graph — the agent loop and any
 * auxiliary calls (e.g. child-side summarization). It does NOT observe
 * deeper subagent levels: each nesting level replaces the callback chain
 * and attaches its own capture handler via the forwarded
 * `subagentUsageSink` on the child graph's input.
 */
function createUsageCaptureHandler(args: {
  sink: SubagentUsageSink;
  subagentType: string;
  subagentKind: 'agent' | 'graph';
  subagentRunId: string;
  subagentAgentId: string;
  parentRunId: string;
  executionContext: SubagentExecutionContext;
  memberInputs: ReadonlyMap<string, AgentInputs>;
}): BaseCallbackHandler {
  const {
    sink,
    subagentType,
    subagentKind,
    subagentRunId,
    subagentAgentId,
    parentRunId,
    executionContext,
    memberInputs,
  } = args;
  const defaultMember =
    memberInputs.size === 1 ? memberInputs.entries().next().value : undefined;
  /**
   * Per-call attribution keyed by LangChain callback runId. `model` joins
   * `ls_model_name` (provider-reported) with `INVOKED_MODEL` (stamped by
   * `tryFallbackProviders` from the fallback's client options); `provider`
   * is `INVOKED_PROVIDER`, stamped by `attemptInvoke` with the SDK enum of
   * the provider that ACTUALLY served the call — correct for
   * fallback-served calls, where the static config provider would mis-tag
   * pricing/cache semantics.
   */
  const callInfoByCallId = new Map<
    string,
    { model?: string; provider?: string; memberAgentId?: string }
  >();
  const handler = BaseCallbackHandler.fromMethods({
    handleChatModelStart: (
      _llm: unknown,
      _messages: unknown,
      runId: string,
      _parentRunId?: string,
      _extraParams?: Record<string, unknown>,
      _tags?: string[],
      metadata?: Record<string, unknown>
    ): void => {
      const callModel =
        asNonEmptyString(metadata?.ls_model_name) ??
        asNonEmptyString(metadata?.[Constants.INVOKED_MODEL]);
      const callProvider = asNonEmptyString(
        metadata?.[Constants.INVOKED_PROVIDER]
      );
      const memberAgentId = asNonEmptyString(metadata?.agentId);
      if (callModel != null || callProvider != null || memberAgentId != null) {
        callInfoByCallId.set(runId, {
          model: callModel,
          provider: callProvider,
          memberAgentId,
        });
      }
    },
    handleLLMEnd: async (output: LLMResult, runId: string): Promise<void> => {
      const callInfo = callInfoByCallId.get(runId);
      callInfoByCallId.delete(runId);
      const observedMemberAgentId = callInfo?.memberAgentId;
      const memberAgentId =
        observedMemberAgentId != null && memberInputs.has(observedMemberAgentId)
          ? observedMemberAgentId
          : defaultMember?.[0];
      const memberInput =
        (memberAgentId == null ? undefined : memberInputs.get(memberAgentId)) ??
        defaultMember?.[1];
      const model =
        callInfo?.model ??
        resolveClientOptionsModel(memberInput?.clientOptions);
      const callProvider = callInfo?.provider ?? memberInput?.provider;
      for (const generationGroup of output.generations) {
        /**
         * At most ONE event per generation group: each group is one
         * provider request (the outer array is per-prompt for batched
         * calls), and with multiple completions (`n > 1`) every choice in
         * a group repeats the request-level `usage_metadata` — emitting
         * per choice would multiply billed tokens.
         */
        for (const generation of generationGroup) {
          const message = (generation as ChatGeneration | undefined)?.message;
          const usage = (
            message as { usage_metadata?: UsageMetadata } | undefined
          )?.usage_metadata;
          if (usage == null) {
            continue;
          }
          /**
           * Awaited so async host sinks (billing/persistence) complete
           * before the model call resolves — `awaitHandlers` only waits on
           * `handleLLMEnd` itself, so a dropped promise here would let the
           * parent run finish before usage is recorded and would turn sink
           * rejections into unhandled rejections.
           */
          try {
            await sink({
              usage,
              modelRunId: runId,
              model,
              provider: callProvider,
              subagentType,
              subagentKind,
              subagentRunId,
              subagentAgentId,
              memberAgentId,
              parentRunId,
              depth: executionContext.depth,
              ancestry: executionContext.ancestry,
              runId: executionContext.rootRunId,
            });
          } catch {
            /* observational — a throwing/rejecting host sink must not break the child run */
          }
          break;
        }
      }
    },
    handleLLMError: (_err: unknown, runId: string): void => {
      callInfoByCallId.delete(runId);
    },
  });
  /**
   * Dispatch usage synchronously with each model call so all entries are
   * sunk before `workflow.invoke` resolves — hosts read their accumulator
   * right after the parent run completes.
   */
  handler.awaitHandlers = true;
  return handler;
}

function asNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined;
}

function getEventAgentId(data: unknown): string | undefined {
  if (data == null || typeof data !== 'object') {
    return undefined;
  }
  const event = data as {
    agentId?: unknown;
    result?: { agentId?: unknown };
  };
  return (
    asNonEmptyString(event.agentId) ?? asNonEmptyString(event.result?.agentId)
  );
}

function getEventStepId(data: unknown): string | undefined {
  if (data == null || typeof data !== 'object') {
    return undefined;
  }
  const event = data as {
    id?: unknown;
    result?: { id?: unknown };
  };
  return asNonEmptyString(event.id) ?? asNonEmptyString(event.result?.id);
}

function getGraphEventMemberAgentId({
  data,
  metadata,
  memberInputs,
  memberAgentIdByStepId,
}: {
  data: unknown;
  metadata?: Record<string, unknown>;
  memberInputs: ReadonlyMap<string, AgentInputs>;
  memberAgentIdByStepId: Map<string, string>;
}): string | undefined {
  const eventAgentId = getEventAgentId(data);
  const stepId = getEventStepId(data);
  if (eventAgentId != null) {
    if (!memberInputs.has(eventAgentId)) {
      return undefined;
    }
    if (stepId != null) {
      memberAgentIdByStepId.set(stepId, eventAgentId);
    }
    return eventAgentId;
  }

  const stepMemberAgentId =
    stepId == null ? undefined : memberAgentIdByStepId.get(stepId);
  if (stepMemberAgentId != null) {
    return stepMemberAgentId;
  }

  const metadataAgentId = asNonEmptyString(metadata?.agentId);
  return metadataAgentId != null && memberInputs.has(metadataAgentId)
    ? metadataAgentId
    : undefined;
}

function sanitizeResolverConfigurable(
  parentConfigurable: Record<string, unknown>
): SubagentResolveConfigurable | undefined {
  const configurable = sanitizeChildConfigurable(parentConfigurable);
  const requestBody = createResolveRequestContext(configurable.requestBody);
  const user = createResolveUserContext(configurable.user);
  const userId = asNonEmptyString(configurable.user_id) ?? user?.id;
  if (requestBody == null && user == null && userId == null) {
    return undefined;
  }
  return {
    ...(requestBody == null ? {} : { requestBody }),
    ...(user == null ? {} : { user }),
    ...(userId == null ? {} : { user_id: userId }),
  };
}

function createResolveRequestContext(
  value: unknown
): SubagentResolveRequestContext | undefined {
  if (!isObjectLike(value)) {
    return undefined;
  }
  const source = value as SubagentResolveRequestContext;
  const conversationId = asNonEmptyString(source.conversationId);
  const messageId = asNonEmptyString(source.messageId);
  const parentMessageId = asNonEmptyString(source.parentMessageId);
  if (conversationId == null && messageId == null && parentMessageId == null) {
    return undefined;
  }
  return {
    ...(conversationId == null ? {} : { conversationId }),
    ...(messageId == null ? {} : { messageId }),
    ...(parentMessageId == null ? {} : { parentMessageId }),
  };
}

function createResolveUserContext(
  value: unknown
): SubagentResolveUserContext | undefined {
  if (!isObjectLike(value)) {
    return undefined;
  }
  const source = value as SubagentResolveUserContext;
  const id = asNonEmptyString(source.id);
  const role = asNonEmptyString(source.role);
  const tenantId = asNonEmptyString(source.tenantId);
  if (id == null && role == null && tenantId == null) {
    return undefined;
  }
  return {
    ...(id == null ? {} : { id }),
    ...(role == null ? {} : { role }),
    ...(tenantId == null ? {} : { tenantId }),
  };
}

function sanitizeChildConfigurable(
  parentConfigurable: Record<string, unknown> | undefined
): Record<string, unknown> {
  if (parentConfigurable == null) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(parentConfigurable).filter(
      ([key]) => !isLangGraphRuntimeConfigKey(key)
    )
  );
}

function sanitizePreparedConfigurable(
  configurable: RunnableConfig['configurable']
): Record<string, unknown> {
  const prepared = sanitizeChildConfigurable(configurable);
  delete prepared.run_id;
  delete prepared.parent_run_id;
  delete prepared.thread_id;
  delete prepared.executionContext;
  return prepared;
}

function isLangGraphRuntimeConfigKey(key: string): boolean {
  return (
    key.startsWith(LANGGRAPH_RUNTIME_CONFIG_PREFIX) ||
    LANGGRAPH_CHECKPOINT_CONFIG_KEYS.has(key) ||
    key === SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY ||
    key === SUBAGENT_RESUME_MANIFEST_CONFIG_KEY ||
    key === SUBAGENT_PARENT_BATCH_CONFIG_KEY ||
    key === TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY ||
    /** The parent batch's breaker scope must not leak into the child
     * workflow's configurable — children own separate controllers. */
    key === RUN_BREAKER_SCOPE_CONFIG_KEY
  );
}

export function sanitizeForwardedSubagentUpdateData(
  eventName: string,
  data: unknown
): unknown {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    return sanitizeToolExecuteUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    return sanitizeRunStepUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
    return sanitizeRunStepDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    return sanitizeRunStepCompletedUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_CLOSED) {
    return sanitizeRunStepClosedUpdateData(data);
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return sanitizeMessageDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_REASONING_DELTA) {
    return sanitizeReasoningDeltaUpdateData(data);
  }
  return undefined;
}

function isLowPrioritySubagentUpdatePhase(phase: SubagentUpdatePhase): boolean {
  return (
    phase === 'message_delta' ||
    phase === 'reasoning_delta' ||
    phase === 'run_step_delta'
  );
}

function sanitizeToolExecuteUpdateData(
  data: unknown
): SanitizedSubagentToolExecuteData {
  const request = data as Partial<ToolExecuteBatchRequest>;
  const toolCalls = Array.isArray(request.toolCalls)
    ? request.toolCalls.map(sanitizeToolCallForUpdate)
    : [];
  const sanitized: SanitizedSubagentToolExecuteData = { toolCalls };
  if (typeof request.agentId === 'string') {
    sanitized.agentId = request.agentId;
  }
  return sanitized;
}

function sanitizeToolCallForUpdate(
  call: ToolExecuteBatchRequest['toolCalls'][number]
): SanitizedSubagentToolCall {
  const sanitized: SanitizedSubagentToolCall = {
    id: call.id,
    name: call.name,
    args: call.args,
  };
  return sanitized;
}

function sanitizeRunStepUpdateData(
  data: unknown
): SanitizedRunStep | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const step = data as Partial<RunStep>;
  const sanitized: SanitizedRunStep = {};
  assignString(sanitized, 'agentId', step.agentId);
  assignNumber(sanitized, 'created_at', step.created_at);
  assignNumber(sanitized, 'groupId', step.groupId);
  assignString(sanitized, 'id', step.id);
  assignNumber(sanitized, 'index', step.index);
  assignString(sanitized, 'runId', step.runId);
  assignString(sanitized, 'status', step.status);
  assignNumber(sanitized, 'stepIndex', step.stepIndex);
  assignString(sanitized, 'type', step.type);
  if (step.summary !== undefined) {
    sanitized.summary = step.summary;
  }
  if (step.usage !== undefined) {
    sanitized.usage = step.usage;
  }
  sanitized.stepDetails = sanitizeStepDetails(step.stepDetails);
  return sanitized;
}

function sanitizeRunStepClosedUpdateData(
  data: unknown
): SanitizedRunStepClosed | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<RunStepClosedEvent>;
  const sanitized: SanitizedRunStepClosed = {};
  assignString(sanitized, 'agentId', event.agentId);
  assignNumber(sanitized, 'closed_at', event.closed_at);
  assignNumber(sanitized, 'created_at', event.created_at);
  assignNumber(sanitized, 'groupId', event.groupId);
  assignString(sanitized, 'id', event.id);
  assignNumber(sanitized, 'index', event.index);
  assignString(sanitized, 'runId', event.runId);
  assignString(sanitized, 'status', event.status);
  assignNumber(sanitized, 'stepIndex', event.stepIndex);
  assignString(sanitized, 'type', event.type);
  return sanitized;
}

function sanitizeRunStepDeltaUpdateData(
  data: unknown
): SanitizedRunStepDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<RunStepDeltaEvent>;
  const sanitized: SanitizedRunStepDelta = {};
  assignString(sanitized, 'id', event.id);
  sanitized.delta = sanitizeToolCallDelta(event.delta);
  return sanitized;
}

function sanitizeRunStepCompletedUpdateData(
  data: unknown
): SanitizedRunStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as { result?: unknown };
  return { result: sanitizeStepCompleted(event.result) };
}

function sanitizeMessageDeltaUpdateData(
  data: unknown
): SanitizedMessageDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<MessageDeltaEvent>;
  const sanitized: SanitizedMessageDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta != null) {
    sanitized.delta = {};
    if (event.delta.content !== undefined) {
      sanitized.delta.content = event.delta.content;
    }
    if (event.delta.tool_call_ids !== undefined) {
      sanitized.delta.tool_call_ids = event.delta.tool_call_ids;
    }
  }
  return sanitized;
}

function sanitizeReasoningDeltaUpdateData(
  data: unknown
): SanitizedReasoningDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<ReasoningDeltaEvent>;
  const sanitized: SanitizedReasoningDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta?.content !== undefined) {
    sanitized.delta = { content: event.delta.content };
  }
  return sanitized;
}

function sanitizeStepDetails(
  stepDetails: unknown
): SanitizedStepDetails | undefined {
  if (!isObjectLike(stepDetails)) {
    return undefined;
  }
  const rawDetails = stepDetails as {
    message_creation?: {
      message_id?: unknown;
      content_type?: unknown;
      phase?: unknown;
    };
    tool_calls?: unknown[];
    type?: unknown;
  };
  if (rawDetails.type === StepTypes.MESSAGE_CREATION) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.MESSAGE_CREATION,
    };
    const rawMessageCreation = rawDetails.message_creation;
    const messageId = rawMessageCreation?.message_id;
    const contentType = rawMessageCreation?.content_type;
    const phase = rawMessageCreation?.phase;
    if (
      typeof messageId === 'string' ||
      contentType === ContentTypes.TEXT ||
      contentType === ContentTypes.THINK ||
      phase === 'commentary' ||
      phase === 'final_answer'
    ) {
      sanitized.message_creation = {
        ...(typeof messageId === 'string' ? { message_id: messageId } : {}),
        ...(contentType === ContentTypes.TEXT ||
        contentType === ContentTypes.THINK
          ? { content_type: contentType }
          : {}),
        ...(phase === 'commentary' || phase === 'final_answer'
          ? { phase }
          : {}),
      };
    }
    return sanitized;
  }
  if (rawDetails.type === StepTypes.TOOL_CALLS) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.TOOL_CALLS,
    };
    if (Array.isArray(rawDetails.tool_calls)) {
      sanitized.tool_calls = rawDetails.tool_calls.map(sanitizeAgentToolCall);
    }
    return sanitized;
  }
  return undefined;
}

function sanitizeToolCallDelta(
  delta: ToolCallDelta | undefined
): SanitizedToolCallDelta | undefined {
  if (!isObjectLike(delta)) {
    return undefined;
  }
  const sanitized: SanitizedToolCallDelta = {};
  assignString(sanitized, 'auth', delta.auth);
  assignNumber(sanitized, 'expires_at', delta.expires_at);
  assignString(sanitized, 'type', delta.type);
  if (delta.summary !== undefined) {
    sanitized.summary = delta.summary;
  }
  if (Array.isArray(delta.tool_calls)) {
    sanitized.tool_calls = delta.tool_calls.map(sanitizeAgentToolCall);
  }
  return sanitized;
}

function sanitizeStepCompleted(
  data: unknown
): SanitizedStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const completed = data as Partial<StepCompleted> & {
    id?: unknown;
    index?: unknown;
    tool_call?: unknown;
    completed_at?: unknown;
  };
  if (completed.type === 'summary') {
    return {
      type: 'summary',
      summary: completed.summary,
    };
  }
  if (completed.type !== 'tool_call') {
    return undefined;
  }
  const sanitized: SanitizedStepCompleted = { type: 'tool_call' };
  assignString(sanitized, 'id', completed.id);
  assignNumber(sanitized, 'index', completed.index);
  assignNumber(sanitized, 'completed_at', completed.completed_at);
  sanitized.tool_call = sanitizeProcessedToolCall(completed.tool_call);
  return sanitized;
}

function sanitizeProcessedToolCall(
  toolCall: unknown
): SanitizedProcessedToolCall | undefined {
  if (!isObjectLike(toolCall)) {
    return undefined;
  }
  const call = toolCall as Partial<ProcessedToolCall>;
  const sanitized: SanitizedProcessedToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  assignString(sanitized, 'output', call.output);
  assignString(sanitized, 'outcome', call.outcome);
  assignNumber(sanitized, 'progress', call.progress);
  return sanitized;
}

function sanitizeAgentToolCall(toolCall: unknown): SanitizedAgentToolCall {
  if (!isObjectLike(toolCall)) {
    return {};
  }
  const call = toolCall as SanitizedAgentToolCall;
  const sanitized: SanitizedAgentToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  assignString(sanitized, 'type', call.type);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  if (isObjectLike(call.function)) {
    const fn: SanitizedAgentToolCall['function'] = {};
    assignString(fn, 'name', call.function.name);
    if (
      typeof call.function.arguments === 'string' ||
      isObjectLike(call.function.arguments)
    ) {
      fn.arguments = call.function.arguments;
    }
    sanitized.function = fn;
  }
  return sanitized;
}

function isObjectLike(value: unknown): value is object {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assignString<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'string') {
    target[key] = value as T[K];
  }
}

function assignNumber<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'number') {
    target[key] = value as T[K];
  }
}

/**
 * Produces a short single-line label for an arbitrary forwarded child event.
 * Used to populate {@link SubagentUpdateEvent.label} so the host UI can show
 * a compact status ticker without parsing the raw payload.
 */
export function summarizeEvent(eventName: string, data: unknown): string {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    const req = data as { toolCalls?: Array<{ name?: string }> };
    const names = (req.toolCalls ?? [])
      .map((c) => c.name)
      .filter((n): n is string => typeof n === 'string');
    return names.length > 0 ? `Calling ${names.join(', ')}` : 'Calling tool';
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    const step = data as {
      type?: string;
      stepDetails?: { type?: string; tool_calls?: Array<{ name?: string }> };
    };
    const detailType = step.stepDetails?.type ?? step.type ?? 'step';
    if (detailType === 'tool_calls') {
      const names = (step.stepDetails?.tool_calls ?? [])
        .map((c) => c.name)
        .filter((n): n is string => typeof n === 'string');
      return names.length > 0
        ? `Using tool: ${names.join(', ')}`
        : 'Planning tool call';
    }
    if (detailType === 'message_creation') {
      return 'Thinking…';
    }
    return `Step: ${detailType}`;
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    const step = data as {
      result?: {
        type?: string;
        tool_call?: { name?: string; output?: string };
      };
    };
    const tool = step.result?.tool_call;
    if (tool?.name != null && tool.name !== '') {
      return `Tool ${tool.name} complete`;
    }
    return 'Step complete';
  }
  if (eventName === GraphEvents.ON_RUN_STEP_CLOSED) {
    const closed = data as { status?: string };
    return closed.status != null && closed.status !== ''
      ? `Step ${closed.status}`
      : 'Step closed';
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return 'Streaming…';
  }
  return eventName;
}

/**
 * Reads only the result member captured by MultiAgentGraph. Worker output is
 * never used as a fallback when the designated result member is textless.
 */
export function filterGraphSubagentResult(
  state: MultiAgentGraphState,
  resultAgentId: string
): string {
  const result = state.subagentResult;
  if (result == null || result.agentId !== resultAgentId) {
    return `Error: Graph subagent result agent "${resultAgentId}" did not produce a result.`;
  }
  if (result.message == null) {
    return 'Task completed';
  }
  return filterSubagentResult([result.message]);
}

/**
 * Walk messages from last to first, returning the text content of the most
 * recent AIMessage that has any. Non-text blocks (tool_use, thinking,
 * redacted_thinking, tool_result) are stripped. If the last AIMessage is
 * pure tool_use (e.g. the subagent hit `maxTurns` mid-tool-call), the walk
 * continues to earlier AIMessages so partial progress is salvaged — this
 * matches Claude Code's behavior in `agentToolUtils.finalizeAgentTool`.
 * Consecutive streamed text-delta blocks with the same provider index are
 * coalesced without adding whitespace. Annotation-only text blocks are
 * ignored; complete text blocks and distinct delta indexes remain separated.
 * Returns "Task completed" only when no AIMessage in the history contains
 * any text.
 */
export function filterSubagentResult(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() !== 'ai') {
      continue;
    }

    const content = messages[i].content;

    if (typeof content === 'string') {
      if (content) return content;
      continue;
    }

    if (!Array.isArray(content)) {
      continue;
    }

    const textParts: string[] = [];
    let textDeltaParts: string[] = [];
    let textDeltaIndex: string | number | undefined;
    const flushTextDeltaParts = (): void => {
      if (textDeltaParts.length === 0) {
        return;
      }
      textParts.push(textDeltaParts.join(''));
      textDeltaParts = [];
      textDeltaIndex = undefined;
    };
    for (const block of content) {
      if (typeof block === 'string') {
        flushTextDeltaParts();
        if (block !== '') {
          textParts.push(block);
        }
        continue;
      }

      const type =
        'type' in block && typeof block.type === 'string' ? block.type : '';
      const isTextDelta = type === TEXT_DELTA_CONTENT_TYPE;
      const isText = type === ContentTypes.TEXT || isTextDelta;
      const text =
        isText && 'text' in block && typeof block.text === 'string'
          ? block.text
          : '';
      if (isTextDelta) {
        if (text === '') {
          continue;
        }
        const index =
          'index' in block &&
          (typeof block.index === 'string' || typeof block.index === 'number')
            ? block.index
            : undefined;
        if (
          textDeltaIndex != null &&
          index != null &&
          index !== textDeltaIndex
        ) {
          flushTextDeltaParts();
        }
        textDeltaIndex ??= index;
        textDeltaParts.push(text);
        continue;
      }

      if (type === ContentTypes.TEXT && text === '') {
        continue;
      }

      flushTextDeltaParts();
      if (text !== '') {
        textParts.push(text);
      }
    }
    flushTextDeltaParts();

    if (textParts.length > 0) {
      return textParts.join('\n');
    }
  }

  return 'Task completed';
}

function truncateErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.length <= ERROR_MESSAGE_MAX_CHARS) {
    return message;
  }
  return `${message.slice(0, ERROR_MESSAGE_MAX_CHARS)}...`;
}

function resolveAgentInputsWithSignal(
  resolve: () => Promise<AgentInputs>,
  signal: AbortSignal
): Promise<AgentInputs> {
  if (signal.aborted) {
    return Promise.reject(getAbortReason(signal));
  }

  return new Promise<AgentInputs>((resolvePromise, rejectPromise) => {
    const cleanup = (): void => signal.removeEventListener('abort', onAbort);
    const onAbort = (): void => {
      cleanup();
      rejectPromise(getAbortReason(signal));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    Promise.resolve()
      .then(resolve)
      .then(
        (agentInputs) => {
          cleanup();
          resolvePromise(agentInputs);
        },
        (error: unknown) => {
          cleanup();
          rejectPromise(error);
        }
      );
  });
}

function getAbortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error('Subagent resolution aborted.');
}

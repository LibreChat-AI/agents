// src/types/graph.ts
import type {
  BaseMessage,
  AIMessageChunk,
  SystemMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { GoogleAIToolType } from '@langchain/google-common';
import type {
  RunStep,
  ModelResponseEvent,
  ModelToolsClaimedEvent,
  RunStepDeltaEvent,
  RunStepResumeState,
  RunStepClosedEvent,
  MessageDeltaEvent,
  ReasoningDeltaEvent,
} from '@/types/stream';
import type {
  SummarizationNodeInput,
  SummarizeCompleteEvent,
  CompactionSemanticIndex,
  SummarizationConfig,
  SummarizeStartEvent,
  SummarizeDeltaEvent,
} from '@/types/summarize';
import type {
  ToolMap,
  ToolSessionMap,
  ToolEndEvent,
  GenericTool,
  LCTool,
  ToolExecutionConfig,
  ToolExecuteBatchRequest,
} from '@/types/tools';
import type {
  TokenCounter,
  StreamLimits,
  StreamPreemption,
  TokenBudgetBreakdown,
} from '@/types/run';
import type { SubagentTaskConfig } from '@/types/subagentTasks';
import type { StandardGraph, MultiAgentGraph } from '@/graphs';
import type { ProviderClientOptionsConfig } from '@/types/llm';
import type { Callback, GraphNodeKeys } from '@/common';

/** Interface for bound model with stream and invoke methods */
export interface ChatModel {
  stream?: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<AsyncIterable<AIMessageChunk>>;
  invoke: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<AIMessageChunk>;
}

export type GraphNode = GraphNodeKeys | typeof START;
export type ClientCallback<T extends unknown[]> = (
  graph: StandardGraph,
  ...args: T
) => void;

export type ClientCallbacks = {
  [Callback.TOOL_ERROR]?: ClientCallback<[Error, string]>;
  [Callback.TOOL_START]?: ClientCallback<unknown[]>;
  [Callback.TOOL_END]?: ClientCallback<unknown[]>;
};

export type SystemCallbacks = {
  [K in keyof ClientCallbacks]: ClientCallbacks[K] extends ClientCallback<
    infer Args
  >
    ? (...args: Args) => void
    : never;
};

/** An executed, SDK-generated handoff, identified independently of stream events. */
export interface HandoffTransition {
  id: string;
  sourceAgentId: string;
  targetAgentId: string;
  toolCallId: string;
  scope: 'turn' | 'conversation';
  /** Causal order: number of handoffs visible at this batch's input. */
  depth: number;
}

/** Versioned graph checkpoint payload, scoped to one logical user turn. */
export interface HandoffState {
  version: 1;
  executionId: string;
  entryAgentId: string;
  maxHandoffs?: number;
  transitions: HandoffTransition[];
  parallel: boolean;
  /** False when resuming an older checkpoint without complete routing provenance. */
  historyComplete?: boolean;
}

/** Only a candidate from a completed top-level run may be promoted by a host. */
export type HandoffOutcome = {
  executionId: string;
  entryAgentId: string;
  transitions: readonly HandoffTransition[];
} & (
  | { status: 'candidate'; agentId: string; transitionId: string }
  | { status: 'unchanged' | 'ambiguous' }
  | { status: 'incomplete'; reason: string }
);

export type BaseGraphState = {
  messages: BaseMessage[];
  runStepState?: RunStepResumeState;
  /** SDK-owned routing state; hosts must not synthesize it from messages. */
  handoffState?: HandoffState;
  /**
   * The summary a summarize-only run produced. Kept in state because such a
   * run has no assistant reply: trace roots report it as the run's output
   * instead of the retained tail or the raw state. Empty once any later run
   * on the same checkpointed thread starts, so it never outlives its run.
   */
  manualSummary?: string;
};

export type AgentSubgraphState = BaseGraphState & {
  summarizationRequest?: SummarizationNodeInput;
};

export type SubagentGraphResult = {
  agentId: string;
  message?: BaseMessage;
};

export type MultiAgentGraphState = BaseGraphState & {
  agentMessages?: BaseMessage[];
  subagentResult?: SubagentGraphResult;
};

export type IState = BaseGraphState;

export interface AgentLogEvent {
  level: 'debug' | 'info' | 'warn' | 'error';
  scope: 'prune' | 'summarize' | 'graph' | 'sanitize' | (string & {});
  message: string;
  data?: Record<string, unknown>;
  runId?: string;
  agentId?: string;
}

/**
 * Per-model-call context window usage snapshot, dispatched after pruning and
 * before the model invocation. Dispatched once per `callModel` invocation:
 * fallback retries reuse the snapshot since the prompt is identical — budget
 * numbers reflect the primary provider's tokenizer, and the calibration
 * ratio self-corrects from whichever provider reports usage.
 */
export interface ContextUsageEvent {
  runId?: string;
  agentId?: string;
  /** Structural token budget snapshot from AgentContext.getTokenBudgetBreakdown */
  breakdown: TokenBudgetBreakdown;
  /** Usable budget this call: maxContextTokens minus output reserve */
  contextBudget?: number;
  /** Calibrated instruction overhead actually applied this call */
  effectiveInstructionTokens?: number;
  /** Calibrated message tokens before pruning (excluding instructions) */
  prePruneContextTokens?: number;
  /** Tokens still free after instructions + pruned messages */
  remainingContextTokens?: number;
  /** EMA ratio of provider-reported vs locally estimated token counts */
  calibrationRatio?: number;
}

/**
 * Latched context-fading tier. Caps for historical tool results derive from
 * `(budgetTokens, masked)` only, so carrying the tier across runs keeps their
 * truncated bytes stable for prefix-based provider prompt caches. Hosts
 * persist it beside `calibrationRatio` and pass it back via
 * `RunConfig.fadingTiers[agentId]`.
 */
export interface FadingTier {
  v: 2;
  /** Token budget the caps derive from, in raw token space. Never grows;
   *  clamped to the current context window when seeded. */
  budgetTokens: number;
  /** Whether observation masking has activated. Never deactivates. */
  masked: boolean;
  /** Whether this tier was reduced, masked, or restored from host state. */
  latched?: true;
}

/** Latched fading tiers keyed by agent ID. */
export type FadingTiers = Record<string, FadingTier>;

export interface EventHandler {
  handle(
    event: string,
    data:
      | StreamEventData
      | ModelEndData
      | ModelResponseEvent
      | ModelToolsClaimedEvent
      | RunStep
      | RunStepDeltaEvent
      | RunStepClosedEvent
      | MessageDeltaEvent
      | ReasoningDeltaEvent
      | SummarizeStartEvent
      | SummarizeDeltaEvent
      | SummarizeCompleteEvent
      | SubagentUpdateEvent
      | AgentLogEvent
      | ContextUsageEvent
      | ToolExecuteBatchRequest
      | { result: ToolEndEvent },
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): void | Promise<void>;
}

export type GraphStateChannels<T extends BaseGraphState> =
  StateGraphArgs<T>['channels'];

export type Workflow<
  T extends BaseGraphState = BaseGraphState,
  U extends Partial<T> = Partial<T>,
  N extends string = string,
> = StateGraph<T, U, N>;

type LangChainEventStreamCallbackHandlerInput = NonNullable<
  Parameters<Runnable['streamEvents']>[2]
>;

export type EventStreamCallbackHandlerInput =
  LangChainEventStreamCallbackHandlerInput & {
    autoClose?: boolean;
    raiseError?: boolean;
    ignoreCustomEvent?: boolean;
  };

export type WorkflowValuesStreamConfig = RunnableConfig & {
  streamMode: 'values';
};

/**
 * LangGraph stream output is mode-dependent (`values`, `updates`, SSE, etc.).
 * Keep the base Runnable stream output as unknown and narrow at callsites that
 * choose a concrete streamMode.
 */
export type CompiledWorkflow<
  TInput extends BaseGraphState = BaseGraphState,
  TOutput extends BaseGraphState = TInput,
> = Omit<Runnable<TInput, unknown>, 'invoke'> & {
  invoke(input: TInput, config?: RunnableConfig): Promise<TOutput>;
};

export type CompiledStateWorkflow = CompiledWorkflow;

export type CompiledMultiAgentWorkflow = CompiledWorkflow<MultiAgentGraphState>;

export type CompiledAgentWorfklow = CompiledWorkflow<
  AgentSubgraphState,
  AgentSubgraphState
>;

export type SystemRunnable =
  | Runnable<
      BaseMessage[],
      (BaseMessage | SystemMessage)[],
      RunnableConfig<Record<string, unknown>>
    >
  | undefined;

/**
 * Optional compile options passed to workflow.compile().
 * These are intentionally untyped to avoid coupling to library internals.
 */
export type CompileOptions = {
  checkpointer?: unknown;
  interruptBefore?: string[];
  interruptAfter?: string[];
};

export type StreamChunk =
  | (ChatGenerationChunk & {
      message: AIMessageChunk;
    })
  | AIMessageChunk;

/**
 * Data associated with a StreamEvent.
 */
export type StreamEventData = {
  /**
   * The input passed to the runnable that generated the event.
   * Inputs will sometimes be available at the *START* of the runnable, and
   * sometimes at the *END* of the runnable.
   * If a runnable is able to stream its inputs, then its input by definition
   * won't be known until the *END* of the runnable when it has finished streaming
   * its inputs.
   */
  input?: unknown;
  /**
   * The output of the runnable that generated the event.
   * Outputs will only be available at the *END* of the runnable.
   * For most runnables, this field can be inferred from the `chunk` field,
   * though there might be some exceptions for special cased runnables (e.g., like
   * chat models), which may return more information.
   */
  output?: unknown;
  /**
   * A streaming chunk from the output that generated the event.
   * chunks support addition in general, and adding them up should result
   * in the output of the runnable that generated the event.
   */
  chunk?: StreamChunk;
  /**
   * Runnable config for invoking other runnables within handlers.
   */
  config?: RunnableConfig;
  /**
   * Custom result from the runnable that generated the event.
   */
  result?: unknown;
  /**
   * Custom field to indicate the event was manually emitted, and may have been handled already
   */
  emitted?: boolean;
};

/**
 * A streaming event.
 *
 * Schema of a streaming event which is produced from the streamEvents method.
 */
export type StreamEvent = {
  /**
   * Event names are of the format: on_[runnable_type]_(start|stream|end).
   *
   * Runnable types are one of:
   * - llm - used by non chat models
   * - chat_model - used by chat models
   * - prompt --  e.g., ChatPromptTemplate
   * - tool -- LangChain tools
   * - chain - most Runnables are of this type
   *
   * Further, the events are categorized as one of:
   * - start - when the runnable starts
   * - stream - when the runnable is streaming
   * - end - when the runnable ends
   *
   * start, stream and end are associated with slightly different `data` payload.
   *
   * Please see the documentation for `EventData` for more details.
   */
  event: string;
  /** The name of the runnable that generated the event. */
  name: string;
  /**
   * An randomly generated ID to keep track of the execution of the given runnable.
   *
   * Each child runnable that gets invoked as part of the execution of a parent runnable
   * is assigned its own unique ID.
   */
  run_id: string;
  /**
   * Tags associated with the runnable that generated this event.
   * Tags are always inherited from parent runnables.
   */
  tags?: string[];
  /** Metadata associated with the runnable that generated this event. */
  metadata: Record<string, unknown>;
  /**
   * Event data.
   *
   * The contents of the event data depend on the event type.
   */
  data: StreamEventData;
};

export type GraphConfig = {
  provider: string;
  thread_id?: string;
  run_id?: string;
};

export type PartMetadata = {
  progress?: number;
  asset_pointer?: string;
  status?: string;
  action?: boolean;
  output?: string;
  auth?: string;
  expires_at?: number;
};

export type ModelEndData =
  | (StreamEventData & { output: AIMessageChunk | undefined })
  | undefined;
export type GraphTools = GenericTool[] | BindToolsInput[] | GoogleAIToolType[];
export type StandardGraphInput = {
  runId?: string;
  signal?: AbortSignal;
  agents: AgentInputs[];
  /** Execution backend used to resolve the effective tool registry. */
  toolExecution?: ToolExecutionConfig;
  /** Trusted single-agent client delegation policy; mixed batches fail closed. */
  clientDelegatedToolNames?: readonly string[];
  langfuse?: LangfuseConfig;
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  calibrationRatio?: number;
  /** Persisted default-agent tier retained for single-agent compatibility. */
  fadingTier?: FadingTier | null;
  /** Persisted fading tiers keyed by agent ID. */
  fadingTiers?: FadingTiers | null;
  /**
   * Receives a {@link SubagentUsageEvent} for every model call that reports
   * usage metadata inside a subagent child run spawned from this graph
   * (including nested subagents and child-side summarization calls). Child graphs run via
   * `invoke()` outside the host's `streamEvents` loop, so their
   * `on_chat_model_end` events never reach the run's handler registry —
   * this sink is the only way hosts can observe child token usage for
   * billing/accounting. Parent-graph model calls are NOT reported here;
   * they already flow through the registry's `CHAT_MODEL_END` handler.
   */
  subagentUsageSink?: SubagentUsageSink;
  /**
   * Optional host-owned process-local task namespace for detached subagents.
   * Presence enables `run_in_background` on the subagent tool. Child graphs
   * do not inherit it, keeping background nesting disabled for the MVP.
   */
  subagentTasks?: SubagentTaskConfig;
  /** Host-owned context and result projection for each isolated child execution. */
  subagentContext?: SubagentContextAdapter;
  /**
   * True when this graph IS a subagent child run (set by `SubagentExecutor`
   * when it constructs the child graph). Drives the hook-input `agentId`
   * subagent-scope marker: hook dispatches from this graph's tool nodes
   * carry `agentId` so run-scoped host hooks — which fire for child scopes
   * too, because children inherit the parent's `run_id` — can tell child
   * scope from the top level. Top-level graphs leave this unset and their
   * hook inputs carry only `executingAgentId`.
   */
  subagentScope?: boolean;
  /**
   * Cooperative preemption, forwarded from `RunConfig.preemption`. Ordinary
   * child graphs do not inherit it. Detached subagent tasks may receive their
   * own internal parent-control source so an interrupt can reuse the same
   * provider-safe sealing path without targeting the top-level conversation.
   */
  preemption?: StreamPreemption;
  /**
   * Stream circuit breakers, forwarded from `RunConfig.streamLimits` and
   * resolved once at graph construction. Enforced by the stream handler on
   * every streamed chunk event. See {@link StreamLimits}.
   */
  streamLimits?: StreamLimits;
  /** Structured lineage for a graph executing as a subagent child. */
  subagentExecutionContext?: SubagentExecutionContext;
};

export type GraphEdge = {
  /** Agent ID; direct edges use an array as an all-of waiting source group. */
  from: string | string[];
  /** Agent ID, use a list for multiple destinations */
  to: string | string[];
  description?: string;
  /** Can return boolean or specific destination(s) */
  condition?: (state: BaseGraphState) => boolean | string | string[];
  /** 'handoff' creates tools for dynamic routing, 'direct' creates direct edges, which also allow parallel execution */
  edgeType?: 'handoff' | 'direct';
  /** Host may promote the destination after successful top-level completion. */
  handoffScope?: 'turn' | 'conversation';
  /**
   * For direct edges: Optional prompt to add when transitioning through this edge.
   * String prompts can include variables like {results} which will be replaced with
   * messages from startIndex onwards. When {results} is used, excludeResults defaults to true.
   *
   * For handoff edges: Description for the input parameter that the handoff tool accepts,
   * allowing the supervisor to pass specific instructions/context to the transferred agent.
   */
  prompt?:
    | string
    | ((
        messages: BaseMessage[],
        runStartIndex: number
      ) => string | Promise<string> | undefined);
  /**
   * When true, excludes messages from startIndex when adding prompt.
   * Automatically set to true when {results} variable is used in prompt.
   */
  excludeResults?: boolean;
  /**
   * For handoff edges: Customizes the parameter name for the handoff input.
   * Defaults to "instructions" if not specified.
   * Only applies when prompt is provided for handoff edges.
   */
  promptKey?: string;
};

export type GraphSubagentEdge = Omit<
  GraphEdge,
  'edgeType' | 'condition' | 'promptKey' | 'handoffScope'
> & {
  edgeType: 'direct';
  condition?: never;
  promptKey?: never;
  handoffScope?: never;
};

export type MultiAgentGraphInput = StandardGraphInput & {
  edges: GraphEdge[];
  /** Explicit fresh-turn entry; absent preserves topology-inferred entry points. */
  entryAgentId?: string;
  /** Shared logical-turn handoff cap. Zero forbids handoffs; absent uses only recursion limits. */
  maxHandoffs?: number;
  /** Captures the designated member's final AI turn in graph state. */
  resultAgentId?: string;
  /** Optional per-member Pregel budget when the outer graph has its own topology budget. */
  memberRecursionLimit?: number;
};

/** Lightweight identity advertised to the model for a spawnable subagent. */
export interface SubagentDescriptor {
  /** Stable identifier used in the tool's `subagent_type` enum (e.g. 'researcher', 'coder'). */
  type: string;
  /** Human-readable display name. */
  name: string;
  /** What this subagent specializes in — shown to the LLM. */
  description: string;
  /**
   * Opaque, versioned identity of the child configuration. Required when
   * `resolveAgentInputs` is used and changed whenever its resolved inputs
   * change incompatibly.
   */
  configId?: string;
}

/** Lazy descriptor with the durable configuration identity required to resolve it. */
export interface LazySubagentDescriptor extends SubagentDescriptor {
  configId: string;
}

/** Stable request identifiers exposed to a selected subagent resolver. */
export interface SubagentResolveRequestContext {
  conversationId?: string;
  messageId?: string;
  parentMessageId?: string;
}

/** Stable user identifiers exposed to a selected subagent resolver. */
export interface SubagentResolveUserContext {
  id?: string;
  role?: string;
  tenantId?: string;
}

/** Sanitized host runtime context safe for lazy subagent resolution. */
export interface SubagentResolveConfigurable {
  requestBody?: Readonly<SubagentResolveRequestContext>;
  user?: Readonly<SubagentResolveUserContext>;
  user_id?: string;
}

/** Runtime context supplied when a host lazily resolves a selected subagent. */
export interface SubagentResolveContext {
  /** Stable subagent identity selected by the model. */
  descriptor: Readonly<LazySubagentDescriptor>;
  /** Stable child execution identity, including across HITL reconstruction. */
  executionId: string;
  /** Parent run that dispatched this execution. */
  parentRunId: string;
  /** Parent agent that dispatched this execution. */
  parentAgentId?: string;
  /** Parent-side tool call that selected the subagent. */
  parentToolCallId?: string;
  /** Durable parent conversation thread, when supplied by the host. */
  threadId?: string;
  /** Parent/breaker cancellation composed for this child execution. */
  signal: AbortSignal;
  /** Stable, sanitized host context from the parent tool invocation. */
  configurable?: Readonly<SubagentResolveConfigurable>;
}

/** Host contract for resolving a selected subagent's full graph inputs. */
export type SubagentAgentInputsResolver = (
  context: SubagentResolveContext
) => Promise<AgentInputs>;

interface SubagentConfigBase extends SubagentDescriptor {
  /** Max AGENT→TOOLS cycles before forced stop (default: 25). */
  maxTurns?: number;
  /** Allow this subagent to spawn its own subagents (default: false). */
  allowNested?: boolean;
}

export interface SubagentConfig extends SubagentConfigBase {
  /** Full agent config for the child graph. Omit when `self` is true. */
  agentInputs?: AgentInputs;
  /**
   * Resolve the full child config only after this descriptor is selected.
   * Eager `agentInputs` take precedence when also supplied. `self` and a
   * resolver are mutually exclusive and normalization rejects that pairing.
   *
   * Resolvers should rebuild inputs from the stable execution context instead
   * of retaining request-owned state. The SDK keeps resolved inputs in memory
   * only for the lifetime of the selected execution and supplies cancellation
   * through `context.signal`. A reconstructed execution may invoke the
   * resolver again, so the same `context.executionId` must produce equivalent
   * child inputs. Durable resolver work should be idempotent on
   * `(context.executionId, context.descriptor.configId)`.
   */
  resolveAgentInputs?: SubagentAgentInputsResolver;
  /** When true, reuse the parent's AgentInputs (context isolation without separate config). */
  self?: boolean;
}

export interface SingleAgentSubagentConfig extends SubagentConfig {
  kind?: 'agent';
  agents?: never;
  edges?: never;
  entryAgentId?: never;
  resultAgentId?: never;
}

export interface GraphSubagentConfig extends SubagentConfigBase {
  kind: 'graph';
  configId?: never;
  resolveAgentInputs?: never;
  allowNested?: false;
  agents: AgentInputs[];
  /**
   * Explicit direct DAG edges. Array-valued sources are all-of waiting edges.
   * Relative message order between parallel branches is implementation-dependent.
   * Prompted edges are supported in chains or on the final converged result transition.
   */
  edges: GraphSubagentEdge[];
  entryAgentId: string;
  resultAgentId: string;
  agentInputs?: never;
  self?: never;
}

/** Any configuration that can be spawned through the subagent tool. */
export type SubagentConfigEntry = SubagentConfig | GraphSubagentConfig;

/** Legacy single-agent config with self-spawn resolution completed. */
export interface ResolvedSubagentConfig extends SubagentConfig {
  agentInputs: AgentInputs;
}

/** Explicit alias for graph-aware code that needs to name the legacy variant. */
export interface ResolvedSingleAgentSubagentConfig
  extends ResolvedSubagentConfig {
  kind?: 'agent';
  agents?: never;
  edges?: never;
  entryAgentId?: never;
  resultAgentId?: never;
}

export type ResolvedSubagentConfigEntry =
  | ResolvedSubagentConfig
  | GraphSubagentConfig;

/** Lazy single-agent entry accepted after descriptor validation. */
export interface LazySingleAgentSubagentConfig
  extends SingleAgentSubagentConfig {
  configId: string;
  resolveAgentInputs: SubagentAgentInputsResolver;
  agentInputs?: never;
  self?: never;
}

/** Single-agent config accepted after eager/self/lazy eligibility checks. */
export type ExecutableSubagentConfig =
  | ResolvedSubagentConfig
  | LazySingleAgentSubagentConfig;

/** Graph-aware config accepted by the executor. Graph configs stay eager. */
export type ExecutableSubagentConfigEntry =
  | ExecutableSubagentConfig
  | GraphSubagentConfig;

/** Lifecycle phase carried on {@link SubagentUpdateEvent}. */
export type SubagentUpdatePhase =
  | 'start'
  | 'run_step'
  | 'run_step_delta'
  | 'run_step_completed'
  | 'run_step_closed'
  | 'message_delta'
  | 'reasoning_delta'
  | 'control'
  | 'stop'
  | 'error';

export interface SubagentAncestryEntry {
  readonly subagentRunId: string;
  readonly subagentType: string;
  readonly subagentKind: 'agent' | 'graph';
  /** Execution subject ID; synthetic for graph subagents. */
  readonly subagentAgentId: string;
  readonly parentRunId: string;
  readonly parentAgentId?: string;
  readonly parentToolCallId?: string;
}

export interface SubagentExecutionContext {
  readonly rootRunId: string;
  readonly hookSessionId: string;
  readonly depth: number;
  readonly ancestry: readonly SubagentAncestryEntry[];
}

/** Trusted identity supplied before a child can access host-owned resources. */
export interface SubagentContextInput {
  executionContext: SubagentExecutionContext;
  parentThreadId?: string;
  memberAgentIds: readonly string[];
  signal: AbortSignal;
  resumed: boolean;
}

export interface PreparedSubagentContext {
  /** Added to the initial child input; checkpoint recovery does not add them again. */
  messages?: BaseMessage[];
  /** Host configuration; SDK execution and checkpoint identities remain authoritative. */
  configurable?: RunnableConfig['configurable'];
  /** Each entry replaces that member's inherited code partition and initial sessions. */
  agentSessions?: Readonly<
    Record<string, Pick<AgentInputs, 'codeSessionKey' | 'initialSessions'>>
  >;
}

export interface SubagentContextResult {
  content: string;
  messages: BaseMessage[];
}

export interface SubagentContextAdapter {
  /** Reauthorizes reconstructed executions; durable work must be idempotent by child run ID. */
  prepare(input: SubagentContextInput): Promise<PreparedSubagentContext>;
  /** Projects durable references into the result returned to the parent. */
  complete?(
    input: SubagentContextInput,
    result: SubagentContextResult
  ): Promise<Pick<SubagentContextResult, 'content'>>;
}

/**
 * Wrapper event emitted when a subagent's child graph dispatches activity.
 * Lets hosts show subagent progress in a UI surface separate from the parent
 * conversation without having to untangle events by agent ID.
 */
export interface SubagentUpdateEvent {
  /** Root run ID that owns this execution tree. */
  runId: string;
  /** Immediate parent run that spawned this child. */
  parentRunId?: string;
  /** Child run ID (unique per subagent execution). */
  subagentRunId: string;
  /**
   * Parent-side `tool_call_id` for the `subagent` tool invocation that
   * triggered this run. Stable for the duration of the child; lets hosts
   * correlate updates deterministically instead of inferring by ordering.
   * Omitted when the executor was invoked outside of a tool-call context.
   */
  parentToolCallId?: string;
  /** Subagent `type` identifier from the SubagentConfig. */
  subagentType: string;
  /** Execution shape. Omitted by older emitters. */
  subagentKind?: 'agent' | 'graph';
  /** Execution subject ID; synthetic for graph subagents. */
  subagentAgentId: string;
  /** Graph member that produced this update, when attributable. */
  memberAgentId?: string;
  /** One-based nesting depth beneath the root graph. */
  depth?: number;
  /** Root-to-leaf execution lineage, without parsing composed run IDs. */
  ancestry?: readonly SubagentAncestryEntry[];
  /** Parent agent ID that spawned this subagent. */
  parentAgentId?: string;
  /** Lifecycle phase carried by this update. */
  phase: SubagentUpdatePhase;
  /** Underlying event payload (shape depends on phase). */
  data?: unknown;
  /** Short human-readable description. Hosts can render this directly. */
  label?: string;
  /** ISO timestamp for ordering / display. */
  timestamp: string;
}

/**
 * Token usage for a single model call made inside a subagent child run.
 * Emitted through {@link SubagentUsageSink} as each call completes, so
 * hosts can bill child-run model usage that never reaches the parent
 * run's `CHAT_MODEL_END` handler (child graphs execute via `invoke()`
 * outside the host's `streamEvents` loop).
 */
export interface SubagentUsageEvent {
  /** Usage metadata reported by the child's model call. */
  usage: UsageMetadata;
  /** Identity of this model call, shared with native failure recovery. */
  modelRunId?: string;
  /**
   * Model that produced this usage. Per-call `ls_model_name` from the
   * model's callback metadata when available (covers child-side
   * summarization or any call that differs from the configured model),
   * then the fallback-invocation's configured model (`INVOKED_MODEL`
   * metadata), then the subagent config's `clientOptions` model.
   */
  model?: string;
  /**
   * Provider that actually served this call — the SDK `Providers` enum
   * value stamped per-invocation by `attemptInvoke` (`INVOKED_PROVIDER`
   * metadata), so fallback-served calls are attributed to the fallback
   * provider, not the configured primary. Falls back to the subagent
   * config's provider. Never LangSmith's `ls_provider` string — derived
   * providers inherit that from their base class, and hosts key
   * pricing/cache semantics off the enum.
   */
  provider?: string;
  /** Subagent `type` identifier from the SubagentConfig. */
  subagentType: string;
  /** Execution shape. Omitted by older emitters. */
  subagentKind?: 'agent' | 'graph';
  /** Child run ID (unique per subagent execution). */
  subagentRunId: string;
  /** Execution subject ID; synthetic for graph subagents. */
  subagentAgentId: string;
  /** Graph member whose model call produced this usage. */
  memberAgentId?: string;
  /** Immediate parent run that spawned this child. */
  parentRunId?: string;
  /** One-based nesting depth beneath the root graph. */
  depth?: number;
  /** Root-to-leaf execution lineage, without parsing composed run IDs. */
  ancestry?: readonly SubagentAncestryEntry[];
  /**
   * ROOT run ID of the host run that owns billing. For nested subagents
   * each forwarding layer rewrites this upward, so events from any depth
   * surface with the outermost run's ID — never an intermediate
   * `*_sub_*` child id (use {@link subagentRunId} to identify the
   * emitting child).
   */
  runId: string;
}

/**
 * Host-provided callback receiving {@link SubagentUsageEvent}s. Invoked as
 * each child model call completes. May return a promise — the executor
 * awaits each dispatch (so all usage is recorded before the child's result
 * resolves to the parent) and swallows both synchronous throws and
 * rejections; implementations should still be cheap, as they sit on the
 * child's model-call path.
 */
export type SubagentUsageSink = (
  event: SubagentUsageEvent
) => void | Promise<void>;

export type LangfuseToolOutputTracingConfig = {
  /**
   * Whether tool outputs should be exported to Langfuse. Defaults to
   * `true`. Set to `false` to keep tool spans and redact their output.
   */
  enabled?: boolean;
  /**
   * Optional allowlist of tool names whose outputs should be redacted even
   * when `enabled` is true.
   */
  redactedToolNames?: string[];
  /**
   * Match strategy for `redactedToolNames`. Defaults to `exact`; use
   * `partial` to redact tools whose names contain a configured value.
   */
  redactedToolNameMatchMode?: 'exact' | 'partial';
  /** Replacement text used for redacted tool outputs. */
  redactionText?: string;
};

export type LangfuseToolNodeTracingConfig = {
  /**
   * Opts into the internal ToolNode batch observation. Graph tool-dispatch
   * and individual tool observations are exported without this wrapper, so
   * the default is false to avoid a redundant hierarchy level.
   */
  enabled?: boolean;
};

export interface LangfuseConfig {
  enabled?: boolean;
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
  /**
   * Environment identifier attached to exported traces (Langfuse
   * `environment`). When unset, falls back to `LANGFUSE_TRACING_ENVIRONMENT`
   * then `NODE_ENV`, so production traces are not collapsed under the
   * `default` environment.
   */
  environment?: string;
  /**
   * Whether the Langfuse span processor should detect and upload base64 media
   * payloads. Defaults to the Langfuse SDK behavior.
   */
  mediaUploadEnabled?: boolean;
  /**
   * Extra HTTP headers sent on every Langfuse export request (OTLP trace
   * export and media uploads), for self-hosted instances behind an
   * authenticating proxy or gateway.
   *
   * Deployment-level only: one exporter is shared by every span routed to a
   * destination, so these cannot carry per-request or per-user identity.
   * Headers participate in destination identity — see
   * `getLangfuseDestinationKey`.
   */
  additionalHeaders?: Record<string, string>;
  metadata?: Record<string, string | number | boolean | null | undefined>;
  /**
   * User identity stamped on every trace this run emits — the agent stream,
   * titles, activity and reasoning labels, and phases. Hosts set it when the
   * observability identity differs from `configurable.user_id` (an email or
   * IdP subject instead of an internal database id); that id remains the
   * fallback when unset or blank.
   */
  userId?: string;
  /**
   * Internal OTLP span attributes to attach to Langfuse observations before
   * export. Intended for collector-side routing/filtering; strip these in the
   * collector before forwarding spans to Langfuse.
   */
  librechatTraceAttributes?: Record<
    string,
    string | number | boolean | null | undefined
  >;
  tags?: string[];
  toolNodeTracing?: LangfuseToolNodeTracingConfig;
  toolOutputTracing?: LangfuseToolOutputTracingConfig;
  /**
   * When true, derive the run's root Langfuse trace id deterministically from
   * its `runId` (`sha256(runId)` → 32 hex chars, matching `@langfuse/tracing`
   * `createTraceId`) instead of a random id. This lets external systems attach
   * scores or observations to the trace afterwards by regenerating the same id
   * from the run/message id, without a trace lookup. Default: random ids.
   */
  deterministicTraceId?: boolean;
}

interface AgentInputFields {
  agentId: string;
  /** Logical endpoint selected by the host before provider resolution. */
  endpoint?: string;
  /**
   * Partition key for transient code-session ids and file refs. Agents with
   * the same key share those refs; different execution profiles/scopes must
   * use different keys. Defaults to the legacy shared `execute_code` slot.
   * Non-default partitions also disable speculative eager execution of code
   * tools because their factory may target a durable backend.
   */
  codeSessionKey?: string;
  /** Human-readable name for the agent (used in handoff context). Defaults to agentId if not provided. */
  name?: string;
  toolEnd?: boolean;
  toolMap?: ToolMap;
  tools?: GraphTools;
  /** Stable/cacheable system instructions. */
  instructions?: string;
  streamBuffer?: number;
  maxContextTokens?: number;
  /** Per-agent Langfuse tracing configuration. */
  langfuse?: LangfuseConfig;
  /** Dynamic system tail appended after stable instructions without provider cache markers. */
  additional_instructions?: string;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  /** Format content blocks as strings (for legacy compatibility i.e. Ollama/Azure Serverless) */
  useLegacyContent?: boolean;
  /**
   * Tool definitions for all tools, including deferred and programmatic.
   * Used for tool search and programmatic tool calling.
   * Maps tool name to LCTool definition.
   */
  toolRegistry?: Map<string, LCTool>;
  /**
   * Serializable tool definitions for event-driven execution.
   * When provided, ToolNode operates in event-driven mode, dispatching
   * ON_TOOL_EXECUTE events instead of invoking tools directly.
   */
  toolDefinitions?: LCTool[];
  /**
   * Tool names discovered from previous conversation history.
   * These tools will be pre-marked as discovered so they're included
   * in tool binding without requiring tool_search.
   */
  discoveredTools?: string[];
  summarizationEnabled?: boolean;
  /**
   * Runs this agent as a summarize-only pass: the first model step requests
   * summarization outright instead of consulting the trigger, and the step
   * after the summary emits its context snapshot and ends the run without a
   * model call. Hosts use it for user-initiated compaction, where the summary
   * is the whole response. Requires `summarizationEnabled`: a run that cannot
   * attempt the summary rejects with `ManualSummarizationSkippedError`
   * rather than ending with nothing, and one whose provider calls all fail
   * keeps the history and reports the failure on the summary step. A
   * multi-agent workflow compiles to the agent that opted in alone; no other
   * agent runs.
   */
  summarizeOnly?: boolean;
  summarizationConfig?: SummarizationConfig;
  /**
   * Optional host-supplied, user-visible guidance for compaction. The SDK
   * validates, bounds, and scopes entries to the messages being compacted;
   * raw conversation messages remain authoritative. Captured when the
   * AgentContext is constructed; labels committed later in the same run are
   * outside this construction-time interface.
   */
  compactionSemanticIndex?: CompactionSemanticIndex;
  /** Cross-run summary from a previous run, forwarded from formatAgentMessages.
   *  Injected into the dynamic system tail via AgentContext. */
  initialSummary?: { text: string; tokenCount: number };
  contextPruningConfig?: ContextPruningConfig;
  maxToolResultChars?: number;
  /** Pre-computed tool schema token count (from cache). Skips recalculation when provided. */
  toolSchemaTokens?: number;
  /** Subagent configurations for hierarchical delegation. Each defines a child agent type. */
  subagentConfigs?: SubagentConfigEntry[];
  /** Maximum subagent nesting depth. Default 1 means top-level agents can spawn subagents but subagents cannot nest further. */
  maxSubagentDepth?: number;
  /**
   * Initial tool-session state owned by this agent when it runs as a
   * subagent. The executor copies these sessions into the isolated child
   * graph before its first invocation. Top-level runs continue to use
   * `RunConfig.initialSessions`.
   */
  initialSessions?: ToolSessionMap;
  /**
   * Host-supplied tool instances that must execute IN-PROCESS inside the graph's
   * ToolNode even when the run is event-driven (`toolDefinitions` non-empty). Each
   * instance is bound to the model alongside the schema-only event tools and its
   * name is marked direct, so calls bypass ON_TOOL_EXECUTE dispatch and run inside
   * the Pregel task frame. This is the only execution mode where a tool body may
   * raise a LangGraph `interrupt()` (e.g. a tool built on `askUserQuestion()`) —
   * the host-side event handler runs outside the graph task, where `interrupt()`
   * throws. Do NOT also list these tools in `toolDefinitions` (they would be bound
   * twice). NOT inherited by SELF-SPAWNED subagent children: `buildChildInputs`
   * scrubs the shallow-spread parent copy so parent-scoped direct tools are not
   * exposed to a child implicitly. An EXPLICIT child config that lists its own
   * `graphTools` keeps them; with HITL enabled, those tools share the parent's
   * checkpointer and may pause and resume inside the child graph.
   *
   * Deliberately `GenericTool[]`, not `GraphTools`: the wider union admits
   * schema-only shapes (OpenAI `BindToolsInput`, Google tool objects) that
   * `initializeTools` cannot register in the ToolNode direct map — the model
   * would bind a tool the SDK advertised as in-process but cannot execute.
   * Every entry must be a real executable tool instance with a `name`.
   */
  graphTools?: GenericTool[];
}

export type AgentInputs = AgentInputFields & ProviderClientOptionsConfig;

export interface ContextPruningConfig {
  enabled?: boolean;
  keepLastAssistants?: number;
  softTrimRatio?: number;
  hardClearRatio?: number;
  minPrunableToolChars?: number;
  softTrim?: {
    maxChars?: number;
    headChars?: number;
    tailChars?: number;
  };
  hardClear?: {
    enabled?: boolean;
    placeholder?: string;
  };
}

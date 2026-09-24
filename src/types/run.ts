// src/types/run.ts
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredTool } from '@langchain/core/tools';
import type * as z from 'zod';
import type {
  ToolSessionMap,
  ToolExecutionConfig,
  ToolOutputReferencesConfig,
  EagerEventToolExecutionConfig,
} from '@/types/tools';
import type { SubagentTaskConfig } from '@/types/subagentTasks';
import type { HumanInTheLoopConfig } from '@/types/hitl';
import type { HookRegistry } from '@/hooks';
import type * as s from '@/types/stream';
import type * as e from '@/common/enum';
import type * as g from '@/types/graph';
import type * as l from '@/types/llm';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ZodObjectAny = z.ZodObject<any, any, any, any>;
export type BaseGraphConfig = {
  llmConfig: l.LLMConfig;
  provider?: l.ProviderName;
  clientOptions?: l.ClientOptions;
  /** Optional compile options for workflow.compile() */
  compileOptions?: g.CompileOptions;
};
export type LegacyGraphConfig = BaseGraphConfig & {
  type?: 'standard';
} & Omit<
    g.StandardGraphInput,
    /** `streamLimits` is excluded because legacy graphs receive limits only
     * via the top-level `RunConfig.streamLimits`; accepting the field here
     * would type-check but be silently ignored by `createLegacyGraph`. */
    'provider' | 'clientOptions' | 'agents' | 'streamLimits'
  > &
  Omit<g.AgentInputs, 'provider' | 'clientOptions' | 'agentId'>;

/* Supervised graph (opt-in) */
export type SupervisedGraphConfig = BaseGraphConfig & {
  type: 'supervised';
  /** Enable supervised router; when false, fall back to standard loop */
  routerEnabled?: boolean;
  /** Table-driven routing policy per stage */
  routingPolicies?: Array<{
    stage: string;
    agents?: string[];
    model?: l.ProviderName;
    parallel?: boolean;
    /** Optional simple condition on content/tools */
    when?:
      | 'always'
      | 'has_tools'
      | 'no_tools'
      | { includes?: string[]; excludes?: string[] };
  }>;
  /** Opt-in feature flags */
  featureFlags?: {
    multi_model_routing?: boolean;
    fan_out?: boolean;
    fan_out_retries?: number;
    fan_out_backoff_ms?: number;
    fan_out_concurrency?: number;
  };
  /** Optional per-stage model configs */
  models?: Record<string, l.LLMConfig>;
} & Omit<g.StandardGraphInput, 'provider' | 'clientOptions'>;

export type RunTitleOptions = {
  inputText: string;
  provider: l.ProviderName;
  contentParts: (s.MessageContentComplex | undefined)[];
  titlePrompt?: string;
  skipLanguage?: boolean;
  clientOptions?: l.ClientOptions;
  chainOptions?: Partial<RunnableConfig> | undefined;
  omitOptions?: Set<string>;
  titleMethod?: e.TitleMethod;
  titlePromptTemplate?: string;
};

export interface AgentStateChannels {
  messages: BaseMessage[];
  next: string;
  [key: string]: unknown;
  /** Stable/cacheable system instructions. */
  instructions?: string;
  /** Dynamic system tail appended after stable instructions. */
  additional_instructions?: string;
}

export interface Member {
  name: string;
  systemPrompt: string;
  tools: StructuredTool[];
  llmConfig: l.LLMConfig;
}

export type CollaborativeGraphConfig = {
  type: 'collaborative';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type TaskManagerGraphConfig = {
  type: 'taskmanager';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type MultiAgentGraphConfig = {
  type: 'multi-agent';
  compileOptions?: g.CompileOptions;
  agents: g.AgentInputs[];
  edges: g.GraphEdge[];
  entryAgentId?: string;
  maxHandoffs?: number;
};

export type StandardGraphConfig = Omit<
  MultiAgentGraphConfig,
  'edges' | 'type' | 'entryAgentId' | 'maxHandoffs'
> & { type?: 'standard'; signal?: AbortSignal };

/**
 * Cooperative mid-generation preemption. Lets a host end the live model stream
 * early and inject whatever it queued, in one of two ways:
 *
 *  - a SEAL, at the next provider-safe token boundary: the run is never
 *    aborted, the partial assistant turn is kept, and the graph self-loops into
 *    a fresh model call once the `PreemptBoundary` hook has injected.
 *  - a RESTART, when the turn holds nothing worth keeping — no text, no tool
 *    call, at most reasoning. The provider request IS torn down, its output is
 *    discarded, and the model is called again with the injection appended to
 *    the same prompt. Requires `subscribe`; see it for why the per-chunk poll
 *    alone cannot reach this case.
 *
 * Preconditions the host MUST satisfy:
 *   - `shouldPreempt` is polled once per streamed chunk on the top-level
 *     graph, and once more per model attempt if `subscribe` is supplied — at
 *     the attempt's start and on each wake. It must be synchronous,
 *     allocation-free and O(1) — never I/O.
 *     It must also be LEVEL-TRIGGERED (non-consuming): the SDK never clears
 *     the host's request, and a true result is only honored once the
 *     accumulated chunk is provider-safe, so the predicate may be polled
 *     many times before a seal. A one-shot read that clears its own pending
 *     flag would silently lose the request on an unsafe chunk (leading
 *     whitespace/reasoning, an in-flight tool call) — keep returning true
 *     until the `PreemptBoundary` drain hands over the queued injection,
 *     then disarm there.
 *   - Sealing is only honored on the SDK's own dispatch loop. A run whose
 *     registered `CHAT_MODEL_STREAM` handler IS the SDK dispatcher — or wraps
 *     it, which `composeEventHandlers` and `createRunHandlers` both do —
 *     consumes chunks through a decoupled `streamEvents` reader that can lag
 *     the accumulated chunk, so those runs never seal.
 *
 *     Detection is by capability, not identity: the dispatcher carries
 *     `SDK_STREAM_DISPATCH` and the SDK's wrappers propagate it. A handler
 *     that merely OBSERVES the raw chunk echo — LibreChat's no-op
 *     `OpenAIChatModelStreamHandler`, say — is unbranded and does NOT disable
 *     sealing, because it assigns no content-part indices and so cannot be
 *     inverted. A host that renders from the raw feed and wants the opt-out
 *     should brand its handler with `SDK_STREAM_DISPATCH`.
 *   - `RunConfig.tokenCounter` should be set. A sealed turn ends before most
 *     providers send their usage chunk, so the synthetic `CHAT_MODEL_END`
 *     falls back to the counter to report `output_tokens`. Without one that
 *     fallback silently no-ops and the sealed turn's usage is lost — verified
 *     live: OpenAI, Azure OpenAI and DeepSeek report no usage for the sealed
 *     segment without a counter, while Anthropic streams usage incrementally
 *     and reports it either way.
 */
export interface StreamPreemption {
  /**
   * Polled once per streamed chunk. Synchronous, allocation-free, O(1), and
   * level-triggered — keep returning true until the `PreemptBoundary` drain
   * consumes the request; a self-clearing read loses it on an unsafe chunk.
   */
  shouldPreempt: () => boolean;
  /**
   * Optional wake-up channel for the window where the poll cannot reach: a
   * provider that has sent nothing yet, or that is streaming reasoning the
   * seal gate will never accept. `shouldPreempt` is only read per chunk, so a
   * request armed during a long silent stretch would otherwise wait for the
   * whole turn.
   *
   * The SDK subscribes once per model attempt and calls the returned
   * unsubscribe when the attempt ends. `wake` is a HINT, never the request
   * itself: it wakes the SDK, which then re-reads `shouldPreempt` as the sole
   * authority. That keeps the level-triggered contract above intact and makes
   * spurious wakes free — a host may call it on every arm without tracking
   * which ones the SDK already saw.
   *
   * A wake is only honored when the accumulated turn holds nothing worth
   * keeping, and it is honored by DISCARDING that turn: the in-flight provider
   * request is torn down, its partial output is dropped, the `PreemptBoundary`
   * hook injects, and the model is called again with the injection appended.
   * Reasoning tokens already spent are billed and lost, which is the trade the
   * host is asking for. A turn that has produced visible text seals instead,
   * through the ordinary per-chunk path, and a wake never discards it.
   */
  subscribe?: (wake: () => void) => () => void;
  /**
   * How long a request waits for the turn to become sealable before it may
   * discard it instead. Defaults to `DEFAULT_PREEMPT_RESTART_GRACE_MS`; `0`
   * converts as soon as the shape allows.
   *
   * The window keeps a restart from stealing a seal that was moments away, and
   * from discarding a first chunk still in flight between the provider and the
   * consumer. Both are why it applies to a silent turn as well as a thinking
   * one.
   */
  restartGraceMs?: number;
  /**
   * Max cooperative seals per run. Each seal costs one extra superstep, so
   * this also bounds the recursion-limit headroom the run reserves.
   */
  maxSeals?: number;
}

/**
 * Per run: seals honored, discards honored, and boundaries that had nothing to
 * inject.
 *
 * `seals` counts only turns that were KEPT — a partial assistant message
 * survived into the next prompt. `restarts` counts turns that were discarded
 * and reissued, which preserve nothing, so a consumer classifying a run as
 * "sealed" must not read them together.
 */
export type PreemptStats = {
  seals: number;
  restarts: number;
  emptyBoundaries: number;
};

/**
 * Circuit breakers for pathological model streams. A malformed generation
 * can stream a single tool call's arguments for many minutes while they
 * never become executable (observed live: one 149,923-char SQL argument
 * streamed for 26 minutes before the 64k output-token ceiling ended the
 * run). When a limit trips, the run aborts with a `StreamLimitExceededError`
 * and the in-flight provider request is torn down mid-stream.
 */
export interface StreamLimits {
  /**
   * Max cumulative UTF-8 bytes a single streamed tool call's arguments may
   * reach before the run is aborted. Defaults to
   * `DEFAULT_MAX_TOOL_CALL_ARG_BYTES` (64 KiB); `0` disables the guard.
   */
  maxToolCallArgBytes?: number;
  /**
   * Per-tool overrides for `maxToolCallArgBytes`, keyed by the model-facing
   * tool name. A matching entry replaces the global cap for that tool's
   * calls (`0` disables the guard for that tool only). Tools that
   * legitimately stream whole documents as arguments (e.g. file creation)
   * can run with a higher cap without loosening every other tool's guard.
   * Calls whose tool name has not yet arrived on the stream use the global
   * cap; providers send the name in the first chunk, so this only matters
   * for malformed streams.
   */
  maxToolCallArgBytesByTool?: Record<string, number>;
  /**
   * Max streamed chunk events a single model generation (turn) may emit
   * before the run is aborted. Defense in depth against looping or
   * duplicated provider streams that a byte limit cannot see (e.g. endless
   * empty chunks). Disabled by default; `0` also disables.
   */
  maxDeltaEventsPerTurn?: number;
}

export type RunConfig = {
  runId: string;
  graphConfig: LegacyGraphConfig | StandardGraphConfig | MultiAgentGraphConfig;
  /**
   * Run-scoped Langfuse configuration. Per-agent `AgentInputs.langfuse`
   * takes precedence for agent-specific spans; this object supplies defaults
   * for run-wide tracing controls such as tool-output redaction.
   */
  langfuse?: g.LangfuseConfig;
  customHandlers?: Record<string, g.EventHandler>;
  /** Explicit client-owned tools for a single-agent graph. A batch mixing
   * client and SDK/provider calls fails closed; omitted means SDK ownership.
   * Hosts must register the corresponding model-facing tool schemas.
   */
  clientDelegatedToolNames?: readonly string[];
  /**
   * Receives token usage for every model call made inside subagent child
   * runs (including nested subagents). Child graphs execute via `invoke()`
   * outside this run's `streamEvents` loop, so their model-end events never
   * reach `customHandlers` — without this sink, child usage is invisible to
   * the host. Parent-graph calls are not reported here; they flow through
   * the registered `CHAT_MODEL_END` handler as usual.
   */
  subagentUsageSink?: g.SubagentUsageSink;
  /**
   * Trusted process-local task namespace for detached subagent execution.
   * Omit to preserve foreground-only subagent behavior.
   */
  subagentTasks?: SubagentTaskConfig;
  /** Authorizes and supplies host-owned context to isolated child runs. */
  subagentContext?: g.SubagentContextAdapter;
  /**
   * Pre-constructed hook registry for this run. Hooks fire at lifecycle
   * points in `processStream` (RunStart, UserPromptSubmit, Stop,
   * StopFailure) and around tool calls (PreToolUse, PostToolUse,
   * PostToolUseFailure, PermissionDenied).
   *
   * Pass `undefined` (the default) to skip all hook dispatch. When a
   * registry is provided, the run attaches it to the `Graph` so internal
   * nodes can fire hooks too, and clears the session in the `finally`
   * block to prevent leaks.
   */
  hooks?: HookRegistry;
  /**
   * Maximum number of times a `Stop` hook may keep this Run warm by returning
   * `decision: 'block'` with messages to inject. Defaults to
   * `DEFAULT_MAX_STOP_CONTINUATIONS`; non-finite values use the default and
   * values at or below zero disable terminal continuation while preserving the
   * final Stop notification.
   */
  maxStopContinuations?: number;
  /**
   * Opt-in cooperative preemption for this run. Requires a `hooks` registry
   * with a `PreemptBoundary` matcher — the seal only stops the stream, the
   * hook is what supplies the messages to resume with. Omit to keep the
   * pre-preemption behavior, where a mid-run injection can only land at a
   * tool boundary.
   */
  preemption?: StreamPreemption;
  /**
   * Circuit breakers for pathological model streams (runaway tool-call
   * argument generation, looping delta streams). Omit for defaults: the
   * tool-call argument byte cap is ON by default, the per-turn event cap is
   * opt-in. See {@link StreamLimits}.
   */
  streamLimits?: StreamLimits;
  returnContent?: boolean;
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  /**
   * Calibration ratio from a previous run's contextMeta.
   * Seeds the pruner's EMA so new messages are scaled immediately.
   *
   * Hosts should persist the value returned by `Run.getCalibrationRatio()`
   * after each run and pass it back here on subsequent runs for the same
   * conversation. Without this, the EMA resets to 1 on every new Run instance.
   */
  calibrationRatio?: number;
  /**
   * Default-agent fading tier retained for single-agent compatibility. New
   * multi-agent integrations should use `fadingTiers`.
   */
  fadingTier?: g.FadingTier | null;
  /**
   * Context-fading tiers keyed by agent ID. Multi-agent hosts should persist
   * the value returned by `Run.getFadingTiers()` and pass it back here.
   */
  fadingTiers?: g.FadingTiers | null;
  /** Skip post-stream cleanup (clearHeavyState) — useful for tests that inspect graph state after processStream */
  skipCleanup?: boolean;
  /**
   * Initial session state to seed the Graph's ToolSessionMap.
   * Used to carry over code environment sessions from skill file priming
   * at run start, so ToolNode can inject session_id + files into tool calls.
   */
  initialSessions?: ToolSessionMap;
  /**
   * Run-scoped tool output reference configuration. When `enabled` is
   * `true`, tool outputs are registered under stable keys
   * (`tool<idx>turn<turn>`) and subsequent tool calls can pipe previous
   * outputs into their arguments via `{{tool<idx>turn<turn>}}`
   * placeholders. Disabled by default so existing runs are unaffected.
   */
  toolOutputReferences?: ToolOutputReferencesConfig;
  /**
   * Opt-in latency optimization for event-driven tools. When enabled,
   * the streaming layer may start a tool call as soon as it sees a
   * complete, parseable tool call; ToolNode still waits for the final
   * assistant message before appending ToolMessages, preserving provider
   * ordering. The SDK automatically falls back to normal batch dispatch
   * when hooks, HITL, output references, or ambiguous stream shapes make
   * eager dispatch unsafe.
   */
  eagerEventToolExecution?: EagerEventToolExecutionConfig;
  /**
   * Names of host tools that write to the code-execution sandbox but are not
   * built-in `CODE_EXECUTION_TOOLS` (e.g. LibreChat's create_file/edit_file).
   * Their successful results fold the returned exec `session_id` into the
   * shared code session, so a file such a tool writes is visible to later
   * bash_tool/execute_code calls running in the same sandbox. Host-declared so
   * the SDK stays name-agnostic.
   */
  codeSessionToolNames?: string[];
  /**
   * Names of host tools whose in-process body may raise a LangGraph
   * `interrupt()` mid-execution — the canonical example is an
   * `ask_user_question` tool that suspends the run to collect a human
   * answer. Within a single tool-call batch, a named tool that is a real
   * in-process graphTool (the only kind whose body can reach `interrupt()`;
   * graphTools are auto-marked direct) is scheduled **ahead of** its
   * non-interrupting direct siblings. That ordering guarantees a mid-body
   * interrupt unwinds the tool batch before a non-idempotent sibling
   * (send_email, billing) executes, so the sibling cannot run once on the
   * first pass and AGAIN when LangGraph re-runs the interrupted batch on
   * resume.
   *
   * This only reorders the direct group — it does NOT force a name onto
   * the direct path. A name that is only an inherited event `toolDefinition`
   * (schema-only stub, e.g. in a self-spawned child) stays event-dispatched;
   * the guard applies only to tools that are independently direct.
   * Host-declared so the SDK stays name-agnostic; omit to keep the prior
   * (unguarded) behavior.
   */
  interruptingToolNames?: string[];
  /**
   * Selects the execution backend for built-in code tools. Omit this to keep
   * the remote LibreChat Code API sandbox. Set `{ engine: 'local' }` to run
   * code execution locally and auto-bind the local coding tool suite unless
   * `local.includeCodingTools` is set to `false`.
   */
  toolExecution?: ToolExecutionConfig;
  /**
   * First-class human-in-the-loop (HITL) flow for this run.
   *
   * **HITL is OFF by default.** Omitting this field — or passing
   * `{ enabled: false }` — keeps the pre-HITL fail-closed semantics
   * where `ask` decisions collapse into a synchronous deny. Hosts opt
   * in explicitly with `{ enabled: true }` once their UI can render
   * and resolve `tool_approval` interrupts (otherwise the run just
   * pauses with no resolver, which surfaces to end users as a hung
   * tool-call card).
   *
   * Plan of record: the default flips back to ON in a future minor
   * once the consumer ecosystem (notably LibreChat) ships HITL UI
   * end-to-end. See `HumanInTheLoopConfig` JSDoc.
   *
   * When enabled (`{ enabled: true }`):
   *   - `PreToolUse` hooks returning `decision: 'ask'` raise a real
   *     LangGraph `interrupt()` instead of being treated as a synchronous
   *     deny. The graph pauses and the run exits cleanly.
   *   - If `graphConfig.compileOptions.checkpointer` is missing, the SDK
   *     installs an in-memory `MemorySaver` as a fallback so scripts and
   *     tests can resume without external infrastructure. Production
   *     hosts should always provide a durable checkpointer.
   *   - Hosts inspect the pending interrupt via `run.getInterrupt()` and
   *     continue with `Run.resume(decisions)` against a Run rebuilt with
   *     the same `thread_id` and checkpointer.
   *
   * When disabled (the default): `ask` decisions remain fail-closed
   * (blocked with an error `ToolMessage`) and no checkpointer is
   * implicitly attached.
   */
  humanInTheLoop?: HumanInTheLoopConfig;
};

export type ProvidedCallbacks = Callbacks | undefined;

export type TokenCounter = (message: BaseMessage) => number;

/** Structured breakdown of how context token budget is consumed. */
export type TokenBudgetBreakdown = {
  /** Total context window budget (maxContextTokens). */
  maxContextTokens: number;
  /** Total instruction tokens (system + tools + summary). */
  instructionTokens: number;
  /** Tokens from the system message text alone. */
  systemMessageTokens: number;
  /** Tokens from instruction text emitted outside the system message. */
  dynamicInstructionTokens: number;
  /** Tokens from tool schema definitions. */
  toolSchemaTokens: number;
  /** Tokens from the conversation summary. */
  summaryTokens: number;
  /** Number of registered tools. */
  toolCount: number;
  /** Number of messages in the conversation. */
  messageCount: number;
  /** Total tokens consumed by messages (excluding system). */
  messageTokens: number;
  /** Tokens available for messages after instructions. */
  availableForMessages: number;
  /** Per-tool schema token counts (post-multiplier), keyed by tool name. */
  toolTokenCounts?: Record<string, number>;
  /** Names of counted tools that are deferred (`defer_loading`) and discovered. */
  deferredToolNames?: string[];
  /** Calibrated retained tool traffic: tool results plus assistant turns made only of tool calls, inline provider tool results and reasoning. A subset of messageTokens, not an additional budget category. */
  toolMessageTokens?: number;
  /** Per-tool result-message share; excludes assistant invocation overhead. */
  toolMessageTokenCounts?: Record<string, number>;
};

export type EventStreamOptions = {
  callbacks?: g.ClientCallbacks;
  keepContent?: boolean;
};

/**
 * When to persist checkpoints during a run. Mirrors langgraph's option:
 * `'async'`/`'sync'` checkpoint every superstep; `'exit'` checkpoints only
 * at the graph's exit/interrupt boundary. Kept as a local union to avoid
 * coupling to langgraph internals (it is not exported from the root).
 */
export type Durability = 'async' | 'sync' | 'exit';

/**
 * Config accepted by `processStream`/`resume`. Extends `RunnableConfig` with
 * the stream `version`, an optional `run_id`, and an optional `durability`
 * override forwarded to langgraph's run options.
 */
export type RunStreamConfig = Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  run_id?: string;
  durability?: Durability;
};

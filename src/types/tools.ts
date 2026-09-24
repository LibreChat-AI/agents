// src/types/tools.ts
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  MessageContentComplex,
  RunStepResumeState,
  ToolErrorData,
} from './stream';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { LangfuseConfig, SubagentExecutionContext } from './graph';
import type { PreparedSubagents } from '@/tools/preparedSubagents';
import type { RunBreakerScope } from '@/llm/streamLimits';
import type { HandoffRouting } from '@/graphs/handoff';
import type { HumanInTheLoopConfig } from './hitl';
import type { HookRegistry } from '@/hooks';

/** Replacement type for `import type { ToolCall } from '@langchain/core/messages/tool'` in order to have stringified args typed */
export type CustomToolCall = {
  name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args: string | Record<string, any>;
  id?: string;
  type?: 'tool_call';
  output?: string;
};

export type GenericTool = (StructuredToolInterface | RunnableToolLike) & {
  mcp?: boolean;
};

export type ToolMap = Map<string, GenericTool>;
export type ToolRefs = {
  tools: GenericTool[];
  toolMap?: ToolMap;
};

export type ToolRefGenerator = (tool_calls: ToolCall[]) => ToolRefs;

export type EagerEventToolExecutionConfig = {
  /**
   * When enabled, event-driven tool calls may be started from the
   * streaming layer as soon as the SDK observes a complete tool call.
   * Results are still held until ToolNode materializes the final
   * ToolMessages so provider message ordering is preserved.
   */
  enabled?: boolean;
  /**
   * Tool names that must never be started eagerly. Eager execution
   * speculates on tool args before the model turn commits, so
   * side-effecting tools (e.g. file writes) should opt out: a
   * speculative write can land even if the turn is superseded, and a
   * later arg revision would otherwise trip the "changed after eager
   * execution" guard. Excluded calls fall through to normal ToolNode
   * execution with final args.
   */
  excludeToolNames?: string[];
  /**
   * Maximum retained early foreground subagent invocations per graph. Defaults
   * to 4; 0 disables them. Checkpoints, parent tool hooks, background calls and
   * control-flow graph tools retain normal batch execution. Once a child has
   * started, a failed/revised model attempt fails instead of automatically retrying.
   */
  maxPendingSubagents?: number;
};

export type EagerEventToolExecutionOutcome =
  | { results: ToolExecuteResult[]; error?: undefined }
  | { results?: undefined; error: Error };

export type EagerEventToolExecution = {
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
  request: ToolCallRequest;
  codeSessionBaselineByName?: ReadonlyMap<string, string>;
  promise: Promise<EagerEventToolExecutionOutcome>;
  /**
   * True when the streaming eager path already emitted the user-visible
   * ON_RUN_STEP_COMPLETED event for this call. ToolNode still consumes the
   * result later to mutate graph state in provider-safe order, but skips
   * duplicate completion emission.
   */
  completionDispatched?: boolean;
};

export type EagerEventToolCallChunkState = {
  id?: string;
  name?: string;
  argsText: string;
  index?: number;
  lastArgsFragment?: string;
  /**
   * Cumulative length of every observed args fragment — the length of the
   * plain in-order concatenation that LangChain's `AIMessageChunk.concat`
   * performs to build the final tool call. Every reconciliation branch in
   * `mergeToolCallArgsText` (and the repeat-fragment dedupe) produces text no
   * longer than plain concatenation, with equality exactly when every merge
   * was a pure append — so `argsText.length === rawArgsLength` proves
   * `argsText` IS the canonical accumulation the final request will carry.
   * Tracking only the length keeps cumulative/restating streams from
   * retaining every prefix (quadratic growth) while still letting seal-time
   * prestart verify the snapshot.
   */
  rawArgsLength?: number;
  /**
   * The non-empty args fragment carried by a chunk whose explicit adapter
   * seal covered this call. Some adapters restate the finished call's full
   * args on the seal chunk (OpenAI Responses `function_call_arguments.done`);
   * only such a restatement may override the canonical-accumulation check at
   * seal time. Pure-signal seals (Bedrock `contentBlockStop`, `args: ''`)
   * never set this.
   */
  sealedArgsFragment?: string;
};

export type ToolNodeOptions = {
  /** @internal Admission shared by tool nodes belonging to one multi-agent graph. */
  handoffRouting?: Pick<HandoffRouting, 'finalize'>;
  name?: string;
  tags?: string[];
  /** Enables LangChain/LangGraph tracing for this ToolNode. Defaults to false. */
  trace?: boolean;
  /** Run-level Langfuse config used to scope ToolNode trace redaction. */
  runLangfuse?: LangfuseConfig;
  /** Agent-level Langfuse config used to scope ToolNode trace redaction. */
  agentLangfuse?: LangfuseConfig;
  handleToolErrors?: boolean;
  loadRuntimeTools?: ToolRefGenerator;
  toolCallStepIds?: Map<string, string>;
  /**
   * Dispatches the error completion event for a failed tool call. Returns
   * whether the event was actually dispatched — `false` (e.g. no run step is
   * registered for the call yet, which happens when a tool fails fast on a
   * resume pass) tells the ToolNode to fall back to its own completion
   * dispatch for the error ToolMessage. A `void` resolution is treated as
   * dispatched for backward compatibility.
   */
  errorHandler?: (
    data: ToolErrorData,
    metadata?: Record<string, unknown>
  ) => Promise<boolean | void>;
  /** Tool registry for lazy computation of programmatic tools and tool search */
  toolRegistry?: LCToolRegistry;
  /** Returns the owning agent's currently discovered deferred tool names. */
  getDiscoveredToolNames?: () => readonly string[];
  /** Reference to Graph's sessions map for automatic session injection */
  sessions?: ToolSessionMap;
  /** Partition within `sessions` used by this agent's code tools. */
  codeSessionKey?: string;
  /** When true, dispatches ON_TOOL_EXECUTE events instead of invoking tools directly */
  eventDrivenMode?: boolean;
  /** Tool definitions for event-driven mode (used for context, not invocation) */
  toolDefinitions?: Map<string, LCTool>;
  /** Agent ID for event-driven mode (used to identify which agent's context to use) */
  agentId?: string;
  /** ID of the agent that owns this tool node, surfaced to hooks as `executingAgentId`
   * so a batch can be attributed to a specific agent even where `agentId` is undefined. */
  executingAgentId?: string;
  /** SDK-owned identity of the child execution, independent of reusable agent IDs. */
  executionContext?: SubagentExecutionContext;
  /** Name of the agent that owns this tool node, used for active Langfuse attribution. */
  executingAgentName?: string;
  /** Root graph-agent identity retained alongside the currently executing tool owner. */
  rootAgentId?: string;
  rootAgentName?: string;
  /** Tool names that must be executed directly (via runTool) even in event-driven mode (e.g., graph-managed handoff tools) */
  directToolNames?: Set<string>;
  /**
   * Tool names whose in-process body may raise a LangGraph `interrupt()`
   * mid-execution — e.g. an `ask_user_question` tool that suspends the
   * run to collect a human answer.
   *
   * Within a single tool-call batch, a named tool that is *already*
   * direct (a real in-process graphTool — the only kind whose body can
   * reach `interrupt()`; graphTools are auto-marked direct by the graph)
   * is executed as its own awaited group **before** its non-interrupting
   * direct siblings. If it interrupts, the ToolNode unwinds before any
   * sibling runs, so a non-idempotent sibling (send_email, billing)
   * cannot execute once on the first pass and AGAIN when LangGraph re-runs
   * the interrupted batch on resume.
   *
   * This set only REORDERS the direct group; it does NOT promote a name
   * into it. A name that resolves to a schema-only event stub (an
   * inherited `toolDefinition` with no executable instance — e.g. in a
   * self-spawned child that scrubs `graphTools`) stays event-dispatched.
   * Forcing such a name direct would invoke the stub, which throws
   * "should not be invoked directly in event-driven mode". For the guard
   * to apply, the interrupting tool must independently be direct.
   *
   * Opt-in and empty by default: when unset (or when no direct batch call
   * matches), direct-batch execution is byte-for-byte unchanged. See the
   * "Resume re-execution" section of {@link HumanInTheLoopConfig} for the
   * batch re-execution contract this guards against.
   */
  interruptingToolNames?: Set<string>;
  /** Opt-in eager execution for event-driven tool calls. */
  eagerEventToolExecution?: EagerEventToolExecutionConfig;
  /**
   * Host tool names that write to the code-execution sandbox but are not
   * built-in `CODE_EXECUTION_TOOLS`. Their exec `session_id` is folded into the
   * shared code session so later bash_tool/execute_code calls see written files.
   */
  codeSessionToolNames?: string[];
  /** Shared per-run eager execution registry populated by the stream handler. */
  eagerEventToolExecutions?: Map<string, EagerEventToolExecution>;
  /** Shared per-run per-tool turn counter used by eager and normal event dispatch. */
  eagerEventToolUsageCount?: Map<string, number>;
  /**
   * Shared per-run circuit breaker for eager prestart. When a prestarted
   * execution's args turn out to differ from the final request ("changed
   * after eager execution started"), the ToolNode adds the tool name here
   * and the stream handler stops prestarting that tool for the remainder of
   * the run, so the model's retry executes normally instead of looping.
   */
  eagerEventToolSuppressions?: Set<string>;
  /**
   * Hook registry for PreToolUse/PostToolUse/PostToolUseFailure/
   * PermissionDenied lifecycle hooks. Fires for **every** tool the
   * ToolNode invokes — both event-dispatched tools (via
   * `dispatchToolEvents`) and direct-path tools (via
   * `runDirectToolWithLifecycleHooks`). The pre-existing limitation
   * that direct-path tools "bypass hook dispatch entirely" was
   * lifted in a follow-up to the HITL surface; both paths now route
   * through the same hook lifecycle.
   */
  hookRegistry?: HookRegistry;
  /**
   * Run-scoped HITL config. **HITL is OFF by default** — omitting this
   * field (or passing `{ enabled: false }`) keeps the pre-HITL
   * fail-closed behavior where a `PreToolUse` `ask` decision collapses
   * into a blocked `ToolMessage`. Hosts opt in with
   * `{ enabled: true }` once their UI can render and resolve a
   * `tool_approval` interrupt; that engages the interrupt path where
   * `ask` raises a real LangGraph `interrupt()` carrying a
   * `HumanInterruptPayload` and the host resumes with
   * `Run.resume(decisions)`.
   *
   * Mirrors `RunConfig.humanInTheLoop` (which is the canonical place
   * to set this); the Graph threads it down to every ToolNode it
   * compiles. The interrupt path is wired into both the event
   * dispatch (`dispatchToolEvents`) and the direct path
   * (`runDirectToolWithLifecycleHooks`) — direct tools no longer
   * bypass HITL. See `HumanInTheLoopConfig` for the resume re-execution
   * contract that applies equally to both paths.
   */
  humanInTheLoop?: HumanInTheLoopConfig;
  /** Max context tokens for the agent — used to compute tool result truncation limits. */
  maxContextTokens?: number;
  /**
   * Maximum characters allowed in a single tool result before truncation.
   * When provided, takes precedence over the value computed from maxContextTokens.
   */
  maxToolResultChars?: number;
  /**
   * Run-scoped tool output reference configuration. When `enabled` is
   * `true`, ToolNode registers successful outputs and substitutes
   * `{{tool<idx>turn<turn>}}` placeholders found in string args.
   *
   * Ignored when `toolOutputRegistry` is also provided (host-supplied
   * registry wins).
   */
  toolOutputReferences?: ToolOutputReferencesConfig;
  /**
   * Pre-constructed registry instance shared across ToolNodes for the
   * run. Graphs pass the same registry to every ToolNode they compile
   * so cross-agent `{{tool<i>turn<n>}}` substitutions resolve. Takes
   * precedence over `toolOutputReferences` when both are set.
   */
  toolOutputRegistry?: ToolOutputReferenceRegistry;
  /**
   * Selects where built-in code execution tools run. Defaults to the
   * remote LibreChat Code API sandbox; `local` swaps those same tool names
   * to process-based local executors at ToolNode construction time.
   */
  toolExecution?: ToolExecutionConfig;
  /**
   * Pre-constructed file checkpointer shared across every ToolNode in
   * a Run so `Run.rewindFiles()` (and direct
   * `Run.getFileCheckpointer()` callers) sees a single unified
   * snapshot store. The Graph layer creates one per Run when
   * `toolExecution.local.fileCheckpointing === true` and threads it
   * to every ToolNode it compiles; without this shared instance,
   * multi-agent graphs would each get their own private checkpointer
   * and the host couldn't reach any of them. Wins over the
   * auto-created bundle checkpointer in
   * `resolveLocalExecutionTools`.
   */
  fileCheckpointer?: LocalFileCheckpointer;
  /**
   * Returns the owning graph's run-scoped breaker signal. Read once per
   * `run()` invocation and composed into the batch config's `signal`, so
   * every tool execution in the batch aborts when a stream circuit breaker
   * trips elsewhere in the run.
   */
  getBreakerSignal?: () => AbortSignal | undefined;
  /**
   * Returns the owning graph's immutable run scope. Read once per batch,
   * BEFORE hooks, and threaded to tools that spawn runs (subagents) so a
   * reset during a hook cannot rebind their children to a newer run's
   * controller.
   */
  getRunScope?: () => RunBreakerScope;
  /** Internal graph-owned prepared invocation registry. */
  preparedSubagents?: PreparedSubagents;
  /** SDK-owned checkpoint bridge for open run-step lifecycle state. */
  restoreRunStepResumeState?: (
    state?: RunStepResumeState,
    config?: RunnableConfig
  ) => void;
  /** SDK-owned checkpoint snapshot for open run-step lifecycle state. */
  createRunStepResumeState?: () => RunStepResumeState;
  /** Internal ownership bridge, awaited before ToolNode can execute a batch. */
  onToolCallsClaimed?: (messageId: string, config: RunnableConfig) => Promise<void>;
};

export type ToolNodeConstructorParams = ToolRefs & ToolNodeOptions;

export type ToolEndEvent = {
  /** The Step Id of the Tool Call */
  id: string;
  /**
   * The Completed Tool Call. Carries the tool-authored `outcome` label when
   * present (see `ProcessedToolCall.outcome`) so `ON_RUN_STEP_COMPLETED`
   * consumers can read it without an unsafe cast.
   */
  tool_call: ToolCall & {
    output?: string;
    progress?: number;
    outcome?: string;
  };
  /** The content index of the tool call */
  index: number;
  type?: 'tool_call';
  /** True when the stream eager path surfaced this completion before ToolNode finalized graph state. */
  eager?: boolean;
};

/**
 * Closed set of resource kinds for sandbox file caching. Defined as a
 * `as const` tuple so the runtime list and the TypeScript union can't
 * drift on future additions — adding a new kind to the tuple updates
 * both at once.
 *
 * - `skill`: shared per skill identity. Cross-user-within-tenant
 *   sharing. Codeapi sessionKey omits the user dimension. `version`
 *   is required (skill's monotonic counter scopes cache per revision).
 * - `agent`: shared per agent identity. Same sharing semantic as
 *   skills.
 * - `user`: user-private. Codeapi sessionKey is keyed by the
 *   requesting user from auth context. Used for chat attachments
 *   and code-output files.
 */
export const CODE_ENV_KINDS = ['skill', 'agent', 'user'] as const;
export type CodeEnvKind = (typeof CODE_ENV_KINDS)[number];

type CodeEnvFileBase = {
  /**
   * **Storage file id** — the per-file uuid the file_server returns
   * at upload time. Identifies the bytes at
   * `<storage_session_id>/<id>` in the object bucket; used by the
   * worker to fetch the file and by codeapi's auth layer for the
   * upload-existence check.
   *
   * Distinct from `resource_id` below — that's the entity that owns
   * this file's storage session (skill `_id`, agent id, etc.), used
   * for sessionKey resolution. Conflating the two caused
   * shared-kind authorization to fail because the sessionKey
   * re-derivation used the storage nanoid where the resource id was
   * required. See codeapi #1455 review.
   */
  id: string;
  /**
   * **Resource id** — the entity that owns this file's storage
   * session. Skill `_id` for `kind: 'skill'`, agent id for `'agent'`,
   * informational only for `'user'` (codeapi resolves user identity
   * from auth context). Kept on the type for shape uniformity across
   * kinds.
   *
   * Drives codeapi's `resolveSessionKey` switch — the sessionKey
   * embeds this value (`<tenant>:<kind>:<resource_id>[:v:<version>]`)
   * so cross-user-within-tenant sharing for shared kinds is a
   * designed property of the kind switch.
   */
  resource_id: string;
  name: string;
  /**
   * Storage session — the long-lived bucket where this file's bytes
   * live in object storage. Distinct from the (transient) execution
   * session id that appears at the top level of an execute response;
   * the two used to share the field name `session_id` and the
   * conflation caused real bugs. See codeapi #1455 / agents #148.
   */
  storage_session_id: string;
};

/**
 * `CodeEnvFile` is a discriminated union on `kind`. `version` is
 * statically required for `kind: 'skill'` and statically forbidden
 * for `agent` / `user` — the constraint holds at compile time on
 * every consumer, not just on codeapi's runtime validator.
 *
 * Codeapi switches on `kind` to derive the sessionKey for cache
 * scoping (`<tenant>:<kind>:<id>[:v:<version>]`). Cross-user sharing
 * for `kind: 'skill'` / `'agent'` is a designed property of the
 * kind switch.
 */
export type CodeEnvFile =
  | (CodeEnvFileBase & { kind: 'skill'; version: number })
  | (CodeEnvFileBase & { kind: 'agent' })
  | (CodeEnvFileBase & { kind: 'user' });

export type CodeExecutionToolParams =
  | undefined
  | {
      /** Trusted Code API endpoint selected by the host for this agent. */
      baseUrl?: string;
      /**
       * Expected Code API execution profile. Sent as a request assertion so
       * endpoint/configuration drift fails closed instead of silently using
       * the wrong sandbox backend.
       */
      executionProfile?: CodeApiExecutionProfile;
      /**
       * Trusted warm-runtime affinity hint selected by the host for this
       * agent invocation. This overrides the legacy ToolNode-injected hint.
       */
      runtimeSessionHint?: string;
      /** Execution session — see `CodeSessionContext.session_id`. */
      session_id?: string;
      user_id?: string;
      files?: CodeEnvFile[];
      /** Optional host-supplied Code API auth headers. */
      authHeaders?: CodeApiAuthHeaders;
      /**
       * Advertise filesystem-tier stateful sessions in the tool description
       * and allow a trusted runtime affinity hint onto the wire. Resolve this,
       * `baseUrl`, `executionProfile`, and `runtimeSessionHint` together from
       * the executing agent so the prompt and backend agree.
       */
      statefulSessions?: boolean;
    };

export type CodeApiExecutionProfile = 'default' | 'stateful';

export type CodeApiAuthHeaderMap = Record<string, string>;

export type CodeApiAuthHeaders =
  | CodeApiAuthHeaderMap
  | (() => CodeApiAuthHeaderMap | Promise<CodeApiAuthHeaderMap>);

export type FileRef = {
  /**
   * Storage file id (the per-file uuid). See `CodeEnvFile.id` for
   * the full motivation behind the storage-vs-resource split.
   */
  id: string;
  /**
   * Resource id — the entity that owns this file's storage session.
   * Optional on `FileRef` (post-execute artifact refs may not carry
   * resource provenance for outputs); required on `CodeEnvFile`
   * (the input wire shape).
   */
  resource_id?: string;
  name: string;
  path?: string;
  /**
   * Storage session this file lives in. See `CodeEnvFile.storage_session_id`
   * for the full motivation.
   */
  storage_session_id?: string;
  /** Resource kind — see `CodeEnvFile.kind`. */
  kind?: CodeEnvKind;
  /** Resource version — see `CodeEnvFile.version`. Only meaningful when
   *  `kind === 'skill'`. */
  version?: number;
  /**
   * `true` when the codeapi sandbox echoed this entry as an unchanged
   * passthrough of an input the caller already owns (skill files,
   * downloaded inputs whose hash matched the baseline, inherited
   * `.dirkeep` markers). The host can use this flag to skip
   * post-processing work for files the caller already owns.
   */
  inherited?: true;
};

export type FileRefs = FileRef[];

export type ArtifactDeliveryFailure = {
  code: 'artifact_delivery_failed';
  status: 'partial' | 'failed';
  attempted: number;
  delivered: number;
  failed: number;
};

export type ExecuteResult = {
  /**
   * Execution session id — the (transient) sandbox run that produced
   * this output. Distinct from per-file `storage_session_id` on the
   * files array.
   */
  session_id: string;
  stdout: string;
  stderr: string;
  files?: FileRefs;
  /** Persisted input paths explicitly removed during this execution. */
  deleted_files?: string[];
  artifact_delivery?: ArtifactDeliveryFailure;
  /**
   * Durable runtime session id echoed by a stateful Code API backend
   * (hash of tenant+user+hint). Additive; absent on stateless servers.
   */
  runtime_session_id?: string;
  /** Whether this execution reused a warm runtime session or started fresh. */
  runtime_status?: 'new' | 'reused';
};

/** JSON Schema type definition for tool parameters */
export type JsonSchemaType = {
  type:
    | 'string'
    | 'number'
    | 'integer'
    | 'float'
    | 'boolean'
    | 'array'
    | 'object';
  enum?: string[];
  items?: JsonSchemaType;
  properties?: Record<string, JsonSchemaType>;
  required?: string[];
  description?: string;
  additionalProperties?: boolean | JsonSchemaType;
};

/**
 * Specifies which contexts can invoke a tool (inspired by Anthropic's allowed_callers)
 * - 'direct': Only callable directly by the LLM (default if omitted)
 * - 'code_execution': Only callable from within programmatic code execution
 */
export type AllowedCaller = 'direct' | 'code_execution';

/** Tool definition with optional deferred loading and caller restrictions */
export type LCTool = {
  name: string;
  description?: string;
  parameters?: JsonSchemaType;
  /** When true, tool is not loaded into context initially (for tool search) */
  defer_loading?: boolean;
  /**
   * Which contexts can invoke this tool.
   * Default: ['direct'] (only callable directly by LLM)
   * Options: 'direct', 'code_execution'
   */
  allowed_callers?: AllowedCaller[];
  /** Response format for the tool output */
  responseFormat?: 'content' | 'content_and_artifact';
  /** Server name for MCP tools */
  serverName?: string;
  /** Tool type classification */
  toolType?: 'builtin' | 'mcp' | 'action';
};

/** Single tool call within a batch request for event-driven execution */
export type ToolCallRequest = {
  /** Tool call ID from the LLM */
  id: string;
  /** Tool name */
  name: string;
  /** Tool arguments */
  args: Record<string, unknown>;
  /** Step ID for tracking */
  stepId?: string;
  /** Usage turn count for this tool */
  turn?: number;
  /** Code execution session context for session continuity in event-driven mode */
  codeSessionContext?: {
    /** Execution session — see `CodeSessionContext.session_id`. */
    session_id: string;
    files?: CodeEnvFile[];
  };
  /** Request-time execution session used to detect host session refreshes. */
  codeSessionBaselineId?: string;
  /** Preserve request-provisioned code inputs after an eager identity mismatch. */
  retainCodeSessionInputs?: true;
  /**
   * Stable runtime session hint for stateful sandbox sessions. Orthogonal to
   * `codeSessionContext` (which threads the transient exec-session for file
   * continuity): the hint identifies the durable server-side runtime session.
   */
  runtimeSessionHint?: string;
};

/**
 * Serializable view of the active tools grouped by their effective caller
 * capabilities. Event-driven hosts use this snapshot to preserve the SDK's
 * live deferred-tool discovery state without reimplementing its policy.
 */
export type CallerCapabilityProjectionSnapshot = {
  version: 1;
  directToolNames: string[];
  codeExecutionToolNames: string[];
  directOnlyToolNames: string[];
  codeExecutionOnlyToolNames: string[];
};

/** Batch request containing ALL tool calls for a graph step */
export type ToolExecuteBatchRequest = {
  /** SDK-owned child lineage. Unset for tools in the root graph. */
  executionContext?: SubagentExecutionContext;
  /** All tool calls from the AIMessage */
  toolCalls: ToolCallRequest[];
  /** User ID for context */
  userId?: string;
  /** Agent ID for context */
  agentId?: string;
  /**
   * SDK-owned snapshot of the active caller capability projection. Optional
   * so older event handlers remain compatible with newer SDKs.
   */
  callerCapabilityProjection?: CallerCapabilityProjectionSnapshot;
  /** Runtime configurable from RunnableConfig (includes user, userMCPAuthMap, etc.) */
  configurable?: Record<string, unknown>;
  /** Runtime metadata from RunnableConfig (includes thread_id, run_id, provider, etc.) */
  metadata?: Record<string, unknown>;
  /**
   * Aborts when the run is cancelled or its stream circuit breaker trips.
   * Handlers SHOULD forward this to their tool executions so in-flight work
   * stops consuming quota once the run is already failing.
   */
  signal?: AbortSignal;
  /** Promise resolver - handler calls this with ALL results */
  resolve: (results: ToolExecuteResult[]) => void;
  /** Promise rejector - handler calls this on fatal error */
  reject: (error: Error) => void;
  /**
   * Optional per-call result channel. When present, the handler MAY invoke
   * this as each tool call settles (before the final `resolve`) so the
   * graph can emit that call's completion event without waiting for the
   * slowest call in the batch. Purely an emission fast-path: the handler
   * must still pass every result to `resolve`, which remains the
   * authoritative batch outcome. Only provided when no post-tool hooks or
   * human-in-the-loop flows could change a result after execution.
   */
  onResult?: (result: ToolExecuteResult) => void;
};

/**
 * A message injected into graph state by any tool execution handler.
 * Generic mechanism: any tool returning `injectedMessages` in its `ToolExecuteResult`
 * will have these appended to state after the ToolMessage for this call.
 */
export type InjectedMessage = {
  /** 'user' for skill body injection, 'system' for context hints.
   *  Both are converted to HumanMessage at runtime; the original role
   *  is preserved in additional_kwargs.role. */
  role: 'user' | 'system';
  /** Message content: string for simple text, array for complex multi-part content */
  content: string | MessageContentComplex[];
  /** When true, the message is framework-internal: not shown in UI, not counted as a user turn */
  isMeta?: boolean;
  /** Origin tag for downstream consumers (UI, pruner, compaction) */
  source?: 'skill' | 'hook' | 'system' | 'steer';
  /** Only set when source is 'skill', for compaction preservation */
  skillName?: string;
};

/**
 * In-place edit of a call's model-authored `intent` label: the first
 * occurrence of `from` in the intent is replaced with `to` (case-sensitive).
 * Lets a tool settle the label while preserving the model's own phrasing,
 * e.g. `{ from: 'Searching', to: 'Searched' }`.
 */
export type OutcomePatch = {
  from: string;
  to: string;
};

/** Result for a single tool call in event-driven execution */
export type ToolExecuteResult = {
  /** Matches ToolCallRequest.id */
  toolCallId: string;
  /** Tool output content */
  content: string | unknown[];
  /** Optional artifact (for content_and_artifact format) */
  artifact?: unknown;
  /** Execution status */
  status: 'success' | 'error';
  /** Error message if status is 'error' */
  errorMessage?: string;
  /**
   * Settled human-readable label for this call, replacing the model-authored
   * `intent` arg in the UI. Full replacement; wins over `outcome_patch`.
   */
  outcome?: string;
  /** In-place edit of the model-authored `intent` label (see {@link OutcomePatch}). */
  outcome_patch?: OutcomePatch;
  /**
   * Messages to inject into graph state after the ToolMessage for this call.
   * Placed after tool results to respect provider message ordering (tool_call -> tool_result adjacency).
   * The host's message formatter may merge injected user messages with the preceding tool_result turn.
   * Generic mechanism: any tool execution handler can use this.
   */
  injectedMessages?: InjectedMessage[];
};

/** Map of tool names to tool definitions */
export type LCToolRegistry = Map<string, LCTool>;

/**
 * Run-scoped configuration for tool output references.
 *
 * When enabled, each successful tool result is registered under a stable
 * key (`tool<idx>turn<turn>`). Later tool calls can pipe a previous
 * output into their arguments by including the literal placeholder
 * `{{tool<idx>turn<turn>}}` anywhere in a string argument; ToolNode
 * substitutes it with the stored output immediately before invoking
 * the tool.
 *
 * The registry stores the *raw, untruncated* tool output (subject to
 * its own size caps) so a later substitution can pipe the full payload
 * into the next tool even when the LLM only saw a head+tail-truncated
 * preview in `ToolMessage.content`. Size limits are decoupled from the
 * LLM-visible truncation budget and default to 5 MB total.
 *
 * Known limitations:
 *  - Tools that return a `ToolMessage` with array-type content
 *    (multi-part content blocks such as text + image) are not
 *    registered and cannot be cited via `{{tool<i>turn<n>}}`. A
 *    warning is logged so the missing reference is visible.
 *  - When a `PostToolUse` hook replaces `ToolMessage.content`, the
 *    *post-hook* content is what gets stored in the registry (and
 *    what the model sees), so `{{…}}` substitutions deliver the
 *    hooked output rather than the raw tool return. This matches the
 *    hook's "authoritative" role for output shaping.
 */
export type ToolOutputReferencesConfig = {
  /** Enable the registry and placeholder substitution. Defaults to `false`. */
  enabled?: boolean;
  /**
   * Maximum characters stored (and substituted) per registered output.
   * Applied to the *raw* output before storage. Defaults to
   * `HARD_MAX_TOOL_RESULT_CHARS` (~400 KB) — matching the
   * LLM-visible tool-result truncation budget, which is also a safe
   * payload size for shell `ARG_MAX` limits when a `{{…}}` expansion
   * gets piped into a bash `command`. Hosts that want to preserve
   * fuller fidelity (for example for non-bash API consumers) can
   * raise this up to `maxTotalSize` (defaults to 5 MB) — be aware
   * that large single-output substitutions may exceed shell
   * argument-size limits on typical Linux/macOS.
   */
  maxOutputSize?: number;
  /**
   * Hard cap on total characters retained across all registered outputs
   * for the run. When exceeded, the oldest entries are evicted FIFO
   * until the total fits. The effective per-output cap is
   * `min(maxOutputSize, maxTotalSize)` so a single stored output can
   * never exceed the aggregate bound. Defaults to
   * `calculateMaxTotalToolOutputSize(maxOutputSize)` (5 MB).
   */
  maxTotalSize?: number;
};

export type ToolExecutionEngine = 'sandbox' | 'local' | 'cloudflare-sandbox';

/**
 * Records pre-write file contents so callers can rewind edits/writes
 * made by local-compatible coding-tool engines. Implementations live
 * in `src/tools/local`.
 */
export interface LocalFileCheckpointer {
  /**
   * Captures the current contents of `absolutePath` before a write or
   * edit. Idempotent: capturing the same path twice keeps the first
   * snapshot. Records "did not exist" so creates can be undone with
   * deletion.
   */
  captureBeforeWrite(absolutePath: string): Promise<void>;
  /** Restores all captured snapshots. Returns the number of files restored. */
  rewind(): Promise<number>;
  /** Returns paths that have been captured during this run. */
  capturedPaths(): string[];
}

/**
 * Pluggable process launcher used by the local execution engine. When
 * provided, the engine calls this in place of `child_process.spawn`,
 * letting callers route shell commands through SSH, containers, or
 * remote runners without forking the engine. The implementation must
 * return a `ChildProcess`-shaped value whose `stdout`/`stderr` streams
 * emit `data` events and that resolves a `close` event when finished.
 */
export type LocalSpawn = (
  command: string,
  args: string[],
  options: import('child_process').SpawnOptions
) => import('child_process').ChildProcessWithoutNullStreams;

/** Bash command-validation strictness for the local engine. */
export type LocalBashAstMode = 'auto' | 'off' | 'strict';

export type LocalSandboxConfig = {
  /**
   * Enable Anthropic Sandbox Runtime wrapping for local process tools.
   * Defaults to false; requires @anthropic-ai/sandbox-runtime to be installed.
   */
  enabled?: boolean;
  /** Throw when native sandbox dependencies are unavailable. Defaults to false. */
  failIfUnavailable?: boolean;
  filesystem?: {
    denyRead?: string[];
    allowRead?: string[];
    allowWrite?: string[];
    denyWrite?: string[];
    allowGitConfig?: boolean;
  };
  network?: {
    allowedDomains?: string[];
    deniedDomains?: string[];
    allowUnixSockets?: string[];
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
    allowMachLookup?: string[];
  };
};

/**
 * Workspace boundary the local-coding tools clamp file operations to.
 *
 * `root` is the canonical "project directory" — every file tool that
 * accepts a path resolves it relative to `root` and refuses to touch
 * paths outside it (after symlink resolution). `additionalRoots` lets
 * monorepos extend the boundary to sibling directories without
 * disabling clamping entirely.
 *
 * `allowReadOutside` and `allowWriteOutside` are escape hatches that
 * disable clamping for read- and write-shaped tools respectively.
 * Most hosts shouldn't need them — prefer wiring
 * `createWorkspacePolicyHook` if you want "ask the user" semantics
 * via the existing PreToolUse / HITL machinery instead of an
 * unconditional bypass.
 */
export type LocalWorkspaceConfig = {
  /** Required. The canonical workspace root. */
  root: string;
  /**
   * Sibling roots that also count as inside the workspace. Useful in
   * monorepos where a project legitimately needs to reach a paired
   * directory without disabling clamping.
   */
  additionalRoots?: readonly string[];
  /** When true, disable the read-outside-workspace clamp. */
  allowReadOutside?: boolean;
  /** When true, disable the write-outside-workspace clamp. */
  allowWriteOutside?: boolean;
};

/**
 * One execution world shared by filesystem and subprocess operations.
 * Backend identity is stable so capability probes remain warm when tool
 * bundles are rebuilt for later agent bindings.
 */
export interface ExecutionWorld {
  /** Launches a process inside this world's filesystem namespace. */
  readonly spawn: LocalSpawn;
  /** Reads and writes the same namespace observed by `spawn`. */
  readonly fs: Readonly<import('@/tools/local/workspaceFS').WorkspaceFS>;
  /** Whether the world already enforces a sandbox boundary. */
  readonly sandboxed: boolean;
}

/**
 * Backward-compatible partial execution-world override. Omitted fields use
 * the Node host world; remote backends should provide the complete trio.
 *
 * Non-bash `execute_code` still stages source through the host temporary
 * directory. Remote runtimes should use their dedicated execution engine for
 * that tool until temporary lifecycle operations join this seam.
 *
 * Command validators and workspace policies are best-effort tripwires. For an
 * adversarial-model threat model, use a backend sandbox boundary rather than
 * relying on validation alone.
 */
export type LocalExecConfig = {
  -readonly [Key in keyof ExecutionWorld]?: Key extends 'fs'
    ? import('@/tools/local/workspaceFS').WorkspaceFS
    : ExecutionWorld[Key];
};

export type LocalExecutionConfig = {
  /**
   * Working directory for local commands. Defaults to process.cwd().
   * Back-compat shorthand for `workspace.root`.
   *
   * @deprecated Prefer `workspace.root`. Kept for backward
   *   compatibility with pre-workspace configs.
   */
  cwd?: string;
  /**
   * Workspace boundary configuration. When omitted, the implementation
   * derives `{ root: cwd ?? process.cwd() }` so existing call sites
   * keep working.
   */
  workspace?: LocalWorkspaceConfig;
  /**
   * Execution seam (spawn + fs). When omitted, defaults to Node-host
   * child_process and fs/promises. Same back-compat: a top-level
   * `spawn` field is honoured if `exec.spawn` isn't set.
   */
  exec?: LocalExecConfig;
  /** Shell executable for bash-style tools. Defaults to `bash`. */
  shell?: string;
  /** Default timeout for local processes, in milliseconds. */
  timeoutMs?: number;
  /** Maximum stdout/stderr characters surfaced to the model. */
  maxOutputChars?: number;
  /**
   * Hard cap on total bytes a single child process can stream across
   * stdout+stderr before its process tree is killed. Independent from
   * `maxOutputChars` (which controls what the model sees); this is the
   * OOM backstop for noisy / runaway commands (`yes`, `cat /dev/urandom`,
   * a verbose build that loops). Defaults to 50 MiB. Set higher for
   * legitimately large logs; setting to 0 disables the cap and risks
   * an OOM crash on the host.
   */
  maxSpawnedBytes?: number;
  /** Extra environment variables merged over process.env. */
  env?: NodeJS.ProcessEnv;
  /** Optional process sandboxing via @anthropic-ai/sandbox-runtime. */
  sandbox?: LocalSandboxConfig;
  /**
   * When true, block obviously mutating shell commands before execution.
   * Useful for read-only agent modes and dry-run workflows.
   */
  readOnly?: boolean;
  /** Permit dangerous commands that the validator otherwise blocks. */
  allowDangerousCommands?: boolean;
  /**
   * Permit file tools to resolve paths outside `cwd`. Defaults to false.
   *
   * @deprecated Prefer the granular
   *   `workspace.allowReadOutside` / `workspace.allowWriteOutside`.
   *   Kept for backward compatibility; setting either of the
   *   workspace fields takes precedence over this flag.
   */
  allowOutsideWorkspace?: boolean;
  /**
   * Add the built-in local coding suite (`read_file`, `write_file`,
   * `edit_file`, `grep_search`, `glob_search`, `list_directory`, plus local
   * code/bash execution tools) when `engine` is `local`. Defaults to true.
   */
  includeCodingTools?: boolean;
  /**
   * Override the process launcher. When set, replaces
   * `child_process.spawn` for every local tool invocation, allowing
   * SSH/container delegation. Default: native spawn.
   */
  spawn?: LocalSpawn;
  /**
   * Tree-sitter-bash AST validation pass on bash commands.
   * - `'off'` skips AST validation (regex + `bash -n` only — current behavior).
   * - `'auto'` runs the AST validator when `tree-sitter` modules are
   *   available; falls back silently otherwise.
   * - `'strict'` requires the AST validator and fails closed when
   *   parsing is unavailable or the command is too complex to verify.
   * Default: `'off'` to preserve historical behavior.
   */
  bashAst?: LocalBashAstMode;
  /**
   * Enable per-Run file checkpointing for `edit_file` / `write_file`
   * so callers can rewind file changes after a failed batch. When
   * true and the run is constructed via `Run.create(...)`, the host
   * can call `Run.rewindFiles()` (or `Run.getFileCheckpointer()` for
   * the raw checkpointer). Defaults to false.
   */
  fileCheckpointing?: boolean;
  /**
   * Maximum bytes to read in `read_file` before returning a stub.
   * Defaults to 10 MiB.
   */
  maxReadBytes?: number;
  /**
   * Controls whether `read_file` returns binary files as inline
   * `MessageContentComplex[]` attachments (so vision-capable models
   * see them) or as a textual stub.
   *
   *   - `'off'`        : never embed; current binary-stub behavior.
   *   - `'images-only'`: embed images (png/jpeg/gif/webp) as
   *     `image_url` blocks; other binaries get the stub.
   *   - `'images-and-pdf'` : also embed PDFs as `image_url` data URLs
   *     (Anthropic accepts these in tool_result; other providers may
   *     degrade to JSON).
   *
   * Defaults to `'off'` to preserve current behavior.
   */
  attachReadAttachments?: 'off' | 'images-only' | 'images-and-pdf';
  /**
   * Maximum pre-encoding byte size to embed inline. Anything larger
   * degrades to an `<oversize>` stub. Defaults to 5 MiB to bound the
   * post-base64 token cost.
   */
  maxAttachmentBytes?: number;
  /**
   * Run a fast per-file syntax check after every successful
   * `edit_file` / `write_file`. When the checker finds an error,
   * the diagnostics are appended to the tool result so the model
   * can self-correct without a separate read round-trip.
   *
   *   - `'off'` (default) : skip; current behavior.
   *   - `'auto'`          : run the checker for known file types
   *     when the corresponding tool is on PATH. Silently skip
   *     otherwise.
   *   - `'strict'`        : same as `'auto'`, plus fail the tool
   *     call with the error so the model is forced to react. Use
   *     when you don't trust the model to read a non-blocking
   *     advisory.
   *
   * Built-in checkers: Node `node --check` for `.js/.mjs/.cjs`,
   * Python `py_compile` for `.py`, `JSON.parse` for `.json`,
   * `bash -n` for `.sh/.bash`. TypeScript falls back to `compile_check`
   * (project-level) since per-file `.ts` syntax check requires the
   * `typescript` package; the host can wire a per-file checker via
   * `local.spawn` if desired.
   */
  postEditSyntaxCheck?: 'off' | 'auto' | 'strict';
  /**
   * Configuration for the `compile_check` tool. When `engine` is
   * `local` and `includeCodingTools` is on, the SDK exposes a
   * `compile_check` tool that runs the project's standard
   * type/lint command (`tsc --noEmit`, `cargo check`, etc.).
   */
  compileCheck?: {
    /**
     * Override the auto-detected command. Runs verbatim from `cwd`
     * via the local engine's standard spawn pipeline (sandbox / AST
     * validation / output overflow all apply).
     */
    command?: string;
    /** Default timeout for `compile_check`, in milliseconds. Defaults to 120s. */
    timeoutMs?: number;
  };
};

export type CloudflareSandboxExecOptions = {
  cwd?: string;
  env?: Record<string, string | undefined>;
  timeout?: number;
  stream?: boolean;
  onOutput?: (stream: 'stdout' | 'stderr', data: string) => void;
  signal?: AbortSignal;
};

export type CloudflareSandboxExecResult = {
  success?: boolean;
  exitCode: number;
  stdout: string;
  stderr: string;
  command?: string;
  duration?: number;
  timestamp?: string;
};

export type CloudflareSandboxReadFileResult =
  | string
  | Buffer
  | Uint8Array
  | {
      content: string | Buffer | Uint8Array | ReadableStream<Uint8Array>;
      size?: number;
      mimeType?: string;
      encoding?: 'utf-8' | 'utf8' | 'base64';
      isBinary?: boolean;
    };

export type CloudflareSandboxListFilesOptions = {
  recursive?: boolean;
  includeHidden?: boolean;
};

export type CloudflareSandboxFileInfo = {
  name: string;
  absolutePath?: string;
  relativePath?: string;
  type?: 'file' | 'directory' | 'symlink' | 'other';
  size?: number;
  modifiedAt?: string;
  mode?: string;
};

export type CloudflareSandboxListFilesResult =
  | CloudflareSandboxFileInfo[]
  | {
      files: CloudflareSandboxFileInfo[];
      count?: number;
      path?: string;
      success?: boolean;
    };

export interface CloudflareSandboxRuntime {
  /**
   * True when this runtime can consume AbortSignal values in exec options.
   * Native Cloudflare Sandbox Durable Object RPC cannot clone AbortSignal,
   * but HTTP bridge runtimes can use it to abort the underlying fetch.
   */
  supportsExecSignal?: boolean;
  exec(
    command: string,
    options?: CloudflareSandboxExecOptions
  ): Promise<CloudflareSandboxExecResult>;
  readFile(
    path: string,
    options?: { encoding?: string }
  ): Promise<CloudflareSandboxReadFileResult>;
  writeFile(
    path: string,
    content: string | ReadableStream<Uint8Array>,
    options?: { encoding?: string }
  ): Promise<unknown>;
  mkdir(path: string, options?: { recursive?: boolean }): Promise<unknown>;
  listFiles(
    path: string,
    options?: CloudflareSandboxListFilesOptions
  ): Promise<CloudflareSandboxListFilesResult>;
  deleteFile(path: string): Promise<unknown>;
}

export type CloudflareSandboxExecutionConfig = {
  sandbox:
    | CloudflareSandboxRuntime
    | (() => CloudflareSandboxRuntime | Promise<CloudflareSandboxRuntime>);
  /** Working directory inside the sandbox. Defaults to `/workspace`. */
  workspaceRoot?: string;
  /** Extra environment variables merged into sandbox command executions. */
  env?: Record<string, string | undefined>;
  /** Default timeout for sandbox commands, in milliseconds. */
  timeoutMs?: number;
  /** Maximum stdout/stderr characters surfaced to the model. */
  maxOutputChars?: number;
  /**
   * Add the built-in coding suite when `engine` is `cloudflare-sandbox`.
   * Defaults to true, matching the local execution backend.
   */
  includeCodingTools?: boolean;
  /**
   * Optional exact coding-tool names to expose when `includeCodingTools`
   * is on. Defaults to the full local-parity Cloudflare bundle. Use this
   * to publish a bash-only sandbox surface, for example file/search tools
   * plus `bash_tool` and `run_tools_with_bash`, without exposing
   * `execute_code`.
   */
  codingToolNames?: readonly string[];
  /** Optional shell executable for bash-style tools. Defaults to `bash`. */
  shell?: string;
  /**
   * Optional project-level compile check configuration. Mirrors the local
   * execution backend.
   */
  compileCheck?: LocalExecutionConfig['compileCheck'];
  /** Optional read-only guard for mutating coding tools. */
  readOnly?: boolean;
  /** Permit dangerous commands that the validator otherwise blocks. */
  allowDangerousCommands?: LocalExecutionConfig['allowDangerousCommands'];
  /** Tree-sitter-bash AST validation pass for bash commands. */
  bashAst?: LocalExecutionConfig['bashAst'];
  /**
   * Enable per-Run file checkpointing for `edit_file` / `write_file`
   * against the Cloudflare Sandbox workspace.
   */
  fileCheckpointing?: LocalExecutionConfig['fileCheckpointing'];
  /** Maximum bytes to read in `read_file` before returning a stub. */
  maxReadBytes?: LocalExecutionConfig['maxReadBytes'];
  /** Controls whether `read_file` returns binary files as attachments. */
  attachReadAttachments?: LocalExecutionConfig['attachReadAttachments'];
  /** Maximum pre-encoding byte size to embed inline. */
  maxAttachmentBytes?: LocalExecutionConfig['maxAttachmentBytes'];
  /** Run a fast per-file syntax check after successful edits/writes. */
  postEditSyntaxCheck?: LocalExecutionConfig['postEditSyntaxCheck'];
};

export type SandboxExecutionConfig = {
  /**
   * Legacy run-scoped opt-in for warm runtime hint injection. New hosts that
   * need mixed execution profiles should resolve `statefulSessions`,
   * `baseUrl`, `executionProfile`, and `runtimeSessionHint` on each executing
   * agent's tool factories instead. Direct executors ignore this injected hint
   * unless their own factory has `statefulSessions: true`.
   *
   * It does NOT change the model-facing tool description. Tool descriptions are
   * bound to the LLM at construction time (`createCodeExecutionTool` /
   * `createBashExecutionTool`), before this run config is applied inside the
   * graph, so they can only be adjusted via the tools' own `statefulSessions`
   * factory param.
   */
  statefulSessions?: boolean;
  /**
   * Stable identity for the runtime session (e.g. the conversation id). The
   * server derives the real session id as hash(tenant, user, hint), so this
   * is never a security boundary. Falls back to `configurable.thread_id` when
   * omitted.
   */
  runtimeSessionHint?: string;
};

export type ToolExecutionConfig = {
  /** `sandbox` preserves the remote Code API behavior and is the default. */
  engine?: ToolExecutionEngine;
  /** Local process execution settings used when `engine` is `local`. */
  local?: LocalExecutionConfig;
  /** Cloudflare Sandbox execution settings used when `engine` is `cloudflare-sandbox`. */
  cloudflare?: CloudflareSandboxExecutionConfig;
  /** Remote sandbox settings; applies when `engine` is `sandbox` or omitted. */
  sandbox?: SandboxExecutionConfig;
};

export type ProgrammaticCache = {
  toolMap: ToolMap;
  toolDefs: LCTool[];
  /** Actual outer runner name used for caller-policy diagnostics. */
  programmaticToolName?: string;
  /**
   * Known tools that cannot be invoked from programmatic execution. Kept
   * separate from `toolDefs` so the Programmatic Tool Manifest can be
   * validated before the execution runtime starts without exposing their
   * schemas to that runtime.
   */
  disallowedToolDefs?: LCTool[];
  /**
   * Hook context plumbed through by ToolNode for the local
   * programmatic-tool path so the in-process bridge can run
   * `PreToolUse` hooks (deny / updatedInput) for inner tool calls.
   * Not present for non-local (remote-engine) programmatic calling
   * which dispatches inner tools through the host's own pipeline.
   */
  hookContext?: ProgrammaticHookContext;
};

export type ProgrammaticHookContext = {
  registry: import('@/hooks').HookRegistry | undefined;
  runId: string;
  threadId?: string;
  agentId?: string;
  executionContext?: SubagentExecutionContext;
  executingAgentId?: string;
};

/** Search mode: code_interpreter uses external sandbox, local uses safe substring matching */
export type ToolSearchMode = 'code_interpreter' | 'local';

/** Format for MCP tool names in search results */
export type McpNameFormat = 'full' | 'base';

/** Parameters for creating a Tool Search tool */
export type ToolSearchParams = {
  toolRegistry?: LCToolRegistry;
  onlyDeferred?: boolean;
  baseUrl?: string;
  /** Search mode: 'code_interpreter' (default) uses sandbox for regex, 'local' uses safe substring matching */
  mode?: ToolSearchMode;
  /** Filter tools to only those from specific MCP server(s). Can be a single name or array of names. */
  mcpServer?: string | string[];
  /** Format for MCP tool names: 'full' (tool_mcp_server) or 'base' (tool only). Default: 'full' */
  mcpNameFormat?: McpNameFormat;
};

/** Simplified tool metadata for search purposes */
export type ToolMetadata = {
  name: string;
  description: string;
  parameters?: JsonSchemaType;
};

/** Individual search result for a matching tool */
export type ToolSearchResult = {
  tool_name: string;
  match_score: number;
  matched_field: string;
  snippet: string;
};

/** Response from the tool search operation */
export type ToolSearchResponse = {
  tool_references: ToolSearchResult[];
  total_tools_searched: number;
  pattern_used: string;
};

/** Artifact returned alongside the formatted search results */
export type ToolSearchArtifact = {
  tool_references: ToolSearchResult[];
  metadata: {
    total_searched: number;
    pattern: string;
    error?: string;
    /** Servers that had tools to search, emitted when a server filter matched
     * nothing so the caller can see what it could have asked for. */
    available_mcp_servers?: string[];
    /** Requested servers that matched no registered server. */
    unmatched_mcp_servers?: string[];
    /** Requested servers that exist but had no tools left to search. */
    idle_mcp_servers?: string[];
  };
};

// ============================================================================
// Programmatic Tool Calling Types
// ============================================================================

/**
 * Tool call requested by the Code API during programmatic execution
 */
export type PTCToolCall = {
  /** Unique ID like "call_001" */
  id: string;
  /** Tool name */
  name: string;
  /** Input parameters. Some bridges may serialize object input as JSON text. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input: Record<string, any> | string;
};

/**
 * Tool result sent back to the Code API
 */
export type PTCToolResult = {
  /** Matches PTCToolCall.id */
  call_id: string;
  /** Tool execution result (any JSON-serializable value) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: any;
  /** Whether tool execution failed */
  is_error: boolean;
  /** Error details if is_error=true */
  error_message?: string;
};

/**
 * Response from the Code API for programmatic execution
 */
export type ProgrammaticExecutionResponse = {
  status: 'tool_call_required' | 'completed' | 'error' | unknown;
  /** Execution session — see `CodeSessionContext.session_id`. */
  session_id?: string;

  /** Present when status='tool_call_required' */
  continuation_token?: string;
  tool_calls?: PTCToolCall[];

  /** Present when status='completed' */
  stdout?: string;
  stderr?: string;
  files?: FileRefs;
  /** Persisted input paths explicitly removed during this execution. */
  deleted_files?: string[];
  artifact_delivery?: ArtifactDeliveryFailure;

  /** Durable runtime session echo from a stateful backend (additive). */
  runtime_session_id?: string;
  runtime_status?: 'new' | 'reused';

  /** Present when status='error' */
  error?: string;
};

/**
 * Artifact returned by the PTC tool
 */
export type ProgrammaticExecutionArtifact = {
  /** Execution session — see `CodeSessionContext.session_id`. */
  session_id?: string;
  files?: FileRefs;
  /** Persisted input paths explicitly removed during this execution. */
  deleted_files?: string[];
  artifact_delivery?: ArtifactDeliveryFailure;
  /** Durable runtime session echo from a stateful backend (additive). */
  runtime_session_id?: string;
  runtime_status?: 'new' | 'reused';
};

/** Parameters for creating a bash execution tool (same API as CodeExecutor, bash-only) */
export type BashExecutionToolParams =
  | undefined
  | (Exclude<CodeExecutionToolParams, undefined> & {
      /** Trusted attached-workspace identifier selected by the host. */
      workspaceId?: string;
      /** Trusted conversation-owned workspace instance selected by the host. */
      workspaceInstanceId?: string;
    });

/** Parameters for creating a bash programmatic tool calling tool (same API as PTC, bash-only) */
export type BashProgrammaticToolCallingParams =
  ProgrammaticToolCallingParams & {
    /** Trusted attached-workspace identifier selected by the host. */
    workspaceId?: string;
    /** Trusted conversation-owned workspace instance selected by the host. */
    workspaceInstanceId?: string;
  };

/**
 * Initialization parameters for the PTC tool
 */
export type ProgrammaticToolCallingParams = {
  /** Code API base URL (or use CODE_BASEURL env var) */
  baseUrl?: string;
  /** Expected Code API execution profile (host-controlled). */
  executionProfile?: CodeApiExecutionProfile;
  /** Trusted warm-runtime affinity hint for this agent invocation. */
  runtimeSessionHint?: string;
  /** Safety limit for round-trips (default: 20) */
  maxRoundTrips?: number;
  /** Maximum per-sandbox-run timeout for PTC's legacy `timeout` field. */
  runTimeoutMs?: number;
  /** HTTP proxy URL */
  proxy?: string;
  /** Enable debug logging (or set PTC_DEBUG=true env var) */
  debug?: boolean;
  /** Optional host-supplied Code API auth headers. */
  authHeaders?: CodeApiAuthHeaders;
  /* No `statefulSessions` here: PTC keeps a stateless runtime prompt in v1.
   * An explicit stateful execution profile may still use a warm filesystem;
   * runtime/in-memory state never carries across calls. */
};

// ============================================================================
// Tool Session Context Types
// ============================================================================

/**
 * Tracks code execution session state for automatic file persistence.
 * Stored in Graph.sessions and injected into subsequent tool invocations.
 */
export type CodeSessionContext = {
  /**
   * Execution session id — the (transient) sandbox run id. Used by
   * ToolNode to thread session continuity into the next code-execution
   * tool call. Distinct from per-file `storage_session_id` carried on
   * `files`.
   */
  session_id: string;
  /** Files generated in this session (for context/tracking) */
  files?: FileRefs;
  /** Timestamp of last update */
  lastUpdated: number;
};

/**
 * Artifact structure returned by code execution tools (CodeExecutor, PTC).
 * Used to extract session context after tool completion.
 */
export type CodeExecutionArtifact = {
  /** Execution session — see `CodeSessionContext.session_id`. */
  session_id?: string;
  files?: FileRefs;
  /** Persisted input paths explicitly removed during this execution. */
  deleted_files?: string[];
  artifact_delivery?: ArtifactDeliveryFailure;
  /** Durable runtime session echo from a stateful backend (additive). */
  runtime_session_id?: string;
  runtime_status?: 'new' | 'reused';
};

/**
 * Generic session context union type for different tool types.
 * Extend this as new tool session types are added.
 */
export type ToolSessionContext = CodeSessionContext;

/**
 * Map of tool names to their session contexts.
 * Keys are tool constants (e.g., Constants.EXECUTE_CODE, Constants.PROGRAMMATIC_TOOL_CALLING).
 */
export type ToolSessionMap = Map<string, ToolSessionContext>;

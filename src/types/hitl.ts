/**
 * First-class human-in-the-loop (HITL) types for `@librechat/agents`.
 * Surfaces the interrupt payload that `ToolNode` raises when a `PreToolUse`
 * hook returns `decision: 'ask'` and HITL is enabled on the run, plus the
 * resume-decision shape the host returns to continue or reject the tool.
 *
 * Mirrors the LangChain HITL middleware shape (action_requests /
 * review_configs) so hosts and clients can share rendering/UI semantics
 * across the langchain ecosystem.
 */

/** Per-tool approval request emitted inside an interrupt payload. */
export interface ToolApprovalRequest {
  /** Stable id of the tool call (matches LangGraph `ToolCall.id`). */
  tool_call_id: string;
  /** Tool name being invoked. */
  name: string;
  /**
   * Arguments the tool is about to be invoked with — already resolved by
   * any `{{tool<i>turn<n>}}` references and any `updatedInput` returned
   * by the firing PreToolUse hook.
   */
  arguments: Record<string, unknown>;
  /**
   * Optional reason the hook supplied for asking (e.g., "destructive
   * filesystem write"). Hosts can render this verbatim.
   */
  description?: string;
}

/** Allowed host-side decisions for a `tool_approval` interrupt. */
export type ToolApprovalDecisionType =
  | 'approve'
  | 'reject'
  | 'edit'
  | 'respond';

/** Per-action review configuration paired with each action_request. */
export interface ToolApprovalReviewConfig {
  /** Tool name (matches the `name` field on the corresponding action_request). */
  action_name: string;
  /**
   * Stable id of the tool call this review_config applies to (matches
   * the `tool_call_id` of the corresponding action_request). Lets a UI
   * map review_configs → action_requests directly when a batch
   * contains the same tool called more than once — by-position
   * mapping breaks down with duplicates.
   */
  tool_call_id: string;
  /** Decisions the host UI is allowed to surface for this action. */
  allowed_decisions: ToolApprovalDecisionType[];
}

/**
 * Resume value the host returns through `Run.resume(decisions)` after a
 * `tool_approval` interrupt. One entry per action_request, in the same
 * order. Hosts may also return a record keyed by `tool_call_id`; the SDK
 * handles either shape.
 *
 * Variants:
 *   - `approve`: run the tool with its original (or hook-rewritten) args.
 *   - `reject`: skip the tool, emit a blocked error `ToolMessage` with
 *     `reason` surfaced to the model.
 *   - `edit`: replace the tool's args with `updatedInput` (re-resolves
 *     any `{{tool<i>turn<n>}}` placeholders) and run the tool.
 *   - `respond`: skip the tool entirely and emit `responseText` as a
 *     successful `ToolMessage`. Mirrors LangChain HITL middleware's
 *     `respond` semantic — the human supplies the result the model sees,
 *     bypassing tool execution. Useful when the user wants to short-circuit
 *     a tool call with a hand-written answer (e.g., "don't actually run
 *     the search, just tell the model 'no relevant results'").
 *
 * Note on hook semantics: `respond` does NOT fire the per-tool
 * `PostToolUse` hook (no real tool execution happened, so the
 * "post-tool" semantic doesn't apply). It DOES appear in the
 * `PostToolBatch` entry array with `status: 'success'` and the
 * user-supplied text as `toolOutput`, so batch-level audit /
 * convention hooks see the full set of outcomes.
 */
export type ToolApprovalDecision =
  | { type: 'approve' }
  | { type: 'reject'; reason?: string }
  | { type: 'edit'; updatedInput: Record<string, unknown> }
  | { type: 'respond'; responseText: string };

/** Map form of resume decisions, keyed by tool call id. */
export type ToolApprovalDecisionMap = Record<string, ToolApprovalDecision>;

/**
 * Categories of human-in-the-loop interrupts the SDK can raise. Hosts
 * narrow on `HumanInterruptPayload.type` to determine which payload
 * shape they're handling and which resume value to send back through
 * `Run.resume()`.
 *
 * Exported as a discrete type so downstream consumers (notably
 * LibreChat's wire types in `librechat-data-provider`) can mirror
 * the discriminator alongside their own host-side `PendingAction`
 * record without re-declaring the union themselves. Internal SDK
 * code narrows directly on the literal strings via the type guards
 * below; this type alias is primarily an integration-layer contract.
 */
export type HumanInterruptType = 'tool_approval' | 'ask_user_question';

/** Identifies an interrupt that originated inside a checkpointed subagent. */
export interface SubagentInterruptScope {
  /** Child execution run id used by subagent update and usage events. */
  run_id: string;
  /** Child agent id that owns the interrupted tool call. */
  agent_id: string;
  /** Configured subagent type selected by the parent tool call. */
  subagent_type: string;
  /** Parent `subagent` tool call that launched this child. */
  parent_tool_call_id?: string;
}

/**
 * Structured payload the SDK passes to `interrupt()` when one or more
 * pending tool calls require host approval. All `ask`-decision tool calls
 * from a single ToolNode batch are bundled into one interrupt so the host
 * can render and resolve them together.
 *
 * Resume value: `ToolApprovalDecision[]` (in `action_requests` order) or
 * `ToolApprovalDecisionMap` (keyed by `tool_call_id`).
 */
export interface ToolApprovalInterruptPayload {
  type: 'tool_approval';
  action_requests: ToolApprovalRequest[];
  review_configs: ToolApprovalReviewConfig[];
  /** Hook-registry session whose policy raised this interrupt. */
  hook_session_id?: string;
  /** Present when the approval request was bridged from a child graph. */
  subagent?: SubagentInterruptScope;
}

/**
 * Pre-defined option the user can pick when answering an
 * `ask_user_question` interrupt. The selected option's `value` becomes
 * the resume value's `answer` field.
 */
export interface AskUserQuestionOption {
  /** Human-readable label rendered in the host UI. */
  label: string;
  /** Value returned via `AskUserQuestionResolution.answer` if picked. */
  value: string;
}

/** Question request emitted inside an `ask_user_question` interrupt. */
export interface AskUserQuestionRequest {
  /** The question to ask the human. */
  question: string;
  /** Optional context / description rendered alongside the question. */
  description?: string;
  /**
   * Optional pre-defined response options. When present, hosts can render
   * a picker; the user may still type a free-form answer when the host
   * UI allows it. Omit to require a free-form answer.
   */
  options?: AskUserQuestionOption[];
  /**
   * When `true`, the host UI may let the user pick several options; the
   * resulting `AskUserQuestionResolution.answer` is the selected option
   * values joined by `", "`. When omitted or `false`, hosts render a
   * single-select picker. Only meaningful alongside `options`.
   */
  multiSelect?: boolean;
}

/** One independently answerable question in a batched question request. */
export interface AskUserQuestionBatchItem extends AskUserQuestionRequest {
  /** Batch-unique identifier (`[A-Za-z][A-Za-z0-9_-]{0,63}`). */
  id: string;
  /** Optional short heading rendered above the question. */
  header?: string;
}

/** Input shape for one tool call that asks one to four questions together. */
export interface AskUserQuestionsRequest {
  questions: AskUserQuestionBatchItem[];
}

/**
 * Structured payload the SDK passes to `interrupt()` when an agent (or
 * a custom node) needs to ask the user a clarifying question. Mirrors
 * Claude Code's `AskUserQuestion` semantic. Resume value is
 * `AskUserQuestionResolution` for a single question, or
 * `AskUserQuestionsResolution` when `questions` is present.
 */
export interface AskUserQuestionInterruptPayload {
  type: 'ask_user_question';
  /**
   * Single-question request, or the first question as a compatibility
   * fallback when `questions` contains a batch. This lets existing hosts show
   * a useful preview during a staged rollout, but they must support `questions`
   * and `AskUserQuestionsResolution` before enabling a batched tool schema.
   */
  question: AskUserQuestionRequest;
  /** One to four questions collected by one `ask_user_question` tool call. */
  questions?: AskUserQuestionsRequest['questions'];
  /**
   * The `tool_call_id` of the ask-tool call that raised this interrupt,
   * when the tool body supplied it (see `askUserQuestion`'s `options`).
   * Lets hosts attribute the question — and later the answer — to the
   * exact tool-call content part instead of guessing by position, which
   * mislabels cards when a model emits several ask calls in one turn.
   */
  tool_call_id?: string;
}

/** Batch-specialized ask payload for hosts that render several questions. */
export interface AskUserQuestionsInterruptPayload
  extends AskUserQuestionInterruptPayload {
  questions: AskUserQuestionsRequest['questions'];
}

/**
 * Discriminated union of every interrupt payload the SDK raises. New
 * variants can be added without breaking existing handlers as long as
 * those handlers check `payload.type` before reading variant-specific
 * fields. Use the `isToolApprovalInterrupt` / `isAskUserQuestionInterrupt`
 * type guards for ergonomic narrowing.
 */
export type HumanInterruptPayload =
  | ToolApprovalInterruptPayload
  | AskUserQuestionInterruptPayload;

/** Resume value the host returns for an `ask_user_question` interrupt. */
export interface AskUserQuestionResolution {
  /**
   * The human's answer. Free-form text, or — when `options` were
   * provided — one of the option `value`s (or, when the request set
   * `multiSelect`, several option `value`s joined by `", "`). Hosts may
   * also send any structured object their custom UI defines; see the
   * host docs for what your downstream consumer expects.
   */
  answer: string;
}

/** Resume value for a batched `ask_user_question` interrupt. */
export interface AskUserQuestionsResolution {
  /** Human answers keyed by each `AskUserQuestionBatchItem.id`. */
  answers: Record<string, string>;
}

/**
 * Type guard narrowing an arbitrary value to a `ToolApprovalInterruptPayload`.
 * Accepts `unknown` (not just `HumanInterruptPayload`) because hosts can
 * raise custom interrupt payloads from custom nodes — `getInterrupt()`
 * surfaces them as-is, and downstream code must validate the shape at
 * runtime before reading variant-specific fields.
 */
export function isToolApprovalInterrupt(
  payload: unknown
): payload is ToolApprovalInterruptPayload {
  return (
    typeof payload === 'object' &&
    payload !== null &&
    (payload as { type?: unknown }).type === 'tool_approval'
  );
}

/**
 * Type guard narrowing an arbitrary value to an
 * `AskUserQuestionInterruptPayload`. Same `unknown`-tolerant contract
 * as `isToolApprovalInterrupt`.
 */
export function isAskUserQuestionInterrupt(
  payload: unknown
): payload is AskUserQuestionInterruptPayload {
  return (
    typeof payload === 'object' &&
    payload !== null &&
    (payload as { type?: unknown }).type === 'ask_user_question'
  );
}

/**
 * Run-level configuration controlling HITL semantics. **HITL is OFF by
 * default** for now — the SDK ships the interrupt machinery, but the
 * default stays opt-in until host UIs (notably LibreChat) ship the
 * approval-rendering affordances needed to surface interrupts to end
 * users. Without that UI, an interrupt with no resolver looks like a
 * hung tool-call card. Hosts opt in explicitly with
 * `{ enabled: true }`. The intent is to flip this default to ON in a
 * future minor once the consumer ecosystem is ready to render
 * interrupts end-to-end.
 *
 * When enabled (`{ enabled: true }`):
 *
 *   - `PreToolUse` hooks returning `decision: 'ask'` raise a real
 *     LangGraph `interrupt()` instead of being treated as a synchronous
 *     deny.
 *   - `Run.create` installs a `MemorySaver` checkpointer fallback on the
 *     run's compile options if the host did not provide one, since
 *     LangGraph requires a checkpointer to suspend and resume.
 *
 * When disabled (the default — omitted, or `{ enabled: false }`):
 * `ask` decisions are fail-closed (blocked with an error
 * `ToolMessage`) and no checkpointer is implicitly attached. This
 * matches the pre-HITL behavior so existing hosts upgrading the SDK
 * see no change until they're ready to wire the resume UI.
 *
 * ## Scope: every tool the ToolNode runs
 *
 * The interrupt path is wired into both `dispatchToolEvents` (the
 * event-driven path) and `runDirectToolWithLifecycleHooks` (the
 * direct path used by `directToolNames` entries — graph-managed
 * handoff/subagent tools and every in-process `graphTool` instance).
 * `PreToolUse` hooks fire for every tool the ToolNode invokes, and
 * HITL approval gates every tool whose hook returns `'ask'` —
 * regardless of whether the tool is dispatched as an event or
 * invoked in-process. This convergence happened in two follow-up
 * commits to the original HITL surface (see `Graph.ts` —
 * `hookRegistry`/`humanInTheLoop` are passed in both
 * event-driven and legacy branches; and `ToolNode.runDirectToolWithLifecycleHooks`
 * — direct-path tools build their own single-tool `tool_approval`
 * payload and raise `interrupt()` the same way the event path does).
 *
 * Practical implications:
 *   - Every host gets the full HITL surface across every tool the
 *     model calls — event-dispatched, direct, mixed.
 *   - `createToolPolicyHook` and `createWorkspacePolicyHook` apply
 *     uniformly. A hook can be registered without knowing or caring
 *     which path the tool will take.
 *   - Direct tools that the host opted into via `directToolNames` no
 *     longer bypass policy. If you need a tool to skip the hook
 *     surface entirely, omit it from any registered matcher.
 *
 * ## Resume re-execution: every tool in the interrupted batch
 *
 * LangGraph rolls back to the start of the interrupted node on
 * resume. That means **every tool in the same batch as the one that
 * interrupted re-runs from the top on the resume pass**, not just
 * the interrupting tool, and not just the direct half (this used to
 * be framed as a direct-tool-specific concern; it is not — it
 * applies to event-dispatched siblings too). Practical contract:
 *
 *   - The body of the interrupting tool itself runs **once** total
 *     (the first pass interrupted *before* the body, the resume pass
 *     ran the body after the host's decision was applied).
 *   - The body of any sibling tool that already executed in the
 *     same batch before the interrupting tool runs **twice** — once
 *     on the first pass, once on the resume pass.
 *   - `PreToolUse` hooks fire **once per pass per tool**. A hook
 *     that always returns `'ask'` will loop forever on resume; real
 *     hooks should be deterministic w.r.t. inputs and use the
 *     `'ask' → host approves → resume → hook returns 'allow'`
 *     pattern, where the second-pass `allow` reflects the host
 *     having recorded the approval (e.g., a session-scoped approved-
 *     paths set keyed by `runId`).
 *
 * Consequence: any tool with side effects MUST be idempotent if
 * there's any chance another tool in the same batch could trigger
 * an interrupt. This applies equally to direct tools (handoffs,
 * subagents) and to event tools.
 *
 * ### Guarding non-idempotent siblings via `interruptingToolNames`
 *
 * The "must be idempotent" rule above is unavoidable in the general
 * case, but the SDK can protect siblings against the one interrupt
 * shape it can predict: a tool whose *body* raises `interrupt()`
 * mid-execution — the `ask_user_question` shape, where the tool
 * suspends the run to collect a human answer. Declare such tools in
 * `RunConfig.interruptingToolNames`
 * ({@link ToolNodeOptions.interruptingToolNames}) and the ToolNode
 * schedules them, within each batch, **ahead of** their
 * non-interrupting direct siblings. When one interrupts, the batch
 * unwinds before any declared-safe sibling has run, so the sibling
 * executes exactly once (on resume) instead of twice. Empirically:
 *
 *   - A **direct** sibling sharing the interrupter's in-process
 *     `Promise.all` is the only shape that double-executes; declaring
 *     the interrupter closes it.
 *   - An **event-dispatched** sibling is already safe without any
 *     config: the ToolNode awaits the whole direct group (where the
 *     body interrupt unwinds) before it dispatches event tools, so a
 *     dispatched sibling never runs on the first pass.
 *
 * This is a *scheduling* guard, not full resume idempotency: it only
 * covers tools that interrupt from their own body and only protects
 * siblings scheduled after them. It does not retroactively make a
 * `PreToolUse` `'ask'` gate on tool B stop tool A (already executed)
 * from re-running — unless B is itself declared interrupting, so it
 * runs first. Tools with side effects should still be written
 * idempotent as defense in depth.
 *
 * The guard only REORDERS the direct group — declaring a name does not
 * force it onto the direct path. The interrupting tool must already be a
 * real in-process graphTool (the only kind whose body can reach
 * `interrupt()`). A name that resolves to a schema-only event stub (an
 * inherited `toolDefinition` with no executable instance, e.g. in a
 * self-spawned child that scrubs `graphTools`) stays event-dispatched
 * and the ordering is a no-op for it.
 *
 * ## Note on idempotency
 *
 * Same root cause as the resume re-execution above: LangGraph
 * re-runs the interrupted node from the start on resume, which
 * fires `PreToolUse` hooks again. Hooks that produce side effects
 * (logging, external calls) will see at least two invocations per
 * paused turn — exactly two for the interrupting tool, possibly
 * more across siblings.
 */
export interface HumanInTheLoopConfig {
  /**
   * Master switch. Defaults to `false` — omit the field (or pass
   * `false`) to keep HITL off, or set `true` to opt in once the host
   * UI is ready to render and resolve `tool_approval` interrupts.
   */
  enabled?: boolean;
  /**
   * Background children cannot pause for human input. `deny` (the default)
   * lets them run but blocks tools whose hooks request approval, and excludes
   * `ask_user_question`. `reject` preserves the legacy admission check.
   * Durable deferred approvals require a separate resume implementation.
   */
  backgroundPausePolicy?: 'deny' | 'reject';
  /** @internal Set only on detached children and their descendants, never on the parent run. */
  backgroundDeny?: true;
}

/** True only on an SDK-created background child, not a foreground HITL run. */
export function isBackgroundDenyMode(
  config: HumanInTheLoopConfig | undefined
): boolean {
  return (
    config?.backgroundDeny === true &&
    config.enabled === false &&
    config.backgroundPausePolicy === 'deny'
  );
}

/**
 * Snapshot of an in-flight interrupt surfaced from `Run.processStream`
 * via `run.getInterrupt()`. Hosts persist this alongside their job
 * record so they can later call `Run.resume(decisions)` against a Run
 * compiled with the same `thread_id` / checkpointer.
 *
 * The `payload` type defaults to `HumanInterruptPayload` (the SDK's
 * built-in `tool_approval` / `ask_user_question` discriminated union)
 * for ergonomic narrowing in the common case. Hosts that raise custom
 * interrupt payloads from custom graph nodes can pass the type
 * parameter (`run.getInterrupt<MyCustom>()` or
 * `RunInterruptResult<MyCustom>`) — the SDK does not validate the
 * runtime shape, it just transports whatever the node passed to
 * `interrupt()`. Use the `isToolApprovalInterrupt` /
 * `isAskUserQuestionInterrupt` guards (which accept `unknown`) when
 * the source of the interrupt isn't statically known.
 */
export interface RunInterruptResult<TPayload = HumanInterruptPayload> {
  /** Stable id of the LangGraph interrupt (from `Interrupt.id`). */
  interruptId: string;
  /** `thread_id` the run was bound to — required to resume. */
  threadId?: string;
  /** LangGraph checkpoint id that contains the paused interrupt task. */
  checkpointId?: string;
  /** LangGraph checkpoint namespace for the paused interrupt task. */
  checkpointNs?: string;
  /** Structured payload describing what needs human input. */
  payload: TPayload;
}

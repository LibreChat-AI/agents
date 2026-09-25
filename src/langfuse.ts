import { tool } from '@langchain/core/tools';
import { CallbackHandler } from '@langfuse/langchain';
import { LangfuseOtelContextKeys } from '@langfuse/core';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { isGraphInterrupt, isParentCommand } from '@langchain/langgraph';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import {
  getLangfuseTracerProvider,
  propagateAttributes,
} from '@langfuse/tracing';
import type {
  AIMessageChunkFields,
  AIMessageFields,
  UsageMetadata,
} from '@langchain/core/messages';
import type {
  ChatGeneration,
  Generation,
  LLMResult,
} from '@langchain/core/outputs';
import type { PropagateAttributesParams } from '@langfuse/tracing';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { Context, SpanContext } from '@opentelemetry/api';
import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import {
  resolveLangfuseConfigForSpan,
  resolveLangfuseScopeAgentId,
  resolveLangfuseScopeRunId,
  resolveTraceIdSeedForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  resolveToolOutputTracingConfig,
  hasLangfuseEnvConfig,
} from '@/langfuseConfig';
import {
  getLangfuseManagedSpanDestination,
  registerLangfuseManagedSpan,
  resolveLangfuseDestinationKey,
} from '@/langfuseSpanRegistry';
import {
  getToolObservationMetadata,
  LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
} from '@/langfuseToolOutputTracing';
import {
  readPreemptRestartedRun,
  PREEMPT_RESTART_CONTROL_FLOW,
} from '@/llm/preempt';
import { filterCallbacks, findCallback } from '@/utils/callbacks';
import { isPresent, parseBooleanEnv } from '@/utils/misc';

export {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
};

const TRACE_METADATA_MAX_LENGTH = 200;
const LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'LANGFUSE_FORCE_FLUSH_ON_DISPOSE';
const GRAPH_INTERRUPT_CONTROL_FLOW = { controlFlow: 'GraphInterrupt' } as const;
const GRAPH_INTERRUPT_TOOL_OUTPUT = JSON.stringify(
  GRAPH_INTERRUPT_CONTROL_FLOW
);

export type LangfuseTraceMetadata = Record<string, string>;
export type LangfuseTraceAttributes = Record<string, string | number | boolean>;
type LangfuseMetadata = NonNullable<t.LangfuseConfig['metadata']>;
type LangfuseConfigTraceAttributes = NonNullable<
  t.LangfuseConfig['librechatTraceAttributes']
>;

type LangfuseHandlerParams = {
  userId?: string;
  sessionId?: string;
  traceMetadata?: LangfuseTraceMetadata;
  tags?: string[];
  traceIdSeed?: string;
  /** Opaque key used by the span processor to capture this run's parents. */
  traceAnchor?: object;
  /** Agent lane this handler owns when ambient callback scope is unavailable. */
  agentId?: string;
  /** Exported observation that auxiliary work should be nested beneath. */
  parentSpanContext?: SpanContext;
  /** Keep an existing parent trace's user, session, name, and metadata. */
  inheritTraceIdentity?: boolean;
  /** Identity of the run this handler traces; ambient runtime scopes are
   *  only adopted when stamped with the same run (see
   *  `LangfuseRuntimeContext.runId`). */
  runId?: string;
  /** Keep this root open while one processStream call executes graph segments. */
  deferRootRunId?: string;
  /** The run's resolved tool-output policy — for multi-agent streams the
   *  conservative aggregate across agents, which `this.langfuse` (the
   *  primary agent's config) cannot reproduce. Applied when a foreign
   *  scope's policy is rejected. */
  toolOutputTracing?: ResolvedLangfuseToolOutputTracingConfig;
  /** The run's propagated trace name, re-propagated when a foreign scope's
   *  attributes are cleared. */
  traceName?: string;
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
};

type HandlerIdentity = {
  userId?: string;
  sessionId?: string;
  tags?: string[];
  metadata?: LangfuseTraceMetadata;
  traceName?: string;
};

type LangfuseAttributeParams = AgentLangfuseHandlerParams & {
  traceName?: string;
};

type FlushableTracerProvider = {
  forceFlush?: () => Promise<void> | void;
};

type BedrockResponseUsage = {
  inputTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
};

type BedrockResponseMetadata = {
  metadata?: {
    usage?: BedrockResponseUsage;
  };
};

function getLangfuseBedrockUsage(
  message: AIMessage | AIMessageChunk
): UsageMetadata | undefined {
  const usageMetadata = message.usage_metadata;
  const bedrockUsage = (message.response_metadata as BedrockResponseMetadata)
    .metadata?.usage;
  if (
    usageMetadata == null ||
    bedrockUsage == null ||
    usageMetadata.input_tokens !== bedrockUsage.inputTokens
  ) {
    return usageMetadata;
  }

  const cacheRead = bedrockUsage.cacheReadInputTokens ?? 0;
  const cacheCreation = bedrockUsage.cacheWriteInputTokens ?? 0;
  if (cacheRead === 0 && cacheCreation === 0) {
    return usageMetadata;
  }

  return {
    ...usageMetadata,
    input_tokens: usageMetadata.input_tokens + cacheRead + cacheCreation,
  };
}

function cloneMessageWithUsage(
  message: AIMessage | AIMessageChunk,
  usageMetadata: UsageMetadata
): AIMessage | AIMessageChunk {
  const fields: AIMessageFields = {
    content: message.content,
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
    id: message.id,
    name: message.name,
    tool_calls: message.tool_calls,
    invalid_tool_calls: message.invalid_tool_calls,
    usage_metadata: usageMetadata,
  };

  if (message instanceof AIMessageChunk) {
    const chunkFields: AIMessageChunkFields = {
      ...fields,
      tool_call_chunks: message.tool_call_chunks,
    };
    return new AIMessageChunk(chunkFields);
  }

  return new AIMessage(fields);
}

function normalizeGenerationForLangfuse(generation: Generation): Generation {
  if (!('message' in generation)) {
    return generation;
  }

  const message = (generation as ChatGeneration).message;
  if (!(message instanceof AIMessage || message instanceof AIMessageChunk)) {
    return generation;
  }

  const usageMetadata = getLangfuseBedrockUsage(message);
  if (usageMetadata == null || usageMetadata === message.usage_metadata) {
    return generation;
  }

  const chatGeneration: ChatGeneration = {
    ...(generation as ChatGeneration),
    message: cloneMessageWithUsage(message, usageMetadata),
  };
  return chatGeneration;
}

function normalizeBedrockUsageForLangfuse(output: LLMResult): LLMResult {
  if (output.generations.length === 0) {
    return output;
  }

  const listIndex = output.generations.length - 1;
  const generationList = output.generations[listIndex];
  if (generationList.length === 0) {
    return output;
  }

  const generationIndex = generationList.length - 1;
  const generation = generationList[generationIndex];
  const normalized = normalizeGenerationForLangfuse(generation);
  if (normalized === generation) {
    return output;
  }

  const generations = [...output.generations];
  generations[listIndex] = [...generationList];
  generations[listIndex][generationIndex] = normalized;
  return { ...output, generations };
}

const LANGGRAPH_NODE_METADATA_KEY = 'langgraph_node';
/** Explicit agent identity in invoke metadata. Every identity-stamping
 *  component (graph model path, ToolNode, summarization node) overwrites the
 *  canonical `agentId` at its own invoke, so spread order guarantees the
 *  closest stamper wins — key priority alone could not (an inherited key of
 *  either casing can name the wrong agent). `agent_id` remains a fallback
 *  for third-party graphs that only stamp the snake-case form. */
const AGENT_ID_METADATA_KEYS = ['agentId', 'agent_id'];
const LANGGRAPH_NODE_AGENT_PREFIXES = ['agent=', 'tools=', 'summarize='];

/** The LangGraph node a callback executes under, from its inherited
 *  `langgraph_node` run metadata. `undefined` when the callback carries no
 *  node identity. */
function getCallbackNode(
  metadata?: Record<string, unknown>
): string | undefined {
  const node = metadata?.[LANGGRAPH_NODE_METADATA_KEY];
  return typeof node === 'string' && node !== '' ? node : undefined;
}

/** Whether a callback's node identifies the given agent. The outer workflow
 *  node carries the agent id VERBATIM — including ids that themselves begin
 *  with an internal prefix (an agent literally named `agent=research`) — so
 *  an exact match is checked before decoding the inner subgraph prefixes
 *  (`agent=` / `tools=` / `summarize=`). */
function callbackNodeMatchesAgent(node: string, agentId: string): boolean {
  if (node === agentId) {
    return true;
  }
  for (const prefix of LANGGRAPH_NODE_AGENT_PREFIXES) {
    if (node.startsWith(prefix) && node.slice(prefix.length) === agentId) {
      return true;
    }
  }
  return false;
}

/**
 * Hosts often execute agent code inside their own OpenTelemetry spans (HTTP
 * server auto-instrumentation on the global provider). Root observations must
 * not inherit that ambient identity: the foreign parent is never exported to
 * Langfuse, which orphans the trace root (root-observation input/output shaping is
 * skipped because the span no longer looks like a root), collapses concurrent
 * runs inside one request context — an agent run and the previous turn's
 * title run — into a single merged trace with racing names and unioned tags,
 * and bypasses the seeded deterministic trace id generator. Only a
 * Langfuse-managed span bound to the same export destination as the starting
 * run is a safe parent — that is the sanctioned way for hosts to group runs
 * under their own Langfuse observations; a managed span from a different
 * destination (another tenant's project) would leave this run's trace
 * dangling in its own destination while inheriting the other trace's id.
 */
function detachForeignAmbientSpan(
  activeContext: Context,
  destinationKey?: string
): Context {
  const activeSpan = otelTrace.getSpan(activeContext);
  if (activeSpan == null) {
    return activeContext;
  }
  const parentDestination = getLangfuseManagedSpanDestination(activeSpan);
  if (parentDestination != null && parentDestination === destinationKey) {
    return activeContext;
  }
  return otelTrace.deleteSpan(activeContext);
}

class ScopedLangfuseCallbackHandler extends CallbackHandler {
  private readonly langfuse?: t.LangfuseConfig;
  private readonly traceIdSeed?: string;
  private readonly traceAnchor?: object;
  private readonly agentId?: string;
  private readonly parentSpanContext?: SpanContext;
  private readonly runId?: string;
  private readonly deferredRootRunId?: string;
  private readonly identity: HandlerIdentity;
  private readonly toolOutputTracing?: ResolvedLangfuseToolOutputTracingConfig;
  private readonly trackedRunIds = new Set<string>();
  private deferredRootStarted = false;
  private deferredRootOutcome:
    | {
        type: 'end';
        output: Parameters<CallbackHandler['handleChainEnd']>[0];
        parentRunId?: string;
      }
    | { type: 'error'; error: Error; parentRunId?: string }
    | undefined;

  constructor(params?: AgentLangfuseHandlerParams) {
    const {
      langfuse,
      traceIdSeed,
      traceAnchor,
      agentId,
      parentSpanContext,
      inheritTraceIdentity,
      runId,
      deferRootRunId,
      toolOutputTracing,
      traceName,
      ...handlerParams
    } = params ?? {};
    super({
      ...handlerParams,
      ...(inheritTraceIdentity === true
        ? {
          userId: undefined,
          sessionId: undefined,
          traceMetadata: undefined,
        }
        : {}),
    });
    this.langfuse = langfuse;
    this.traceIdSeed = traceIdSeed;
    this.traceAnchor = traceAnchor;
    this.agentId = agentId;
    this.parentSpanContext = parentSpanContext;
    this.runId = runId;
    this.deferredRootRunId = deferRootRunId;
    if (deferRootRunId != null) {
      this.awaitHandlers = true;
    }
    this.toolOutputTracing = toolOutputTracing;
    this.identity = {
      userId: inheritTraceIdentity === true ? undefined : handlerParams.userId,
      sessionId:
        inheritTraceIdentity === true ? undefined : handlerParams.sessionId,
      tags: handlerParams.tags,
      metadata:
        inheritTraceIdentity === true ? undefined : handlerParams.traceMetadata,
      traceName: inheritTraceIdentity === true ? undefined : traceName,
    };
  }

  private getDeterministicTraceSeed(): string | undefined {
    return this.langfuse?.deterministicTraceId === true
      ? this.traceIdSeed
      : undefined;
  }

  private applyExplicitParent(
    activeContext: Context,
    langfuse?: t.LangfuseConfig
  ): Context {
    if (this.parentSpanContext == null) {
      return activeContext;
    }
    const destinationKey = resolveLangfuseDestinationKey(langfuse);
    if (destinationKey == null) {
      return activeContext;
    }
    const parentSpan = otelTrace.wrapSpanContext(this.parentSpanContext);
    registerLangfuseManagedSpan(parentSpan, destinationKey);
    return otelTrace.setSpan(activeContext, parentSpan);
  }

  /**
   * Mirrors the base handler's `runMap`: a start callback whose `parentRunId`
   * this handler never observed gets no explicit parent span and falls back
   * to the ambient OTEL context (`startAndRegisterOtelSpan`), so it needs the
   * same foreign-span detachment as a true root. This happens whenever a
   * handler is attached mid-graph — e.g. the per-agent handler created for a
   * detached subagent's model invocations, whose surrounding graph runs were
   * never traced.
   */
  private startsDetachedRun(runId: string, parentRunId?: string): boolean {
    const detached =
      parentRunId == null || !this.trackedRunIds.has(parentRunId);
    this.trackedRunIds.add(runId);
    return detached;
  }

  /**
   * Whether the ambient runtime scope belongs to a different run — or, for
   * per-agent overlay scopes, to a different concurrently executing agent of
   * the same run than the one this callback reports via its inherited
   * `langgraph_node` metadata. Unstamped scopes on unstamped handlers are
   * never foreign (host-managed handler semantics).
   */
  private isForeignScope(
    scopeRunId: string | undefined,
    scopeAgentId: string | undefined,
    callbackMetadata?: Record<string, unknown>
  ): boolean {
    if (this.runId == null || scopeRunId == null) {
      return false;
    }
    if (scopeRunId !== this.runId) {
      return true;
    }
    if (scopeAgentId == null) {
      return false;
    }
    // Explicit agent identity (stamped into invoke metadata by the graph's
    // model path and ToolNode) is unambiguous; node names are a fallback —
    // an agent literally named `agent=research` makes its outer node
    // indistinguishable from agent `research`'s inner model node.
    for (const key of AGENT_ID_METADATA_KEYS) {
      const explicitAgentId = callbackMetadata?.[key];
      if (typeof explicitAgentId === 'string' && explicitAgentId !== '') {
        return explicitAgentId !== scopeAgentId;
      }
    }
    const callbackNode = getCallbackNode(callbackMetadata);
    return (
      callbackNode != null &&
      !callbackNodeMatchesAgent(callbackNode, scopeAgentId)
    );
  }

  /**
   * LangChain executes non-awaited callbacks on a process-wide background
   * queue (`consumeCallback`), so this callback may be running inside a
   * DIFFERENT concurrent run's async context. The ambient runtime scope is
   * therefore only adopted when it belongs to this handler's run (and, for
   * agent sub-scopes, to this callback's agent): scopes are stamped at their
   * call sites, and a foreign stamp means the scope's config would route
   * spans to the wrong destination, its seed would collapse this run's spans
   * into the foreign trace, and its tool-output policy could leak output the
   * foreign run permits but this run redacts — so config, seed, AND redaction
   * policy all fall back to this handler's own run. Unstamped scopes on
   * unstamped handlers keep scope-first semantics (agent overlays and
   * per-path seeds like the title/label scopes, host-managed handlers).
   *
   * Detached runs (roots, or starts whose parent this handler never tracked)
   * take their span parent from the ambient OTEL context, so drop any
   * foreign ambient span first — a run launched from a host's instrumented
   * request context must start its own trace, not join an unexported
   * foreign one.
   */
  private withRuntimeContext<T>(
    action: () => T,
    isDetachedRun = false,
    callbackMetadata?: Record<string, unknown>
  ): T {
    const currentContext = otelContext.active();
    const scopeRunId = resolveLangfuseScopeRunId(currentContext);
    const scopeAgentId = resolveLangfuseScopeAgentId(currentContext);
    if (this.isForeignScope(scopeRunId, scopeAgentId, callbackMetadata)) {
      return this.withForeignScopeRejected(action);
    }
    const langfuse =
      resolveLangfuseConfigForSpan(currentContext) ?? this.langfuse;
    const parentedContext = isDetachedRun
      ? this.applyExplicitParent(currentContext, langfuse)
      : currentContext;
    const activeContext = isDetachedRun
      ? detachForeignAmbientSpan(
        parentedContext,
        resolveLangfuseDestinationKey(langfuse)
      )
      : parentedContext;
    const scoped = (): T =>
      withLangfuseRuntimeScope(
        {
          langfuse,
          traceIdSeed:
            resolveTraceIdSeedForSpan(activeContext) ??
            this.getDeterministicTraceSeed(),
          traceAnchor: this.traceAnchor,
          runId: scopeRunId ?? this.runId,
          agentId: scopeAgentId ?? this.agentId,
        },
        action
      );
    return activeContext === currentContext
      ? scoped()
      : otelContext.with(activeContext, scoped);
  }

  /**
   * A foreign concurrent run's context must be replaced wholesale, not
   * merged: this library's scope keys (config, seed, tool-output policy,
   * identity stamps) via the replace-mode runtime scope, `@langfuse/tracing`'s
   * propagated trace attributes (userId, sessionId, tags, metadata, …) by
   * deleting their context keys and re-propagating this handler's own
   * identity, and the foreign active span — removed both so detached runs
   * root their own trace and so `propagateAttributes` cannot stamp this
   * run's identity onto the foreign run's still-recording span.
   */
  private withForeignScopeRejected<T>(action: () => T): T {
    let cleanContext = otelTrace.deleteSpan(otelContext.active());
    for (const key of Object.values(LangfuseOtelContextKeys)) {
      cleanContext = cleanContext.deleteValue(key);
    }
    cleanContext = this.applyExplicitParent(cleanContext, this.langfuse);
    const scoped = (): T =>
      withLangfuseRuntimeScope(
        {
          langfuse: this.langfuse,
          traceIdSeed: this.getDeterministicTraceSeed(),
          traceAnchor: this.traceAnchor,
          runId: this.runId,
          agentId: this.agentId,
          toolOutputTracing:
            this.toolOutputTracing ??
            resolveToolOutputTracingConfig(this.langfuse),
        },
        action,
        { replace: true }
      );
    const { userId, sessionId, tags, metadata, traceName } = this.identity;
    const hasIdentity =
      userId != null ||
      sessionId != null ||
      metadata != null ||
      traceName != null ||
      (tags?.length ?? 0) > 0;
    return otelContext.with(cleanContext, () =>
      hasIdentity
        ? propagateAttributes(
          { userId, sessionId, tags, metadata, traceName },
          scoped
        )
        : scoped()
    );
  }

  // LangChain may invoke callback handlers outside the caller's OTEL context.
  // Re-enter tenant scope only for callbacks that start Langfuse observations;
  // end/error/token callbacks use spans already bound to a processor at start.
  override handleChainStart(
    ...args: Parameters<CallbackHandler['handleChainStart']>
  ): ReturnType<CallbackHandler['handleChainStart']> {
    const [, , runId, parentRunId] = args;
    if (
      runId === this.deferredRootRunId &&
      parentRunId == null &&
      this.deferredRootStarted
    ) {
      return Promise.resolve();
    }
    if (runId === this.deferredRootRunId && parentRunId == null) {
      this.deferredRootStarted = true;
    }
    return this.withRuntimeContext(
      () => super.handleChainStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }

  override handleChainError(
    ...args: Parameters<CallbackHandler['handleChainError']>
  ): ReturnType<CallbackHandler['handleChainError']> {
    const [error, runId, parentRunId] = args;
    if (runId === this.deferredRootRunId && parentRunId == null) {
      this.deferredRootOutcome = {
        type: 'error',
        error: error instanceof Error ? error : new Error(String(error)),
      };
      return Promise.resolve();
    }
    if (error != null && parentRunId != null && isGraphInterrupt(error)) {
      return super.handleChainEnd(
        GRAPH_INTERRUPT_CONTROL_FLOW,
        runId,
        parentRunId
      );
    }
    if (error != null && parentRunId != null && isParentCommand(error)) {
      return super.handleChainEnd(
        { controlFlow: 'ParentCommand' },
        runId,
        parentRunId
      );
    }
    return super.handleChainError(...args);
  }

  override handleChainEnd(
    ...args: Parameters<CallbackHandler['handleChainEnd']>
  ): ReturnType<CallbackHandler['handleChainEnd']> {
    const [output, runId, parentRunId] = args;
    if (runId === this.deferredRootRunId && parentRunId == null) {
      this.deferredRootOutcome = { type: 'end', output };
      return Promise.resolve();
    }
    return super.handleChainEnd(...args);
  }

  async finishDeferredRoot(): Promise<void> {
    const runId = this.deferredRootRunId;
    const outcome = this.deferredRootOutcome;
    if (!this.deferredRootStarted || runId == null || outcome == null) {
      return;
    }
    this.deferredRootStarted = false;
    this.deferredRootOutcome = undefined;
    if (outcome.type === 'error') {
      await super.handleChainError(outcome.error, runId, outcome.parentRunId);
      return;
    }
    await super.handleChainEnd(outcome.output, runId, outcome.parentRunId);
  }

  override handleAgentAction(
    ...args: Parameters<CallbackHandler['handleAgentAction']>
  ): ReturnType<CallbackHandler['handleAgentAction']> {
    return this.withRuntimeContext(
      () => super.handleAgentAction(...args),
      this.startsDetachedRun(args[1], args[2])
    );
  }

  override handleGenerationStart(
    ...args: Parameters<CallbackHandler['handleGenerationStart']>
  ): ReturnType<CallbackHandler['handleGenerationStart']> {
    return this.withRuntimeContext(
      () => super.handleGenerationStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleChatModelStart(
    ...args: Parameters<CallbackHandler['handleChatModelStart']>
  ): ReturnType<CallbackHandler['handleChatModelStart']> {
    return this.withRuntimeContext(
      () => super.handleChatModelStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleLLMStart(
    ...args: Parameters<CallbackHandler['handleLLMStart']>
  ): ReturnType<CallbackHandler['handleLLMStart']> {
    return this.withRuntimeContext(
      () => super.handleLLMStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string
  ): Promise<void> {
    return super.handleLLMEnd(
      normalizeBedrockUsageForLangfuse(output),
      runId,
      parentRunId
    );
  }

  /**
   * A cooperative restart tears the provider stream down mid-flight, so the
   * adapter reports cancellation and LangChain closes the generation through
   * this path. That is control flow, not a failure — the graph catches it,
   * injects, and calls the model again — so it closes as a successful
   * generation rather than polluting an otherwise healthy trace with a failed
   * one.
   *
   * Keyed on the run id the tear-down recorded, not on the error's shape:
   * every provider adapter raises its own cancellation error, and matching on
   * those would make the invariant depend on which provider served the call.
   */
  override handleLLMError(
    ...args: Parameters<CallbackHandler['handleLLMError']>
  ): ReturnType<CallbackHandler['handleLLMError']> {
    const [, runId, parentRunId] = args;
    const restarted = readPreemptRestartedRun(runId);
    if (restarted != null) {
      /**
       * Closed WITH the discarded turn, not as an empty end. The provider
       * consumed the whole prompt and may have billed reasoning tokens before
       * the tear-down, and this is the only close the generation will get —
       * the synthetic `CHAT_MODEL_END` that follows is a host stream event and
       * cannot reopen it. Routed through the override so Bedrock usage is
       * normalized exactly as it is for an ordinary end.
       *
       * The generation text stays empty on purpose: a discarded turn produced
       * no answer, and replaying its reasoning as output would read as one.
       *
       * The lookup does not consume the record: a run can be closed through
       * several handlers at once, and each must reach this branch rather than
       * the first one leaving the others to export an error.
       */
      const generation: ChatGeneration = {
        text: '',
        message: restarted.message,
      };
      return this.handleLLMEnd(
        {
          generations: [[generation]],
          llmOutput: PREEMPT_RESTART_CONTROL_FLOW,
        },
        runId,
        parentRunId
      );
    }
    return super.handleLLMError(...args);
  }

  override handleToolStart(
    ...args: Parameters<CallbackHandler['handleToolStart']>
  ): ReturnType<CallbackHandler['handleToolStart']> {
    return this.withRuntimeContext(
      () => super.handleToolStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }

  override handleToolError(
    ...args: Parameters<CallbackHandler['handleToolError']>
  ): ReturnType<CallbackHandler['handleToolError']> {
    const [error, runId, parentRunId] = args;
    if (error != null && parentRunId != null && isGraphInterrupt(error)) {
      return super.handleToolEnd(
        GRAPH_INTERRUPT_TOOL_OUTPUT,
        runId,
        parentRunId
      );
    }
    return super.handleToolError(...args);
  }

  override handleRetrieverStart(
    ...args: Parameters<CallbackHandler['handleRetrieverStart']>
  ): ReturnType<CallbackHandler['handleRetrieverStart']> {
    return this.withRuntimeContext(
      () => super.handleRetrieverStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }
}

function hasLangfuseTracingConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.toolNodeTracing != null ||
    langfuse?.toolOutputTracing != null ||
    langfuse?.privacy != null
  );
}

function hasLangfuseTraceAttributes(langfuse?: t.LangfuseConfig): boolean {
  return (
    Object.keys(createTraceMetadata(langfuse?.metadata ?? {})).length > 0 ||
    Object.keys(
      createLibreChatTraceAttributes(langfuse?.librechatTraceAttributes ?? {})
    ).length > 0 ||
    (mergeLangfuseTags(undefined, langfuse?.tags)?.length ?? 0) > 0
  );
}

function hasLangfuseConfigBaseUrl(langfuse?: t.LangfuseConfig): boolean {
  return isPresent(langfuse?.baseUrl);
}

export function isExplicitLangfuseConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.enabled != null ||
    isPresent(langfuse?.publicKey) ||
    isPresent(langfuse?.secretKey) ||
    isPresent(langfuse?.baseUrl) ||
    hasLangfuseTraceAttributes(langfuse) ||
    hasLangfuseTracingConfig(langfuse)
  );
}

function createTraceMetadata(
  metadata: Record<string, unknown>
): LangfuseTraceMetadata {
  const traceMetadata: LangfuseTraceMetadata = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (value == null) {
      continue;
    }
    const stringValue = typeof value === 'string' ? value : String(value);
    if (
      stringValue.trim() === '' ||
      stringValue.length > TRACE_METADATA_MAX_LENGTH
    ) {
      continue;
    }
    traceMetadata[key] = stringValue;
  }
  return traceMetadata;
}

export function createLibreChatTraceAttributes(
  attributes: LangfuseConfigTraceAttributes
): LangfuseTraceAttributes {
  const librechatTraceAttributes: LangfuseTraceAttributes = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (value == null || key.trim() === '') {
      continue;
    }
    if (typeof value === 'string') {
      if (value.trim() === '' || value.length > TRACE_METADATA_MAX_LENGTH) {
        continue;
      }
      librechatTraceAttributes[key] = value;
      continue;
    }
    librechatTraceAttributes[key] = value;
  }
  return librechatTraceAttributes;
}

export function createLangfuseTraceMetadata({
  messageId,
  parentMessageId,
  agentId,
  agentName,
  rootAgentId,
  rootAgentName,
  activeAgentId,
  activeAgentName,
  endpoint,
  model,
  provider,
  resolvedProvider,
}: {
  messageId?: unknown;
  parentMessageId?: unknown;
  agentId?: unknown;
  agentName?: unknown;
  rootAgentId?: unknown;
  rootAgentName?: unknown;
  activeAgentId?: unknown;
  activeAgentName?: unknown;
  endpoint?: unknown;
  model?: unknown;
  provider?: unknown;
  resolvedProvider?: unknown;
}): LangfuseTraceMetadata {
  return createTraceMetadata({
    messageId,
    parentMessageId,
    agentId,
    agentName,
    rootAgentId,
    rootAgentName,
    activeAgentId,
    activeAgentName,
    endpoint,
    model,
    provider,
    resolvedProvider,
  });
}

function mergeLangfuseTraceMetadata(
  traceMetadata?: LangfuseTraceMetadata,
  metadata?: LangfuseMetadata
): LangfuseTraceMetadata | undefined {
  const merged = createTraceMetadata({
    ...(metadata ?? {}),
    ...(traceMetadata ?? {}),
  });
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function mergeLangfuseTags(
  tags?: string[],
  configTags?: string[]
): string[] | undefined {
  const merged = [...(tags ?? []), ...(configTags ?? [])].filter(
    (tag) => tag.trim() !== ''
  );
  return merged.length > 0 ? [...new Set(merged)] : undefined;
}

/**
 * The trace's user identity: the host-configured `langfuse.userId` when
 * present, else the caller's (normally `configurable.user_id`).
 */
export function resolveLangfuseTraceUserId(
  langfuse: t.LangfuseConfig | undefined,
  userId: string | undefined
): string | undefined {
  const configured = langfuse?.userId?.trim();
  return isPresent(configured) ? configured : userId;
}

export function getLangfuseTraceName(
  traceMetadata?: LangfuseTraceMetadata,
  fallback: string = 'LibreChat Agent'
): string {
  const agentName = traceMetadata?.agentName;
  return isPresent(agentName) ? `${fallback}: ${agentName}` : fallback;
}

export function shouldCreateLangfuseHandler(
  langfuse?: t.LangfuseConfig
): boolean {
  if (langfuse?.enabled === false) {
    return false;
  }
  return (
    hasLangfuseEnvConfig() ||
    hasLangfuseConfigCredentials(langfuse) ||
    (hasLangfuseConfigBaseUrl(langfuse) && hasLangfuseEnvCredentials())
  );
}

export function createLegacyLangfuseHandler(
  params: LangfuseHandlerParams
): CallbackHandler {
  return new ScopedLangfuseCallbackHandler(params);
}

export function createLangfuseHandler({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  tags,
  traceIdSeed,
  traceAnchor,
  agentId,
  parentSpanContext,
  inheritTraceIdentity,
  runId,
  deferRootRunId,
  toolOutputTracing,
  traceName,
}: AgentLangfuseHandlerParams): CallbackHandler | undefined {
  if (!shouldCreateLangfuseHandler(langfuse)) {
    return undefined;
  }
  return new ScopedLangfuseCallbackHandler({
    userId: resolveLangfuseTraceUserId(langfuse, userId),
    sessionId,
    traceMetadata:
      inheritTraceIdentity === true
        ? undefined
        : mergeLangfuseTraceMetadata(traceMetadata, langfuse?.metadata),
    tags: mergeLangfuseTags(tags, langfuse?.tags),
    langfuse,
    traceIdSeed,
    traceAnchor,
    agentId,
    parentSpanContext,
    inheritTraceIdentity,
    runId,
    deferRootRunId,
    toolOutputTracing,
    traceName,
  });
}

function createPropagateAttributeParams({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  traceName,
  tags,
  inheritTraceIdentity,
}: LangfuseAttributeParams): PropagateAttributesParams {
  return {
    userId:
      inheritTraceIdentity === true
        ? undefined
        : resolveLangfuseTraceUserId(langfuse, userId),
    sessionId: inheritTraceIdentity === true ? undefined : sessionId,
    traceName: inheritTraceIdentity === true ? undefined : traceName,
    tags: mergeLangfuseTags(tags, langfuse?.tags),
    metadata:
      inheritTraceIdentity === true
        ? undefined
        : mergeLangfuseTraceMetadata(traceMetadata, langfuse?.metadata),
  };
}

export function withLangfuseAttributes<T>(
  params: LangfuseAttributeParams,
  action: () => T
): T {
  if (!shouldCreateLangfuseHandler(params.langfuse)) {
    return action();
  }
  return propagateAttributes(createPropagateAttributeParams(params), action);
}

export function hasExplicitLangfuseConfig(
  contexts: Iterable<{ langfuse?: t.LangfuseConfig }>
): boolean {
  for (const context of contexts) {
    if (isExplicitLangfuseConfig(context.langfuse)) {
      return true;
    }
  }
  return false;
}

export function isLangfuseCallbackHandler(value: unknown): boolean {
  return value instanceof CallbackHandler;
}

/** Host results have passed host output filters. Record only reserved metadata,
 * not arguments, rows, errors, or attachments. These completion observations
 * do not measure tool latency; execution timing belongs in the metadata. */
export async function traceHostToolResults(
  request: t.ToolExecuteBatchRequest,
  results: t.ToolExecuteResult[],
  config?: RunnableConfig
): Promise<void> {
  if (findCallback(config?.callbacks, isLangfuseCallbackHandler) == null) {
    return;
  }
  const callbacks = filterCallbacks(
    config?.callbacks,
    isLangfuseCallbackHandler
  );
  const callsById = new Map(request.toolCalls.map((call) => [call.id, call]));
  for (const result of results) {
    const metadata = getToolObservationMetadata(result.artifact);
    const call = callsById.get(result.toolCallId);
    if (call == null || Object.keys(metadata).length === 0) {
      continue;
    }
    const failure =
      result.status === 'error'
        ? new Error('Host tool execution failed')
        : undefined;
    const observation = tool(
      async () => {
        if (failure != null) {
          throw failure;
        }
        return LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT;
      },
      {
        name: call.name,
        description: 'Host tool execution metadata',
        schema: { type: 'object', properties: {} },
      }
    );
    try {
      await observation.invoke(
        { type: 'tool_call', id: call.id, name: call.name, args: {} },
        {
          callbacks,
          metadata: { ...metadata, ...request.metadata },
        }
      );
    } catch (error) {
      if (failure == null || error !== failure) {
        throw error;
      }
    }
  }
}

export async function disposeLangfuseHandler(value: unknown): Promise<void> {
  if (value instanceof ScopedLangfuseCallbackHandler) {
    await value.finishDeferredRoot();
  }
  if (
    value == null ||
    parseBooleanEnv(process.env[LANGFUSE_FORCE_FLUSH_ON_DISPOSE]) !== true
  ) {
    return;
  }
  const provider = getLangfuseTracerProvider() as FlushableTracerProvider;
  await provider.forceFlush?.();
}

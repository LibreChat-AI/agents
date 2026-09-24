// src/run.ts
import { nanoid } from 'nanoid';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import {
  Command,
  INTERRUPT,
  MemorySaver,
  Overwrite,
  isInterrupted,
} from '@langchain/langgraph';
import type {
  MessageContentComplex,
  UsageMetadata,
} from '@langchain/core/messages';
import type { StringPromptValue } from '@langchain/core/prompt_values';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import type { StandardGraph } from '@/graphs/Graph';
import type * as t from '@/types';
import {
  Callback,
  GraphEvents,
  TitleMethod,
  ACTIVITY_LABEL_RUN_NAME,
  REASONING_LABEL_RUN_NAME,
  ACTIVITY_PHASE_RUN_NAME,
  ACTIVITY_PHASE_LABEL_RUN_NAME,
  DEFAULT_MAX_STOP_CONTINUATIONS,
  DEFAULT_RECURSION_LIMIT,
} from '@/common';
import {
  ACTIVITY_PHASE_LABEL_PROMPT,
  ACTIVITY_LABEL_PROMPT,
  buildActivityLabelPrompt,
  buildActivityPhaseLabelPrompt,
  normalizeActivityPhaseLabel,
} from '@/prompts/activityLabel';
import {
  createLangfuseTraceMetadata,
  createLangfuseHandler,
  disposeLangfuseHandler,
  getLangfuseTraceName,
  isLangfuseCallbackHandler,
  withLangfuseAttributes,
} from '@/langfuse';
import {
  requireValidSubagentResumeManifest,
  SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY,
  SUBAGENT_RESUME_MANIFEST_CONFIG_KEY,
} from '@/tools/subagent/SubagentReplay';
import {
  REASONING_LABEL_PROMPT,
  buildReasoningLabelTraceSeed,
  buildReasoningLabelPrompt,
  normalizeReasoningLabel,
} from '@/prompts/reasoningLabel';
import {
  TOOL_BATCH_REPLAY_KEY,
  restoreToolReplayConfig,
  stripToolBatchReplayState,
  getPublicToolInterruptPayload,
} from '@/tools/toolBatchReplay';
import {
  hasToolOutputTracingConfig,
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
} from '@/langfuseConfig';
import {
  cloneToolApprovalInterruptPayload,
  TOOL_APPROVAL_REVIEW_CONFIG_KEY,
} from '@/hitl/approvalReview';
import {
  resolveLangfuseDestinationKey,
  resolveLangfuseTraceAnchorParent,
} from '@/langfuseSpanRegistry';
import {
  appendCallbacks,
  filterCallbacks,
  findCallback,
  type CallbackEntry,
} from '@/utils/callbacks';
import {
  resolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  getRunStepResumeState,
  stripRunStepResumeState,
} from '@/tools/runStepResume';
import {
  createCompletionTitleRunnable,
  createTitleRunnable,
} from '@/utils/title';
import { applyGraphRuntimeConfig } from '@/graphs/applyGraphRuntimeConfig';
import { LANGFUSE_OPERATION_METADATA_KEY } from '@/langfuseOperation';
import { createTokenCounter, encodingForModel } from '@/utils/tokens';
import { stampSyntheticProviderMessage } from '@/messages/provenance';
import { isOpenAILike, isLibreChatOpenAIModel } from '@/utils/llm';
import { executeHooks, mergeAggregatedHookResults } from '@/hooks';
import { convertInjectedMessages } from '@/messages/injected';
import { initializeLangfuseTracing } from './instrumentation';
import { seedRunInitialSessions } from '@/utils/toolSessions';
import { getTraceIdSeed } from '@/langfuseRuntimeContext';
import { resolveClientOptionsModel } from '@/llm/request';
import { HandoffLimitError } from '@/graphs/handoff';
import { createGraph } from '@/graphs/createGraph';
import { isFadingTier } from '@/messages/fading';
import { resolveMaxSeals } from '@/llm/preempt';
import { isBuiltRuntime } from '@/lazyRequire';
import { initializeModel } from '@/llm/init';
import { HandlerRegistry } from '@/events';

/** Source-mode runs have no dist siblings for the lazy-loading seam, so every
 *  lazily loadable module is imported through the active loader before the
 *  first model resolves. The built package never enters this branch. */
async function ensureSourceModeProviders(): Promise<void> {
  if (isBuiltRuntime()) {
    return;
  }
  await import('@/llm/providers.eager');
}

export const defaultOmitOptions = new Set([
  'stream',
  'thinking',
  'streaming',
  'maxTokens',
  'clientOptions',
  'thinkingConfig',
  'thinkingBudget',
  'includeThoughts',
  'maxOutputTokens',
  'additionalModelRequestFields',
]);

const ACTIVITY_LABEL_TRACE_NAME = 'LibreChat Activity Label';
const ACTIVITY_PHASE_TRACE_NAME = 'LibreChat Activity Phase';
const REASONING_LABEL_TRACE_NAME = 'LibreChat Reasoning Label';
const OUTPUT_TRUNCATED_HALT_REASON = 'output_truncated';

function resolveMaxStopContinuations(value: number | undefined): number {
  if (value == null || !Number.isFinite(value)) {
    return DEFAULT_MAX_STOP_CONTINUATIONS;
  }
  return Math.max(0, Math.floor(value));
}

function materializeStopContinuation(
  result: AggregatedHookResult
): BaseMessage[] {
  const messages: BaseMessage[] = [];
  const contexts = result.additionalContexts.filter(
    (context) => context.trim() !== ''
  );
  if (contexts.length > 0) {
    messages.push(
      stampSyntheticProviderMessage(
        new HumanMessage({
          content: contexts.join('\n\n'),
          additional_kwargs: {
            role: 'system',
            isMeta: true,
            source: 'hook',
          },
        })
      )
    );
  }
  messages.push(...convertInjectedMessages(result.injectedMessages));
  return messages;
}

function advanceCheckpointCursor(config: t.RunStreamConfig): t.RunStreamConfig {
  const configurable = { ...config.configurable };
  delete configurable.checkpoint_id;
  delete configurable.checkpoint_map;
  return { ...config, configurable };
}

function assertFinalAdmissionSucceeded(result: AggregatedHookResult): void {
  if (result.hasHookFailures !== true && result.errors.length === 0) {
    return;
  }
  const detail =
    result.errors.length > 0
      ? result.errors.join('; ')
      : 'one or more internal finalizers failed';
  throw new Error(`StopFinalize terminal admission failed: ${detail}`);
}

const CUSTOM_GRAPH_EVENTS = new Set<string>([
  GraphEvents.ON_AGENT_UPDATE,
  GraphEvents.ON_RUN_STEP,
  GraphEvents.ON_RUN_STEP_DELTA,
  GraphEvents.ON_RUN_STEP_COMPLETED,
  GraphEvents.ON_RUN_STEP_CLOSED,
  GraphEvents.ON_MESSAGE_DELTA,
  GraphEvents.ON_REASONING_DELTA,
  GraphEvents.ON_TOOL_EXECUTE,
  GraphEvents.ON_SUMMARIZE_START,
  GraphEvents.ON_SUMMARIZE_DELTA,
  GraphEvents.ON_SUMMARIZE_COMPLETE,
  GraphEvents.ON_SUBAGENT_UPDATE,
  GraphEvents.ON_AGENT_LOG,
  GraphEvents.ON_CONTEXT_USAGE,
  GraphEvents.ON_CUSTOM_EVENT,
]);

const DIRECT_DISPATCHED_STEP_EVENTS = new Set<string>([
  GraphEvents.ON_RUN_STEP,
  GraphEvents.ON_RUN_STEP_DELTA,
  GraphEvents.ON_RUN_STEP_CLOSED,
  GraphEvents.ON_MESSAGE_DELTA,
  GraphEvents.ON_REASONING_DELTA,
]);

function getStepScopedEventId(data: unknown): string | undefined {
  if (data == null || typeof data !== 'object') {
    return undefined;
  }
  const candidate = data as { id?: unknown };
  return typeof candidate.id === 'string' ? candidate.id : undefined;
}

/**
 * Narrows an ON_RUN_STEP_COMPLETED payload (`{ result: ToolCompleteEvent }`)
 * to the ids step closure needs. Returns undefined for malformed payloads and
 * the resume race's empty step id.
 */
function getToolCompletion(
  data: unknown
): { stepId: string; toolCallId?: string; completedAt?: number } | undefined {
  if (data == null || typeof data !== 'object') {
    return undefined;
  }
  const { result } = data as { result?: unknown };
  if (result == null || typeof result !== 'object') {
    return undefined;
  }
  const candidate = result as {
    id?: unknown;
    tool_call?: { id?: unknown };
    completed_at?: unknown;
  };
  if (typeof candidate.id !== 'string' || candidate.id === '') {
    return undefined;
  }
  const toolCallId = candidate.tool_call?.id;
  const completedAt = candidate.completed_at;
  return {
    stepId: candidate.id,
    ...(typeof toolCallId === 'string' ? { toolCallId } : {}),
    ...(typeof completedAt === 'number' ? { completedAt } : {}),
  };
}

function isLangGraphResumeMapForInterrupt(
  value: unknown,
  interruptId: string
): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(value, interruptId);
}

function getInterruptHookSessionId(payload: unknown): string | undefined {
  const publicPayload = getPublicToolInterruptPayload(payload);
  if (
    publicPayload == null ||
    typeof publicPayload !== 'object' ||
    (publicPayload as { type?: unknown }).type !== 'tool_approval'
  ) {
    return undefined;
  }
  const sessionId = (publicPayload as { hook_session_id?: unknown })
    .hook_session_id;
  return typeof sessionId === 'string' && sessionId.length > 0
    ? sessionId
    : undefined;
}

type InterruptStateSnapshot = {
  config?: RunnableConfig;
  values?: {
    handoffState?: t.HandoffState;
    messages?: BaseMessage[];
    runStepState?: t.RunStepResumeState;
  };
  tasks?: Array<{
    interrupts?: Array<{ id?: string; value?: unknown }>;
    state?: RunnableConfig | InterruptStateSnapshot;
  }>;
};

type WorkflowWithStateHistory = {
  getState?(
    config: RunnableConfig,
    options?: { subgraphs?: boolean }
  ): Promise<InterruptStateSnapshot>;
  getStateHistory?(
    config: RunnableConfig,
    options?: { filter?: Record<string, unknown>; limit?: number }
  ): AsyncIterableIterator<InterruptStateSnapshot>;
};

async function resolveCompletedSegmentConfig(
  workflow: t.CompiledStateWorkflow,
  config: t.RunStreamConfig,
  executionId: string,
  streamSegment: number
): Promise<t.RunStreamConfig> {
  const stateWorkflow = workflow as t.CompiledStateWorkflow &
    WorkflowWithStateHistory;
  const stateHistory = stateWorkflow.getStateHistory;
  if (typeof stateHistory !== 'function') {
    throw new Error(
      'Cannot continue a checkpointed run without exact state history.'
    );
  }
  const lookup = advanceCheckpointCursor(config);

  const resolveSnapshot = (
    snapshot: InterruptStateSnapshot | undefined
  ): t.RunStreamConfig | undefined => {
    if (snapshot == null) {
      return undefined;
    }
    const runStepState =
      snapshot.values?.runStepState ??
      snapshot.tasks
        ?.flatMap((task) => task.interrupts ?? [])
        .map((pendingInterrupt) =>
          getRunStepResumeState(pendingInterrupt.value)
        )
        .find((state): state is t.RunStepResumeState => state != null);
    if (
      runStepState?.stopContinuationExecutionId !== executionId ||
      runStepState.streamSegment !== streamSegment
    ) {
      return undefined;
    }
    const checkpointId = snapshot.config?.configurable?.checkpoint_id;
    if (typeof checkpointId !== 'string' || checkpointId.length === 0) {
      return undefined;
    }
    return {
      ...config,
      configurable: {
        ...config.configurable,
        ...snapshot.config?.configurable,
      },
    };
  };

  if (typeof stateWorkflow.getState === 'function') {
    const latest = await stateWorkflow.getState(lookup);
    const latestConfig = resolveSnapshot(latest);
    if (latestConfig != null) {
      return latestConfig;
    }
  }

  for await (const snapshot of stateHistory.call(workflow, lookup)) {
    const snapshotConfig = resolveSnapshot(snapshot);
    if (snapshotConfig != null) {
      return snapshotConfig;
    }
  }
  throw new Error(
    'Cannot identify the checkpoint committed by the completed graph segment.'
  );
}

function getFirstPersistedInterrupt(
  snapshot: InterruptStateSnapshot
): { id: string; value: unknown } | undefined {
  for (const task of snapshot.tasks ?? []) {
    for (const pendingInterrupt of task.interrupts ?? []) {
      if (
        typeof pendingInterrupt.id === 'string' &&
        pendingInterrupt.id.length > 0
      ) {
        return { id: pendingInterrupt.id, value: pendingInterrupt.value };
      }
    }
    const nestedState = task.state;
    if (
      nestedState != null &&
      'tasks' in nestedState &&
      Array.isArray(nestedState.tasks)
    ) {
      const nestedInterrupt = getFirstPersistedInterrupt(nestedState);
      if (nestedInterrupt != null) {
        return nestedInterrupt;
      }
    }
  }
  return undefined;
}

function getPersistedMessages(
  snapshot: InterruptStateSnapshot
): BaseMessage[] | undefined {
  const messages = snapshot.values?.messages;
  if (!Array.isArray(messages) || !messages.every(BaseMessage.isInstance)) {
    return undefined;
  }
  return messages;
}

type ResumeCommandUpdate = ConstructorParameters<typeof Command>[0]['update'];

function overwriteResumeRunStepState(
  command: Command,
  state: t.RunStepResumeState
): Command {
  const overwrite = new Overwrite(state);
  const update = Array.isArray(command.update)
    ? [
      ...command.update.filter(([key]) => key !== 'runStepState'),
        ['runStepState', overwrite] as [string, unknown],
    ]
    : { ...(command.update ?? {}), runStepState: overwrite };
  const resumed = new Command({
    ...(command.graph == null ? {} : { graph: command.graph }),
    ...(command.resume === undefined ? {} : { resume: command.resume }),
    update,
    ...(command.goto === undefined ? {} : { goto: command.goto }),
  });
  resumed.lc_direct_tool_output = command.lc_direct_tool_output;
  return resumed;
}

function getResumeUpdateMessages(
  update: ResumeCommandUpdate
): BaseMessage[] | undefined {
  if (update == null) {
    return undefined;
  }
  const messages = Array.isArray(update)
    ? update.find(([key]) => key === 'messages')?.[1]
    : update.messages;
  if (BaseMessage.isInstance(messages)) {
    return [messages];
  }
  if (!Array.isArray(messages) || !messages.every(BaseMessage.isInstance)) {
    return undefined;
  }
  return messages;
}

export class Run<_T extends t.BaseGraphState> {
  id: string;
  private tokenCounter?: t.TokenCounter;
  private handlerRegistry?: HandlerRegistry;
  private hookRegistry?: HookRegistry;
  private humanInTheLoop?: t.HumanInTheLoopConfig;
  private langfuse?: t.LangfuseConfig;
  private toolOutputReferences?: t.ToolOutputReferencesConfig;
  private eagerEventToolExecution?: t.EagerEventToolExecutionConfig;
  private codeSessionToolNames?: string[];
  private interruptingToolNames?: string[];
  private toolExecution?: t.ToolExecutionConfig;
  private subagentUsageSink?: t.SubagentUsageSink;
  private preemption?: t.StreamPreemption;
  private maxStopContinuations: number;
  private streamLimits?: t.StreamLimits;
  private subagentTasks?: t.SubagentTaskConfig;
  private subagentContext?: t.SubagentContextAdapter;
  private indexTokenCountMap?: Record<string, number>;
  calibrationRatio: number = 1;
  fadingTier?: t.FadingTier;
  fadingTiers: t.FadingTiers = {};
  private fadingTierReset: boolean = false;
  private fadingTierResetAgentIds: string[] = [];
  graphRunnable?: t.CompiledStateWorkflow;
  Graph: StandardGraph | MultiAgentGraph | undefined;
  returnContent: boolean = false;
  private skipCleanup: boolean = false;
  /**
   * Whether the compiled graph was built with a checkpointer (host-supplied
   * or the HITL `MemorySaver` fallback). Captured at graph creation because
   * the constructor can later overwrite `Graph.compileOptions` with the raw
   * caller options, dropping the fallback checkpointer from that metadata.
   */
  private hasCheckpointer: boolean = false;
  private _streamResult: t.MessageContentComplex[] | undefined;
  /**
   * Captured interrupt payload typed as `unknown` because the SDK
   * does not validate the runtime shape — custom graph nodes can
   * raise interrupts with arbitrary payloads (not just the SDK's
   * `HumanInterruptPayload` union). The public `getInterrupt<T>()`
   * lets callers assert the type they expect.
   */
  private _interrupt: t.RunInterruptResult<unknown> | undefined;
  /** Per-run sequence for batch-unique activity-label trace-seed fallbacks. */
  private activityLabelSeq = 0;
  /** Per-run sequence for parent activity-phase trace and invocation ids. */
  private activityPhaseLabelSeq = 0;
  /** Per-run sequence for reasoning-label trace and invocation ids. */
  private reasoningLabelSeq = 0;
  /** Latest user turn used to keep detached phase roots conversation-shaped. */
  private activityPhaseTraceInput?: string;
  /** Distinguishes sibling forks started from the same explicit checkpoint. */
  private checkpointForkSeq = 0;
  private _haltedReason: string | undefined;
  private _handoffOutcome?: t.HandoffOutcome;

  private constructor(config: Partial<t.RunConfig>) {
    const runId = config.runId ?? '';
    if (!runId) {
      throw new Error('Run ID not provided');
    }

    this.id = runId;
    this.tokenCounter = config.tokenCounter;
    this.indexTokenCountMap = config.indexTokenCountMap;
    if (config.calibrationRatio != null && config.calibrationRatio > 0) {
      this.calibrationRatio = config.calibrationRatio;
    }
    if (isFadingTier(config.fadingTier)) {
      this.fadingTier = { ...config.fadingTier };
    }
    if (config.fadingTiers != null) {
      this.fadingTiers = Object.fromEntries(
        Object.entries(config.fadingTiers).flatMap(([agentId, tier]) =>
          isFadingTier(tier) ? [[agentId, { ...tier }] as const] : []
        )
      );
    }

    const handlerRegistry = new HandlerRegistry();

    if (config.customHandlers) {
      for (const [eventType, handler] of Object.entries(
        config.customHandlers
      )) {
        handlerRegistry.register(eventType, handler);
      }
    }

    this.handlerRegistry = handlerRegistry;
    this.hookRegistry = config.hooks;
    this.humanInTheLoop = config.humanInTheLoop;
    this.langfuse = config.langfuse;
    this.toolOutputReferences = config.toolOutputReferences;
    this.eagerEventToolExecution = config.eagerEventToolExecution;
    this.codeSessionToolNames = config.codeSessionToolNames;
    this.interruptingToolNames = config.interruptingToolNames;
    this.toolExecution = config.toolExecution;
    this.subagentUsageSink = config.subagentUsageSink;
    this.subagentTasks = config.subagentTasks;
    this.subagentContext = config.subagentContext;
    this.preemption = config.preemption;
    this.maxStopContinuations = resolveMaxStopContinuations(
      config.maxStopContinuations
    );
    this.streamLimits = config.streamLimits;

    if (!config.graphConfig) {
      throw new Error('Graph config not provided');
    }

    /** Handle different graph types */
    if (config.graphConfig.type === 'multi-agent') {
      this.graphRunnable = this.createMultiAgentGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.handlerRegistry = handlerRegistry;
      }
    } else {
      /** Default to legacy graph for 'standard' or undefined type */
      this.graphRunnable = this.createLegacyGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.handlerRegistry = handlerRegistry;
      }
    }

    if (config.initialSessions && this.Graph) {
      const configuredAgents =
        'agents' in config.graphConfig &&
        Array.isArray(config.graphConfig.agents)
          ? config.graphConfig.agents
          : undefined;
      const agents: Array<Pick<t.AgentInputs, 'codeSessionKey'>> =
        configuredAgents ?? [
          {
            codeSessionKey:
              'codeSessionKey' in config.graphConfig
                ? config.graphConfig.codeSessionKey
                : undefined,
          },
        ];
      seedRunInitialSessions({
        sessions: this.Graph.sessions,
        initialSessions: config.initialSessions,
        agents,
      });
    }

    this.returnContent = config.returnContent ?? false;
    this.skipCleanup = config.skipCleanup ?? false;
  }

  private createLegacyGraph(
    config: t.LegacyGraphConfig | t.StandardGraphConfig
  ): t.CompiledStateWorkflow {
    let agentConfig: t.AgentInputs;
    let signal: AbortSignal | undefined;

    /** Check if this is a multi-agent style config (has agents array) */
    if ('agents' in config && Array.isArray(config.agents)) {
      if (config.agents.length === 0) {
        throw new Error('At least one agent must be provided');
      }
      agentConfig = config.agents[0];
      signal = config.signal;
    } else {
      /** Legacy path: build agent config from llmConfig */
      const {
        type: _type,
        llmConfig,
        signal: legacySignal,
        tools = [],
        ...agentInputs
      } = config as t.LegacyGraphConfig;
      const { provider, ...clientOptions } = llmConfig;

      agentConfig = {
        ...agentInputs,
        tools,
        provider,
        clientOptions,
        agentId: 'default',
      } as t.AgentInputs;
      signal = legacySignal;
    }

    const standardGraph = createGraph({
      kind: 'standard',
      input: {
        signal,
        runId: this.id,
        agents: [agentConfig],
        langfuse: this.langfuse,
        tokenCounter: this.tokenCounter,
        indexTokenCountMap: this.indexTokenCountMap,
        calibrationRatio: this.calibrationRatio,
        fadingTier: this.fadingTier,
        fadingTiers: this.fadingTiers,
        subagentUsageSink: this.subagentUsageSink,
        subagentTasks: this.subagentTasks,
        subagentContext: this.subagentContext,
        preemption: this.preemption,
        streamLimits: this.streamLimits,
        toolExecution: this.toolExecution,
      },
    });
    /** Propagate compile options from graph config */
    standardGraph.compileOptions = this.applyHITLCheckpointerFallback(
      config.compileOptions
    );
    this.hasCheckpointer = standardGraph.compileOptions?.checkpointer != null;
    applyGraphRuntimeConfig(standardGraph, {
      hookRegistry: this.hookRegistry,
      humanInTheLoop: this.humanInTheLoop,
      toolOutputReferences: this.toolOutputReferences,
      eagerEventToolExecution: this.eagerEventToolExecution,
      codeSessionToolNames: this.codeSessionToolNames,
      interruptingToolNames: this.interruptingToolNames,
      toolExecution: this.toolExecution,
    });
    this.Graph = standardGraph;
    return standardGraph.createWorkflow();
  }

  private createMultiAgentGraph(
    config: t.MultiAgentGraphConfig
  ): t.CompiledStateWorkflow {
    const { agents, edges, compileOptions, entryAgentId, maxHandoffs } = config;

    const multiAgentGraph = createGraph({
      kind: 'multi-agent',
      input: {
        runId: this.id,
        agents,
        edges,
        entryAgentId,
        maxHandoffs,
        langfuse: this.langfuse,
        tokenCounter: this.tokenCounter,
        indexTokenCountMap: this.indexTokenCountMap,
        calibrationRatio: this.calibrationRatio,
        fadingTier: this.fadingTier,
        fadingTiers: this.fadingTiers,
        subagentUsageSink: this.subagentUsageSink,
        subagentTasks: this.subagentTasks,
        subagentContext: this.subagentContext,
        preemption: this.preemption,
        streamLimits: this.streamLimits,
        toolExecution: this.toolExecution,
      },
    });

    multiAgentGraph.compileOptions =
      this.applyHITLCheckpointerFallback(compileOptions);
    this.hasCheckpointer = multiAgentGraph.compileOptions?.checkpointer != null;

    applyGraphRuntimeConfig(multiAgentGraph, {
      hookRegistry: this.hookRegistry,
      humanInTheLoop: this.humanInTheLoop,
      toolOutputReferences: this.toolOutputReferences,
      eagerEventToolExecution: this.eagerEventToolExecution,
      codeSessionToolNames: this.codeSessionToolNames,
      interruptingToolNames: this.interruptingToolNames,
      toolExecution: this.toolExecution,
    });
    this.Graph = multiAgentGraph;
    return multiAgentGraph.createWorkflow();
  }

  /**
   * When the host opted into HITL via `humanInTheLoop: { enabled: true }`
   * and did not supply a checkpointer, install an in-memory `MemorySaver`
   * so `interrupt()` can persist checkpoints and `Command({ resume })`
   * can rebuild state. The fallback is intentionally process-local:
   * production hosts that need durable resumption across processes /
   * restarts must provide their own checkpointer (Redis, Postgres, etc.)
   * on `compileOptions.checkpointer`.
   *
   * No-op when HITL is off (the default — omitted, or
   * `{ enabled: false }`) or the host already supplied a checkpointer
   * of their own. See `HumanInTheLoopConfig` JSDoc for the rationale
   * behind the default-off stance.
   */
  private applyHITLCheckpointerFallback(
    compileOptions: t.CompileOptions | undefined
  ): t.CompileOptions | undefined {
    if (this.humanInTheLoop?.enabled !== true) {
      return compileOptions;
    }
    if (compileOptions?.checkpointer != null) {
      return compileOptions;
    }
    return {
      ...(compileOptions ?? {}),
      checkpointer: new MemorySaver(),
    };
  }

  /**
   * Run RunStart + UserPromptSubmit hooks before the graph stream
   * begins, accumulate any `additionalContext` strings into the input
   * messages, and short-circuit when a hook signals the run should not
   * proceed (deny / ask decision on the prompt, or `preventContinuation`
   * on either hook).
   *
   * Returns `true` when the caller should bail with `undefined` (run
   * was halted before any model call); returns `false` to proceed
   * into the stream loop.
   *
   * ## Side effects
   *
   * On the success path:
   *   - Mutates `stateInputs.messages` in place to append a
   *     consolidated `HumanMessage` carrying any hook
   *     `additionalContext` strings. Safe because the host owns the
   *     array and `processStream` is the only consumer until LangGraph
   *     reads it.
   *
   * On the halt path (returning `true`):
   *   - Sets `this._haltedReason` so callers (and the eventual host)
   *     can distinguish a hook-driven halt from a natural completion.
   *   - Calls `registry.clearSession(this.id)` and
   *     `registry.clearHaltSignal(this.id)` because no resume is
   *     expected from a pre-stream halt — the run never entered the
   *     graph, so the session/halt state for this run would otherwise
   *     leak to the next `processStream` invocation on the same
   *     registry. Other concurrent runs on the same registry are
   *     untouched (halt signals are scoped per session id).
   *   - Sets `config.callbacks = undefined` to drop the callback
   *     references the caller built (langfuse handler, custom event
   *     handler, etc.) since they won't be exercised. Mirrors the
   *     equivalent cleanup the `processStream` `finally` block does
   *     on the natural-completion path.
   */
  private async runPreStreamHooks(
    stateInputs: t.IState,
    threadId: string | undefined,
    config: Partial<RunnableConfig>
  ): Promise<boolean> {
    const registry = this.hookRegistry;
    /**
     * Defensive guard: `processStream` already validated `this.Graph`
     * before calling this helper, but TypeScript can't propagate that
     * narrowing across method boundaries. The check keeps the body
     * free of `this.Graph!` non-null assertions.
     */
    if (registry == null || this.Graph == null) {
      return false;
    }

    const preStreamContexts: string[] = [];

    const runStartResult = await executeHooks({
      registry,
      input: {
        hook_event_name: 'RunStart',
        runId: this.id,
        threadId,
        agentId: this.Graph.defaultAgentId,
        messages: stateInputs.messages,
      },
      sessionId: this.id,
    });
    for (const ctx of runStartResult.additionalContexts) {
      preStreamContexts.push(ctx);
    }
    /**
     * Honor `preventContinuation` from RunStart before the stream
     * starts. Mid-flight halts (from tool/compact/subagent hooks)
     * route through `HookRegistry.haltRun` and are polled by the
     * stream loop in `processStream` — different mechanism, same
     * intent.
     */
    if (runStartResult.preventContinuation === true) {
      this._haltedReason = runStartResult.stopReason ?? 'preventContinuation';
      registry.clearSession(this.id);
      registry.clearHaltSignal(this.id);
      config.callbacks = undefined;
      return true;
    }

    const lastHuman = findLastMessageOfType(stateInputs.messages, 'human');
    if (lastHuman != null) {
      const promptResult = await executeHooks({
        registry,
        input: {
          hook_event_name: 'UserPromptSubmit',
          runId: this.id,
          threadId,
          agentId: this.Graph.defaultAgentId,
          prompt: extractPromptText(lastHuman),
          // attachments: not yet wired — Phase 2 will extract
          // non-text content blocks (images, files) from messages
        },
        sessionId: this.id,
      });
      if (
        promptResult.decision === 'deny' ||
        promptResult.decision === 'ask' ||
        promptResult.preventContinuation === true
      ) {
        /**
         * Always set `_haltedReason` so the host can call
         * `getHaltReason()` and distinguish a hook-blocked prompt
         * from a natural empty-output completion. Three signals can
         * land here, each with its own canonical reason string when
         * the hook didn't supply one.
         */
        if (promptResult.preventContinuation === true) {
          this._haltedReason = promptResult.stopReason ?? 'preventContinuation';
        } else if (promptResult.decision === 'deny') {
          this._haltedReason = promptResult.reason ?? 'prompt_denied';
        } else {
          this._haltedReason =
            promptResult.reason ?? 'prompt_requires_approval';
        }
        registry.clearSession(this.id);
        registry.clearHaltSignal(this.id);
        config.callbacks = undefined;
        return true;
      }
      for (const ctx of promptResult.additionalContexts) {
        preStreamContexts.push(ctx);
      }
    }

    if (preStreamContexts.length > 0) {
      /**
       * Wraps the joined hook contexts as a `HumanMessage` even though
       * the intent is system-level guidance. Using a `SystemMessage`
       * mid-conversation is rejected by Anthropic and Google providers
       * (system messages must be the leading entry), so the LangChain
       * convention — also used by `ToolNode.convertInjectedMessages`
       * — is `HumanMessage` carrying `additional_kwargs.role` as a
       * marker for hosts inspecting state. The model still sees a
       * user-role message; the `role: 'system'` field is metadata
       * only. Hosts that want a true system message should compose
       * it into the agent's `instructions` config instead.
       */
      stateInputs.messages.push(
        stampSyntheticProviderMessage(
          new HumanMessage({
            content: preStreamContexts.join('\n\n'),
            additional_kwargs: { role: 'system', source: 'hook' },
          })
        )
      );
    }

    return false;
  }

  static async create<T extends t.BaseGraphState>(
    config: t.RunConfig
  ): Promise<Run<T>> {
    await ensureSourceModeProviders();
    /** Create tokenCounter if indexTokenCountMap is provided but tokenCounter is
     *  not. The model is read through both option keys: `modelName` is
     *  LangChain's alias for `model`, so consulting one alone would give a
     *  Claude-backed agent an `o200k_base` counter and undercount every message
     *  it measures. */
    if (config.indexTokenCountMap && !config.tokenCounter) {
      const gc = config.graphConfig;
      const clientOpts =
        'agents' in gc ? gc.agents[0]?.clientOptions : gc.clientOptions;
      const model = resolveClientOptionsModel(clientOpts) ?? '';
      config.tokenCounter = await createTokenCounter(encodingForModel(model));
    }
    return new Run<T>(config);
  }

  getRunMessages(): BaseMessage[] | undefined {
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    return this.Graph.getRunMessages();
  }

  getChildCheckpointThreadIds(): string[] {
    return this.Graph?.getChildCheckpointThreadIds() ?? [];
  }

  /**
   * Returns a defensive snapshot of tools discovered by the current run.
   * Pass an agent id for that context, or omit it for the ordered union across
   * contexts. Interrupted state is available immediately for host persistence;
   * completed runs retain their final snapshot through graph cleanup.
   */
  getDiscoveredTools(agentId?: string): string[] {
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    return this.Graph.getDiscoveredTools(agentId);
  }

  /**
   * Returns the current calibration ratio (EMA of provider-vs-estimate token ratios).
   * Hosts should persist this value and pass it back as `RunConfig.calibrationRatio`
   * on the next run for the same conversation so the pruner starts with an accurate
   * scaling factor instead of the default (1).
   */
  getCalibrationRatio(): number {
    return this.calibrationRatio;
  }

  /**
   * Returns the default agent's latched context-fading tier. Single-agent hosts
   * may persist it beside the calibration ratio; multi-agent hosts should use
   * `getFadingTiers()` so independent agent tiers are retained.
   */
  getFadingTier(): t.FadingTier | undefined {
    return this.fadingTier == null ? undefined : { ...this.fadingTier };
  }

  /** Returns a defensive snapshot of latched tiers keyed by agent ID. */
  getFadingTiers(): t.FadingTiers {
    return Object.fromEntries(
      Object.entries(this.fadingTiers).map(([agentId, tier]) => [
        agentId,
        { ...tier },
      ])
    );
  }

  /** Agent IDs whose canonical history was compacted during this run. */
  getFadingTierResetAgentIds(): string[] {
    return [...this.fadingTierResetAgentIds];
  }

  /** Whether the default agent compacted canonical history during this run. */
  didResetFadingTier(): boolean {
    return this.fadingTierReset;
  }

  getResolvedInstructionOverhead(): number | undefined {
    return this.Graph?.getResolvedInstructionOverhead();
  }

  /**
   * Cooperative-preemption counters for this run. `emptyBoundaries` is the one
   * to watch: it counts SEALS whose `PreemptBoundary` produced nothing to
   * inject, which ends the turn early and leaves the answer unfinished. A
   * restart that injects nothing simply reissues the call, so it is not
   * counted there.
   */
  getPreemptStats(): t.PreemptStats {
    return (
      this.Graph?.getPreemptStats() ?? {
        seals: 0,
        restarts: 0,
        emptyBoundaries: 0,
      }
    );
  }

  getToolCount(): number {
    return this.Graph?.getToolCount() ?? 0;
  }

  /**
   * True when the run's last turn ended at `END` because the provider hit
   * its output token ceiling while producing plain text/reasoning — no tool
   * call, so `assertNotTruncatedToolCall` never sees it and the graph reads
   * the turn as an ordinary completion. Hosts check this alongside
   * `getPreemptStats()` / `getHaltReason()` to decide whether to persist the
   * response as unfinished instead of a silently truncated "complete" one.
   */
  getOutputTruncated(): boolean {
    return this.Graph?.outputTruncatedIncomplete ?? false;
  }

  /**
   * Creates a custom event callback handler that intercepts custom events
   * and processes them through our handler registry instead of EventStreamCallbackHandler
   */
  private createCustomEventCallback() {
    return async (
      eventName: string,
      data: unknown,
      runId: string,
      tags?: string[],
      metadata?: Record<string, unknown>
    ): Promise<void> => {
      // Step-scoped SDK events are dispatched directly via the handler
      // registry first. Skip callback-based echoes to prevent double
      // handling when LangGraph invokes custom callbacks more than once.
      const stepScopedEventId = getStepScopedEventId(data);
      if (
        DIRECT_DISPATCHED_STEP_EVENTS.has(eventName) &&
        this.Graph != null &&
        stepScopedEventId != null &&
        this.Graph.hasHandlerDispatchedEvent(eventName, stepScopedEventId)
      ) {
        return;
      }
      // Accepted results are graph-owned, never inferred from provider/tool callbacks.
      if (
        eventName === GraphEvents.ON_MODEL_RESPONSE ||
        eventName === GraphEvents.ON_MODEL_TOOLS_CLAIMED
      ) return;
      const handler = this.handlerRegistry?.getHandler(eventName);
      /**
       * Tool completions arriving over the custom-event channel are the only
       * signal ToolNode (which holds no graph reference) emits — observe them
       * here to drive step closure. Runs in `finally`, independent of handler
       * registration, so an absent or throwing host handler cannot lose the
       * close; duplicate callback echoes are absorbed by the terminal-status
       * guard in `closeRunStep`.
       */
      try {
        if (handler && this.Graph) {
          return await handler.handle(
            eventName,
            data as
              | t.StreamEventData
              | t.ModelEndData
              | t.RunStep
              | t.RunStepDeltaEvent
              | t.RunStepClosedEvent
              | t.MessageDeltaEvent
              | t.ReasoningDeltaEvent
              | { result: t.ToolEndEvent },
            metadata,
            this.Graph
          );
        }
      } finally {
        if (
          eventName === GraphEvents.ON_RUN_STEP_COMPLETED &&
          this.Graph != null
        ) {
          const completion = getToolCompletion(data);
          if (completion != null) {
            /**
             * The producer stamped `completed_at` before dispatch. Carrying it
             * through keeps the recorded duration the tool's, not the host
             * handler's — this runs after an arbitrarily slow handler resolves.
             *
             * Isolated because this is a `finally`: a throw here would
             * REPLACE an in-flight error from the handler above, so the
             * host's original failure would be reported as whatever went
             * wrong during closure instead. Closing is best-effort at this
             * point either way — the end-of-run sweep still stamps any step
             * this misses.
             */
            try {
              await this.Graph.recordStepCompletion(completion.stepId, {
                toolCallId: completion.toolCallId,
                metadata,
                at: completion.completedAt,
              });
            } catch (_e) {
              /** Never mask the handler's error with a closure failure */
            }
          }
        }
      }
    };
  }

  private shouldClearHookSession(streamThrew: boolean): boolean {
    return this._interrupt == null || this._haltedReason != null || streamThrew;
  }

  private isAwaitingResume(streamThrew: boolean): boolean {
    return (
      this._interrupt != null && this._haltedReason == null && !streamThrew
    );
  }

  /**
   * Terminal status for steps still open at end-of-run: `cancelled` for
   * intentional stops (caller abort, hook halt), `failed` for unexpected
   * stream errors, `completed` for a natural finish. Reads `_haltedReason`
   * behind a method boundary on purpose — it is assigned inside the
   * `consumeStream` closure, which control-flow narrowing cannot see.
   */
  private resolveSweepStatus(
    streamThrew: boolean,
    streamAborted: boolean
  ): Exclude<t.RunStepStatus, 'in_progress'> {
    if (streamThrew) {
      return streamAborted ? 'cancelled' : 'failed';
    }
    if (this._haltedReason != null) {
      return 'cancelled';
    }
    return 'completed';
  }

  private getStreamLangfuseConfig(
    graph: StandardGraph | MultiAgentGraph
  ): t.LangfuseConfig | undefined {
    const primaryContext = graph.agentContexts.get(graph.defaultAgentId);
    if (primaryContext != null) {
      return resolveLangfuseConfig(this.langfuse, primaryContext.langfuse);
    }

    for (const context of graph.agentContexts.values()) {
      const langfuse = resolveLangfuseConfig(this.langfuse, context.langfuse);
      if (langfuse != null) {
        return langfuse;
      }
    }

    return this.langfuse;
  }

  private getStreamToolOutputTracingLangfuseConfig(
    graph: StandardGraph | MultiAgentGraph
  ): t.LangfuseConfig | undefined {
    const toolOutputTracingConfigs = Array.from(graph.agentContexts.values())
      .map((context) => {
        return resolveLangfuseConfig(this.langfuse, context.langfuse)
          ?.toolOutputTracing;
      })
      .filter((config): config is t.LangfuseToolOutputTracingConfig => {
        return config != null;
      });

    if (toolOutputTracingConfigs.length === 0) {
      return this.langfuse?.toolOutputTracing != null
        ? { toolOutputTracing: this.langfuse.toolOutputTracing }
        : undefined;
    }
    if (toolOutputTracingConfigs.length === 1) {
      return { toolOutputTracing: toolOutputTracingConfigs[0] };
    }

    let enabled: boolean | undefined;
    let redactionText: string | undefined;
    let redactedToolNameMatchMode: 'exact' | 'partial' | undefined;
    const redactedToolNames = new Set<string>();

    for (const config of toolOutputTracingConfigs) {
      if (config.enabled === false) {
        enabled = false;
      } else if (enabled !== false && config.enabled != null) {
        enabled = config.enabled;
      }

      redactionText ??= config.redactionText;
      if (config.redactedToolNameMatchMode === 'partial') {
        redactedToolNameMatchMode = 'partial';
      } else {
        redactedToolNameMatchMode ??= config.redactedToolNameMatchMode;
      }

      for (const toolName of config.redactedToolNames ?? []) {
        redactedToolNames.add(toolName);
      }
    }

    return {
      toolOutputTracing: {
        ...(enabled != null ? { enabled } : {}),
        ...(redactedToolNames.size > 0
          ? { redactedToolNames: Array.from(redactedToolNames) }
          : {}),
        ...(redactedToolNameMatchMode != null
          ? { redactedToolNameMatchMode }
          : {}),
        ...(redactionText != null ? { redactionText } : {}),
      },
    };
  }

  async processStream(
    inputs: t.IState | Command,
    callerConfig: t.RunStreamConfig,
    streamOptions?: t.EventStreamOptions
  ): Promise<MessageContentComplex[] | undefined> {
    if (this.graphRunnable == null) {
      throw new Error(
        'Run not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    const graphRunnable = this.graphRunnable;
    const graph = this.Graph;
    this._handoffOutcome = undefined;

    /**
     * `Command` inputs (`Command({ resume, update?, goto? })`) are
     * resume-mode invocations: LangGraph rebuilds graph state from the
     * checkpointer, so we skip RunStart / UserPromptSubmit hooks (no
     * new prompt to evaluate) and read run-state from the Graph wrapper
     * instead of `inputs.messages`.
     */
    const isResume = inputs instanceof Command;
    if (!isResume) {
      graph.startFreshLangfuseExecution();
    }
    const stateInputs = isResume ? undefined : (inputs as t.IState);
    if (stateInputs != null) {
      this.activityPhaseTraceInput = findActivityPhaseTraceInput(
        stateInputs.messages
      );
    }

    /**
     * Every honored seal costs one extra superstep, so a preemption-enabled
     * run reserves headroom for its whole seal budget. Without it, a
     * tool-heavy agent that gets preempted could hit `GraphRecursionError` —
     * which surfaces as a thrown stream, setting `streamThrew`, firing
     * `StopFailure`, and wiping via `clearHeavyState()` exactly the partial
     * content the seal existed to preserve.
     */
    const recursionLimit =
      (callerConfig.recursionLimit ?? DEFAULT_RECURSION_LIMIT) +
      (this.preemption != null ? resolveMaxSeals(this.preemption.maxSeals) : 0);

    const config: t.RunStreamConfig = {
      ...callerConfig,
      recursionLimit,
      configurable: { ...callerConfig.configurable },
    };
    delete config.configurable?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
    delete config.configurable?.[TOOL_BATCH_REPLAY_KEY];
    if (!isResume) {
      delete config.configurable?.[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY];
      delete config.configurable?.[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY];
    }
    let overwriteLegacyResumeState = false;
    if (isResume) {
      await this.restoreInterruptFromCheckpoint(
        config,
        (inputs as Command).update
      );
      const publicPayload = stripRunStepResumeState(this._interrupt?.payload);
      if (config.configurable != null) {
        restoreToolReplayConfig(
          config.configurable,
          this._interrupt?.interruptId,
          publicPayload
        );
      }
      if (graph.getStopContinuationExecutionId() === '') {
        graph.startStopContinuationExecution(nanoid());
        overwriteLegacyResumeState = this.hasCheckpointer;
      }
    }

    /**
     * Cancellation can arrive either at graph construction or per-call through
     * `callerConfig.signal`, and boundary hooks need to observe both — for a
     * multi-agent run the construction signal does not exist at all, since
     * `MultiAgentGraphConfig` exposes none. Carried on its own field, assigned
     * unconditionally: writing into `graph.signal` would leak this call's
     * controller into later calls (model-call config and subagent
     * parentSignal read that field, and `clearHeavyState()` is skipped on
     * HITL interrupts), while a conditional write would keep observing a
     * stale controller the host has since aborted. The boundary dispatch
     * composes both channels; see `dispatchPreemptBoundary`.
     */
    graph.callerSignal = callerConfig.signal;

    /**
     * Skip `resetValues` on resume — we're continuing an in-flight
     * run, not starting a fresh one. Resetting would wipe the
     * sidecars (`toolCallStepIds`, `stepKeyIds`, accumulated
     * `messages`, etc.) the resumed `ToolNode` needs to dispatch
     * tool completions with the correct step ids and re-resolve
     * `{{tool<i>turn<n>}}` references. Pairs with the
     * `awaitingResume` gate on `clearHeavyState` in the `finally`
     * block so the sidecars survive both ends of the interrupt
     * boundary.
     */
    if (!isResume) {
      const checkpointThreadId =
        typeof config.configurable?.thread_id === 'string'
          ? config.configurable.thread_id
          : undefined;
      const checkpointNamespace =
        typeof config.configurable?.checkpoint_ns === 'string'
          ? config.configurable.checkpoint_ns
          : '';
      const checkpointId =
        typeof config.configurable?.checkpoint_id === 'string'
          ? config.configurable.checkpoint_id
          : '';
      const checkpointScope =
        checkpointThreadId == null
          ? undefined
          : JSON.stringify([
            checkpointThreadId,
            checkpointNamespace,
            checkpointId,
            checkpointId === '' ? 0 : ++this.checkpointForkSeq,
          ]);
      graph.resetValues(streamOptions?.keepContent, checkpointScope);
      graph.handoffRouting?.start();
      graph.startStopContinuationExecution(nanoid());
    }
    this._interrupt = undefined;
    this._haltedReason = undefined;
    this.hookRegistry?.clearHaltSignal(this.id);

    /** Custom event callback to intercept and handle custom events */
    const customEventCallback = this.createCustomEventCallback();

    const streamCallbacks = streamOptions?.callbacks
      ? this.getCallbacks(streamOptions.callbacks)
      : undefined;

    const customHandler = BaseCallbackHandler.fromMethods({
      [Callback.CUSTOM_EVENT]: customEventCallback,
    });
    customHandler.awaitHandlers = true;

    config.callbacks = appendCallbacks(
      config.callbacks,
      streamCallbacks ? [streamCallbacks, customHandler] : [customHandler]
    );

    const primaryContext = graph.agentContexts.get(graph.defaultAgentId);
    const userId =
      typeof config.configurable?.user_id === 'string'
        ? config.configurable.user_id
        : undefined;
    const sessionId =
      typeof config.configurable?.thread_id === 'string'
        ? config.configurable.thread_id
        : undefined;
    const traceMetadata = createLangfuseTraceMetadata({
      messageId: this.id,
      parentMessageId: config.configurable?.requestBody?.parentMessageId,
      agentId: graph.defaultAgentId,
      agentName: primaryContext?.name,
      rootAgentId: graph.defaultAgentId,
      rootAgentName: primaryContext?.name,
      activeAgentId: graph.defaultAgentId,
      activeAgentName: primaryContext?.name,
    });
    const traceName = config.runName ?? getLangfuseTraceName(traceMetadata);
    const streamLangfuseConfig = this.getStreamLangfuseConfig(graph);
    initializeLangfuseTracing(streamLangfuseConfig);
    const streamRuntimeScope = resolveLangfuseRuntimeScope({
      runLangfuse: streamLangfuseConfig,
      langfuseOverlay: this.getStreamToolOutputTracingLangfuseConfig(graph),
      traceIdSeed:
        streamLangfuseConfig?.deterministicTraceId === true
          ? this.id
          : undefined,
      traceAnchor: graph.langfuseTraceAnchor,
      // The graph's per-execution stamp, NOT the public run id: public ids
      // may repeat across concurrent executions (retries, tenant-local
      // message ids), and equal stamps defeat foreign-scope rejection.
      runId: graph.langfuseScopeRunId,
    });
    const langfuseHandler = createLangfuseHandler({
      langfuse: streamLangfuseConfig,
      userId,
      sessionId,
      traceMetadata,
      tags: ['librechat', 'agent'],
      traceIdSeed:
        streamLangfuseConfig?.deterministicTraceId === true
          ? this.id
          : undefined,
      traceAnchor: graph.langfuseTraceAnchor,
      runId: graph.langfuseScopeRunId,
      deferRootRunId: this.id,
      // The aggregate multi-agent policy from the runtime scope — the
      // handler must restore THIS (not the primary agent's config-derived
      // policy) when rejecting a foreign scope.
      toolOutputTracing: streamRuntimeScope.toolOutputTracing,
      traceName,
    });
    if (langfuseHandler != null) {
      config.callbacks = appendCallbacks(config.callbacks, [langfuseHandler]);
    }

    if (!this.id) {
      throw new Error('Run ID not provided');
    }

    config.runId = this.id;
    config.run_id = this.id;
    config.configurable = Object.assign(config.configurable ?? {}, {
      run_id: this.id,
    });

    /**
     * Default `durability: 'exit'` whenever a checkpointer is active so
     * runs skip per-superstep checkpoint writes and persist only at the
     * exit/interrupt boundary (all HITL/resume needs). An explicit caller
     * value wins; no checkpointer leaves it unset (langgraph default).
     */
    if (config.durability == null && this.hasCheckpointer) {
      config.durability = 'exit';
    }

    const threadId = config.configurable.thread_id as string | undefined;

    if (this.hookRegistry != null && stateInputs != null) {
      const shouldHalt = await this.runPreStreamHooks(
        stateInputs,
        threadId,
        config
      );
      if (shouldHalt) {
        return undefined;
      }
    }

    /**
     * Tracks whether the stream loop threw. Used by the `finally`
     * block to decide whether to honor the interrupt-preservation
     * guard for session hooks: a captured `_interrupt` is only
     * meaningful if the stream completed cleanly. If the loop errored
     * after stashing an interrupt (e.g. a downstream handler throws
     * after the interrupt event landed), the interrupt is stale —
     * preserving session hooks would leak them into the next run.
     */
    let streamThrew = false;
    let streamAborted = false;
    /**
     * When the stream itself ended — captured before the post-stream work in
     * the `finally` (Stop/StopFailure hooks, Langfuse disposal, which can
     * force-flush) so a slow hook cannot inflate the terminal stamps that the
     * sweep writes onto steps that were still open.
     */
    let terminalAt: number | undefined;

    const consumeStream = async (): Promise<void> => {
      let streamInputs: t.IState | Command = inputs;
      if (!isResume && graph.handoffRouting != null) {
        streamInputs = {
          ...(inputs as t.IState),
          handoffState: graph.handoffRouting.snapshot(),
        };
      }
      if (!isResume && this.hasCheckpointer) {
        streamInputs = {
          ...(streamInputs as t.IState),
          ...(graph.handoffRouting == null
            ? {}
            : { handoffState: new Overwrite(graph.handoffRouting.snapshot()) }),
          runStepState: new Overwrite(graph.createRunStepResumeState()),
        } as unknown as t.IState;
      } else if (overwriteLegacyResumeState) {
        streamInputs = overwriteResumeRunStepState(
          inputs as Command,
          graph.createRunStepResumeState()
        );
      }
      let streamConfig = config;

      for (;;) {
        /**
         * `streamEvents` accepts both state inputs and `Command` (resume) at
         * runtime, but our `CompiledStateWorkflow` type narrows the first
         * arg to `BaseGraphState`. Cast on the call so the resume path
         * type-checks without widening the wrapper for every caller.
         */
        const stream = graphRunnable.streamEvents(
          streamInputs as t.IState,
          { ...streamConfig, runName: graph.runName },
          {
            raiseError: true,
            /** Custom events are handled by createCustomEventCallback(). */
            ignoreCustomEvent: true,
          }
        );

        for await (const event of stream) {
          const { data, metadata, ...info } = event;
          const eventName: t.EventName = info.event;

          if (CUSTOM_GRAPH_EVENTS.has(eventName)) {
            continue;
          }

          if (
            this._interrupt == null &&
            data.chunk != null &&
            isInterrupted<unknown>(data.chunk)
          ) {
            const interrupts = data.chunk[INTERRUPT];
            if (interrupts.length > 0) {
              const first = interrupts[0];
              this._interrupt = {
                interruptId: first.id ?? '',
                threadId,
                payload: cloneToolApprovalInterruptPayload(first.value),
              };
            }
          }

          const modelEndAt =
            eventName === GraphEvents.CHAT_MODEL_END ? Date.now() : undefined;
          if (
            eventName === GraphEvents.ON_MODEL_RESPONSE ||
            eventName === GraphEvents.ON_MODEL_TOOLS_CLAIMED
          ) continue;
          const handler = this.handlerRegistry?.getHandler(eventName);
          if (handler) {
            await handler.handle(eventName, data, metadata, this.Graph);
          }

          if (eventName === GraphEvents.CHAT_MODEL_END && this.Graph != null) {
            await this.Graph.closeOpenMessageStep(metadata, modelEndAt);
          }

          const haltSignal = this.hookRegistry?.getHaltSignal(this.id);
          if (haltSignal != null) {
            this._haltedReason = haltSignal.reason;
            break;
          }
        }

        terminalAt = Date.now();

        if (this._interrupt != null) {
          const interruptConfig = this.hasCheckpointer
            ? advanceCheckpointCursor(streamConfig)
            : streamConfig;
          await this.resolveInterruptResumeConfig(interruptConfig);
          return;
        }
        if (this._haltedReason != null) {
          return;
        }

        let stopReason = graph.preemptHaltReason;
        if (stopReason == null && graph.preemptIncomplete) {
          stopReason = 'preempt_incomplete';
        }
        if (stopReason == null && graph.outputTruncatedIncomplete) {
          stopReason = OUTPUT_TRUNCATED_HALT_REASON;
        }

        const stopMessages =
          graph.getRunMessages() ?? stateInputs?.messages ?? [];
        const stopContinuationCount = graph.getStopContinuationCount();
        const continuationBudgetRemaining = Math.max(
          0,
          this.maxStopContinuations - stopContinuationCount
        );
        let stopResult: AggregatedHookResult | undefined;
        if (this.hookRegistry?.hasHookFor('Stop', this.id) === true) {
          stopResult = await executeHooks({
            registry: this.hookRegistry,
            input: {
              hook_event_name: 'Stop',
              runId: this.id,
              threadId,
              agentId: graph.defaultAgentId,
              messages: stopMessages,
              stopReason,
              stopHookActive: stopContinuationCount > 0,
              continuationCount: stopContinuationCount,
              continuationBudgetRemaining,
            },
            sessionId: this.id,
            signal: config.signal,
          }).catch((): undefined => undefined);
        }

        /** A summarize-only run is complete once its summary exists: there
         *  is no assistant reply for a Stop hook to demand, and its one
         *  summarization claim is spent, so it admits no continuation. */
        const continuationAdmissible = graph.summarizeOnlyAgentId == null;
        if (this.hookRegistry?.hasHookFor('StopFinalize', this.id) === true) {
          const stopInjected =
            stopResult == null ? [] : materializeStopContinuation(stopResult);
          const continuationPrevented =
            !continuationAdmissible ||
            stopReason != null ||
            stopResult?.preventContinuation === true;
          const finalized = await executeHooks({
            registry: this.hookRegistry,
            input: {
              hook_event_name: 'StopFinalize',
              runId: this.id,
              threadId,
              agentId: graph.defaultAgentId,
              messages: stopMessages,
              stopReason,
              stopHookActive: stopContinuationCount > 0,
              continuationCount: stopContinuationCount,
              continuationBudgetRemaining,
              continuationPlanned:
                !continuationPrevented &&
                stopResult?.stopDecision === 'block' &&
                stopInjected.length > 0 &&
                continuationBudgetRemaining > 0,
              continuationPrevented,
            },
            sessionId: this.id,
            signal: config.signal,
          });
          assertFinalAdmissionSucceeded(finalized);
          stopResult = mergeAggregatedHookResults(stopResult, finalized);
        }

        if (stopResult?.preventContinuation === true) {
          this._haltedReason =
            stopResult.stopReason ?? stopResult.reason ?? 'preventContinuation';
          return;
        }

        const requestsContinuation =
          continuationAdmissible &&
          stopReason == null &&
          stopResult?.stopDecision === 'block';
        if (requestsContinuation && stopResult != null) {
          const injected = materializeStopContinuation(stopResult);
          if (injected.length > 0) {
            if (stopContinuationCount >= this.maxStopContinuations) {
              throw new Error(
                'Stop hook attempted to inject messages after the terminal continuation budget was exhausted.'
              );
            }
            const completedSegmentConfig = this.hasCheckpointer
              ? await resolveCompletedSegmentConfig(
                graphRunnable,
                streamConfig,
                graph.getStopContinuationExecutionId(),
                graph.getStreamSegment()
              )
              : streamConfig;
            const nextContinuationCount = stopContinuationCount + 1;
            graph.setStopContinuationCount(nextContinuationCount);
            graph.advanceStreamSegment();
            /**
             * A checkpointer already owns the completed graph state, so only
             * the delta is submitted. Without one, each invocation starts from
             * empty state and must be seeded with the live full transcript.
             * Graph sidecars and the outer Run stay intact in both cases.
             */
            streamInputs = {
              messages: this.hasCheckpointer
                ? injected
                : [...graph.messages, ...injected],
              runStepState: graph.createRunStepResumeState(),
              handoffState: graph.handoffRouting?.snapshot(),
            };
            streamConfig = completedSegmentConfig;
            continue;
          }
        }

        if (graph.preemptHaltReason != null) {
          this._haltedReason = graph.preemptHaltReason;
        } else if (graph.preemptIncomplete) {
          this._haltedReason = 'preempt_incomplete';
        } else if (graph.outputTruncatedIncomplete) {
          this._haltedReason = OUTPUT_TRUNCATED_HALT_REASON;
        }
        return;
      }
    };

    try {
      // When opted in, seed the root trace id from this run's id so feedback /
      // other external signals can be attached to the trace later without a
      // lookup (see SeededTraceIdGenerator in ./instrumentation).
      await withLangfuseRuntimeScope(streamRuntimeScope, () =>
        withLangfuseAttributes(
          {
            langfuse: streamLangfuseConfig,
            userId,
            sessionId,
            traceName,
            traceMetadata,
            tags: ['librechat', 'agent'],
          },
          consumeStream
        )
      );
    } catch (err) {
      terminalAt = Date.now();
      streamThrew = true;
      if (err instanceof HandoffLimitError)
        this._haltedReason = 'handoff_limit';
      await langfuseHandler?.handleChainError(
        err instanceof Error ? err : new Error(String(err)),
        this.id
      );
      /**
       * Corroborate cancellation against an actually-aborted signal. A
       * provider SDK or host handler can reject with an `AbortError` while
       * nothing was cancelled — that is an unexpected failure (it also fires
       * `StopFailure`), and naming it `cancelled` would misreport abort
       * forensics.
       */
      streamAborted =
        config.signal?.aborted === true || this.Graph.signal?.aborted === true;
      if (this.hookRegistry?.hasHookFor('StopFailure', this.id) === true) {
        const runMessages = this.Graph.getRunMessages() ?? [];
        await executeHooks({
          registry: this.hookRegistry,
          input: {
            hook_event_name: 'StopFailure',
            runId: this.id,
            threadId,
            agentId: this.Graph.defaultAgentId,
            error: err instanceof Error ? err.message : String(err),
            lastAssistantMessage: findLastMessageOfType(runMessages, 'ai'),
          },
          sessionId: this.id,
        }).catch(() => {
          /* swallow hook errors — the original error must propagate */
        });
      }
      throw err;
    } finally {
      /**
       * Preserve session-scoped hooks when the run paused on a HITL
       * interrupt — the very next call will be `Run.resume()`, which
       * needs the same policy hooks (e.g., the `PreToolUse` matcher
       * that triggered the interrupt) to fire on the re-executed node
       * and uphold the approval flow. Clearing here would leak the
       * approval gate on resume. The session is cleared instead at
       * natural completion, error (including errors that happen AFTER
       * an interrupt was captured — those interrupts are stale), or
       * hook-driven halt (including hooks that returned BOTH `ask`
       * and `preventContinuation` — the halt wins, no resume is
       * expected, sessions must drop). Every state where no resume
       * is expected clears.
       */
      if (this.shouldClearHookSession(streamThrew)) {
        this.hookRegistry?.clearSession(this.id);
      }
      /**
       * Drop any halt signal raised mid-stream for this run so a
       * subsequent `processStream` / `resume` starts with clean state.
       * The Run captured `_haltedReason` already; the registry entry
       * for this `sessionId` would otherwise spuriously trip the next
       * loop. Other concurrent runs sharing this registry are
       * unaffected — their entries live under their own session ids.
       */
      this.hookRegistry?.clearHaltSignal(this.id);
      await disposeLangfuseHandler(langfuseHandler);

      /**
       * Terminal sweep: close every step that never reached a terminal
       * status — `completed` on a natural end, `cancelled` on caller abort
       * or hook halt, `failed` on an unexpected stream error. Skipped on a
       * HITL pause, where the open steps continue after `resume()`.
       *
       * Runs BEFORE the callback teardown below, so a caller observing
       * lifecycle events only through `RunnableConfig.callbacks` still
       * receives these closures rather than being left with unmatched
       * starts, and before `getContentParts()` so terminal stamps flow into
       * content and session serialization.
       */
      if (!this.isAwaitingResume(streamThrew)) {
        try {
          await this.Graph.closeUnfinishedRunSteps(
            this.resolveSweepStatus(streamThrew, streamAborted),
            terminalAt
          );
        } catch {
          /* the sweep must never mask the stream outcome */
        }
      }

      /**
       * Break the reference chain that keeps heavy data alive via
       * LangGraph's internal `__pregel_scratchpad.currentTaskInput` →
       * `@langchain/core` `RunTree.extra[lc:child_config]` →
       * Node.js `AsyncLocalStorage` context captured by timers/promises.
       *
       * Without this, base64-encoded images/PDFs in message content remain
       * reachable from lingering `Timeout` handles until GC runs.
       */
      if (!this.skipCleanup) {
        if (
          (config.configurable as Record<string, unknown> | undefined) != null
        ) {
          for (const key of Object.getOwnPropertySymbols(config.configurable)) {
            const val = config.configurable[key as unknown as string] as
              | Record<string, unknown>
              | undefined;
            if (
              val != null &&
              typeof val === 'object' &&
              'currentTaskInput' in val
            ) {
              (val as Record<string, unknown>).currentTaskInput = undefined;
            }
            delete config.configurable[key as unknown as string];
          }
          config.configurable = undefined;
        }
        config.callbacks = undefined;
      }

      const result = this.returnContent
        ? this.Graph.getContentParts()
        : undefined;

      this.calibrationRatio = this.Graph.getCalibrationRatio();
      this.fadingTier = this.Graph.getFadingTier();
      this.fadingTiers = this.Graph.getFadingTiers();
      this.fadingTierReset = this.Graph.didResetFadingTier();
      this.fadingTierResetAgentIds = this.Graph.getFadingTierResetAgentIds();

      /**
       * Skip `clearHeavyState()` when the run paused on a clean HITL
       * interrupt awaiting resume — `Run.resume()` re-enters the same
       * `ToolNode` instance and needs the sidecars `clearHeavyState`
       * would wipe (`toolCallStepIds` for completion-event step ids,
       * the `_toolOutputRegistry` for `{{tool<i>turn<n>}}`
       * substitutions, `sessions` for code-env continuity, plus the
       * `hookRegistry` and `humanInTheLoop` config the interrupt
       * branch itself relies on). Without preservation, the resumed
       * tool completion would dispatch `ON_RUN_STEP_COMPLETED` with
       * an empty step id and downstream stream consumers would drop
       * the result.
       *
       * The natural-completion / error / hook-driven-halt paths still
       * clean up — `_haltedReason != null` or `streamThrew` mean no
       * resume is expected. Cross-process resume (host rebuilds the
       * Run from scratch) is a separate concern; see
       * `HumanInTheLoopConfig` JSDoc.
       */
      const awaitingResume = this.isAwaitingResume(streamThrew);
      this._handoffOutcome = graph.handoffRouting?.outcome(
        this.getHandoffIncompleteReason(
          streamThrew,
          awaitingResume,
          config.signal
        )
      );
      if (!this.skipCleanup && !awaitingResume) {
        this.Graph.clearHeavyState();
      }

      this._streamResult = result;
    }

    return this._streamResult;
  }

  /**
   * Returns the pending interrupt captured during the most recent
   * `processStream` (or `resume`) invocation. `undefined` when the run
   * either has not been streamed yet or completed without pausing.
   *
   * Hosts call this immediately after `processStream` returns to decide
   * whether the run is awaiting human input. Persist the returned
   * descriptor (alongside `thread_id` and the agent run config) so a
   * later `resume(decisions)` can rebuild the run.
   *
   * The default `TPayload` is the SDK's `HumanInterruptPayload` union
   * (`tool_approval` / `ask_user_question`), suitable for the common
   * case where interrupts come from the built-in ToolNode or
   * `askUserQuestion()` helper. Hosts that raise custom interrupts
   * from custom graph nodes pass their own type — the SDK does not
   * validate the runtime shape, it just transports whatever the
   * `interrupt()` call carried. When in doubt, narrow with the
   * `isToolApprovalInterrupt` / `isAskUserQuestionInterrupt` type
   * guards (which accept `unknown`) before reading variant-specific
   * fields.
   */
  getInterrupt<TPayload = t.HumanInterruptPayload>():
    | t.RunInterruptResult<TPayload>
    | undefined {
    if (this._interrupt == null) {
      return undefined;
    }
    return {
      ...this._interrupt,
      payload: cloneToolApprovalInterruptPayload(
        getPublicToolInterruptPayload(this._interrupt.payload)
      ),
    } as t.RunInterruptResult<TPayload>;
  }

  /**
   * Returns why the run ended without a natural completion, or `undefined`
   * when it completed normally. Reasons include hook- and prompt-driven
   * halts, `preempt_incomplete` when a cooperative seal ended the turn
   * without continuation content, and `output_truncated` when the provider
   * stopped a plain-text/reasoning response at its output-token ceiling.
   *
   * Hosts inspect this after `processStream` returns to distinguish a
   * natural completion from a terminal partial response. Independent from
   * `getInterrupt()` — a halted run has no interrupt; an interrupted run has
   * no halt reason.
   */
  getHaltReason(): string | undefined {
    return this._haltedReason;
  }

  private getHandoffIncompleteReason(
    streamThrew: boolean,
    awaitingResume: boolean,
    signal?: AbortSignal
  ): string | undefined {
    if (this._haltedReason != null) return this._haltedReason;
    if (signal?.aborted === true || this.Graph?.signal?.aborted === true) {
      return 'aborted';
    }
    if (streamThrew) return 'error';
    if (awaitingResume) return 'interrupted';
    if (this.Graph?.subagentScope === true) return 'subagent';
    return undefined;
  }

  /** Execution evidence only. The host must authorize and durably commit a candidate. */
  getHandoffOutcome(): t.HandoffOutcome | undefined {
    const outcome = this._handoffOutcome;
    if (outcome == null) return undefined;
    return {
      ...outcome,
      transitions: outcome.transitions.map((item) => ({ ...item })),
    };
  }

  /**
   * Resume a paused HITL run with the value the user (or whatever
   * decided the interrupt) supplied. The default `TResume` covers the
   * `tool_approval` interrupt (the common case): an array of decisions
   * in `action_requests` order, or a record keyed by `tool_call_id`.
   *
   * For other interrupt types (e.g., `ask_user_question` →
   * `AskUserQuestionResolution`, or any custom interrupt a host raises
   * from a custom node), pass the type parameter and the SDK forwards
   * the value through unchanged. LangGraph delivers it as the return
   * value of the original `interrupt()` call inside the paused node.
   *
   * The host MUST construct this Run with the same `thread_id` and the
   * same checkpointer as the original paused run; LangGraph rebuilds
   * graph state from the checkpoint and re-enters the interrupted node
   * from the start.
   */
  /**
   * Returns the per-Run file checkpointer when
   * `toolExecution.local.fileCheckpointing === true` was set on the
   * RunConfig. Hosts can capture extra paths or call `rewind()`
   * directly. Returns undefined when checkpointing is disabled.
   *
   * Construction-time invariant: the checkpointer is shared across
   * every ToolNode the graph compiles (single-agent and multi-agent),
   * so a `rewind()` call here unwinds writes made by ANY agent in the
   * run.
   */
  getFileCheckpointer(): t.LocalFileCheckpointer | undefined {
    return this.Graph?.getOrCreateFileCheckpointer();
  }

  /**
   * Convenience wrapper that calls `rewind()` on the per-Run file
   * checkpointer. Restores every file the local engine snapshotted
   * during this Run to its pre-write content (and deletes any path
   * that didn't exist before being created). Returns the count of
   * paths processed; returns 0 when checkpointing is disabled.
   */
  async rewindFiles(): Promise<number> {
    const cp = this.getFileCheckpointer();
    return cp == null ? 0 : cp.rewind();
  }

  /**
   * Resume an interrupted run. `commandOptions` forwards langgraph 1.4.5
   * `Command` fields applied together with `resume` in one superstep:
   * - `update`: channel updates committed at the resume point. On a *rebuilt*
   *   Run (new instance + durable checkpointer), `update.messages` are the first
   *   write the fresh wrapper sees, so they seed the `getRunMessages()` /
   *   `returnContent` baseline and are excluded from them (still committed to the
   *   checkpoint). Hosts that rebuild + inject messages should persist them
   *   directly or read from `getState`. Unreachable without a durable checkpointer.
   * - `goto`: a *dynamic* edge that does not cancel static `addEdge` routes. On
   *   the built-in standard graph the fixed `toolNode -> agentNode/END` edge still
   *   fires, so `goto` adds rather than replaces (e.g. `goto: END` will not stop a
   *   tool-node resume). Intended for custom, Command-routed graphs.
   */
  async resume<TResume = t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap>(
    resumeValue: TResume,
    callerConfig: t.RunStreamConfig,
    streamOptions?: t.EventStreamOptions,
    commandOptions?: Pick<
      ConstructorParameters<typeof Command>[0],
      'update' | 'goto'
    >
  ): Promise<MessageContentComplex[] | undefined> {
    const resumeConfig = await this.resolveInterruptResumeConfig(
      callerConfig,
      commandOptions?.update
    );
    const interruptId = this._interrupt?.interruptId;
    const scopedResume =
      typeof interruptId === 'string' &&
      interruptId.length > 0 &&
      !isLangGraphResumeMapForInterrupt(resumeValue, interruptId)
        ? { [interruptId]: resumeValue }
        : resumeValue;
    // langgraph 1.4.5 applies resume + state update + reroute in one superstep
    // (single checkpoint). `update`/`goto` are omitted unless the caller sets them.
    return this.processStream(
      new Command({
        resume: scopedResume,
        ...(commandOptions?.update !== undefined
          ? { update: commandOptions.update }
          : {}),
        ...(commandOptions?.goto !== undefined
          ? { goto: commandOptions.goto }
          : {}),
      }),
      resumeConfig,
      streamOptions
    );
  }

  private async resolveInterruptResumeConfig(
    callerConfig: t.RunStreamConfig,
    resumeUpdate?: ResumeCommandUpdate
  ): Promise<t.RunStreamConfig> {
    await this.restoreInterruptFromCheckpoint(callerConfig, resumeUpdate);
    const interrupt = this._interrupt;
    const replayPayload = stripRunStepResumeState(interrupt?.payload);
    const resumeManifest =
      requireValidSubagentResumeManifest(replayPayload) ??
      requireValidSubagentResumeManifest(
        stripToolBatchReplayState(replayPayload)
      );
    const resumeConfigurable = { ...callerConfig.configurable };
    delete resumeConfigurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
    delete resumeConfigurable[TOOL_BATCH_REPLAY_KEY];
    delete resumeConfigurable[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY];
    delete resumeConfigurable[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY];
    resumeConfigurable[SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY] = nanoid();
    if (resumeManifest != null) {
      resumeConfigurable[SUBAGENT_RESUME_MANIFEST_CONFIG_KEY] = resumeManifest;
    }
    const publicPayload = stripRunStepResumeState(interrupt?.payload);
    restoreToolReplayConfig(
      resumeConfigurable,
      interrupt?.interruptId,
      publicPayload
    );
    const manifestConfig = {
      ...callerConfig,
      configurable: resumeConfigurable,
    };
    const hookSessionId = getInterruptHookSessionId(interrupt?.payload);
    if (hookSessionId != null) {
      this.hookRegistry?.copySession(hookSessionId, this.id);
    }
    const interruptId = interrupt?.interruptId;
    const workflow = this.graphRunnable as
      | (t.CompiledStateWorkflow & WorkflowWithStateHistory)
      | undefined;
    const stateHistory = workflow?.getStateHistory;
    if (interrupt?.checkpointId != null && interrupt.checkpointId.length > 0) {
      return {
        ...manifestConfig,
        configurable: {
          ...manifestConfig.configurable,
          checkpoint_id: interrupt.checkpointId,
          ...(typeof interrupt.checkpointNs === 'string'
            ? { checkpoint_ns: interrupt.checkpointNs }
            : {}),
        },
      };
    }
    if (
      interrupt == null ||
      typeof interruptId !== 'string' ||
      interruptId.length === 0 ||
      typeof stateHistory !== 'function'
    ) {
      return manifestConfig;
    }

    for await (const snapshot of stateHistory.call(
      this.graphRunnable,
      manifestConfig as RunnableConfig
    )) {
      const hasMatchingInterrupt =
        snapshot.tasks?.some(
          (task) =>
            task.interrupts?.some(
              (interrupt) => interrupt.id === interruptId
            ) === true
        ) === true;
      const checkpointConfigurable = snapshot.config?.configurable;
      if (!hasMatchingInterrupt || checkpointConfigurable == null) {
        continue;
      }

      const checkpointId = checkpointConfigurable.checkpoint_id;
      const checkpointNs = checkpointConfigurable.checkpoint_ns;
      if (typeof checkpointId === 'string' && checkpointId.length > 0) {
        this._interrupt = {
          ...interrupt,
          checkpointId,
          ...(typeof checkpointNs === 'string' ? { checkpointNs } : {}),
        };
        return {
          ...manifestConfig,
          configurable: {
            ...manifestConfig.configurable,
            checkpoint_id: checkpointId,
            ...(typeof checkpointNs === 'string'
              ? { checkpoint_ns: checkpointNs }
              : {}),
          },
        };
      }
    }

    return manifestConfig;
  }

  private async restoreInterruptFromCheckpoint(
    callerConfig: t.RunStreamConfig,
    resumeUpdate?: ResumeCommandUpdate
  ): Promise<void> {
    if (this._interrupt != null || !this.hasCheckpointer) {
      return;
    }
    const workflow = this.graphRunnable as
      | (t.CompiledStateWorkflow & WorkflowWithStateHistory)
      | undefined;
    if (typeof workflow?.getState !== 'function') {
      return;
    }

    const snapshot = await workflow.getState(callerConfig as RunnableConfig, {
      subgraphs: true,
    });
    this.Graph?.handoffRouting?.resume(snapshot.values?.handoffState);
    const persistedInterrupt = getFirstPersistedInterrupt(snapshot);
    if (persistedInterrupt == null) {
      return;
    }
    this.Graph?.restoreRunStepResumeState(
      getRunStepResumeState(persistedInterrupt.value) ??
        snapshot.values?.runStepState
    );
    const persistedMessages = getPersistedMessages(snapshot);
    if (persistedMessages != null) {
      const resumeMessages = getResumeUpdateMessages(resumeUpdate);
      this.Graph?.restoreCheckpointMessages(persistedMessages, resumeMessages);
      this.activityPhaseTraceInput = findActivityPhaseTraceInput(
        resumeMessages == null
          ? persistedMessages
          : [...persistedMessages, ...resumeMessages]
      );
    }

    const checkpointConfigurable = snapshot.config?.configurable;
    const checkpointId = checkpointConfigurable?.checkpoint_id;
    const checkpointNs = checkpointConfigurable?.checkpoint_ns;
    const threadId = callerConfig.configurable?.thread_id;
    this._interrupt = {
      interruptId: persistedInterrupt.id,
      payload: cloneToolApprovalInterruptPayload(persistedInterrupt.value),
      ...(typeof threadId === 'string' ? { threadId } : {}),
      ...(typeof checkpointId === 'string' ? { checkpointId } : {}),
      ...(typeof checkpointNs === 'string' ? { checkpointNs } : {}),
    };
  }

  private createSystemCallback<K extends keyof t.ClientCallbacks>(
    clientCallbacks: t.ClientCallbacks,
    key: K
  ): t.SystemCallbacks[K] {
    return ((...args: unknown[]) => {
      const clientCallback = clientCallbacks[key];
      if (clientCallback && this.Graph) {
        (clientCallback as (...args: unknown[]) => void)(this.Graph, ...args);
      }
    }) as t.SystemCallbacks[K];
  }

  getCallbacks(clientCallbacks: t.ClientCallbacks): t.SystemCallbacks {
    return {
      [Callback.TOOL_ERROR]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_ERROR
      ),
      [Callback.TOOL_START]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_START
      ),
      [Callback.TOOL_END]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_END
      ),
    };
  }

  async generateTitle({
    provider,
    inputText,
    contentParts,
    titlePrompt,
    clientOptions,
    chainOptions,
    skipLanguage,
    titleMethod = TitleMethod.COMPLETION,
    titlePromptTemplate,
  }: t.RunTitleOptions): Promise<{ language?: string; title?: string }> {
    let titleLangfuseHandler: CallbackEntry | undefined;
    let titleUserId: string | undefined;
    let titleSessionId: string | undefined;
    const titleContext =
      this.Graph == null
        ? undefined
        : this.Graph.agentContexts.get(this.Graph.defaultAgentId);
    const titleLangfuseConfig = resolveLangfuseConfig(
      this.langfuse,
      titleContext?.langfuse
    );
    const traceMetadata = createLangfuseTraceMetadata({
      messageId: 'title-' + this.id,
      agentName: titleContext?.name,
    });
    const titleRunName = getLangfuseTraceName(traceMetadata, 'LibreChat Title');
    /** Scope identity carries an opaque per-execution component: public run
     *  ids are unrestricted, so a purely derived id (`title-<runId>`) could
     *  collide with an ordinary concurrent run literally named that way and
     *  defeat foreign-scope rejection. */
    const titleScopeRunId = `title:${this.id}:${nanoid()}`;
    /** Seed policy mirrors `generateActivityLabel`:
     *  `runWithLangfuseRuntimeContext` SPREADS the surrounding context, so an
     *  absent seed INHERITS an active parent run's and collapses the title
     *  into that run's trace. Seeded when determinism is opted into OR a
     *  parent seed is live; otherwise unseeded, matching the other paths. */
    const inheritedTraceSeed = getTraceIdSeed();
    const titleRuntimeScope = resolveLangfuseRuntimeScope({
      runLangfuse: this.langfuse,
      langfuseOverlay: titleContext?.langfuse,
      traceIdSeed:
        titleLangfuseConfig?.deterministicTraceId === true ||
        inheritedTraceSeed != null
          ? 'title-' + this.id
          : undefined,
      runId: titleScopeRunId,
    });

    if (chainOptions != null) {
      titleUserId =
        typeof chainOptions.configurable?.user_id === 'string'
          ? chainOptions.configurable.user_id
          : undefined;
      titleSessionId =
        typeof chainOptions.configurable?.thread_id === 'string'
          ? chainOptions.configurable.thread_id
          : undefined;
      initializeLangfuseTracing(titleLangfuseConfig);
      titleLangfuseHandler = createLangfuseHandler({
        langfuse: titleLangfuseConfig,
        userId: titleUserId,
        sessionId: titleSessionId,
        traceMetadata,
        tags: ['librechat', 'title'],
        traceIdSeed:
          titleLangfuseConfig?.deterministicTraceId === true
            ? 'title-' + this.id
            : undefined,
        runId: titleScopeRunId,
        toolOutputTracing: titleRuntimeScope.toolOutputTracing,
        traceName: chainOptions.runName ?? titleRunName,
      });

      if (titleLangfuseHandler != null) {
        chainOptions.callbacks = appendCallbacks(chainOptions.callbacks, [
          titleLangfuseHandler,
        ]);
      }
    }

    const convoTemplate = PromptTemplate.fromTemplate(
      titlePromptTemplate ?? 'User: {input}\nAI: {output}'
    );

    const response = contentParts
      .map((part) => {
        if (part?.type === 'text') return part.text;
        return '';
      })
      .join('\n');

    const model = initializeModel({
      provider,
      clientOptions,
    }) as t.ChatModelInstance;

    if (isOpenAILike(provider) && isLibreChatOpenAIModel(model)) {
      model.temperature = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.temperature as number;
      model.topP = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.topP as number;
      model.frequencyPenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.frequencyPenalty as number;
      model.presencePenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.presencePenalty as number;
      model.n = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.n as number;
    }

    const convoToTitleInput = new RunnableLambda({
      func: (
        promptValue: StringPromptValue
      ): { convo: string; inputText: string; skipLanguage?: boolean } => ({
        convo: promptValue.value,
        inputText,
        skipLanguage,
      }),
    }).withConfig({ runName: 'PrepareTitleInput' });

    const titleChain =
      titleMethod === TitleMethod.COMPLETION
        ? await createCompletionTitleRunnable(model, titlePrompt)
        : await createTitleRunnable(model, titlePrompt);

    /** Pipes `convoTemplate` -> `transformer` -> `titleChain` */
    const fullChain = convoTemplate
      .withConfig({ runName: 'FormatConversation' })
      .pipe(convoToTitleInput)
      .pipe(titleChain)
      .withConfig({ runName: 'GenerateConversationTitle' });

    const invokeConfig = Object.assign({}, chainOptions, {
      run_id: this.id,
      runId: this.id,
      runName: chainOptions?.runName ?? titleRunName,
    });

    const invokeTitleChain = (
      runtimeConfig: Partial<RunnableConfig>
    ): Promise<{ language?: string; title?: string }> =>
      withLangfuseAttributes(
        {
          langfuse: titleLangfuseConfig,
          userId: titleUserId,
          sessionId: titleSessionId,
          traceName: runtimeConfig.runName ?? titleRunName,
          traceMetadata,
          tags: ['librechat', 'title'],
        },
        () =>
          fullChain.invoke(
            { input: inputText, output: response },
            runtimeConfig
          )
      );

    try {
      try {
        return await withLangfuseRuntimeScope(titleRuntimeScope, () =>
          invokeTitleChain(invokeConfig)
        );
      } catch (_e) {
        // Fallback: strip callbacks to avoid EventStream tracer errors in certain environments
        // but preserve Langfuse tracing if it exists.
        const langfuseHandler = findCallback(
          invokeConfig.callbacks,
          isLangfuseCallbackHandler
        );
        const { callbacks: _cb, ...rest } = invokeConfig;
        const safeConfig = Object.assign({}, rest, {
          callbacks: langfuseHandler ? [langfuseHandler] : [],
        });
        return await withLangfuseRuntimeScope(titleRuntimeScope, () =>
          invokeTitleChain(safeConfig as Partial<RunnableConfig>)
        );
      }
    } finally {
      await disposeLangfuseHandler(titleLangfuseHandler);
    }
  }

  /**
   * Generates a short activity label for a completed tool/reasoning block
   * using a fast model. Mirrors `generateTitle`'s Langfuse wiring so the
   * call is traced under the conversation's session (sessionId from
   * `chainOptions.configurable.thread_id`) with its own tags — never as an
   * orphan trace. The payload contains no human messages by design: intent
   * comes from `lastAssistantText`, content from reasoning excerpts and
   * tool entries.
   */
  async generateActivityLabel({
    provider,
    clientOptions,
    entries,
    thinkingExcerpts,
    lastAssistantText,
    lastAssistantPhase,
    previousLabels,
    prompt,
    charLimit = 600,
    chainOptions,
    traceSeed,
    agentId,
  }: t.RunActivityLabelOptions): Promise<{ label?: string }> {
    if (
      entries.length === 0 &&
      !(thinkingExcerpts && thinkingExcerpts.length > 0)
    ) {
      return {};
    }
    const labelSeq = ++this.activityLabelSeq;

    /** Resolve the LABELED agent's context: its Langfuse overlay carries the
     *  trace metadata and the tool-output redaction policy that must govern
     *  this label. */
    const requestedContext =
      this.Graph == null || agentId == null
        ? undefined
        : this.Graph.agentContexts.get(agentId);
    /** Fail closed: an explicit but unknown/stale `agentId` must NOT silently
     *  fall back to the default agent, whose redaction policy may be weaker
     *  than the labeled agent's. Skip generation entirely instead. */
    if (agentId != null && requestedContext == null) {
      return {};
    }
    const labelContext =
      this.Graph == null
        ? undefined
        : (requestedContext ??
          this.Graph.agentContexts.get(this.Graph.defaultAgentId));
    /** Shallow-cloned: activity labels run once per tool batch, and writing
     *  the Langfuse handler back onto a host-reused `chainOptions` would
     *  accumulate duplicate callbacks across batches. */
    const labelChainOptions = {
      ...(chainOptions ?? {}),
    } as Partial<RunnableConfig> & {
      configurable?: Record<string, unknown> & {
        requestBody?: { parentMessageId?: unknown };
      };
    };
    const labelUserId =
      typeof labelChainOptions.configurable?.user_id === 'string'
        ? (labelChainOptions.configurable.user_id as string)
        : undefined;
    const labelSessionId =
      typeof labelChainOptions.configurable?.thread_id === 'string'
        ? (labelChainOptions.configurable.thread_id as string)
        : undefined;
    const labelIndex = labelSeq - 1;
    const labelParentMessageId =
      labelChainOptions.configurable?.requestBody?.parentMessageId;
    /** An omitted `agentId` is attributable only when exactly one context
     *  exists. Multi-agent callers remain unattributed instead of being
     *  incorrectly assigned to the graph's default agent. */
    const labelAgentId =
      agentId ??
      (this.Graph?.agentContexts.size === 1
        ? this.Graph.defaultAgentId
        : undefined);
    const labelAgentName =
      labelAgentId == null ? undefined : labelContext?.name;
    const labelMetadata: Record<string, unknown> = {
      [LANGFUSE_OPERATION_METADATA_KEY]: ACTIVITY_LABEL_RUN_NAME,
      sourceRunId: this.id,
      responseId: this.id,
      activityIndex: labelIndex,
      ...(typeof labelParentMessageId === 'string'
        ? { parentMessageId: labelParentMessageId }
        : {}),
      ...(labelAgentId == null ? {} : { agentId: labelAgentId }),
      ...(labelAgentName == null ? {} : { agentName: labelAgentName }),
    };
    const traceMetadata: Record<string, string> = {
      ...createLangfuseTraceMetadata({
        messageId: 'activity-label-' + this.id,
        parentMessageId: labelParentMessageId,
        agentId: labelAgentId,
        agentName: labelAgentName,
      }),
      sourceRunId: this.id,
      responseId: this.id,
      activityIndex: String(labelIndex),
    };
    const labelRunName = labelChainOptions.runName ?? ACTIVITY_LABEL_TRACE_NAME;
    const labelTags = ['librechat', 'activity-label'];
    const labelLangfuseConfig = resolveLangfuseConfig(
      this.langfuse,
      labelContext?.langfuse
    );
    initializeLangfuseTracing(labelLangfuseConfig);
    /** A captured source observation parents the label in the agent trace.
     *  The distinct seed remains the standalone fallback when labeling runs
     *  before tracing starts or against another Langfuse destination. */
    const inheritedTraceSeed = getTraceIdSeed();
    const labelTraceSeed =
      labelLangfuseConfig?.deterministicTraceId === true ||
      inheritedTraceSeed != null
        ? (traceSeed ?? `activity-label-${this.id}-${labelSeq}`)
        : undefined;
    /** Opaque per-execution component: see `titleScopeRunId`. */
    const labelScopeRunId = `activity-label:${this.id}:${labelSeq}:${nanoid()}`;
    const labelRuntimeScope = resolveLangfuseRuntimeScope({
      runLangfuse: this.langfuse,
      langfuseOverlay: labelContext?.langfuse,
      traceIdSeed: labelTraceSeed,
      runId: labelScopeRunId,
    });
    const labelParentSpanContext = resolveLangfuseTraceAnchorParent(
      this.Graph?.langfuseTraceAnchor,
      resolveLangfuseDestinationKey(labelLangfuseConfig),
      labelAgentId
    );
    if (labelParentSpanContext != null) {
      labelMetadata.sourceTraceId = labelParentSpanContext.traceId;
      traceMetadata.sourceTraceId = labelParentSpanContext.traceId;
    }
    /** Handler only when a session id resolved from
     *  `chainOptions.configurable.thread_id`: without it the label call has
     *  no conversation identity, and tracing it would create an orphan
     *  trace outside any session — worse than not tracing at all. */
    /** Declared then conditionally assigned (title precedent): a ternary
     *  around the object literal makes eslint's indent rule and prettier
     *  disagree, and both gate CI. */
    let labelLangfuseHandler: CallbackEntry | undefined;
    if (labelSessionId != null) {
      labelLangfuseHandler = createLangfuseHandler({
        langfuse: labelLangfuseConfig,
        userId: labelUserId,
        sessionId: labelSessionId,
        traceMetadata:
          labelParentSpanContext == null ? traceMetadata : undefined,
        tags: labelTags,
        traceIdSeed:
          labelLangfuseConfig?.deterministicTraceId === true
            ? labelTraceSeed
            : undefined,
        parentSpanContext: labelParentSpanContext,
        inheritTraceIdentity: labelParentSpanContext != null,
        runId: labelScopeRunId,
        toolOutputTracing: labelRuntimeScope.toolOutputTracing,
        traceName: labelParentSpanContext == null ? labelRunName : undefined,
      });
    }
    if (labelLangfuseHandler != null) {
      labelChainOptions.callbacks = appendCallbacks(
        labelChainOptions.callbacks,
        [labelLangfuseHandler]
      );
    }

    /** The label prompt becomes Langfuse generation input, so the resolved
     *  tool-output redaction policy (global disable / redactedToolNames)
     *  applies to it exactly as to structured tool observations. */
    let redaction = hasToolOutputTracingConfig(
      this.langfuse,
      labelContext?.langfuse
    )
      ? resolveToolOutputTracingConfig(this.langfuse, labelContext?.langfuse)
      : undefined;
    /** Multi-agent graph with no `agentId`: the caller did not say WHICH
     *  agent ran this batch, so resolving from the default agent could trace
     *  raw output that a stricter sibling's policy forbids. Fold every
     *  agent's policy into the strictest one instead of guessing. */
    const agentContexts = this.Graph?.agentContexts;
    if (agentId == null && agentContexts != null && agentContexts.size > 1) {
      for (const context of agentContexts.values()) {
        if (!hasToolOutputTracingConfig(this.langfuse, context.langfuse)) {
          continue;
        }
        const candidate = resolveToolOutputTracingConfig(
          this.langfuse,
          context.langfuse
        );
        if (redaction == null) {
          redaction = candidate;
          continue;
        }
        redaction = {
          enabled: redaction.enabled === false ? false : candidate.enabled,
          redactedToolNames: new Set([
            ...redaction.redactedToolNames,
            ...candidate.redactedToolNames,
          ]),
          redactedToolNameMatchMode:
            redaction.redactedToolNameMatchMode === 'partial' ||
            candidate.redactedToolNameMatchMode === 'partial'
              ? 'partial'
              : 'exact',
          redactionText: redaction.redactionText,
        };
      }
    }
    /** An active redaction policy suppresses free-form reasoning/intent, so
     *  a reasoning-only block has nothing describable left — skip the model
     *  call rather than paying for a label built from the prompt alone. */
    const freeFormSuppressed =
      redaction != null &&
      (redaction.enabled === false || redaction.redactedToolNames.size > 0);
    if (entries.length === 0 && freeFormSuppressed) {
      return {};
    }
    const userPrompt = buildActivityLabelPrompt({
      entries,
      charLimit,
      thinkingExcerpts,
      lastAssistantText:
        lastAssistantPhase === 'final_answer' ? undefined : lastAssistantText,
      previousLabels,
      redaction,
    });

    const model = initializeModel({
      provider,
      clientOptions: {
        ...(clientOptions ?? {}),
        streaming: false,
      } as t.ClientOptions,
    }) as t.ChatModelInstance;

    /** Distinct run id per label call: callback/tracing integrations key
     *  in-flight runs by it, so reusing the parent run's id would collide
     *  across successive (or concurrent) label batches. */
    const labelRunId = `${this.id}-activity-${labelSeq}`;
    const invokeConfig = Object.assign({}, labelChainOptions, {
      run_id: labelRunId,
      runId: labelRunId,
      runName: labelRunName,
      tags: [...new Set([...(labelChainOptions.tags ?? []), ...labelTags])],
      metadata: {
        ...(labelChainOptions.metadata ?? {}),
        ...labelMetadata,
      },
    }) as Partial<RunnableConfig>;

    const invokeLabel = (
      runtimeConfig: Partial<RunnableConfig>
    ): Promise<unknown> =>
      withLangfuseAttributes(
        {
          langfuse: labelLangfuseConfig,
          userId: labelUserId,
          sessionId: labelSessionId,
          traceName: labelParentSpanContext == null ? labelRunName : undefined,
          traceMetadata:
            labelParentSpanContext == null ? traceMetadata : undefined,
          tags: labelTags,
          inheritTraceIdentity: labelParentSpanContext != null,
        },
        () =>
          model.invoke(
            [
              new SystemMessage(prompt ?? ACTIVITY_LABEL_PROMPT),
              new HumanMessage(userPrompt),
            ],
            runtimeConfig
          )
      );

    const extractLabel = (response: unknown): string => {
      const content = (response as { content?: unknown } | null)?.content;
      let text = '';
      if (typeof content === 'string') {
        text = content;
      } else if (Array.isArray(content)) {
        text = content
          .map((block) =>
            typeof block === 'string'
              ? block
              : ((block as { text?: string }).text ?? '')
          )
          .join('');
      }
      /** Collapsed to one line at the source: a header renders as a single
       *  row, and hosts feed committed labels back as continuity context —
       *  so a multi-line result would carry its line breaks into every later
       *  prompt in the run. */
      return text
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/^["']|["']$/g, '');
    };

    try {
      let response: unknown;
      try {
        response = await withLangfuseRuntimeScope(labelRuntimeScope, () =>
          invokeLabel(invokeConfig)
        );
      } catch (error) {
        /** Retry ONLY recognized callback/tracer failures (the EventStream
         *  tracer class of errors the stripped-callbacks fallback exists
         *  for). Aborts and provider failures rethrow — retrying those
         *  doubles traffic/cost and can restart cancelled requests. */
        const aborted =
          (labelChainOptions as { signal?: AbortSignal }).signal?.aborted ===
            true || (error as Error | null)?.name === 'AbortError';
        const callbackFailure = /callback|tracer|event.?stream/i.test(
          String(
            (error as Error | null)?.stack ??
              (error as Error | null)?.message ??
              ''
          )
        );
        if (aborted || !callbackFailure) {
          throw error;
        }
        const langfuseHandler = findCallback(
          invokeConfig.callbacks,
          isLangfuseCallbackHandler
        );
        const { callbacks, ...rest } = invokeConfig;
        const safeConfig = Object.assign({}, rest, {
          callbacks: filterCallbacks(
            callbacks,
            (callback) => callback === langfuseHandler
          ),
        });
        response = await withLangfuseRuntimeScope(labelRuntimeScope, () =>
          invokeLabel(safeConfig as Partial<RunnableConfig>)
        );
      }
      const label = extractLabel(response);
      return label.length > 0 ? { label } : {};
    } finally {
      await disposeLangfuseHandler(labelLangfuseHandler);
    }
  }

  /**
   * Generates one replacement title for a user-visible reasoning step. Hosts
   * own accumulation, scheduling, revision ordering, durable delivery, and
   * billing; the SDK only performs one bounded, redaction-aware generation.
   */
  async generateReasoningLabel({
    provider,
    clientOptions,
    visibleReasoning,
    reasoningStepId,
    revision,
    status = 'streaming',
    previousLabel,
    agentId,
    prompt,
    charLimit = 6_000,
    chainOptions,
    traceSeed,
    sourceRunId,
    sourceTraceId,
    responseId,
  }: t.RunReasoningLabelOptions): Promise<t.ReasoningLabelResult> {
    const normalizedStepId = reasoningStepId.trim();
    const snapshotChars = visibleReasoning.trim().length;
    if (
      normalizedStepId === '' ||
      snapshotChars === 0 ||
      !Number.isInteger(revision) ||
      revision < 0
    ) {
      return {};
    }

    const reasoningSeq = ++this.reasoningLabelSeq;
    const requestedContext =
      this.Graph == null || agentId == null
        ? undefined
        : this.Graph.agentContexts.get(agentId);
    if (agentId != null && requestedContext == null) {
      return {};
    }
    if (agentId == null && (this.Graph?.agentContexts.size ?? 0) > 1) {
      return {};
    }
    const reasoningContext =
      this.Graph == null
        ? undefined
        : (requestedContext ??
          this.Graph.agentContexts.get(this.Graph.defaultAgentId));
    const reasoningChainOptions = {
      ...(chainOptions ?? {}),
    } as Partial<RunnableConfig> & {
      configurable?: Record<string, unknown> & {
        requestBody?: { parentMessageId?: unknown };
      };
    };
    const reasoningUserId =
      typeof reasoningChainOptions.configurable?.user_id === 'string'
        ? reasoningChainOptions.configurable.user_id
        : undefined;
    const reasoningSessionId =
      typeof reasoningChainOptions.configurable?.thread_id === 'string'
        ? reasoningChainOptions.configurable.thread_id
        : undefined;
    const reasoningParentMessageId =
      reasoningChainOptions.configurable?.requestBody?.parentMessageId;
    const reasoningAgentId =
      agentId ??
      (this.Graph?.agentContexts.size === 1
        ? this.Graph.defaultAgentId
        : undefined);
    const reasoningAgentName =
      reasoningAgentId == null ? undefined : reasoningContext?.name;
    const resolvedSourceRunId = sourceRunId ?? this.id;
    const reasoningResponseId = responseId ?? this.id;
    const reasoningMetadata: Record<string, unknown> = {
      [LANGFUSE_OPERATION_METADATA_KEY]: REASONING_LABEL_RUN_NAME,
      sourceRunId: resolvedSourceRunId,
      ...(sourceTraceId == null ? {} : { sourceTraceId }),
      responseId: reasoningResponseId,
      reasoningStepId: normalizedStepId,
      revision,
      status,
      snapshotChars,
      ...(typeof reasoningParentMessageId === 'string'
        ? { parentMessageId: reasoningParentMessageId }
        : {}),
      ...(reasoningAgentId == null ? {} : { agentId: reasoningAgentId }),
      ...(reasoningAgentName == null ? {} : { agentName: reasoningAgentName }),
    };
    const traceMetadata: Record<string, string> = {
      ...createLangfuseTraceMetadata({
        messageId: `reasoning-label-${reasoningResponseId}`,
        parentMessageId: reasoningParentMessageId,
        agentId: reasoningAgentId,
        agentName: reasoningAgentName,
      }),
      sourceRunId: resolvedSourceRunId,
      ...(sourceTraceId == null ? {} : { sourceTraceId }),
      responseId: reasoningResponseId,
      reasoningStepId: normalizedStepId,
      revision: String(revision),
      status,
      snapshotChars: String(snapshotChars),
    };
    const reasoningRunName =
      reasoningChainOptions.runName ?? REASONING_LABEL_TRACE_NAME;
    const reasoningTags = ['librechat', 'reasoning-label', 'reasoning-step'];
    const reasoningLangfuseConfig = resolveLangfuseConfig(
      this.langfuse,
      reasoningContext?.langfuse
    );
    initializeLangfuseTracing(reasoningLangfuseConfig);

    const inheritedTraceSeed = getTraceIdSeed();
    const reasoningTraceSeed =
      reasoningLangfuseConfig?.deterministicTraceId === true ||
      inheritedTraceSeed != null
        ? (traceSeed ??
          buildReasoningLabelTraceSeed(
            resolvedSourceRunId,
            normalizedStepId,
            revision
          ))
        : undefined;
    const reasoningScopeRunId = `reasoning-label:${this.id}:${reasoningSeq}:${nanoid()}`;
    const reasoningRuntimeScope = resolveLangfuseRuntimeScope({
      runLangfuse: this.langfuse,
      langfuseOverlay: reasoningContext?.langfuse,
      traceIdSeed: reasoningTraceSeed,
      runId: reasoningScopeRunId,
    });
    const candidateReasoningParent = resolveLangfuseTraceAnchorParent(
      this.Graph?.langfuseTraceAnchor,
      resolveLangfuseDestinationKey(reasoningLangfuseConfig),
      reasoningAgentId
    );
    const reasoningParentSpanContext =
      resolvedSourceRunId === this.id &&
      (sourceTraceId == null ||
        sourceTraceId === candidateReasoningParent?.traceId)
        ? candidateReasoningParent
        : undefined;
    if (reasoningParentSpanContext != null) {
      reasoningMetadata.sourceTraceId = reasoningParentSpanContext.traceId;
      traceMetadata.sourceTraceId = reasoningParentSpanContext.traceId;
    }
    let reasoningLangfuseHandler: CallbackEntry | undefined;
    if (reasoningSessionId != null) {
      reasoningLangfuseHandler = createLangfuseHandler({
        langfuse: reasoningLangfuseConfig,
        userId: reasoningUserId,
        sessionId: reasoningSessionId,
        traceMetadata:
          reasoningParentSpanContext == null ? traceMetadata : undefined,
        tags: reasoningTags,
        traceIdSeed:
          reasoningLangfuseConfig?.deterministicTraceId === true
            ? reasoningTraceSeed
            : undefined,
        parentSpanContext: reasoningParentSpanContext,
        inheritTraceIdentity: reasoningParentSpanContext != null,
        runId: reasoningScopeRunId,
        toolOutputTracing: reasoningRuntimeScope.toolOutputTracing,
        traceName:
          reasoningParentSpanContext == null ? reasoningRunName : undefined,
      });
    }
    if (reasoningLangfuseHandler != null) {
      reasoningChainOptions.callbacks = appendCallbacks(
        reasoningChainOptions.callbacks,
        [reasoningLangfuseHandler]
      );
    }

    const redaction = hasToolOutputTracingConfig(
      this.langfuse,
      reasoningContext?.langfuse
    )
      ? resolveToolOutputTracingConfig(
        this.langfuse,
        reasoningContext?.langfuse
      )
      : undefined;
    const userPrompt = buildReasoningLabelPrompt({
      visibleReasoning,
      status,
      charLimit,
      previousLabel,
      redaction,
    });
    if (userPrompt === '') {
      await disposeLangfuseHandler(reasoningLangfuseHandler);
      return {};
    }

    const model = initializeModel({
      provider,
      clientOptions: {
        ...(clientOptions ?? {}),
        streaming: false,
      } as t.ClientOptions,
    }) as t.ChatModelInstance;
    const reasoningRunId = `${this.id}-reasoning-${reasoningSeq}`;
    const invokeConfig = Object.assign({}, reasoningChainOptions, {
      run_id: reasoningRunId,
      runId: reasoningRunId,
      runName: reasoningRunName,
      tags: [
        ...new Set([...(reasoningChainOptions.tags ?? []), ...reasoningTags]),
      ],
      metadata: {
        ...(reasoningChainOptions.metadata ?? {}),
        ...reasoningMetadata,
      },
    }) as Partial<RunnableConfig>;
    const invokeLabel = (
      runtimeConfig: Partial<RunnableConfig>
    ): Promise<unknown> =>
      withLangfuseAttributes(
        {
          langfuse: reasoningLangfuseConfig,
          userId: reasoningUserId,
          sessionId: reasoningSessionId,
          traceName:
            reasoningParentSpanContext == null ? reasoningRunName : undefined,
          traceMetadata:
            reasoningParentSpanContext == null ? traceMetadata : undefined,
          tags: reasoningTags,
          inheritTraceIdentity: reasoningParentSpanContext != null,
        },
        () =>
          model.invoke(
            [
              new SystemMessage(prompt ?? REASONING_LABEL_PROMPT),
              new HumanMessage(userPrompt),
            ],
            runtimeConfig
          )
      );
    const extractResult = (response: unknown): t.ReasoningLabelResult => {
      const result = response as {
        content?: unknown;
        usage_metadata?: UsageMetadata;
      } | null;
      const content = result?.content;
      let text = '';
      if (typeof content === 'string') {
        text = content;
      } else if (Array.isArray(content)) {
        text = content
          .map((block) =>
            typeof block === 'string'
              ? block
              : ((block as { text?: string }).text ?? '')
          )
          .join('');
      }
      const label = normalizeReasoningLabel(text);
      return {
        ...(label === '' ? {} : { label }),
        ...(result?.usage_metadata == null
          ? {}
          : { usage: result.usage_metadata }),
      };
    };

    try {
      const response = await withLangfuseRuntimeScope(
        reasoningRuntimeScope,
        () => invokeLabel(invokeConfig)
      );
      return extractResult(response);
    } finally {
      await disposeLangfuseHandler(reasoningLangfuseHandler);
    }
  }

  /**
   * Generates one parent summary for two or more logical activities. The
   * summary model is traced as an activity-phase chain beneath the source
   * agent run, with the model callback recorded as its generation child. It
   * falls back to a dedicated trace when no source observation is available.
   * No session id means no phase trace, avoiding orphan observations.
   */
  async generateActivityPhaseLabel({
    provider,
    clientOptions,
    activities,
    totalActivityCount,
    assistantContext,
    closingTextPhase,
    prompt,
    charLimit = 600,
    chainOptions,
    traceSeed,
    sourceRunId,
    sourceTraceId,
    responseId,
    phaseIndex,
    status = 'completed',
    agentIds,
  }: t.RunActivityPhaseLabelOptions): Promise<{ label?: string }> {
    if (activities.length < 2) {
      return {};
    }

    const phaseSeq = ++this.activityPhaseLabelSeq;
    const hasUnattributedActivity = activities.some(
      (activity) => activity.agentId == null
    );
    const hasOmittedActivitiesWithoutAgentIds =
      agentIds == null &&
      (totalActivityCount ?? activities.length) > activities.length;
    const contributingAgentIds = [
      ...new Set([
        ...(agentIds ?? []),
        ...activities.flatMap((activity) =>
          activity.agentId == null ? [] : [activity.agentId]
        ),
      ]),
    ];
    const agentContexts = this.Graph?.agentContexts;
    if (
      contributingAgentIds.some(
        (agentId) => agentContexts?.get(agentId) == null
      )
    ) {
      return {};
    }
    const phaseContext =
      this.Graph == null
        ? undefined
        : this.Graph.agentContexts.get(this.Graph.defaultAgentId);
    const phaseLangfuseConfig = resolveLangfuseConfig(
      this.langfuse,
      phaseContext?.langfuse
    );

    let redaction = hasToolOutputTracingConfig(
      this.langfuse,
      phaseContext?.langfuse
    )
      ? resolveToolOutputTracingConfig(this.langfuse, phaseContext?.langfuse)
      : undefined;
    const redactionContexts =
      contributingAgentIds.length > 0 &&
      !hasUnattributedActivity &&
      !hasOmittedActivitiesWithoutAgentIds
        ? contributingAgentIds.flatMap((agentId) => {
          const context = agentContexts?.get(agentId);
          return context == null ? [] : [context];
        })
        : Array.from(agentContexts?.values() ?? []);
    for (const context of redactionContexts) {
      if (!hasToolOutputTracingConfig(this.langfuse, context.langfuse)) {
        continue;
      }
      const candidate = resolveToolOutputTracingConfig(
        this.langfuse,
        context.langfuse
      );
      if (redaction == null) {
        redaction = candidate;
        continue;
      }
      redaction = {
        enabled: redaction.enabled === false ? false : candidate.enabled,
        redactedToolNames: new Set([
          ...redaction.redactedToolNames,
          ...candidate.redactedToolNames,
        ]),
        redactedToolNameMatchMode:
          redaction.redactedToolNameMatchMode === 'partial' ||
          candidate.redactedToolNameMatchMode === 'partial'
            ? 'partial'
            : 'exact',
        redactionText: redaction.redactionText,
      };
    }

    const userPrompt = buildActivityPhaseLabelPrompt({
      activities,
      totalActivityCount,
      charLimit,
      assistantContext,
      redaction,
    });
    if (userPrompt === '') {
      return {};
    }
    const phaseChainOptions = {
      ...(chainOptions ?? {}),
    } as Partial<RunnableConfig> & {
      configurable?: Record<string, unknown> & {
        requestBody?: { parentMessageId?: unknown };
      };
    };
    const phaseUserId =
      typeof phaseChainOptions.configurable?.user_id === 'string'
        ? phaseChainOptions.configurable.user_id
        : undefined;
    const phaseSessionId =
      typeof phaseChainOptions.configurable?.thread_id === 'string'
        ? phaseChainOptions.configurable.thread_id
        : undefined;
    const resolvedPhaseIndex = phaseIndex ?? phaseSeq - 1;
    const phaseMessageId =
      responseId ?? `activity-phase-${this.id}-${resolvedPhaseIndex}`;
    const phaseAgentId = this.Graph?.defaultAgentId;
    const phaseAgentName = phaseContext?.name;
    const phaseParentMessageId =
      phaseChainOptions.configurable?.requestBody?.parentMessageId;
    const phaseMetadata: Record<string, unknown> = {
      [LANGFUSE_OPERATION_METADATA_KEY]: ACTIVITY_PHASE_LABEL_RUN_NAME,
      sourceRunId: sourceRunId ?? this.id,
      ...(sourceTraceId == null ? {} : { sourceTraceId }),
      responseId: phaseMessageId,
      phaseIndex: resolvedPhaseIndex,
      activityCount: Math.max(activities.length, totalActivityCount ?? 0),
      status,
      contributingAgentIds,
      ...(typeof phaseParentMessageId === 'string'
        ? { parentMessageId: phaseParentMessageId }
        : {}),
      ...(phaseAgentId == null ? {} : { agentId: phaseAgentId }),
      ...(phaseAgentName == null ? {} : { agentName: phaseAgentName }),
      ...(closingTextPhase == null ? {} : { closingTextPhase }),
    };
    const traceMetadata: Record<string, string> = {
      ...createLangfuseTraceMetadata({
        messageId: phaseMessageId,
        parentMessageId: phaseParentMessageId,
        agentId: phaseAgentId,
        agentName: phaseAgentName,
      }),
      sourceRunId: String(phaseMetadata.sourceRunId),
      ...(sourceTraceId == null ? {} : { sourceTraceId }),
      responseId: phaseMessageId,
      phaseIndex: String(resolvedPhaseIndex),
      activityCount: String(phaseMetadata.activityCount),
      status,
      ...(contributingAgentIds.length === 0
        ? {}
        : { contributingAgentIds: contributingAgentIds.join(',') }),
      ...(closingTextPhase == null ? {} : { closingTextPhase }),
    };
    const phaseTraceName =
      phaseChainOptions.runName ?? ACTIVITY_PHASE_TRACE_NAME;
    const phaseTags = [
      'librechat',
      'activity-phase',
      'agent-run-summary',
      'agent',
    ];
    initializeLangfuseTracing(phaseLangfuseConfig);

    const inheritedTraceSeed = getTraceIdSeed();
    const phaseTraceSeed =
      phaseLangfuseConfig?.deterministicTraceId === true ||
      inheritedTraceSeed != null
        ? (traceSeed ?? `activity-phase-${this.id}-${resolvedPhaseIndex}`)
        : undefined;
    const phaseScopeRunId = `activity-phase:${this.id}:${phaseSeq}:${nanoid()}`;
    const phaseRuntimeScope = resolveLangfuseRuntimeScope({
      runLangfuse: this.langfuse,
      langfuseOverlay: phaseContext?.langfuse,
      traceIdSeed: phaseTraceSeed,
      runId: phaseScopeRunId,
    });
    const candidatePhaseParent = resolveLangfuseTraceAnchorParent(
      this.Graph?.langfuseTraceAnchor,
      resolveLangfuseDestinationKey(phaseLangfuseConfig)
    );
    const phaseParentSpanContext =
      (sourceRunId == null || sourceRunId === this.id) &&
      (sourceTraceId == null || sourceTraceId === candidatePhaseParent?.traceId)
        ? candidatePhaseParent
        : undefined;
    if (phaseParentSpanContext != null) {
      phaseMetadata.sourceTraceId = phaseParentSpanContext.traceId;
      traceMetadata.sourceTraceId = phaseParentSpanContext.traceId;
    }
    let phaseLangfuseHandler: CallbackEntry | undefined;
    const sourceUserText =
      this.activityPhaseTraceInput ??
      findActivityPhaseTraceInput(this.Graph?.getRunMessages() ?? []);
    if (phaseSessionId != null && sourceUserText != null) {
      phaseLangfuseHandler = createLangfuseHandler({
        langfuse: phaseLangfuseConfig,
        userId: phaseUserId,
        sessionId: phaseSessionId,
        traceMetadata:
          phaseParentSpanContext == null ? traceMetadata : undefined,
        tags: phaseTags,
        traceIdSeed:
          phaseLangfuseConfig?.deterministicTraceId === true
            ? phaseTraceSeed
            : undefined,
        parentSpanContext: phaseParentSpanContext,
        inheritTraceIdentity: phaseParentSpanContext != null,
        runId: phaseScopeRunId,
        toolOutputTracing: phaseRuntimeScope.toolOutputTracing,
        traceName: phaseParentSpanContext == null ? phaseTraceName : undefined,
      });
    }
    if (phaseLangfuseHandler != null) {
      phaseChainOptions.callbacks = appendCallbacks(
        phaseChainOptions.callbacks,
        [phaseLangfuseHandler]
      );
    }

    const model = initializeModel({
      provider,
      clientOptions: {
        ...(clientOptions ?? {}),
        streaming: false,
      } as t.ClientOptions,
    }) as t.ChatModelInstance;
    const phaseRunId = `${this.id}-activity-phase-${phaseSeq}`;
    const invokeConfig = Object.assign({}, phaseChainOptions, {
      run_id: phaseRunId,
      runId: phaseRunId,
      runName: ACTIVITY_PHASE_RUN_NAME,
      tags: [...new Set([...(phaseChainOptions.tags ?? []), ...phaseTags])],
      metadata: {
        ...(phaseChainOptions.metadata ?? {}),
        ...phaseMetadata,
      },
    }) as Partial<RunnableConfig>;
    const invokeModel = (
      runtimeConfig: Partial<RunnableConfig>
    ): Promise<unknown> =>
      model.invoke(
        [
          new SystemMessage(prompt ?? ACTIVITY_PHASE_LABEL_PROMPT),
          new HumanMessage(userPrompt),
        ],
        runtimeConfig
      );
    const invokeWithCallbackFallback = async (
      runtimeConfig: Partial<RunnableConfig>
    ): Promise<unknown> => {
      try {
        return await invokeModel(runtimeConfig);
      } catch (error) {
        const aborted =
          (runtimeConfig as { signal?: AbortSignal }).signal?.aborted ===
            true || (error as Error | null)?.name === 'AbortError';
        const callbackFailure = /callback|tracer|event.?stream/i.test(
          String(
            (error as Error | null)?.stack ??
              (error as Error | null)?.message ??
              ''
          )
        );
        if (aborted || !callbackFailure) {
          throw error;
        }
        const langfuseHandler = findCallback(
          runtimeConfig.callbacks,
          isLangfuseCallbackHandler
        );
        const { callbacks, ...rest } = runtimeConfig;
        const safeConfig = Object.assign({}, rest, {
          callbacks: filterCallbacks(
            callbacks,
            (callback) => callback === langfuseHandler
          ),
        });
        return invokeModel(safeConfig as Partial<RunnableConfig>);
      }
    };
    const extractPhaseLabel = (response: unknown): string => {
      const content = (response as { content?: unknown } | null)?.content;
      if (typeof content === 'string') {
        return normalizeActivityPhaseLabel(content);
      }
      if (!Array.isArray(content)) {
        return '';
      }
      return normalizeActivityPhaseLabel(
        content
          .map((block) =>
            typeof block === 'string'
              ? block
              : ((block as { text?: string }).text ?? '')
          )
          .join('')
      );
    };
    const phaseRunnable = new RunnableLambda({
      func: async (
        _input: { messages: BaseMessage[] },
        runtimeConfig?: Partial<RunnableConfig>
      ): Promise<{ label?: string; messages: BaseMessage[] }> => {
        const response = await invokeWithCallbackFallback(runtimeConfig ?? {});
        const label = extractPhaseLabel(response);
        return label.length > 0
          ? { label, messages: [new AIMessage(label)] }
          : { messages: [] };
      },
    }).withConfig({ runName: ACTIVITY_PHASE_RUN_NAME });

    try {
      const result = await withLangfuseRuntimeScope(phaseRuntimeScope, () =>
        withLangfuseAttributes(
          {
            langfuse: phaseLangfuseConfig,
            userId: phaseUserId,
            sessionId: phaseSessionId,
            traceName:
              phaseParentSpanContext == null ? phaseTraceName : undefined,
            traceMetadata:
              phaseParentSpanContext == null ? traceMetadata : undefined,
            tags: phaseTags,
            inheritTraceIdentity: phaseParentSpanContext != null,
          },
          () =>
            phaseRunnable.invoke(
              {
                messages:
                  sourceUserText == null
                    ? []
                    : [new HumanMessage(sourceUserText)],
              },
              invokeConfig
            )
        )
      );
      return result.label == null ? {} : { label: result.label };
    } finally {
      await disposeLangfuseHandler(phaseLangfuseHandler);
    }
  }
}

function findLastMessageOfType(
  messages: BaseMessage[],
  type: string
): BaseMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].getType() === type) {
      return messages[i];
    }
  }
  return undefined;
}

function findActivityPhaseTraceInput(
  messages: BaseMessage[]
): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (
      message.getType() !== 'human' ||
      message.additional_kwargs.role === 'system' ||
      message.additional_kwargs.isMeta === true
    ) {
      continue;
    }
    const input = extractPromptText(message).trim();
    if (input !== '') {
      return input;
    }
  }
  return undefined;
}

function extractPromptText(message: BaseMessage): string {
  const content = message.content;
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return String(content);
  }
  const parts: string[] = [];
  for (const block of content) {
    const textBlock = block as { type?: unknown; text?: unknown } | null;
    if (
      textBlock != null &&
      (textBlock.type === 'text' || textBlock.type === 'input_text') &&
      typeof textBlock.text === 'string'
    ) {
      parts.push(textBlock.text);
    }
  }
  return parts.join('\n');
}

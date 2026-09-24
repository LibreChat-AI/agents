/* eslint-disable no-console */
import { v4 } from 'uuid';
import { nanoid } from 'nanoid';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ContextOverflowError } from '@langchain/core/errors';
import { Runnable, RunnableConfig } from '@langchain/core/runnables';
import { START, END, StateGraph, Annotation } from '@langchain/langgraph';
import {
  ToolMessage,
  HumanMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import type {
  UsageMetadata,
  BaseMessage,
  MessageContent,
} from '@langchain/core/messages';
import type { BaseChannel, OverwriteValue } from '@langchain/langgraph';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  ReplayableSubagentTool,
  SubagentGraphResumeState,
  SubagentResumeManifest,
  SubagentToolNodeResumeState,
  SettledSubagentToolOutput,
} from '@/tools/subagent/SubagentReplay';
import type {
  ResolvedStreamLimits,
  RunBreakerScope,
  StreamedToolCallArgTally,
  StreamDeltaEventTally,
} from '@/llm/streamLimits';
import type {
  GraphFactory,
  GraphFactoryDependencies,
  GraphFactoryRequest,
} from '@/graphs/graphFactory';
import type { OverflowRecoveryPlan } from '@/llm/contextOverflowRecovery';
import type { FallbackErrorContext } from '@/llm/invoke';
import type { HandoffRouting } from './handoff';
import type { HookRegistry } from '@/hooks';
import type * as t from '@/types';
import {
  projectAnthropicArtifactContent,
  ensureThinkingBlockInMessages,
  foldToolBlocksForToollessAgent,
  convertMessagesToContent,
  sanitizeOrphanToolBlocks,
  extractToolDiscoveries,
  addBedrockTailCacheControl,
  projectArtifactPayload,
  formatContentStrings,
  cloneMessage,
  CALIBRATION_RATIO_MAX,
  createPruneMessages,
  projectToolMessagesForProvider,
  syncBudgetDerivedFields,
  addTailCacheControl,
  resolvePromptCacheTtl,
  resolveBedrockPromptCacheTtl,
  isSyntheticProviderContextMessage,
  compactSyntheticProviderContextMessage,
  getMessageId,
  getMessageCreationContentMetadata,
  splitAssistantTextContentByPhase,
  makeIsDeferred,
  DEFAULT_RETAIN_RECENT_TURNS,
  resolveIntraTurnRetainTokens,
  splitAtRecencyBoundary,
  convertInjectedMessages,
  coalesceAdjacentUserTurns,
  appendPredecessorHandoffCue,
  appendInstructionlessHandoffCue,
  stampSyntheticProviderMessage,
} from '@/messages';
import {
  Constants,
  GraphNodeKeys,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
  STANDARD_GRAPH_RUN_NAME,
  AGENT_MODEL_CALL_RUN_NAME,
  PREEMPT_BOUNDARY_HOOK_TIMEOUT_MS,
} from '@/common';
import {
  resetIfNotEmpty,
  isAnthropicLike,
  isOpenAILike,
  isGoogleLike,
  apportionTokenCounts,
  calculateMaxToolResultChars,
  composeAbortSignals,
  joinKeys,
  sleep,
} from '@/utils';
import {
  resolveStreamLimits,
  StreamLimitExceededError,
  sweepStaleStreamLimitEntries,
  STREAM_LIMIT_EPOCH_KEY,
  RUN_BREAKER_SCOPE_CONFIG_KEY,
} from '@/llm/streamLimits';
import {
  DEFAULT_SUBAGENT_DESCRIPTION,
  SubagentExecutor,
  isGraphSubagentConfig,
  normalizeSubagentConfigEntries,
} from '@/tools/subagent';
import {
  createLangfuseHandler,
  createLangfuseTraceMetadata,
  disposeLangfuseHandler,
  isLangfuseCallbackHandler,
} from '@/langfuse';
import {
  getBlindRecoveryBudget,
  planContextOverflowRecovery,
  translateRecoveryBudget,
} from '@/llm/contextOverflowRecovery';
import {
  attemptInvoke,
  tryFallbackProviders,
  getFallbackErrorContext,
  getFallbackOverflowCandidates,
} from '@/llm/invoke';
import {
  hasToolOutputTracingConfig,
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
} from '@/langfuseConfig';
import {
  annotateMessagesForLLM,
  ToolOutputReferenceRegistry,
} from '@/tools/toolOutputReferences';
import {
  prepareProviderRequest,
  usesNativeOpenAIResponses,
} from '@/llm/prepareProviderRequest';
import {
  resolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  ManualSummarizationSkippedError,
  shouldTriggerSummarization,
} from '@/summarization';
import {
  getToolContentCharLength,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  appendCallbacks,
  findCallback,
  type CallbackEntry,
} from '@/utils/callbacks';
import {
  PreparedSubagents,
  PreparedSubagentError,
} from '@/tools/preparedSubagents';
import {
  createRemoveAllMessage,
  messagesStateReducer,
} from '@/messages/reducer';
import { createToolHistoryPreparation } from '@/messages/toolHistoryProjection';
import { ToolNode as CustomToolNode, toolsCondition } from '@/tools/ToolNode';
import { shouldTraceToolNodeForLangfuse } from '@/langfuseToolOutputTracing';
import { createLocalCodingToolBundle } from '@/tools/local/LocalCodingTools';
import { SUBAGENT_REPLAY_CONTROLLER } from '@/tools/subagent/SubagentReplay';
import { applyGraphRuntimeConfig } from '@/graphs/applyGraphRuntimeConfig';
import { isFadingTier, isInformativeFadingTier } from '@/messages/fading';
import { createContextPressureMeter } from '@/llm/contextPressureMeter';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { createCloudflareCodingToolBundle } from '@/tools/cloudflare';
import { calculateMaxToolCallInputChars } from '@/utils/truncation';
import { prepareToolsForPromptCache } from '@/llm/promptCacheTools';
import { providerRequiresStrictAlternation } from '@/llm/providers';
import { buildSubagentToolParams } from '@/tools/SubagentTool';
import { initializeLangfuseTracing } from '@/instrumentation';
import { isRunStepResumeState } from '@/tools/runStepResume';
import { resolveLocalToolsForBinding } from '@/tools/local';
import { createSummarizeNode } from '@/summarization/node';
import { getTruncationStopReason } from '@/llm/truncation';
import { createSchemaOnlyTools } from '@/tools/schema';
import { AgentContext } from '@/agents/AgentContext';
import { createFakeStreamingLLM } from '@/llm/fake';
import { handoffStateAnnotation } from './handoff';
import { handleToolCalls } from '@/tools/handlers';
import { isThinkingEnabled } from '@/llm/request';
import { resolveMaxSeals } from '@/llm/preempt';
import { initializeModel } from '@/llm/init';
import { HandlerRegistry } from '@/events';
import { ChatOpenAI } from '@/llm/openai';
import { executeHooks } from '@/hooks';

const { AGENT, TOOLS, SUMMARIZE } = GraphNodeKeys;

/** What a `PreemptBoundary` drain resolved to. */
type PreemptBoundaryResult = {
  messages: BaseMessage[];
  /** A hook asked for no further model turn; the seal must not self-loop. */
  preventContinuation: boolean;
};

const EMPTY_PREEMPT_BOUNDARY: PreemptBoundaryResult = {
  messages: [],
  preventContinuation: false,
};

/** Minimum relative variance before calibrated toolSchemaTokens overrides current value. */
const CALIBRATION_VARIANCE_THRESHOLD = 0.15;

function createChildHandlerRegistry(
  source: HandlerRegistry | undefined
): HandlerRegistry | undefined {
  const toolHandler = source?.getHandler(GraphEvents.ON_TOOL_EXECUTE);
  const updateHandler = source?.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
  if (toolHandler == null && updateHandler == null) {
    return undefined;
  }
  const registry = new HandlerRegistry();
  if (toolHandler != null) {
    registry.register(GraphEvents.ON_TOOL_EXECUTE, toolHandler);
  }
  if (updateHandler != null) {
    registry.register(GraphEvents.ON_SUBAGENT_UPDATE, updateHandler);
  }
  return registry;
}

type ReasoningKey = 'reasoning_content' | 'reasoning';
type ReasoningSummary = { summary?: Array<{ text?: string }> };
type ReasoningDetail = { type?: string; text?: string };

function getHandlerDispatchedEventKey(
  eventName: string,
  stepId: string
): string {
  return `${eventName}:${stepId}`;
}

function getReasoningText(
  value: string | Partial<ReasoningSummary> | null | undefined
): string | undefined {
  if (typeof value === 'string') {
    return value !== '' ? value : undefined;
  }
  const summaryText = value?.summary
    ?.map((summary) => summary.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return summaryText != null && summaryText !== '' ? summaryText : undefined;
}

function getReasoningDetailsText(
  value: ReasoningDetail[] | null | undefined
): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const reasoningText = value
    .filter((detail) => detail.type === 'reasoning.text')
    .map((detail) => detail.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return reasoningText !== '' ? reasoningText : undefined;
}

function getResponseReasoningContent({
  responseMessage,
  reasoningKey,
}: {
  responseMessage?: Partial<AIMessageChunk>;
  reasoningKey: ReasoningKey;
}): string | undefined {
  const additionalKwargs = responseMessage?.additional_kwargs;
  if (additionalKwargs == null) {
    return undefined;
  }

  const keyedReasoning = getReasoningText(
    additionalKwargs[reasoningKey] as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (keyedReasoning != null) {
    return keyedReasoning;
  }

  const reasoningContent = getReasoningText(
    additionalKwargs.reasoning_content as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (reasoningContent != null) {
    return reasoningContent;
  }

  const reasoning = getReasoningText(
    additionalKwargs.reasoning as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (reasoning != null) {
    return reasoning;
  }

  return getReasoningDetailsText(
    additionalKwargs.reasoning_details as ReasoningDetail[] | null | undefined
  );
}

function isTextMessageContentPart(
  contentPart: MessageContent[number] | t.MessageContentComplex
): boolean {
  return (
    typeof contentPart === 'object' &&
    'type' in contentPart &&
    typeof contentPart.type === 'string' &&
    contentPart.type.startsWith('text')
  );
}

function isGoogleServerSideToolMessageContentPart(
  contentPart: MessageContent[number] | t.MessageContentComplex
): boolean {
  return (
    typeof contentPart === 'object' &&
    'type' in contentPart &&
    (contentPart.type === 'toolCall' || contentPart.type === 'toolResponse')
  );
}

function hasGoogleServerSideToolDeltaContent(
  provider: t.ProviderName | undefined,
  content: t.MessageDelta['content']
): content is t.MessageContentComplex[] {
  return (
    isGoogleLike(provider) &&
    Array.isArray(content) &&
    content.some((contentPart) =>
      isGoogleServerSideToolMessageContentPart(contentPart)
    )
  );
}

function getMessageDeltaContent(
  provider: t.ProviderName | undefined,
  content: MessageContent | undefined
): t.MessageDelta['content'] | undefined {
  if (content == null) {
    return undefined;
  }
  if (typeof content === 'string') {
    return content !== ''
      ? [{ type: ContentTypes.TEXT, text: content }]
      : undefined;
  }
  if (content.length === 0) {
    return undefined;
  }

  const hasGoogleServerSideToolPart =
    isGoogleLike(provider) &&
    content.some((contentPart) =>
      isGoogleServerSideToolMessageContentPart(contentPart)
    );
  if (content.every((contentPart) => isTextMessageContentPart(contentPart))) {
    return content as t.MessageDelta['content'];
  }
  if (!hasGoogleServerSideToolPart) {
    return undefined;
  }
  const messageContent = content.filter(
    (contentPart) =>
      isTextMessageContentPart(contentPart) ||
      isGoogleServerSideToolMessageContentPart(contentPart)
  );
  return messageContent.length > 0
    ? (messageContent as t.MessageDelta['content'])
    : undefined;
}

function hasTextDeltaContent(
  content: t.MessageDelta['content'] | undefined
): boolean {
  if (content == null) {
    return false;
  }
  return content.some((contentPart) => {
    if (contentPart.type?.startsWith(ContentTypes.TEXT) !== true) {
      return false;
    }
    const text = (contentPart as Partial<{ text: string }>).text;
    return typeof text === 'string' && text !== '';
  });
}

function hasReasoningDeltaContent(
  content: t.ReasoningDelta['content'] | undefined
): boolean {
  if (content == null) {
    return false;
  }
  return content.some(
    (contentPart) =>
      contentPart.type === ContentTypes.THINK && contentPart.think !== ''
  );
}

function getCurrentStepIds({
  graph,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  metadata: Record<string, unknown>;
}): string[] {
  const baseStepKey = graph.getStepBaseKey(metadata);
  const currentStepIds: string[] = [];
  for (const [stepKey, stepIds] of graph.stepKeyIds) {
    if (stepKey !== baseStepKey && !stepKey.startsWith(`${baseStepKey}_`)) {
      continue;
    }
    for (const stepId of stepIds) {
      currentStepIds.push(stepId);
    }
  }
  return currentStepIds;
}

function hasCurrentTextDeltaStep({
  graph,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  metadata: Record<string, unknown>;
}): boolean {
  return getCurrentStepIds({ graph, metadata }).some((stepId) =>
    graph.messageStepHasTextDeltas.has(stepId)
  );
}

function hasCurrentReasoningDeltaStep({
  graph,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  metadata: Record<string, unknown>;
}): boolean {
  return getCurrentStepIds({ graph, metadata }).some((stepId) =>
    graph.reasoningStepHasDeltas.has(stepId)
  );
}

function clearCurrentDeltaStepMarkers({
  graph,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  metadata: Record<string, unknown>;
}): void {
  for (const stepId of getCurrentStepIds({ graph, metadata })) {
    graph.messageStepHasTextDeltas.delete(stepId);
    graph.reasoningStepHasDeltas.delete(stepId);
  }
}

/**
 * The completion allowance the caller configured, under whichever key the
 * provider's client uses. Providers count it against the same ceiling as the
 * prompt, so overflow recovery has to reserve it when the error did not
 * itemize the total.
 */
function getConfiguredCompletionTokens(
  clientOptions: t.ClientOptions | undefined
): number | undefined {
  const options = clientOptions as
    | { maxTokens?: unknown; maxOutputTokens?: unknown }
    | undefined;
  for (const value of [options?.maxTokens, options?.maxOutputTokens]) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  return undefined;
}

/**
 * Our own estimate of the prompt that was actually sent, derived from the
 * pre-invoke usage snapshot. Used to corroborate ambiguous provider errors
 * and to measure how far our token accounting sits from the provider's.
 */
function getEstimatedPromptTokens(
  contextUsage: t.ContextUsageEvent | null
): number | undefined {
  const budget = contextUsage?.contextBudget;
  const remaining = contextUsage?.remainingContextTokens;
  if (
    budget == null ||
    remaining == null ||
    !Number.isFinite(budget) ||
    !Number.isFinite(remaining)
  ) {
    return undefined;
  }
  const used = budget - remaining;
  return used > 0 ? used : undefined;
}

function minDefined(
  left: number | undefined,
  right: number | undefined
): number | undefined {
  if (left == null) {
    return right;
  }
  if (right == null) {
    return left;
  }
  return Math.min(left, right);
}

async function dispatchMessageCreationStep({
  graph,
  stepKey,
  messageId,
  content,
  contentType,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  stepKey: string;
  messageId: string;
  content?: string | t.MessageContentComplex[];
  contentType?: ContentTypes.TEXT | ContentTypes.THINK;
  metadata: Record<string, unknown>;
}): Promise<string> {
  await graph.dispatchRunStep(
    stepKey,
    {
      type: StepTypes.MESSAGE_CREATION,
      message_creation: {
        message_id: messageId,
        ...getMessageCreationContentMetadata(content, contentType),
      },
    },
    metadata
  );
  return graph.getStepIdByKey(stepKey);
}

async function dispatchTextMessageContent({
  graph,
  stepKey,
  provider,
  content,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  stepKey: string;
  provider?: t.ProviderName;
  content: t.MessageDelta['content'];
  metadata: Record<string, unknown>;
}): Promise<boolean> {
  const messageId = getMessageId(stepKey, graph) ?? '';
  if (!messageId) {
    return false;
  }
  if (hasGoogleServerSideToolDeltaContent(provider, content)) {
    for (const contentPart of content) {
      const stepId = await dispatchMessageCreationStep({
        graph,
        stepKey,
        messageId,
        content: [contentPart],
        metadata,
      });
      await graph.dispatchMessageDelta(
        stepId,
        { content: [contentPart] },
        metadata
      );
    }
    return true;
  }
  const contentGroups = Array.isArray(content)
    ? splitAssistantTextContentByPhase(content)
    : [content];
  for (const contentGroup of contentGroups) {
    const stepId = await dispatchMessageCreationStep({
      graph,
      stepKey,
      messageId,
      content: contentGroup,
      contentType: ContentTypes.TEXT,
      metadata,
    });
    await graph.dispatchMessageDelta(
      stepId,
      { content: contentGroup },
      metadata
    );
  }
  return true;
}

async function dispatchReasoningContent({
  graph,
  agentContext,
  reasoningContent,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  agentContext: AgentContext;
  reasoningContent: string;
  metadata: Record<string, unknown>;
}): Promise<boolean> {
  const previousTokenType = agentContext.currentTokenType;
  const previousTokenTypeSwitch = agentContext.tokenTypeSwitch;
  const previousTransitionCount = agentContext.reasoningTransitionCount;

  agentContext.currentTokenType = ContentTypes.THINK;
  agentContext.tokenTypeSwitch = 'reasoning';

  const stepKey = graph.getStepKey(metadata);
  const messageId = getMessageId(stepKey, graph) ?? '';
  if (!messageId) {
    agentContext.currentTokenType = previousTokenType;
    agentContext.tokenTypeSwitch = previousTokenTypeSwitch;
    agentContext.reasoningTransitionCount = previousTransitionCount;
    return false;
  }

  await graph.dispatchRunStep(
    stepKey,
    {
      type: StepTypes.MESSAGE_CREATION,
      message_creation: {
        message_id: messageId,
        content_type: ContentTypes.THINK,
      },
    },
    metadata
  );
  const stepId = graph.getStepIdByKey(stepKey);
  await graph.dispatchReasoningDelta(
    stepId,
    {
      content: [{ type: ContentTypes.THINK, think: reasoningContent }],
    },
    metadata
  );
  return true;
}

function markPostReasoningContent(agentContext: AgentContext): void {
  if (
    agentContext.tokenTypeSwitch !== 'reasoning' ||
    agentContext.currentTokenType === ContentTypes.TEXT
  ) {
    return;
  }
  agentContext.currentTokenType = ContentTypes.TEXT;
  agentContext.tokenTypeSwitch = 'content';
  agentContext.reasoningTransitionCount++;
}

function getDispatchableFinalReasoningContent({
  agentContext,
  responseReasoningContent,
  hasStreamedTextDeltaStep,
  hasStreamedReasoningDeltaStep,
}: {
  agentContext: AgentContext;
  responseReasoningContent: string | undefined;
  hasStreamedTextDeltaStep: boolean;
  hasStreamedReasoningDeltaStep: boolean;
}): string | undefined {
  if (responseReasoningContent == null || hasStreamedReasoningDeltaStep) {
    return undefined;
  }
  if (
    agentContext.provider === Providers.OPENROUTER &&
    hasStreamedTextDeltaStep
  ) {
    return undefined;
  }
  return responseReasoningContent;
}

function createEmptyRunStepResumeState(): t.RunStepResumeState {
  return {
    version: 1,
    revision: 0,
    nextIndex: 0,
    toolCallSteps: [],
    steps: [],
  };
}

type RunStepStateChannel = BaseChannel<
  t.RunStepResumeState,
  t.RunStepResumeState | OverwriteValue<t.RunStepResumeState>
>;

function buildRunStepStateAnnotation(): RunStepStateChannel {
  return Annotation<t.RunStepResumeState>({
    reducer: (current, update) =>
      update.revision >= current.revision ? update : current,
    default: createEmptyRunStepResumeState,
  });
}

export abstract class Graph<
  T extends t.BaseGraphState = t.BaseGraphState,
  _TNodeName extends string = string,
> {
  abstract resetValues(keepContent?: boolean, checkpointScope?: string): void;
  abstract createRunStepResumeState(): t.RunStepResumeState;
  abstract restoreRunStepResumeState(state?: t.RunStepResumeState): void;
  restoreCheckpointMessages(
    _messages: BaseMessage[],
    _pendingMessages?: BaseMessage[]
  ): void {}
  abstract initializeTools({
    currentTools,
    currentToolMap,
  }: {
    currentTools?: t.GraphTools;
    currentToolMap?: t.ToolMap;
  }): CustomToolNode<T> | ToolNode<T>;
  abstract getRunMessages(): BaseMessage[] | undefined;
  /** Returns a snapshot of deferred tools discovered by this graph. */
  getDiscoveredTools(_agentId?: string): string[] {
    return [];
  }
  abstract getContentParts(): t.MessageContentComplex[] | undefined;
  abstract generateStepId(stepKey: string): [string, number];
  abstract getKeyList(
    metadata: Record<string, unknown> | undefined
  ): (string | number | undefined)[];
  abstract getStepBaseKey(
    metadata: Record<string, unknown> | undefined
  ): string;
  abstract getStepKey(metadata: Record<string, unknown> | undefined): string;
  abstract checkKeyList(keyList: (string | number | undefined)[]): boolean;
  abstract getStepIdByKey(stepKey: string, index?: number): string;
  abstract getRunStep(stepId: string): t.RunStep | undefined;
  abstract dispatchRunStep(
    stepKey: string,
    stepDetails: t.StepDetails,
    metadata?: Record<string, unknown>
  ): Promise<string>;
  abstract dispatchRunStepDelta(
    id: string,
    delta: t.ToolCallDelta,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  abstract dispatchMessageDelta(
    id: string,
    delta: t.MessageDelta,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  abstract dispatchReasoningDelta(
    stepId: string,
    delta: t.ReasoningDelta,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  abstract createCallModel(
    agentId?: string,
    currentModel?: t.ChatModel
  ): (
    state: t.AgentSubgraphState,
    config?: RunnableConfig
  ) => Promise<Partial<t.AgentSubgraphState>>;
  messageStepHasTextDeltas: Set<string> = new Set();
  messageStepHasToolCalls: Map<string, boolean> = new Map();
  messageIdsByStepKey: Map<string, string> = new Map();
  prelimMessageIdsByStepKey: Map<string, string> = new Map();
  config: RunnableConfig | undefined;
  contentData: t.RunStep[] = [];
  protected nextContentIndex = 0;
  protected runStepStateRevision = 0;
  protected stopContinuationCount = 0;
  protected stopContinuationExecutionId = '';
  protected streamSegment = 0;
  stepKeyIds: Map<string, string[]> = new Map<string, string[]>();
  contentIndexMap: Map<string, number> = new Map();
  toolCallStepIds: Map<string, string> = new Map();
  /** Step ID -> tool call IDs whose completions have not yet arrived. */
  pendingToolCallsByStep: Map<string, Set<string>> = new Map();
  /**
   * Step ID -> latest producer completion time seen for that step. Parallel
   * calls sharing a step can settle out of producer order when their host
   * handlers differ in latency, so the call that happens to drain the set is
   * not necessarily the one that finished last.
   */
  latestCompletionByStep: Map<string, number> = new Map();
  /** Agent key ('' for single-agent) -> currently open MESSAGE_CREATION step ID. */
  openMessageStepByAgent: Map<string, string> = new Map();
  /**
   * Step IDs dispatched through the handler registry during this run.
   * Event echo suppression is tracked separately so repeated deltas for
   * the same step are scoped to the active custom event dispatch.
   */
  handlerDispatchedStepIds: Set<string> = new Set();
  reasoningStepHasDeltas: Set<string> = new Set();
  protected handlerDispatchedEventCounts: Map<string, number> = new Map();
  signal?: AbortSignal;
  /**
   * The abort signal the CALLER handed to the current `processStream` call,
   * assigned unconditionally — including back to `undefined` — on every call.
   *
   * Kept separate from {@link signal} on purpose. That field is construction
   * state with its own consumers (model-call config, subagent parentSignal),
   * so adopting a per-call signal into it would leak one call's controller
   * into the next — `clearHeavyState()` is skipped on HITL interrupts, so a
   * host that aborts a finished request's controller would poison the resumed
   * run's model calls and boundary drains with an already-aborted signal.
   * Boundary dispatch composes the two instead; see
   * `StandardGraph.dispatchPreemptBoundary`.
   */
  callerSignal?: AbortSignal;
  /** Set of invoked tool call IDs from non-message run steps completed mid-run, if any */
  invokedToolIds?: Set<string>;
  handlerRegistry: HandlerRegistry | undefined;
  /** Host registry retained only for forwarding tools from nested child graphs. */
  protected parentToolHandlerRegistry: HandlerRegistry | undefined;
  /**
   * True when event-driven tool execution can be routed through callbacks even
   * though this graph intentionally does not own the full handler registry.
   * Self-spawned subagent graphs use this shape: their callback forwarder sends
   * `ON_TOOL_EXECUTE` to the parent's handler, while child run-step events stay
   * wrapped as `ON_SUBAGENT_UPDATE` instead of leaking as parent events.
   */
  eventToolExecutionAvailable: boolean = false;
  hookRegistry: HookRegistry | undefined;
  /**
   * Run-scoped HITL configuration. When `humanInTheLoop?.enabled` is
   * `true`, `ToolNode` raises a real `interrupt()` for `PreToolUse`
   * `ask` decisions instead of treating them as a synchronous deny.
   * Threaded from `RunConfig.humanInTheLoop`.
   */
  humanInTheLoop: t.HumanInTheLoopConfig | undefined;
  /**
   * Run-scoped config for the tool output reference registry. Threaded
   * from `RunConfig.toolOutputReferences` down into every ToolNode this
   * graph compiles.
   */
  toolOutputReferences: t.ToolOutputReferencesConfig | undefined;
  /**
   * Run-scoped Langfuse defaults. Per-agent config wins when present.
   */
  langfuse: t.LangfuseConfig | undefined;
  /**
   * Run-scoped opt-in for eager event-driven tool execution. The stream
   * handler may prestart eligible event-driven tools; ToolNode later
   * consumes the settled promises while preserving final ToolMessage order.
   */
  eagerEventToolExecution: t.EagerEventToolExecutionConfig | undefined;
  codeSessionToolNames: string[] | undefined;
  /**
   * Run-scoped names of tools whose in-process body may raise a LangGraph
   * `interrupt()` (e.g. `ask_user_question`). Threaded from
   * `RunConfig.interruptingToolNames` into every ToolNode this graph
   * compiles so a mid-batch interrupt cannot double-execute non-idempotent
   * siblings on resume. See {@link t.ToolNodeOptions.interruptingToolNames}.
   */
  interruptingToolNames: string[] | undefined;
  eagerEventToolExecutions: Map<string, t.EagerEventToolExecution> = new Map();
  eagerEventToolUsageCount: Map<string, number> = new Map();
  private eagerEventToolUsageCountsByAgentId: Map<string, Map<string, number>> =
    new Map();
  eagerEventToolCallChunks: Map<string, t.EagerEventToolCallChunkState> =
    new Map();
  /**
   * Per-run eager prestart circuit breaker, shared by reference with every
   * ToolNode this graph compiles. When a prestarted execution's args turn
   * out to differ from the final request, ToolNode records the tool name
   * here and the stream handler stops prestarting that tool for the rest of
   * the run — the retry then executes normally instead of re-diverging in a
   * loop (LibreChat#14371).
   */
  eagerEventToolSuppressions: Set<string> = new Set();
  /**
   * Run-scoped execution backend for built-in code tools. Defaults to the
   * remote Code API sandbox when unset.
   */
  toolExecution: t.ToolExecutionConfig | undefined;
  /**
   * Shared registry instance used by every ToolNode compiled from this
   * graph. Lazily constructed on first access so multi-agent graphs
   * produce one registry per run (not one per agent), letting cross-
   * agent `{{tool<i>turn<n>}}` substitutions resolve.
   */
  private _toolOutputRegistry?: ToolOutputReferenceRegistry;
  /**
   * Tool session contexts for automatic state persistence across tool invocations.
   * Keyed by tool name (e.g., Constants.EXECUTE_CODE).
   * Currently supports code execution session tracking (session_id, files).
   */
  sessions: t.ToolSessionMap = new Map();

  /**
   * Clears heavy references to allow GC to reclaim memory held by
   * LangGraph's internal config / AsyncLocalStorage RunTree chain.
   * Call after a run completes and content has been extracted.
   */
  clearHeavyState(): void {
    this.preparedSubagents.clear();
    this.config = undefined;
    this.signal = undefined;
    this.callerSignal = undefined;
    this.contentData = [];
    this.nextContentIndex = 0;
    this.runStepStateRevision = 0;
    this.stopContinuationCount = 0;
    this.stopContinuationExecutionId = '';
    this.streamSegment = 0;
    this.contentIndexMap = new Map();
    this.stepKeyIds = new Map();
    this.toolCallStepIds.clear();
    this.pendingToolCallsByStep.clear();
    this.latestCompletionByStep.clear();
    this.openMessageStepByAgent.clear();
    this.messageIdsByStepKey = new Map();
    this.messageStepHasTextDeltas = new Set();
    this.reasoningStepHasDeltas = new Set();
    this.messageStepHasToolCalls = new Map();
    this.prelimMessageIdsByStepKey = new Map();
    this.invokedToolIds = undefined;
    this.handlerRegistry = undefined;
    this.parentToolHandlerRegistry = undefined;
    this.hookRegistry = undefined;
    this.humanInTheLoop = undefined;
    this.toolOutputReferences = undefined;
    this.eagerEventToolExecution = undefined;
    this.codeSessionToolNames = undefined;
    this.interruptingToolNames = undefined;
    this.eagerEventToolExecutions.clear();
    this.clearEagerEventToolUsageCounts();
    this.eagerEventToolCallChunks.clear();
    this.eagerEventToolSuppressions.clear();
    this.toolExecution = undefined;
    this.handlerDispatchedEventCounts.clear();
    /**
     * ToolNodes compiled from this graph captured the registry
     * instance at construction time, so simply dropping the Graph's
     * own reference would leave their captured reference — and every
     * stored `tool<i>turn<n>` entry, plus up to `maxTotalSize` of raw
     * output — alive across subsequent `processStream()` calls. Wipe
     * the registry's contents first so subsequent runs start fresh.
     */
    this._toolOutputRegistry?.clear();
    this._toolOutputRegistry = undefined;
    // NB: `_fileCheckpointer` is intentionally NOT cleared here.
    // `Run.processStream()` calls `clearHeavyState()` in its
    // finally block on natural-completion / error paths — exactly
    // when the host is most likely to want `Run.rewindFiles()` (for
    // rollback after a failed batch). Per-Run isolation is already
    // automatic because each `Run.create()` constructs a brand-new
    // Graph instance, so the next Run gets its own checkpointer
    // without us needing to reset this field. Codex P1 #32: pre-fix
    // the checkpointer was nulled before the caller could reach it.
    // Flush each compiled ToolNode's direct-path turn cache so it
    // doesn't leak across Runs (Codex P2 #33). The cache survives
    // `run()` re-entry by design (resume-stable), but end-of-Run
    // is the right point to reset it. Retain the registrations because
    // the compiled workflow can be reused for later Runs; compilation
    // will not register these instances again.
    for (const node of this._compiledToolNodes) {
      node.clearDirectPathTurns();
    }
    // Subagent executors are likewise compiled once and reused. Clear
    // their per-Run state without dropping the registrations needed by
    // subsequent cleanup cycles.
    for (const executor of this._subagentExecutors) {
      executor.clearHeavyState();
    }
    this.sessions.clear();
  }

  getEagerEventToolUsageCount(agentId?: string): Map<string, number> {
    if (agentId == null || agentId === '') {
      return this.eagerEventToolUsageCount;
    }
    let usageCount = this.eagerEventToolUsageCountsByAgentId.get(agentId);
    if (usageCount == null) {
      usageCount = new Map<string, number>();
      this.eagerEventToolUsageCountsByAgentId.set(agentId, usageCount);
    }
    return usageCount;
  }

  protected clearEagerEventToolUsageCounts(): void {
    this.eagerEventToolUsageCount.clear();
    for (const usageCount of this.eagerEventToolUsageCountsByAgentId.values()) {
      usageCount.clear();
    }
  }

  /**
   * Tracks a tool call whose completion must arrive before its step can be
   * considered finished. Registered wherever `toolCallStepIds` gains entries.
   */
  registerPendingToolCall(toolCallId: string, stepId: string): void {
    if (!toolCallId || !stepId) {
      return;
    }
    const pending = this.getPendingToolCallSet(stepId);
    const size = pending.size;
    pending.add(toolCallId);
    if (pending.size !== size) {
      this.runStepStateRevision += 1;
    }
  }

  /** Lazily creates a step's pending-completions set; callers registering a
   *  batch hoist this lookup out of their per-call loop. */
  protected getPendingToolCallSet(stepId: string): Set<string> {
    let pending = this.pendingToolCallsByStep.get(stepId);
    if (!pending) {
      pending = new Set();
      this.pendingToolCallsByStep.set(stepId, pending);
    }
    return pending;
  }

  markHandlerDispatchedEvent(eventName: string, stepId: string): () => void {
    const key = getHandlerDispatchedEventKey(eventName, stepId);
    this.handlerDispatchedEventCounts.set(
      key,
      (this.handlerDispatchedEventCounts.get(key) ?? 0) + 1
    );
    return () => {
      const count = this.handlerDispatchedEventCounts.get(key) ?? 0;
      if (count <= 1) {
        this.handlerDispatchedEventCounts.delete(key);
        return;
      }
      this.handlerDispatchedEventCounts.set(key, count - 1);
    };
  }

  hasHandlerDispatchedEvent(eventName: string, stepId: string): boolean {
    const key = getHandlerDispatchedEventKey(eventName, stepId);
    return (this.handlerDispatchedEventCounts.get(key) ?? 0) > 0;
  }

  /**
   * Subclass hook to register a freshly compiled ToolNode so
   * `clearHeavyState` can flush its per-Run direct-path turn cache
   * at end-of-Run. Internal — called from `initializeTools` in the
   * concrete graph subclasses.
   */
  protected registerCompiledToolNode(node: {
    clearDirectPathTurns(): void;
    createSubagentResumeState(): SubagentToolNodeResumeState;
    restoreSubagentResumeState(state: SubagentToolNodeResumeState): void;
  }): void {
    this._compiledToolNodes.add(node);
  }

  protected registerSubagentExecutor(executor: SubagentExecutor): void {
    this._subagentExecutors.add(executor);
  }

  protected resetSubagentCheckpointThreadIds(): void {
    for (const executor of this._subagentExecutors) {
      executor.resetCheckpointThreadIds();
    }
  }

  getChildCheckpointThreadIds(): string[] {
    const threadIds = new Set<string>();
    for (const executor of this._subagentExecutors) {
      for (const threadId of executor.getChildCheckpointThreadIds()) {
        threadIds.add(threadId);
      }
    }
    return [...threadIds];
  }

  protected createSubagentFadingResumeState(): Pick<
    SubagentGraphResumeState,
    'fadingTier' | 'fadingTiers'
    > {
    return {};
  }

  createSubagentResumeState(runId: string): SubagentGraphResumeState {
    return {
      toolCallSteps: [...this.toolCallStepIds].map(([toolCallId, stepId]) => ({
        toolCallId,
        stepId,
      })),
      toolSessions: [...this.sessions].map(([toolName, context]) => ({
        toolName,
        context: {
          ...context,
          ...(context.files == null
            ? {}
            : { files: context.files.map((file) => ({ ...file })) }),
        },
      })),
      toolNodes: [...this._compiledToolNodes].map((node) =>
        node.createSubagentResumeState()
      ),
      eagerToolUsage: [
        {
          agentId: '',
          toolUsageCounts: [...this.eagerEventToolUsageCount].map(
            ([toolName, count]) => ({ toolName, count })
          ),
        },
        ...[...this.eagerEventToolUsageCountsByAgentId].map(
          ([agentId, usageCounts]) => ({
            agentId,
            toolUsageCounts: [...usageCounts].map(([toolName, count]) => ({
              toolName,
              count,
            })),
          })
        ),
      ],
      eagerToolSuppressions: [...this.eagerEventToolSuppressions],
      runStepState: this.createRunStepResumeState(),
      ...this.createSubagentFadingResumeState(),
      ...(this._toolOutputRegistry == null
        ? {}
        : {
          toolOutputReferences: this._toolOutputRegistry.snapshotState(runId),
        }),
    };
  }

  restoreSubagentResumeState(
    state: SubagentGraphResumeState,
    runId: string
  ): void {
    const toolNodesByKey = new Map(
      [...this._compiledToolNodes].map((node) => {
        const nodeState = node.createSubagentResumeState();
        return [nodeState.stateKey, node] as const;
      })
    );
    if (toolNodesByKey.size !== state.toolNodes.length) {
      throw new Error('Cannot restore changed subagent tool topology.');
    }
    for (const nodeState of state.toolNodes) {
      if (!toolNodesByKey.has(nodeState.stateKey)) {
        throw new Error(
          `Cannot restore subagent tool state for "${nodeState.stateKey}".`
        );
      }
    }
    const registry =
      state.toolOutputReferences == null
        ? undefined
        : this.getOrCreateToolOutputRegistry();
    if (state.toolOutputReferences != null && registry == null) {
      throw new Error('Cannot restore disabled tool output references.');
    }

    this.restoreRunStepResumeState(state.runStepState);
    this.toolCallStepIds.clear();
    for (const { toolCallId, stepId } of state.toolCallSteps) {
      this.toolCallStepIds.set(toolCallId, stepId);
    }
    this.sessions.clear();
    for (const { toolName, context } of state.toolSessions) {
      this.sessions.set(toolName, {
        ...context,
        ...(context.files == null
          ? {}
          : { files: context.files.map((file) => ({ ...file })) }),
      });
    }
    for (const nodeState of state.toolNodes) {
      const node = toolNodesByKey.get(nodeState.stateKey)!;
      node.restoreSubagentResumeState(nodeState);
    }
    this.clearEagerEventToolUsageCounts();
    for (const usageState of state.eagerToolUsage) {
      const usageCounts = this.getEagerEventToolUsageCount(usageState.agentId);
      for (const { toolName, count } of usageState.toolUsageCounts) {
        usageCounts.set(toolName, count);
      }
    }
    this.eagerEventToolSuppressions.clear();
    for (const toolName of state.eagerToolSuppressions) {
      this.eagerEventToolSuppressions.add(toolName);
    }
    if (state.toolOutputReferences != null && registry != null) {
      registry.restoreState(runId, state.toolOutputReferences);
    }
  }

  /**
   * Returns the shared `ToolOutputReferenceRegistry` for this run,
   * constructing it on first access. Returns `undefined` when the
   * feature is disabled. All ToolNodes compiled from this graph share
   * this single instance so cross-agent `{{…}}` references resolve.
   *
   * @internal Public so `attemptInvoke` can read it through the typed
   * `InvokeContext` and project ToolMessages into LLM-facing annotated
   * copies right before each provider call (see
   * `annotateMessagesForLLM`). Host code should not call this directly
   * — registry mutations outside the ToolNode lifecycle break the
   * partitioning, eviction, and turn-counter invariants.
   */
  public getOrCreateToolOutputRegistry():
    | ToolOutputReferenceRegistry
    | undefined {
    if (this.toolOutputReferences?.enabled !== true) {
      return undefined;
    }
    if (this._toolOutputRegistry == null) {
      this._toolOutputRegistry = new ToolOutputReferenceRegistry({
        maxOutputSize: this.toolOutputReferences.maxOutputSize,
        maxTotalSize: this.toolOutputReferences.maxTotalSize,
      });
    }
    return this._toolOutputRegistry;
  }

  /**
   * Single per-Run file checkpointer shared across every ToolNode the
   * graph compiles. Lazily constructed when
   * `toolExecution.local.fileCheckpointing === true` or
   * `toolExecution.cloudflare.fileCheckpointing === true` so
   * multi-agent graphs see ONE snapshot store, not one-per-agent.
   * Returns undefined when checkpointing is disabled or a supported
   * coding-tool engine isn't selected. Exposed via
   * `Run.getFileCheckpointer()` / `Run.rewindFiles()`.
   */
  private _fileCheckpointer?: t.LocalFileCheckpointer;
  /**
   * ToolNodes compiled into this Graph's workflow. Tracked so
   * `clearHeavyState()` can flush their per-Run direct-path turn
   * cache (`directPathTurns`) at end-of-Run — that map intentionally
   * survives `run()` re-entry (resume-stable per Codex P2 #30) but
   * would otherwise grow linearly with tool calls and could collide
   * across Runs if a provider reuses call ids (Codex P2 #33).
   */
  private _compiledToolNodes: Set<{
    clearDirectPathTurns(): void;
    createSubagentResumeState(): SubagentToolNodeResumeState;
    restoreSubagentResumeState(state: SubagentToolNodeResumeState): void;
  }> = new Set();
  private _subagentExecutors = new Set<SubagentExecutor>();
  readonly preparedSubagents = new PreparedSubagents();
  protected readonly subagentToolNodes = new Map<
    string,
    CustomToolNode<t.BaseGraphState>
  >();
  public getOrCreateFileCheckpointer(): t.LocalFileCheckpointer | undefined {
    // Return the cached instance unconditionally if one exists. The
    // toolExecution check below decides whether to *create* a new
    // one — `clearHeavyState` nulls `this.toolExecution` at end-of-
    // Run, but we want post-Run `Run.rewindFiles()` to still resolve
    // to the checkpointer that captured the writes. Codex P1 #32.
    if (this._fileCheckpointer != null) {
      return this._fileCheckpointer;
    }
    // Eagerly create via the bundle factory so the construction path
    // matches the bundle-only callers (and future bundle-internal
    // cleanup hooks fire). The bundle factory itself accepts a pre-
    // supplied checkpointer when present, so re-injecting this one
    // into every ToolNode is idempotent.
    if (
      this.toolExecution?.engine === 'local' &&
      this.toolExecution.local?.fileCheckpointing === true
    ) {
      const bundle = createLocalCodingToolBundle(
        this.toolExecution.local ?? {}
      );
      this._fileCheckpointer = bundle.checkpointer;
      return this._fileCheckpointer;
    }
    if (
      this.toolExecution?.engine === 'cloudflare-sandbox' &&
      this.toolExecution.cloudflare?.fileCheckpointing === true
    ) {
      const bundle = createCloudflareCodingToolBundle(
        this.toolExecution.cloudflare
      );
      this._fileCheckpointer = bundle.checkpointer;
      return this._fileCheckpointer;
    }
    return undefined;
  }
}

export class StandardGraph extends Graph<t.BaseGraphState, t.GraphNode> {
  readonly runName: string = STANDARD_GRAPH_RUN_NAME;
  overrideModel?: t.ChatModel;
  private subagentModelOverride?: t.ChatModel;
  private readonly graphFactory: GraphFactory;
  private readonly supportsMultiAgentChildren: boolean;
  /** Optional compile options passed into workflow.compile() */
  compileOptions?: t.CompileOptions | undefined;
  /** Whether the workflow was actually compiled with a checkpointer. */
  hasCompiledCheckpointer: boolean = false;
  messages: BaseMessage[] = [];
  /** Whether a rebuilt resume seeded the message baseline from its checkpoint. */
  private hasRestoredCheckpointMessages = false;
  /** Cached run messages preserved before clearHeavyState() so getRunMessages() works after cleanup. */
  private cachedRunMessages?: BaseMessage[];
  /** Per-agent discovery snapshots preserved before contexts are reset on cleanup. */
  private cachedDiscoveredTools?: Map<string, string[]>;
  /** Ids of AI turns the agent node returned THIS run; see isRunProducedMessage. */
  protected runProducedAiMessageIds = new Set<string>();
  /** Checkpoint scope whose messages match index-keyed tool snapshots. */
  private originalToolContentCheckpointScope?: string;
  runId: string | undefined;
  /**
   * Identity used to stamp Langfuse runtime scopes and handlers (see
   * `LangfuseRuntimeContext.runId`). Carries an opaque per-instance
   * component: public run ids are unrestricted and may repeat across
   * concurrently executing runs (retries, duplicate submissions,
   * tenant-local message ids), and equal stamps would let those runs adopt
   * each other's scopes. Fresh stream executions rotate this stamp; resumes
   * retain it so the continuation stays in the interrupted execution.
   */
  langfuseScopeRunId: string;
  /** Opaque key for parenting post-processing observations to this run. */
  langfuseTraceAnchor: object = {};
  /**
   * Boundary between historical messages (loaded from conversation state)
   * and messages produced during the current run.  Set once in the state
   * reducer when messages first arrive.  Used by `getRunMessages()` and
   * multi-agent message filtering — NOT for pruner token counting (the
   * pruner maintains its own `lastTurnStartIndex` in its closure).
   */
  startIndex: number = 0;
  signal?: AbortSignal;
  /** Map of agent contexts by agent ID */
  agentContexts: Map<string, AgentContext> = new Map();
  /** Default agent ID to use */
  defaultAgentId: string;
  /**
   * Host sink for model usage emitted inside subagent child runs. Threaded
   * into each `SubagentExecutor` this graph creates (and from there into
   * child graphs, so nested subagents report too). See
   * {@link t.StandardGraphInput.subagentUsageSink}.
   */
  subagentUsageSink?: t.SubagentUsageSink;
  /** See {@link t.StandardGraphInput.subagentScope}. */
  subagentScope: boolean;
  handoffRouting?: HandoffRouting;
  /** See {@link t.StandardGraphInput.subagentTasks}. */
  subagentTasks: t.SubagentTaskConfig | undefined;
  /** See {@link t.StandardGraphInput.subagentExecutionContext}. */
  readonly subagentExecutionContext?: t.SubagentExecutionContext;
  private readonly subagentContext?: t.SubagentContextAdapter;
  /** See {@link t.StandardGraphInput.preemption}. */
  preemption?: t.StreamPreemption;
  /**
   * Stream circuit breakers, resolved once from
   * {@link t.StandardGraphInput.streamLimits}. The stream handler enforces
   * these on every streamed chunk event.
   */
  streamLimits: ResolvedStreamLimits;
  /**
   * Cumulative streamed argument bytes per in-flight tool call, keyed by
   * generation key + chunk index (see `resolveGenerationKey`). Per-run
   * accumulation state, cleared by both reset paths.
   */
  streamedToolCallArgTallies: Map<string, StreamedToolCallArgTally> = new Map();
  /** Streamed chunk events per model generation, keyed by generation key. */
  streamDeltaEventCounts: Map<string, StreamDeltaEventTally> = new Map();
  /** Per-chunk-object, per-generation charge balances (lazily created; see
   * `StreamLimitState`). Reinitialized by both reset paths: a model may
   * retain and re-yield one mutable chunk object, whose nested map would
   * otherwise grow by one attempt-stamped entry per model call for the
   * graph's lifetime. */
  streamLimitChargeCredits?: WeakMap<object, Map<string, number>>;
  /** Run-scoped breaker abort: composed into every model invocation and
   * every SubagentExecutor child signal, and tripped when a stream circuit
   * breaker fires anywhere in the run, so parallel agent nodes' in-flight
   * provider calls and subagents stop consuming quota while the rejection
   * propagates. Recreated by both reset paths — the abort is one-way within
   * a run, and a reused graph must start its next run unaborted. */
  breakerAbort = new AbortController();

  /** Incremented whenever `breakerAbort` is replaced. Stamped into each
   * model attempt's metadata ({@link STREAM_LIMIT_EPOCH_KEY}) so the stream
   * handler's consumer-side trip binds to the run that produced the event
   * rather than whichever controller is live when a straggling chunk is
   * finally handled. */
  breakerEpoch = 0;

  /** Immutable snapshot of the run's breaker identity (epoch + controller),
   * replaced as ONE object whenever `resetValues` installs a fresh
   * controller. Sites that pause across awaits capture it at entry and
   * revalidate by REFERENCE afterwards — a single identity comparison
   * proves no reset happened while suspended, where separate epoch and
   * controller reads could interleave with one. */
  runScope: RunBreakerScope = Object.freeze({
    epoch: 0,
    controller: this.breakerAbort,
  });

  /** Generation keys of model attempts still in flight (see
   * `StreamLimitState.activeStreamLimitGenerations`). Lazily created by the
   * attempt lease; spans resets on purpose. */
  activeStreamLimitGenerations?: Set<string>;

  /** The stream-limit error behind an already-fired breaker, whether this
   * graph's own controller tripped or a parent run's breaker arrived through
   * the composed constructor signal (child graphs own separate controllers).
   * Providers can translate either abort into a generic error, and recovery
   * paths must not run in that state. */
  protected resolveTrippedBreakerReason(
    breakerSignal: AbortSignal = this.breakerAbort.signal
  ): StreamLimitExceededError | PreparedSubagentError | undefined {
    if (
      breakerSignal.aborted &&
      (breakerSignal.reason instanceof StreamLimitExceededError ||
        breakerSignal.reason instanceof PreparedSubagentError)
    ) {
      return breakerSignal.reason;
    }
    if (
      this.signal?.aborted === true &&
      (this.signal.reason instanceof StreamLimitExceededError ||
        this.signal.reason instanceof PreparedSubagentError)
    ) {
      return this.signal.reason;
    }
    return undefined;
  }
  /**
   * Seals charged against `preemption.maxSeals`. Per-turn: cleared by both
   * reset paths so a fresh turn gets a fresh budget, while a HITL resume —
   * which skips `resetValues` — keeps what it had left.
   */
  private preemptSealBudgetUsed = 0;
  /**
   * Seals honored over the graph's lifetime. Reported by
   * {@link getPreemptStats}, so it deliberately SURVIVES `clearHeavyState()`
   * — a host reads it after `processStream` returns, which is strictly after
   * cleanup runs.
   */
  preemptSealCount = 0;
  /** Discards honored, counted apart from seals — see
   *  {@link claimPreemptRestart}. */
  preemptRestartCount = 0;
  /** Boundaries that produced nothing to inject, so the turn stopped early. */
  preemptEmptyBoundaries = 0;
  /**
   * Set between claiming a seal and resolving its boundary. `MultiAgentGraph`
   * fans parallel agents through this one instance against a single host
   * request, so without a one-at-a-time gate several streams would each seal
   * for the same queued message and every loser would take the
   * nothing-to-inject path and cut its answer short.
   */
  private preemptSealInFlight = false;
  /**
   * True when a seal ended the turn without a resume. The assistant turn is
   * real and kept, but it is not the answer the model intended to finish —
   * hosts persist it as unfinished rather than complete.
   */
  preemptIncomplete = false;
  /**
   * True when `routeMessage` sent a turn to `END` because the last AI
   * message carries no tool call, AND the provider reports it stopped for
   * hitting the output token ceiling (`getTruncationStopReason`). Plain-text
   * and reasoning turns cut off this way carry no tool call for
   * `assertNotTruncatedToolCall` to catch, so `toolsCondition` reads them as
   * an ordinary finished turn otherwise — hosts read this flag to persist
   * the turn as unfinished instead of a silently truncated "complete" answer.
   *
   * Deliberately separate from `preemptIncomplete`/`preemptHaltReason`: this
   * has no interaction with the preempt/seal machinery (in particular the
   * `preemptHaltReason` check at each model node's entry), so setting it
   * cannot suppress an unrelated agent's turn in a multi-agent graph.
   */
  outputTruncatedIncomplete = false;
  /**
   * The agent a summarize-only run summarizes with. Set from the first agent
   * input that opted in; while set, the model step after the summary routes
   * to END, and a multi-agent workflow compiles to this agent alone.
   */
  summarizeOnlyAgentId?: string;
  /**
   * `stopReason` from a `PreemptBoundary` hook that halted the turn.
   *
   * Clearing the registry halt is what keeps the sealed turn alive, but the
   * registry held the only copy of the reason — so it is captured here first.
   * Without it `getHaltReason()` returns undefined and a host records a
   * hook-halted turn as an ordinary completion.
   */
  preemptHaltReason: string | undefined;
  /**
   * Agent IDs whose next superstep must return to the agent node. Keyed by
   * agent because `MultiAgentGraph` routes every parallel agent through this
   * same instance, and a single field would let one agent's boundary resume
   * another's turn.
   */
  pendingPreemptReturn = new Set<string>();
  /**
   * Agent IDs whose model turn a preempt DISCARDED rather than sealed. A
   * discard returns no message, so the node cannot read
   * `response_metadata.preempted` to learn a boundary is owed — this set is
   * the only carrier.
   *
   * Keyed by agent for the same reason {@link pendingPreemptReturn} is: one
   * graph instance serves every parallel agent, and a single flag would let
   * whichever lane finished first consume another lane's boundary.
   *
   * Not derivable from `preemptSealInFlight`: an ordinary seal holds that slot
   * too, and a claim that never reached its boundary would be indistinguishable
   * from a discard.
   */
  private preemptRestartPending = new Set<string>();

  constructor(
    {
      runId,
      signal,
      agents,
      langfuse,
      tokenCounter,
      indexTokenCountMap,
      calibrationRatio,
      fadingTier,
      fadingTiers,
      subagentUsageSink,
      subagentTasks,
      subagentScope,
      subagentExecutionContext,
      subagentContext,
      preemption,
      streamLimits,
      toolExecution,
    }: t.StandardGraphInput,
    dependencies?: GraphFactoryDependencies
  ) {
    super();
    this.supportsMultiAgentChildren = dependencies != null;
    this.graphFactory =
      dependencies?.graphFactory ??
      ((request): StandardGraph => {
        if (request.kind !== 'standard') {
          throw new Error(
            'A polymorphic graph factory is required for multi-agent graph construction.'
          );
        }
        return new StandardGraph(request.input);
      });
    this.runId = runId;
    this.langfuseScopeRunId = `${runId ?? 'graph'}:${nanoid()}`;
    this.signal = signal;
    this.langfuse = langfuse;
    this.subagentUsageSink = subagentUsageSink;
    this.subagentTasks = subagentTasks;
    this.subagentScope = subagentScope === true;
    this.subagentExecutionContext = subagentExecutionContext;
    this.subagentContext = subagentContext;
    this.preemption = preemption;
    this.streamLimits = resolveStreamLimits(streamLimits);
    this.toolExecution = toolExecution;

    if (agents.length === 0) {
      throw new Error('At least one agent configuration is required');
    }

    for (const agentConfig of agents) {
      const agentContext = AgentContext.fromConfig(
        agentConfig,
        tokenCounter,
        indexTokenCountMap,
        toolExecution
      );
      if (agentConfig.summarizeOnly === true) {
        this.summarizeOnlyAgentId ??= agentContext.agentId;
      }
      if (calibrationRatio != null && calibrationRatio > 0) {
        agentContext.calibrationRatio = calibrationRatio;
      }
      let restoredFadingTier: t.FadingTier | null | undefined;
      if (
        fadingTiers != null &&
        Object.prototype.hasOwnProperty.call(fadingTiers, agentConfig.agentId)
      ) {
        restoredFadingTier = fadingTiers[agentConfig.agentId];
      } else if (agentConfig.agentId === agents[0].agentId) {
        restoredFadingTier = fadingTier;
      }
      /** A supplied tier is assumed to describe the supplied history. Hosts
       * must clear it when they create a new summary; this also permits tiers
       * learned on turns after a durable summary to survive subsequent Runs. */
      if (isFadingTier(restoredFadingTier)) {
        agentContext.fadingTier = { ...restoredFadingTier };
      }

      this.agentContexts.set(agentConfig.agentId, agentContext);
    }

    /** The run's agent: root trace identity and Langfuse routing follow it,
     *  so a summarize-only run is attributed to the agent that summarizes. */
    this.defaultAgentId = this.summarizeOnlyAgentId ?? agents[0].agentId;
  }

  /**
   * A compaction applies its remove-all inside the agent subgraph, so the
   * subgraph's result carries only the retained tail, and an outer reducer
   * would merge that tail back into the history it already holds. Re-issuing
   * the remove-all across the boundary keeps a checkpointed outer state as
   * compacted as the subgraph's. A run that produced no summary changed
   * nothing and passes through.
   */
  protected propagateManualCompaction<
    S extends { messages: BaseMessage[]; manualSummary?: string },
  >(result: S): S {
    if (
      this.summarizeOnlyAgentId == null ||
      result.manualSummary == null ||
      result.manualSummary.length === 0
    ) {
      return result;
    }
    return {
      ...result,
      messages: [createRemoveAllMessage(), ...result.messages],
    };
  }

  /** Rotates the Langfuse identities that must never cross fresh executions. */
  startFreshLangfuseExecution(): void {
    this.langfuseTraceAnchor = {};
    this.langfuseScopeRunId = `${this.runId ?? 'graph'}:${nanoid()}`;
  }

  /* Init */

  resetValues(keepContent?: boolean, checkpointScope?: string): void {
    this.preparedSubagents.clear();
    this.resetSubagentCheckpointThreadIds();
    this.messages = [];
    this.hasRestoredCheckpointMessages = false;
    this.cachedRunMessages = undefined;
    this.cachedDiscoveredTools = undefined;
    this.config = resetIfNotEmpty(this.config, undefined);
    this.stopContinuationCount = 0;
    this.stopContinuationExecutionId = '';
    this.streamSegment = 0;
    if (keepContent !== true) {
      this.contentData = resetIfNotEmpty(this.contentData, []);
      this.nextContentIndex = 0;
      this.runStepStateRevision = 0;
      this.contentIndexMap = resetIfNotEmpty(this.contentIndexMap, new Map());
    }
    this.stepKeyIds = resetIfNotEmpty(this.stepKeyIds, new Map());
    /**
     * Clear in-place instead of replacing with a new Map to preserve the
     * shared reference held by ToolNode (passed at construction time).
     * Using resetIfNotEmpty would create a new Map, leaving ToolNode with
     * a stale reference on 2nd+ processStream calls.
     */
    this.toolCallStepIds.clear();
    this.pendingToolCallsByStep.clear();
    this.latestCompletionByStep.clear();
    this.openMessageStepByAgent.clear();
    this.runProducedAiMessageIds.clear();
    this.eagerEventToolExecutions.clear();
    this.clearEagerEventToolUsageCounts();
    this.eagerEventToolCallChunks.clear();
    this.eagerEventToolSuppressions.clear();
    /** Grace sweep instead of a clear: producer loops of straggling
     * attempts use these maps directly and sit outside the consumer-only
     * epoch gate — clearing would hand a cancellation-ignoring provider a
     * fresh allowance at every run start. Entries from the epoch that is
     * ending survive exactly one reset so those stragglers stay on their
     * original budgets; older entries are removed. */
    sweepStaleStreamLimitEntries(
      this.streamedToolCallArgTallies,
      this.breakerEpoch,
      this.activeStreamLimitGenerations
    );
    sweepStaleStreamLimitEntries(
      this.streamDeltaEventCounts,
      this.breakerEpoch,
      this.activeStreamLimitGenerations
    );
    this.streamLimitChargeCredits = undefined;
    /** Run-start is the only safe replacement point for the breaker:
     * end-of-run cleanup must leave it in place so straggling parallel
     * children from the failed run cannot start on a fresh signal.
     * Replaced UNCONDITIONALLY here — a run that failed on an ordinary
     * error leaves the controller un-aborted, and stragglers still settling
     * hold their entry-time capture of it; a late stream-limit trip on that
     * old controller must not cancel the run starting now. */
    this.breakerAbort = new AbortController();
    this.breakerEpoch += 1;
    this.runScope = Object.freeze({
      epoch: this.breakerEpoch,
      controller: this.breakerAbort,
    });
    this.handlerDispatchedStepIds = resetIfNotEmpty(
      this.handlerDispatchedStepIds,
      new Set()
    );
    this.handlerDispatchedEventCounts = resetIfNotEmpty(
      this.handlerDispatchedEventCounts,
      new Map()
    );
    this.messageIdsByStepKey = resetIfNotEmpty(
      this.messageIdsByStepKey,
      new Map()
    );
    this.messageStepHasToolCalls = resetIfNotEmpty(
      this.messageStepHasToolCalls,
      new Map()
    );
    this.messageStepHasTextDeltas = resetIfNotEmpty(
      this.messageStepHasTextDeltas,
      new Set()
    );
    this.reasoningStepHasDeltas = resetIfNotEmpty(
      this.reasoningStepHasDeltas,
      new Set()
    );
    this.prelimMessageIdsByStepKey = resetIfNotEmpty(
      this.prelimMessageIdsByStepKey,
      new Map()
    );
    this.invokedToolIds = resetIfNotEmpty(this.invokedToolIds, undefined);
    this.resetPreemptTurnState();
    this.resetPreemptTotals();
    const hasScopedCheckpoint =
      this.hasCompiledCheckpointer &&
      checkpointScope != null &&
      checkpointScope !== '';
    const preserveOriginalToolContent =
      hasScopedCheckpoint &&
      this.originalToolContentCheckpointScope === checkpointScope;
    for (const context of this.agentContexts.values()) {
      context.reset({ preserveOriginalToolContent });
    }
    this.originalToolContentCheckpointScope = hasScopedCheckpoint
      ? checkpointScope
      : undefined;
  }

  /** Seeds the sidecar message view that checkpoint restoration bypasses. */
  override restoreCheckpointMessages(
    messages: BaseMessage[],
    pendingMessages?: BaseMessage[]
  ): void {
    if (this.messages.length > 0) {
      return;
    }
    this.messages =
      pendingMessages == null
        ? [...messages]
        : messagesStateReducer(messages, pendingMessages);
    this.startIndex = this.messages.length;
    this.hasRestoredCheckpointMessages = true;
    this.cachedRunMessages = undefined;
  }

  override clearHeavyState(): void {
    this.cachedRunMessages = this.messages.slice(this.startIndex);
    this.cachedDiscoveredTools = new Map(
      Array.from(this.agentContexts, ([agentId, context]) => [
        agentId,
        context.getDiscoveredTools(),
      ])
    );
    super.clearHeavyState();
    this.messages = [];
    this.overrideModel = undefined;
    this.subagentModelOverride = undefined;
    /** Stream-limit accounting (argument tallies, event counts, charge
     * credits) deliberately SURVIVES cleanup: this runs in `processStream`'s
     * finally, which an ordinary parallel-branch failure reaches while
     * sibling attempts are still unwinding on the retained breaker — a
     * cancellation-ignoring provider's late chunks would otherwise recreate
     * their budgets from zero and stream another full allowance. The maps
     * are bounded by in-flight call sizes and `resetValues` clears them at
     * the next run start, where the epoch bump already drops stamped
     * straggler events before accounting. */
    /** Deliberately NOT recreating a tripped breakerAbort here: this runs in
     * `processStream`'s cleanup, which a rejected parallel batch reaches
     * while sibling subagents can still be pre-invoke — a fresh controller
     * would hand them an un-aborted signal and let provider requests start
     * after the run already failed. `resetValues` recreates it when the next
     * run begins. */
    /**
     * Turn state only. The reported totals must outlive cleanup — this runs
     * in `processStream`'s `finally`, and the host reads `getPreemptStats()`
     * after that returns.
     */
    this.resetPreemptTurnState();
    const preserveOriginalToolContent =
      this.hasCompiledCheckpointer &&
      this.originalToolContentCheckpointScope != null;
    for (const context of this.agentContexts.values()) {
      context.reset({ preserveOriginalToolContent });
    }
  }

  /**
   * Per-turn seal budget and routing markers. Cleared by both reset paths so
   * a new turn starts with a full budget and no stale resume marker.
   *
   * The REPORTED counters are deliberately not touched here — see
   * {@link resetPreemptTotals}.
   */
  private resetPreemptTurnState(): void {
    this.preemptSealBudgetUsed = 0;
    this.preemptSealInFlight = false;
    this.preemptRestartPending.clear();
    this.pendingPreemptReturn.clear();
  }

  /**
   * Lifetime seal totals, cleared only when a genuinely new run starts.
   * `clearHeavyState()` must NOT call this: it runs in `processStream`'s
   * `finally`, so zeroing here would make {@link getPreemptStats} and
   * `preemptIncomplete` unreadable for every caller of the method that just
   * produced them.
   */
  private resetPreemptTotals(): void {
    this.preemptSealCount = 0;
    this.preemptRestartCount = 0;
    this.preemptEmptyBoundaries = 0;
    this.preemptIncomplete = false;
    this.preemptHaltReason = undefined;
    this.outputTruncatedIncomplete = false;
  }

  /**
   * True when the host has requested a cooperative seal AND this graph may
   * honor it. Read once per streamed chunk, so it stays property reads plus
   * one host callback — no I/O, no allocation.
   *
   * Non-mutating: a true result only means a seal is worth evaluating. The
   * budget is taken by {@link claimPreemptSeal} once the accumulated chunk is
   * known to be safe, so a chunk that cannot seal never spends budget.
   *
   * Ordinary subagent scopes never receive `preemption`. A detached child may
   * receive a dedicated parent-control preemption source, in which case the
   * same provider-safe seal path is intentionally reused inside that child.
   */
  /** Internal seal preconditions only — no host callback, no side effects. */
  private canClaimPreemptSeal(): boolean {
    /**
     * Resolved and required here with the same rule `dispatchPreemptBoundary`
     * uses. Without it a direct `StandardGraph` consumer that supplies no
     * `runId` could claim a seal on the strength of a global matcher, then hit
     * the boundary's own null-runId guard and get nothing back — truncating
     * the answer for a drain that provably could not run.
     */
    const runId =
      (this.config?.configurable?.run_id as string | undefined) ?? this.runId;
    return (
      this.preemption != null &&
      !this.preemptSealInFlight &&
      this.preemptSealBudgetUsed < resolveMaxSeals(this.preemption.maxSeals) &&
      runId != null &&
      /**
       * A seal only buys room for an injection. With no `PreemptBoundary`
       * matcher live — never registered, or a `once` matcher already
       * consumed — the boundary provably returns nothing and the answer is
       * cut short for no gain, so refuse the seal instead. Failing closed
       * lands on the documented no-preemption behavior: the model finishes
       * and the message waits for the next tool boundary.
       *
       * Same session resolution as `dispatchPreemptBoundary`, or a
       * session-scoped matcher would be visible at one site and not the other.
       */
      this.hookRegistry?.hasDispatchableHookFor('PreemptBoundary', runId) ===
        true
    );
  }

  shouldPreemptStream(): boolean {
    return (
      this.canClaimPreemptSeal() && this.preemption?.shouldPreempt() === true
    );
  }

  /**
   * Takes the seal slot, or returns false if another stream already holds it.
   *
   * Assumes the caller already polled `shouldPreemptStream()` for THIS chunk,
   * and deliberately does not poll the host again — `StreamPreemption`
   * documents `shouldPreempt` as once per chunk, and a host that consumes a
   * pending flag on read would lose the request to a second call.
   *
   * The guard and both mutations remain one synchronous body, which is what
   * makes this safe under a parallel `MultiAgentGraph`: several agents share
   * one graph and can each see the poll as true, but no `await` can split the
   * claim, so only one takes the slot. The loser keeps streaming normally
   * rather than sealing for a message it would never receive.
   */
  claimPreemptSeal(): boolean {
    return this.claimPreemptSlot('seal');
  }

  /**
   * The restart twin. It takes the SAME slot and spends the SAME budget —
   * both moves end one model turn and cost one extra superstep, and the
   * mutual exclusion argument above applies identically — but it is counted
   * apart: a restart preserved no partial assistant message, so reporting it
   * as a seal would tell every consumer of `getPreemptStats().seals` and the
   * boundary hook's `sealCount` that a turn was kept when it was discarded.
   */
  claimPreemptRestart(): boolean {
    return this.claimPreemptSlot('restart');
  }

  private claimPreemptSlot(kind: 'seal' | 'restart'): boolean {
    if (!this.canClaimPreemptSeal()) {
      return false;
    }
    this.preemptSealInFlight = true;
    this.preemptSealBudgetUsed += 1;
    if (kind === 'seal') {
      this.preemptSealCount += 1;
      return true;
    }
    this.preemptRestartCount += 1;
    return true;
  }

  /** Releases the seal slot once its boundary has resolved, win or lose. */
  releasePreemptSeal(): void {
    this.preemptSealInFlight = false;
  }

  /** See {@link preemptRestartPending}. Called from the model attempt. */
  notePreemptRestart(agentId: string): void {
    this.preemptRestartPending.add(agentId);
  }

  /**
   * Reads and clears the lane's mark in one step, so a boundary is dispatched
   * exactly once per discard even though the model node runs again immediately
   * after.
   */
  private consumePreemptRestart(agentId: string): boolean {
    return this.preemptRestartPending.delete(agentId);
  }

  getPreemptStats(): t.PreemptStats {
    return {
      seals: this.preemptSealCount,
      restarts: this.preemptRestartCount,
      emptyBoundaries: this.preemptEmptyBoundaries,
    };
  }

  /* Run Step Processing */

  createRunStepResumeState(): t.RunStepResumeState {
    const steps: t.RunStepResumeEntry[] = [];
    for (const runStep of this.contentData) {
      if (runStep.status != null && runStep.status !== 'in_progress') {
        continue;
      }
      const agentKey = runStep.agentId ?? '';
      steps.push({
        step: structuredClone(runStep),
        pendingToolCallIds: [
          ...(this.pendingToolCallsByStep.get(runStep.id) ?? []),
        ],
        ...(this.latestCompletionByStep.has(runStep.id)
          ? {
            latestCompletionAt: this.latestCompletionByStep.get(runStep.id),
          }
          : {}),
        openMessageStep:
          this.openMessageStepByAgent.get(agentKey) === runStep.id,
      });
    }
    return {
      version: 1,
      revision: this.runStepStateRevision,
      nextIndex: this.nextContentIndex,
      stopContinuationCount: this.stopContinuationCount,
      ...(this.stopContinuationExecutionId === ''
        ? {}
        : {
          stopContinuationExecutionId: this.stopContinuationExecutionId,
        }),
      streamSegment: this.streamSegment,
      toolCallSteps: [...this.toolCallStepIds].map(([toolCallId, stepId]) => ({
        toolCallId,
        stepId,
      })),
      steps,
    };
  }

  restoreRunStepResumeState(state?: t.RunStepResumeState): void {
    if (
      !isRunStepResumeState(state) ||
      this.contentData.length > 0 ||
      this.runStepStateRevision > 0
    ) {
      return;
    }

    this.nextContentIndex = state.nextIndex;
    this.runStepStateRevision = state.revision;
    this.stopContinuationCount = state.stopContinuationCount ?? 0;
    this.stopContinuationExecutionId = state.stopContinuationExecutionId ?? '';
    this.streamSegment = state.streamSegment ?? 0;
    for (const { toolCallId, stepId } of state.toolCallSteps) {
      this.toolCallStepIds.set(toolCallId, stepId);
    }
    for (const entry of state.steps) {
      const runStep = structuredClone(entry.step);
      runStep.status = 'in_progress';
      this.nextContentIndex = Math.max(
        this.nextContentIndex,
        runStep.index + 1
      );
      const position = this.contentData.length;
      this.contentData.push(runStep);
      this.contentIndexMap.set(runStep.id, position);
      if (entry.pendingToolCallIds.length > 0) {
        const pending = new Set(entry.pendingToolCallIds);
        this.pendingToolCallsByStep.set(runStep.id, pending);
      }
      if (entry.latestCompletionAt != null) {
        this.latestCompletionByStep.set(runStep.id, entry.latestCompletionAt);
      }
      if (entry.openMessageStep) {
        this.openMessageStepByAgent.set(runStep.agentId ?? '', runStep.id);
      }
    }
  }

  getRunStep(stepId: string): t.RunStep | undefined {
    const index = this.contentIndexMap.get(stepId);
    if (index !== undefined) {
      return this.contentData[index];
    }
    return undefined;
  }

  getStopContinuationCount(): number {
    return this.stopContinuationCount;
  }

  setStopContinuationCount(count: number): void {
    this.stopContinuationCount = count;
  }

  getStopContinuationExecutionId(): string {
    return this.stopContinuationExecutionId;
  }

  startStopContinuationExecution(executionId: string): void {
    this.stopContinuationExecutionId = executionId;
  }

  getStreamSegment(): number {
    return this.streamSegment;
  }

  advanceStreamSegment(): void {
    this.streamSegment += 1;
  }

  /**
   * Derives the same lane key `dispatchRunStep` stamps as `runStep.agentId`.
   * The multi-agent check gates the lookup because `getAgentContext` signals
   * a miss by throwing: single-agent graphs key every step under `''`, so
   * resolving the context could only ever produce a thrown-and-discarded
   * Error on a per-model-call path.
   */
  protected getStepAgentKey(metadata?: Record<string, unknown>): string {
    if (!metadata || !this.isMultiAgentGraph()) {
      return '';
    }
    try {
      const agentContext = this.getAgentContext(metadata);
      if (agentContext.agentId) {
        return agentContext.agentId;
      }
    } catch (_e) {
      /** No agent context — fall back to the default lane */
    }
    return '';
  }

  /**
   * O(1) reverse lookup: both dispatch funnels key the open-message map by
   * `runStep.agentId ?? ''`, so the entry is addressable without scanning.
   */
  private untrackRunStep(runStep: t.RunStep): void {
    this.pendingToolCallsByStep.delete(runStep.id);
    this.latestCompletionByStep.delete(runStep.id);
    const agentKey = runStep.agentId ?? '';
    if (this.openMessageStepByAgent.get(agentKey) === runStep.id) {
      this.openMessageStepByAgent.delete(agentKey);
    }
  }

  /**
   * Shared step accounting for both dispatch funnels: a successor step in
   * the same agent lane marks the previous message step as finished — its
   * CLOSED event must precede the successor's ON_RUN_STEP so hosts observe
   * a consistent open-step timeline — then the step is registered in the
   * content maps and tracked as the lane's open message step when
   * applicable.
   */
  protected async trackDispatchedRunStep(
    runStep: t.RunStep,
    metadata?: Record<string, unknown>,
    /**
     * Summarization steps are typed MESSAGE_CREATION but own an explicit
     * completion, and their model call emits `CHAT_MODEL_END` well before the
     * summary is assembled. Tracking one as the lane's open step would let
     * model-end publish an authoritative `completed` closure early — the
     * measured duration would exclude the remaining work and a later
     * post-model failure could no longer change the status. They close
     * through `recordStepCompletion` instead.
     */
    trackAsOpenMessageStep: boolean = true
  ): Promise<void> {
    const agentKey = runStep.agentId ?? '';
    const openMessageStepId = this.openMessageStepByAgent.get(agentKey);
    /**
     * Reserve the content index and register the step SYNCHRONOUSLY, before
     * awaiting the predecessor's closure. Parallel agent lanes dispatch
     * successors concurrently: if both yielded here first, they would push
     * with the same `contentData.length` and `contentIndexMap` would resolve
     * one step id to the other lane's entry, so later deltas and completions
     * would mutate the wrong agent's step. Assigning the index at push time
     * also supersedes whatever the caller computed before this await point.
     */
    runStep.index = Math.max(this.nextContentIndex, this.contentData.length);
    this.nextContentIndex = runStep.index + 1;
    const position = this.contentData.length;
    this.contentData.push(runStep);
    this.contentIndexMap.set(runStep.id, position);
    if (trackAsOpenMessageStep && runStep.type === StepTypes.MESSAGE_CREATION) {
      this.openMessageStepByAgent.set(agentKey, runStep.id);
    } else {
      this.openMessageStepByAgent.delete(agentKey);
    }
    /**
     * Awaited after registration but before the caller dispatches this step's
     * ON_RUN_STEP, so the predecessor's CLOSED event still precedes the
     * successor's start event.
     */
    if (openMessageStepId != null && openMessageStepId !== runStep.id) {
      /**
       * Isolated: this step is already registered in `contentData`, so a
       * predecessor delivery failure rejecting here would abort the caller
       * before it publishes this step's ON_RUN_STEP — leaving the sweep to
       * emit a terminal event for a step that never announced a start.
       */
      try {
        await this.closeRunStep(openMessageStepId, 'completed', { metadata });
      } catch (_e) {
        /** Predecessor delivery must not abort the successor's lifecycle */
      }
    }
    /**
     * Stamped last, after the predecessor's closure has been delivered and
     * immediately before the caller publishes this step's ON_RUN_STEP.
     * `created_at` documents when the step was dispatched, so it must not
     * absorb the latency of an arbitrarily slow predecessor handler.
     */
    runStep.created_at = Date.now();
    this.runStepStateRevision += 1;
  }

  /**
   * Closes a run step: stamps its terminal status + timestamp on the stored
   * `RunStep` and emits `ON_RUN_STEP_CLOSED`. First close wins — later calls
   * are no-ops with no exceptions; every terminal status is immutable once
   * stamped. (A `restamp` option once let a completed TOOL_CALLS step
   * refresh `completed_at` for the eager-execution race, but that race is
   * unreachable — each step registers its calls before any completion can
   * reference it — and the mechanism was removed.)
   */
  /**
   * Closes the lane's open message step as `cancelled` when a preempt discards
   * the turn that step belongs to.
   *
   * Without it the step is closed `completed` by the discard's own model-end,
   * so `getRunSteps()` and every run-step subscriber keep reasoning the
   * replacement prompt does not contain — the graph state forgets the attempt
   * while the step view still shows it as a finished one.
   *
   * Only for turns that were CUT SHORT. A turn whose stream reached its own
   * end really did complete, and its step is already closed by then; the
   * terminal-status guard in {@link closeRunStep} leaves that alone.
   */
  async cancelOpenMessageStep(
    metadata?: Record<string, unknown>
  ): Promise<void> {
    const stepId = this.openMessageStepByAgent.get(
      this.getStepAgentKey(metadata)
    );
    if (stepId == null) {
      return;
    }
    try {
      await this.closeRunStep(stepId, 'cancelled', { metadata });
    } catch (_e) {
      /** A step-view detail must never take down the run it describes. */
    }
  }

  async closeRunStep(
    stepId: string,
    status: Exclude<t.RunStepStatus, 'in_progress'>,
    options?: t.RunStepCloseOptions
  ): Promise<boolean> {
    if (!stepId) {
      return false;
    }
    const runStep = this.getRunStep(stepId);
    if (!runStep) {
      return false;
    }
    if (runStep.status != null && runStep.status !== 'in_progress') {
      this.untrackRunStep(runStep);
      return false;
    }

    const closedAt = options?.at ?? Date.now();
    runStep.status = status;
    if (status === 'completed') {
      runStep.completed_at = closedAt;
    } else if (status === 'cancelled') {
      runStep.cancelled_at = closedAt;
    } else {
      runStep.failed_at = closedAt;
    }

    const closedEvent: t.RunStepClosedEvent = {
      id: stepId,
      index: runStep.index,
      type: runStep.type,
      status,
      closed_at: closedAt,
    };
    if (runStep.created_at != null) {
      closedEvent.created_at = runStep.created_at;
    }
    if (runStep.runId != null) {
      closedEvent.runId = runStep.runId;
    }
    if (runStep.agentId != null) {
      closedEvent.agentId = runStep.agentId;
    }
    if (runStep.groupId != null) {
      closedEvent.groupId = runStep.groupId;
    }
    if (runStep.stepIndex != null) {
      closedEvent.stepIndex = runStep.stepIndex;
    }
    this.untrackRunStep(runStep);
    this.runStepStateRevision += 1;

    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_RUN_STEP_CLOSED
    );
    if (handler) {
      /**
       * Isolated, unlike the other dual-dispatch sites, because this one
       * reports state that is already committed: the step was stamped
       * terminal and untracked above, and nothing a failed delivery can do
       * will undo that. Propagating instead costs two things.
       *
       * First, it fails an entire run over an observational event —
       * `closeOpenMessageStep` awaits this inside the stream loop on every
       * CHAT_MODEL_END, where a rejection sets `streamThrew` and fires the
       * StopFailure hooks for a response that was fully delivered.
       *
       * Second, and worse, it skips the secondary custom-event dispatch
       * below. That channel exists precisely as the fallback for when the
       * primary path does not deliver, so letting the primary's failure
       * suppress it removes the redundancy exactly when it is needed.
       *
       * `closeUnfinishedRunSteps` and `dispatchRunStep` already wrap their
       * own calls for the same reason; this closes the gap inside, which
       * those wrappers cannot reach.
       */
      try {
        await handler.handle(
          GraphEvents.ON_RUN_STEP_CLOSED,
          closedEvent,
          options?.metadata,
          this
        );
      } catch (_e) {
        /** Host delivery failure must not fail the run or block the echo */
      }
      this.handlerDispatchedStepIds.add(stepId);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_RUN_STEP_CLOSED, stepId)
      : undefined;
    try {
      if (this.config) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_RUN_STEP_CLOSED,
          closedEvent,
          this.config
        );
      }
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
    return true;
  }

  /**
   * Observes one `ON_RUN_STEP_COMPLETED` for a step and closes the step when
   * no registered tool calls remain pending. Steps without pending tracking
   * (summaries, cross-process resume) close on their first completion; the
   * terminal-status guard in `closeRunStep` absorbs duplicate echoes.
   */
  async recordStepCompletion(
    stepId: string,
    options?: t.RecordStepCompletionOptions
  ): Promise<void> {
    if (!stepId) {
      return;
    }
    const { toolCallId, metadata, at } = options ?? {};
    let lifecycleChanged = false;
    if (at != null) {
      const latest = this.latestCompletionByStep.get(stepId);
      if (latest == null || at > latest) {
        this.latestCompletionByStep.set(stepId, at);
        lifecycleChanged = true;
      }
    }
    const closeAt = this.latestCompletionByStep.get(stepId) ?? at;
    const pending = this.pendingToolCallsByStep.get(stepId);
    if (pending == null) {
      await this.closeRunStep(stepId, 'completed', { metadata, at: closeAt });
      return;
    }
    if (toolCallId != null && toolCallId !== '') {
      lifecycleChanged = pending.delete(toolCallId) || lifecycleChanged;
    }
    if (pending.size > 0) {
      if (lifecycleChanged) {
        this.runStepStateRevision += 1;
      }
      return;
    }
    await this.closeRunStep(stepId, 'completed', { metadata, at: closeAt });
  }

  /**
   * Closes the tracked open MESSAGE_CREATION step for the event's agent lane.
   * Fires on every model end, so the empty-map check short-circuits ahead of
   * resolving the lane key — a turn whose message step already closed through
   * successor-close does no work here.
   */
  async closeOpenMessageStep(
    metadata?: Record<string, unknown>,
    /** Model-end time captured before host handlers ran, so a slow usage sink
     *  cannot inflate the step's measured duration. */
    at?: number
  ): Promise<void> {
    if (this.openMessageStepByAgent.size === 0) {
      return;
    }
    const agentKey = this.getStepAgentKey(metadata);
    const openStepId = this.openMessageStepByAgent.get(agentKey);
    if (openStepId == null) {
      return;
    }
    await this.closeRunStep(openStepId, 'completed', { metadata, at });
  }

  /**
   * End-of-run sweep: closes every step that never reached a terminal
   * status. Dual-dispatches like any other close — the custom-event channel
   * is usually already torn down here and `safeDispatchCustomEvent` reports
   * that quietly, but callback-only subscribers still receive the terminal
   * signal whenever it is alive.
   */
  async closeUnfinishedRunSteps(
    status: Exclude<t.RunStepStatus, 'in_progress'>,
    at?: number
  ): Promise<void> {
    const closedAt = at ?? Date.now();
    for (const runStep of this.contentData) {
      if (runStep.status != null && runStep.status !== 'in_progress') {
        continue;
      }
      /**
       * Isolated per step: one host handler throwing must not strand the
       * remaining steps `in_progress` with no terminal event. The step is
       * stamped before dispatch either way, so a thrown handler still leaves
       * consistent state behind.
       */
      try {
        await this.closeRunStep(runStep.id, status, { at: closedAt });
      } catch (_e) {
        /** Delivery failure for one step must not halt the sweep */
      }
    }
    this.pendingToolCallsByStep.clear();
    this.latestCompletionByStep.clear();
    this.openMessageStepByAgent.clear();
  }

  getAgentContext(metadata: Record<string, unknown> | undefined): AgentContext {
    if (!metadata) {
      throw new Error('No metadata provided to retrieve agent context');
    }

    const currentNode = metadata.langgraph_node as string;
    if (!currentNode) {
      throw new Error(
        'No langgraph_node in metadata to retrieve agent context'
      );
    }

    let agentId: string | undefined;
    if (currentNode.startsWith(AGENT)) {
      agentId = currentNode.substring(AGENT.length);
    } else if (currentNode.startsWith(TOOLS)) {
      agentId = currentNode.substring(TOOLS.length);
    } else if (currentNode.startsWith(SUMMARIZE)) {
      agentId = currentNode.substring(SUMMARIZE.length);
    }

    const agentContext = this.agentContexts.get(agentId ?? '');
    if (!agentContext) {
      throw new Error(`No agent context found for agent ID ${agentId}`);
    }

    return agentContext;
  }

  getStepBaseKey(metadata: Record<string, unknown> | undefined): string {
    if (!metadata) return '';

    const keyList = this.getInvocationKeyList(metadata);
    if (this.checkKeyList(keyList)) {
      throw new Error('Missing metadata');
    }

    return joinKeys(keyList);
  }

  getStepKey(metadata: Record<string, unknown> | undefined): string {
    if (!metadata) return '';

    const keyList = this.getKeyList(metadata);
    if (this.checkKeyList(keyList)) {
      throw new Error('Missing metadata');
    }

    return joinKeys(keyList);
  }

  getStepIdByKey(stepKey: string, index?: number): string {
    const stepIds = this.stepKeyIds.get(stepKey);
    if (!stepIds) {
      throw new Error(`No step IDs found for stepKey ${stepKey}`);
    }

    if (index === undefined) {
      return stepIds[stepIds.length - 1];
    }

    return stepIds[index];
  }

  generateStepId(stepKey: string): [string, number] {
    const stepIds = this.stepKeyIds.get(stepKey);
    let newStepId: string | undefined;
    let stepIndex = 0;
    if (stepIds) {
      stepIndex = stepIds.length;
      newStepId = `step_${nanoid()}`;
      stepIds.push(newStepId);
      this.stepKeyIds.set(stepKey, stepIds);
    } else {
      newStepId = `step_${nanoid()}`;
      this.stepKeyIds.set(stepKey, [newStepId]);
    }

    return [newStepId, stepIndex];
  }

  getKeyList(
    metadata: Record<string, unknown> | undefined
  ): (string | number | undefined)[] {
    if (!metadata) return [];

    const keyList = this.getInvocationKeyList(metadata);
    const agentContext = this.getAgentContext(metadata);
    if (
      agentContext.currentTokenType === ContentTypes.THINK ||
      agentContext.currentTokenType === 'think_and_text'
    ) {
      keyList.push('reasoning');
    } else if (agentContext.tokenTypeSwitch === 'content') {
      keyList.push(`post-reasoning-${agentContext.reasoningTransitionCount}`);
    }

    return keyList;
  }

  private getInvocationKeyList(
    metadata: Record<string, unknown>
  ): (string | number | undefined)[] {
    const keyList = this.getBaseKeyList(metadata);
    if (this.invokedToolIds != null && this.invokedToolIds.size > 0) {
      keyList.push(this.invokedToolIds.size + '');
    }
    return keyList;
  }

  private getBaseKeyList(
    metadata: Record<string, unknown>
  ): (string | number | undefined)[] {
    const configurable = this.config?.configurable;
    const runId =
      (metadata.run_id as string | undefined) ??
      (configurable?.run_id as string | undefined) ??
      this.runId;
    const threadId =
      (metadata.thread_id as string | undefined) ??
      (configurable?.thread_id as string | undefined) ??
      runId;
    const checkpointNs =
      (metadata.checkpoint_ns as string | undefined) ??
      (metadata.langgraph_checkpoint_ns as string | undefined) ??
      '';
    const keyList = [
      runId,
      threadId,
      metadata.langgraph_node as string,
      metadata.langgraph_step as number,
      checkpointNs,
      this.streamSegment,
    ];

    return keyList;
  }

  checkKeyList(keyList: (string | number | undefined)[]): boolean {
    return keyList.some((key) => key === undefined);
  }

  /* Misc.*/

  getRunMessages(): BaseMessage[] | undefined {
    /** Runtime-honest widening: a disposed-but-cached graph (HITL
     * resume/reconnect through a WeakRef cache) can carry null here despite
     * the field type. */
    const messages = this.messages as BaseMessage[] | undefined;
    if (messages == null) {
      return this.cachedRunMessages;
    }
    if (messages.length === 0 && this.cachedRunMessages != null) {
      return this.cachedRunMessages;
    }
    return messages.slice(this.startIndex);
  }

  override getDiscoveredTools(agentId?: string): string[] {
    if (agentId != null) {
      const current =
        this.agentContexts.get(agentId)?.getDiscoveredTools() ?? [];
      if (current.length > 0 || this.cachedDiscoveredTools == null) {
        return current;
      }
      return [...(this.cachedDiscoveredTools.get(agentId) ?? [])];
    }

    const discoveredTools = new Set<string>();
    for (const context of this.agentContexts.values()) {
      for (const toolName of context.getDiscoveredTools()) {
        discoveredTools.add(toolName);
      }
    }
    if (discoveredTools.size === 0 && this.cachedDiscoveredTools != null) {
      for (const snapshot of this.cachedDiscoveredTools.values()) {
        for (const toolName of snapshot) {
          discoveredTools.add(toolName);
        }
      }
    }
    return Array.from(discoveredTools);
  }

  /**
   * True when THIS RUN produced `message` — the provenance the handoff cue
   * gate needs. Tracked as an id set rather than inferred from `startIndex`
   * arithmetic: summarization's remove-all compaction rewrites the live
   * array and leaves `startIndex` stale, so index-based run/host
   * discrimination silently breaks right after a mid-run summarize. Ids
   * survive compaction (retained messages keep theirs), host-supplied
   * prefill messages are never in the set, and membership is O(1) per
   * model call.
   */
  isRunProducedMessage(message: BaseMessage): boolean {
    const id = message.id;
    return (
      typeof id === 'string' &&
      id !== '' &&
      this.runProducedAiMessageIds.has(id)
    );
  }

  getContentParts(): t.MessageContentComplex[] | undefined {
    // `messages` can be null/undefined on a graph that has been disposed
    // (clearHeavyState) but is still reachable via a cache (e.g. RedisJobStore's
    // WeakRef) during a HITL resume/reconnect. Guard instead of dereferencing null.
    const messages = this.messages as BaseMessage[] | undefined;
    if (messages == null) {
      return undefined;
    }
    return convertMessagesToContent(messages.slice(this.startIndex));
  }

  getCalibrationRatio(): number {
    const context = this.agentContexts.get(this.defaultAgentId);
    return context?.calibrationRatio ?? 1;
  }

  /**
   * Latched context-fading tier for the default agent, or undefined while it
   * is still fresh. Hosts persist it beside the calibration ratio.
   */
  getFadingTier(): t.FadingTier | undefined {
    const context = this.agentContexts.get(this.defaultAgentId);
    const tier = context?.fadingTier;
    return isInformativeFadingTier(tier, context?.maxContextTokens)
      ? { ...tier }
      : undefined;
  }

  /** Latched context-fading tiers keyed by agent ID. */
  getFadingTiers(): t.FadingTiers {
    const tiers: Array<[string, t.FadingTier]> = [];
    for (const [agentId, context] of this.agentContexts) {
      if (
        isInformativeFadingTier(context.fadingTier, context.maxContextTokens)
      ) {
        tiers.push([agentId, { ...context.fadingTier }]);
      }
    }
    return Object.fromEntries(tiers);
  }

  /** Agent IDs whose canonical history was compacted during this run. */
  getFadingTierResetAgentIds(): string[] {
    return Array.from(this.agentContexts)
      .filter(([, context]) => context.fadingTierReset)
      .map(([agentId]) => agentId);
  }

  /** Whether the default agent compacted canonical history during this run. */
  didResetFadingTier(): boolean {
    return (
      this.agentContexts.get(this.defaultAgentId)?.fadingTierReset === true
    );
  }

  protected override createSubagentFadingResumeState(): Pick<
    SubagentGraphResumeState,
    'fadingTier' | 'fadingTiers'
    > {
    const fadingTier = this.getFadingTier();
    const fadingTiers = this.getFadingTiers();
    return {
      ...(fadingTier == null ? {} : { fadingTier }),
      ...(Object.keys(fadingTiers).length === 0 ? {} : { fadingTiers }),
    };
  }

  getResolvedInstructionOverhead(): number | undefined {
    const context = this.agentContexts.get(this.defaultAgentId);
    return context?.resolvedInstructionOverhead;
  }

  getToolCount(): number {
    const context = this.agentContexts.get(this.defaultAgentId);
    return (
      (context?.tools?.length ?? 0) +
      (context?.toolDefinitions?.length ?? 0) +
      /**
       * Graph-managed + host-supplied direct tools (handoff, subagent,
       * `AgentInputs.graphTools`) are bound to the model and token-accounted,
       * so a count that omits them under-reports the run's tool surface
       * (Codex #289 P3).
       */
      (context?.graphTools?.length ?? 0)
    );
  }

  /**
   * Get all run steps, optionally filtered by agent ID
   */
  getRunSteps(agentId?: string): t.RunStep[] {
    // `contentData` can be null/undefined on a disposed-but-cached graph during a
    // HITL resume/reconnect; without this guard `[...this.contentData]` throws
    // "this.contentData is not iterable".
    const contentData = this.contentData as t.RunStep[] | undefined;
    if (contentData == null) {
      return [];
    }
    if (agentId == null || agentId === '') {
      return [...contentData];
    }
    return contentData.filter((step) => step.agentId === agentId);
  }

  /**
   * Get run steps grouped by agent ID
   */
  getRunStepsByAgent(): Map<string, t.RunStep[]> {
    const stepsByAgent = new Map<string, t.RunStep[]>();

    for (const step of this.contentData) {
      if (step.agentId == null || step.agentId === '') continue;

      const steps = stepsByAgent.get(step.agentId) ?? [];
      steps.push(step);
      stepsByAgent.set(step.agentId, steps);
    }

    return stepsByAgent;
  }

  /**
   * Get agent IDs that participated in this run
   */
  getActiveAgentIds(): string[] {
    const agentIds = new Set<string>();
    for (const step of this.contentData) {
      if (step.agentId != null && step.agentId !== '') {
        agentIds.add(step.agentId);
      }
    }
    return Array.from(agentIds);
  }

  /**
   * Maps contentPart indices to agent IDs for post-run analysis
   * Returns a map where key is the contentPart index and value is the agentId
   */
  getContentPartAgentMap(): Map<number, string> {
    const contentPartAgentMap = new Map<number, string>();

    for (const step of this.contentData) {
      if (
        step.agentId != null &&
        step.agentId !== '' &&
        Number.isFinite(step.index)
      ) {
        contentPartAgentMap.set(step.index, step.agentId);
      }
    }

    return contentPartAgentMap;
  }

  /* Graph */

  canPrestartSubagents(agentContext?: AgentContext): boolean {
    if (
      this.eagerEventToolExecution?.enabled !== true ||
      (this.eagerEventToolExecution.maxPendingSubagents ?? 4) <= 0 ||
      this.compileOptions?.checkpointer != null ||
      this.humanInTheLoop?.enabled === true ||
      (this.interruptingToolNames?.length ?? 0) > 0 ||
      agentContext == null ||
      !this.subagentToolNodes.has(agentContext.agentId)
    ) {
      return false;
    }
    const graphTools = agentContext.graphTools as t.GenericTool[] | undefined;
    return (
      graphTools?.length === 1 &&
      graphTools[0].name === Constants.SUBAGENT &&
      SUBAGENT_REPLAY_CONTROLLER in graphTools[0]
    );
  }

  prestartSubagent(
    call: ToolCall,
    attempt: string,
    agentContext?: AgentContext
  ): void {
    const config = this.preparedSubagents.getConfig(attempt);
    if (
      !this.canPrestartSubagents(agentContext) ||
      config == null ||
      agentContext == null
    ) {
      return;
    }
    this.subagentToolNodes
      .get(agentContext.agentId)
      ?.prestartSubagent(
        call,
        attempt,
        config,
        this.eagerEventToolExecution?.maxPendingSubagents ?? 4
      );
  }

  initializeTools({
    currentTools,
    currentToolMap,
    agentContext,
  }: {
    currentTools?: t.GraphTools;
    currentToolMap?: t.ToolMap;
    agentContext?: AgentContext;
  }): CustomToolNode<t.BaseGraphState> | ToolNode<t.BaseGraphState> {
    const toolDefinitions = agentContext?.toolDefinitions;
    const eventDrivenMode =
      toolDefinitions != null && toolDefinitions.length > 0;
    const traceToolNode = shouldTraceToolNodeForLangfuse({
      runLangfuse: this.langfuse,
      agentLangfuse: agentContext?.langfuse,
    });
    const interruptingToolNames = new Set(this.interruptingToolNames ?? []);
    if (
      this.humanInTheLoop?.enabled === true &&
      (agentContext?.subagentConfigs?.length ?? 0) > 0
    ) {
      interruptingToolNames.add(Constants.SUBAGENT);
    }
    const effectiveInterruptingToolNames =
      interruptingToolNames.size > 0 ? interruptingToolNames : undefined;

    if (eventDrivenMode) {
      const schemaTools = createSchemaOnlyTools(toolDefinitions);
      const toolDefMap = new Map(toolDefinitions.map((def) => [def.name, def]));
      const graphTools = agentContext?.graphTools as
        | t.GenericTool[]
        | undefined;

      const directToolNames = new Set<string>();
      const allTools = [...schemaTools] as t.GenericTool[];
      const allToolMap: t.ToolMap = new Map(
        schemaTools.map((tool) => [tool.name, tool])
      );

      if (graphTools && graphTools.length > 0) {
        for (const tool of graphTools) {
          if ('name' in tool) {
            allTools.push(tool);
            allToolMap.set(tool.name, tool);
            directToolNames.add(tool.name);
          }
        }
      }

      const node = new CustomToolNode<t.BaseGraphState>({
        handoffRouting: this.handoffRouting,
        tools: allTools,
        toolMap: allToolMap,
        trace: traceToolNode,
        runLangfuse: this.langfuse,
        agentLangfuse: agentContext?.langfuse,
        eventDrivenMode: true,
        sessions: this.sessions,
        codeSessionKey: agentContext?.codeSessionKey,
        toolDefinitions: toolDefMap,
        // `agentId` is the subagent-scope marker — set ONLY for child-run
        // graphs (hooks fire for child scopes too, via the inherited
        // run_id); `executingAgentId` always identifies the owning agent.
        agentId: this.subagentScope ? agentContext?.agentId : undefined,
        executingAgentId: agentContext?.agentId,
        executionContext: this.subagentExecutionContext,
        executingAgentName: agentContext?.name,
        rootAgentId: this.defaultAgentId,
        rootAgentName: this.agentContexts.get(this.defaultAgentId)?.name,
        toolCallStepIds: this.toolCallStepIds,
        toolRegistry: agentContext?.toolRegistry,
        getDiscoveredToolNames: (): readonly string[] =>
          agentContext?.getDiscoveredTools() ?? [],
        hookRegistry: this.hookRegistry,
        humanInTheLoop: this.humanInTheLoop,
        eagerEventToolExecution: this.eagerEventToolExecution,
        preparedSubagents: this.preparedSubagents,
        codeSessionToolNames: this.codeSessionToolNames,
        eagerEventToolExecutions: this.eagerEventToolExecutions,
        eagerEventToolUsageCount: this.getEagerEventToolUsageCount(
          agentContext?.agentId
        ),
        eagerEventToolSuppressions: this.eagerEventToolSuppressions,
        toolExecution: this.toolExecution,
        directToolNames: directToolNames.size > 0 ? directToolNames : undefined,
        interruptingToolNames: effectiveInterruptingToolNames,
        maxContextTokens: agentContext?.maxContextTokens,
        maxToolResultChars: agentContext?.maxToolResultChars,
        toolOutputRegistry: this.getOrCreateToolOutputRegistry(),
        fileCheckpointer: this.getOrCreateFileCheckpointer(),
        getBreakerSignal: (): AbortSignal => this.breakerAbort.signal,
        getRunScope: (): RunBreakerScope => this.runScope,
        restoreRunStepResumeState: (state, config): void => {
          this.config = config;
          this.restoreRunStepResumeState(state);
        },
        createRunStepResumeState: (): t.RunStepResumeState =>
          this.createRunStepResumeState(),
        errorHandler: (data, metadata): Promise<boolean> =>
          StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      });
      this.registerCompiledToolNode(node);
      if (agentContext != null) {
        this.subagentToolNodes.set(agentContext.agentId, node);
      }
      return node;
    }

    const graphTools = agentContext?.graphTools as t.GenericTool[] | undefined;
    const baseTools = (currentTools as t.GenericTool[] | undefined) ?? [];
    const allTraditionalTools =
      graphTools && graphTools.length > 0
        ? [...baseTools, ...graphTools]
        : baseTools;
    /**
     * ToolNode treats a supplied `toolMap` as authoritative (it only derives
     * one from `tools` when the param is undefined), so when graphTools force
     * us to build a merged map here, an absent `currentToolMap` must be
     * seeded from the BASE tools first — otherwise ordinary tools stay bound
     * to the model but vanish from the execution map and every call to them
     * fails as an unknown tool (Codex #289 round 2).
     */
    const traditionalToolMap =
      graphTools && graphTools.length > 0
        ? new Map([
          ...(currentToolMap ??
              new Map(
                baseTools
                  .filter(
                    (t): t is t.GenericTool & { name: string } => 'name' in t
                  )
                  .map((t) => [t.name, t] as [string, t.GenericTool])
              )),
          ...graphTools
            .filter((t): t is t.GenericTool & { name: string } => 'name' in t)
            .map((t) => [t.name, t] as [string, t.GenericTool]),
        ])
        : currentToolMap;

    const node = new CustomToolNode<t.BaseGraphState>({
      handoffRouting: this.handoffRouting,
      tools: allTraditionalTools,
      toolMap: traditionalToolMap,
      trace: traceToolNode,
      runLangfuse: this.langfuse,
      agentLangfuse: agentContext?.langfuse,
      // `agentId` is the subagent-scope marker — set ONLY for child-run
      // graphs; `executingAgentId` always identifies the owning agent so
      // hooks can attribute the batch even at the top level.
      agentId: this.subagentScope ? agentContext?.agentId : undefined,
      executingAgentId: agentContext?.agentId,
      executionContext: this.subagentExecutionContext,
      executingAgentName: agentContext?.name,
      rootAgentId: this.defaultAgentId,
      rootAgentName: this.agentContexts.get(this.defaultAgentId)?.name,
      toolCallStepIds: this.toolCallStepIds,
      errorHandler: (data, metadata): Promise<boolean> =>
        StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      toolRegistry: agentContext?.toolRegistry,
      getDiscoveredToolNames: (): readonly string[] =>
        agentContext?.getDiscoveredTools() ?? [],
      sessions: this.sessions,
      codeSessionKey: agentContext?.codeSessionKey,
      toolExecution: this.toolExecution,
      codeSessionToolNames: this.codeSessionToolNames,
      interruptingToolNames: effectiveInterruptingToolNames,
      hookRegistry: this.hookRegistry,
      humanInTheLoop: this.humanInTheLoop,
      maxContextTokens: agentContext?.maxContextTokens,
      maxToolResultChars: agentContext?.maxToolResultChars,
      toolOutputRegistry: this.getOrCreateToolOutputRegistry(),
      fileCheckpointer: this.getOrCreateFileCheckpointer(),
      getBreakerSignal: (): AbortSignal => this.breakerAbort.signal,
      getRunScope: (): RunBreakerScope => this.runScope,
      restoreRunStepResumeState: (state, config): void => {
        this.config = config;
        this.restoreRunStepResumeState(state);
      },
      createRunStepResumeState: (): t.RunStepResumeState =>
        this.createRunStepResumeState(),
    });
    this.registerCompiledToolNode(node);
    return node;
  }

  overrideTestModel(
    responses: string[],
    sleep?: number,
    toolCalls?: ToolCall[]
  ): void {
    this.overrideModel = createFakeStreamingLLM({
      responses,
      sleep,
      toolCalls,
    });
  }

  /** Explicitly overrides the model used by isolated descendant subagent graphs. */
  setSubagentModelOverride(model: t.ChatModel): void {
    this.subagentModelOverride = model;
  }

  getUsageMetadata(
    finalMessage?: BaseMessage
  ): Partial<UsageMetadata> | undefined {
    if (
      finalMessage &&
      'usage_metadata' in finalMessage &&
      finalMessage.usage_metadata != null
    ) {
      return finalMessage.usage_metadata as Partial<UsageMetadata>;
    }
  }

  cleanupSignalListener(currentModel?: t.ChatModel): void {
    if (!this.signal) {
      return;
    }
    const model = this.overrideModel ?? currentModel;
    if (!model) {
      return;
    }
    const client = (model as ChatOpenAI | undefined)?.exposedClient;
    if (!client?.abortHandler) {
      return;
    }
    this.signal.removeEventListener('abort', client.abortHandler);
    client.abortHandler = undefined;
  }

  /**
   * Applies a context-overflow recovery plan and hands control to the
   * summarize node, which compacts and then routes straight back here for a
   * retry against the corrected budget.
   *
   * Returning the detour rather than rethrowing is the whole point: the
   * caller never sees the provider's rejection, only a slightly longer turn.
   */
  private beginOverflowRecovery({
    recovery,
    agentContext,
    agentId,
    config,
    originalToolContent,
    estimatedPromptTokens,
  }: {
    recovery: OverflowRecoveryPlan;
    agentContext: AgentContext;
    agentId: string;
    config?: RunnableConfig;
    /** Masking record from the prune pass that built the rejected prompt. */
    originalToolContent?: Map<number, string>;
    /** Size of the rejected prompt, recorded to detect a correction that changed nothing. */
    estimatedPromptTokens?: number;
  }): Partial<t.AgentSubgraphState> {
    const previousBudget = agentContext.maxContextTokens;
    /**
     * Deterministic compaction first. Re-pruning against the corrected budget
     * raises context pressure, which is what drives the pruner's tool-output
     * truncation and observation masking — no model call, no cost, and no
     * message content lost. A summarization call is held back until that has
     * been tried and the provider rejected the prompt again.
     */
    const allowSummarization = agentContext.shouldSummarizeOverflow();

    agentContext.preserveOriginalToolContent(originalToolContent);
    agentContext.applyContextBudgetCorrection(
      recovery.budgetTokens,
      estimatedPromptTokens
    );
    agentContext.applyObservedOverflowCalibration(
      recovery.info.provider,
      recovery.observedCalibrationRatio
    );

    emitAgentLog(
      config,
      'warn',
      'graph',
      'Provider rejected the prompt as too large — compacting and retrying',
      {
        kind: recovery.info.kind,
        previousBudget,
        recoveredBudget: recovery.budgetTokens,
        providerReportedLimit: recovery.info.limitTokens,
        providerReportedTokens: recovery.info.requestedTokens,
        providerReportedPromptTokens: recovery.info.promptTokens,
        observedCalibrationRatio: recovery.observedCalibrationRatio,
        detectedBy: recovery.info.source,
        attempt: agentContext.overflowRecoveryAttempts,
        compaction: allowSummarization ? 'summarize' : 'compress',
      },
      { runId: this.runId, agentId },
      { force: true }
    );

    return {
      summarizationRequest: {
        remainingContextTokens: 0,
        agentId: agentId || agentContext.agentId,
        reason: 'overflow',
        allowSummarization,
      },
    };
  }

  private getPreparedToolsForBinding(
    agentContext: AgentContext,
    provider: t.ProviderName = agentContext.provider,
    clientOptions: t.ClientOptions | undefined = agentContext.clientOptions
  ): t.GraphTools | undefined {
    const tools = resolveLocalToolsForBinding({
      tools: agentContext.getToolsForBinding(),
      toolExecution: this.toolExecution,
      toolRegistry: agentContext.toolRegistry,
      discoveredToolNames: new Set(agentContext.getDiscoveredTools()),
    });
    return prepareToolsForPromptCache({
      provider,
      clientOptions,
      tools,
      isDeferred: makeIsDeferred(agentContext.getEffectiveToolDefinitions()),
    });
  }

  /**
   * The two model steps of a summarize-only run, neither of which calls the
   * model. The first requests the summary outright — no trigger consulted,
   * because the user asked for it — and the second, entered after the
   * summarize node, dispatches the post-summary usage snapshot (the host's
   * gauge and summary baseline read it) and ends the run. Returns `null`
   * for every other run so the ordinary step proceeds.
   */
  private async summarizeOnlyStep({
    agentContext,
    agentId,
    messageCount,
    contextUsage,
    config,
  }: {
    agentContext: AgentContext;
    agentId: string;
    messageCount: number;
    contextUsage: t.ContextUsageEvent | null;
    config: RunnableConfig;
  }): Promise<Partial<t.AgentSubgraphState> | null> {
    if (agentContext.summarizeOnly !== true) {
      return null;
    }
    const meta = { runId: this.runId, agentId };
    if (agentContext.claimManualSummarization()) {
      if (agentContext.summarizationEnabled !== true) {
        throw new ManualSummarizationSkippedError(
          'disabled',
          'Compaction skipped: summarization is not enabled for this agent'
        );
      }
      emitAgentLog(
        config,
        'info',
        'graph',
        'Manual summarization requested',
        {
          totalMessages: messageCount,
          summaryVersion: agentContext.summaryVersion + 1,
        },
        meta
      );
      agentContext.markSummarizationTriggered(messageCount);
      return {
        summarizationRequest: {
          remainingContextTokens: contextUsage?.remainingContextTokens ?? 0,
          agentId: agentId || agentContext.agentId,
          reason: 'manual',
        },
        /** A checkpointed thread may still hold the previous compaction's
         *  summary; only the one this run produces may be reported. */
        manualSummary: '',
      };
    }
    if (contextUsage != null) {
      await safeDispatchCustomEvent(
        GraphEvents.ON_CONTEXT_USAGE,
        contextUsage,
        config
      );
    }
    emitAgentLog(
      config,
      'debug',
      'graph',
      'Summarize-only run complete — ending without a model call',
      { summaryVersion: agentContext.summaryVersion },
      meta
    );
    return { messages: [] };
  }

  createCallModel(agentId = 'default') {
    return async (
      state: t.AgentSubgraphState,
      config?: RunnableConfig
    ): Promise<Partial<t.AgentSubgraphState>> => {
      /** Captured at node ENTRY, before any host-facing await (context-usage
       * dispatch, hooks): a sibling's trip can fail the run and a prompt
       * next run can reset the controller while this node is paused in one
       * of those awaits, and a later capture would bind this attempt to the
       * fresh controller. Trips and reason reads below stay on this
       * capture. */
      const attemptBreaker = this.breakerAbort;
      const attemptBreakerEpoch = this.breakerEpoch;
      /** Already-tripped-at-entry: a parallel sibling's breach has failed
       * the run before this node was scheduled. Rethrow before hooks or the
       * provider call — a custom provider that doesn't synchronously reject
       * an aborted signal would otherwise start another model request on a
       * failed run. */
      const entryTripReason = this.resolveTrippedBreakerReason(
        attemptBreaker.signal
      );
      if (entryTripReason != null) {
        throw entryTripReason;
      }
      const agentContext = this.agentContexts.get(agentId);
      if (!agentContext) {
        throw new Error(`Agent context not found for agentId: ${agentId}`);
      }

      if (!config) {
        throw new Error('No config provided');
      }

      /**
       * A `PreemptBoundary` hook halted this run and the sealed commit is
       * already in state. Enforced at every model node's ENTRY because that
       * is the only site that covers all of `MultiAgentGraph`'s onward
       * routing at once — static direct edges, Command fan-out, fan-in
       * wrappers, and parallel siblings' subsequent inner-loop turns — none
       * of which consult the halt (the registry signal was deliberately
       * cleared to keep the stream-cancel from destroying the sealed turn).
       * Declining the model call turns every routed-to successor into a
       * no-op, so the outer workflow drains to END without new turns or tool
       * side effects. Reset per turn in `resetPreemptTotals`, so the next
       * `processStream` call starts clean.
       */
      if (this.preemptHaltReason != null) {
        return { messages: [] };
      }

      const { messages } = state;

      const discoveredNames = extractToolDiscoveries(messages);
      if (discoveredNames.length > 0) {
        agentContext.markToolsAsDiscovered(discoveredNames);
      }

      if (agentContext.tokenCalculationPromise) {
        await agentContext.tokenCalculationPromise;
      }
      if (!config.signal) {
        config.signal = this.signal;
      }
      this.config = config;

      let messagesToUse = messages;
      let contextUsage: t.ContextUsageEvent | null = null;
      /**
       * Held outside the prune block so overflow recovery — which detours to
       * the summarize node from the invoke catch below — can preserve the
       * same masking record the configured trigger preserves.
       */
      let prunedOriginalToolContent: Map<number, string> | undefined;
      const canProjectContext =
        agentContext.tokenCounter != null &&
        agentContext.maxContextTokens != null;
      const providerMessages = canProjectContext
        ? agentContext.getProviderProjectedMessages(messages)
        : messages;
      if (
        !agentContext.pruneMessages &&
        agentContext.tokenCounter &&
        agentContext.maxContextTokens != null
      ) {
        agentContext.pruneMessages = createPruneMessages({
          startIndex:
            agentContext.indexTokenCountMap[0] != null ? this.startIndex : 0,
          provider: agentContext.provider,
          tokenCounter: agentContext.tokenCounter,
          maxTokens: agentContext.maxContextTokens,
          maxToolResultChars: agentContext.maxToolResultChars,
          thinkingEnabled: isThinkingEnabled(
            agentContext.provider,
            agentContext.clientOptions
          ),
          indexTokenCountMap: agentContext.indexTokenCountMap,
          contextPruningConfig: agentContext.contextPruningConfig,
          summarizationEnabled: agentContext.summarizationEnabled,
          reserveRatio: agentContext.summarizationConfig?.reserveRatio,
          calibrationRatio: agentContext.calibrationRatio,
          fadingTier: agentContext.fadingTier,
          getInstructionTokens: () => agentContext.instructionTokens,
          log: (level, message, data) => {
            emitAgentLog(config, level, 'prune', message, data, {
              runId: this.runId,
              agentId,
            });
          },
        });
      }
      if (agentContext.pruneMessages) {
        const {
          context,
          indexTokenCountMap,
          messagesToRefine,
          prePruneContextTokens,
          remainingContextTokens,
          newOriginalToolContent,
          calibrationRatio,
          fadingTier,
          resolvedInstructionOverhead,
          contextBudget,
          effectiveInstructionTokens,
        } = agentContext.pruneMessages({
          messages: providerMessages,
          canonicalMessages: messages,
          canonicalPrefixStable: true,
          usageMetadata: agentContext.currentUsage,
          lastCallUsage: agentContext.lastCallUsage,
          totalTokensFresh: agentContext.totalTokensFresh,
        });
        prunedOriginalToolContent = newOriginalToolContent;
        /** Preserve the masking record for the existing overflow/summarization
         * path. The graph history itself remains canonical; only the provider
         * projection above is rewritten by fading. */
        agentContext.preserveOriginalToolContent(newOriginalToolContent);
        agentContext.indexTokenCountMap = indexTokenCountMap;
        if (calibrationRatio != null && calibrationRatio > 0) {
          agentContext.calibrationRatio = calibrationRatio;
        }
        agentContext.fadingTier = fadingTier;
        if (resolvedInstructionOverhead != null) {
          agentContext.resolvedInstructionOverhead =
            resolvedInstructionOverhead;
          const nonToolOverhead =
            agentContext.instructionTokens - agentContext.toolSchemaTokens;
          const calibratedToolTokens = Math.max(
            0,
            resolvedInstructionOverhead - nonToolOverhead
          );
          const currentToolTokens = agentContext.toolSchemaTokens;
          const variance =
            currentToolTokens > 0
              ? Math.abs(calibratedToolTokens - currentToolTokens) /
                currentToolTokens
              : 1;
          if (variance > CALIBRATION_VARIANCE_THRESHOLD) {
            agentContext.toolSchemaTokens = calibratedToolTokens;
            /** Largest-remainder apportionment keeps the per-tool breakdown
             *  summing exactly to the calibrated aggregate */
            if (agentContext.toolTokenCounts != null && currentToolTokens > 0) {
              agentContext.toolTokenCounts = apportionTokenCounts(
                agentContext.toolTokenCounts,
                calibratedToolTokens / currentToolTokens,
                calibratedToolTokens
              );
            }
          }
        }
        messagesToUse = context;

        /** Dispatched right before the model invoke — a summarization
         *  detour returns from this node without an LLM call, and the
         *  post-summary retry produces its own snapshot.
         *
         *  The breakdown describes the post-prune prompt: counts from the
         *  kept context, message tokens derived from the same calibrated
         *  budget math as `remainingContextTokens` (the index map is keyed
         *  by pre-prune state indices, so summing it over `context` would
         *  missum); `prePruneContextTokens` carries the pre-prune metric. */
        const usageBreakdown = agentContext.getTokenBudgetBreakdown(messages);
        usageBreakdown.messageCount = context.length;
        contextUsage = {
          runId: this.runId,
          agentId,
          breakdown: usageBreakdown,
          contextBudget,
          effectiveInstructionTokens,
          prePruneContextTokens,
          remainingContextTokens,
          calibrationRatio: agentContext.calibrationRatio,
        };
        syncBudgetDerivedFields(contextUsage);

        const summarizeOnlyStep = await this.summarizeOnlyStep({
          agentContext,
          agentId,
          messageCount: messages.length,
          contextUsage,
          config,
        });
        if (summarizeOnlyStep != null) {
          return summarizeOnlyStep;
        }

        const hasPrunedMessages =
          agentContext.summarizationEnabled === true &&
          Array.isArray(messagesToRefine) &&
          messagesToRefine.length > 0;

        if (hasPrunedMessages) {
          const shouldSkip =
            agentContext.summarizationExhausted ||
            agentContext.shouldSkipSummarization(messages.length);
          const triggerResult =
            !shouldSkip &&
            shouldTriggerSummarization({
              trigger: agentContext.summarizationConfig?.trigger,
              maxContextTokens: agentContext.maxContextTokens,
              prePruneContextTokens:
                prePruneContextTokens != null
                  ? prePruneContextTokens + agentContext.instructionTokens
                  : undefined,
              remainingContextTokens,
              messagesToRefineCount: messagesToRefine.length,
            });

          if (triggerResult) {
            emitAgentLog(
              config,
              'info',
              'graph',
              'Summarization triggered',
              undefined,
              { runId: this.runId, agentId }
            );
            emitAgentLog(
              config,
              'debug',
              'graph',
              'Summarization trigger details',
              {
                totalMessages: messages.length,
                remainingContextTokens: remainingContextTokens ?? 0,
                summaryVersion: agentContext.summaryVersion + 1,
                toolSchemaTokens: agentContext.toolSchemaTokens,
                instructionTokens: agentContext.instructionTokens,
                systemMessageTokens: agentContext.systemMessageTokens,
              },
              { runId: this.runId, agentId }
            );
            agentContext.markSummarizationTriggered(messages.length);
            return {
              summarizationRequest: {
                remainingContextTokens: remainingContextTokens ?? 0,
                agentId: agentId || agentContext.agentId,
              },
            };
          }

          if (shouldSkip) {
            emitAgentLog(
              config,
              'debug',
              'graph',
              'Summarization skipped — no new messages or per-run cap reached',
              {
                messageCount: messages.length,
                messagesToRefineCount: messagesToRefine.length,
                contextLength: context.length,
                summarizationFailures: agentContext.summarizationFailures,
                summarizationExhausted: agentContext.summarizationExhausted,
              },
              { runId: this.runId, agentId }
            );
          }
        }
      } else {
        /** No pruner (no token counter or budget): the summarize-only run
         *  still has to request its summary and stop, just without usage. */
        const summarizeOnlyStep = await this.summarizeOnlyStep({
          agentContext,
          agentId,
          messageCount: messages.length,
          contextUsage: null,
          config,
        });
        if (summarizeOnlyStep != null) {
          return summarizeOnlyStep;
        }
      }

      /**
       * Anthropic prompt-cache breakpoint on the tool definitions.
       *
       * Without this, the (often static) tool inventory shows up as
       * fresh input on every turn — measured at ~28k tokens/turn for
       * the local engine's coding-tool bundle, dominating per-turn
       * cost even when message-level caching is on.
       *
       * Strategy: partition tools into [static, deferred] and stamp
       * `cache_control: ephemeral` on the last static tool.
       * Discovered deferred tools that arrive across turns sit *after*
       * the breakpoint and don't invalidate the prefix.
       *
       * Built only once a model call is certain: a summarize-only step
       * returns above without one, and must not fail on a primary model
       * it never invokes.
       */
      const toolsForBinding = this.getPreparedToolsForBinding(agentContext);

      let model =
        this.overrideModel ??
        initializeModel({
          tools: toolsForBinding,
          provider: agentContext.provider,
          clientOptions: agentContext.clientOptions,
        });

      if (agentContext.systemRunnable) {
        model = agentContext.systemRunnable
          .pipe(model as Runnable)
          .withConfig({ runName: AGENT_MODEL_CALL_RUN_NAME });
      }

      let finalMessages = messagesToUse;
      /** Primary wire shaping is lossy; each fallback starts from pruned source history. */
      const fallbackBaseMessages = messagesToUse;
      /**
       * Keep the pruner's provider-grounded aggregate as the authoritative
       * baseline, then attribute it across retained messages. Provider
       * transforms can shrink one message while expanding or adding another;
       * per-origin accounting prevents that unrelated shrink from canceling
       * the expansion. Exact counts are memoized across repeated projections.
       */
      const contextPressure = createContextPressureMeter({
        provider: agentContext.provider,
        tokenCounter: agentContext.tokenCounter,
        tokenCountCache: agentContext.contextPressureTokenCounts,
        sourceMessages: messages,
        retainedMessages: messagesToUse,
        indexTokenCountMap: agentContext.indexTokenCountMap,
        contextUsage,
        instructionTokens: agentContext.instructionTokens,
        calibrationRatio: agentContext.calibrationRatio,
      });
      const trackProviderMessageOrigins = contextPressure.trackProjection;
      const toolHistory = createToolHistoryPreparation();

      if (agentContext.useLegacyContent) {
        const before = finalMessages;
        finalMessages = trackProviderMessageOrigins(
          before,
          formatContentStrings(before)
        );
      }

      const maxProviderToolResultChars =
        agentContext.maxToolResultChars ??
        calculateMaxToolResultChars(agentContext.maxContextTokens);
      const beforeToolInputProjection = finalMessages;
      finalMessages = trackProviderMessageOrigins(
        beforeToolInputProjection,
        projectToolMessagesForProvider(
          beforeToolInputProjection,
          calculateMaxToolCallInputChars(agentContext.maxContextTokens),
          agentContext.provider
        )
      );

      const lastMessageX =
        finalMessages.length >= 2
          ? finalMessages[finalMessages.length - 2]
          : null;
      const lastMessageY =
        finalMessages.length >= 1
          ? finalMessages[finalMessages.length - 1]
          : null;

      const anthropicLike = isAnthropicLike(
        agentContext.provider,
        agentContext.clientOptions as { model?: string }
      );

      if (
        agentContext.provider === Providers.BEDROCK &&
        lastMessageX instanceof AIMessageChunk &&
        lastMessageY instanceof ToolMessage &&
        typeof lastMessageX.content === 'string'
      ) {
        const trimmed = lastMessageX.content.trim();
        const before = finalMessages;
        finalMessages = [...before];
        finalMessages[finalMessages.length - 2] = cloneMessage(
          lastMessageX,
          trimmed.length > 0 ? [{ type: 'text' as const, text: trimmed }] : ''
        );
        finalMessages = trackProviderMessageOrigins(before, finalMessages);
      }

      const localProviderOverflowMeasurements = new WeakMap<
        object,
        {
          contextBudget: number;
          estimatedPromptTokens: number;
        }
      >();
      const measureProviderPayload = contextPressure.measure;

      const createProviderPayloadOverflowError = ({
        projection,
        provider,
        info,
      }: {
        projection: ReturnType<typeof measureProviderPayload>;
        provider?: t.ProviderName;
        info: string;
      }): ContextOverflowError => {
        const error = new ContextOverflowError(
          JSON.stringify({
            type: 'final_context_overflow',
            info,
            provider,
            projectedMessageTokens: projection.projectedMessageTokens,
            availableMessageTokens: projection.availableMessageTokens,
          })
        );
        if (
          projection.projectedMessageTokens != null &&
          projection.contextBudget != null &&
          projection.effectiveInstructionTokens != null
        ) {
          localProviderOverflowMeasurements.set(error, {
            contextBudget: projection.contextBudget,
            estimatedPromptTokens:
              projection.projectedMessageTokens +
              projection.effectiveInstructionTokens,
          });
        }
        return error;
      };

      const applyProviderMessageTransforms = (
        candidate: BaseMessage[],
        serving?: {
          provider: t.ProviderName;
          clientOptions?: t.ClientOptions;
          model: t.ChatModel;
          config?: RunnableConfig;
          history: ReturnType<typeof createToolHistoryPreparation>;
        }
      ): BaseMessage[] => {
        const provider = serving?.provider ?? agentContext.provider;
        const clientOptions =
          serving == null ? agentContext.clientOptions : serving.clientOptions;
        const callConfig = serving == null ? config : serving.config;
        const history = serving?.history ?? toolHistory;
        const servingModel =
          serving?.model ?? ((this.overrideModel ?? model) as t.ChatModel);
        let transformed = candidate;
        if (isThinkingEnabled(provider, clientOptions)) {
          /**
           * Current-run AI messages may validly omit a thinking block. The
           * boundary prevents them from being mistaken for foreign history.
           */
          const before = transformed;
          transformed = trackProviderMessageOrigins(
            before,
            ensureThinkingBlockInMessages(
              before,
              provider,
              callConfig,
              this.startIndex,
              history
            )
          );
        }

        /**
         * Tool-less destinations cannot send inherited tool blocks without a
         * tool schema, so fold those interactions into provider-valid content.
         */
        if (toolsForBinding == null || toolsForBinding.length === 0) {
          const before = transformed;
          transformed = trackProviderMessageOrigins(
            before,
            foldToolBlocksForToollessAgent(
              before,
              callConfig,
              usesNativeOpenAIResponses(servingModel, provider, callConfig),
              history
            )
          );
          if (agentContext.useLegacyContent) {
            const beforeLegacyFormat = transformed;
            transformed = trackProviderMessageOrigins(
              beforeLegacyFormat,
              formatContentStrings(beforeLegacyFormat)
            );
          }
        }
        /**
         * Applied HERE for the primary so the cue is part of the MEASURED
         * payload — the pre-invoke projection and overflow guard run on this
         * stage's output, and a post-measure append could push a just-fits
         * prompt over budget unreported (#346 round 2). Instructionless tool
         * handoffs are grounded for every transport. Only the direct-edge
         * predecessor cue is re-keyed per SERVING provider: tolerant
         * fallbacks strip it, while Claude fallbacks add it.
         */
        const beforeHandoffCue = transformed;
        transformed = trackProviderMessageOrigins(
          beforeHandoffCue,
          appendInstructionlessHandoffCue(beforeHandoffCue, callConfig)
        );
        if (isAnthropicLike(provider, clientOptions as { model?: string })) {
          const before = transformed;
          transformed = trackProviderMessageOrigins(
            before,
            appendPredecessorHandoffCue(before, (message) =>
              this.isRunProducedMessage(message)
            )
          );
        }
        return transformed;
      };

      const toolOutputRegistry = this.getOrCreateToolOutputRegistry();
      const providerRunId = config.configurable?.run_id as string | undefined;
      const projectProviderReferences = (
        candidate: BaseMessage[]
      ): BaseMessage[] =>
        trackProviderMessageOrigins(
          candidate,
          annotateMessagesForLLM(candidate, toolOutputRegistry, providerRunId)
        );

      const compactSyntheticProviderContext = (
        candidate: BaseMessage[],
        measure = measureProviderPayload
      ): BaseMessage[] => {
        const synthetic: Array<{
          index: number;
          message: HumanMessage;
          chars: number;
        }> = [];
        for (let i = 0; i < candidate.length; i++) {
          const message = candidate[i];
          if (
            !(message instanceof HumanMessage) ||
            !isSyntheticProviderContextMessage(message)
          ) {
            continue;
          }
          const content = message.content;
          synthetic.push({
            index: i,
            message,
            chars: getToolContentCharLength(content),
          });
        }
        if (synthetic.length === 0) {
          return candidate;
        }

        const buildCandidate = (scale: number): BaseMessage[] => {
          const compacted = [...candidate];
          for (const { index, message, chars } of synthetic) {
            compacted[index] = compactSyntheticProviderContextMessage(
              message,
              Math.floor(chars * scale)
            );
          }
          return compacted;
        };

        let best = buildCandidate(0);
        if (!measure(best).fits) {
          return candidate;
        }
        let low = 0;
        let high = 1;
        for (let i = 0; i < 12; i++) {
          const scale = (low + high) / 2;
          const attempt = buildCandidate(scale);
          if (measure(attempt).fits) {
            best = attempt;
            low = scale;
          } else {
            high = scale;
          }
        }
        return best;
      };

      const projectServingArtifacts = (
        candidate: BaseMessage[],
        provider: t.ProviderName,
        clientOptions: t.ClientOptions | undefined,
        maxChars: number
      ): BaseMessage[] => {
        if (!(candidate[candidate.length - 1] instanceof ToolMessage)) {
          return candidate;
        }
        if (isAnthropicLike(provider, clientOptions as { model?: string })) {
          return trackProviderMessageOrigins(
            candidate,
            projectAnthropicArtifactContent(candidate, maxChars)
          );
        }
        if (
          (isOpenAILike(provider) && provider !== Providers.DEEPSEEK) ||
          isGoogleLike(provider)
        ) {
          return trackProviderMessageOrigins(
            candidate,
            projectArtifactPayload(candidate, maxChars)
          );
        }
        return candidate;
      };

      const applyServingTailCache = (
        candidate: BaseMessage[],
        provider: t.ProviderName,
        clientOptions: t.ClientOptions | undefined,
        systemOwnsTail = false
      ): BaseMessage[] => {
        if (provider === Providers.BEDROCK) {
          const options = clientOptions as
            | t.BedrockAnthropicClientOptions
            | undefined;
          return options?.promptCache === true
            ? trackProviderMessageOrigins(
              candidate,
              addBedrockTailCacheControl(
                candidate,
                resolveBedrockPromptCacheTtl(
                  options.promptCacheTtl,
                  options.model
                )
              )
            )
            : candidate;
        }
        if (
          systemOwnsTail ||
          (provider !== Providers.ANTHROPIC &&
            provider !== Providers.OPENROUTER)
        ) {
          return candidate;
        }
        const options = clientOptions as t.AnthropicClientOptions | undefined;
        return options?.promptCache === true
          ? trackProviderMessageOrigins(
            candidate,
            addTailCacheControl(
              candidate,
              resolvePromptCacheTtl(options.promptCacheTtl)
            )
          )
          : candidate;
      };

      let artifactBaseMessages: BaseMessage[] | undefined;
      if (lastMessageY instanceof ToolMessage) {
        const artifactCandidate = projectServingArtifacts(
          finalMessages,
          agentContext.provider,
          agentContext.clientOptions,
          maxProviderToolResultChars
        );

        if (artifactCandidate !== finalMessages) {
          const projection = measureProviderPayload(artifactCandidate);
          if (projection.fits) {
            artifactBaseMessages = finalMessages;
            finalMessages = artifactCandidate;
          } else {
            emitAgentLog(
              config,
              'warn',
              'graph',
              'Artifact payload omitted because it exceeds the remaining context budget',
              {
                projectedMessageTokens: projection.projectedMessageTokens,
                availableMessageTokens: projection.availableMessageTokens,
              },
              { runId: this.runId, agentId }
            );
          }
        }
      }

      finalMessages = projectProviderReferences(
        applyProviderMessageTransforms(finalMessages)
      );
      let finalProjection = measureProviderPayload(finalMessages);
      if (artifactBaseMessages != null) {
        if (!finalProjection.fits) {
          finalMessages = projectProviderReferences(
            applyProviderMessageTransforms(artifactBaseMessages)
          );
          finalProjection = measureProviderPayload(finalMessages);
          emitAgentLog(
            config,
            'warn',
            'graph',
            'Artifact payload omitted after final provider formatting exceeded the remaining context budget',
            {
              projectedMessageTokens: finalProjection.projectedMessageTokens,
              availableMessageTokens: finalProjection.availableMessageTokens,
            },
            { runId: this.runId, agentId }
          );
        }
      }
      if (!finalProjection.fits) {
        const compacted = compactSyntheticProviderContext(finalMessages);
        if (compacted !== finalMessages) {
          finalMessages = compacted;
          finalProjection = measureProviderPayload(finalMessages);
          emitAgentLog(
            config,
            finalProjection.fits ? 'warn' : 'error',
            'graph',
            finalProjection.fits
              ? 'Synthetic provider context compacted to fit the final payload budget'
              : 'Final provider payload still exceeds budget after synthetic context compaction',
            {
              projectedMessageTokens: finalProjection.projectedMessageTokens,
              availableMessageTokens: finalProjection.availableMessageTokens,
            },
            { runId: this.runId, agentId }
          );
        }
      }

      /**
       * Mistral rejects consecutive user turns outright; Bedrock's Converse
       * API documents strict user/assistant alternation across its model
       * families, with enforcement varying by family (Claude on Converse
       * currently tolerates the shape — verified live — but the payload is
       * normalized for all of them rather than betting on leniency). Four
       * sites can emit them — the `PostToolBatch` and `PreemptBoundary` hook
       * boundaries (a consolidated context message followed by one
       * `HumanMessage` per injected entry), a queue drain carrying more than
       * one steer, and `run.ts`'s pre-stream context push onto a payload that
       * already ends on a user turn.
       *
       * Normalized here, at the last provider-facing hop, rather than at any
       * one boundary: the boundaries must keep per-message identity, because
       * `additional_kwargs.source`/`skillName` drive steer rendering and the
       * trailing-steer anchor downstream. Graph state and the host's
       * persisted messages are untouched — this shapes only what goes on the
       * wire, for the providers that actually care.
       *
       * Runs AFTER synthetic-context compaction: that pass can rewrite or
       * drop messages, so coalescing has to see its output, and it is the
       * last shaping step before the cache breakpoint is chosen.
       */
      if (providerRequiresStrictAlternation(agentContext.provider)) {
        /**
         * Wrapped like every other provider transform: the merged message is
         * a NEW object, and without re-attachment the final pre-invoke
         * measurement would drop both source turns' calibrated shares and
         * recharge the merge at full raw estimate — enough to flip a
         * just-fits payload (the synthetic-context compaction above binary
         * searches to exactly that) into a spurious pre-invoke overflow. The
         * merge keeps the first source's id, so the keyed branch re-attaches
         * that origin; the absorbed turn's tokens are charged as new raw
         * growth, which only ever under-estimates by less than the old
         * behavior over-estimated.
         */
        const beforeCoalesce = finalMessages;
        finalMessages = trackProviderMessageOrigins(
          beforeCoalesce,
          coalesceAdjacentUserTurns(beforeCoalesce)
        );
      }

      // Determine the prompt-cache strategy up front. Two distinct facts:
      //
      //   `providerPromptCacheEnabled` — prompt caching is on for this provider
      //   at all. This drives orphan cleanup, because EVERY cached send must be
      //   sanitized — including the system-runnable path, where AgentContext (not
      //   this node) adds the body marker.
      //
      //   `willAddTailCache` — THIS node will add the marker itself. Anthropic /
      //   OpenRouter defer to the system runnable when one owns the system-prompt
      //   breakpoint, so they exclude that case; Bedrock always marks here.
      const anthropicPromptCacheEnabled =
        agentContext.provider === Providers.ANTHROPIC &&
        (agentContext.clientOptions as t.AnthropicClientOptions | undefined)
          ?.promptCache === true;
      const openRouterPromptCacheEnabled =
        agentContext.provider === Providers.OPENROUTER &&
        (
          agentContext.clientOptions as
            | t.ProviderOptionsMap[Providers.OPENROUTER]
            | undefined
        )?.promptCache === true;
      // Message/system cache points work on all cache-capable Bedrock models,
      // including Nova (verified live: HTTP 200 with cacheWriteInputTokens). Only
      // the tool checkpoint is Claude-only, so this is gated on promptCache alone.
      const bedrockPromptCacheEnabled =
        agentContext.provider === Providers.BEDROCK &&
        (
          agentContext.clientOptions as
            | t.BedrockAnthropicClientOptions
            | undefined
        )?.promptCache === true;
      const providerPromptCacheEnabled =
        anthropicPromptCacheEnabled ||
        openRouterPromptCacheEnabled ||
        bedrockPromptCacheEnabled;

      // Intentionally broad: runs when the pruner wasn't used, when any
      // post-pruning transform (ensureThinkingBlock, etc.) reassigned
      // finalMessages, OR when this is a prompt-cached send. The last clause
      // matters because the marker is now applied AFTER this gate (and, for the
      // system-runnable path, in AgentContext entirely): without it, a cached
      // send whose pruner returned the context unchanged would skip cleanup and
      // could ship orphaned AI/tool pairs from persisted history.
      // sanitizeOrphanToolBlocks fast-paths to a Set diff check when no orphans
      // exist, so the cost is negligible.
      const needsOrphanSanitize =
        anthropicLike &&
        (!agentContext.pruneMessages ||
          finalMessages !== messagesToUse ||
          providerPromptCacheEnabled);
      if (needsOrphanSanitize) {
        const beforeSanitize = finalMessages.length;
        const beforeSanitizeMessages = finalMessages;
        finalMessages = trackProviderMessageOrigins(
          beforeSanitizeMessages,
          sanitizeOrphanToolBlocks(beforeSanitizeMessages, (source, clone) => {
            contextPressure.trackClone(source, clone);
          })
        );
        if (finalMessages.length !== beforeSanitize) {
          emitAgentLog(
            config,
            'warn',
            'sanitize',
            'Orphan tool blocks removed',
            {
              before: beforeSanitize,
              after: finalMessages.length,
              dropped: beforeSanitize - finalMessages.length,
            },
            { runId: this.runId, agentId }
          );
        }
      }

      // Place the single tail prompt-cache breakpoint LAST, after thinking
      // normalization and orphan sanitization. ensureThinkingBlockInMessages can
      // fold a trailing non-thinking AI→Tool chain into a `[Previous agent
      // context]` HumanMessage whose builder copies text but not cache_control /
      // cachePoint, and sanitizeOrphanToolBlocks can drop the anchored block — so
      // marking earlier would let the only breakpoint vanish before the model
      // call (zero message caching). Anchoring on the final message list keeps
      // the marker on a block that actually ships. The system-runnable path
      // adds its body marker in AgentContext, so this node skips it there.
      finalMessages = applyServingTailCache(
        finalMessages,
        agentContext.provider,
        agentContext.clientOptions,
        agentContext.systemRunnable != null
      );

      const beforeFinalProviderProjection = finalMessages;
      const preparedRequest = prepareProviderRequest({
        model: (this.overrideModel ?? model) as t.ChatModel,
        messages: beforeFinalProviderProjection,
        provider: agentContext.provider,
        context: this,
        config,
        maxToolResultChars: maxProviderToolResultChars,
        toolHistory,
        measure: (preparedMessages) =>
          measureProviderPayload(
            trackProviderMessageOrigins(
              beforeFinalProviderProjection,
              preparedMessages
            )
          ),
      });
      finalMessages = preparedRequest.messages;

      /**
       * Prompt-cache placement and orphan sanitization are provider-wire
       * transforms too. Re-measure after both so no content added after the
       * earlier artifact/synthetic compaction decision can bypass the guard.
       */
      finalProjection =
        preparedRequest.measurement ?? measureProviderPayload(finalMessages);
      const preInvokeContextOverflowError = !finalProjection.fits
        ? createProviderPayloadOverflowError({
          projection: finalProjection,
          provider: agentContext.provider,
          info: 'Provider message formatting exceeded the context budget and no safe synthetic-context compaction could make it fit.',
        })
        : undefined;

      if (
        agentContext.lastStreamCall != null &&
        agentContext.streamBuffer != null
      ) {
        const timeSinceLastCall = Date.now() - agentContext.lastStreamCall;
        if (timeSinceLastCall < agentContext.streamBuffer) {
          const timeToWait =
            Math.ceil((agentContext.streamBuffer - timeSinceLastCall) / 1000) *
            1000;
          await sleep(timeToWait);
        }
      }

      agentContext.lastStreamCall = Date.now();
      agentContext.markTokensStale();

      let result: Partial<t.BaseGraphState> | undefined;
      const fallbacks =
        (agentContext.clientOptions as t.LLMConfig | undefined)?.fallbacks ??
        [];

      if (
        finalMessages.length === 0 &&
        !agentContext.hasPendingCompactionSummary()
      ) {
        const budgetBreakdown = agentContext.getTokenBudgetBreakdown(messages);
        const breakdown = agentContext.formatTokenBudgetBreakdown(messages);
        const instructionsExceedBudget =
          budgetBreakdown.instructionTokens > budgetBreakdown.maxContextTokens;

        let guidance: string;
        if (instructionsExceedBudget) {
          const toolPct =
            budgetBreakdown.toolSchemaTokens > 0
              ? Math.round(
                (budgetBreakdown.toolSchemaTokens /
                    budgetBreakdown.instructionTokens) *
                    100
              )
              : 0;
          guidance =
            toolPct > 50
              ? `Tool definitions consume ${budgetBreakdown.toolSchemaTokens} tokens (${toolPct}% of instructions) across ${budgetBreakdown.toolCount} tools, exceeding maxContextTokens (${budgetBreakdown.maxContextTokens}). Reduce the number of tools or increase maxContextTokens.`
              : `Instructions (${budgetBreakdown.instructionTokens} tokens) exceed maxContextTokens (${budgetBreakdown.maxContextTokens}). Increase maxContextTokens or shorten the system prompt.`;
          if (agentContext.summarizationEnabled === true) {
            guidance +=
              ' Summarization was skipped because the summary would further increase the instruction overhead.';
          }
        } else {
          guidance =
            'Please increase the context window size or make your message shorter.';
        }

        emitAgentLog(
          config,
          'error',
          'graph',
          'Empty messages after pruning',
          {
            messageCount: messages.length,
            instructionsExceedBudget,
            breakdown,
          },
          { runId: this.runId, agentId }
        );
        throw new Error(
          JSON.stringify({
            type: 'empty_messages',
            info: `Message pruning removed all messages as none fit in the context window. ${guidance}\n${breakdown}`,
          })
        );
      }

      /** Past the empty-prompt guard — a model call is now guaranteed */
      if (contextUsage != null) {
        if (
          finalProjection.projectedMessageTokens != null &&
          finalProjection.availableMessageTokens != null
        ) {
          contextUsage.breakdown.messageCount = finalMessages.length;
          contextUsage.remainingContextTokens = Math.max(
            0,
            finalProjection.availableMessageTokens -
              finalProjection.projectedMessageTokens
          );
        }
        contextUsage.breakdown.toolMessageTokens =
          finalProjection.toolMessageTokens;
        contextUsage.breakdown.toolMessageTokenCounts =
          finalProjection.toolMessageTokenCounts;
        syncBudgetDerivedFields(
          contextUsage,
          undefined,
          undefined,
          config,
          agentContext.provider,
          finalProjection.toolMessageUsageError
        );
        /** Awaited so async host handlers receive the pre-invoke snapshot
         *  before any model deltas are emitted */
        await safeDispatchCustomEvent(
          GraphEvents.ON_CONTEXT_USAGE,
          contextUsage,
          config
        );
      }

      const invokeStart = Date.now();
      const invokeMeta = { runId: this.runId, agentId };
      emitAgentLog(
        config,
        'debug',
        'graph',
        'Invoking LLM',
        {
          messageCount: finalMessages.length,
          provider: agentContext.provider,
        },
        invokeMeta,
        { force: true }
      );

      const langfuse = resolveLangfuseConfig(
        this.langfuse,
        agentContext.langfuse
      );
      const traceMetadata = createLangfuseTraceMetadata({
        messageId: this.runId,
        parentMessageId: config.configurable?.requestBody?.parentMessageId,
        agentId,
        agentName: agentContext.name,
        rootAgentId: this.defaultAgentId,
        rootAgentName: this.agentContexts.get(this.defaultAgentId)?.name,
        activeAgentId: agentId,
        activeAgentName: agentContext.name,
        endpoint: agentContext.endpoint,
      });
      let langfuseHandler: CallbackEntry | undefined;
      let invokeConfig = {
        ...config,
        /** The run-scoped breaker composed in, so a stream-limit trip in one
         * parallel agent node also cancels sibling nodes' in-flight model
         * calls, not only their subagents. */
        signal: composeAbortSignals(config.signal, attemptBreaker.signal),
        metadata: {
          ...(config.metadata ?? {}),
          ...traceMetadata,
          /** Canonical agent identity, stamped OUTSIDE trace-metadata
           *  filtering: `createLangfuseTraceMetadata` drops values over its
           *  length cap, but scope trust (`isForeignScope`) needs the id
           *  verbatim regardless of length. */
          agentId,
          [STREAM_LIMIT_EPOCH_KEY]: attemptBreakerEpoch,
        },
      };
      initializeLangfuseTracing(langfuse);
      if (findCallback(config.callbacks, isLangfuseCallbackHandler) == null) {
        langfuseHandler = createLangfuseHandler({
          langfuse,
          userId: config.configurable?.user_id as string | undefined,
          sessionId: config.configurable?.thread_id as string | undefined,
          traceMetadata,
          tags: ['librechat', 'agent'],
          traceIdSeed:
            langfuse?.deterministicTraceId === true ? this.runId : undefined,
          traceAnchor: this.langfuseTraceAnchor,
          agentId,
          runId: this.langfuseScopeRunId,
          toolOutputTracing: hasToolOutputTracingConfig(
            this.langfuse,
            agentContext.langfuse
          )
            ? resolveToolOutputTracingConfig(
              this.langfuse,
              agentContext.langfuse
            )
            : undefined,
        });
        if (langfuseHandler != null) {
          invokeConfig = {
            ...invokeConfig,
            callbacks: appendCallbacks(invokeConfig.callbacks, [
              langfuseHandler,
            ]),
          };
        }
      }
      const metadata = config.metadata as Record<string, unknown>;

      try {
        if (preInvokeContextOverflowError != null) {
          throw preInvokeContextOverflowError;
        }
        /** Rechecked after the pre-invoke awaits (context-usage dispatch,
         * hooks): a sibling can trip the breaker while this node is paused
         * in one of them, and a provider that doesn't synchronously reject
         * an aborted signal would still start the request. */
        {
          const preInvokeTrip = this.resolveTrippedBreakerReason(
            attemptBreaker.signal
          );
          if (preInvokeTrip != null) {
            throw preInvokeTrip;
          }
        }
        result = await withLangfuseRuntimeScope(
          resolveLangfuseRuntimeScope({
            runLangfuse: this.langfuse,
            langfuseOverlay: agentContext.langfuse,
            traceAnchor: this.langfuseTraceAnchor,
            runId: this.langfuseScopeRunId,
            agentId,
          }),
          () =>
            attemptInvoke(
              {
                request: preparedRequest,
                context: this,
                preemptAgentId: agentId,
              },
              invokeConfig
            )
        );
      } catch (primaryError) {
        clearCurrentDeltaStepMarkers({
          graph: this,
          metadata,
        });
        /**
         * A tripped stream circuit breaker is a deliberate abort, not a
         * provider failure: entering overflow recovery or the fallback chain
         * would spend more provider work after the safety limit fired, and a
         * succeeding fallback would resolve a run the public contract says
         * must reject. Rethrow before any recovery path.
         */
        if (
          primaryError instanceof StreamLimitExceededError ||
          primaryError instanceof PreparedSubagentError
        ) {
          /** Tripped before rethrowing so parallel agent nodes' in-flight
           * model calls and subagents stop while the rejection propagates. */
          attemptBreaker.abort(primaryError);
          throw primaryError;
        }
        /** A sibling that tripped the shared breaker aborts this branch's
         * composed signal, and some providers surface that as a generic
         * abort error; entering overflow planning or the fallback chain
         * would start new provider work after the run-wide breaker fired.
         * Rethrow the breaker's own stream-limit reason instead. */
        {
          const trippedReason = this.resolveTrippedBreakerReason(
            attemptBreaker.signal
          );
          if (trippedReason != null) {
            throw trippedReason;
          }
        }
        /**
         * A context overflow is a deterministic consequence of the payload,
         * not a provider being unavailable — so it is answered by compacting
         * and retrying rather than by re-sending the same oversized prompt
         * down the fallback chain. Fallbacks still run for every other
         * failure, and for an overflow whose recovery budget is spent.
         */
        /**
         * Compaction has to have something to work with. Without a token
         * counter there is no pruner, and with summarization disabled the
         * summarize node deliberately no-ops — so in that combination the
         * retry would resend a byte-identical prompt. Skipping the detour
         * keeps the original error and one round trip instead of three.
         */
        const estimatedPromptTokens = getEstimatedPromptTokens(contextUsage);

        const recencyTokenCounter =
          agentContext.contextPressureTokenCounts?.count ??
          agentContext.tokenCounter;
        const canSummarizeOverflow =
          agentContext.summarizationEnabled === true &&
          splitAtRecencyBoundary(messages, {
            provider: agentContext.provider,
            turns:
              agentContext.summarizationConfig?.retainRecent?.turns ??
              DEFAULT_RETAIN_RECENT_TURNS,
            tokens: agentContext.summarizationConfig?.retainRecent?.tokens,
            tokenCounter: recencyTokenCounter,
            intraTurnTokens: resolveIntraTurnRetainTokens({
              tokens: agentContext.summarizationConfig?.retainRecent?.tokens,
              maxContextTokens: agentContext.maxContextTokens,
            }),
          }).head.length > 0;

        const getLocalProviderOverflowMeasurement = (
          error: unknown
        ):
          | {
              contextBudget: number;
              estimatedPromptTokens: number;
            }
          | undefined =>
          typeof error === 'object' && error !== null
            ? localProviderOverflowMeasurements.get(error)
            : undefined;

        const getRecoveryPromptEstimate = (
          error: unknown,
          fallbackContext?: FallbackErrorContext
        ): number | undefined => {
          const resolvedFallbackContext =
            fallbackContext ?? getFallbackErrorContext(error);
          return (
            getLocalProviderOverflowMeasurement(error)?.estimatedPromptTokens ??
            (resolvedFallbackContext == null
              ? estimatedPromptTokens
              : undefined)
          );
        };

        const planRecovery = (
          error: unknown,
          attributedFallbackContext?: FallbackErrorContext
        ): OverflowRecoveryPlan | null => {
          /**
           * When the rejection came from a fallback, plan against *that*
           * client: its window and output allowance are why it was configured
           * as an alternative in the first place.
           */
          const fallbackContext =
            attributedFallbackContext ?? getFallbackErrorContext(error);
          const localMeasurement = getLocalProviderOverflowMeasurement(error);
          const recoveryPromptEstimate = getRecoveryPromptEstimate(
            error,
            fallbackContext
          );
          /**
           * A previous correction that left the rejected prompt no smaller
           * proves this state has nothing left to compact. Use the fallback
           * projection when one exists so unlike provider formats are never
           * compared through the primary's cheaper pre-projection estimate.
           */
          if (agentContext.overflowRecoveryStalled(recoveryPromptEstimate)) {
            return null;
          }
          const recovery = planContextOverflowRecovery({
            error,
            provider: fallbackContext?.provider ?? agentContext.provider,
            maxContextTokens:
              localMeasurement?.contextBudget ??
              fallbackContext?.maxContextTokens ??
              agentContext.maxContextTokens,
            estimatedPromptTokens: recoveryPromptEstimate,
            calibrationRatio: agentContext.calibrationRatio,
            instructionTokens: agentContext.instructionTokens,
            canSummarize: agentContext.summarizationEnabled === true,
            configuredCompletionTokens: getConfiguredCompletionTokens(
              fallbackContext?.clientOptions ?? agentContext.clientOptions
            ),
            attemptsSoFar: agentContext.overflowRecoveryAttempts,
          });
          if (recovery == null) {
            return null;
          }
          const translatedRecovery =
            fallbackContext != null
              ? {
                ...recovery,
                budgetTokens: minDefined(
                  getBlindRecoveryBudget(agentContext.maxContextTokens),
                  localMeasurement != null
                    ? recovery.budgetTokens
                    : translateRecoveryBudget(
                      recovery.budgetTokens,
                      recovery.observedCalibrationRatio ??
                            CALIBRATION_RATIO_MAX,
                      agentContext.calibrationRatio
                    )
                ),
                observedCalibrationRatio: undefined,
              }
              : recovery;
          const canReduceContext =
            canSummarizeOverflow ||
            (agentContext.tokenCounter != null &&
              translatedRecovery.budgetTokens != null);
          return canReduceContext ? translatedRecovery : null;
        };

        const recovery = planRecovery(primaryError);
        if (recovery != null) {
          const recoveryPromptEstimate =
            getRecoveryPromptEstimate(primaryError);
          return this.beginOverflowRecovery({
            recovery,
            agentContext,
            agentId,
            config,
            originalToolContent: prunedOriginalToolContent,
            estimatedPromptTokens: recoveryPromptEstimate,
          });
        }
        /**
         * A fallback can reject the same prompt as too large even when the
         * primary failed for an unrelated reason — a fallback with a smaller
         * window is the obvious case. Planning against the exhausted-chain
         * error keeps that path recoverable instead of surfacing it.
         */
        try {
          result = await withLangfuseRuntimeScope(
            resolveLangfuseRuntimeScope({
              runLangfuse: this.langfuse,
              langfuseOverlay: agentContext.langfuse,
              runId: this.langfuseScopeRunId,
              agentId,
            }),
            () =>
              tryFallbackProviders({
                fallbacks,
                tools: agentContext.tools,
                messages: fallbackBaseMessages,
                config: invokeConfig,
                primaryError,
                context: this,
                preemptAgentId: agentId,
                /**
                 * Lets the chain recognise a fallback overflow whose signature
                 * carries no reason of its own (Vertex AI's bare 400) and
                 * surface it rather than a later unrelated failure.
                 */
                overflowContext: {
                  provider: agentContext.provider,
                  estimatedPromptTokens: getEstimatedPromptTokens(contextUsage),
                  maxContextTokens: agentContext.maxContextTokens,
                },
                prepareProviderRequest: ({
                  model: fallbackModel,
                  messages: fallbackMessages,
                  provider: fallbackProvider,
                  clientOptions: fallbackClientOptions,
                  maxContextTokens: fallbackMaxContextTokens,
                  config: fallbackConfig,
                }) => {
                  const fallbackToolHistory = createToolHistoryPreparation();
                  const fallbackToolResultChars =
                    agentContext.maxToolResultChars ??
                    calculateMaxToolResultChars(
                      fallbackMaxContextTokens ?? agentContext.maxContextTokens
                    );
                  const primaryContextBudget = contextUsage?.contextBudget;
                  const fallbackContextBudget =
                    fallbackMaxContextTokens == null
                      ? primaryContextBudget
                      : Math.min(
                        primaryContextBudget ?? fallbackMaxContextTokens,
                        fallbackMaxContextTokens
                      );
                  const measureFallback = (
                    candidate: BaseMessage[]
                  ): ReturnType<typeof measureProviderPayload> =>
                    measureProviderPayload(candidate, {
                      contextBudget: fallbackContextBudget,
                      forceRawRecount: true,
                    });
                  const source = agentContext.useLegacyContent
                    ? trackProviderMessageOrigins(
                      fallbackMessages,
                      formatContentStrings(fallbackMessages)
                    )
                    : fallbackMessages;
                  const boundedSource = trackProviderMessageOrigins(
                    source,
                    projectToolMessagesForProvider(
                      source,
                      calculateMaxToolCallInputChars(fallbackMaxContextTokens),
                      fallbackProvider
                    )
                  );
                  const artifactSource = projectServingArtifacts(
                    boundedSource,
                    fallbackProvider,
                    fallbackClientOptions,
                    fallbackToolResultChars
                  );
                  const transformFallback = (
                    candidate: BaseMessage[]
                  ): BaseMessage[] =>
                    applyProviderMessageTransforms(candidate, {
                      provider: fallbackProvider,
                      clientOptions: fallbackClientOptions,
                      model: fallbackModel,
                      config: fallbackConfig,
                      history: fallbackToolHistory,
                    });
                  const prepareFallback = (
                    candidate: BaseMessage[]
                  ): {
                    request: ReturnType<typeof prepareProviderRequest>;
                    projection: ReturnType<typeof measureProviderPayload>;
                  } => {
                    const sanitized = isAnthropicLike(
                      fallbackProvider,
                      fallbackClientOptions as { model?: string }
                    )
                      ? trackProviderMessageOrigins(
                        candidate,
                        sanitizeOrphanToolBlocks(
                          candidate,
                          (original, clone) =>
                            contextPressure.trackClone(original, clone)
                        )
                      )
                      : candidate;
                    const cached = applyServingTailCache(
                      sanitized,
                      fallbackProvider,
                      fallbackClientOptions
                    );
                    let projection:
                      | ReturnType<typeof measureProviderPayload>
                      | undefined;
                    const request = prepareProviderRequest({
                      model: fallbackModel,
                      messages: cached,
                      provider: fallbackProvider,
                      context: this,
                      config: fallbackConfig,
                      maxToolResultChars: fallbackToolResultChars,
                      toolHistory: fallbackToolHistory,
                      measure: (prepared) => {
                        projection = measureFallback(
                          trackProviderMessageOrigins(cached, prepared)
                        );
                        return projection;
                      },
                    });
                    if (projection == null) {
                      throw new Error(
                        'Fallback preparation did not measure its payload'
                      );
                    }
                    return { request, projection };
                  };
                  let servingFallbackMessages =
                    transformFallback(artifactSource);
                  let preparedFallbackRequest = prepareFallback(
                    servingFallbackMessages
                  );
                  let projection = preparedFallbackRequest.projection;
                  if (!projection.fits && artifactSource !== boundedSource) {
                    servingFallbackMessages = transformFallback(boundedSource);
                    preparedFallbackRequest = prepareFallback(
                      servingFallbackMessages
                    );
                    projection = preparedFallbackRequest.projection;
                  }
                  if (!projection.fits) {
                    const compacted = compactSyntheticProviderContext(
                      servingFallbackMessages,
                      (candidate) => prepareFallback(candidate).projection
                    );
                    if (compacted !== servingFallbackMessages) {
                      preparedFallbackRequest = prepareFallback(compacted);
                      projection = preparedFallbackRequest.projection;
                    }
                  }
                  if (!projection.fits) {
                    throw createProviderPayloadOverflowError({
                      projection,
                      provider: fallbackProvider,
                      info: 'Fallback provider message formatting exceeded the context budget before invocation.',
                    });
                  }
                  return preparedFallbackRequest.request;
                },
              })
          );
        } catch (fallbackError) {
          if (
            fallbackError instanceof StreamLimitExceededError ||
            fallbackError instanceof PreparedSubagentError
          ) {
            /** Same treatment as the primary path: a fallback stream that
             * trips the breaker must stop parallel agent nodes' model calls
             * and subagents before the rejection propagates. */
            attemptBreaker.abort(fallbackError);
            throw fallbackError;
          }
          {
            /** Same sibling-abort translation guard as the primary catch. */
            const trippedReason = this.resolveTrippedBreakerReason(
              attemptBreaker.signal
            );
            if (trippedReason != null) {
              throw trippedReason;
            }
          }
          const overflowCandidates =
            getFallbackOverflowCandidates(fallbackError);
          let fallbackRecovery: OverflowRecoveryPlan | null = null;
          let fallbackRecoveryPromptEstimate: number | undefined;
          for (const candidate of overflowCandidates) {
            fallbackRecovery = planRecovery(candidate.error, candidate.context);
            if (fallbackRecovery != null) {
              fallbackRecoveryPromptEstimate = getRecoveryPromptEstimate(
                candidate.error,
                candidate.context
              );
              break;
            }
          }
          if (overflowCandidates.length === 0) {
            fallbackRecovery = planRecovery(fallbackError);
            fallbackRecoveryPromptEstimate =
              getRecoveryPromptEstimate(fallbackError);
          }
          if (fallbackRecovery == null) {
            throw fallbackError;
          }
          return this.beginOverflowRecovery({
            recovery: fallbackRecovery,
            agentContext,
            agentId,
            config,
            originalToolContent: prunedOriginalToolContent,
            estimatedPromptTokens: fallbackRecoveryPromptEstimate,
          });
        }
      } finally {
        await disposeLangfuseHandler(langfuseHandler);
      }

      if (!result) {
        throw new Error('No result after model invocation');
      }

      /**
       * Fallback: populate toolCallStepIds in the graph execution context.
       *
       * When model.stream() is available (the common case), attemptInvoke
       * processes all chunks through a local ChatModelStreamHandler which
       * creates run steps and populates toolCallStepIds before returning.
       * The code below is a fallback for the rare case where model.stream
       * is unavailable and model.invoke() was used instead.
       *
       * Text content is dispatched FIRST so that MESSAGE_CREATION is the
       * current step when handleToolCalls runs. handleToolCalls then creates
       * TOOL_CALLS on top of it. The dedup in getMessageId and
       * toolCallStepIds.has makes this safe when attemptInvoke already
       * handled everything — both paths become no-ops.
       */
      const responseMessage = result.messages?.[0];
      /**
       * Provenance for the handoff-cue gate: recorded at the node, where the
       * produced turn is unambiguous. The public ChatModel contract does not
       * require implementations to set message ids — the reducer would
       * assign one AFTER this node returns, which is too late for the set —
       * so an id is assigned here first, the same way the reducer does it
       * (`v4()`, mirrored into `lc_kwargs`), and the reducer's
       * keep-existing-id rule makes the state message match.
       */
      if (responseMessage?.getType() === 'ai') {
        if (
          typeof responseMessage.id !== 'string' ||
          responseMessage.id === ''
        ) {
          responseMessage.id = v4();
          responseMessage.lc_kwargs.id = responseMessage.id;
        }
        this.runProducedAiMessageIds.add(responseMessage.id);
      }
      const toolCalls = (responseMessage as AIMessageChunk | undefined)
        ?.tool_calls;
      const hasToolCalls = Array.isArray(toolCalls) && toolCalls.length > 0;
      const responseReasoningContent = getResponseReasoningContent({
        responseMessage: responseMessage as Partial<AIMessageChunk> | undefined,
        reasoningKey: agentContext.reasoningKey,
      });
      const textMessageContent = getMessageDeltaContent(
        agentContext.provider,
        responseMessage?.content as MessageContent | undefined
      );
      const hasStreamedTextDeltaStep = hasCurrentTextDeltaStep({
        graph: this,
        metadata,
      });
      const hasStreamedReasoningDeltaStep = hasCurrentReasoningDeltaStep({
        graph: this,
        metadata,
      });
      const dispatchableFinalReasoningContent =
        getDispatchableFinalReasoningContent({
          agentContext,
          responseReasoningContent,
          hasStreamedTextDeltaStep,
          hasStreamedReasoningDeltaStep,
        });

      if (hasToolCalls) {
        const dispatchedReasoning =
          dispatchableFinalReasoningContent != null &&
          (await dispatchReasoningContent({
            graph: this,
            agentContext,
            reasoningContent: dispatchableFinalReasoningContent,
            metadata,
          }));
        if (dispatchedReasoning) {
          markPostReasoningContent(agentContext);
        }
        if (textMessageContent != null && !hasStreamedTextDeltaStep) {
          const stepKey = this.getStepKey(metadata);
          const dispatchedText = await dispatchTextMessageContent({
            graph: this,
            stepKey,
            provider: agentContext.provider,
            content: textMessageContent,
            metadata,
          });
          if (dispatchedText) {
            markPostReasoningContent(agentContext);
          }
        }

        await handleToolCalls(toolCalls as ToolCall[], metadata, this);
      }

      /**
       * When streaming events are unavailable, ChatModelStreamHandler never
       * fires. Dispatch final reasoning/text content here. getMessageId makes
       * this a no-op when the streaming path already handled the same step.
       */
      if (!hasToolCalls && responseMessage != null) {
        const dispatchedReasoning =
          dispatchableFinalReasoningContent != null &&
          (await dispatchReasoningContent({
            graph: this,
            agentContext,
            reasoningContent: dispatchableFinalReasoningContent,
            metadata,
          }));
        if (dispatchedReasoning && textMessageContent != null) {
          markPostReasoningContent(agentContext);
        }
        if (textMessageContent != null && !hasStreamedTextDeltaStep) {
          const stepKey = this.getStepKey(metadata);
          await dispatchTextMessageContent({
            graph: this,
            stepKey,
            provider: agentContext.provider,
            content: textMessageContent,
            metadata,
          });
        }
      }

      const invokeElapsed = ((Date.now() - invokeStart) / 1000).toFixed(2);
      agentContext.currentUsage = this.getUsageMetadata(result.messages?.[0]);
      /**
       * Synthetic usage from a sealed turn is an estimate derived from the
       * host's own counter, so feeding it to calibration would teach a ratio
       * of exactly 1.0 — self-consistent by construction, and wrong for any
       * provider whose real ratio differs. It still flows to `currentUsage`
       * for host billing; it just does not get to move the EMA.
       */
      const estimatedUsage =
        (result.messages?.[0] as AIMessageChunk | undefined)?.response_metadata
          .estimated_usage === true;
      if (agentContext.currentUsage) {
        if (!estimatedUsage) {
          agentContext.updateLastCallUsage(agentContext.currentUsage);
        }
        emitAgentLog(
          config,
          'debug',
          'graph',
          `LLM call complete (${invokeElapsed}s)`,
          {
            ...agentContext.currentUsage,
            elapsedSeconds: Number(invokeElapsed),
            instructionTokens: agentContext.instructionTokens,
            toolSchemaTokens: agentContext.toolSchemaTokens,
            messageCount: finalMessages.length,
          },
          invokeMeta,
          { force: true }
        );
      } else {
        emitAgentLog(
          config,
          'debug',
          'graph',
          `LLM call complete (${invokeElapsed}s)`,
          {
            elapsedSeconds: Number(invokeElapsed),
            messageCount: finalMessages.length,
          },
          invokeMeta,
          { force: true }
        );
      }
      const preemptRestarted = this.consumePreemptRestart(agentId);
      if (
        preemptRestarted ||
        (responseMessage as AIMessageChunk | undefined)?.response_metadata
          .preempted === true
      ) {
        const { messages: injected, preventContinuation } =
          await this.dispatchPreemptBoundary(agentId, config);
        /**
         * Release before branching: the slot is held only for the duration of
         * the drain, and an early return below must not strand it.
         */
        this.releasePreemptSeal();
        if (preventContinuation) {
          /**
           * A hook halted at the boundary. Commit the sealed turn and anything
           * it injected, but do NOT self-loop: `preventContinuation` promises
           * no further model turn, and the run-loop poll in `processStream`
           * only sees the halt AFTER the next call would already have started
           * — direct graph consumers never poll it at all. A trailing injected
           * HumanMessage carries no tool calls, so `toolsCondition` routes it
           * to END.
           */
          this.preemptIncomplete = true;
          /**
           * A halting boundary that ALSO injected nothing is still an empty
           * boundary by the `getPreemptStats().emptyBoundaries` contract —
           * hosts use the counter for truncated-seal telemetry, and both
           * paths end the turn with nothing to resume from.
           */
          /**
           * Seals only. `emptyBoundaries` means a kept assistant turn that was
           * truncated with nothing to resume from; a restart preserved no turn
           * at all, so a halted one is not that — `preemptIncomplete` above
           * already records that the answer never arrived.
           */
          if (injected.length === 0 && !preemptRestarted) {
            this.preemptEmptyBoundaries += 1;
          }
          this.cleanupSignalListener();
          return injected.length > 0
            ? { messages: [...(result.messages ?? []), ...injected] }
            : result;
        }
        if (injected.length > 0) {
          this.pendingPreemptReturn.add(agentId);
          this.cleanupSignalListener();
          return { messages: [...(result.messages ?? []), ...injected] };
        }
        /**
         * Nothing to inject — the host cancelled or already drained.
         *
         * After a SEAL, do not self-loop: a trailing model turn with no new
         * input is dropped by some Gemini models and read as prefill by
         * Anthropic. Do not pretend the turn completed either; the answer
         * really was cut short.
         *
         * After a RESTART there is no such trailing turn. The discard left
         * graph state exactly as the model node found it, so returning to the
         * node re-issues the same call — the only way the run can still
         * produce an answer, and the honest outcome of an interrupt whose
         * words were withdrawn before they landed. Bounded by the same seal
         * budget, so a host stuck arming and cancelling cannot loop forever.
         */
        if (preemptRestarted) {
          /**
           * NOT counted as an empty boundary. That counter means a seal that
           * ended the turn early with nothing to resume from — hosts read it
           * as truncated-answer telemetry — and this branch is the opposite:
           * the call is reissued and the run goes on to produce a complete
           * answer. Counting it here would make a successful restart look like
           * a truncated one in every host that persists on that signal.
           */
          this.pendingPreemptReturn.add(agentId);
          this.cleanupSignalListener();
          return result;
        }
        this.preemptEmptyBoundaries += 1;
        this.preemptIncomplete = true;
      }

      const responseHandler = this.handlerRegistry?.getHandler(
        GraphEvents.ON_MODEL_RESPONSE
      );
      if (responseHandler != null && responseMessage?.getType() === 'ai') {
        try {
          // One graph-owned accepted result after all primary/fallback/overflow paths.
          // No inference from provider chunks, run-step IDs, or attempt callback metadata.
          invokeConfig.signal?.throwIfAborted();
          const finalResponse = responseMessage as AIMessageChunk;
          let snapshot: Pick<t.ModelResponseEvent, 'toolCalls' | 'invalidToolCalls'>;
          try {
            snapshot = structuredClone({
              toolCalls: finalResponse.tool_calls ?? [],
              invalidToolCalls: finalResponse.invalid_tool_calls ?? [],
            });
          } catch {
            // DataCloneError may include source values. Never surface tool arguments.
            throw new Error('Accepted model response contains non-serializable tool calls');
          }
          const accepted: t.ModelResponseEvent = {
            type: 'model_response',
            id: v4(),
            agentId,
            ...snapshot,
          };
          // Awaited, registry-only: no trace replay, usage recording or side effects.
          // Detached calls prevent a consumer from changing tools about to execute.
          await responseHandler.handle(
            GraphEvents.ON_MODEL_RESPONSE,
            accepted,
            metadata,
            this
          );
          invokeConfig.signal?.throwIfAborted();
        } catch (error) {
          this.cleanupSignalListener();
          throw error;
        }
      }

      this.cleanupSignalListener();
      return result;
    };
  }

  /**
   * Fires `PreemptBoundary` after a sealed turn and returns whatever the
   * hooks asked to inject, converted through the same `convertInjectedMessages`
   * the tool boundary uses so the two sites cannot emit different shapes.
   *
   * Never throws: a drain that fails or times out costs the injection, not the
   * run. The caller treats an empty result as "nothing to resume with".
   *
   * `preventContinuation` is surfaced alongside the messages rather than left
   * to the registry halt signal, which `processStream` only polls between
   * stream events — by then the self-loop it was meant to prevent has already
   * issued another model call, and a direct graph consumer never polls it.
   */
  private async dispatchPreemptBoundary(
    agentId: string,
    config: RunnableConfig | undefined
  ): Promise<PreemptBoundaryResult> {
    if (this.hookRegistry == null) {
      return EMPTY_PREEMPT_BOUNDARY;
    }
    const configurable = config?.configurable;
    const runId = (configurable?.run_id as string | undefined) ?? this.runId;
    if (runId == null) {
      return EMPTY_PREEMPT_BOUNDARY;
    }
    const result = await executeHooks({
      registry: this.hookRegistry,
      input: {
        hook_event_name: 'PreemptBoundary',
        runId,
        threadId: configurable?.thread_id as string | undefined,
        agentId: this.subagentScope ? agentId : undefined,
        executionContext: this.subagentExecutionContext,
        executingAgentId: agentId,
        sealCount: this.preemptSealCount,
      },
      sessionId: runId,
      timeoutMs: PREEMPT_BOUNDARY_HOOK_TIMEOUT_MS,
      /**
       * The host's own abort signal(s), deliberately NOT `config.signal` —
       * inside a node the latter is LangGraph's composed signal, which also
       * fires when an unrelated sibling in the same superstep throws.
       * Cancellation already returns control in milliseconds without this;
       * what it buys is that a drain does not keep running after the run it
       * belongs to died.
       *
       * Composed because the host can cancel through either channel: the
       * construction signal, or the per-call `callerConfig.signal` — the only
       * one a multi-agent run has, since `MultiAgentGraphConfig` exposes no
       * construction signal. When both exist they may be different
       * controllers, and a drain must stop when EITHER fires.
       */
      signal: composeAbortSignals(this.signal, this.callerSignal),
    }).catch((): undefined => undefined);
    if (result == null) {
      return EMPTY_PREEMPT_BOUNDARY;
    }
    /**
     * `executeHooks` raises a registry halt whenever a hook returns
     * `preventContinuation`. That halt has exactly one consumer — the poll in
     * `Run.processStream` — and its `break` cancels the stream iterator, which
     * aborts Pregel. The abort lands BEFORE the outer reducer commits
     * `StandardGraph.messages`, so honoring the halt here would destroy the
     * sealed assistant turn: the run returns empty content and the host
     * persists nothing. Measured deterministically — the commit is several
     * stream events downstream of the point the halt becomes observable.
     *
     * The `preventContinuation` branch in `createCallModel` already enforces
     * the contract locally by declining to self-loop, and a sealed chunk
     * provably carries no tool calls, so the turn routes to END after exactly
     * one model call either way. Clearing the halt therefore costs nothing it
     * was buying and saves the content the seal exists to preserve.
     *
     * Scoped to a halt this event raised, so a halt from an earlier hook in
     * the same run — `haltRun` is first-write-wins — is left alone.
     */
    const halt = this.hookRegistry.getHaltSignal(runId);
    if (
      result.preventContinuation === true &&
      halt?.source === 'PreemptBoundary'
    ) {
      this.preemptHaltReason = halt.reason;
      this.hookRegistry.clearHaltSignal(runId);
    }
    const injected: BaseMessage[] = [];
    /**
     * `PreemptBoundaryHookOutput` is `BaseHookOutput`, so `additionalContext`
     * is part of the contract here just as it is at the tool boundary. It has
     * to be materialized BEFORE the emptiness test, or a hook that returns
     * context alone would read as "nothing to resume with" and cut the answer
     * short. Same system-flavored `HumanMessage` convention `ToolNode` uses —
     * Anthropic and Google reject a mid-conversation `SystemMessage`.
     */
    /**
     * Whitespace-only entries are dropped for the same reason empty
     * `injectedMessages` are: `executeHooks` keeps them because their raw
     * length is nonzero, but a blank turn is not something to resume from —
     * it costs a model call and strict providers reject it outright.
     */
    const contexts = result.additionalContexts.filter(
      (context) => context.trim() !== ''
    );
    if (contexts.length > 0) {
      injected.push(
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
    if (result.injectedMessages.length > 0) {
      try {
        const convertedMessages = convertInjectedMessages(
          result.injectedMessages
        );
        for (const convertedMessage of convertedMessages) {
          injected.push(convertedMessage);
        }
      } catch (e) {
        console.warn(
          '[StandardGraph] Failed to convert PreemptBoundary injectedMessages:',
          e instanceof Error ? e.message : e
        );
      }
    }
    return {
      messages: injected,
      preventContinuation: result.preventContinuation === true,
    };
  }

  protected createRunStepStateAnnotation(): RunStepStateChannel {
    return buildRunStepStateAnnotation();
  }

  createAgentNode(agentId: string): t.CompiledAgentWorfklow {
    const getConfig = (): RunnableConfig | undefined => this.config;
    const agentContext = this.agentContexts.get(agentId);
    if (!agentContext) {
      throw new Error(`Agent context not found for agentId: ${agentId}`);
    }

    /**
     * Depth countdown across graph boundaries: the parent's `maxSubagentDepth`
     * becomes this executor's `maxDepth`. When the child graph is constructed,
     * `buildChildInputs()` decrements `maxSubagentDepth` on the child's
     * `AgentInputs` (only when `allowNested: true`; otherwise subagentConfigs
     * are stripped entirely). The child graph's own `createAgentNode()` then
     * reads the decremented value here and creates a narrower executor —
     * recursion is bounded even though each graph has its own separate
     * executor instance.
     */
    const effectiveSubagentDepth = agentContext.maxSubagentDepth ?? 1;
    if (
      agentContext.subagentConfigs != null &&
      agentContext.subagentConfigs.length > 0 &&
      effectiveSubagentDepth > 0
    ) {
      const executableConfigs = normalizeSubagentConfigEntries(
        agentContext.subagentConfigs,
        agentContext
      );
      if (executableConfigs.length > 0) {
        if (
          !this.supportsMultiAgentChildren &&
          executableConfigs.some(isGraphSubagentConfig)
        ) {
          throw new Error(
            'Graph subagents require constructing the parent with createGraph() or an injected GraphFactory dependency.'
          );
        }
        const getParentHandlerRegistry = (): HandlerRegistry | undefined =>
          this.handlerRegistry ?? this.parentToolHandlerRegistry;
        const snapshotChildGraphFactory = (
          parentHandlerRegistry: HandlerRegistry | undefined
        ): GraphFactory => {
          const graphFactory = this.graphFactory;
          const subagentModelOverride = this.subagentModelOverride;
          const runtimeConfig = {
            hookRegistry: this.hookRegistry,
            humanInTheLoop: this.humanInTheLoop,
            toolOutputReferences: this.toolOutputReferences,
            eagerEventToolExecution: this.eagerEventToolExecution,
            codeSessionToolNames: this.codeSessionToolNames,
            interruptingToolNames: this.interruptingToolNames,
            toolExecution: this.toolExecution,
          };
          const checkpointer = this.compileOptions?.checkpointer;
          return (request): StandardGraph => {
            const configuredRequest: GraphFactoryRequest =
              request.kind === 'multi-agent'
                ? {
                  kind: 'multi-agent',
                  input: {
                    ...request.input,
                    toolExecution: runtimeConfig.toolExecution,
                  },
                }
                : {
                  kind: 'standard',
                  input: {
                    ...request.input,
                    toolExecution: runtimeConfig.toolExecution,
                  },
                };
            const childGraph = graphFactory(configuredRequest);
            if (subagentModelOverride != null) {
              childGraph.overrideModel = subagentModelOverride;
              childGraph.setSubagentModelOverride(subagentModelOverride);
            }
            const childHandlerRegistry = createChildHandlerRegistry(
              parentHandlerRegistry
            );
            // Pure execution-ordering hint (unlike `humanInTheLoop`). It only
            // reorders tools already in the child's direct group; it does not
            // force a schema-only event tool onto the direct execution path.
            applyGraphRuntimeConfig(childGraph, runtimeConfig);
            if (runtimeConfig.humanInTheLoop?.enabled === true) {
              childGraph.compileOptions = { checkpointer };
            }
            childGraph.parentToolHandlerRegistry = childHandlerRegistry;
            childGraph.eventToolExecutionAvailable =
              childHandlerRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) !=
              null;
            return childGraph;
          };
        };
        const createConfiguredChildGraph: GraphFactory = (request) =>
          snapshotChildGraphFactory(getParentHandlerRegistry())(request);
        const executor = new SubagentExecutor({
          configs: new Map(
            executableConfigs.map((config) => [config.type, config])
          ),
          parentSignal: this.signal,
          breakerScope: {
            controller: (): AbortController => this.breakerAbort,
          },
          hookRegistry: this.hookRegistry,
          /** Lazy — Run wires the registry onto the graph AFTER
           *  `createWorkflow()` runs, so a direct capture here would be
           *  `undefined` at construction time. */
          parentHandlerRegistry: getParentHandlerRegistry,
          parentRunId: this.runId ?? '',
          parentAgentId: agentContext.agentId,
          executionContext: this.subagentExecutionContext,
          langfuse: this.langfuse,
          tokenCounter: agentContext.tokenCounter,
          usageSink: this.subagentUsageSink,
          taskConfig: this.subagentTasks,
          subagentContext: this.subagentContext,
          streamLimits: this.streamLimits,
          humanInTheLoop: this.humanInTheLoop,
          checkpointer: this.compileOptions?.checkpointer,
          maxDepth: effectiveSubagentDepth,
          createChildGraph: (input): StandardGraph =>
            createConfiguredChildGraph({
              kind: 'standard',
              input,
            }),
          createChildGraphByKind: createConfiguredChildGraph,
          createDetachedChildGraphFactory: (
            parentHandlerRegistry
          ): GraphFactory => snapshotChildGraphFactory(parentHandlerRegistry),
        });
        this.registerSubagentExecutor(executor);

        const subagentTool = tool(
          async (rawInput, config) => {
            const input = rawInput as {
              description?: string;
              subagent_type?: string;
              subagent_thread_id?: string;
              run_in_background?: boolean;
            };
            const description =
              typeof input.description === 'string' &&
              input.description.trim().length > 0
                ? input.description
                : DEFAULT_SUBAGENT_DESCRIPTION;
            const subagentType =
              typeof input.subagent_type === 'string'
                ? input.subagent_type
                : '';
            const subagentThreadId =
              typeof input.subagent_thread_id === 'string' &&
              input.subagent_thread_id.trim() !== ''
                ? input.subagent_thread_id.trim()
                : undefined;
            const threadId = config.configurable?.thread_id as
              | string
              | undefined;
            /** Surface the parent call id so child checkpoints, interrupts, and
             * update events remain correlated across replay and resume. */
            const toolRuntime = config as {
              toolCallId?: string;
              toolCall?: { id?: string };
            };
            const toolCall = toolRuntime.toolCall;
            let parentToolCallId: string | undefined;
            if (
              typeof toolRuntime.toolCallId === 'string' &&
              toolRuntime.toolCallId !== ''
            ) {
              parentToolCallId = toolRuntime.toolCallId;
            } else if (typeof toolCall?.id === 'string' && toolCall.id !== '') {
              parentToolCallId = toolCall.id;
            }
            /** The parent tool batch's entry-captured scope (stamped by
             * ToolNode before PreToolUse hooks) — binds this child to the
             * run that dispatched it, not to whatever controller a reset
             * installed while the hooks were awaited. */
            const batchScope = config.configurable?.[
              RUN_BREAKER_SCOPE_CONFIG_KEY
            ] as RunBreakerScope | undefined;
            const executeParams = {
              description,
              subagentType,
              threadId,
              signal: config.signal,
              parentToolCallId,
              breaker: batchScope?.controller,
              /**
               * Forward the parent's `configurable` so host-set fields
               * (`requestBody`, `user`, etc.) propagate into the child
               * workflow. The executor scrubs run-identity fields before
               * forwarding — see `SubagentExecuteParams.parentConfigurable`.
               */
              parentConfigurable: config.configurable as
                | Record<string, unknown>
                | undefined,
            };
            if (input.run_in_background === true) {
              return executor.executeInBackground({
                ...executeParams,
                ...(subagentThreadId == null ? {} : { subagentThreadId }),
              });
            }
            if (subagentThreadId != null) {
              return JSON.stringify({
                status: 'rejected',
                tool: Constants.SUBAGENT,
                message:
                  'Child-thread continuation requires run_in_background.',
              });
            }
            const result = await executor.execute(executeParams);
            if (result.retryableDelivery === true) {
              throw new Error(result.content);
            }
            return result.content;
          },
          buildSubagentToolParams(executableConfigs, {
            background: this.subagentTasks != null,
            threadContinuation:
              this.subagentTasks?.store.supportsThreadContinuation === true,
          })
        );
        const replayableSubagentTool = subagentTool as typeof subagentTool &
          ReplayableSubagentTool;
        replayableSubagentTool[SUBAGENT_REPLAY_CONTROLLER] = {
          getResumeManifest: (
            parentToolCallIds,
            config
          ): Promise<SubagentResumeManifest | undefined> =>
            executor.getResumeManifest(parentToolCallIds, config),
          getSettledOutput: (
            call,
            config
          ): Promise<SettledSubagentToolOutput | undefined> =>
            executor.getSettledToolOutput(call, config),
          persistSettledOutput: (call, config, output): Promise<void> =>
            executor.persistSettledToolOutput(call, config, output),
        };

        if (!agentContext.graphTools) {
          agentContext.graphTools = [];
        }
        (agentContext.graphTools as t.GenericTool[]).push(subagentTool);

        /**
         * Refresh toolSchemaTokens to include the subagent tool's schema.
         * `calculateInstructionTokens()` was kicked off in `fromConfig()`
         * before graphTools was populated, so its result did not count this
         * tool. Without this retrigger, token-budget/pruning logic
         * underestimates prompt overhead.
         */
        if (agentContext.tokenCounter) {
          const { tokenCounter, baseIndexTokenCountMap } = agentContext;
          agentContext.tokenCalculationPromise = agentContext
            .calculateInstructionTokens(tokenCounter)
            .then(() => {
              agentContext.updateTokenMapWithInstructions(
                baseIndexTokenCountMap
              );
            })
            .catch((err) => {
              console.error(
                'Error recalculating instruction tokens after subagent tool injection:',
                err
              );
            });
        }
      }
    }

    const agentNode = `${AGENT}${agentId}` as const;
    const toolNode = `${TOOLS}${agentId}` as const;
    const summarizeNode = `${SUMMARIZE}${agentId}` as const;
    const callModel = this.createCallModel(agentId);
    const callTools = this.initializeTools({
      currentTools: agentContext.tools,
      currentToolMap: agentContext.toolMap,
      agentContext,
    });
    const invokeWithRunStepState = async (
      state: t.AgentSubgraphState,
      config: RunnableConfig | undefined,
      invoke: () => Promise<Partial<t.AgentSubgraphState>>
    ): Promise<Partial<t.AgentSubgraphState>> => {
      this.config = config;
      this.restoreRunStepResumeState(state.runStepState);
      this.handoffRouting?.restore(state.handoffState);
      const result = await invoke();
      /** An ordinary run on a checkpointed thread inherits the last
       *  compaction's summary in state; it is not this run's output. */
      const clearsManualSummary =
        this.summarizeOnlyAgentId == null &&
        typeof state.manualSummary === 'string' &&
        state.manualSummary.length > 0;
      return {
        ...result,
        ...(clearsManualSummary ? { manualSummary: '' } : {}),
        runStepState: this.createRunStepResumeState(),
      };
    };

    const routeMessage = (
      state: t.AgentSubgraphState,
      config?: RunnableConfig
    ): string => {
      this.config = config;
      /**
       * A sealed turn that injected messages resumes in the SAME pregel run:
       * back to the agent node as a new superstep, so the model continues in
       * one assistant message instead of restarting the graph.
       */
      if (this.pendingPreemptReturn.delete(agentId)) {
        return agentNode;
      }
      if (state.summarizationRequest != null) {
        return summarizeNode;
      }
      /** A summarize-only run never calls the model, so the step after the
       *  summary has nothing to route: the summary itself is the result. */
      if (this.summarizeOnlyAgentId != null) {
        return END;
      }
      const decision = toolsCondition(
        state as t.BaseGraphState,
        toolNode,
        this.invokedToolIds
      );
      /**
       * `toolsCondition` only looks at `tool_calls` — a plain-text/reasoning
       * turn cut off by the output token ceiling has none, so it reads as an
       * ordinary finished turn and routes here to END. Flag it so hosts can
       * tell a genuinely finished answer from one the model never got to
       * complete. See `outputTruncatedIncomplete` for why this stays clear
       * of the preempt/seal halt fields.
       */
      if (decision === END) {
        const { messages } = state as t.BaseGraphState;
        const lastMessage = messages[messages.length - 1];
        if (getTruncationStopReason(lastMessage) != null) {
          this.outputTruncatedIncomplete = true;
        }
      }
      return decision;
    };

    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => {
          agentContext.invalidateProviderProjectionForMessageUpdates(b);
          return messagesStateReducer(a, b);
        },
        default: () => [],
      }),
      summarizationRequest: Annotation<t.SummarizationNodeInput | undefined>({
        reducer: (
          _: t.SummarizationNodeInput | undefined,
          b: t.SummarizationNodeInput | undefined
        ) => b,
        default: () => undefined,
      }),
      manualSummary: Annotation<string | undefined>({
        reducer: (_: string | undefined, b: string | undefined) => b,
        default: () => undefined,
      }),
      runStepState: this.createRunStepStateAnnotation(),
      handoffState: handoffStateAnnotation(),
    });

    const readChargeCredits = ():
      | WeakMap<object, Map<string, number>>
      | undefined => this.streamLimitChargeCredits;
    const readBreakerEpoch = (): number => this.breakerEpoch;
    const readActiveGenerations = (): Set<string> | undefined =>
      this.activeStreamLimitGenerations;
    const writeActiveGenerations = (value: Set<string> | undefined): void => {
      this.activeStreamLimitGenerations = value;
    };
    const writeChargeCredits = (
      value: WeakMap<object, Map<string, number>> | undefined
    ): void => {
      this.streamLimitChargeCredits = value;
    };

    const workflow = new StateGraph(StateAnnotation)
      .addNode(agentNode, (state, config) =>
        invokeWithRunStepState(state, config, () => callModel(state, config))
      )
      .addNode(toolNode, callTools)
      .addNode(
        summarizeNode,
        createSummarizeNode({
          agentContext,
          graph: {
            contentData: this.contentData,
            contentIndexMap: this.contentIndexMap,
            get config() {
              return getConfig();
            },
            runId: this.runId,
            isMultiAgent: this.isMultiAgentGraph(),
            getSubagentExecutionContext: () => this.subagentExecutionContext,
            hookRegistry: this.hookRegistry,
            getToolsForBinding: (
              provider: t.ProviderName,
              clientOptions: t.ClientOptions | undefined
            ): t.GraphTools | undefined =>
              this.getPreparedToolsForBinding(
                agentContext,
                provider,
                clientOptions
              ),
            /**
             * Live references (both maps are cleared in place, never
             * replaced), so summarization streams share the run's event
             * budget accounting under their own generation key.
             */
            streamLimits: this.streamLimits,
            streamDeltaEventCounts: this.streamDeltaEventCounts,
            streamedToolCallArgTallies: this.streamedToolCallArgTallies,
            /** Accessor pair, not a value copy: unlike the two maps above,
             * the credit map is REPLACED by graph resets rather than cleared
             * in place, and the guards' lazy `??=` must install onto the
             * graph — a copy held here would survive resets and grow one
             * attempt-stamped entry per compaction for a retained reused
             * chunk object. */
            get streamLimitChargeCredits() {
              return readChargeCredits();
            },
            set streamLimitChargeCredits(
              value: WeakMap<object, Map<string, number>> | undefined
            ) {
              writeChargeCredits(value);
            },
            /** Live epoch, so summary tallies are creation-tagged and the
             * resetValues grace sweep treats them like model-attempt
             * entries. */
            get breakerEpoch(): number {
              return readBreakerEpoch();
            },
            /** Accessor pair like the charge credits: summary attempts
             * lease their generations on the GRAPH's active set, and the
             * lazy `??=` in the lease helper must install there. */
            get activeStreamLimitGenerations(): Set<string> | undefined {
              return readActiveGenerations();
            },
            set activeStreamLimitGenerations(value: Set<string> | undefined) {
              writeActiveGenerations(value);
            },
            /** Read per attempt: a sibling branch tripping the run breaker
             * must also cancel in-flight summarization model calls. */
            getBreakerSignal: (): AbortSignal => this.breakerAbort.signal,
            /** The node captures this at entry so its own chunk handler's
             * breach trips the run that STARTED the summarization, not a
             * controller installed by a later reset. */
            getBreakerController: (): AbortController => this.breakerAbort,
            /** Captured at node entry and stamped into the summary attempt
             * metadata, so the wire consumer epoch-gates old-run summary
             * chunks exactly like model-attempt chunks. */
            getBreakerEpoch: (): number => this.breakerEpoch,
            dispatchRunStep: async (runStep, nodeConfig) => {
              const resolvedConfig = nodeConfig ?? this.config;
              if (runStep.agentId != null) {
                const groupId = this.resolveParallelGroupId(
                  runStep.agentId,
                  resolvedConfig?.metadata
                );
                if (groupId != null) {
                  runStep.groupId = groupId;
                }
              }
              runStep.status ??= 'in_progress';
              await this.trackDispatchedRunStep(
                runStep,
                resolvedConfig?.metadata,
                false
              );

              const handler = this.handlerRegistry?.getHandler(
                GraphEvents.ON_RUN_STEP
              );
              if (handler) {
                await handler.handle(
                  GraphEvents.ON_RUN_STEP,
                  runStep,
                  resolvedConfig?.configurable,
                  this
                );
                this.handlerDispatchedStepIds.add(runStep.id);
              }

              const unmarkHandlerDispatchedEvent = handler
                ? this.markHandlerDispatchedEvent(
                  GraphEvents.ON_RUN_STEP,
                  runStep.id
                )
                : undefined;
              try {
                if (resolvedConfig) {
                  await safeDispatchCustomEvent(
                    GraphEvents.ON_RUN_STEP,
                    runStep,
                    resolvedConfig
                  );
                }
              } finally {
                unmarkHandlerDispatchedEvent?.();
              }
            },
            closeRunStep: async (
              stepId: string,
              status: Exclude<t.RunStepStatus, 'in_progress'>,
              nodeConfig?: RunnableConfig
            ) => {
              await this.closeRunStep(stepId, status, {
                metadata: (nodeConfig ?? this.config)?.metadata,
              });
            },
            dispatchRunStepCompleted: async (
              stepId: string,
              result: t.StepCompleted,
              nodeConfig?: RunnableConfig
            ) => {
              const resolvedConfig = nodeConfig ?? this.config;
              const completedAt = Date.now();
              const runStep = this.getRunStep(stepId);
              const handler = this.handlerRegistry?.getHandler(
                GraphEvents.ON_RUN_STEP_COMPLETED
              );
              if (handler) {
                await handler.handle(
                  GraphEvents.ON_RUN_STEP_COMPLETED,
                  {
                    result: {
                      ...result,
                      id: stepId,
                      index: runStep?.index ?? 0,
                      completed_at: completedAt,
                    },
                  },
                  resolvedConfig?.configurable,
                  this
                );
              }
              await this.recordStepCompletion(stepId, {
                metadata: resolvedConfig?.metadata,
                at: completedAt,
              });
            },
          },
          generateStepId: (stepKey: string) => this.generateStepId(stepKey),
        })
      )
      .addEdge(START, agentNode)
      .addConditionalEdges(agentNode, routeMessage)
      .addEdge(summarizeNode, agentNode)
      .addEdge(toolNode, agentContext.toolEnd ? END : agentNode);

    return workflow.compile({ name: STANDARD_GRAPH_RUN_NAME });
  }

  createWorkflow(): t.CompiledStateWorkflow {
    this.hasCompiledCheckpointer = this.compileOptions?.checkpointer != null;
    const agentNode = this.createAgentNode(this.defaultAgentId);
    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => {
          if (!this.messages.length && !this.hasRestoredCheckpointMessages) {
            this.startIndex = a.length + b.length;
          }
          const result = messagesStateReducer(a, b);
          this.messages = result;
          return result;
        },
        default: () => [],
      }),
      /** Surfaced from the agent subgraph on a summarize-only run. */
      manualSummary: Annotation<string | undefined>({
        reducer: (_: string | undefined, b: string | undefined) => b,
        default: () => undefined,
      }),
      runStepState: this.createRunStepStateAnnotation(),
      handoffState: handoffStateAnnotation(),
    });
    const compactingAgentNode = async (
      state: t.AgentSubgraphState,
      config?: RunnableConfig
    ): Promise<Partial<t.AgentSubgraphState>> =>
      this.propagateManualCompaction(await agentNode.invoke(state, config));
    const workflow = new StateGraph(StateAnnotation)
      .addNode(
        this.defaultAgentId,
        this.summarizeOnlyAgentId != null
          ? compactingAgentNode
          : (agentNode as Runnable<
              t.AgentSubgraphState,
              Partial<t.AgentSubgraphState>
            >),
        { ends: [END] }
      )
      .addEdge(START, this.defaultAgentId);

    // LangGraph compile() types are overly strict for opt-in options
    return workflow.compile({
      ...this.compileOptions,
      name: STANDARD_GRAPH_RUN_NAME,
    } as unknown as never);
  }

  /**
   * Indicates if this is a multi-agent graph.
   * Override in MultiAgentGraph to return true.
   * Used to conditionally include agentId in RunStep for frontend rendering.
   */
  protected isMultiAgentGraph(): boolean {
    return false;
  }

  /**
   * Get the parallel group ID for an agent, if any.
   * Override in MultiAgentGraph to provide actual group IDs.
   * Group IDs are incrementing numbers (1, 2, 3...) reflecting execution order.
   * @param _agentId - The agent ID to look up
   * @returns undefined for StandardGraph (no parallel groups), or group number for MultiAgentGraph
   */
  protected getParallelGroupIdForAgent(_agentId: string): number | undefined {
    return undefined;
  }

  protected resolveParallelGroupId(
    agentId: string,
    metadata?: Record<string, unknown>
  ): number | undefined {
    if (
      metadata == null ||
      !Object.prototype.hasOwnProperty.call(
        metadata,
        Constants.HANDOFF_GROUP_ID
      )
    ) {
      return this.getParallelGroupIdForAgent(agentId);
    }
    const runtimeGroupId = metadata[Constants.HANDOFF_GROUP_ID];
    if (runtimeGroupId === null) {
      return undefined;
    }
    if (
      typeof runtimeGroupId === 'number' &&
      Number.isSafeInteger(runtimeGroupId) &&
      runtimeGroupId > 0
    ) {
      return runtimeGroupId;
    }
    return this.getParallelGroupIdForAgent(agentId);
  }

  /* Dispatchers */

  /**
   * Dispatches a run step to the client, returns the step ID
   */
  async dispatchRunStep(
    stepKey: string,
    stepDetails: t.StepDetails,
    metadata?: Record<string, unknown>
  ): Promise<string> {
    if (!this.config) {
      throw new Error('No config provided');
    }

    if (stepDetails.type === StepTypes.TOOL_CALLS && stepDetails.tool_calls) {
      let replayStepId: string | undefined;
      let reusesOpenStep = stepDetails.tool_calls.length > 0;
      for (const toolCall of stepDetails.tool_calls) {
        const toolCallId = toolCall.id ?? '';
        const mappedStepId = this.toolCallStepIds.get(toolCallId);
        if (!toolCallId || mappedStepId == null) {
          reusesOpenStep = false;
          continue;
        }
        replayStepId ??= mappedStepId;
        if (mappedStepId !== replayStepId) {
          reusesOpenStep = false;
        }
      }
      const replayStep =
        replayStepId == null ? undefined : this.getRunStep(replayStepId);
      if (
        reusesOpenStep &&
        replayStep != null &&
        (replayStep.status == null || replayStep.status === 'in_progress')
      ) {
        return replayStep.id;
      }
    }

    const [stepId, stepIndex] = this.generateStepId(stepKey);
    if (stepDetails.type === StepTypes.TOOL_CALLS && stepDetails.tool_calls) {
      let pendingToolCalls: Set<string> | undefined;
      for (const tool_call of stepDetails.tool_calls) {
        const toolCallId = tool_call.id ?? '';
        if (!toolCallId || this.toolCallStepIds.has(toolCallId)) {
          continue;
        }
        this.toolCallStepIds.set(toolCallId, stepId);
        pendingToolCalls ??= this.getPendingToolCallSet(stepId);
        pendingToolCalls.add(toolCallId);
      }
    }

    const runStep: t.RunStep = {
      stepIndex,
      id: stepId,
      type: stepDetails.type,
      index: this.contentData.length,
      stepDetails,
      usage: null,
      status: 'in_progress',
    };

    const runId = this.runId ?? '';
    if (runId) {
      runStep.runId = runId;
    }

    /**
     * `agentId`/`groupId` are multi-agent-only, and `getAgentContext` signals a
     * miss by throwing — so for a single-agent graph the lookup could only ever
     * build and discard an Error while producing the same undefined fields.
     * The constant-time check gates it out of the per-step dispatch path.
     */
    if (metadata && this.isMultiAgentGraph()) {
      try {
        const agentContext = this.getAgentContext(metadata);
        if (agentContext.agentId) {
          runStep.agentId = agentContext.agentId;
          const groupId = this.resolveParallelGroupId(
            agentContext.agentId,
            metadata
          );
          if (groupId != null) {
            runStep.groupId = groupId;
          }
        }
      } catch (_e) {
        /** If we can't get agent context, that's okay - agentId remains undefined */
      }
    }

    await this.trackDispatchedRunStep(runStep, metadata);

    // Primary dispatch: handler registry (reliable, always works).
    // This mirrors how handleToolCallCompleted dispatches ON_RUN_STEP_COMPLETED
    // via the handler registry, ensuring the event always reaches the handler
    // even when LangGraph's callback system drops the custom event.
    const handler = this.handlerRegistry?.getHandler(GraphEvents.ON_RUN_STEP);
    if (handler) {
      await handler.handle(GraphEvents.ON_RUN_STEP, runStep, metadata, this);
      this.handlerDispatchedStepIds.add(stepId);
    }

    // Secondary dispatch: custom event for LangGraph callback chain
    // (tracing, Langfuse, external consumers).  May be silently dropped
    // in some scenarios (stale run ID, subgraph callback propagation issues),
    // but the primary dispatch above guarantees the event reaches the handler.
    // The customEventCallback in run.ts skips events already dispatched above
    // to prevent double handling.
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_RUN_STEP, stepId)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP,
        runStep,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
    return stepId;
  }

  /**
   * Static version of handleToolCallError to avoid creating strong references
   * that prevent garbage collection.
   *
   * Returns whether the error completion event was actually dispatched. A
   * tool can error before this graph instance has a run step for the call —
   * on a resume pass the interrupted batch re-executes IMMEDIATELY on graph
   * re-entry, before any step replay has registered `toolCallStepIds` (a
   * fast-failing tool, e.g. a schema-validation reject, loses that race).
   * That is a caller-recoverable condition, not an invariant violation: the
   * ToolNode falls back to its normal completion dispatch for the error
   * ToolMessage when this returns `false`, so throwing here would only
   * replace a recoverable miss with a lost completion event and a scary log.
   */
  static async handleToolCallErrorStatic(
    graph: StandardGraph,
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): Promise<boolean> {
    if (!data.id) {
      console.warn('No Tool ID provided for Tool Error');
      return false;
    }

    const stepId = graph.toolCallStepIds.get(data.id) ?? '';
    if (!stepId) {
      return false;
    }

    const { name, input: args, error } = data;
    const eventValueLimit = calculateMaxToolResultChars();
    const errorOutputPrefix = 'Error processing tool';
    const errorDetail =
      error?.message != null
        ? `: ${serializeToolContentBounded(
          error.message,
          Math.max(0, eventValueLimit - errorOutputPrefix.length - 2)
        )}`
        : '';

    const runStep = graph.getRunStep(stepId);
    if (!runStep) {
      return false;
    }

    const completedAt = Date.now();
    const tool_call: t.ProcessedToolCall = {
      id: data.id,
      name: name || '',
      args: serializeToolContentBounded(args, eventValueLimit),
      output: `${errorOutputPrefix}${errorDetail}`,
      progress: 1,
    };

    // No registered ON_RUN_STEP_COMPLETED handler ⇒ nothing was dispatched.
    // Report `false` so the ToolNode runs its own fallback dispatch; returning
    // `true` here would silently drop the error completion for hosts that wire
    // completions through callback-based custom events instead of a handler.
    const handler = graph.handlerRegistry?.getHandler(
      GraphEvents.ON_RUN_STEP_COMPLETED
    );
    if (!handler) {
      return false;
    }

    await handler.handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: stepId,
          index: runStep.index,
          type: 'tool_call',
          tool_call,
          completed_at: completedAt,
        } as t.ToolCompleteEvent,
      },
      metadata,
      graph
    );
    await graph.recordStepCompletion(stepId, {
      toolCallId: data.id,
      metadata,
      at: completedAt,
    });
    return true;
  }

  /**
   * Instance method that delegates to the static method
   * Kept for backward compatibility
   */
  async handleToolCallError(
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): Promise<boolean> {
    return StandardGraph.handleToolCallErrorStatic(this, data, metadata);
  }

  async dispatchRunStepDelta(
    id: string,
    delta: t.ToolCallDelta,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    } else if (!id) {
      throw new Error('No step ID found');
    }
    const runStepDelta: t.RunStepDeltaEvent = {
      id,
      delta,
    };
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_RUN_STEP_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_RUN_STEP_DELTA,
        runStepDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(id);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_RUN_STEP_DELTA, id)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_DELTA,
        runStepDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  }

  async dispatchMessageDelta(
    id: string,
    delta: t.MessageDelta,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const messageDelta: t.MessageDeltaEvent = {
      id,
      delta,
    };
    if (hasTextDeltaContent(delta.content)) {
      this.messageStepHasTextDeltas.add(id);
    }
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_MESSAGE_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_MESSAGE_DELTA,
        messageDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(id);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_MESSAGE_DELTA, id)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_MESSAGE_DELTA,
        messageDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  }

  dispatchReasoningDelta = async (
    stepId: string,
    delta: t.ReasoningDelta,
    metadata?: Record<string, unknown>
  ): Promise<void> => {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const reasoningDelta: t.ReasoningDeltaEvent = {
      id: stepId,
      delta,
    };
    if (hasReasoningDeltaContent(delta.content)) {
      this.reasoningStepHasDeltas.add(stepId);
    }
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_REASONING_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_REASONING_DELTA,
        reasoningDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(stepId);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_REASONING_DELTA, stepId)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_REASONING_DELTA,
        reasoningDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  };
}

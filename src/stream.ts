// src/stream.ts
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { ChatOpenAIReasoningSummary } from '@langchain/openai';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { AgentContext } from '@/agents/AgentContext';
import type { RunBreakerScope } from '@/llm/streamLimits';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  claimStreamLimitCharge,
  combineCompleteToolCalls,
  enforceCompleteToolCallArgLimit,
  enforceStreamedToolCallArgLimit,
  enforceStreamDeltaEventLimit,
  requiresStreamLimitAccounting,
  StreamLimitExceededError,
  STREAM_LIMIT_EPOCH_KEY,
  resolveGenerationKey,
} from '@/llm/streamLimits';
import {
  getStreamedToolCallSeal,
  getStreamedToolCallAdapter,
  streamedToolCallAdapterAllowsSequentialSeal,
  type StreamedToolCallSeal,
} from '@/tools/streamedToolCallSeals';
import {
  ToolCallTypes,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
  Constants,
  CODE_EXECUTION_TOOLS,
  LOCAL_CODING_BUNDLE_NAMES,
} from '@/common';
import {
  getMessageCreationContentMetadata,
  splitAssistantTextContentByPhase,
} from '@/messages/assistantPhase';
import {
  buildToolExecutionRequestPlan,
  coerceRecordArgs,
  normalizeError,
} from '@/tools/eagerEventExecution';
import {
  serializeStructuredValueBounded,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  handleServerToolResult,
  handleToolCallChunks,
  handleToolCalls,
} from '@/tools/handlers';
import {
  calculateMaxToolResultChars,
  truncateToolResultContent,
} from '@/utils/truncation';
import { resolveToolOutcome, outcomeFieldsFromResult } from '@/tools/intentArg';
import { TOOL_OUTPUT_REF_PATTERN } from '@/tools/toolOutputReferences';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { isReasoningContentBlock } from '@/messages/core';
import { safeDispatchCustomEvent } from '@/utils/events';
import { composeAbortSignals } from '@/utils/misc';
import { isGoogleLike } from '@/utils/llm';
import { getMessageId } from '@/messages';

const LOCAL_CODING_BUNDLE_NAME_SET: ReadonlySet<string> = new Set(
  LOCAL_CODING_BUNDLE_NAMES
);

type ReasoningSummaryLike = {
  summary?: Array<{ text?: string }>;
};

/**
 * Parses content to extract thinking sections enclosed in <think> tags using string operations
 * @param content The content to parse
 * @returns An object with separated text and thinking content
 */
function parseThinkingContent(content: string): {
  text: string;
  thinking: string;
} {
  // If no think tags, return the original content as text
  if (!content.includes('<think>')) {
    return { text: content, thinking: '' };
  }

  let textResult = '';
  const thinkingResult: string[] = [];
  let position = 0;

  while (position < content.length) {
    const thinkStart = content.indexOf('<think>', position);

    if (thinkStart === -1) {
      // No more think tags, add the rest and break
      textResult += content.slice(position);
      break;
    }

    // Add text before the think tag
    textResult += content.slice(position, thinkStart);

    const thinkEnd = content.indexOf('</think>', thinkStart);
    if (thinkEnd === -1) {
      // Malformed input, no closing tag
      textResult += content.slice(thinkStart);
      break;
    }

    // Add the thinking content
    const thinkContent = content.slice(thinkStart + 7, thinkEnd);
    thinkingResult.push(thinkContent);

    // Move position to after the think tag
    position = thinkEnd + 8; // 8 is the length of '</think>'
  }

  return {
    text: textResult.trim(),
    thinking: thinkingResult.join('\n').trim(),
  };
}

function getNonEmptyValue(possibleValues: string[]): string | undefined {
  for (const value of possibleValues) {
    if (value && value.trim() !== '') {
      return value;
    }
  }
  return undefined;
}

function isBatchSensitiveToolExecution(
  graph: StandardGraph,
  metadata?: Record<string, unknown>
): boolean {
  /**
   * Resolve the hook-session id exactly the way ToolNode will: its hook
   * lookups read `config.configurable.run_id` ONLY, so that source wins here
   * too (a subagent child graph carries its own `graph.runId`, but its
   * ToolNode executes hooks under the PARENT's inherited configurable run
   * id). The metadata / graph.runId fallbacks only apply when no
   * configurable id exists and can only be conservative — a false positive
   * merely skips eager prestart.
   */
  const runId =
    (graph.config?.configurable?.run_id as string | undefined) ??
    (metadata?.run_id as string | undefined) ??
    graph.runId;
  return (
    graph.hookRegistry?.hasResultAlteringHooks(runId) === true ||
    graph.humanInTheLoop?.enabled === true
  );
}

function hasToolOutputReference(value: unknown): boolean {
  if (typeof value === 'string') {
    return TOOL_OUTPUT_REF_PATTERN.test(value);
  }
  if (Array.isArray(value)) {
    return value.some((item) => hasToolOutputReference(item));
  }
  if (value !== null && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).some((item) =>
      hasToolOutputReference(item)
    );
  }
  return false;
}

function isEagerExecutionExcludedTool(
  name: string,
  graph: StandardGraph,
  agentContext?: AgentContext
): boolean {
  if (name === '') {
    return false;
  }
  const excluded = graph.eagerEventToolExecution?.excludeToolNames;
  if (excluded != null && excluded.includes(name)) {
    return true;
  }
  // Run-scoped circuit breaker: once a prestart for this tool diverged from
  // the final request ("changed after eager execution started"), stop
  // prestarting it so the model's retry executes normally instead of
  // re-diverging in a loop (LibreChat#14371).
  if (
    (graph.eagerEventToolSuppressions as Set<string> | undefined)?.has(name) ===
    true
  ) {
    return true;
  }
  // A code-session participant writes to the shared sandbox, so it is
  // side-effecting: never prestart it speculatively (a revised/superseded turn
  // would leave the write applied). Implies exclusion without the host having
  // to also list the name in excludeToolNames.
  if (graph.codeSessionToolNames?.includes(name) === true) {
    return true;
  }
  // With stateful sessions on, execute_code/bash run against a DURABLE warm
  // runtime. Speculative prestart there is unsafe: if the model revises the
  // args, ToolNode discards the eager result but the mutation has already
  // landed in the session workspace, corrupting later runs. Stateless mode
  // uses a throwaway VM per call, so eager prestart stays safe there.
  if (!CODE_EXECUTION_TOOLS.has(name)) {
    return false;
  }
  if (graph.toolExecution?.sandbox?.statefulSessions === true) {
    return true;
  }
  // A non-default code-session partition is the trusted per-agent signal that
  // this call must remain isolated from the graph-wide stateless session. The
  // actual tool factory may route it to a durable backend, which the stream
  // layer cannot inspect in event-driven mode. Conservatively avoid speculative
  // execution for these calls so discarded eager results cannot mutate that
  // agent's workspace.
  return (
    agentContext?.codeSessionKey != null &&
    agentContext.codeSessionKey !== Constants.EXECUTE_CODE
  );
}

function isDirectGraphTool(
  name: string,
  agentContext: AgentContext | undefined
): boolean {
  if (name.startsWith(Constants.LC_TRANSFER_TO_)) {
    return true;
  }
  return (
    (agentContext?.graphTools as t.GenericTool[] | undefined)?.some(
      (tool) => 'name' in tool && tool.name === name
    ) === true
  );
}

function isDirectLocalTool(name: string, graph: StandardGraph): boolean {
  const toolExecution = graph.toolExecution;
  const engine = toolExecution?.engine;
  if (
    toolExecution == null ||
    (engine !== 'local' && engine !== 'cloudflare-sandbox')
  ) {
    return false;
  }
  const includeCodingTools =
    engine === 'cloudflare-sandbox'
      ? toolExecution.cloudflare?.includeCodingTools
      : toolExecution.local?.includeCodingTools;
  if (includeCodingTools === false) {
    return CODE_EXECUTION_TOOLS.has(name);
  }
  return LOCAL_CODING_BUNDLE_NAME_SET.has(name);
}

function toCodeEnvFile(file: t.FileRef, execSessionId: string): t.CodeEnvFile {
  const base = {
    id: file.id,
    resource_id: file.resource_id ?? file.id,
    name: file.name,
    storage_session_id: file.storage_session_id ?? execSessionId,
  };
  const kind = file.kind ?? 'user';
  if (kind === 'skill' && file.version != null) {
    return { ...base, kind: 'skill', version: file.version };
  }
  if (kind === 'agent') {
    return { ...base, kind: 'agent' };
  }
  return { ...base, kind: 'user' };
}

function getCodeSessionContext(
  graph: StandardGraph,
  name: string,
  agentContext?: AgentContext
): t.ToolCallRequest['codeSessionContext'] | undefined {
  if (
    !CODE_EXECUTION_TOOLS.has(name) &&
    name !== Constants.SKILL_TOOL &&
    name !== Constants.READ_FILE &&
    graph.codeSessionToolNames?.includes(name) !== true
  ) {
    return undefined;
  }

  const codeSession = graph.sessions.get(
    agentContext?.codeSessionKey ?? Constants.EXECUTE_CODE
  ) as t.CodeSessionContext | undefined;
  if (codeSession?.session_id == null || codeSession.session_id === '') {
    return undefined;
  }

  return {
    session_id: codeSession.session_id,
    files: codeSession.files?.map((file) =>
      toCodeEnvFile(file, codeSession.session_id)
    ),
  };
}

function isEagerToolExecutionEnabledForBatch(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
}): boolean {
  const { graph, metadata, agentContext } = args;
  if (graph.eagerEventToolExecution?.enabled !== true) {
    return false;
  }
  if ((agentContext?.toolDefinitions?.length ?? 0) === 0) {
    return false;
  }
  if (isBatchSensitiveToolExecution(graph, metadata)) {
    return false;
  }
  if (
    metadata?.[Constants.PROGRAMMATIC_TOOL_CALLING] === true ||
    metadata?.[Constants.BASH_PROGRAMMATIC_TOOL_CALLING] === true
  ) {
    return false;
  }
  if (
    graph.handlerRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) == null &&
    graph.eventToolExecutionAvailable !== true
  ) {
    return false;
  }
  return true;
}

function hasFinalToolCallSignal(chunk: Partial<AIMessageChunk>): boolean {
  const metadata = chunk.response_metadata as
    | Record<string, unknown>
    | undefined;
  const finishReason =
    metadata?.finish_reason ??
    metadata?.finishReason ??
    metadata?.stop_reason ??
    metadata?.stopReason;
  return finishReason === 'tool_calls' || finishReason === 'tool_use';
}

function canPrestartSequentialStreamedToolChunks(
  agentContext: AgentContext | undefined
): boolean {
  // Anthropic seals each prior streamed tool-use block when the next indexed
  // tool-use block begins. Live Kimi/Moonshot streams can still revise prior
  // args after advancing to the next index, so keep those on the final
  // tool-call path unless they grow an explicit adapter seal.
  return agentContext?.provider === Providers.ANTHROPIC;
}

function hasExplicitStreamedToolCallSeals(
  chunk: Partial<AIMessageChunk>
): boolean {
  return (
    getStreamedToolCallAdapter(
      chunk.response_metadata as Record<string, unknown> | undefined
    ) != null
  );
}

/**
 * True when a provider adapter marked every tool call on this chunk as
 * complete on arrival (seal kind `all`), e.g. Google GenAI / Vertex AI, whose
 * protocol delivers function calls as whole objects rather than arg deltas.
 */
function hasOnArrivalToolCallSeal(chunk: Partial<AIMessageChunk>): boolean {
  const metadata = chunk.response_metadata as
    | Record<string, unknown>
    | undefined;
  return (
    getStreamedToolCallAdapter(metadata) != null &&
    getStreamedToolCallSeal(metadata)?.kind === 'all'
  );
}

function hasDirectToolCallInBatch(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
}): boolean {
  const { graph, agentContext, toolCalls } = args;
  return toolCalls.some(
    (toolCall) =>
      toolCall.name !== '' &&
      (isDirectGraphTool(toolCall.name, agentContext) ||
        isDirectLocalTool(toolCall.name, graph))
  );
}

function hasPotentialDirectToolInStreamContext(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
}): boolean {
  const { graph, agentContext } = args;
  const engine = graph.toolExecution?.engine;
  if (engine === 'local' || engine === 'cloudflare-sandbox') {
    return true;
  }
  if ((agentContext?.graphTools?.length ?? 0) > 0) {
    return true;
  }
  return false;
}

function hasDirectToolCallChunkInBatch(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  toolCallChunks?: ToolCallChunk[];
}): boolean {
  const { graph, agentContext, toolCallChunks } = args;
  return (
    toolCallChunks?.some(
      (toolCallChunk) =>
        toolCallChunk.name != null &&
        toolCallChunk.name !== '' &&
        (isDirectGraphTool(toolCallChunk.name, agentContext) ||
          isDirectLocalTool(toolCallChunk.name, graph))
    ) === true
  );
}

function hasDirectToolCallChunkStateInStep(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  stepKey: string;
}): boolean {
  const { graph, agentContext, stepKey } = args;
  const prefix = `${stepKey}\u0000`;
  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    const name = state.name;
    if (
      name != null &&
      name !== '' &&
      (isDirectGraphTool(name, agentContext) || isDirectLocalTool(name, graph))
    ) {
      return true;
    }
  }
  return false;
}

function isGoogleServerSideToolContentPart(
  contentPart: t.MessageContentComplex
): boolean {
  return contentPart.type === 'toolCall' || contentPart.type === 'toolResponse';
}

function isNativeMediaContentPart(
  contentPart: t.MessageContentComplex
): boolean {
  return (
    contentPart.type === ContentTypes.IMAGE_FILE ||
    contentPart.native_media != null
  );
}

function isTextContentPart(contentPart: t.MessageContentComplex): boolean {
  return contentPart.type?.startsWith(ContentTypes.TEXT) ?? false;
}

function getReasoningTextFromContentPart(
  contentPart: t.MessageContentComplex
): string {
  return (
    (contentPart as t.ThinkingContentText).thinking ??
    (contentPart as Partial<t.GoogleReasoningContentText>).reasoning ??
    (contentPart as Partial<t.BedrockReasoningContentText>).reasoningText
      ?.text ??
    ''
  );
}

function getReasoningTextFromChunk(
  chunk: Partial<AIMessageChunk>,
  agentContext: AgentContext
): string {
  const reasoning = chunk.additional_kwargs?.[agentContext.reasoningKey] as
    | string
    | Partial<ChatOpenAIReasoningSummary>
    | undefined;
  if (typeof reasoning === 'string') {
    return reasoning;
  }
  return reasoning?.summary?.[0]?.text ?? '';
}

const googleServerSideToolStepIdsByGraph = new WeakMap<
  StandardGraph,
  Set<string>
>();

function markGoogleServerSideToolMessageStep(
  graph: StandardGraph,
  stepId: string
): void {
  const stepIds = googleServerSideToolStepIdsByGraph.get(graph) ?? new Set();
  stepIds.add(stepId);
  googleServerSideToolStepIdsByGraph.set(graph, stepIds);
}

function isGoogleServerSideToolMessageStep(
  graph: StandardGraph,
  stepId: string
): boolean {
  return googleServerSideToolStepIdsByGraph.get(graph)?.has(stepId) === true;
}

function shouldStartFreshMessageStepAfterGoogleServerSideTool({
  graph,
  stepId,
  runStep,
  content,
}: {
  graph: StandardGraph;
  stepId: string;
  runStep?: t.RunStep;
  content: string | t.MessageContentComplex[];
}): boolean {
  if (
    runStep?.type !== StepTypes.MESSAGE_CREATION ||
    !isGoogleServerSideToolMessageStep(graph, stepId)
  ) {
    return false;
  }
  if (typeof content === 'string') {
    return true;
  }
  return (
    content.every((c) => isTextContentPart(c)) ||
    content.every((c) => isReasoningContentBlock(c))
  );
}

async function dispatchMessageCreationStep({
  graph,
  stepKey,
  content,
  contentType,
  metadata,
}: {
  graph: StandardGraph;
  stepKey: string;
  content?: string | t.MessageContentComplex[];
  contentType?: ContentTypes.TEXT | ContentTypes.THINK;
  metadata?: Record<string, unknown>;
}): Promise<string> {
  const messageId = getMessageId(stepKey, graph, true) ?? '';
  return graph.dispatchRunStep(
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
}

async function dispatchMessageContentParts({
  graph,
  stepKey,
  content,
  metadata,
}: {
  graph: StandardGraph;
  stepKey: string;
  content: t.MessageContentComplex[];
  metadata?: Record<string, unknown>;
}): Promise<void> {
  for (const contentPart of content) {
    const currentStepId = await dispatchMessageCreationStep({
      graph,
      stepKey,
      content: [contentPart],
      metadata,
    });
    if (
      isGoogleServerSideToolContentPart(contentPart) ||
      isNativeMediaContentPart(contentPart)
    ) {
      markGoogleServerSideToolMessageStep(graph, currentStepId);
    }
    await graph.dispatchMessageDelta(
      currentStepId,
      {
        content: [contentPart],
      },
      metadata
    );
  }
}

async function dispatchReasoningContentParts({
  graph,
  stepKey,
  content,
  metadata,
}: {
  graph: StandardGraph;
  stepKey: string;
  content: t.MessageContentComplex[];
  metadata?: Record<string, unknown>;
}): Promise<void> {
  if (content.length === 0) {
    return;
  }
  const currentStepId = await dispatchMessageCreationStep({
    graph,
    stepKey,
    content,
    contentType: ContentTypes.THINK,
    metadata,
  });
  await graph.dispatchReasoningDelta(
    currentStepId,
    {
      content,
    },
    metadata
  );
}

async function dispatchGoogleServerSideToolStreamContent({
  graph,
  stepKey,
  chunk,
  agentContext,
  content,
  metadata,
}: {
  graph: StandardGraph;
  stepKey: string;
  chunk: Partial<AIMessageChunk>;
  agentContext: AgentContext;
  content: t.MessageContentComplex[];
  metadata?: Record<string, unknown>;
}): Promise<void> {
  const reasoningContent: t.MessageContentComplex[] = [];
  const reasoningText = getReasoningTextFromChunk(chunk, agentContext);
  if (reasoningText !== '') {
    reasoningContent.push({
      type: ContentTypes.THINK,
      think: reasoningText,
    });
  }
  reasoningContent.push(
    ...content
      .filter((contentPart) => isReasoningContentBlock(contentPart))
      .map((contentPart) => ({
        type: ContentTypes.THINK,
        think: getReasoningTextFromContentPart(contentPart),
      }))
      .filter((contentPart) => contentPart.think !== '')
  );
  await dispatchReasoningContentParts({
    graph,
    stepKey,
    content: reasoningContent,
    metadata,
  });

  const messageContent = content.filter(
    (contentPart) =>
      isTextContentPart(contentPart) ||
      isGoogleServerSideToolContentPart(contentPart) ||
      isNativeMediaContentPart(contentPart)
  );
  await dispatchMessageContentParts({
    graph,
    stepKey,
    content: messageContent,
    metadata,
  });
}

type EagerToolExecutionEntry = {
  id: string;
  toolName: string;
  coercedArgs: Record<string, unknown>;
  request: t.ToolCallRequest;
};

function createEagerToolExecutionPlan(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
  skipExisting?: boolean;
}): EagerToolExecutionEntry[] | undefined {
  const {
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting = false,
  } = args;
  if (
    !isEagerToolExecutionEnabledForBatch({
      graph,
      metadata,
      agentContext,
    })
  ) {
    return undefined;
  }

  if (hasDirectToolCallInBatch({ graph, agentContext, toolCalls })) {
    return undefined;
  }
  if (
    graph.toolOutputReferences?.enabled === true &&
    toolCalls.some((toolCall) => hasToolOutputReference(toolCall.args))
  ) {
    return undefined;
  }

  const unstartedToolCalls = skipExisting
    ? toolCalls.filter((toolCall) => {
      if (toolCall.id == null || toolCall.id === '') {
        return true;
      }
      return !graph.eagerEventToolExecutions.has(toolCall.id);
    })
    : toolCalls;
  // Drop host-excluded tools only AFTER the batch-level guards above have run
  // against the full batch, so excluding a call never hides a sibling direct
  // tool from `hasDirectToolCallInBatch`. Excluded calls fall through to normal
  // ToolNode execution; siblings may still eager-execute.
  const candidateToolCalls = unstartedToolCalls.filter(
    (toolCall) =>
      !isEagerExecutionExcludedTool(toolCall.name, graph, agentContext)
  );
  if (candidateToolCalls.length === 0) {
    return [];
  }
  const toolSchemas = new Map(
    agentContext?.toolDefinitions?.map(({ name, parameters }) => [
      name,
      parameters,
    ])
  );

  // Eager execution must preserve ToolNode batch semantics exactly for every
  // unstarted call. If any candidate cannot be planned, fall back for that
  // candidate set.
  if (
    candidateToolCalls.some(
      (toolCall) =>
        toolCall.id == null ||
        toolCall.id === '' ||
        toolCall.name === '' ||
        (!skipExisting && graph.eagerEventToolExecutions.has(toolCall.id))
    )
  ) {
    return undefined;
  }

  /* No runtimeSessionHint here on purpose: the eager path is speculative, and
   * code-session tools (the only ones that carry a hint) are excluded from
   * eager prestart entirely when stateful sessions are on — see
   * isEagerExecutionExcludedTool. The durable runtime is only ever touched by
   * the committed ToolNode path. */
  const plan = buildToolExecutionRequestPlan({
    toolCalls: candidateToolCalls.map((toolCall) => ({
      id: toolCall.id,
      name: toolCall.name,
      args: toolCall.args,
      stepId: graph.toolCallStepIds.get(toolCall.id!) ?? '',
      codeSessionContext: getCodeSessionContext(
        graph,
        toolCall.name,
        agentContext
      ),
    })),
    usageCount: graph.getEagerEventToolUsageCount(agentContext?.agentId),
    getToolSchema: (toolName) => toolSchemas.get(toolName),
  });
  if (plan == null) {
    return undefined;
  }

  return plan.requests.map(
    (request): EagerToolExecutionEntry => ({
      id: request.id,
      toolName: request.name,
      coercedArgs: request.args,
      request,
    })
  );
}

function startEagerToolExecutions(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
  skipExisting?: boolean;
}): void {
  const { graph, metadata, agentContext, toolCalls, skipExisting } = args;
  const entries = createEagerToolExecutionPlan({
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting,
  });
  if (entries == null || entries.length === 0) {
    return;
  }

  const codeSessionBaselineById = new Map(
    entries.map((entry) => [
      entry.id,
      new Map(
        (entry.request.codeSessionContext?.files ?? []).map((file) => [
          file.name,
          `${file.storage_session_id}\0${file.id}`,
        ])
      ),
    ])
  );
  for (const entry of entries) {
    entry.request.codeSessionBaselineId =
      entry.request.codeSessionContext?.session_id;
  }
  const records: t.EagerEventToolExecution[] = [];
  const promise: Promise<t.EagerEventToolExecutionOutcome> = new Promise<
    t.ToolExecuteResult[]
  >((resolve, reject) => {
    let dispatchSettled = false;
    let resultSettled = false;
    let settledResults: t.ToolExecuteResult[] | undefined;
    const maybeResolve = (): void => {
      if (dispatchSettled && resultSettled) {
        resolve(settledResults ?? []);
      }
    };
    const batchRequest: t.ToolExecuteBatchRequest = {
      executionContext: graph.subagentExecutionContext,
      toolCalls: entries.map((entry) => entry.request),
      userId: graph.config?.configurable?.user_id as string | undefined,
      agentId: agentContext?.agentId,
      callerCapabilityProjection: (
        agentContext as
          | Partial<Pick<AgentContext, 'getCallerCapabilityProjectionSnapshot'>>
          | undefined
      )?.getCallerCapabilityProjectionSnapshot?.(),
      configurable: {
        ...graph.config?.configurable,
        executionContext: graph.subagentExecutionContext,
      },
      metadata: {
        ...metadata,
        executionContext: graph.subagentExecutionContext,
      },
      signal: composeAbortSignals(
        graph.config?.signal,
        graph.breakerAbort.signal
      ),
      resolve: (results): void => {
        resultSettled = true;
        settledResults = results;
        maybeResolve();
      },
      reject,
    };

    void safeDispatchCustomEvent(
      GraphEvents.ON_TOOL_EXECUTE,
      batchRequest,
      graph.config
    )
      .then(() => {
        dispatchSettled = true;
        maybeResolve();
      })
      .catch(reject);
  }).then(
    async (results): Promise<t.EagerEventToolExecutionOutcome> => {
      await dispatchEagerToolCompletions({
        graph,
        agentContext,
        records,
        results,
      });
      return { results };
    },
    (error): t.EagerEventToolExecutionOutcome => ({
      error: normalizeError(error),
    })
  );

  for (const entry of entries) {
    const record: t.EagerEventToolExecution = {
      toolCallId: entry.id,
      toolName: entry.toolName,
      args: entry.coercedArgs,
      request: entry.request,
      codeSessionBaselineByName: codeSessionBaselineById.get(entry.id),
      promise,
    };
    records.push(record);
    graph.eagerEventToolExecutions.set(entry.id, record);
  }
}

async function dispatchEagerToolCompletions(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  records: t.EagerEventToolExecution[];
  results: t.ToolExecuteResult[];
}): Promise<void> {
  const { graph, agentContext, records, results } = args;
  const recordById = new Map(
    records.map((record) => [record.toolCallId, record])
  );
  const maxToolResultChars =
    agentContext?.maxToolResultChars ??
    calculateMaxToolResultChars(agentContext?.maxContextTokens);

  for (const result of results) {
    const record = recordById.get(result.toolCallId);
    if (record == null) {
      continue;
    }
    if (graph.eagerEventToolExecutions.get(result.toolCallId) !== record) {
      continue;
    }
    const stepId =
      record.request.stepId ??
      graph.toolCallStepIds.get(result.toolCallId) ??
      '';
    if (stepId === '') {
      continue;
    }
    let output: string;
    if (result.status === 'error') {
      output = truncateToolResultContent(
        `Error: ${result.errorMessage ?? 'Unknown error'}\n Please fix your mistakes.`,
        maxToolResultChars
      );
    } else if (typeof result.content === 'string') {
      output = truncateToolResultContent(result.content, maxToolResultChars);
    } else {
      output = serializeStructuredValueBounded(
        result.content,
        maxToolResultChars
      ).content;
    }
    const outcome = resolveToolOutcome(
      record.request.args,
      outcomeFieldsFromResult(result),
      { isError: result.status === 'error' }
    );

    try {
      const dispatched = await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_COMPLETED,
        {
          result: {
            id: stepId,
            index: record.request.turn ?? 0,
            type: 'tool_call' as const,
            eager: true,
            tool_call: {
              args: serializeToolContentBounded(
                record.request.args,
                maxToolResultChars
              ),
              name: record.toolName,
              id: result.toolCallId,
              output,
              progress: 1,
              ...(outcome != null && { outcome }),
            } as t.ProcessedToolCall,
            completed_at: Date.now(),
          },
        },
        graph.config
      );
      if (dispatched === false) {
        continue;
      }
      record.completionDispatched = true;
    } catch (error) {
      // Let ToolNode dispatch the completion through the normal path later.

      console.warn(
        `[stream] eager completion dispatch failed for toolCallId=${result.toolCallId}:`,
        error instanceof Error ? error.message : error
      );
    }
  }
}

function getEagerToolChunkKey(
  stepKey: string,
  toolCallChunk: ToolCallChunk
): string | undefined {
  let chunkKey: string | undefined;
  if (typeof toolCallChunk.index === 'number') {
    chunkKey = String(toolCallChunk.index);
  } else if (toolCallChunk.id != null && toolCallChunk.id !== '') {
    chunkKey = toolCallChunk.id;
  }
  if (chunkKey == null) {
    return undefined;
  }
  return `${stepKey}\u0000${chunkKey}`;
}

function getEagerToolChunkIndex(
  toolCallChunk: ToolCallChunk
): number | undefined {
  return typeof toolCallChunk.index === 'number'
    ? toolCallChunk.index
    : undefined;
}

function pruneEagerToolCallChunkStates(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallIds?: ReadonlySet<string>;
  clearStep?: boolean;
}): void {
  const { graph, stepKey, toolCallIds, clearStep = false } = args;
  const prefix = `${stepKey}\u0000`;
  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    if (
      clearStep ||
      (state.id != null && toolCallIds?.has(state.id) === true)
    ) {
      graph.eagerEventToolCallChunks.delete(key);
    }
  }
}

function mergeToolCallArgsText(existing: string, incoming: string): string {
  if (incoming === '') {
    return existing;
  }
  if (existing === '') {
    return incoming;
  }
  if (incoming === existing) {
    try {
      JSON.parse(incoming);
      return incoming;
    } catch {
      return `${existing}${incoming}`;
    }
  }
  if (incoming.startsWith(existing)) {
    return incoming;
  }
  if (existing.startsWith(incoming)) {
    return existing;
  }
  try {
    JSON.parse(existing);
    JSON.parse(incoming);
    return incoming;
  } catch {
    // Fall through to delta concatenation.
  }
  for (
    let overlap = Math.min(existing.length, incoming.length);
    overlap >= 8;
    overlap -= 1
  ) {
    if (existing.endsWith(incoming.slice(0, overlap))) {
      return `${existing}${incoming.slice(overlap)}`;
    }
  }
  return `${existing}${incoming}`;
}

function recordEagerToolCallChunks(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
  seal?: StreamedToolCallSeal;
}): void {
  const { graph, stepKey, toolCallChunks, seal } = args;
  if (toolCallChunks == null || toolCallChunks.length === 0) {
    return;
  }

  // Streamed args can be cumulative and parseable before the provider has
  // sealed the call. Recording stays separate from dispatch so the boundary
  // logic can wait for either a later tool index or the final tool-call signal.
  for (const toolCallChunk of toolCallChunks) {
    const key = getEagerToolChunkKey(stepKey, toolCallChunk);
    if (key == null) {
      continue;
    }

    const incomingId =
      toolCallChunk.id != null && toolCallChunk.id !== ''
        ? toolCallChunk.id
        : undefined;
    const incomingName =
      toolCallChunk.name != null && toolCallChunk.name !== ''
        ? toolCallChunk.name
        : undefined;
    const previous = graph.eagerEventToolCallChunks.get(key);
    const shouldReset =
      previous != null &&
      ((incomingId != null &&
        previous.id != null &&
        incomingId !== previous.id) ||
        (incomingName != null &&
          previous.name != null &&
          incomingName !== previous.name));
    const existing =
      previous == null || shouldReset
        ? {
          argsText: '',
        }
        : previous;
    const id = incomingId ?? existing.id;
    const name = incomingName ?? existing.name;
    const incomingArgs = toolCallChunk.args ?? '';
    const isRepeatedObservedFragment =
      incomingArgs !== '' &&
      incomingArgs.length > 1 &&
      incomingArgs === existing.lastArgsFragment;
    const argsText = isRepeatedObservedFragment
      ? existing.argsText
      : mergeToolCallArgsText(existing.argsText, incomingArgs);
    const index = getEagerToolChunkIndex(toolCallChunk) ?? existing.index;
    // Only a chunk whose explicit adapter seal covers THIS call may supply a
    // full-args restatement (OpenAI Responses `arguments.done`). Pure-signal
    // seals carry empty args and never set this.
    const sealCoversChunk =
      seal != null &&
      (seal.kind === 'all' ||
        (seal.id != null && seal.id === id) ||
        (seal.index != null && seal.index === index));
    const next = {
      id,
      name,
      argsText,
      // Canonical accumulation length: LangChain concats fragments verbatim
      // to build the final request, and every reconciliation branch above
      // yields text no longer than that concat — equal exactly when every
      // merge was a pure append. Tracking the length (not the text) keeps
      // cumulative/restating streams from retaining every prefix.
      rawArgsLength: (existing.rawArgsLength ?? 0) + incomingArgs.length,
      index,
      lastArgsFragment:
        incomingArgs !== '' ? incomingArgs : existing.lastArgsFragment,
      sealedArgsFragment:
        sealCoversChunk && incomingArgs !== ''
          ? incomingArgs
          : existing.sealedArgsFragment,
    };
    graph.eagerEventToolCallChunks.set(key, next);
  }
}

function getStreamedReadyToolCalls(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
  seal?: StreamedToolCallSeal;
  allowSequentialSeal?: boolean;
  sealAll?: boolean;
}): ToolCall[] {
  const {
    graph,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal = false,
    sealAll = false,
  } = args;
  const currentIndices = new Set<number>();
  for (const toolCallChunk of toolCallChunks ?? []) {
    const index = getEagerToolChunkIndex(toolCallChunk);
    if (index != null) {
      currentIndices.add(index);
    }
  }
  const highestCurrentIndex =
    currentIndices.size > 0 ? Math.max(...currentIndices) : undefined;
  const prefix = `${stepKey}\u0000`;
  const readyEntries: Array<{
    key: string;
    state: t.EagerEventToolCallChunkState;
    sealedByAdapter: boolean;
    parsedArgs: ToolCall['args'];
  }> = [];

  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    if (state.id != null && graph.eagerEventToolExecutions.has(state.id)) {
      graph.eagerEventToolCallChunks.delete(key);
      continue;
    }
    if (
      state.id == null ||
      state.id === '' ||
      state.name == null ||
      state.name === ''
    ) {
      continue;
    }
    const isSealedByLaterChunk =
      allowSequentialSeal &&
      highestCurrentIndex != null &&
      state.index != null &&
      state.index < highestCurrentIndex &&
      !currentIndices.has(state.index);
    const isSealedExplicitly =
      seal?.kind === 'single' &&
      ((seal.id != null && state.id === seal.id) ||
        (seal.index != null && state.index === seal.index));
    if (
      sealAll ||
      seal?.kind === 'all' ||
      isSealedByLaterChunk ||
      isSealedExplicitly
    ) {
      const parsedArgs = coerceRecordArgs(state.argsText);
      if (parsedArgs == null) {
        continue;
      }
      readyEntries.push({
        key,
        state,
        sealedByAdapter: isSealedExplicitly || seal?.kind === 'all',
        parsedArgs,
      });
    }
  }

  pruneEagerToolCallChunkStates({
    graph,
    stepKey,
    toolCallIds: new Set(
      readyEntries
        .map(({ state }) => state.id)
        .filter((id): id is string => id != null && id !== '')
    ),
  });
  if (sealAll) {
    pruneEagerToolCallChunkStates({ graph, stepKey, clearStep: true });
  }

  return readyEntries
    .sort((left, right) => (left.state.index ?? 0) - (right.state.index ?? 0))
    .flatMap(({ state, sealedByAdapter, parsedArgs }) => {
      // The final request's args come from LangChain's canonical verbatim
      // concatenation of fragments, while `argsText` reconciles provider
      // quirks with lossy heuristics that can also swallow legitimately
      // repetitive payload fragments (LibreChat#14371). `argsText` can never
      // be LONGER than the plain concat, so length equality proves it IS the
      // canonical accumulation.
      const isCanonicalAccumulation =
        state.rawArgsLength != null &&
        state.argsText.length === state.rawArgsLength;
      // Adapter seals may instead restate the finished call's full args on
      // the seal chunk itself (OpenAI Responses
      // `function_call_arguments.done`). Only when the seal-carrying chunk
      // supplied that fragment AND the accumulated text IS that restatement
      // has the adapter vouched for it — plain concatenation intentionally
      // differs there. Pure-signal seals (Bedrock contentBlockStop,
      // `args: ''`) never qualify.
      const isAuthoritativeRestatement =
        sealedByAdapter &&
        state.sealedArgsFragment != null &&
        state.sealedArgsFragment === state.argsText;
      // Prestarting an unconfirmed snapshot trips the "changed after eager
      // execution started" guard and burns a retry loop — leave unconfirmed
      // calls to normal ToolNode execution with final args.
      if (!isCanonicalAccumulation && !isAuthoritativeRestatement) {
        return [];
      }
      return [
        {
          id: state.id,
          name: state.name ?? '',
          args: parsedArgs,
        },
      ];
    });
}

function startReadyStreamedEagerToolExecutions(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
  seal?: StreamedToolCallSeal;
  allowSequentialSeal?: boolean;
  sealAll?: boolean;
}): void {
  const {
    graph,
    metadata,
    agentContext,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal,
    sealAll,
  } = args;
  if (
    hasPotentialDirectToolInStreamContext({ graph, agentContext }) ||
    hasDirectToolCallChunkInBatch({ graph, agentContext, toolCallChunks }) ||
    hasDirectToolCallChunkStateInStep({ graph, agentContext, stepKey }) ||
    !isEagerToolExecutionEnabledForBatch({ graph, metadata, agentContext })
  ) {
    return;
  }
  const toolCalls = getStreamedReadyToolCalls({
    graph,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal,
    sealAll,
  });
  if (toolCalls.length === 0) {
    return;
  }
  startEagerToolExecutions({
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting: true,
  });
}

/** Shares provider sealing/parsing with event tools, but never relaxes their batch guard. */
function startPreparedSubagents(
  graph: StandardGraph,
  agentContext: AgentContext | undefined,
  chunk: Partial<AIMessageChunk>,
  metadata?: Record<string, unknown>
): void {
  const attempt = resolveGenerationKey(metadata);
  if (
    (graph as Partial<StandardGraph>).canPrestartSubagents?.(agentContext) !==
      true ||
    graph.preparedSubagents.isOpen(attempt) !== true ||
    graph.hookRegistry?.hasResultAlteringHooks(
      graph.preparedSubagents.getConfig(attempt)?.configurable?.run_id
    ) === true
  ) {
    return;
  }
  let calls: ToolCall[];
  if (hasOnArrivalToolCallSeal(chunk) && (chunk.tool_calls?.length ?? 0) > 0) {
    calls = chunk.tool_calls!;
  } else {
    const seal = getStreamedToolCallSeal(chunk.response_metadata);
    const allowSequentialSeal =
      canPrestartSequentialStreamedToolChunks(agentContext) ||
      streamedToolCallAdapterAllowsSequentialSeal(chunk.response_metadata);
    if (!hasExplicitStreamedToolCallSeals(chunk) && !allowSequentialSeal) {
      return;
    }
    const stepKey = `prepared:${attempt}`;
    recordEagerToolCallChunks({
      graph,
      stepKey,
      toolCallChunks: chunk.tool_call_chunks ?? [],
      seal,
    });
    calls = getStreamedReadyToolCalls({
      graph,
      stepKey,
      toolCallChunks: chunk.tool_call_chunks,
      seal,
      allowSequentialSeal,
    });
    pruneEagerToolCallChunkStates({
      graph,
      stepKey,
      toolCallIds: new Set(calls.map((call) => call.id!)),
    });
  }
  for (const call of calls) {
    if (
      call.name === Constants.SUBAGENT &&
      !hasToolOutputReference(call.args)
    ) {
      graph.prestartSubagent(call, attempt, agentContext);
    }
  }
}

export function getChunkContent({
  chunk,
  provider,
  reasoningKey,
}: {
  chunk?: Partial<AIMessageChunk>;
  provider?: t.ProviderName;
  reasoningKey: 'reasoning_content' | 'reasoning';
}): string | t.MessageContentComplex[] | undefined {
  if (
    isGoogleLike(provider) &&
    Array.isArray(chunk?.content) &&
    chunk.content.some(
      (c) => isGoogleServerSideToolContentPart(c) || isNativeMediaContentPart(c)
    )
  ) {
    return chunk.content;
  }

  if (
    (provider === Providers.OPENAI || provider === Providers.AZURE) &&
    (
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text != null &&
    ((
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text?.length ?? 0) > 0
  ) {
    return (
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text;
  }
  if (provider === Providers.OPENROUTER) {
    // Content presence signals end of reasoning phase - prefer content over reasoning
    // This handles transitional chunks that may have both reasoning and content
    if (typeof chunk?.content === 'string' && chunk.content !== '') {
      return chunk.content;
    }
    const reasoning = chunk?.additional_kwargs?.reasoning as string | undefined;
    if (reasoning != null && reasoning !== '') {
      return reasoning;
    }
    const reasoningContent = chunk?.additional_kwargs?.reasoning_content as
      | string
      | undefined;
    if (reasoningContent != null && reasoningContent !== '') {
      return reasoningContent;
    }
    return chunk?.content;
  }
  const keyedReasoning = chunk?.additional_kwargs?.[reasoningKey] as
    | string
    | undefined;
  if (
    typeof chunk?.content === 'string' &&
    chunk.content !== '' &&
    keyedReasoning != null &&
    keyedReasoning !== ''
  ) {
    return chunk.content;
  }
  return ((keyedReasoning as string | undefined) ?? '') || chunk?.content;
}

function isDisableStreamingEnabled(
  clientOptions: t.ClientOptions | undefined
): boolean {
  return (
    clientOptions != null &&
    'disableStreaming' in clientOptions &&
    clientOptions.disableStreaming === true
  );
}

function hasReasoningContent(
  value: string | ReasoningSummaryLike | object[] | null | undefined
): boolean {
  if (typeof value === 'string') {
    return value !== '';
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (value == null) {
    return false;
  }
  return (
    value.summary?.some(
      (summary) => summary.text != null && summary.text.length > 0
    ) === true
  );
}

function shouldDeferMixedFinalReasoningChunk({
  chunk,
  agentContext,
}: {
  chunk: Partial<AIMessageChunk>;
  agentContext: AgentContext;
}): boolean {
  if (
    (chunk.tool_calls?.length ?? 0) > 0 ||
    (chunk.tool_call_chunks?.length ?? 0) > 0 ||
    typeof chunk.content !== 'string' ||
    chunk.content === ''
  ) {
    return false;
  }
  const additionalKwargs = chunk.additional_kwargs;
  if (
    agentContext.provider === Providers.OPENROUTER &&
    hasReasoningContent(additionalKwargs?.reasoning_details as object[])
  ) {
    return true;
  }
  if (!isDisableStreamingEnabled(agentContext.clientOptions)) {
    return false;
  }
  return (
    hasReasoningContent(
      additionalKwargs?.[agentContext.reasoningKey] as
        | string
        | ReasoningSummaryLike
        | null
        | undefined
    ) ||
    hasReasoningContent(
      additionalKwargs?.reasoning_content as
        | string
        | ReasoningSummaryLike
        | null
        | undefined
    ) ||
    hasReasoningContent(
      additionalKwargs?.reasoning as
        | string
        | ReasoningSummaryLike
        | null
        | undefined
    ) ||
    hasReasoningContent(additionalKwargs?.reasoning_details as object[])
  );
}

function hasCurrentTextDeltaStep({
  graph,
  metadata,
}: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
}): boolean {
  if (metadata == null) {
    return false;
  }
  const baseStepKey = graph.getStepBaseKey(metadata);
  for (const [stepKey, stepIds] of graph.stepKeyIds) {
    if (stepKey !== baseStepKey && !stepKey.startsWith(`${baseStepKey}_`)) {
      continue;
    }
    if (stepIds.some((stepId) => graph.messageStepHasTextDeltas.has(stepId))) {
      return true;
    }
  }
  return false;
}

function shouldSkipLateOpenRouterReasoningChunk({
  chunk,
  agentContext,
  graph,
  metadata,
}: {
  chunk: Partial<AIMessageChunk>;
  agentContext: AgentContext;
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
}): boolean {
  if (
    agentContext.provider !== Providers.OPENROUTER ||
    (chunk.tool_calls?.length ?? 0) > 0 ||
    (chunk.tool_call_chunks?.length ?? 0) > 0 ||
    (chunk.content != null && chunk.content !== '')
  ) {
    return false;
  }
  return (
    (hasReasoningContent(chunk.additional_kwargs?.reasoning as string) ||
      hasReasoningContent(
        chunk.additional_kwargs?.reasoning_content as string
      ) ||
      hasReasoningContent(
        chunk.additional_kwargs?.reasoning_details as object[]
      )) &&
    hasCurrentTextDeltaStep({ graph, metadata })
  );
}

/**
 * Brands a handler as one that dispatches content parts for the SDK — either
 * `ChatModelStreamHandler` itself or a wrapper forwarding to one.
 *
 * Identity alone is not a usable contract here. Hosts compose and wrap
 * handlers (`composeEventHandlers`, `createRunHandlers`), and every wrapper
 * fails `instanceof` while still driving the same dispatch. A brand survives
 * wrapping, so "does this handler own content-part dispatch" can be answered
 * about a value the SDK did not construct.
 */
export const SDK_STREAM_DISPATCH = Symbol.for(
  '@librechat/agents:chatModelStreamDispatch'
);

/** True when `handler` is, or forwards to, the SDK's stream dispatcher. */
export function dispatchesChatModelStream(handler?: t.EventHandler): boolean {
  if (handler == null) {
    return false;
  }
  if (handler instanceof ChatModelStreamHandler) {
    return true;
  }
  return Reflect.get(handler, SDK_STREAM_DISPATCH) === true;
}

export class ChatModelStreamHandler implements t.EventHandler {
  readonly [SDK_STREAM_DISPATCH] = true;

  async handle(
    event: string,
    data: t.StreamEventData,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph
  ): Promise<void> {
    if (!graph) {
      throw new Error('Graph not found');
    }
    if (!graph.config) {
      throw new Error('Config not found in graph');
    }

    if (!data.chunk) {
      console.warn(`No chunk found in ${event} event`);
      return;
    }

    const chunk = data.chunk as Partial<AIMessageChunk>;

    /** Attempts stamp their breaker epoch into event metadata; a mismatch
     * marks a straggling chunk from a failed run that outlived
     * `resetValues()`. Dropped OUTRIGHT: content handling and the eager
     * paths below compose the LIVE controller, so acting on a dead run's
     * chunk could dispatch host tools into the run now using it. Events
     * without a stamp (direct handler callers, partial stubs) keep the
     * live-controller behavior. */
    const eventEpoch = metadata?.[STREAM_LIMIT_EPOCH_KEY];
    /** Runtime-honest widening: partial handler stubs carry neither an
     * epoch nor a run scope despite the field types. */
    const liveEpoch = graph.breakerEpoch as number | undefined;
    if (eventEpoch != null && liveEpoch != null && eventEpoch !== liveEpoch) {
      return;
    }
    const eventBreaker =
      graph.breakerAbort instanceof AbortController
        ? graph.breakerAbort
        : undefined;
    /** Immutable scope captured at handler entry. A reset while this
     * handler is suspended in an await replaces the object, so ONE
     * reference comparison proves the event still belongs to the live run
     * before anything composes `graph.breakerAbort` or `graph.config`. */
    const entryRunScope = graph.runScope as RunBreakerScope | undefined;
    const runScopeInvalidated = (): boolean =>
      entryRunScope != null && graph.runScope !== entryRunScope;
    const throwIfRunBreakerTripped = (): void => {
      if (
        eventBreaker != null &&
        eventBreaker.signal.aborted &&
        (eventBreaker.signal.reason instanceof StreamLimitExceededError ||
          eventBreaker.signal.reason instanceof PreparedSubagentError)
      ) {
        throw eventBreaker.signal.reason;
      }
    };

    /**
     * Enforced before every content-specific early return below
     * (server-tool results, deferred mixed reasoning, late OpenRouter
     * reasoning): a looping provider can flood through any of those paths,
     * a coalesced event can carry client `tool_call_chunks` alongside a
     * server-tool result, and the complete-call dispatch branch further
     * down can prestart a side-effecting tool from an arrival-sealed
     * oversized call. Charging is claim-based: the producer loop and this
     * decoupled echo can observe the same chunk object in either order, and
     * only the first claimer charges it. The argument guard is
     * deliberately NOT gated on numeric chunk indices, so id-only or
     * index-less runaway streams stay bounded, and complete parsed
     * `tool_calls` without a raw chunk representation are judged standalone.
     */
    if (
      requiresStreamLimitAccounting(graph, chunk) &&
      claimStreamLimitCharge(graph, data.chunk, 'consumer', metadata)
    ) {
      try {
        enforceStreamDeltaEventLimit({ graph, metadata });
        /** Combined first so raw-chunk name correlation sees invalid calls
         * too; an unnamed raw chunk twinned with a named invalid call must
         * select that tool's override, not the global cap. */
        const completeCalls = combineCompleteToolCalls(chunk);
        if (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) {
          enforceStreamedToolCallArgLimit({
            graph,
            metadata,
            toolCallChunks: chunk.tool_call_chunks,
            responseMetadata: chunk.response_metadata as
              | Record<string, unknown>
              | undefined,
            parsedToolCalls: completeCalls,
          });
        }
        /** Judged whenever parsed calls are present, not only when raw
         * chunks are absent — an adapter can pair an empty or partial raw
         * chunk with a complete parsed call; the standalone check is
         * stateless, so the common both-present case is not double-tallied.
         * Invalid calls are included because ToolNode processes and
         * promotes them. */
        if (completeCalls != null) {
          enforceCompleteToolCallArgLimit({
            graph,
            metadata,
            toolCalls: completeCalls,
          });
        }
      } catch (error) {
        /** A breach detected on this consumer path must still stop parallel
         * fan-out work: the producer skips its own enforcement once this
         * side has claimed the emission, so createCallModel's breaker-abort
         * never fires for it. Trip the EVENT's run-bound breaker before the
         * throw rejects the run — never the live controller of a newer run. */
        if (error instanceof StreamLimitExceededError && eventBreaker != null) {
          eventBreaker.abort(error);
        }
        throw error;
      }
    }

    /** A parallel producer can trip the shared breaker while this event was
     * already queued in `streamEvents`. Stop before content handling or the
     * eager-tool paths below — those can dispatch a side-effecting host
     * tool with an already-aborted signal the handler never inspects.
     * Rechecked again immediately before each eager dispatch: the awaits in
     * between (server-tool results, tool-call handling, content dispatch)
     * are windows for a sibling's trip — or for a full reset, after which
     * this event belongs to a dead run and is dropped. */
    if (runScopeInvalidated()) {
      return;
    }
    throwIfRunBreakerTripped();

    const agentContext = graph.getAgentContext(metadata);

    const content = getChunkContent({
      chunk,
      reasoningKey: agentContext.reasoningKey,
      provider: agentContext.provider,
    });
    const skipHandling = await handleServerToolResult({
      graph,
      content,
      metadata,
      agentContext,
    });
    if (skipHandling) {
      return;
    }
    if (shouldDeferMixedFinalReasoningChunk({ chunk, agentContext })) {
      return;
    }
    if (
      shouldSkipLateOpenRouterReasoningChunk({
        chunk,
        agentContext,
        graph,
        metadata,
      })
    ) {
      return;
    }
    this.handleReasoning(chunk, agentContext);
    const stepKey = graph.getStepKey(metadata);
    let hasToolCalls = false;
    const hasToolCallChunks =
      (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) ?? false;
    const hasGoogleServerSideToolContent =
      isGoogleLike(agentContext.provider) &&
      Array.isArray(content) &&
      content.some(
        (c) =>
          isGoogleServerSideToolContentPart(c) || isNativeMediaContentPart(c)
      );
    if (hasGoogleServerSideToolContent && Array.isArray(content)) {
      await dispatchGoogleServerSideToolStreamContent({
        graph,
        stepKey,
        chunk,
        agentContext,
        content,
        metadata,
      });
    }

    if (
      chunk.tool_calls &&
      chunk.tool_calls.length > 0 &&
      chunk.tool_calls.every(
        (tc) =>
          tc.id != null &&
          tc.id !== '' &&
          (tc as Partial<ToolCall>).name != null &&
          tc.name !== ''
      )
    ) {
      hasToolCalls = true;
      await handleToolCalls(chunk.tool_calls, metadata, graph);
      if (runScopeInvalidated()) {
        return;
      }
      throwIfRunBreakerTripped();
      if (hasOnArrivalToolCallSeal(chunk)) {
        startPreparedSubagents(graph, agentContext, chunk, metadata);
      }
      if (hasFinalToolCallSignal(chunk)) {
        startEagerToolExecutions({
          graph,
          metadata,
          agentContext,
          toolCalls: chunk.tool_calls,
          skipExisting: true,
        });
        if (!hasToolCallChunks) {
          pruneEagerToolCallChunkStates({ graph, stepKey, clearStep: true });
        }
      } else if (
        hasOnArrivalToolCallSeal(chunk) &&
        !hasPotentialDirectToolInStreamContext({ graph, agentContext })
      ) {
        // Providers like Google never signal `tool_calls`/`tool_use` as the
        // finish reason, but their adapters seal calls on arrival — prestart
        // these mid-stream under the same direct-tool guard as streamed
        // chunk sealing.
        startEagerToolExecutions({
          graph,
          metadata,
          agentContext,
          toolCalls: chunk.tool_calls,
          skipExisting: true,
        });
      }
    }

    const isEmptyContent =
      typeof content === 'undefined' ||
      !content.length ||
      (typeof content === 'string' && !content);

    /** Set a preliminary message ID if found in empty chunk */
    const isEmptyChunk = isEmptyContent && !hasToolCallChunks;
    if (
      isEmptyChunk &&
      (chunk.id ?? '') !== '' &&
      !graph.prelimMessageIdsByStepKey.has(chunk.id ?? '')
    ) {
      graph.prelimMessageIdsByStepKey.set(stepKey, chunk.id ?? '');
    } else if (isEmptyChunk) {
      return;
    }

    if (
      hasToolCallChunks &&
      chunk.tool_call_chunks &&
      chunk.tool_call_chunks.length &&
      typeof chunk.tool_call_chunks[0]?.index === 'number'
    ) {
      const streamedToolCallSeal = getStreamedToolCallSeal(
        chunk.response_metadata as Record<string, unknown> | undefined
      );
      const allowSequentialSeal =
        canPrestartSequentialStreamedToolChunks(agentContext) ||
        streamedToolCallAdapterAllowsSequentialSeal(
          chunk.response_metadata as Record<string, unknown> | undefined
        );
      const canStreamEager =
        (allowSequentialSeal || hasExplicitStreamedToolCallSeals(chunk)) &&
        !hasPotentialDirectToolInStreamContext({ graph, agentContext }) &&
        isEagerToolExecutionEnabledForBatch({ graph, metadata, agentContext });
      if (canStreamEager) {
        recordEagerToolCallChunks({
          graph,
          stepKey,
          toolCallChunks: chunk.tool_call_chunks,
          seal: streamedToolCallSeal,
        });
      }
      await handleToolCallChunks({
        graph,
        stepKey,
        toolCallChunks: chunk.tool_call_chunks,
        metadata,
      });
      if (canStreamEager) {
        if (runScopeInvalidated()) {
          return;
        }
        throwIfRunBreakerTripped();
        startReadyStreamedEagerToolExecutions({
          graph,
          metadata,
          agentContext,
          stepKey,
          toolCallChunks: chunk.tool_call_chunks,
          seal: streamedToolCallSeal,
          allowSequentialSeal,
          sealAll: hasFinalToolCallSignal(chunk),
        });
      }
    }

    if (!hasOnArrivalToolCallSeal(chunk) && !runScopeInvalidated()) {
      throwIfRunBreakerTripped();
      startPreparedSubagents(graph, agentContext, chunk, metadata);
    }

    if (isEmptyContent) {
      return;
    }

    if (hasGoogleServerSideToolContent) {
      return;
    }

    if (Array.isArray(content) && content.every(isTextContentPart)) {
      const contentGroups = splitAssistantTextContentByPhase(content);
      const currentStepId = graph.stepKeyIds.get(stepKey)?.at(-1);
      const currentStep =
        currentStepId == null ? undefined : graph.getRunStep(currentStepId);
      const currentPhase =
        currentStep?.stepDetails.type === StepTypes.MESSAGE_CREATION
          ? currentStep.stepDetails.message_creation.phase
          : undefined;
      const nextPhase = getMessageCreationContentMetadata(
        contentGroups[0]
      ).phase;
      const phaseChanged =
        currentPhase != null && nextPhase != null && currentPhase !== nextPhase;
      if (contentGroups.length > 1 || phaseChanged) {
        for (const contentGroup of contentGroups) {
          const currentStepId = await dispatchMessageCreationStep({
            graph,
            stepKey,
            content: contentGroup,
            metadata,
          });
          await graph.dispatchMessageDelta(
            currentStepId,
            { content: contentGroup },
            metadata
          );
        }
        return;
      }
    }

    const message_id = getMessageId(stepKey, graph) ?? '';
    if (message_id) {
      const fallbackContentType =
        agentContext.currentTokenType === ContentTypes.TEXT
          ? ContentTypes.TEXT
          : ContentTypes.THINK;
      await graph.dispatchRunStep(
        stepKey,
        {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: {
            message_id,
            ...getMessageCreationContentMetadata(content, fallbackContentType),
          },
        },
        metadata
      );
    }

    let stepId = graph.getStepIdByKey(stepKey);
    let runStep = graph.getRunStep(stepId);
    if (
      shouldStartFreshMessageStepAfterGoogleServerSideTool({
        graph,
        stepId,
        runStep,
        content,
      })
    ) {
      stepId = await dispatchMessageCreationStep({
        graph,
        stepKey,
        content,
        metadata,
      });
      runStep = graph.getRunStep(stepId);
    }
    if (!runStep) {
      console.warn(`\n
==============================================================


Run step for ${stepId} does not exist, cannot dispatch delta event.

event: ${event}
stepId: ${stepId}
stepKey: ${stepKey}
message_id: ${message_id}
hasToolCalls: ${hasToolCalls}
hasToolCallChunks: ${hasToolCallChunks}

==============================================================
\n`);
      return;
    }

    /* Note: tool call chunks may have non-empty content that matches the current tool chunk generation */
    if (typeof content === 'string' && runStep.type === StepTypes.TOOL_CALLS) {
      return;
    } else if (
      hasToolCallChunks &&
      (chunk.tool_call_chunks?.some((tc) => tc.args === content) ?? false)
    ) {
      return;
    } else if (typeof content === 'string') {
      if (agentContext.currentTokenType === ContentTypes.TEXT) {
        await graph.dispatchMessageDelta(
          stepId,
          {
            content: [
              {
                type: ContentTypes.TEXT,
                text: content,
              },
            ],
          },
          metadata
        );
      } else if (agentContext.currentTokenType === 'think_and_text') {
        const { text, thinking } = parseThinkingContent(content);
        if (thinking) {
          await graph.dispatchReasoningDelta(
            stepId,
            {
              content: [
                {
                  type: ContentTypes.THINK,
                  think: thinking,
                },
              ],
            },
            metadata
          );
        }
        if (text) {
          agentContext.currentTokenType = ContentTypes.TEXT;
          agentContext.tokenTypeSwitch = 'content';
          const newStepKey = graph.getStepKey(metadata);
          const message_id = getMessageId(newStepKey, graph) ?? '';
          await graph.dispatchRunStep(
            newStepKey,
            {
              type: StepTypes.MESSAGE_CREATION,
              message_creation: {
                message_id,
                content_type: ContentTypes.TEXT,
              },
            },
            metadata
          );

          const newStepId = graph.getStepIdByKey(newStepKey);
          await graph.dispatchMessageDelta(
            newStepId,
            {
              content: [
                {
                  type: ContentTypes.TEXT,
                  text: text,
                },
              ],
            },
            metadata
          );
        }
      } else {
        await graph.dispatchReasoningDelta(
          stepId,
          {
            content: [
              {
                type: ContentTypes.THINK,
                think: content,
              },
            ],
          },
          metadata
        );
      }
    } else if (content.every((c) => isTextContentPart(c))) {
      await graph.dispatchMessageDelta(
        stepId,
        {
          content,
        },
        metadata
      );
    } else if (content.every((c) => isReasoningContentBlock(c))) {
      await graph.dispatchReasoningDelta(
        stepId,
        {
          content: content.map((c) => ({
            type: ContentTypes.THINK,
            think:
              (c as t.ThinkingContentText).thinking ??
              (c as Partial<t.GoogleReasoningContentText>).reasoning ??
              (c as Partial<t.BedrockReasoningContentText>).reasoningText
                ?.text ??
              '',
          })),
        },
        metadata
      );
    }
  }
  handleReasoning(
    chunk: Partial<AIMessageChunk>,
    agentContext: AgentContext
  ): void {
    let reasoning_content = chunk.additional_kwargs?.[
      agentContext.reasoningKey
    ] as string | Partial<ChatOpenAIReasoningSummary> | undefined;
    if (
      Array.isArray(chunk.content) &&
      (chunk.content[0]?.type === ContentTypes.THINKING ||
        chunk.content[0]?.type === ContentTypes.REASONING ||
        chunk.content[0]?.type === ContentTypes.REASONING_CONTENT ||
        chunk.content[0]?.type === 'redacted_thinking')
    ) {
      reasoning_content = 'valid';
    } else if (
      (agentContext.provider === Providers.OPENAI ||
        agentContext.provider === Providers.AZURE) &&
      reasoning_content != null &&
      typeof reasoning_content !== 'string' &&
      reasoning_content.summary?.[0]?.text != null &&
      reasoning_content.summary[0].text
    ) {
      reasoning_content = 'valid';
    } else if (
      agentContext.provider === Providers.OPENROUTER &&
      // Only set reasoning as valid if content is NOT present (content signals end of reasoning)
      (chunk.content == null || chunk.content === '') &&
      // Check for reasoning_details (final chunk) OR reasoning string (intermediate chunks)
      ((chunk.additional_kwargs?.reasoning_details != null &&
        Array.isArray(chunk.additional_kwargs.reasoning_details) &&
        chunk.additional_kwargs.reasoning_details.length > 0) ||
        (typeof chunk.additional_kwargs?.reasoning === 'string' &&
          chunk.additional_kwargs.reasoning !== '') ||
        (typeof chunk.additional_kwargs?.reasoning_content === 'string' &&
          chunk.additional_kwargs.reasoning_content !== ''))
    ) {
      reasoning_content = 'valid';
    }
    if (
      reasoning_content != null &&
      reasoning_content !== '' &&
      (chunk.content == null ||
        chunk.content === '' ||
        reasoning_content === 'valid')
    ) {
      agentContext.currentTokenType = ContentTypes.THINK;
      agentContext.tokenTypeSwitch = 'reasoning';
      return;
    } else if (
      agentContext.tokenTypeSwitch === 'reasoning' &&
      agentContext.currentTokenType !== ContentTypes.TEXT &&
      ((chunk.content != null && chunk.content !== '') ||
        (chunk.tool_calls?.length ?? 0) > 0 ||
        (chunk.tool_call_chunks?.length ?? 0) > 0)
    ) {
      agentContext.currentTokenType = ContentTypes.TEXT;
      agentContext.tokenTypeSwitch = 'content';
      agentContext.reasoningTransitionCount++;
    } else if (
      chunk.content != null &&
      typeof chunk.content === 'string' &&
      chunk.content.includes('<think>') &&
      chunk.content.includes('</think>')
    ) {
      agentContext.currentTokenType = 'think_and_text';
      agentContext.tokenTypeSwitch = 'content';
    } else if (
      chunk.content != null &&
      typeof chunk.content === 'string' &&
      chunk.content.includes('<think>')
    ) {
      agentContext.currentTokenType = ContentTypes.THINK;
      agentContext.tokenTypeSwitch = 'content';
    } else if (
      agentContext.lastToken != null &&
      agentContext.lastToken.includes('</think>')
    ) {
      agentContext.currentTokenType = ContentTypes.TEXT;
      agentContext.tokenTypeSwitch = 'content';
    }
    if (typeof chunk.content !== 'string') {
      return;
    }
    agentContext.lastToken = chunk.content;
  }
}

export function createContentAggregator(): t.ContentAggregatorResult {
  type ToolStepContentState = {
    indices: Set<number>;
    chunkIndices: Map<number, number>;
    unclaimedIndices: Set<number>;
    unboundIndices: Set<number>;
    callIdsByIndex: Map<number, string>;
  };

  const contentParts: Array<t.MessageContentComplex | undefined> = [];
  const stepMap = new Map<string, t.RunStep>();
  const toolCallContentIndexMap = new Map<string, number>();
  const sourceContentIndexMap = new Map<number, number>();
  const toolStepContentMap = new Map<string, ToolStepContentState>();
  let indexedContentLength = 0;
  /** Physical append cursor; event/chunk indices are correlation keys, not array offsets. */
  let nextContentIndex = 0;
  // Track agentId and groupId for each content index (applied to content parts)
  const contentMetaMap = new Map<
    number,
    { agentId?: string; groupId?: number }
  >();
  /** A delta's content may carry several parts (e.g. Google server-side tool
   *  chunks emit multiple reasoning entries at once); every entry must reach
   *  the step's slot, in order, or streamed text is silently lost. */
  const getDeltaContentParts = (
    content?: t.MessageDelta['content'] | t.MessageContentComplex
  ): t.MessageContentComplex[] => {
    if (content == null) {
      return [];
    }
    return Array.isArray(content) ? content : [content];
  };
  const indexContentPart = (
    index: number,
    contentPart?: t.MessageContentComplex
  ): void => {
    if (contentPart == null) {
      return;
    }
    if (contentPart.type === ContentTypes.TOOL_CALL) {
      const toolCallId = contentPart.tool_call.id;
      if (toolCallId != null && toolCallId !== '') {
        toolCallContentIndexMap.set(toolCallId, index);
      }
    }
    const hasAgentId =
      contentPart.agentId != null && contentPart.agentId !== '';
    const hasGroupId = contentPart.groupId != null;
    if (hasAgentId || hasGroupId) {
      const existingMeta = contentMetaMap.get(index) ?? {};
      if (hasAgentId) {
        existingMeta.agentId = contentPart.agentId;
      }
      if (hasGroupId) {
        existingMeta.groupId = contentPart.groupId;
      }
      contentMetaMap.set(index, existingMeta);
    }
  };
  const syncSeededContent = (): void => {
    /** Hosts can seed pre-pause content after creating the aggregator. */
    for (
      let index = indexedContentLength;
      index < contentParts.length;
      index++
    ) {
      indexContentPart(index, contentParts[index]);
    }
    indexedContentLength = contentParts.length;
    nextContentIndex = Math.max(nextContentIndex, contentParts.length);
  };
  const getToolCallContentIndex = (
    toolCallId: string | undefined
  ): number | undefined => {
    if (toolCallId == null || toolCallId === '') {
      return undefined;
    }
    syncSeededContent();
    return toolCallContentIndexMap.get(toolCallId);
  };
  const allocateContentIndex = (minimumIndex = 0): number => {
    syncSeededContent();
    const contentIndex = Math.max(nextContentIndex, minimumIndex);
    nextContentIndex = contentIndex + 1;
    return contentIndex;
  };
  const resolveSourceContentIndex = (sourceIndex: number): number => {
    const existingIndex = sourceContentIndexMap.get(sourceIndex);
    if (existingIndex != null) {
      return existingIndex;
    }
    const contentIndex = allocateContentIndex(sourceIndex);
    sourceContentIndexMap.set(sourceIndex, contentIndex);
    return contentIndex;
  };
  const createToolStepContentState = (
    contentIndex: number
  ): ToolStepContentState => ({
    indices: new Set<number>([contentIndex]),
    chunkIndices: new Map<number, number>(),
    unclaimedIndices: new Set<number>([contentIndex]),
    unboundIndices: new Set<number>([contentIndex]),
    callIdsByIndex: new Map<number, string>(),
  });
  const registerToolContentIndex = (
    state: ToolStepContentState,
    contentIndex: number,
    toolCallId?: string
  ): void => {
    if (!state.indices.has(contentIndex)) {
      state.indices.add(contentIndex);
      state.unclaimedIndices.add(contentIndex);
      state.unboundIndices.add(contentIndex);
    }
    if (toolCallId != null && toolCallId !== '') {
      const existingToolCallId = state.callIdsByIndex.get(contentIndex);
      if (
        existingToolCallId != null &&
        existingToolCallId !== toolCallId &&
        toolCallContentIndexMap.get(existingToolCallId) === contentIndex
      ) {
        toolCallContentIndexMap.delete(existingToolCallId);
      }
      toolCallContentIndexMap.set(toolCallId, contentIndex);
      state.callIdsByIndex.set(contentIndex, toolCallId);
      state.unboundIndices.delete(contentIndex);
    }
  };
  const takeFirstIndex = (indices: Set<number>): number | undefined => {
    const contentIndex = indices.values().next().value;
    if (contentIndex != null) {
      indices.delete(contentIndex);
    }
    return contentIndex;
  };
  const setContentMeta = (index: number, runStep: t.RunStep): void => {
    const hasAgentId = runStep.agentId != null && runStep.agentId !== '';
    const hasGroupId = runStep.groupId != null;
    if (!hasAgentId && !hasGroupId) {
      return;
    }
    const existingMeta = contentMetaMap.get(index) ?? {};
    if (hasAgentId) {
      existingMeta.agentId = runStep.agentId;
    }
    if (hasGroupId) {
      existingMeta.groupId = runStep.groupId;
    }
    contentMetaMap.set(index, existingMeta);
  };
  const applyContentMetadata = (index: number): void => {
    const contentPart = contentParts[index];
    if (contentPart == null) {
      return;
    }
    const meta = contentMetaMap.get(index);
    if (meta?.agentId != null) {
      contentPart.agentId = meta.agentId;
    }
    if (meta?.groupId != null) {
      contentPart.groupId = meta.groupId;
    }
  };

  const updateContent = (
    index: number,
    contentPart?: t.MessageContentComplex,
    finalUpdate = false
  ): void => {
    if (!contentPart) {
      console.warn('No content part found in \'updateContent\'');
      return;
    }
    const partType = contentPart.type ?? '';
    if (!partType) {
      console.warn('No content type found in content part');
      return;
    }

    if (!contentParts[index] && partType !== ContentTypes.TOOL_CALL) {
      contentParts[index] = { type: partType };
    }

    if (!partType.startsWith(contentParts[index]?.type ?? '')) {
      console.warn('Content type mismatch');
      return;
    }

    const incomingText =
      ContentTypes.TEXT in contentPart && typeof contentPart.text === 'string'
        ? contentPart.text
        : undefined;
    /**
     * Anthropic emits a search turn's citations as their own `citations_delta`,
     * which arrives here as a text part carrying `citations` and no `text`.
     */
    const { citations } = contentPart as {
      citations?: t.MessageDeltaUpdate['citations'];
    };
    const incomingCitations = Array.isArray(citations) ? citations : undefined;

    if (
      partType.startsWith(ContentTypes.TEXT) &&
      (incomingText !== undefined || incomingCitations !== undefined)
    ) {
      // TODO: update this!!
      const currentContent = contentParts[index] as t.MessageDeltaUpdate;
      const update: t.MessageDeltaUpdate = {
        type: ContentTypes.TEXT,
        text: (currentContent.text || '') + (incomingText ?? ''),
      };

      if (contentPart.tool_call_ids) {
        update.tool_call_ids = contentPart.tool_call_ids;
      } else if (incomingText === undefined && currentContent.tool_call_ids) {
        /** A citations-only delta must not drop ids already accumulated */
        update.tool_call_ids = currentContent.tool_call_ids;
      }

      if (incomingCitations !== undefined) {
        update.citations =
          currentContent.citations !== undefined
            ? [...currentContent.citations, ...incomingCitations]
            : [...incomingCitations];
      } else if (currentContent.citations !== undefined) {
        update.citations = currentContent.citations;
      }
      if (contentPart.native_media != null) {
        update.native_media = contentPart.native_media;
      }
      contentParts[index] = update;
    } else if (
      partType.startsWith(ContentTypes.THINK) &&
      ContentTypes.THINK in contentPart &&
      typeof contentPart.think === 'string'
    ) {
      const currentContent = contentParts[index] as t.ReasoningDeltaUpdate;
      const update: t.ReasoningDeltaUpdate = {
        type: ContentTypes.THINK,
        think: (currentContent.think || '') + contentPart.think,
      };
      contentParts[index] = update;
    } else if (
      partType.startsWith(ContentTypes.AGENT_UPDATE) &&
      ContentTypes.AGENT_UPDATE in contentPart &&
      contentPart.agent_update != null
    ) {
      const update: t.AgentUpdate = {
        type: ContentTypes.AGENT_UPDATE,
        agent_update: contentPart.agent_update,
      };

      contentParts[index] = update;
    } else if (
      partType === ContentTypes.IMAGE_FILE &&
      'image_file' in contentPart
    ) {
      contentParts[index] = { ...contentPart };
    } else if (partType === 'toolCall' || partType === 'toolResponse') {
      contentParts[index] = contentPart;
    } else if (partType === ContentTypes.SUMMARY) {
      const currentSummary = contentParts[index] as
        | t.SummaryContentBlock
        | undefined;
      const incoming = contentPart as t.SummaryContentBlock;
      contentParts[index] = {
        ...incoming,
        content: [
          ...(currentSummary?.content ?? []),
          ...(incoming.content ?? []),
        ],
      };
    } else if (
      partType === ContentTypes.IMAGE_URL &&
      'image_url' in contentPart
    ) {
      const currentContent = contentParts[index] as {
        type: 'image_url';
        image_url: string;
      };
      contentParts[index] = {
        ...currentContent,
      };
    } else if (
      partType === ContentTypes.TOOL_CALL &&
      'tool_call' in contentPart
    ) {
      const incomingName = contentPart.tool_call.name;
      const incomingId = contentPart.tool_call.id;
      const toolCallArgs = (contentPart.tool_call as t.ToolCallPart).args;

      // When we receive a tool call with a name, it's the complete tool call
      // Consolidate with any previously accumulated args from chunks
      const hasValidName = incomingName != null && incomingName !== '';

      // Only process if incoming has a valid name (complete tool call)
      // or if we're doing a final update with complete data
      if (!hasValidName && !finalUpdate) {
        return;
      }

      const existingContent = contentParts[index] as
        | (Omit<t.ToolCallContent, 'tool_call'> & {
            tool_call?: t.ToolCallPart & t.PartMetadata;
          })
        | undefined;
      if (!finalUpdate && existingContent?.tool_call?.progress === 1) {
        return;
      }

      /** When args are a valid object, they are likely already invoked */
      let args =
        finalUpdate ||
        typeof existingContent?.tool_call?.args === 'object' ||
        typeof toolCallArgs === 'object'
          ? contentPart.tool_call.args
          : (existingContent?.tool_call?.args ?? '') + (toolCallArgs ?? '');
      if (
        finalUpdate &&
        args == null &&
        existingContent?.tool_call?.args != null
      ) {
        args = existingContent.tool_call.args;
      }

      const id =
        getNonEmptyValue([incomingId, existingContent?.tool_call?.id]) ?? '';
      const name =
        getNonEmptyValue([incomingName, existingContent?.tool_call?.name]) ??
        '';
      const existingToolCallId = existingContent?.tool_call?.id;
      if (
        existingToolCallId != null &&
        existingToolCallId !== '' &&
        existingToolCallId !== id &&
        toolCallContentIndexMap.get(existingToolCallId) === index
      ) {
        toolCallContentIndexMap.delete(existingToolCallId);
      }

      const newToolCall: ToolCall & t.PartMetadata & { outcome?: string } = {
        id,
        name,
        args,
        type: ToolCallTypes.TOOL_CALL,
      };

      const auth =
        contentPart.tool_call.auth ?? existingContent?.tool_call?.auth;
      const expiresAt =
        contentPart.tool_call.expires_at ??
        existingContent?.tool_call?.expires_at;
      if (auth != null) {
        newToolCall.auth = auth;
        newToolCall.expires_at = expiresAt;
      }

      if (finalUpdate) {
        newToolCall.progress = 1;
        newToolCall.output = contentPart.tool_call.output;
        const outcome = (contentPart.tool_call as t.ToolCallPart).outcome;
        if (outcome != null) {
          newToolCall.outcome = outcome;
        }
      }

      contentParts[index] = {
        type: ContentTypes.TOOL_CALL,
        tool_call: newToolCall,
      };
      indexContentPart(index, contentParts[index]);
      indexedContentLength = Math.max(indexedContentLength, index + 1);
    }

    // Apply agentId (for MultiAgentGraph) and groupId (for parallel execution) to content parts
    // - agentId present → MultiAgentGraph (show agent labels)
    // - groupId present → parallel execution (render columns)
    applyContentMetadata(index);
  };

  const aggregateContent = ({
    event,
    data,
  }: {
    event: GraphEvents;
    data:
      | t.RunStep
      | t.AgentUpdate
      | t.MessageDeltaEvent
      | t.ReasoningDeltaEvent
      | t.RunStepDeltaEvent
      | t.SummarizeDeltaData
      | t.SummarizeCompleteEvent
      | { result: t.ToolEndEvent };
  }): void => {
    if (event === GraphEvents.ON_SUMMARIZE_DELTA) {
      const deltaData = data as t.SummarizeDeltaData;
      const runStep = stepMap.get(deltaData.id);
      if (!runStep) {
        console.warn('No run step found for summarize delta event');
        return;
      }
      updateContent(runStep.index, deltaData.delta.summary);
      return;
    }

    if (event === GraphEvents.ON_SUMMARIZE_COMPLETE) {
      const completeData = data as t.SummarizeCompleteEvent;
      const summary = completeData.summary;
      if (!summary?.boundary) {
        return;
      }
      const runStep = stepMap.get(summary.boundary.messageId);
      if (!runStep) {
        return;
      }
      // Replace accumulated delta text with the authoritative final summary.
      // Multi-stage summarization streams deltas from each chunk, which
      // concatenate in updateContent.  This event carries only the correct
      // final text from the last stage.
      contentParts[runStep.index] = summary;
      applyContentMetadata(runStep.index);
      return;
    }

    if (event === GraphEvents.ON_RUN_STEP) {
      const incomingRunStep = data as t.RunStep;
      const toolCalls =
        incomingRunStep.stepDetails.type === StepTypes.TOOL_CALLS
          ? (incomingRunStep.stepDetails.tool_calls as ToolCall[] | undefined)
          : undefined;
      syncSeededContent();
      let runStepIndex: number;
      const toolCallIndices: number[] = [];
      if (toolCalls && toolCalls.length > 0) {
        for (let index = 0; index < toolCalls.length; index++) {
          const toolCallId = toolCalls[index].id;
          let contentIndex = getToolCallContentIndex(toolCallId);
          if (contentIndex == null) {
            contentIndex =
              index === 0
                ? resolveSourceContentIndex(incomingRunStep.index)
                : allocateContentIndex();
          } else if (index === 0) {
            sourceContentIndexMap.set(incomingRunStep.index, contentIndex);
          }
          toolCallIndices.push(contentIndex);
        }
        runStepIndex = toolCallIndices[0];
      } else {
        runStepIndex = resolveSourceContentIndex(incomingRunStep.index);
      }
      const runStep =
        runStepIndex !== incomingRunStep.index
          ? { ...incomingRunStep, index: runStepIndex }
          : incomingRunStep;
      stepMap.set(runStep.id, runStep);

      setContentMeta(runStep.index, runStep);

      if (runStep.summary != null) {
        updateContent(runStep.index, runStep.summary);
      }

      if (runStep.stepDetails.type === StepTypes.TOOL_CALLS) {
        const toolStepContent =
          toolStepContentMap.get(runStep.id) ??
          createToolStepContentState(runStep.index);
        registerToolContentIndex(toolStepContent, runStep.index);
        (runStep.stepDetails.tool_calls as ToolCall[] | undefined)?.forEach(
          (toolCall, toolCallIndex) => {
            const contentIndex =
              toolCallIndices[toolCallIndex] ?? runStep.index;
            const toolCallId = toolCall.id ?? '';
            registerToolContentIndex(toolStepContent, contentIndex, toolCallId);
            const contentPart: t.MessageContentComplex = {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                args: toolCall.args,
                name: toolCall.name,
                id: toolCallId,
              },
            };

            setContentMeta(contentIndex, runStep);
            updateContent(contentIndex, contentPart);
          }
        );
        toolStepContentMap.set(runStep.id, toolStepContent);
      }
    } else if (event === GraphEvents.ON_MESSAGE_DELTA) {
      const messageDelta = data as t.MessageDeltaEvent;
      const runStep = stepMap.get(messageDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for message delta event');
        return;
      }

      for (const contentPart of getDeltaContentParts(
        messageDelta.delta.content
      )) {
        updateContent(runStep.index, contentPart);
      }
    } else if (
      event === GraphEvents.ON_AGENT_UPDATE &&
      (data as t.AgentUpdate | undefined)?.agent_update
    ) {
      const contentPart = data as t.AgentUpdate | undefined;
      if (!contentPart) {
        return;
      }
      const contentIndex = resolveSourceContentIndex(
        contentPart.agent_update.index
      );
      updateContent(contentIndex, {
        ...contentPart,
        agent_update: {
          ...contentPart.agent_update,
          index: contentIndex,
        },
      });
    } else if (event === GraphEvents.ON_REASONING_DELTA) {
      const reasoningDelta = data as t.ReasoningDeltaEvent;
      const runStep = stepMap.get(reasoningDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for reasoning delta event');
        return;
      }

      for (const contentPart of getDeltaContentParts(
        reasoningDelta.delta.content
      )) {
        updateContent(runStep.index, contentPart);
      }
    } else if (event === GraphEvents.ON_RUN_STEP_DELTA) {
      const runStepDelta = data as t.RunStepDeltaEvent;
      const runStep = stepMap.get(runStepDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for run step delta event');
        return;
      }

      if (
        runStepDelta.delta.type === StepTypes.TOOL_CALLS &&
        runStepDelta.delta.tool_calls
      ) {
        const toolStepContent =
          toolStepContentMap.get(runStepDelta.id) ??
          createToolStepContentState(runStep.index);
        runStepDelta.delta.tool_calls.forEach((toolCallDelta) => {
          const chunkIndex =
            typeof toolCallDelta.index === 'number'
              ? toolCallDelta.index
              : undefined;
          const explicitToolCallId =
            toolCallDelta.id != null && toolCallDelta.id !== ''
              ? toolCallDelta.id
              : undefined;
          let contentIndex = getToolCallContentIndex(explicitToolCallId);
          if (contentIndex == null && chunkIndex != null) {
            contentIndex = toolStepContent.chunkIndices.get(chunkIndex);
          }
          const soleContentIndex =
            toolStepContent.indices.size === 1
              ? toolStepContent.indices.values().next().value
              : undefined;
          if (
            contentIndex == null &&
            explicitToolCallId == null &&
            soleContentIndex != null &&
            (chunkIndex == null ||
              toolStepContent.callIdsByIndex.has(soleContentIndex) ||
              toolStepContent.unclaimedIndices.has(soleContentIndex))
          ) {
            contentIndex = soleContentIndex;
          }
          if (contentIndex == null) {
            if (chunkIndex == null && explicitToolCallId == null) {
              console.warn(
                'No tool call id or chunk index found for run step delta event'
              );
              return;
            }
            contentIndex =
              explicitToolCallId == null
                ? takeFirstIndex(toolStepContent.unclaimedIndices)
                : takeFirstIndex(toolStepContent.unboundIndices);
            contentIndex ??= allocateContentIndex();
          }
          registerToolContentIndex(
            toolStepContent,
            contentIndex,
            explicitToolCallId
          );
          if (chunkIndex != null) {
            toolStepContent.chunkIndices.set(chunkIndex, contentIndex);
            toolStepContent.unclaimedIndices.delete(contentIndex);
            toolStepContent.unboundIndices.delete(contentIndex);
          }
          const toolCallId =
            explicitToolCallId ??
            toolStepContent.callIdsByIndex.get(contentIndex);

          const contentPart: t.MessageContentComplex = {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              args: toolCallDelta.args ?? '',
              name: toolCallDelta.name,
              id: toolCallId,
              auth: runStepDelta.delta.auth,
              expires_at: runStepDelta.delta.expires_at,
            },
          };

          setContentMeta(contentIndex, runStep);
          updateContent(contentIndex, contentPart);
        });
        toolStepContentMap.set(runStepDelta.id, toolStepContent);
      }
    } else if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
      const { result } = data as unknown as {
        result:
          | t.ToolEndEvent
          | (t.SummaryCompleted & { id: string; index: number });
      };

      const { id: stepId } = result;

      const runStep = stepMap.get(stepId);

      if (result.type === ContentTypes.SUMMARY && 'summary' in result) {
        if (!runStep) {
          console.warn('No run step or runId found for completed step event');
          return;
        }
        contentParts[runStep.index] = result.summary as t.MessageContentComplex;
        applyContentMetadata(runStep.index);
      } else if ('tool_call' in result) {
        let contentIndex = getToolCallContentIndex(result.tool_call.id);
        if (contentIndex == null && runStep != null) {
          const toolStepContent = toolStepContentMap.get(runStep.id);
          if (toolStepContent?.indices.size === 1) {
            contentIndex = toolStepContent.indices.values().next().value;
          } else if (toolStepContent == null) {
            const declaredToolCalls =
              runStep.stepDetails.type === StepTypes.TOOL_CALLS
                ? runStep.stepDetails.tool_calls
                : undefined;
            if ((declaredToolCalls?.length ?? 0) <= 1) {
              contentIndex = runStep.index;
            }
          }
        }
        if (contentIndex == null) {
          console.warn(
            'No run step or tool call found for completed step event'
          );
          return;
        }
        if (runStep != null) {
          setContentMeta(contentIndex, runStep);
        }
        const contentPart: t.MessageContentComplex = {
          type: ContentTypes.TOOL_CALL,
          tool_call: (result as t.ToolEndEvent).tool_call,
        };
        updateContent(contentIndex, contentPart, true);
      }
    }
  };

  return { contentParts, aggregateContent, stepMap };
}

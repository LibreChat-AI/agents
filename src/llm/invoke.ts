import { concat } from '@langchain/core/utils/stream';
import { AIMessageChunk } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { getCallbackManagerForConfig } from '@langchain/core/runnables';
import {
  CallbackManager,
  CallbackManagerForLLMRun,
  type Callbacks,
} from '@langchain/core/callbacks/manager';
import type { Serialized } from '@langchain/core/load/serializable';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ChatGeneration } from '@langchain/core/outputs';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { PreparedProviderRequest } from '@/llm/prepareProviderRequest';
import type { ContextOverflowContext } from '@/utils/errors';
import type { StreamLimitState } from '@/llm/streamLimits';
import type { PreemptAction } from '@/llm/preempt';
import type * as t from '@/types';
import {
  enforceStreamLimitsForWireChunk,
  registerActiveStreamLimitGeneration,
  releaseStreamLimitGeneration,
  resolveGenerationKey,
  streamLimitAccountingEnabled,
  StreamLimitExceededError,
  STREAM_LIMIT_REDISPATCH_KEY,
  STREAM_LIMIT_ATTEMPT_KEY,
} from '@/llm/streamLimits';
import {
  inspectProviderMessageProjection,
  ProviderMessageProjectionInvariantError,
  resolveProviderMessageProjectionInvariantMode,
  modifyDeltaProperties,
} from '@/messages';
import {
  canRestartPreempt,
  notePreemptRestartedRun,
  PREEMPT_RESTART_CONTROL_FLOW,
  resolvePreemptAction,
  resolveRestartGraceMs,
} from '@/llm/preempt';
import {
  detachValidatedModelToolCalls,
  snapshotValidatedModelChunk,
  InvalidModelToolCallError,
} from '@/graphs/acceptedModelResponse';
import {
  assertPreparedProviderRequestFor,
  prepareProviderRequest,
} from '@/llm/prepareProviderRequest';
import {
  getProviderFamily,
  providerUsesManualToolStream,
} from '@/llm/providers';
import { ChatModelStreamHandler, dispatchesChatModelStream } from '@/stream';
import { Constants, ContentTypes, GraphEvents, Providers } from '@/common';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { assertNotTruncatedToolCall } from '@/llm/truncation';
import { resolveClientOptionsModel } from '@/llm/request';
import { safeDispatchCustomEvent } from '@/utils/events';
import { getContextOverflowInfo } from '@/utils/errors';
import { appendCallbacks } from '@/utils/callbacks';
import { composeAbortSignals } from '@/utils/misc';
import { initializeModel } from '@/llm/init';

export {
  projectMessagesForProvider,
  resolveServingModelId,
  usesNativeOpenAIResponses,
} from '@/llm/prepareProviderRequest';
export type {
  PreparedProviderRequest,
  PrepareProviderRequestParams,
  ProviderMessageProjectionMode,
  ProviderPayloadMeasurement,
  ProviderRequestContext,
} from '@/llm/prepareProviderRequest';

/**
 * Context passed to `attemptInvoke`. Matches the subset of Graph that
 * `ChatModelStreamHandler.handle` needs *plus* the explicit
 * `getOrCreateToolOutputRegistry()` accessor used while preparing a
 * provider request. Raw callers prepare inside `attemptInvoke`; Graph
 * callers prepare before final payload measurement and pass the exact
 * artifact through.
 *
 * The intersection is intentional: `Parameters<...>[3]` resolves
 * indirectly through the stream handler's signature (which returns
 * `StandardGraph` and already exposes the accessor since #117), but
 * stating it explicitly here surfaces the contract at the call site —
 * a developer reading `attemptInvoke` doesn't have to chase the
 * upstream handler's parameter list to discover that
 * `context?.getOrCreateToolOutputRegistry()` is a real thing. Single
 * optional chain only — the method itself is required on the
 * `StandardGraph` branch of the intersection, so the second `?.` is
 * unnecessary at the call site.
 *
 * `NonNullable<...>` strips `undefined` from the upstream parameter
 * type so the intersection doesn't collapse to `never` on the
 * undefined branch; callers express optionality via `context?:
 * InvokeContext` on the function signature instead.
 *
 * Callers without a registry (e.g. summarization) simply pass no
 * `context` and the transform safely no-ops.
 */
export type InvokeContext = NonNullable<
  Parameters<ChatModelStreamHandler['handle']>[3]
> & {
  getOrCreateToolOutputRegistry?(): ToolOutputReferenceRegistry | undefined;
};

/**
 * Per-chunk callback for custom stream processing.
 * When provided, replaces the default `ChatModelStreamHandler`.
 *
 * `metadata` is the attempt's callback metadata (carrying the provider and
 * stream-limit attempt stamps), so consumers that count against the stream
 * limits key each model attempt separately.
 */
export type OnChunk = (
  chunk: AIMessageChunk,
  metadata?: Record<string, unknown>
) => void | Promise<void>;

/** Node coerces a `setTimeout` delay past this to 1ms, so a longer grace is
 *  reached by chaining rather than by one out-of-range timer. */
const MAX_TIMEOUT_MS = 2_147_483_647;

/** Unique per-model-attempt sequence; see the stamp in `attemptInvoke`. */
let streamLimitAttemptSeq = 0;

function createModelStartHandler({
  config,
  mode,
  provider,
  captureRunId,
}: {
  config: RunnableConfig;
  mode: ReturnType<typeof resolveProviderMessageProjectionInvariantMode>;
  provider: t.ProviderName;
  captureRunId?: (runId: string) => void;
}): BaseCallbackHandler {
  let inspected = false;
  const handler = BaseCallbackHandler.fromMethods({
    handleChatModelStart: async (
      _llm: Serialized,
      messageBatches: BaseMessage[][],
      runId: string
    ): Promise<void> => {
      captureRunId?.(runId);
      if (mode === 'off' || inspected) {
        return;
      }
      inspected = true;
      const report = inspectProviderMessageProjection(messageBatches[0] ?? []);
      if (report.valid) {
        return;
      }
      if (mode === 'assert') {
        throw new ProviderMessageProjectionInvariantError(report);
      }
      try {
        const callbackManager = await getCallbackManagerForConfig(config);
        await callbackManager?.handleCustomEvent?.(
          GraphEvents.ON_AGENT_LOG,
          {
            level: 'warn',
            scope: 'projection',
            message: 'Provider message projection has provenance gaps',
            data: { provider, report },
            runId,
          } satisfies t.AgentLogEvent,
          runId
        );
      } catch {
        return;
      }
    },
  });
  handler.name = 'provider-message-projection-invariant';
  handler.raiseError = mode === 'assert';
  handler.awaitHandlers = true;
  return handler;
}

function withModelStartHandler({
  config,
  mode,
  provider,
  captureRunId,
}: {
  config: RunnableConfig;
  mode: ReturnType<typeof resolveProviderMessageProjectionInvariantMode>;
  provider: t.ProviderName;
  captureRunId?: (runId: string) => void;
}): RunnableConfig {
  return {
    ...config,
    callbacks: appendCallbacks(config.callbacks, [
      createModelStartHandler({ config, mode, provider, captureRunId }),
    ]),
  };
}

function getManualToolStreamNormalizationProvider(
  provider: t.ProviderName
): t.ProviderName {
  const family = getProviderFamily(provider);
  if (family === 'anthropic') {
    return Providers.ANTHROPIC;
  }
  if (family === 'bedrock') {
    return Providers.BEDROCK;
  }
  return provider;
}

/**
 * The registered handler that owns content-part dispatch, if any.
 *
 * Detected by brand rather than by `instanceof`: a host that registers
 * `new ChatModelStreamHandler()` to opt out of sealing gets wrapped by
 * `createRunHandlers` on every `AgentSession` run, and by
 * `composeEventHandlers` on a key collision. Both wrappers forward to the same
 * dispatcher while failing an identity check, so an identity test would
 * silently revoke the opt-out documented on `StreamPreemption`.
 */
function getRegisteredDefaultChatStreamHandler(
  context?: InvokeContext
): t.EventHandler | undefined {
  const handler = context?.handlerRegistry?.getHandler(
    GraphEvents.CHAT_MODEL_STREAM
  );
  return dispatchesChatModelStream(handler) ? handler : undefined;
}

function hasReasoningDetails(chunk: AIMessageChunk): boolean {
  const reasoningDetails = chunk.additional_kwargs.reasoning_details;
  return Array.isArray(reasoningDetails) && reasoningDetails.length > 0;
}

function removeOpenRouterFinalReasoningReplayContent({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: t.ProviderName;
}): AIMessageChunk {
  const content = getOpenRouterFinalReasoningContent({
    current,
    next,
    provider,
  });
  if (content == null || content === next.content) {
    return next;
  }

  return new AIMessageChunk(
    Object.assign({}, next, {
      content,
    })
  );
}

function getOpenRouterFinalReasoningContent({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: t.ProviderName;
}): string | undefined {
  if (
    provider !== Providers.OPENROUTER ||
    current == null ||
    !hasReasoningDetails(next) ||
    typeof current.content !== 'string' ||
    current.content === '' ||
    typeof next.content !== 'string' ||
    next.content === ''
  ) {
    return undefined;
  }
  if (!next.content.startsWith(current.content)) {
    return next.content;
  }
  return next.content.slice(current.content.length);
}

function removeReasoningDetails(
  additionalKwargs: AIMessageChunk['additional_kwargs']
): AIMessageChunk['additional_kwargs'] {
  return Object.fromEntries(
    Object.entries(additionalKwargs).filter(
      ([key]) => key !== 'reasoning_details'
    )
  );
}

function getStreamHandlingChunk({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: t.ProviderName;
}): AIMessageChunk | undefined {
  const content = getOpenRouterFinalReasoningContent({
    current,
    next,
    provider,
  });
  if (content == null) {
    return next;
  }
  if (content === '') {
    return undefined;
  }
  return new AIMessageChunk(
    Object.assign({}, next, {
      content,
      additional_kwargs: removeReasoningDetails(next.additional_kwargs),
    })
  );
}

/**
 * Best-effort output-token count for a sealed turn, used only when the
 * provider never got to send its usage chunk.
 */
function countSealedTokens(
  context: InvokeContext | undefined,
  metadata: Record<string, unknown> | undefined,
  messages: BaseMessage[]
): number | undefined {
  try {
    const counter = context?.getAgentContext(metadata).tokenCounter;
    if (counter == null) {
      return undefined;
    }
    let total = 0;
    for (const message of messages) {
      total += counter(message);
    }
    return total;
  } catch {
    return undefined;
  }
}

/**
 * Instruction overhead the provider processed but that never appears in the
 * message array: `createCallModel` pipes the model through
 * `agentContext.systemRunnable` and binds tool schemas AFTER `messages` is
 * formed, so the system prompt, dynamic instructions, summary and tool
 * schemas are all billed yet invisible here.
 *
 * Read per-node via `getAgentContext(metadata)` rather than the graph-level
 * accessor, which is hardcoded to `defaultAgentId` and would report the wrong
 * agent's overhead in a `MultiAgentGraph`.
 */
function sealedInstructionOverhead(
  context: InvokeContext | undefined,
  metadata: Record<string, unknown> | undefined
): number {
  try {
    const agentContext = context?.getAgentContext(metadata);
    return (
      agentContext?.resolvedInstructionOverhead ??
      agentContext?.instructionTokens ??
      0
    );
  } catch {
    return 0;
  }
}

/**
 * Best-effort usage for a turn the provider never got to bill us for.
 *
 * The prompt matters as much as the completion: the provider processed the
 * ENTIRE prompt — messages plus instruction overhead — before we sealed, and
 * every resume re-sends it, so under-counting input hides the expensive half
 * of a preempted run.
 *
 * ESTIMATE, NOT MEASUREMENT. Messages are counted with the host's tokenizer
 * rather than the provider's, and `toolSchemaTokens` applies a heuristic
 * multiplier. It is also an over-count on the fallback path, where
 * `tryFallbackProviders` builds a bare model with no `systemRunnable` pipe so
 * the system prompt genuinely is not sent. Accepted rather than threaded
 * through a flag: only the fallback-plus-seal combination is affected, and an
 * over-count is safer than the previous fabricated `input_tokens: 0`.
 *
 * Marked `estimated_usage` so calibration can refuse to learn from it — a
 * ratio derived from the same counter that produced the estimate is
 * self-consistent by construction and would drag a provider's real
 * calibration toward 1.0.
 */
function synthesizeSealedUsage(
  context: InvokeContext | undefined,
  chunk: AIMessageChunk,
  prompt: BaseMessage[],
  metadata: Record<string, unknown> | undefined
): void {
  if (chunk.usage_metadata != null) {
    return;
  }
  const outputTokens = countSealedTokens(context, metadata, [chunk]);
  if (outputTokens == null) {
    return;
  }
  const inputTokens =
    (countSealedTokens(context, metadata, prompt) ?? 0) +
    sealedInstructionOverhead(context, metadata);
  const usageMetadata = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
  };
  chunk.usage_metadata = usageMetadata;
  chunk.lc_kwargs.usage_metadata = usageMetadata;
  const responseMetadata = {
    ...chunk.response_metadata,
    estimated_usage: true,
  };
  chunk.response_metadata = responseMetadata;
  chunk.lc_kwargs.response_metadata = responseMetadata;
}

function getMessageText(chunk: AIMessageChunk): string {
  if (typeof chunk.content === 'string') {
    return chunk.content;
  }
  let text = '';
  for (const block of chunk.content) {
    if (block.type === ContentTypes.TEXT) {
      const value = block[ContentTypes.TEXT];
      if (typeof value === 'string') {
        text += value;
      }
    }
  }
  return text;
}

/**
 * Ends the real model run for a turn that was sealed mid-stream.
 *
 * Mandatory, not cosmetic. `@langchain/core`'s `_streamIterator` calls
 * `handleLLMEnd` after its try/catch with no `finally`, so breaking out of the
 * consumer's `for await` produces a *return* completion that fires neither
 * `handleLLMError` nor `handleLLMEnd`. The run would stay open in every
 * callback handler: the host records no usage — and since each seal re-sends
 * the whole prompt, N preemptions cost N unrecorded prompts — while LangSmith
 * and Langfuse hold a span that never closes.
 *
 * `runId` cannot be dictated from here (the bound runnable consumes
 * `config.runId` for its own run and hands the chat model a fresh one), but it
 * can be OBSERVED: the capture handler installed at the `model.stream` call
 * records it from `handleChatModelStart`, which fires before the first chunk.
 * Rebuilding the manager against that id closes the real run, and the host's
 * `on_chat_model_end` then arrives through the ordinary `streamEvents` path.
 *
 * Falls back to a custom-event dispatch if the id was never observed, so the
 * host still records usage even when the native close is unavailable.
 */
/**
 * Every callbacks source the real model run would compose beyond the per-call
 * config. `model` here is whatever `createCallModel` produced — with tools
 * that is `bindTools(...)`'s `RunnableBinding`, and a system runnable pipes a
 * `RunnableSequence` on top — while `clientOptions.callbacks` lives on the
 * chat model at the BOTTOM of that stack. Walks `bound` (bindings) and
 * `last`/`steps` (sequences), collecting each wrapper's own `callbacks` and
 * any binding-config callbacks along the way, since the binding merges its
 * config into the call before the chat model composes.
 */
function collectModelCallbackSources(model: unknown): Callbacks[] {
  const sources: Callbacks[] = [];
  const seen = new Set<unknown>();
  let current: unknown = model;
  while (current != null && typeof current === 'object' && !seen.has(current)) {
    seen.add(current);
    const wrapper = current as {
      callbacks?: Callbacks;
      config?: { callbacks?: Callbacks };
      bound?: unknown;
      last?: unknown;
      steps?: unknown[];
    };
    if (wrapper.callbacks != null) {
      sources.push(wrapper.callbacks);
    }
    if (wrapper.config?.callbacks != null) {
      sources.push(wrapper.config.callbacks);
    }
    current =
      wrapper.bound ??
      wrapper.last ??
      (Array.isArray(wrapper.steps)
        ? wrapper.steps[wrapper.steps.length - 1]
        : undefined);
  }
  return sources;
}

async function endSealedModelRun(
  context: InvokeContext | undefined,
  chunk: AIMessageChunk,
  prompt: BaseMessage[],
  llmRunId: string | undefined,
  config?: RunnableConfig,
  model?: t.ChatModel,
  /**
   * Shapes the close for a DISCARDED turn. A seal's output is the partial
   * answer it kept; a restart kept nothing, and marking it here is what keeps
   * the two restart routes reading alike in a trace — the aborted one is
   * relabelled through the tracing callback, and this is the manual twin.
   */
  llmOutput: Record<string, unknown> = {},
  /**
   * Set when the run was probably closed already, so a failed native close is
   * the expected outcome rather than a fault. The synthetic fallback below is
   * the real close in that case, and warning on every interrupt would bury the
   * failures that do matter.
   */
  nativeCloseOptional = false
): Promise<void> {
  const metadata = config?.metadata as Record<string, unknown> | undefined;
  synthesizeSealedUsage(context, chunk, prompt, metadata);
  if (llmRunId != null) {
    try {
      let callbackManager = await getCallbackManagerForConfig(config);
      /**
       * The real model run composes the per-call config's callbacks WITH the
       * model's own (`CallbackManager.configure(config.callbacks,
       * this.callbacks, …)` in `@langchain/core`'s base chat model), so a
       * handler supplied via `clientOptions.callbacks` received
       * `handleChatModelStart` for this run. Rebuilding from the config alone
       * would close the run for every handler EXCEPT those — leaving their
       * span open forever. Composed the same way the real run composes:
       * model callbacks appended non-inheritable, parent run id preserved by
       * `copy`, tracers deduped by `configure`.
       */
      for (const source of collectModelCallbackSources(model)) {
        callbackManager =
          CallbackManager.configure(callbackManager ?? undefined, source) ??
          callbackManager;
      }
      if (callbackManager != null) {
        const runManager = new CallbackManagerForLLMRun(
          llmRunId,
          callbackManager.handlers,
          callbackManager.inheritableHandlers,
          callbackManager.tags,
          callbackManager.inheritableTags,
          callbackManager.metadata,
          callbackManager.inheritableMetadata,
          callbackManager.getParentRunId()
        );
        const generation: ChatGeneration = {
          text: getMessageText(chunk),
          message: chunk,
        };
        await runManager.handleLLMEnd({
          generations: [[generation]],
          llmOutput,
        });
        return;
      }
    } catch (e) {
      /**
       * A sealed answer that reaches the user is worth more than a tidy
       * trace. Fall through to the custom event rather than failing the run.
       */
      if (!nativeCloseOptional) {
        // eslint-disable-next-line no-console
        console.warn(
          '[attemptInvoke] Native close of the sealed model run failed; falling back to a custom event:',
          e instanceof Error ? e.message : e
        );
      }
    }
  }
  await safeDispatchCustomEvent(
    GraphEvents.CHAT_MODEL_END,
    { output: chunk },
    config
  );
}

function appendStreamChunk({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: t.ProviderName;
}): AIMessageChunk {
  if (current == null) {
    return next;
  }
  return concat(
    current,
    removeOpenRouterFinalReasoningReplayContent({ current, next, provider })
  );
}

/**
 * Invokes a chat model with the given messages, handling both streaming and
 * non-streaming paths.
 *
 * By default, stream chunks are processed through a `ChatModelStreamHandler`
 * that dispatches run steps (MESSAGE_CREATION, TOOL_CALLS) for the graph.
 * Pass an `onChunk` callback to override this with custom chunk processing
 * (e.g. summarization delta events).
 */
interface AttemptInvokeCommonParams {
  context?: InvokeContext;
  onChunk?: OnChunk;
  /**
   * The agent lane this attempt belongs to, as the model node names it. Only
   * read to record a discarded turn, which the node cannot otherwise detect —
   * a discard returns no message to carry `response_metadata.preempted`.
   *
   * Keyed rather than global for the same reason `pendingPreemptReturn` is:
   * `MultiAgentGraph` routes every parallel agent through ONE graph instance,
   * so a shared flag would let whichever lane finished first consume another
   * lane's boundary — and inject that lane's words into the wrong turn.
   */
  preemptAgentId?: string;
  /** Accounting owner for callers that deliberately pass no `context`
   * (summarization) — used ONLY for the attempt's accounting lease, never
   * for charge claims. */
  streamLimitState?: StreamLimitState;
}

type AttemptInvokeParams = AttemptInvokeCommonParams &
  (
    | {
        request: PreparedProviderRequest;
        model?: never;
        messages?: never;
        provider?: never;
      }
    | {
        request?: never;
        model: t.ChatModel;
        messages: BaseMessage[];
        provider: t.ProviderName;
      }
  );

function resolveAttemptProvider(params: AttemptInvokeParams): t.ProviderName {
  if (params.request != null) {
    return params.request.provider;
  }
  return params.provider;
}

function resolveAttemptRequest(
  params: AttemptInvokeParams,
  config: RunnableConfig
): PreparedProviderRequest {
  if (params.request != null) {
    assertPreparedProviderRequestFor(
      params.request,
      params.request.model,
      params.request.provider,
      config
    );
    return params.request;
  }
  return prepareProviderRequest({
    model: params.model,
    messages: params.messages,
    provider: params.provider,
    context: params.context,
    config,
  });
}

/**
 * One model attempt. Stamps the attempt identity into callback metadata
 * (see the generation-key notes in `streamLimits.ts`), leases the attempt's
 * accounting for its LIFETIME, and releases both from `finally`: retention
 * must follow the attempt — a cancellation-ignoring straggler keeps its
 * original budget no matter how many runs start and reset while it drains.
 */
export async function attemptInvoke(
  params: AttemptInvokeParams,
  config?: RunnableConfig
): Promise<Partial<t.BaseGraphState>> {
  const provider = resolveAttemptProvider(params);
  const providerStampedConfig: RunnableConfig = {
    ...config,
    metadata: {
      ...(config?.metadata ?? {}),
      [Constants.INVOKED_PROVIDER]: provider,
      provider,
      resolvedProvider: provider,
    },
  };
  const request = resolveAttemptRequest(params, providerStampedConfig);
  const configuredModel =
    providerStampedConfig.metadata?.[Constants.INVOKED_MODEL];
  const modelId =
    request.modelId ??
    (typeof configuredModel === 'string' ? configuredModel : undefined);
  const stampedConfig: RunnableConfig = {
    ...providerStampedConfig,
    metadata: {
      ...providerStampedConfig.metadata,
      ...(modelId == null
        ? {}
        : {
          [Constants.INVOKED_MODEL]: modelId,
          model: modelId,
        }),
      /**
       * One `attemptInvoke` call is one model attempt; primary, fallback,
       * and retry attempts within a node otherwise share the same langgraph
       * metadata, so without a unique attempt stamp a fallback re-streaming
       * a tool call from scratch would be charged the failed primary's
       * partial bytes (or a same-named sibling fallback's) and could
       * falsely trip the stream limits. The stamp rides the same metadata
       * rebuild that already attributes the serving provider.
       */
      [STREAM_LIMIT_ATTEMPT_KEY]: ++streamLimitAttemptSeq,
    },
  };
  const rawLeaseTarget = params.context ?? params.streamLimitState;
  /** No lease when no guard can fire: the lease only protects accounting
   * entries, and fully disabled guards must allocate no bookkeeping at
   * all — per-attempt included. */
  const leaseTarget =
    rawLeaseTarget != null && streamLimitAccountingEnabled(rawLeaseTarget)
      ? rawLeaseTarget
      : undefined;
  const generationKey =
    leaseTarget != null
      ? resolveGenerationKey(stampedConfig.metadata as Record<string, unknown>)
      : undefined;
  if (leaseTarget != null && generationKey != null) {
    registerActiveStreamLimitGeneration(leaseTarget, generationKey);
  }
  const prepared =
    params.context?.eagerEventToolExecution?.enabled === true
      ? params.context.preparedSubagents
      : undefined;
  const preparedAttempt = resolveGenerationKey(stampedConfig.metadata);
  prepared?.begin(preparedAttempt, stampedConfig);
  try {
    const result = await attemptInvokeBody(
      {
        request,
        context: params.context,
        onChunk: params.onChunk,
        preemptAgentId: params.preemptAgentId,
      },
      stampedConfig
    );
    const calls =
      result.messages?.flatMap((message) =>
        'tool_calls' in message
          ? ((message.tool_calls as ToolCall[] | undefined) ?? [])
          : []
      ) ?? [];
    prepared?.finish(preparedAttempt, calls);
    return result;
  } catch (error) {
    prepared?.finish(preparedAttempt, undefined, error);
    throw error;
  } finally {
    const chunks = params.context?.eagerEventToolCallChunks;
    if (prepared != null && chunks != null) {
      const prefix = `prepared:${preparedAttempt}\u0000`;
      for (const key of chunks.keys()) {
        if (key.startsWith(prefix)) {
          chunks.delete(key);
        }
      }
    }
    if (leaseTarget != null && generationKey != null) {
      releaseStreamLimitGeneration(leaseTarget, generationKey);
    }
  }
}

async function attemptInvokeBody(
  {
    request,
    context,
    onChunk,
    preemptAgentId,
  }: Pick<
    AttemptInvokeCommonParams,
    'context' | 'onChunk' | 'preemptAgentId'
  > & {
    request: PreparedProviderRequest;
  },
  config: RunnableConfig
): Promise<Partial<t.BaseGraphState>> {
  const { model, messages: messagesForProvider, provider } = request;
  const projectionInvariantMode =
    resolveProviderMessageProjectionInvariantMode();
  let sealedRunId: string | undefined;
  let invocationConfig = config;
  const captureModelRunId = model.stream != null && context?.preemption != null;
  if (projectionInvariantMode !== 'off' || captureModelRunId) {
    invocationConfig = withModelStartHandler({
      config,
      mode: projectionInvariantMode,
      provider,
      captureRunId: captureModelRunId
        ? (runId: string): void => {
          sealedRunId ??= runId;
        }
        : undefined,
    });
  }

  /**
   * Stamp the provider that is ACTUALLY serving this invocation onto the
   * callback metadata. `attemptInvoke` is the single funnel for primary,
   * fallback, and summarization model calls, so consumers that need
   * provider attribution per call (the subagent usage-capture handler)
   * read this key instead of trusting static agent config — which is
   * wrong for fallback-served calls — or `ls_provider` — which derived
   * providers inherit from their base class.
   */
  if (model.stream) {
    /**
     * Observed, not dictated. `handleChatModelStart` fires with the chat
     * model's real run id before the first chunk, which is the only way to
     * name the run a seal has to close — pinning `config.runId` does not
     * survive the bound runnable. The same handler owns the opt-in projection
     * invariant so enabled diagnostics do not stack a second model callback.
     */
    let finalChunk: AIMessageChunk | undefined;
    let preemptAction: PreemptAction = 'none';
    const registeredStreamHandler =
      getRegisteredDefaultChatStreamHandler(context);
    /**
     * The wake channel is armed for the seal-capable branch only. The other
     * two consume through readers that lag the accumulation — the same reason
     * the per-chunk poll lives in that branch alone — and an `onChunk`
     * consumer owns the stream outright.
     *
     * An unnamed lane disarms it too. A caller that supplies a preemption
     * source but no `preemptAgentId` cannot have its discard routed back to
     * the right model node, so it keeps today's behavior — the request waits
     * for a boundary — rather than ending a turn nothing will resume.
     */
    const restartLane =
      onChunk == null &&
      registeredStreamHandler == null &&
      context?.preemption?.subscribe != null
        ? preemptAgentId
        : undefined;
    const restartController =
      restartLane == null ? undefined : new AbortController();
    /**
     * Composed, never replaced: the run's own signal must keep tearing this
     * stream down. `composeAbortSignals` collapses back to a single signal
     * when there is nothing to compose.
     */
    const streamConfig =
      restartController == null
        ? invocationConfig
        : {
          ...invocationConfig,
          signal: composeAbortSignals(
            invocationConfig.signal,
            restartController.signal
          ),
        };
    const restartGraceMs = resolveRestartGraceMs(
      context?.preemption?.restartGraceMs
    );
    /**
     * When THIS attempt first saw the request, which is what the grace window
     * is measured against. Recorded on first observation rather than on arm:
     * the host's flag is level-triggered and may already have been true for a
     * previous attempt, and a retry inheriting an aged request would discard
     * its very first chunk.
     */
    let preemptRequestedAt: number | undefined;
    let restartGraceTimer: ReturnType<typeof setTimeout> | undefined;
    /** True while a chunk is being handed to the stream handler and has not
     *  yet been folded into `finalChunk`. See the guard in
     *  {@link evaluatePreemptRestart}. */
    let dispatchingChunk = false;
    /** Read through a call so the check sees the value the wake handler
     *  writes: control-flow analysis at the loop head still holds the
     *  initializer, and would narrow the comparison away as unreachable. */
    const restartDecided = (): boolean => preemptAction === 'restart';
    /**
     * How the stream ended when a restart was chosen, because each way leaves
     * the model run in a different state and needs a different close:
     *  - `aborted`: LangChain closed it through its error path, where the
     *    marker relabels it and supplies the discarded attempt's usage.
     *  - `broke`: the iterator was closed early, which fires NEITHER end nor
     *    error callbacks, so this is the one route that must close natively.
     *  - `exhausted`: the stream finished on its own, so LangChain already
     *    emitted its end. Closing again would warn and change nothing — and
     *    the generation is honestly an ordinary completed one, since nothing
     *    was torn down; only the turn built on it is being discarded.
     */
    let restartRoute: 'aborted' | 'broke' | 'exhausted' | undefined;
    /**
     * Cleared when the attempt returns. A host may deliver its wake through a
     * queue or a message bus, so one can already be in flight when the
     * unsubscribe runs — and a late claim would take the seal slot, abort a
     * finished controller, and never reach a boundary to release it, blocking
     * every later preemption in the run.
     */
    let attemptActive = true;
    /**
     * Only an exhausted stream leaves nothing for this attempt to close: it
     * emitted its own end. Both other routes attempt the native close, because
     * whether the run is still open is NOT observable from here — an adapter
     * that ignores the abort keeps it open, while one that honors it (or that
     * simply hands over a chunk buffered before the abort landed) has already
     * closed it through the error path. `endSealedModelRun` degrades to the
     * synthetic event when the run is gone, which is what makes both right.
     */
    const restartNeedsNativeClose = (): boolean => restartRoute !== 'exhausted';
    /**
     * An aborted run is EXPECTED to be closed already, so a failed native
     * close is the normal outcome there and must not warn on every interrupt.
     * A broken iterator fired no callback at all, so a failure is worth
     * hearing about.
     */
    const restartCloseMayHaveHappened = (): boolean =>
      restartRoute === 'aborted';

    /**
     * The wake is a hint; this is where the request is actually read. The
     * shape is judged at the instant we look at it, and the teardown that
     * follows is what makes any later chunk irrelevant — so a wake arriving
     * once an answer has started, or one that loses the shared seal slot,
     * leaves the stream untouched and waits for the ordinary boundary.
     *
     * A `seal` verdict is deliberately ignored here. Sealing KEEPS the
     * accumulated turn, and only the chunk loop can hand it over intact; this
     * path exists solely to end turns that have nothing to hand over.
     *
     * The self-rescheduling timer is what covers the silent window. Nothing
     * else will look again — a provider that has gone quiet produces no chunk
     * to poll on — so without it a request arriving mid-grace would wait out
     * the whole turn, which is the stall this path exists to remove.
     */
    const evaluatePreemptRestart = (): boolean => {
      /** Reports whether a restart IS decided, not whether this call decided
       *  it. A synchronous wake during `subscribe` can settle the action
       *  before the pre-call read runs, and a caller that only learned "I did
       *  not convert" would go on to issue a provider request against an
       *  already-aborted signal. */
      if (!attemptActive) {
        return false;
      }
      if (restartController == null || preemptAction !== 'none') {
        return preemptAction === 'restart';
      }
      if (context?.shouldPreemptStream() !== true) {
        /** The request was withdrawn. Forget when it arrived, or a NEW request
         *  later in this same attempt would inherit the old clock and convert
         *  without ever getting its grace. */
        preemptRequestedAt = undefined;
        return false;
      }
      /** A chunk is mid-dispatch to the host, so `finalChunk` describes only
       *  the chunks BEFORE it. Judging the turn now could read a text chunk
       *  the host has already been shown as an empty turn and discard it. The
       *  per-chunk poll runs immediately after the append with the complete
       *  accumulation, so nothing is lost by declining here. */
      if (dispatchingChunk) {
        return false;
      }
      preemptRequestedAt ??= Date.now();
      const requestAgeMs = Date.now() - preemptRequestedAt;
      const action = resolvePreemptAction({
        chunk: finalChunk,
        requestAgeMs,
        graceMs: restartGraceMs,
      });
      if (action !== 'restart') {
        if (
          action === 'none' &&
          restartGraceTimer == null &&
          canRestartPreempt(finalChunk)
        ) {
          /**
           * Scheduled for what is LEFT of the window, not a fresh one: the
           * clock starts when the request is first seen, so a wake arriving
           * mid-window must not push the conversion further out than a wake
           * arriving at its start.
           *
           * The handle is released as the timer fires so a look that changes
           * nothing — the shape moved, the host disarmed and re-armed — can
           * still schedule the next one.
           */
          restartGraceTimer = setTimeout(
            () => {
              restartGraceTimer = undefined;
              /**
               * Contained, because this frame has no invocation to reject
               * into: a host predicate that throws here would surface as an
               * uncaught exception and can take the process down, while the
               * same throw from the per-chunk poll is held by the attempt's
               * promise. The cost of swallowing is one missed look, and a
               * predicate that throws has no request to honor anyway.
               */
              try {
                evaluatePreemptRestart();
              } catch {
                /** empty */
              }
            },
            /**
             * Clamped, then CHAINED: a delay past Node's ceiling silently
             * becomes 1ms, and this timer reschedules itself, so an
             * out-of-range grace would spin at ~1ms for the life of the
             * stream. Each firing re-reads the real age, so the requested
             * deadline survives being reached in several hops.
             */
            Math.min(MAX_TIMEOUT_MS, Math.max(0, restartGraceMs - requestAgeMs))
          );
          /** The grace is a fallback, never a reason to hold the process
           *  open: a run that ends before the window elapses must not be kept
           *  alive by a timer whose only job is to look again. */
          restartGraceTimer.unref();
        }
        return false;
      }
      if (!context.claimPreemptRestart()) {
        return false;
      }
      preemptAction = 'restart';
      /**
       * Recorded BEFORE the abort: the adapter's cancellation error can reach
       * the tracing callback synchronously, and a run marked afterwards would
       * already have closed as a failure.
       *
       * `sealedRunId` being set is also what proves the request went OUT, so
       * charging the prompt against it is honest. Usage is resolved here, onto
       * the same chunk the discard reports later — the run closes through the
       * error path, which carries no output, so a marker without it would make
       * every restart look free. `synthesizeSealedUsage` no-ops when the
       * provider already streamed usage, and again when the discard path runs,
       * so this is one computation, not two.
       */
      if (sealedRunId != null) {
        finalChunk ??= new AIMessageChunk({ content: '' });
        synthesizeSealedUsage(
          context,
          finalChunk,
          messagesForProvider,
          config.metadata as Record<string, unknown> | undefined
        );
        notePreemptRestartedRun(sealedRunId, finalChunk);
      }
      restartRoute = 'aborted';
      restartController.abort();
      return true;
    };
    const unsubscribeWake =
      restartController == null
        ? undefined
        : context?.preemption?.subscribe?.(evaluatePreemptRestart);
    /** A sibling's trip aborts the composed signal, but an adapter that
     * ignores cancellation keeps yielding — and text-only chunks with the
     * event cap off never throw in enforcement, so nothing else would stop
     * the drain. Checked on every yielded chunk in all three loops;
     * throwing closes the iterator and tears down the provider stream. */
    const throwIfBreakerTripped = (): void => {
      const signal = config.signal;
      if (
        signal?.aborted === true &&
        (signal.reason instanceof StreamLimitExceededError ||
          signal.reason instanceof PreparedSubagentError)
      ) {
        throw signal.reason;
      }
    };

    try {
      /**
       * Subscribing is not enough on its own. `shouldPreempt` is
       * level-triggered, so a request armed BEFORE this attempt began — during
       * setup, or on a previous attempt a fallback replaced — is already true
       * and will never produce another wake. Reading it once here is the
       * difference between honoring that request and waiting out the turn.
       *
       * Read before the provider request is issued so the grace clock starts
       * from the attempt's own beginning rather than from its first chunk. A
       * host that set `restartGraceMs` to zero skips the call entirely here,
       * having streamed nothing and opened no model run — which is why the
       * close below is conditional.
       */
      if (!evaluatePreemptRestart()) {
        const stream = await model.stream(messagesForProvider, streamConfig);
        if (onChunk) {
          const attemptMetadata = config.metadata as
            | Record<string, unknown>
            | undefined;
          for await (const rawChunk of stream) {
            throwIfBreakerTripped();
            const chunk = snapshotValidatedModelChunk(rawChunk);
            /** An onChunk consumer replaces the stream handler entirely, so
             * stream limits are enforced here for every such caller — public
             * package consumers get no other accounting. The internal
             * summarization onChunk charges producer-side itself and passes no
             * context, precisely so this claim and its own never stack. */
            if (context != null) {
              enforceStreamLimitsForWireChunk({
                graph: context,
                metadata: attemptMetadata,
                chunk,
              });
            }
            await onChunk(chunk, attemptMetadata);
            finalChunk = appendStreamChunk({
              current: finalChunk,
              next: chunk,
              provider,
            });
          }
        } else if (registeredStreamHandler == null) {
          const metadata = config.metadata as
            | Record<string, unknown>
            | undefined;
          const streamHandler = new ChatModelStreamHandler();
          for await (const rawChunk of stream) {
            throwIfBreakerTripped();
            const chunk = snapshotValidatedModelChunk(rawChunk);
            /**
             * The decision is final, so stop consuming here rather than
             * trusting the adapter to honor the abort. An adapter that ignores
             * cancellation — or one whose iterator hands over a chunk buffered
             * before the abort landed — would otherwise keep dispatching text
             * the host is about to be told was discarded, and in the ignoring
             * case would hold the boundary until the whole response finished:
             * exactly the stall a restart exists to end.
             */
            if (restartDecided()) {
              break;
            }
            const handlingChunk = getStreamHandlingChunk({
              current: finalChunk,
              next: chunk,
              provider,
            });
            if (handlingChunk != null) {
              dispatchingChunk = true;
              try {
                await streamHandler.handle(
                  GraphEvents.CHAT_MODEL_STREAM,
                  { chunk: handlingChunk },
                  metadata,
                  context
                );
              } finally {
                dispatchingChunk = false;
              }
            } else if (context != null) {
              /**
               * A replay-skipped chunk yields no handling chunk, and in this
               * local branch no `streamEvents` consumer judges the wire event
               * either — yet a cumulative OpenRouter replay can still carry
               * `tool_call_chunks` or complete `tool_calls` that are appended
               * below. Charge the full limits (event budget and argument bytes)
               * directly so neither cap can be bypassed. Consumer side: the
               * local handler.handle call above claims as consumer, and one
               * reused chunk object can alternate between these two arms.
               */
              enforceStreamLimitsForWireChunk({
                graph: context,
                metadata,
                chunk,
                side: 'consumer',
              });
            }
            finalChunk = appendStreamChunk({
              current: finalChunk,
              next: chunk,
              provider,
            });
            /**
             * Only this loop may seal. The registered-handler branch below
             * dispatches through `run.ts`'s decoupled `streamEvents` consumer,
             * which can lag the accumulated chunk — sealing there would let the
             * host index a content part the user has not been shown yet.
             */
            /**
             * Cheap poll first, shape check second, budget claim last. The claim
             * is what makes this safe under a parallel `MultiAgentGraph`: several
             * agents share one graph and can each see the poll as true, but only
             * one can take the slot, and a chunk that cannot seal never spends it.
             *
             * `resolvePreemptAction` prefers a seal wherever one is available,
             * so a turn that already produced an answer keeps it. `restart` is
             * reached only for an accumulation holding nothing but reasoning —
             * the case a seal can never accept, and the reason an interrupt
             * armed during a long thinking stretch used to wait for the whole
             * turn.
             */
            if (context?.shouldPreemptStream() !== true) {
              /** Withdrawn: forget the clock, so a later request in this same
               *  attempt still gets its own grace. */
              preemptRequestedAt = undefined;
            } else {
              preemptRequestedAt ??= Date.now();
              const action = resolvePreemptAction({
                chunk: finalChunk,
                requestAgeMs: Date.now() - preemptRequestedAt,
                graceMs: restartGraceMs,
              });
              /**
               * A restart needs the lane that names where its boundary is
               * owed. Without one — a host still on the seal-only contract,
               * which supplies no `subscribe` — the discard would return no
               * message AND record no lane, so the node would dispatch no
               * boundary, inject nothing, and end the turn empty with the
               * steer still queued. Such a host also never opted into having
               * its turns discarded, so its request waits for a seal exactly
               * as it does today.
               */
              const claimed =
                action === 'seal'
                  ? context.claimPreemptSeal()
                  : action === 'restart' &&
                    restartLane != null &&
                    context.claimPreemptRestart();
              if (claimed) {
                preemptAction = action;
                if (action === 'restart') {
                  restartRoute = 'broke';
                }
                break;
              }
            }
          }
        } else {
          const metadata = config.metadata as
            | Record<string, unknown>
            | undefined;
          /**
           * The original wire chunk still reaches the registered handler through
           * `streamEvents` (where the late-reasoning skip discards it AFTER the
           * event guard counts it), so this inline re-dispatch of the transformed
           * chunk is marked to not consume a second event-budget slot. Allocated
           * once per attempt, only when a transformation occurs.
           */
          let redispatchMetadata: Record<string, unknown> | undefined;
          for await (const rawChunk of stream) {
            throwIfBreakerTripped();
            const chunk = snapshotValidatedModelChunk(rawChunk);
            /**
             * Charged synchronously, ahead of the decoupled `streamEvents`
             * reader that will echo this same chunk to the registered handler:
             * a lagging reader would otherwise let an oversized complete call
             * return to LangGraph and reach ToolNode before the queued handler
             * throws. The chunk is marked so the echo skips accounting.
             */
            if (context != null) {
              enforceStreamLimitsForWireChunk({
                graph: context,
                metadata,
                chunk,
              });
            }
            const handlingChunk = getStreamHandlingChunk({
              current: finalChunk,
              next: chunk,
              provider,
            });
            if (handlingChunk != null && handlingChunk !== chunk) {
              redispatchMetadata ??= {
                ...(metadata ?? {}),
                [STREAM_LIMIT_REDISPATCH_KEY]: true,
              };
              await registeredStreamHandler.handle(
                GraphEvents.CHAT_MODEL_STREAM,
                { chunk: handlingChunk },
                redispatchMetadata,
                context
              );
            }
            finalChunk = appendStreamChunk({
              current: finalChunk,
              next: chunk,
              provider,
            });
          }
        }
        /**
         * The stream reached its natural end while a request was still armed
         * and the turn still holds nothing worth keeping. The grace exists to
         * avoid stealing a seal that was about to become possible — and the
         * final shape is proof none was: no text ever arrived. Converting here
         * spends no extra provider call (the stream is over) and spares the
         * host a reasoning-only turn followed by its own steer, which is the
         * shape the seal gate refuses to build in the first place.
         */
        if (
          preemptAction === 'none' &&
          restartLane != null &&
          context?.shouldPreemptStream() === true &&
          canRestartPreempt(finalChunk) &&
          context.claimPreemptRestart()
        ) {
          preemptAction = 'restart';
          restartRoute = 'exhausted';
        }
      }
    } catch (error) {
      /**
       * A restart tears the provider stream down mid-flight, which surfaces
       * as the composed signal's abort — from the iteration, or from stream
       * creation when the request was already outstanding. Swallowed only when
       * THIS attempt asked for it: `preemptAction` is set immediately before
       * the abort and by nothing else, so a run-level abort, a tripped stream
       * limit and every provider error still propagate.
       */
      const ownAbort =
        restartRoute === 'aborted' &&
        !(error instanceof StreamLimitExceededError) &&
        !(error instanceof PreparedSubagentError) &&
        config.signal?.aborted !== true;
      if (!ownAbort) {
        throw error;
      }
    } finally {
      attemptActive = false;
      unsubscribeWake?.();
      clearTimeout(restartGraceTimer);
    }

    if (providerUsesManualToolStream(provider)) {
      finalChunk = modifyDeltaProperties(
        getManualToolStreamNormalizationProvider(provider),
        finalChunk
      );
    }

    if (preemptAction === 'restart') {
      /**
       * The turn is thrown away, but a model run that OPENED still has to be
       * closed:
       * its callback span is open, and a host that renders from the stream
       * has already drawn the reasoning that is about to vanish. The synthetic
       * end carries `preemptDiscarded` so both can tell this apart from a seal
       * — a seal's content survives into the next prompt, a discard's does
       * not, and a host that keeps rendering it would show the user words the
       * model no longer has.
       *
       * Usage is still synthesized from whatever accumulated. Those reasoning
       * tokens were spent and billed; dropping them from the report would make
       * an interrupted turn look free.
       *
       * Skipped entirely when the request never went out — a preempt already
       * outstanding when the attempt began short-circuits above the provider
       * call, so there is no open span and no stream for a host to unwind, and
       * a synthetic end would announce a run that never started.
       */
      if (restartRoute !== 'exhausted') {
        /**
         * The step really was cut short, so it closes `cancelled` before the
         * model-end below can close it `completed`. An exhausted turn is left
         * alone: its stream reached its own end, and the step describing it is
         * already closed and honest — only the turn built on it is discarded.
         */
        await context?.cancelOpenMessageStep(
          config.metadata as Record<string, unknown> | undefined
        );
      }
      if (finalChunk != null || sealedRunId != null) {
        const discardedChunk =
          finalChunk ?? new AIMessageChunk({ content: '' });
        const responseMetadata = {
          ...discardedChunk.response_metadata,
          preempted: true,
          preemptDiscarded: true,
        };
        discardedChunk.response_metadata = responseMetadata;
        discardedChunk.lc_kwargs.response_metadata = responseMetadata;
        if (restartRoute === 'exhausted') {
          /**
           * The stream finished on its own, so the host has ALREADY had this
           * turn's model-end, with its usage. Replaying it through
           * `endSealedModelRun` would charge the same tokens twice in any host
           * accumulating usage from that event. What the host still needs is
           * the one thing the natural end could not say — that the turn it
           * just completed is being thrown away — so that notification carries
           * the marker and nothing else.
           */
          await safeDispatchCustomEvent(
            GraphEvents.CHAT_MODEL_END,
            {
              output: new AIMessageChunk({
                content: '',
                response_metadata: responseMetadata,
              }),
            },
            config
          );
        } else {
          /**
           * Both cut-short routes come here and both pass the run id: see
           * {@link restartNeedsNativeClose} for why the run's state is not
           * observable and the close has to work either way. An aborted run
           * that WAS already closed falls through to the synthetic event,
           * where the marker relabels it and supplies this usage.
           */
          await endSealedModelRun(
            context,
            discardedChunk,
            messagesForProvider,
            restartNeedsNativeClose() ? sealedRunId : undefined,
            config,
            model,
            PREEMPT_RESTART_CONTROL_FLOW,
            restartCloseMayHaveHappened()
          );
        }
      }
      /** Non-null on every path that can reach here: it arms the controller
       *  the wake path needs, and the per-chunk path refuses a restart
       *  without it. */
      if (restartLane != null) {
        context?.notePreemptRestart(restartLane);
      }
      /**
       * No message reaches graph state, so the injected user turn lands
       * directly after the previous one — which is what makes this safe on
       * every provider: adjacent user turns are native on Anthropic, OpenAI
       * and Gemini, and normalized by `coalesceAdjacentUserTurns` for the
       * strict-alternation providers. A seal needs a non-empty assistant turn
       * precisely because it leaves one behind; a discard leaves none.
       */
      return { messages: [] };
    }

    if (preemptAction === 'seal' && finalChunk != null) {
      const responseMetadata = {
        ...finalChunk.response_metadata,
        preempted: true,
      };
      finalChunk.response_metadata = responseMetadata;
      finalChunk.lc_kwargs.response_metadata = responseMetadata;
      await endSealedModelRun(
        context,
        finalChunk,
        messagesForProvider,
        sealedRunId,
        config,
        model
      );
    }

    if (finalChunk != null) detachValidatedModelToolCalls(finalChunk);
    if ((finalChunk?.tool_calls?.length ?? 0) > 0) {
      finalChunk!.tool_calls = finalChunk!.tool_calls?.filter(
        (tool_call: ToolCall) => !!tool_call.name
      );
    }

    assertNotTruncatedToolCall(finalChunk, provider);
    return { messages: [finalChunk as AIMessageChunk] };
  }

  const finalMessage = snapshotValidatedModelChunk(
    await model.invoke(messagesForProvider, invocationConfig),
    false
  );
  if ((finalMessage.tool_calls?.length ?? 0) > 0) {
    finalMessage.tool_calls = finalMessage.tool_calls?.filter(
      (tool_call: ToolCall) => !!tool_call.name
    );
  }
  assertNotTruncatedToolCall(finalMessage, provider);
  return { messages: [finalMessage] };
}

/**
 * Identifies which fallback produced an error, so a caller planning a
 * recovery can reason about the client that actually failed rather than the
 * primary's configuration — their context windows and output allowances
 * differ, which is the whole reason a fallback exists.
 */
export interface FallbackErrorContext {
  provider: t.ProviderName;
  clientOptions?: t.ClientOptions;
  maxContextTokens?: number;
}

export interface FallbackOverflowCandidate {
  error: unknown;
  context: FallbackErrorContext;
}

const fallbackErrorContexts = new WeakMap<object, FallbackErrorContext>();
const fallbackOverflowCandidates = new WeakMap<
  object,
  FallbackOverflowCandidate[]
>();

function attachFallbackErrorContext(
  error: unknown,
  fallbackContext: FallbackErrorContext
): void {
  if (typeof error !== 'object' || error === null) {
    return;
  }
  fallbackErrorContexts.set(error, fallbackContext);
}

/** Reads back the fallback attribution attached by `tryFallbackProviders`. */
export function getFallbackErrorContext(
  error: unknown
): FallbackErrorContext | undefined {
  if (typeof error !== 'object' || error === null) {
    return undefined;
  }
  return fallbackErrorContexts.get(error);
}

/** Returns every fallback overflow retained from an exhausted provider chain. */
export function getFallbackOverflowCandidates(
  error: unknown
): FallbackOverflowCandidate[] {
  if (typeof error !== 'object' || error === null) {
    return [];
  }
  return [...(fallbackOverflowCandidates.get(error) ?? [])];
}

/**
 * Attempts each fallback provider in order until one succeeds.
 *
 * When every fallback fails, a context overflow among them is thrown in
 * preference to whichever failure happened to come last. An overflow is the
 * one failure the caller can act on — it compacts and retries — and losing it
 * behind a later unrelated error would surface a dead end instead. Ordinary
 * failures still throw last-error-wins.
 */
export async function tryFallbackProviders({
  fallbacks,
  tools,
  messages,
  config,
  primaryError,
  context,
  onChunk,
  streamLimitState,
  preemptAgentId,
  overflowContext,
  prepareProviderRequest: prepareFallbackRequest,
  prepareProviderMessages,
}: {
  fallbacks: t.FallbackConfig[];
  tools?: t.GraphTools;
  messages: BaseMessage[];
  config?: RunnableConfig;
  primaryError: unknown;
  context?: InvokeContext;
  onChunk?: OnChunk;
  /** Accounting-lease owner forwarded to each fallback attempt (see
   * `AttemptInvokeParams.streamLimitState`). */
  streamLimitState?: StreamLimitState;
  /** Forwarded so a fallback-served attempt records a discarded turn in the
   * SAME lane the primary would have (see
   * `AttemptInvokeParams.preemptAgentId`). Dropping it here would leave the
   * node's boundary undispatched and the lane holding an empty turn. */
  preemptAgentId?: string;
  /**
   * Prompt-size corroboration for signatures that are not self-describing.
   * Vertex AI's overflow is a bare `400` with no reason, so without this a
   * fallback that overflows is indistinguishable from any other 400 and would
   * be dropped in favour of whichever failure came last.
   */
  overflowContext?: ContextOverflowContext;
  /**
   * Optional exact-payload preparation used by Graph. The returned request is
   * measured and sent without another provider projection.
   */
  prepareProviderRequest?: (input: {
    model: t.ChatModel;
    messages: BaseMessage[];
    provider: t.ProviderName;
    clientOptions?: t.ClientOptions;
    maxContextTokens?: number;
    config?: RunnableConfig;
  }) => PreparedProviderRequest | Promise<PreparedProviderRequest>;
  /** @deprecated Return a `PreparedProviderRequest` instead. */
  prepareProviderMessages?: (input: {
    model: t.ChatModel;
    messages: BaseMessage[];
    provider: t.ProviderName;
    clientOptions?: t.ClientOptions;
    maxContextTokens?: number;
    config?: RunnableConfig;
  }) => BaseMessage[] | Promise<BaseMessage[]>;
}): Promise<Partial<t.BaseGraphState> | undefined> {
  const isOverflow = (
    error: unknown,
    contextOverride = overflowContext
  ): boolean => getContextOverflowInfo(error, contextOverride) != null;
  let lastError: unknown = primaryError;
  /**
   * Tracked apart from the primary's overflow. A caller reaching this
   * function with an overflowing primary has already failed to recover from
   * it, so a fallback overflow — which may sit against a different window and
   * output allowance — is the more useful of the two to surface.
   */
  const overflowCandidates: FallbackOverflowCandidate[] = [];
  const primaryOverflowError: unknown = isOverflow(primaryError)
    ? primaryError
    : undefined;
  for (const fb of fallbacks) {
    try {
      const fbModel = initializeModel({
        provider: fb.provider,
        clientOptions: fb.clientOptions,
        tools,
      });
      /**
       * Stamp the fallback's configured model onto callback metadata so
       * per-call attribution (subagent usage capture) doesn't fall back to
       * the PRIMARY config's model when the provider reports no
       * `ls_model_name`. The serving provider is stamped uniformly by
       * `attemptInvoke` (`INVOKED_PROVIDER`).
       */
      const fbModelName = resolveClientOptionsModel(fb.clientOptions);
      const fbConfig: RunnableConfig | undefined =
        fbModelName == null
          ? config
          : {
            ...config,
            metadata: {
              ...(config?.metadata ?? {}),
              [Constants.INVOKED_MODEL]: fbModelName,
            },
          };
      const preparationInput = {
        model: fbModel as t.ChatModel,
        messages,
        provider: fb.provider,
        clientOptions: fb.clientOptions,
        maxContextTokens: fb.maxContextTokens,
        config: fbConfig,
      };
      const preparedRequest = await prepareFallbackRequest?.(preparationInput);
      if (preparedRequest != null) {
        assertPreparedProviderRequestFor(
          preparedRequest,
          fbModel as t.ChatModel,
          fb.provider,
          fbConfig
        );
      }
      let fallbackMessages = messages;
      if (preparedRequest == null) {
        fallbackMessages =
          (await prepareProviderMessages?.({
            model: fbModel as t.ChatModel,
            messages,
            provider: fb.provider,
            clientOptions: fb.clientOptions,
            maxContextTokens: fb.maxContextTokens,
            config: fbConfig,
          })) ?? messages;
      }
      /** A sibling can trip the breaker while the preparation above is
       * awaited — and the catch below only sees attempts that THROW, so a
       * provider that ignores an aborted signal and succeeds would resolve
       * a run that must reject. Check before every fallback invocation. */
      if (
        config?.signal?.aborted === true &&
        (config.signal.reason instanceof StreamLimitExceededError ||
          config.signal.reason instanceof PreparedSubagentError)
      ) {
        throw config.signal.reason;
      }
      if (preparedRequest != null) {
        return await attemptInvoke(
          {
            request: preparedRequest,
            context,
            onChunk,
            streamLimitState,
            preemptAgentId,
          },
          fbConfig
        );
      }
      return await attemptInvoke(
        {
          model: fbModel as t.ChatModel,
          messages: fallbackMessages,
          provider: fb.provider,
          context,
          onChunk,
          streamLimitState,
          preemptAgentId,
        },
        fbConfig
      );
    } catch (e) {
      /**
       * A tripped stream circuit breaker is a deliberate abort, not a
       * provider failure. Continuing would try the remaining fallbacks and a
       * succeeding one would resolve a run that must reject.
       */
      if (
        e instanceof StreamLimitExceededError ||
        e instanceof PreparedSubagentError ||
        e instanceof InvalidModelToolCallError
      ) {
        throw e;
      }
      /** A parallel sibling's trip aborts this branch's composed signal, and
       * a provider can surface that as a generic abort error; advancing to
       * the next fallback would start new provider work after the safety
       * abort. Rethrow the breaker's own reason instead. */
      if (
        config?.signal?.aborted === true &&
        (config.signal.reason instanceof StreamLimitExceededError ||
          config.signal.reason instanceof PreparedSubagentError)
      ) {
        throw config.signal.reason;
      }
      lastError = e;
      const fallbackOverflowContext: ContextOverflowContext = {
        provider: fb.provider,
        maxContextTokens: fb.maxContextTokens,
        ...(overflowContext?.provider === fb.provider
          ? {
            estimatedPromptTokens: overflowContext.estimatedPromptTokens,
          }
          : {}),
      };
      if (isOverflow(e, fallbackOverflowContext)) {
        const errorContext: FallbackErrorContext = {
          provider: fb.provider,
          clientOptions: fb.clientOptions,
          maxContextTokens: fb.maxContextTokens,
        };
        attachFallbackErrorContext(e, errorContext);
        overflowCandidates.push({ error: e, context: errorContext });
      }
      continue;
    }
  }
  /**
   * Preference order: a fallback overflow, then the primary's overflow, then
   * whichever failure came last. An overflow is the only one of the three a
   * caller can act on, and the fallback's carries the client attribution that
   * makes a correct retry budget possible.
   */
  const preferred =
    overflowCandidates[0]?.error ?? primaryOverflowError ?? lastError;
  if (
    overflowCandidates.length > 0 &&
    typeof preferred === 'object' &&
    preferred !== null
  ) {
    fallbackOverflowCandidates.set(preferred, overflowCandidates);
  }
  if (preferred !== undefined) {
    throw preferred;
  }
  return undefined;
}

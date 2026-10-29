import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { RenderedCompactionSemanticIndex } from '@/summarization/semanticIndex';
import type { StreamLimitState } from '@/llm/streamLimits';
import type { AgentContext } from '@/agents/AgentContext';
import type { EncodingName } from '@/utils/tokens';
import type { HookRegistry } from '@/hooks';
import type { OnChunk } from '@/llm/invoke';
import type * as t from '@/types';
import {
  DEFAULT_SUMMARIZATION_PROMPT,
  DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
  buildSummaryCarrierText,
  separateSummarizationParameters,
  buildSummarizationInstruction,
  ManualSummarizationSkippedError,
} from './shared';
import {
  addTailCacheControl,
  addBedrockTailCacheControl,
  resolvePromptCacheTtl,
  resolveBedrockPromptCacheTtl,
  type PromptCacheTtl,
} from '@/messages/cache';
import {
  enforceStreamLimitsForWireChunk,
  StreamLimitExceededError,
  STREAM_LIMIT_EPOCH_KEY,
  STREAM_LIMIT_ATTEMPT_KEY,
} from '@/llm/streamLimits';
import { classifySummarizationError, withSummarizationRetry, withSummarizationRetryOwner, ReasoningOnlySummaryError } from './retry';
import {
  DEFAULT_RETAIN_RECENT_TURNS,
  resolveIntraTurnRetainTokens,
  splitAtRecencyBoundary,
} from '@/messages/recency';
import {
  boundSummarizationInput,
  DEFAULT_SUMMARY_OUTPUT_TOKENS,
  DEFAULT_RETAIN_RECENT_TOKENS,
} from './input';
import { isReasoningContentBlock, hasMeaningfulReasoningContentBlock } from '@/messages/reasoningTypes';
import {
  createTokenCounter,
  encodingForModel,
  encodingOfTokenCounter,
} from '@/utils/tokens';
import { prepareProviderRequest, resolveServingModelId } from '@/llm/prepareProviderRequest';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
} from '@/common';
import {
  getMaxOutputTokensKey,
  resolveClientOptionsModel,
} from '@/llm/request';
import { renderCompactionSemanticIndex } from '@/summarization/semanticIndex';
import { readSummarizationUsage, buildSummarizationStepUsage } from './usage';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { prepareToolsForPromptCache } from '@/llm/promptCacheTools';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { serializeToolContentBounded } from '@/utils/toolContent';
import { hasReasoningKwargs } from '@/llm/stream/chunkAdapters';
import { makeIsDeferred } from '@/messages/anthropicToolCache';
import { countSummarizationToolTokens } from './toolBudget';
import { createRemoveAllMessage } from '@/messages/reducer';
import { getProviderFamily } from '@/llm/providerRegistry';
import { getProviderStopReasons } from '@/llm/truncation';
import { isProviderCancellation } from '@/utils/errors';
import { initializeModel } from '@/llm/init';
import { getChunkContent } from '@/stream';
import { executeHooks } from '@/hooks';

/** Maximum number of tool failures to include in the enrichment section. */
const MAX_TOOL_FAILURES = 8;
/** Maximum chars per failure summary line. */
const MAX_TOOL_FAILURE_CHARS = 240;

/**
 * Extracts failed tool results from messages and formats them as a structured
 * section. LLMs often omit specific failure details (exit codes, error messages)
 * from their summaries, this mechanical enrichment guarantees they survive.
 */
function extractToolFailuresSection(messages: BaseMessage[]): string {
  const failures: Array<{ toolName: string; summary: string }> = [];
  const seen = new Set<string>();

  for (const msg of messages) {
    if (msg.getType() !== 'tool') {
      continue;
    }
    const toolMsg = msg as ToolMessage;
    if (toolMsg.status !== 'error') {
      continue;
    }
    // Deduplicate by tool_call_id
    const callId = toolMsg.tool_call_id;
    if (callId && seen.has(callId)) {
      continue;
    }
    if (callId) {
      seen.add(callId);
    }

    const toolName = toolMsg.name ?? 'tool';
    const content = serializeToolContentBounded(
      toolMsg.content,
      MAX_TOOL_FAILURE_CHARS * 4
    );
    const normalized = content.replace(/\s+/g, ' ').trim();
    const summary =
      normalized.length > MAX_TOOL_FAILURE_CHARS
        ? `${normalized.slice(0, MAX_TOOL_FAILURE_CHARS - 3)}...`
        : normalized;

    failures.push({ toolName, summary });
  }

  if (failures.length === 0) {
    return '';
  }

  const lines = failures
    .slice(0, MAX_TOOL_FAILURES)
    .map((f) => `- ${f.toolName}: ${f.summary}`);
  if (failures.length > MAX_TOOL_FAILURES) {
    lines.push(`- ...and ${failures.length - MAX_TOOL_FAILURES} more`);
  }

  return `\n\n## Tool Failures\n${lines.join('\n')}`;
}

/**
 * Appends mechanical enrichment sections to an LLM-generated summary.
 * Tool failures are appended verbatim because LLMs often omit specific
 * error details from their summaries.
 */
function enrichSummary(summaryText: string, messages: BaseMessage[]): string {
  return summaryText + extractToolFailuresSection(messages);
}

// ---------------------------------------------------------------------------
// Extracted helpers for createSummarizeNode
// ---------------------------------------------------------------------------

interface SummarizationClientConfig {
  provider: string;
  modelName?: string;
  clientOptions: Record<string, unknown>;
  effectiveMaxSummaryTokens: number;
  promptText: string;
  updatePromptText: string;
}

/** Assembles the summarization model's client options from agent and config. */
function buildSummarizationClientConfig(
  agentContext: AgentContext,
  summarizationConfig?: t.SummarizationConfig
): SummarizationClientConfig {
  const provider = (summarizationConfig?.provider ??
    agentContext.provider) as string;
  const modelName = summarizationConfig?.model;
  const parameters = summarizationConfig?.parameters ?? {};
  const promptText =
    summarizationConfig?.prompt ?? DEFAULT_SUMMARIZATION_PROMPT;
  const updatePromptText =
    summarizationConfig?.updatePrompt ?? DEFAULT_UPDATE_SUMMARIZATION_PROMPT;

  const { llmParams, maxSummaryTokens: paramMaxSummaryTokens } =
    separateSummarizationParameters(parameters);

  const isSelfSummarize = provider === (agentContext.provider as string);
  const baseOptions: Record<string, unknown> =
    isSelfSummarize && agentContext.clientOptions
      ? { ...agentContext.clientOptions }
      : {};

  const clientOptions: Record<string, unknown> = withSummarizationRetryOwner(provider, {
    ...baseOptions,
    ...llmParams,
  });

  if (modelName != null && modelName !== '') {
    clientOptions.model = modelName;
    clientOptions.modelName = modelName;
  }

  const outputKey = getMaxOutputTokensKey(provider);
  const inheritedOutputTokens = readSummaryOutputLimit(provider, baseOptions);
  const explicitOutputTokens = readSummaryOutputLimit(provider, llmParams);
  clearSummaryOutputLimits(provider, clientOptions);
  const defaultOutputTokens = resolveSummaryOutputTokens(
    provider, clientOptions, agentContext.maxContextTokens
  );
  const reasoningOutputTokens = resolveSummaryReasoning(provider, clientOptions).enabled &&
    typeof inheritedOutputTokens === 'number' && Number.isFinite(inheritedOutputTokens) && inheritedOutputTokens > 0
    ? inheritedOutputTokens : 0;
  const effectiveMaxSummaryTokens =
    paramMaxSummaryTokens ??
    summarizationConfig?.maxSummaryTokens ??
    explicitOutputTokens ??
    Math.max(defaultOutputTokens, reasoningOutputTokens);
  clientOptions[outputKey] = effectiveMaxSummaryTokens;

  return {
    provider,
    modelName,
    clientOptions,
    effectiveMaxSummaryTokens,
    promptText,
    updatePromptText,
  };
}

function usesCompletionTokenAlias(provider: t.ProviderName): boolean {
  return getProviderFamily(provider) === 'openai' || provider === Providers.MOONSHOT;
}

function readSummaryOutputLimit(provider: t.ProviderName, options: Record<string, unknown>): number | undefined {
  const openai = usesCompletionTokenAlias(provider);
  const native = (openai ? options.modelKwargs : options.invocationKwargs) as
    {max_output_tokens?: number; max_tokens?: number} | undefined;
  const anthropic = provider === Providers.ANTHROPIC || getProviderFamily(provider) === 'anthropic';
  // Completions rewrites its active output field after merging modelKwargs.
  let nativeLimit: number | undefined;
  if (openai) nativeLimit = native?.max_output_tokens;
  else if (anthropic) nativeLimit = native?.max_tokens;
  return (nativeLimit ?? (openai ? options.maxCompletionTokens : undefined) ??
    options[getMaxOutputTokensKey(provider)]) as number | undefined;
}

function clearSummaryOutputLimits(provider: t.ProviderName, options: Record<string, unknown>): void {
  delete options[getMaxOutputTokensKey(provider)];
  const openai = usesCompletionTokenAlias(provider);
  if (openai) delete options.maxCompletionTokens;
  const key = openai ? 'modelKwargs' : 'invocationKwargs';
  if (!openai && provider !== Providers.ANTHROPIC && getProviderFamily(provider) !== 'anthropic') return;
  const native = options[key] as object | undefined;
  if (native == null) return;
  const normalized = {...native};
  for (const field of openai ? ['max_output_tokens', 'max_completion_tokens', 'max_tokens'] : ['max_tokens']) {
    Reflect.deleteProperty(normalized, field);
  }
  options[key] = normalized;
}

function resolveSummaryReasoning(provider: t.ProviderName, clientOptions: Record<string, unknown>): {enabled: boolean; fixedBudget: number} {
  type ReasoningOptions = {
    reasoningEffort?: string;
    reasoning_effort?: string;
    reasoning?: {effort?: string; enabled?: boolean; max_tokens?: number};
    thinking?: t.AnthropicReasoning['thinking'];
    thinkingBudget?: number;
    thinkingConfig?: t.GoogleThinkingConfig;
  };
  const options = clientOptions as ReasoningOptions & {
    include_reasoning?: boolean;
    modelKwargs?: ReasoningOptions;
    additionalModelRequestFields?: ReasoningOptions;
    invocationKwargs?: ReasoningOptions;
  };
  const family = getProviderFamily(provider);
  const google = options.thinkingConfig;
  if (provider === Providers.GOOGLE || provider === Providers.VERTEXAI || family === 'google') {
    const budget = google?.thinkingBudget ?? options.thinkingBudget;
    const hasLevel = google?.thinkingLevel != null && google.thinkingLevel !== 'THINKING_LEVEL_UNSPECIFIED';
    const fixedBudget = budget != null && Number.isFinite(budget) && budget > 0 ? budget : 0;
    return {
      enabled: hasLevel || (budget != null && (budget > 0 || budget === -1)),
      fixedBudget: provider === Providers.VERTEXAI && hasLevel ? 0 : fixedBudget,
    };
  }
  const thinking = (provider === Providers.ANTHROPIC || family === 'anthropic'
    ? options.invocationKwargs?.thinking : undefined) ?? options.thinking ??
    options.additionalModelRequestFields?.thinking ?? options.modelKwargs?.thinking;
  if (thinking != null) {
    const enabled = thinking === true || (typeof thinking === 'object' &&
      (thinking.type === 'enabled' || thinking.type === 'adaptive'));
    let budget = thinking === true ? options.thinkingBudget ?? options.additionalModelRequestFields?.thinkingBudget : 0;
    if (typeof thinking === 'object' && thinking.type === 'enabled') budget = thinking.budget_tokens;
    return {enabled, fixedBudget: enabled && budget != null && Number.isFinite(budget) && budget > 0 ? budget : 0};
  }
  const reasoning = {...options.modelKwargs?.reasoning, ...options.reasoning};
  const effort = reasoning.effort ?? options.reasoningEffort ?? options.reasoning_effort ??
    options.modelKwargs?.reasoning_effort ?? options.modelKwargs?.reasoningEffort;
  return {enabled: reasoning.enabled !== false && (
    (effort != null && effort !== '' && effort !== 'none' && effort !== 'disabled') ||
    reasoning.enabled === true || (reasoning.max_tokens != null && reasoning.max_tokens > 0) ||
    (provider === Providers.OPENROUTER && options.reasoning == null &&
      options.modelKwargs?.reasoning == null && options.include_reasoning === true)
  ), fixedBudget: 0};
}

function resolveSummaryOutputTokens(
  provider: t.ProviderName,
  clientOptions: Record<string, unknown>,
  maxContextTokens?: number
): number {
  const thinkingBudget = resolveSummaryReasoning(provider, clientOptions).fixedBudget;
  const defaultOutputTokens =
    thinkingBudget +
    Math.min(
      DEFAULT_SUMMARY_OUTPUT_TOKENS,
      Math.max(1, Math.floor((maxContextTokens ?? 40_960) * 0.1))
    );
  return (
    readSummaryOutputLimit(provider, clientOptions) ?? defaultOutputTokens
  );
}

/**
 * Sizes a summary from the text that will actually be re-injected, carrier
 * included, so the stored count is a measurement rather than a body count plus
 * a remembered constant.
 *
 * The provider's `output_tokens` is deliberately not consulted. On a reasoning
 * summarizer the two diverge by the hidden thinking, which is billed but never
 * written into the checkpoint, so using it here would make every later context
 * calculation reserve room for tokens that are never sent. Provider usage stays
 * exclusively a billing input.
 *
 * A missing host counter is not hypothetical: `shouldSummarizeOverflow` fires
 * precisely when there is nothing to count with, so that branch is the
 * overflow-recovery summary, and its count is persisted and then reserved by
 * `AgentContext.instructionTokens` on the retry. Undercounting it is what makes
 * the retry overflow again, which rules out a character heuristic: measured
 * against `o200k_base` and Anthropic's tokenizer, four-characters-per-token
 * understates base64 by 1.5x and Korean by 4.6x, and coefficients large enough
 * to cover those overestimate English prose by roughly 4x. So this falls back
 * to the tokenizer this package already bundles rather than to an estimate.
 *
 * A host counter that is present is not automatically the right one either. In
 * a heterogeneous multi-agent run `Run.create` derives a single counter from
 * `agents[0]` and `StandardGraph` hands that same counter to every
 * `AgentContext`, so a Claude agent behind a GPT first agent would otherwise
 * measure its carrier in `o200k_base` and under-reserve by up to 1.9x on CJK.
 * When the counter came from `createTokenCounter` its encoding is known, so a
 * disagreement with the receiving agent's encoding takes the bundled path
 * instead. A counter the host built itself is unstamped and stays authoritative:
 * its units are the ones the rest of that host's accounting is denominated in.
 */
async function computeSummaryTokenCount(
  summaryText: string,
  agentContext: AgentContext
): Promise<number> {
  const carrier = new HumanMessage(buildSummaryCarrierText(summaryText));
  const encoding = encodingForReceivingAgent(agentContext);
  const hostCounter = agentContext.tokenCounter;
  const hostEncoding =
    hostCounter != null ? encodingOfTokenCounter(hostCounter) : undefined;
  if (
    hostCounter != null &&
    (hostEncoding == null || hostEncoding === encoding)
  ) {
    return hostCounter(carrier);
  }
  const bundledCounter = await createTokenCounter(encoding);
  return bundledCounter(carrier);
}

/**
 * Encoding of the model that will *receive* the carrier.
 *
 * That is the agent's own model, never the summarizer's. The two are the same
 * only by default: `summarizationConfig.model` is undefined for ordinary
 * self-summarization and can name a different provider entirely when a cheap
 * dedicated summarizer is configured. The count produced here is spent by
 * `AgentContext.instructionTokens` against the agent's context window, so it
 * has to be denominated in the agent's tokenizer, and Anthropic's counts run
 * well above `o200k_base` on the same text: measuring CJK with the wrong one
 * understates it by up to 1.9x.
 *
 * Both signals are read, because only one of them can ever be positive.
 * `encodingForModel` matches the substring `claude`, so a name it accepts is
 * proof, which is what keeps a Claude model reached through Bedrock or
 * OpenRouter resolving to `claude`. A name it rejects is only the absence of
 * proof: `production` is an opaque deployment alias, not a statement that the
 * model behind it is something other than Claude. Both option keys are
 * consulted for that name, since `modelName` is LangChain's alias for `model`
 * and hosts configure agents through either, so reading one key alone would
 * report an unconfigured model and quietly hand a Claude agent the
 * `o200k_base` tokenizer.
 *
 * The provider therefore decides every case the name leaves open, not just the
 * case where no model was recorded at all, and it is read as a family rather
 * than as one enum value: a host can register its own provider with
 * `family: 'anthropic'` (the trait `isThinkingEnabled` already reads the same
 * way), and such a provider serves Claude behind whatever name and deployment
 * alias the host chose. `BEDROCK` is left out by the same rule, since its
 * family is `bedrock` and it also serves Llama, Titan and Mistral. The enum
 * check stays ahead of the family lookup because the registry is populated by
 * importing `@/llm/providers`, which a root-barrel consumer defers.
 *
 * On an opaque alias this deliberately parts ways with `Run.create`, which
 * infers from the model name alone and stamps its counter `o200k_base`. That
 * stamp records which tokenizer the counter is, not which one the agent needs,
 * so the disagreement routes the carrier to the bundled Claude tokenizer by
 * the same rule that rejects another agent's counter.
 */
function encodingForReceivingAgent(agentContext: AgentContext): EncodingName {
  return encodingForSummarizationModel(
    agentContext.provider,
    agentContext.clientOptions
  );
}

function encodingForSummarizationModel(
  provider: t.ProviderName,
  clientOptions?: t.ClientOptions
): EncodingName {
  const model = resolveClientOptionsModel(clientOptions);
  if (model != null && encodingForModel(model) === 'claude') {
    return 'claude';
  }
  return provider === Providers.ANTHROPIC ||
    getProviderFamily(provider) === 'anthropic'
    ? 'claude'
    : 'o200k_base';
}

async function prepareSummarizationBudget({
  agentContext,
  provider,
  clientOptions,
  maxContextTokens,
  outputTokens,
  contextBudgetMode = 'input',
  instruction,
  tools,
  config,
}: {
  agentContext: AgentContext;
  provider: t.ProviderName;
  clientOptions?: t.ClientOptions;
  maxContextTokens?: number;
  outputTokens: number;
  contextBudgetMode?: t.ContextBudgetMode;
  instruction: string;
  tools?: t.GraphTools;
  config?: RunnableConfig;
}): Promise<{
  count: t.TokenCounter;
  system: AgentContext['systemRunnable'];
  budget: number;
}> {
  const encoding = encodingForSummarizationModel(provider, clientOptions);
  const hostCounter = agentContext.tokenCounter;
  const hostEncoding =
    hostCounter == null ? undefined : encodingOfTokenCounter(hostCounter);
  const usesAgentCounter =
    provider === agentContext.provider &&
    encoding === encodingForReceivingAgent(agentContext) &&
    hostCounter != null &&
    (hostEncoding == null || hostEncoding === encoding);
  const count = usesAgentCounter
    ? (agentContext.contextPressureTokenCounts?.count ?? hostCounter)
    : await createTokenCounter(encoding);
  const system =
    provider === agentContext.provider
      ? agentContext.systemRunnable
      : undefined;
  const toolTokens = usesAgentCounter ? agentContext.toolSchemaTokens :
    countSummarizationToolTokens({tools, provider, clientOptions, count});
  let overhead = toolTokens;
  if (system != null) {
    overhead = usesAgentCounter
      ? agentContext.instructionTokens
      : (await system.invoke([], config)).reduce(
        (total, message) => total + count(message),
        overhead
      );
  }
  const ratio = usesAgentCounter
    ? Math.max(1, agentContext.calibrationRatio)
    : 1;
  const budget =
    maxContextTokens == null
      ? Number.MAX_SAFE_INTEGER
      : Math.floor(
        (maxContextTokens -
            overhead -
            (contextBudgetMode === 'window' ? outputTokens : 0) -
            count(new HumanMessage(instruction))) /
            ratio
      );
  return { count, system, budget };
}

async function flushSummarizationDeltas(
  deltas: t.SummarizeDeltaEvent[],
  config?: RunnableConfig,
  graph?: { getBreakerSignal?: () => AbortSignal }
): Promise<void> {
  const signal = withBreakerSignal(config, graph)?.signal;
  signal?.throwIfAborted();
  for (const delta of deltas) {
    signal?.throwIfAborted();
    if (config != null)
      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_DELTA,
        delta,
        config
      );
  }
  signal?.throwIfAborted();
}

/**
 * Names the first retained message so the summary declares its own extent
 * rather than leaving the next run to infer coverage from where the block
 * happens to sit.
 *
 * Anchored to the retained side, not the covered side. One source message can
 * expand into several messages — a steer splits an assistant entry into
 * pre-steer, steer, and post-steer entries sharing its ID — and the recency
 * split lands on any human-type message, including the steer. Naming the last
 * *covered* message would then name a half-covered ID with no correct reading;
 * naming the first *retained* message makes that same message the anchor, so it
 * survives whole and everything before it is unambiguously covered.
 *
 * Synthetic entries are skipped, because they can sit *before* a resolvable
 * one. `formatAgentMessages` reconstructs skill bodies inside its payload loop
 * (see the `pendingSkillNames` block) and keeps processing payload entries
 * afterwards, so an unstamped skill body — which `messagesStateReducer` then
 * gives a UUID no payload entry carries — is followed by stamped messages.
 * Anchoring on the UUID would look resolvable at write time and degrade to
 * positional trimming on read, dropping the retained tail. Skipping it reaches
 * the stamped message behind it.
 *
 * `convertInjectedMessages` records `injected` on everything it builds, which is
 * what makes this decidable: `isMeta` and `source` are both optional on
 * `InjectedMessage`, so a bare entry carries no marker of its own, and an
 * injected `source: 'steer'` is otherwise indistinguishable from a replayed one.
 * The remaining `isMeta`/`source` checks cover the constructors that build
 * synthetic entries directly instead of going through that funnel — hook context
 * in `ToolNode` and `StandardGraph`, handoff cues, reconstructed skill bodies.
 *
 * `steer` is exempt from the `source` check because a replayed steer *is*
 * correlated with its payload entry; rejecting every marked `source` once
 * dropped exactly those retained steers. Injected steers are still caught, by
 * `injected`.
 *
 * Known limitation: a payload entry that omits `messageId` is never stamped, so
 * the reducer's UUID is recorded and cannot resolve on the next run. There is no
 * write-time fix — such an entry has no stable ID to name in the next payload
 * either — and the reader's positional fallback is what `main` already does, so
 * the anchor degrades rather than misleads.
 */
function isSyntheticContext(message: BaseMessage): boolean {
  const { additional_kwargs: kwargs } = message;
  if (kwargs.injected === true || kwargs.isMeta === true) {
    return true;
  }
  return kwargs.source != null && kwargs.source !== 'steer';
}

function restoreRetainedMessages(
  retained: BaseMessage[],
  canonical: BaseMessage[]
): BaseMessage[] {
  if (retained.length === 0) return retained;
  const pending = new Map<string, number>();
  for (let i = 0; i < retained.length; i++) {
    const id = retained[i].id;
    if (id != null) pending.set(id, i);
  }
  const restored = [...retained];
  for (let i = canonical.length - 1; i >= 0 && pending.size > 0; i--) {
    const message = canonical[i];
    if (message.id == null) continue;
    const index = pending.get(message.id);
    if (index == null) continue;
    restored[index] = message;
    pending.delete(message.id);
  }
  return restored;
}

function resolveSummaryCoverage(
  messagesToRetain: BaseMessage[]
): t.SummaryCoverage | undefined {
  for (let i = 0; i < messagesToRetain.length; i++) {
    const message = messagesToRetain[i];
    if (isSyntheticContext(message)) {
      continue;
    }
    const sourceMessageId = message.additional_kwargs.sourceMessageId;
    const sourceId =
      typeof sourceMessageId === 'string' ? sourceMessageId.trim() : '';
    const id = sourceId !== '' ? sourceId : message.id?.trim();
    if (id != null && id !== '') {
      return { retainedFromMessageId: id };
    }
  }
  return undefined;
}

/** Constructs the SummaryContentBlock persisted in the run step and dispatched to events. */
function buildSummaryBlock(params: {
  summaryText: string;
  tokenCount: number;
  coverage?: t.SummaryCoverage;
  stepId: string;
  stepIndex: number;
  modelName?: string;
  provider: string;
  summaryVersion: number;
}): t.SummaryContentBlock {
  return {
    type: ContentTypes.SUMMARY,
    content: [
      {
        type: ContentTypes.TEXT,
        text: params.summaryText,
      } as t.MessageContentComplex,
    ],
    tokenCount: params.tokenCount,
    ...(params.coverage != null ? { coverage: params.coverage } : {}),
    summaryVersion: params.summaryVersion,
    boundary: {
      messageId: params.stepId,
      contentIndex: params.stepIndex,
    },
    model: params.modelName,
    provider: params.provider,
    createdAt: new Date().toISOString(),
  };
}

type LogFn = (
  level: 'debug' | 'info' | 'warn' | 'error',
  message: string,
  data?: Record<string, unknown>
) => void;

/**
 * Extracts an HTTP status code from a thrown LLM-provider error. Returns
 * `undefined` for non-object values (including `null` or `undefined`, both
 * valid `throw` targets in JS) so callers never dereference a nullish
 * value.
 */
function extractHttpStatus(err: unknown): number | undefined {
  if (err == null || typeof err !== 'object') {
    return undefined;
  }
  const errRecord = err as Record<string, unknown>;
  const direct = errRecord.status;
  if (typeof direct === 'number') {
    return direct;
  }
  const statusCode = errRecord.statusCode;
  if (typeof statusCode === 'number') {
    return statusCode;
  }
  const response = errRecord.response;
  if (response != null && typeof response === 'object') {
    const nested = (response as Record<string, unknown>).status;
    if (typeof nested === 'number') {
      return nested;
    }
  }
  return undefined;
}

/**
 * Formats a provider-level error for logging. Returns both a human-readable
 * suffix (safe to include in the message string so it survives any host-side
 * formatter) and a structured metadata bag for rich log backends.
 */
function describeProviderError(
  err: unknown,
  provider: string,
  modelName?: string
): { suffix: string; data: Record<string, unknown> } {
  const providerLabel = `${provider}/${modelName ?? '(no-model)'}`;
  const errMsg = err instanceof Error ? err.message : String(err);

  const data: Record<string, unknown> = {
    provider,
    model: modelName,
  };
  if (err instanceof Error) {
    data.errorName = err.name;
    data.errorStack = err.stack;
  }

  const status = extractHttpStatus(err);
  const statusSuffix = status != null ? ` (HTTP ${status})` : '';
  if (status != null) {
    data.status = status;
  }

  return {
    suffix: `[${providerLabel}]${statusSuffix}: ${errMsg}`,
    data,
  };
}

/**
 * Formats an exhausted-fallback error. `tryFallbackProviders` throws the
 * last fallback provider's error, which may be from any of the configured
 * fallbacks — not the primary — so we label the log with the list of
 * fallback providers attempted rather than mis-attributing to the primary.
 *
 * Entries in `fallbacks` are normally strongly typed, but we defend against
 * malformed runtime config (null/undefined entries, missing `provider`
 * field) so a recoverable summarization failure is never promoted to an
 * uncaught exception from inside the logging path.
 */
function describeFallbackError(
  err: unknown,
  fallbacks: unknown
): { suffix: string; data: Record<string, unknown> } {
  const errMsg = err instanceof Error ? err.message : String(err);
  const list: ReadonlyArray<unknown> = Array.isArray(fallbacks)
    ? fallbacks
    : [];
  const providerNames = list
    .map((f) => {
      if (f == null || typeof f !== 'object') {
        return undefined;
      }
      const raw = (f as { provider?: unknown }).provider;
      return raw != null ? String(raw) : undefined;
    })
    .filter((p): p is string => typeof p === 'string');
  const label =
    providerNames.length > 0
      ? `fallbacks=[${providerNames.join(',')}]`
      : 'no-fallbacks';

  const data: Record<string, unknown> = {
    fallbackProviders: providerNames,
    fallbackCount: list.length,
  };
  if (err instanceof Error) {
    data.errorName = err.name;
    data.errorStack = err.stack;
  }
  const status = extractHttpStatus(err);
  const statusSuffix = status != null ? ` (HTTP ${status})` : '';
  if (status != null) {
    data.status = status;
  }

  return {
    suffix: `[${label}]${statusSuffix}: ${errMsg}`,
    data,
  };
}

/**
 * Runs the summarization LLM call with primary + fallback providers,
 * preserving history when every provider fails.
 */
async function executeSummarizationWithFallback(params: {
  agentContext: AgentContext;
  messages: BaseMessage[];
  requiredHeadLength: number;
  clientConfig: SummarizationClientConfig;
  summarizeConfig?: RunnableConfig;
  stepId: string;
  usePromptCache: boolean;
  shrinkInput?: boolean;
  semanticIndex: RenderedCompactionSemanticIndex;
  log: LogFn;
  onUsage: (attempt: t.SummarizationAttemptUsage) => void;
  /** Carries the run's stream limits so the event cap covers summary streams. */
  graph?: StreamLimitState & {
    getBreakerSignal?: () => AbortSignal;
    getToolsForBinding?: (
      provider: t.ProviderName,
      clientOptions: t.ClientOptions | undefined
    ) => t.GraphTools | undefined;
  };
}): Promise<{
  text: string;
  usage?: Partial<UsageMetadata>;
  provider: string;
  model?: string;
  errorCode?: t.SummarizationErrorCode;
}> {
  const {
    agentContext,
    messages,
    requiredHeadLength,
    clientConfig,
    summarizeConfig,
    stepId,
    usePromptCache,
    shrinkInput,
    semanticIndex,
    log,
    graph,
    onUsage,
  } = params;

  const priorSummaryText = agentContext.getSummaryText()?.trim() ?? '';

  let summaryText = '';
  let summaryProvider = clientConfig.provider;
  let summaryModel = clientConfig.modelName;
  let summaryUsage: Partial<UsageMetadata> | undefined;
  let errorCode: t.SummarizationErrorCode | undefined;

  try {
    /**
     * Initialize inside the try so that a misconfigured provider
     * (e.g. an unrecognized summarization.provider) surfaces through the
     * `log('error', ...)` path below rather than bubbling up silently.
     */
    const isSelfSummarize = clientConfig.provider === agentContext.provider;
    const summarizationTools = isSelfSummarize
      ? (graph?.getToolsForBinding?.(
          clientConfig.provider as t.ProviderName,
          clientConfig.clientOptions as t.ClientOptions
      ) ??
        prepareToolsForPromptCache({
          provider: clientConfig.provider as t.ProviderName,
          clientOptions: clientConfig.clientOptions as t.ClientOptions,
          tools: agentContext.getToolsForBinding(),
          isDeferred: makeIsDeferred(
            agentContext.getEffectiveToolDefinitions()
          ),
        }))
      : agentContext.getToolsForBinding();
    const summarizationModel = initializeModel({
      provider: clientConfig.provider,
      clientOptions: clientConfig.clientOptions as t.ClientOptions,
      tools: summarizationTools,
      disableToolCalling: true,
    }) as t.ChatModel;

    const signal = withBreakerSignal(summarizeConfig, graph)?.signal;
    const result = await withSummarizationRetry(
      () =>
        summarizeWithCacheHit({
          agentContext,
          onUsage: usage => onUsage({
            provider: clientConfig.provider as t.ProviderName,
            model: clientConfig.modelName ?? resolveServingModelId(summarizationModel) ?? resolveClientOptionsModel(clientConfig.clientOptions as t.ClientOptions),
            usage,
          }),
          shrinkInput,
          outputTokens: clientConfig.effectiveMaxSummaryTokens,
          clientOptions: clientConfig.clientOptions as t.ClientOptions,
          model: summarizationModel,
          tools: summarizationTools,
          messages,
          requiredHeadLength,
          promptText: clientConfig.promptText,
          updatePromptText: clientConfig.updatePromptText,
          priorSummaryText,
          semanticIndexAppendix: semanticIndex.appendix,
          config: summarizeConfig,
          stepId,
          provider: clientConfig.provider,
          reasoningKey: agentContext.reasoningKey,
          graph,
          usePromptCache,
          promptCacheTtl:
        clientConfig.provider === Providers.ANTHROPIC ||
        clientConfig.provider === Providers.OPENROUTER ||
        clientConfig.provider === Providers.BEDROCK
          ? (
              clientConfig.clientOptions as {
                promptCacheTtl?: PromptCacheTtl;
              }
          ).promptCacheTtl
          : undefined,
          bedrockModelId:
        clientConfig.provider === Providers.BEDROCK
          ? resolveBedrockCompactionCacheModel(
              clientConfig.clientOptions as
                | {
                    applicationInferenceProfile?: string;
                    model?: string;
                  }
                | undefined
          )
          : undefined,
          log,
        }),
      {
        signal,
        onRetry: (retry) => log('debug', 'Retrying summarization', retry),
      }
    );
    summaryText = result.text;
    summaryUsage = result.usage;
  } catch (primaryError) {
    /**
     * A tripped stream circuit breaker is a safety abort, not a
     * summarization failure to recover from: retrying on fallback providers
     * would spend more provider work after the limit fired despite the
     * `StreamLimits` contract that a breach rejects the run. Rethrown so the summarize node fails the run with
     * the actionable limit error, consistent with agent turns and subagents.
     */
    if (
      primaryError instanceof StreamLimitExceededError ||
      primaryError instanceof ProviderTextProtectionError ||
      isProviderCancellation(primaryError)
    ) {
      throw primaryError;
    }
    /** A parallel branch's trip aborts the composed summarization signal,
     * and a provider can surface that as a generic abort error; entering
     * fallback recovery would continue work after the run-wide safety abort. Checked on BOTH channels: the
     * graph's own breaker, and the composed config signal that carries a
     * parent run's trip into subagent child graphs. */
    {
      const trippedReason = findStreamLimitAbortReason(
        graph?.getBreakerSignal?.(),
        summarizeConfig?.signal
      );
      if (trippedReason != null) {
        throw trippedReason;
      }
    }

    withBreakerSignal(summarizeConfig, graph)?.signal?.throwIfAborted();
    errorCode = classifySummarizationError(primaryError);

    const primaryDescribed = describeProviderError(
      primaryError,
      clientConfig.provider,
      clientConfig.modelName
    );
    log('error', `Summarization LLM call failed ${primaryDescribed.suffix}`, {
      ...primaryDescribed.data,
      messagesToRefineCount: messages.length,
    });

    const rawFallbacks = (
      clientConfig.clientOptions as unknown as t.LLMConfig | undefined
    )?.fallbacks;
    const fallbacks = Array.isArray(rawFallbacks) ? rawFallbacks : [];
    if (fallbacks.length > 0) {
      try {
        const boundedFallbacks = fallbacks.map((fallback) => {
          const maxContextTokens =
            fallback.maxContextTokens ?? agentContext.maxContextTokens;
          const outputTokens = resolveSummaryOutputTokens(
            fallback.provider, {...fallback.clientOptions}, maxContextTokens
          );
          const clientOptions: Record<string, unknown> = withSummarizationRetryOwner(fallback.provider, {...fallback.clientOptions});
          clearSummaryOutputLimits(fallback.provider, clientOptions);
          clientOptions[getMaxOutputTokensKey(fallback.provider)] = outputTokens;
          return {
            ...fallback,
            clientOptions,
            maxContextTokens,
            contextBudgetMode: fallback.maxContextTokens == null
              ? agentContext.summarizationBudgetMode : fallback.contextBudgetMode ?? 'window',
          } as t.FallbackConfig;
        });
        let fallbackIdentity: {provider: string; model?: string} | undefined;
        const pendingDeltas: t.SummarizeDeltaEvent[] = [];
        const onChunk = createSummarizationChunkHandler({
          pendingDeltas,
          stepId,
          config: traceConfig(summarizeConfig, 'cache_hit_compaction'),
          provider: clientConfig.provider,
          reasoningKey: agentContext.reasoningKey,
          graph,
        });
        const fallbackTools = agentContext.getToolsForBinding();
        const fbResult = await tryFallbackProviders({
          fallbacks: boundedFallbacks,
          disableToolCalling: true,
          tools: fallbackTools,
          messages,
          prepareProviderRequest: async ({
            model,
            provider,
            clientOptions,
            maxContextTokens,
            contextBudgetMode,
            config,
          }) => {
            config?.signal?.throwIfAborted();
            pendingDeltas.length = 0;
            const instruction = buildSummarizationInstruction(
              clientConfig.promptText, clientConfig.updatePromptText, priorSummaryText,
              semanticIndex.appendix,
              provider === agentContext.provider && agentContext.systemRunnable != null,
            );
            const { count, system, budget } = await prepareSummarizationBudget({
              agentContext,
              provider,
              clientOptions,
              maxContextTokens,
              contextBudgetMode,
              outputTokens: resolveSummaryOutputTokens(provider, {...clientOptions}, maxContextTokens),
              instruction,
              tools: fallbackTools,
              config,
            });
            const bounded = boundSummarizationInput({
              messages,
              requiredHeadLength,
              tokenCounter: count,
              budget,
              provider,
            });
            const bedrock = clientOptions as t.BedrockAnthropicInput | undefined;
            const windowWithCache = provider === Providers.BEDROCK
              ? applySummarizationHistoryCache({messages: bounded, provider,
                enabled: bedrock?.promptCache === true,
                promptCacheTtl: bedrock?.promptCacheTtl, bedrockModelId: resolveClientOptionsModel(clientOptions),
              }) : bounded;
            const history = system == null ? windowWithCache : await system.invoke(windowWithCache, config);
            const request = prepareProviderRequest({
              model,
              messages: [...history, new HumanMessage(instruction)],
              provider,
              config,
            });
            fallbackIdentity = {provider, model: request.modelId ?? resolveClientOptionsModel(clientOptions)};
            return request;
          },
          config: withBreakerSignal(
            traceConfig(summarizeConfig, 'cache_hit_compaction'),
            graph
          ),
          primaryError,
          onChunk,
          streamLimitState: graph,
          invokeModel: (params, config) => withSummarizationRetry(async () => {
            pendingDeltas.length = 0;
            const result = await attemptInvoke({...params, onModelComplete: response => {
              params.onModelComplete?.(response);
              const usage = readSummarizationUsage(response);
              if (usage != null && fallbackIdentity != null) onUsage({
                provider: fallbackIdentity.provider as t.ProviderName, model: fallbackIdentity.model, usage,
              });
            }}, config);
            config?.signal?.throwIfAborted();
            const response = result.messages?.[0];
            validateSummaryResponse(response);
            return result;
          }, {signal: config?.signal, onRetry: retry => log('debug', 'Retrying fallback summarization', retry)}),
        });
        const fbMsg = fbResult?.messages?.[0];
        if (fbMsg) {
          summaryUsage = readSummarizationUsage(fbMsg);
          summaryText = extractResponseText(
            fbMsg as { content: string | object }
          );
          if (summaryText !== '') {
            if (fallbackIdentity == null) throw new Error('Missing successful fallback identity', {cause: primaryError});
            summaryProvider = fallbackIdentity.provider;
            summaryModel = fallbackIdentity.model;
            await flushSummarizationDeltas(
              pendingDeltas,
              summarizeConfig,
              graph
            );
          }
        }
      } catch (fbErr) {
        const trip = findStreamLimitAbortReason(
          graph?.getBreakerSignal?.(),
          summarizeConfig?.signal
        );
        if (trip != null) throw trip;
        if (
          fbErr instanceof StreamLimitExceededError ||
          fbErr instanceof ProviderTextProtectionError ||
          isProviderCancellation(fbErr)
        ) {
          throw fbErr;
        }
        withBreakerSignal(summarizeConfig, graph)?.signal?.throwIfAborted();
        errorCode = classifySummarizationError(fbErr);
        const fbDescribed = describeFallbackError(fbErr, fallbacks);
        log('warn', `Fallback providers also failed ${fbDescribed.suffix}`, {
          ...fbDescribed.data,
        });
      }
    }
    if (!summaryText) {
      log(
        'warn',
        `Summarization failed; preserving conversation history ${primaryDescribed.suffix}`,
        {
          ...primaryDescribed.data,
          messagesToRefineCount: messages.length,
        }
      );
      summaryText = '';
    }
  }

  const trip = findStreamLimitAbortReason(
    graph?.getBreakerSignal?.(),
    summarizeConfig?.signal
  );
  if (trip != null) throw trip;
  return {
    text: summaryText,
    usage: summaryUsage,
    provider: summaryProvider,
    model: summaryModel,
    ...(summaryText === '' ? { errorCode } : {}),
  };
}

/** Dispatches run step completion, ON_SUMMARIZE_COMPLETE, and rebuilds token map. */
async function dispatchCompletionEvents(params: {
  graph: CreateSummarizeNodeParams['graph'];
  runnableConfig?: RunnableConfig;
  stepId: string;
  summaryBlock: t.SummaryContentBlock;
  agentContext: AgentContext;
  runStep: t.RunStep;
  summaryUsage?: Partial<UsageMetadata>;
  attemptUsage: readonly t.SummarizationAttemptUsage[];
  agentId: string;
  /**
   * Number of messages preserved verbatim by the recency window after
   * compaction.  Reported via the PostCompact hook payload so observers
   * (metrics, cleanup) see the true post-compaction message count
   * instead of always-zero.
   */
  messagesAfterCount: number;
}): Promise<void> {
  const {
    graph,
    runnableConfig,
    stepId,
    summaryBlock,
    agentContext,
    runStep,
    summaryUsage,
    attemptUsage,
    agentId,
    messagesAfterCount,
  } = params;

  const assertNotTripped = (): void => {
    const trip = findStreamLimitAbortReason(
      graph.getBreakerSignal?.(),
      runnableConfig?.signal
    );
    if (trip != null) throw trip;
  };
  assertNotTripped();
  runStep.summary = summaryBlock;
  if (summaryUsage && runStep.usage == null) {
    runStep.usage = {
      prompt_tokens: Number(summaryUsage.input_tokens) || 0,
      completion_tokens: Number(summaryUsage.output_tokens) || 0,
      total_tokens:
        (Number(summaryUsage.input_tokens) || 0) +
        (Number(summaryUsage.output_tokens) || 0),
    };
  }

  await graph.dispatchRunStepCompleted(
    stepId,
    { type: 'summary', summary: summaryBlock } satisfies t.SummaryCompleted,
    runnableConfig
  );

  assertNotTripped();
  if (runnableConfig) {
    await safeDispatchCustomEvent(
      GraphEvents.ON_SUMMARIZE_COMPLETE,
      {
        id: stepId,
        agentId,
        summary: summaryBlock,
        ...(attemptUsage.length > 0 ? {attemptUsage: [...attemptUsage]} : {}),
      } satisfies t.SummarizeCompleteEvent,
      runnableConfig
    );
  }

  assertNotTripped();
  const sessionId = graph.runId ?? '';
  if (graph.hookRegistry?.hasHookFor('PostCompact', sessionId) === true) {
    const threadId = (
      runnableConfig?.configurable as Record<string, unknown> | undefined
    )?.thread_id as string | undefined;
    const firstBlock = summaryBlock.content?.[0];
    const summaryText =
      firstBlock != null &&
      typeof firstBlock === 'object' &&
      'text' in firstBlock &&
      typeof firstBlock.text === 'string'
        ? firstBlock.text
        : '';
    await executeHooks({
      registry: graph.hookRegistry,
      input: {
        hook_event_name: 'PostCompact',
        runId: sessionId,
        threadId,
        agentId,
        executionContext: graph.getSubagentExecutionContext?.(),
        summary: summaryText,
        messagesAfterCount,
      },
      sessionId,
    }).catch(() => {
      /* PostCompact is observational — swallow errors */
    });
  }

  agentContext.rebuildTokenMapAfterSummarization({});
}

// ---------------------------------------------------------------------------
// createSummarizeNode
// ---------------------------------------------------------------------------

interface CreateSummarizeNodeParams {
  agentContext: AgentContext;
  graph: {
    contentData: t.RunStep[];
    contentIndexMap: Map<string, number>;
    config?: RunnableConfig;
    runId?: string;
    isMultiAgent: boolean;
    getSubagentExecutionContext?: () => t.SubagentExecutionContext | undefined;
    hookRegistry?: HookRegistry;
    getToolsForBinding?: (
      provider: t.ProviderName,
      clientOptions: t.ClientOptions | undefined
    ) => t.GraphTools | undefined;
    dispatchRunStep: (
      runStep: t.RunStep,
      config?: RunnableConfig
    ) => Promise<void>;
    dispatchRunStepCompleted: (
      stepId: string,
      result: t.StepCompleted,
      config?: RunnableConfig
    ) => Promise<void>;
    /**
     * Terminal close for a summary step that ends without a completion —
     * an errored or empty summary would otherwise stay `in_progress` until
     * the run-end sweep reported it as `completed`.
     */
    closeRunStep?: (
      stepId: string,
      status: Exclude<t.RunStepStatus, 'in_progress'>,
      config?: RunnableConfig
    ) => Promise<void>;
    /** The run's shared breaker signal, composed into every summarization
     * model attempt so a sibling branch tripping a stream limit also
     * cancels in-flight summaries. */
    getBreakerSignal?: () => AbortSignal;
    /** The run's shared breaker controller. Preferred over the bare signal
     * accessor: the node captures it at entry so a breach detected by its
     * own chunk handler trips the run that STARTED the summarization. */
    getBreakerController?: () => AbortController;
    /** The controller's epoch, captured at entry and stamped into summary
     * attempt metadata so the wire consumer epoch-gates old-run summary
     * chunks like model-attempt chunks. */
    getBreakerEpoch?: () => number;
  } & StreamLimitState;
  generateStepId: (stepKey: string) => [string, number];
}

/** Composes the run's breaker signal (when the adapter provides one) into a
 * summarization attempt's config, so breaches in sibling branches cancel
 * summaries too. */
function withBreakerSignal<T extends { signal?: AbortSignal }>(
  config: T | undefined,
  graph?: { getBreakerSignal?: () => AbortSignal }
): T | undefined {
  const breakerSignal = graph?.getBreakerSignal?.();
  if (breakerSignal == null) {
    return config;
  }
  if (config == null) {
    return { signal: breakerSignal } as T;
  }
  const existing = config.signal;
  if (existing == null || existing === breakerSignal) {
    return { ...config, signal: breakerSignal };
  }
  return { ...config, signal: AbortSignal.any([existing, breakerSignal]) };
}

/** First stream-limit reason found on any of the given signals. Checked
 * alongside the graph's own breaker because a subagent child graph receives
 * a ROOT sibling's trip through its constructor/invocation signal — the
 * child's private breaker never fires for it. */
function findStreamLimitAbortReason(
  ...signals: Array<AbortSignal | undefined>
): StreamLimitExceededError | ProviderTextProtectionError | undefined {
  for (const signal of signals) {
    if (
      signal?.aborted === true &&
      (signal.reason instanceof StreamLimitExceededError ||
        signal.reason instanceof ProviderTextProtectionError)
    ) {
      return signal.reason;
    }
  }
  return undefined;
}

/** Wording for the history-preserved log and completion, by request reason. */
const PRESERVATION_REASONS: Partial<
  Record<NonNullable<t.SummarizationNodeInput['reason']>, string>
> = {
  overflow: 'overflow recovery',
  manual: 'manual compaction',
};

export function createSummarizeNode({
  agentContext,
  graph: adapterGraph,
  generateStepId,
}: CreateSummarizeNodeParams) {
  return async (
    state: {
      messages: BaseMessage[];
      summarizationRequest?: t.SummarizationNodeInput;
    },
    config?: RunnableConfig
  ): Promise<{
    summarizationRequest: undefined;
    messages?: BaseMessage[];
    manualSummary?: string;
  }> => {
    /** The breaker signal is captured at node ENTRY, before dispatchRunStep,
     * ON_SUMMARIZE_START, or PreCompact awaits: a sibling's trip plus a
     * prompt next run can reset the controller during one of those awaits,
     * and resolving the live accessor later would bind this summarization
     * to the fresh controller. `Object.create` freezes the signal while
     * DELEGATING every other adapter member — spreading would snapshot the
     * charge-credit accessor pair and detach it from graph resets. */
    const entryBreakerController = adapterGraph.getBreakerController?.();
    const entryBreakerSignal =
      entryBreakerController?.signal ?? adapterGraph.getBreakerSignal?.();
    const entryBreakerEpoch = adapterGraph.getBreakerEpoch?.();
    const graph =
      entryBreakerSignal == null
        ? adapterGraph
        : (Object.create(adapterGraph, {
          getBreakerSignal: { value: (): AbortSignal => entryBreakerSignal },
          ...(entryBreakerController != null && {
            getBreakerController: {
              value: (): AbortController => entryBreakerController,
            },
          }),
        }) as typeof adapterGraph);
    const request = state.summarizationRequest;
    if (request == null) {
      return { summarizationRequest: undefined };
    }
    /** Already-tripped-at-entry: a parallel sibling's breach failed the run
     * before this node was scheduled — via this graph's own breaker or, in
     * a subagent child graph, the parent trip on the invocation signal.
     * Rethrow before dispatching steps, hooks, or the model call — an
     * adapter that doesn't synchronously reject an aborted signal would
     * otherwise spend summarization provider work on a failed run. */
    const entryTripReason = findStreamLimitAbortReason(
      entryBreakerSignal,
      config?.signal
    );
    if (entryTripReason != null) {
      throw entryTripReason;
    }

    /**
     * Overflow recovery routes through this node purely to get back to the
     * agent node with a corrected budget, and deliberately spends no model
     * call on its first attempt: re-pruning under the raised context pressure
     * drives the pruner's tool-output compression and masking, which is
     * cheaper than a summary and cannot lose message content. Summarization
     * is also skipped outright when the caller never enabled it.
     */
    if (
      request.reason === 'overflow' &&
      (request.allowSummarization !== true ||
        agentContext.summarizationEnabled !== true)
    ) {
      emitAgentLog(
        config,
        'debug',
        'summarize',
        'Overflow recovery re-prune — compressing tool output without a summarization call',
        {
          maxContextTokens: agentContext.maxContextTokens,
          summarizationEnabled: agentContext.summarizationEnabled === true,
          allowSummarization: request.allowSummarization === true,
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      return { summarizationRequest: undefined };
    }

    /**
     * A summarizer that has already returned nothing several times in a row
     * keeps returning nothing, and every empty result leaves the message set
     * exactly as it was — so the next prune cycle re-triggers on identical
     * state. Stopping here bounds that loop instead of letting the run spend
     * its recursion budget dispatching empty summary steps.
     */
    if (agentContext.summarizationExhausted) {
      if (request.reason === 'manual') {
        throw new ManualSummarizationSkippedError(
          'exhausted',
          'Compaction skipped: consecutive summarization attempts produced no usable summary'
        );
      }
      emitAgentLog(
        config,
        'warn',
        'summarize',
        'Summarization skipped — consecutive attempts produced no usable summary',
        {
          failures: agentContext.summarizationFailures,
          reason: request.reason ?? 'trigger',
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      agentContext.markSummarizationTriggered(state.messages.length);
      return { summarizationRequest: undefined };
    }

    const maxCtx = agentContext.maxContextTokens ?? 0;
    const usesAgentInstructions =
      (agentContext.summarizationConfig?.provider ?? agentContext.provider) ===
      agentContext.provider;
    const instructionOverhead = usesAgentInstructions
      ? agentContext.instructionTokens
      : agentContext.toolSchemaTokens;
    if (usesAgentInstructions && maxCtx > 0 && instructionOverhead >= maxCtx) {
      if (request.reason === 'manual') {
        throw new ManualSummarizationSkippedError(
          'instructions_exceed_budget',
          'Compaction skipped: instructions and tool definitions exceed the context budget'
        );
      }
      emitAgentLog(
        config,
        'warn',
        'summarize',
        'Summarization skipped, instructions exceed context budget. Reduce the number of tools or increase maxContextTokens.',
        {
          instructionTokens: agentContext.instructionTokens,
          maxContextTokens: maxCtx,
          breakdown: agentContext.formatTokenBudgetBreakdown(),
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      return { summarizationRequest: undefined };
    }

    const runnableConfig = config ?? graph.config;
    const clientConfig = buildSummarizationClientConfig(
      agentContext,
      agentContext.summarizationConfig
    );
    const windowMessages = request.windowMessages ?? state.messages;
    const retainRecent = agentContext.summarizationConfig?.retainRecent;
    const retainTurns = retainRecent?.turns ?? DEFAULT_RETAIN_RECENT_TURNS;
    const recencyTokenCounter =
      agentContext.contextPressureTokenCounts?.count ??
      agentContext.tokenCounter ??
      (await createTokenCounter(encodingForReceivingAgent(agentContext)));
    const retainTokens =
      retainRecent?.tokens ??
      (retainRecent == null
        ? Math.min(
          DEFAULT_RETAIN_RECENT_TOKENS,
          Math.max(
            1,
            Math.floor((agentContext.maxContextTokens ?? 100_000) * 0.16)
          )
        )
        : undefined);
    const {
      head: compactableHead,
      tail: projectedTail,
    } = splitAtRecencyBoundary(windowMessages, {
      provider: agentContext.provider,
      turns:
        retainRecent == null
          ? retainTurns
          : (retainRecent.turns ?? DEFAULT_RETAIN_RECENT_TURNS),
      tokens: retainTokens,
      tokenCounter: recencyTokenCounter,
      intraTurnTokens: resolveIntraTurnRetainTokens({
        tokens: retainTokens,
        maxContextTokens: agentContext.maxContextTokens,
      }),
    });

    const messagesToRefine = compactableHead;

    if (messagesToRefine.length === 0) {
      if (request.reason === 'manual') {
        throw new ManualSummarizationSkippedError(
          'nothing_to_summarize',
          'Nothing to compact: the retained recent turns cover the whole conversation'
        );
      }
      /**
       * Recency window covers the entire conversation — there is no
       * older content to summarize.  Skipping prevents the model from
       * destroying the user's most recent message (e.g. a large pasted
       * payload on the first turn) by replacing it with a generic
       * checkpoint summary.  Mark the trigger so the same unchanged
       * state is not re-evaluated on the next prune cycle.
       */
      emitAgentLog(
        config,
        'debug',
        'summarize',
        'Summarization skipped — recency window retains all messages',
        {
          messagesRetained: projectedTail.length,
          retainTurns,
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      agentContext.markSummarizationTriggered(state.messages.length);
      return { summarizationRequest: undefined };
    }

    const messagesToRetain = windowMessages === state.messages
      ? projectedTail : restoreRetainedMessages(projectedTail, state.messages);

    const semanticIndex = renderCompactionSemanticIndex(
      agentContext.compactionSemanticIndex,
      messagesToRefine
    );

    const stepKey = `summarize-${request.agentId}`;
    const [stepId, stepIndex] = generateStepId(stepKey);

    const placeholderSummary: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      model: clientConfig.modelName,
      provider: clientConfig.provider,
    };

    const attemptUsage: t.SummarizationAttemptUsage[] = [];
    const runStep: t.RunStep = {
      stepIndex,
      id: stepId,
      type: StepTypes.MESSAGE_CREATION,
      index: graph.contentData.length,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: stepId },
      },
      summary: placeholderSummary,
      usage: null,
    };

    if (graph.runId != null && graph.runId !== '') {
      runStep.runId = graph.runId;
    }
    if (graph.isMultiAgent && agentContext.agentId) {
      runStep.agentId = agentContext.agentId;
    }

    await graph.dispatchRunStep(runStep, runnableConfig);

    if (runnableConfig) {
      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_START,
        {
          agentId: request.agentId,
          provider: clientConfig.provider,
          model: clientConfig.modelName,
          messagesToRefineCount: messagesToRefine.length,
          summaryVersion: agentContext.summaryVersion + 1,
          semanticIndexEntryCount: semanticIndex.entryCount,
          semanticIndexCharCount: semanticIndex.charCount,
        } satisfies t.SummarizeStartEvent,
        runnableConfig
      );
    }

    const sessionId = graph.runId ?? '';
    if (graph.hookRegistry?.hasHookFor('PreCompact', sessionId) === true) {
      const threadId = (
        runnableConfig?.configurable as Record<string, unknown> | undefined
      )?.thread_id as string | undefined;
      await executeHooks({
        registry: graph.hookRegistry,
        input: {
          hook_event_name: 'PreCompact',
          runId: sessionId,
          threadId,
          agentId: request.agentId,
          executionContext: graph.getSubagentExecutionContext?.(),
          messagesBeforeCount: messagesToRefine.length,
          trigger:
            request.reason === 'manual'
              ? 'manual'
              : (agentContext.summarizationConfig?.trigger?.type ?? 'default'),
        },
        sessionId,
      }).catch(() => {
        /* PreCompact is observational — swallow errors */
      });
    }

    const isSelfSummarizeModel =
      clientConfig.provider === (agentContext.provider as string);
    const hasPromptCache =
      isSelfSummarizeModel &&
      (clientConfig.clientOptions as Record<string, unknown> | undefined)
        ?.promptCache === true;

    const log: LogFn = (level, message, data) => {
      emitAgentLog(runnableConfig, level, 'summarize', message, data, {
        runId: graph.runId,
        agentId: request.agentId,
      });
    };

    log('debug', 'Summarization starting', {
      messagesToRefineCount: messagesToRefine.length,
      hasPriorSummary: (agentContext.getSummaryText()?.trim() ?? '') !== '',
      summaryVersion: agentContext.summaryVersion + 1,
      isSelfSummarize: isSelfSummarizeModel,
      hasPromptCache,
      provider: clientConfig.provider,
      semanticIndexEntryCount: semanticIndex.entryCount,
      semanticIndexCharCount: semanticIndex.charCount,
      semanticIndexOmittedEntryCount: semanticIndex.omittedEntryCount,
    });

    const summarizeConfig: RunnableConfig | undefined = config
      ? {
        ...config,
        metadata: {
          ...config.metadata,
          agent_id: request.agentId,
          /** Canonical agent-identity key, overwritten by every component
             *  that stamps identity (graph model path, ToolNode, here) so the
             *  closest stamper always wins via spread order — Langfuse scope
             *  trust relies on it (see `isForeignScope`). */
          agentId: request.agentId,
          summarization_provider: clientConfig.provider,
          summarization_model: clientConfig.modelName,
          compaction_semantic_index_entries: semanticIndex.entryCount,
          compaction_semantic_index_chars: semanticIndex.charCount,
          compaction_semantic_index_omitted_entries:
              semanticIndex.omittedEntryCount,
          /**
             * Per-call model attribution for usage consumers (the subagent
             * usage-capture handler): the summarizer's model can differ from
             * the agent's primary, and providers that emit no `ls_model_name`
             * would otherwise be billed against the primary config's model.
             * Omitted for self-summarize (no explicit model — the primary
             * config fallback is then correct). `tryFallbackProviders`
             * overrides this per fallback attempt; `INVOKED_PROVIDER` is
             * stamped by `attemptInvoke` itself.
             */
          ...(clientConfig.modelName != null && clientConfig.modelName !== ''
            ? { [Constants.INVOKED_MODEL]: clientConfig.modelName }
            : {}),
          /** Entry-captured breaker epoch: the wire consumer epoch-gates
             * old-run summary chunks exactly like model-attempt chunks —
             * without the stamp, a straggling summary from a failed run
             * reads as current and could abort the next run's controller. */
          ...(entryBreakerEpoch != null
            ? { [STREAM_LIMIT_EPOCH_KEY]: entryBreakerEpoch }
            : {}),
        },
      }
      : undefined;

    /** Rechecked after the pre-call awaits (dispatchRunStep,
     * ON_SUMMARIZE_START, PreCompact): a sibling can trip the breaker while
     * this node is paused in one of them, and a provider that doesn't
     * synchronously reject an aborted signal would still start the
     * summarization call. Mirrors the model node's pre-invoke recheck. */
    {
      const preCallTrip = findStreamLimitAbortReason(
        entryBreakerSignal,
        config?.signal
      );
      if (preCallTrip != null) {
        throw preCallTrip;
      }
    }
    const {
      text: rawText,
      usage: summaryUsage,
      provider: summaryProvider,
      model: summaryModel,
      errorCode,
    } = await executeSummarizationWithFallback({
      agentContext,
      messages: windowMessages,
      requiredHeadLength: messagesToRefine.length,
      onUsage: attempt => {
        attemptUsage.push(attempt);
        runStep.usage = buildSummarizationStepUsage(attemptUsage);
      },
      clientConfig,
      summarizeConfig,
      stepId,
      usePromptCache: isSelfSummarizeModel && hasPromptCache,
      shrinkInput: request.reason === 'overflow',
      semanticIndex,
      log,
      graph,
    });

    if (errorCode != null) {
      const preservationReason =
        PRESERVATION_REASONS[request.reason ?? 'trigger'] ??
        'automatic compaction';
      log(
        'warn',
        `Summarization failed during ${preservationReason}; keeping history unchanged`
      );
      agentContext.markSummarizationTriggered(state.messages.length);
      agentContext.recordSummarizationFailure();
      /**
       * The run step was already dispatched, so it has to be resolved here or
       * consumers tracking step lifecycle keep an unfinished placeholder for
       * the rest of the run.
       */
      if (graph.closeRunStep != null) {
        await graph.closeRunStep(stepId, 'failed', runnableConfig);
      } else {
        await graph.dispatchRunStepCompleted(
          stepId,
        {
          type: 'summary',
          summary: placeholderSummary,
        } satisfies t.SummaryCompleted,
        runnableConfig
        );
      }
      if (runnableConfig) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            id: stepId,
            agentId: request.agentId,
            error: `Summarization failed during ${preservationReason}; conversation history was preserved`,
            errorCode,
            ...(attemptUsage.length > 0 ? {attemptUsage: [...attemptUsage]} : {}),
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      return { summarizationRequest: undefined };
    }

    if (!rawText) {
      /**
       * An empty summary compacts nothing, so the state the pruner sees next
       * is byte-identical to the one that just triggered. Resetting the guard
       * to `0` here made `shouldSkipSummarization` answer `false` forever,
       * and the agent node re-triggered on that unchanged state until the
       * graph hit its recursion cap — a loop of empty summary steps, each one
       * a billed model call. Recording the current count keeps the guard
       * honest; the failure tally bounds the retries once new messages do
       * arrive and legitimately lift it.
       */
      agentContext.markSummarizationTriggered(state.messages.length);
      agentContext.recordSummarizationFailure();
      log('warn', 'Summarization produced empty output', {
        failures: agentContext.summarizationFailures,
        messagesToRefineCount: messagesToRefine.length,
      });
      if (runnableConfig) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            id: stepId,
            agentId: request.agentId,
            error: 'Summarization produced empty output',
            errorCode: 'unknown',
            ...(attemptUsage.length > 0 ? {attemptUsage: [...attemptUsage]} : {}),
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      await graph.closeRunStep?.(stepId, 'failed', runnableConfig);
      return { summarizationRequest: undefined };
    }

    const postCallTrip = findStreamLimitAbortReason(
      entryBreakerSignal,
      config?.signal
    );
    if (postCallTrip != null) throw postCallTrip;
    const summaryText = enrichSummary(rawText, messagesToRefine);

    const tokenCount = await computeSummaryTokenCount(
      summaryText,
      agentContext
    );

    const commitTrip = findStreamLimitAbortReason(
      entryBreakerSignal,
      config?.signal
    );
    if (commitTrip != null) throw commitTrip;
    runnableConfig?.signal?.throwIfAborted();
    entryBreakerSignal?.throwIfAborted();
    if (messagesToRetain.length > 0) {
      agentContext.setSummary(summaryText, tokenCount, {
        precedesMessages: true,
      });
    } else {
      agentContext.setSummary(summaryText, tokenCount);
    }

    log('info', 'Summary persisted');
    log('debug', 'Summary details', {
      summaryTokens: tokenCount,
      textLength: summaryText.length,
      messagesCompacted: messagesToRefine.length,
      summaryVersion: agentContext.summaryVersion,
      ...(summaryUsage != null
        ? {
          input_tokens: summaryUsage.input_tokens,
          output_tokens: summaryUsage.output_tokens,
          cache_read: summaryUsage.input_token_details?.cache_read,
          cache_creation: summaryUsage.input_token_details?.cache_creation,
        }
        : {}),
    });

    const summaryBlock = buildSummaryBlock({
      summaryText,
      tokenCount,
      coverage: resolveSummaryCoverage(messagesToRetain),
      stepId,
      stepIndex: runStep.index,
      modelName: summaryModel,
      provider: summaryProvider,
      summaryVersion: agentContext.summaryVersion,
    });

    await dispatchCompletionEvents({
      graph,
      runnableConfig,
      stepId,
      summaryBlock,
      agentContext,
      runStep,
      summaryUsage,
      attemptUsage,
      agentId: request.agentId,
      messagesAfterCount: messagesToRetain.length,
    });

    /**
     * `dispatchCompletionEvents` calls `rebuildTokenMapAfterSummarization({})`
     * which resets the dedupe baseline to 0 — correct under the legacy
     * "remove-all only" shape where no messages survived, but stale once
     * the recency window keeps a tail.  Realign the baseline to the
     * surviving tail length so a subsequent prune cycle on the unchanged
     * tail short-circuits via `shouldSkipSummarization` instead of
     * looping back into another summarize call.
     */
    agentContext.markSummarizationTriggered(messagesToRetain.length);

    agentContext.pendingOriginalToolContent = undefined;

    return {
      summarizationRequest: undefined,
      messages:
        messagesToRetain.length > 0
          ? [createRemoveAllMessage(), ...messagesToRetain]
          : [createRemoveAllMessage()],
      /** The run's result, for trace roots that otherwise see no reply. */
      ...(request.reason === 'manual' ? { manualSummary: summaryText } : {}),
    };
  };
}

/** Extracts text from an LLM response, skipping reasoning/thinking blocks. */
function extractResponseText(response: { content: string | object }): string {
  const { content } = response;
  if (typeof content === 'string') {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return '';
  }
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === 'string') {
      parts.push(block);
      continue;
    }
    if (block == null || typeof block !== 'object') {
      continue;
    }
    const rec = block as Record<string, unknown>;
    if (
      rec.type === ContentTypes.THINKING ||
      rec.type === ContentTypes.REASONING_CONTENT ||
      rec.type === 'redacted_thinking'
    ) {
      continue;
    }
    if (rec.type === 'text' && typeof rec.text === 'string') {
      parts.push(rec.text);
    }
  }
  return parts.join('').trim();
}

class ToolOnlySummaryError extends ReasoningOnlySummaryError {
  constructor() {
    super();
    this.name = 'ToolOnlySummaryError';
    this.message = 'The summarizer requested tools instead of summary text';
  }
}

function validateSummaryResponse(response: BaseMessage | undefined): void {
  if (response == null) return;
  if (isReasoningOnlySummary(response)) throw new ReasoningOnlySummaryError();
  if (extractResponseText(response) === '' && 'tool_calls' in response && Array.isArray(response.tool_calls) && response.tool_calls.length > 0) {
    const kwargs = response.additional_kwargs as {refusal?: string} | undefined;
    if (kwargs?.refusal != null || getProviderStopReasons(response).some(reason =>
      reason !== 'end_turn' && reason !== 'stop' && reason !== 'tool_use' && reason !== 'tool_calls'
    )) return;
    throw new ToolOnlySummaryError();
  }
}

function isReasoningOnlySummary(response: BaseMessage): boolean {
  const kwargs = response.additional_kwargs as (BaseMessage['additional_kwargs'] & {refusal?: string}) | undefined;
  if (kwargs?.refusal != null || getProviderStopReasons(response).some(
    reason => reason !== 'end_turn' && reason !== 'stop'
  )) return false;
  if ('tool_calls' in response && Array.isArray(response.tool_calls) && response.tool_calls.length > 0) return false;
  if ('invalid_tool_calls' in response && Array.isArray(response.invalid_tool_calls) && response.invalid_tool_calls.length > 0) return false;
  let hasReasoning = kwargs != null && hasReasoningKwargs(response);
  if (typeof response.content === 'string') return hasReasoning && response.content.trim() === '';
  if (!Array.isArray(response.content)) return false;
  for (const block of response.content) {
    if (isReasoningContentBlock(block)) {
      hasReasoning ||= hasMeaningfulReasoningContentBlock(block);
      continue;
    }
    if (block.type !== 'text' || typeof block.text !== 'string' || block.text.trim() !== '') return false;
  }
  return hasReasoning;
}

/** Creates an `onChunk` callback that dispatches `ON_SUMMARIZE_DELTA` events for streaming. */
export function createSummarizationChunkHandler({
  stepId,
  config,
  provider,
  reasoningKey = 'reasoning_content',
  graph,
  pendingDeltas,
}: {
  stepId?: string;
  config?: RunnableConfig;
  provider?: t.ProviderName;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  graph?: StreamLimitState & { getBreakerController?: () => AbortController };
  pendingDeltas?: t.SummarizeDeltaEvent[];
}): OnChunk | undefined {
  if (stepId == null || stepId === '' || !config) {
    return undefined;
  }
  const limitMetadata = config.metadata as Record<string, unknown> | undefined;
  let bufferedAttempt: unknown;
  return (chunk, attemptMetadata) => {
    if (
      pendingDeltas != null &&
      bufferedAttempt !== attemptMetadata?.[STREAM_LIMIT_ATTEMPT_KEY]
    ) {
      pendingDeltas.length = 0;
      bufferedAttempt = attemptMetadata?.[STREAM_LIMIT_ATTEMPT_KEY];
    }
    /**
     * Charged as `producer` so it PAIRS with the run's registered stream
     * handler: inside a live run the provider callbacks route each chunk
     * through the graph's `on_chat_model_stream` consumer too, and two
     * same-side consumer claims would charge the chunk twice — halving an
     * opt-in event cap. Standalone callers (tests, custom hosts, runs with
     * no registered handler) still get single-sided enforcement from this
     * one claim. Summarization deliberately passes NO `context` to
     * `attemptInvoke`, whose onChunk branch would otherwise add a second
     * producer claim. A breach trips the run's shared breaker (the
     * ENTRY-captured controller when the adapter provides one) before
     * rethrowing: this producer claim wins the race, so the wire consumer's
     * breaker-aborting catch never fires for the same chunk — without the
     * trip here, sibling model calls, tools, and subagents would keep
     * consuming quota while the summary branch rejects.
     * `executeSummarizationWithFallback` rethrows past fallbacks, failing
     * the run like any other breach.
     */
    if (graph != null) {
      try {
        enforceStreamLimitsForWireChunk({
          graph,
          metadata: attemptMetadata ?? limitMetadata,
          chunk: chunk as Parameters<
            typeof enforceStreamLimitsForWireChunk
          >[0]['chunk'],
        });
      } catch (error) {
        if (error instanceof StreamLimitExceededError) {
          graph.getBreakerController?.().abort(error);
        }
        throw error;
      }
    }
    const protectionTrip = findStreamLimitAbortReason(
      graph?.getBreakerController?.().signal,
      config.signal
    );
    if (protectionTrip != null) throw protectionTrip;
    const chunkAny = chunk as Parameters<typeof getChunkContent>[0]['chunk'];
    const servingProvider =
      (attemptMetadata?.[Constants.INVOKED_PROVIDER] as
        | t.ProviderName
        | undefined) ?? provider;
    const raw = getChunkContent({
      chunk: chunkAny,
      provider: servingProvider,
      reasoningKey,
    });
    if (raw == null || (typeof raw === 'string' && !raw)) {
      return;
    }
    const contentBlocks: t.MessageContentComplex[] =
      typeof raw === 'string'
        ? [{ type: ContentTypes.TEXT, text: raw } as t.MessageContentComplex]
        : raw;

    const event: t.SummarizeDeltaEvent = {
      id: stepId,
      delta: {
        summary: {
          type: ContentTypes.SUMMARY,
          content: contentBlocks,
          provider: String(
            servingProvider ?? config.metadata?.summarization_provider ?? ''
          ),
          model: String(
            attemptMetadata?.[Constants.INVOKED_MODEL] ??
              config.metadata?.summarization_model ??
              ''
          ),
        },
      },
    };
    if (pendingDeltas != null) {
      pendingDeltas.push(event);
      return;
    }
    void safeDispatchCustomEvent(GraphEvents.ON_SUMMARIZE_DELTA, event, config);
  };
}

function traceConfig(
  config: RunnableConfig | undefined,
  stage: string
): RunnableConfig | undefined {
  if (!config) {
    return undefined;
  }
  return {
    ...config,
    runName: `summarization:${stage}`,
    metadata: { ...config.metadata, summarization: true, stage },
  };
}

export function applySummarizationHistoryCache(params: {
  messages: BaseMessage[];
  provider: t.ProviderName;
  enabled: boolean;
  promptCacheTtl?: PromptCacheTtl;
  bedrockModelId?: string;
}): BaseMessage[] {
  if (!params.enabled) {
    return params.messages;
  }
  if (params.provider === Providers.BEDROCK) {
    return addBedrockTailCacheControl(
      [...params.messages],
      resolveBedrockPromptCacheTtl(params.promptCacheTtl, params.bedrockModelId)
    );
  }
  if (
    params.provider !== Providers.ANTHROPIC &&
    params.provider !== Providers.OPENROUTER
  ) {
    return params.messages;
  }
  return addTailCacheControl(
    [...params.messages],
    resolvePromptCacheTtl(params.promptCacheTtl)
  );
}

export function resolveBedrockCompactionCacheModel(
  options: { applicationInferenceProfile?: string; model?: string } | undefined
): string | undefined {
  return options?.model;
}

/**
 * Cache-friendly compaction: sends raw conversation messages with the
 * summarization instruction appended as the final HumanMessage. Bound tool
 * definitions and the turn's system runnable preserve the cached prefix.
 * Tool output stays masked and the final instruction has reserved headroom.
 */
async function summarizeWithCacheHit({
  agentContext,
  onUsage,
  shrinkInput,
  outputTokens,
  clientOptions,
  model,
  tools,
  messages,
  requiredHeadLength,
  promptText,
  updatePromptText,
  priorSummaryText,
  semanticIndexAppendix,
  config,
  stepId,
  provider,
  reasoningKey,
  graph,
  usePromptCache,
  promptCacheTtl,
  bedrockModelId,
  log,
}: {
  agentContext: AgentContext;
  onUsage: (usage: Partial<UsageMetadata>) => void;
  shrinkInput?: boolean;
  outputTokens: number;
  clientOptions?: t.ClientOptions;
  model: t.ChatModel;
  tools?: t.GraphTools;
  messages: BaseMessage[];
  requiredHeadLength: number;
  promptText: string;
  updatePromptText?: string;
  priorSummaryText: string;
  semanticIndexAppendix?: string;
  config?: RunnableConfig;
  stepId?: string;
  provider: t.ProviderName;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  graph?: StreamLimitState & { getBreakerSignal?: () => AbortSignal };
  usePromptCache?: boolean;
  promptCacheTtl?: PromptCacheTtl;
  bedrockModelId?: string;
  log?: LogFn;
}): Promise<{ text: string; usage?: Partial<UsageMetadata> }> {
  const instruction = buildSummarizationInstruction(
    promptText,
    updatePromptText,
    priorSummaryText,
    semanticIndexAppendix,
    provider === agentContext.provider && agentContext.systemRunnable != null
  );

  const { count, system, budget } = await prepareSummarizationBudget({
    agentContext,
    provider,
    clientOptions,
    maxContextTokens: agentContext.maxContextTokens,
    outputTokens,
    contextBudgetMode: agentContext.summarizationBudgetMode,
    instruction,
    tools,
    config,
  });
  const scales = [1, 0.85, 0.65, 0.45, 0.3];
  let result: Partial<t.BaseGraphState> | undefined;
  const initialTokens = messages.reduce(
    (total, message) => total + count(message),
    0
  );
  let lastError: unknown;
  for (let attempt = 0; attempt < scales.length; attempt++) {
    const signalConfig = withBreakerSignal(
      traceConfig(config, 'cache_hit_compaction'),
      graph
    );
    signalConfig?.signal?.throwIfAborted();
    const boundedHistory = boundSummarizationInput({
      messages,
      requiredHeadLength,
      tokenCounter: count,
      budget: Math.floor(
        Math.min(
          budget,
          attempt === 0 && shrinkInput !== true ? budget : initialTokens
        ) * scales[attempt]
      ),
      provider,
      shrink: attempt > 0 || shrinkInput === true,
    });
    const windowWithCache = provider === Providers.BEDROCK
      ? applySummarizationHistoryCache({messages: boundedHistory, provider, enabled: usePromptCache === true, promptCacheTtl, bedrockModelId})
      : boundedHistory;
    const history = system == null ? windowWithCache : await system.invoke(windowWithCache, config);
    const cachedHistory =
      system != null || provider === Providers.BEDROCK
        ? history
        : applySummarizationHistoryCache({
          messages: history,
          provider,
          enabled: usePromptCache === true,
          promptCacheTtl,
          bedrockModelId,
        });
    const pendingDeltas: t.SummarizeDeltaEvent[] = [];
    try {
      result = await attemptInvoke(
        {
          model,
          messages: [...cachedHistory, new HumanMessage(instruction)],
          provider,
          onModelComplete: response => {
            const usage = readSummarizationUsage(response);
            if (usage != null) onUsage(usage);
          },
          onChunk: createSummarizationChunkHandler({
            stepId,
            config: traceConfig(config, 'cache_hit_compaction'),
            provider,
            reasoningKey,
            graph,
            pendingDeltas,
          }),
          streamLimitState: graph,
        },
        signalConfig
      );
      const response = result.messages?.[0];
      signalConfig?.signal?.throwIfAborted();
      validateSummaryResponse(response);
      if (response != null && extractResponseText(response) !== '') {
        await flushSummarizationDeltas(pendingDeltas, config, graph);
      }
      break;
    } catch (error) {
      signalConfig?.signal?.throwIfAborted();
      if (classifySummarizationError(error) !== 'context_exceeded') throw error;
      lastError = error;
      log?.('debug', 'Shrinking summarization context', {
        attempt: attempt + 1,
      });
    }
  }
  if (result == null) throw lastError;

  const responseMsg = result.messages?.[0];
  const text = responseMsg
    ? extractResponseText(responseMsg as { content: string | object })
    : '';
  const usage = readSummarizationUsage(responseMsg);
  const usageSource = responseMsg != null && 'usage_metadata' in responseMsg && responseMsg.usage_metadata != null
    ? 'usage_metadata' : 'response_metadata';
  const cacheDetails = (
    usage as
      | {
          input_token_details?: {
            cache_read?: number;
            cache_creation?: number;
          };
        }
      | undefined
  )?.input_token_details;
  log?.('debug', 'Summarization LLM usage', {
    source: usage == null ? 'none' : usageSource,
    input_tokens: usage?.input_tokens,
    output_tokens: usage?.output_tokens,
    ...(cacheDetails?.cache_read != null || cacheDetails?.cache_creation != null
      ? {
        'input_token_details.cache_read': cacheDetails.cache_read,
        'input_token_details.cache_creation': cacheDetails.cache_creation,
      }
      : {}),
  });
  return { text, usage };
}

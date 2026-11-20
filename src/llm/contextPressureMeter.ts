import { isProxy } from 'node:util/types';
import { ToolMessage } from '@langchain/core/messages';
import type { AIMessage, BaseMessage } from '@langchain/core/messages';
import type { ProviderPayloadMeasurement } from '@/llm/prepareProviderRequest';
import type * as t from '@/types';
import {
  REPLY_PRIMER_TOKENS,
  isSyntheticProviderContextMessage,
} from '@/messages';
import { readTokenMetadataProperty, apportionTokenCounts } from '@/utils';
import { createToolMessageUsageAccumulator } from '@/messages/budget';

interface ContextPressureUsage {
  contextBudget?: number;
  effectiveInstructionTokens?: number;
  remainingContextTokens?: number;
  calibrationRatio?: number;
}

interface ContextPressureMeterParams {
  provider?: t.ProviderName;
  tokenCounter?: t.TokenCounter;
  tokenCountCache?: ExactTokenCountCache;
  sourceMessages: BaseMessage[];
  retainedMessages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  contextUsage?: ContextPressureUsage | null;
  instructionTokens: number;
  calibrationRatio: number;
}

interface ProviderPayloadMeasureOptions {
  contextBudget?: number;
  forceRawRecount?: boolean;
}

interface ProviderMessageBaseline {
  message: BaseMessage;
  /** Lazily tokenized: reading it costs a full count, so only read it on a changed projection. */
  readonly rawTokens: number;
  accountingWeight: number;
}

interface ProviderBaselineAttribution {
  attributedByOrigin: number[];
  projectedBaseTokens: number;
}

export interface ContextPressureMeter {
  trackProjection(before: BaseMessage[], after: BaseMessage[]): BaseMessage[];
  trackClone(source: BaseMessage, clone: BaseMessage): void;
  measure(
    messages: BaseMessage[],
    options?: ProviderPayloadMeasureOptions
  ): ProviderPayloadMeasurement;
}

export interface ExactTokenCountCache {
  count(message: BaseMessage): number;
}

interface StableTokenSurface {
  content: string;
  messageType: string;
  role?: string;
  additionalType?: string;
}

interface ExactTokenCountCacheEntry {
  surface: StableTokenSurface;
  tokens: number;
}

function readDataProperty(
  owner: object,
  property: PropertyKey
): { own: boolean; safe: boolean; value?: unknown } {
  let descriptor: PropertyDescriptor | undefined;
  try {
    descriptor = Object.getOwnPropertyDescriptor(owner, property);
  } catch {
    return { own: false, safe: false };
  }
  if (descriptor == null) {
    return { own: false, safe: true };
  }
  return 'value' in descriptor
    ? { own: true, safe: true, value: descriptor.value }
    : { own: true, safe: false };
}

function readCacheMetadataProperty(
  owner: object,
  property: PropertyKey,
  path: string
): { safe: boolean; value?: unknown } {
  try {
    return {
      safe: true,
      value: readTokenMetadataProperty(owner, property, path),
    };
  } catch {
    return { safe: false };
  }
}

function getStableTokenSurface(
  message: BaseMessage
): StableTokenSurface | undefined {
  if (isProxy(message)) {
    return undefined;
  }
  const contentProperty = readDataProperty(message, 'content');
  if (
    !contentProperty.safe ||
    !contentProperty.own ||
    typeof contentProperty.value !== 'string'
  ) {
    return undefined;
  }
  const messageType = message.getType();
  const roleProperty = readDataProperty(message, 'role');
  const resolvedRoleProperty = roleProperty.own
    ? roleProperty
    : readCacheMetadataProperty(message, 'role', 'message.role');
  if (!resolvedRoleProperty.safe) {
    return undefined;
  }
  const role = resolvedRoleProperty.value;
  if (role != null && typeof role !== 'string') {
    return undefined;
  }

  const additionalKwargsProperty = readDataProperty(
    message,
    'additional_kwargs'
  );
  if (!additionalKwargsProperty.safe || !additionalKwargsProperty.own) {
    return undefined;
  }
  const rawAdditionalKwargs = additionalKwargsProperty.value;
  if (
    rawAdditionalKwargs == null ||
    typeof rawAdditionalKwargs !== 'object' ||
    isProxy(rawAdditionalKwargs)
  ) {
    return undefined;
  }
  const additionalKwargs = rawAdditionalKwargs;
  const rawToolCalls = readCacheMetadataProperty(
    additionalKwargs,
    'tool_calls',
    'additional_kwargs.tool_calls'
  );
  if (!rawToolCalls.safe || rawToolCalls.value != null) {
    return undefined;
  }
  const typeProperty = readDataProperty(additionalKwargs, 'type');
  if (!typeProperty.safe) {
    return undefined;
  }
  if (messageType === 'ai' || role === 'assistant') {
    const toolCallsProperty = readDataProperty(
      message as AIMessage,
      'tool_calls'
    );
    const resolvedToolCallsProperty = toolCallsProperty.own
      ? toolCallsProperty
      : readCacheMetadataProperty(message, 'tool_calls', 'message.tool_calls');
    if (!resolvedToolCallsProperty.safe) {
      return undefined;
    }
    const toolCalls = resolvedToolCallsProperty.value;
    if (
      toolCalls != null &&
      (isProxy(toolCalls) || !Array.isArray(toolCalls) || toolCalls.length > 0)
    ) {
      return undefined;
    }
    const functionCall = readDataProperty(additionalKwargs, 'function_call');
    if (!functionCall.safe || functionCall.value != null) {
      return undefined;
    }
  }

  let additionalType: string | undefined;
  if (messageType === 'tool') {
    const typeValue = typeProperty.value;
    if (typeValue != null && typeof typeValue !== 'string') {
      return undefined;
    }
    additionalType = typeof typeValue === 'string' ? typeValue : undefined;
  }

  return {
    content: contentProperty.value,
    messageType,
    ...(role != null && { role }),
    ...(additionalType != null && { additionalType }),
  };
}

function tokenSurfacesMatch(
  left: StableTokenSurface,
  right: StableTokenSurface
): boolean {
  return (
    left.content === right.content &&
    left.messageType === right.messageType &&
    left.role === right.role &&
    left.additionalType === right.additionalType
  );
}

/** Reuses exact counts only while every token-relevant stable field matches. */
export function createExactTokenCountCache(
  tokenCounter: t.TokenCounter
): ExactTokenCountCache {
  const entries = new WeakMap<BaseMessage, ExactTokenCountCacheEntry>();
  return {
    count(message): number {
      const surface = getStableTokenSurface(message);
      const cached = entries.get(message);
      if (
        surface != null &&
        cached != null &&
        tokenSurfacesMatch(cached.surface, surface)
      ) {
        return cached.tokens;
      }
      const tokens = tokenCounter(message);
      if (surface != null) {
        entries.set(message, { surface, tokens });
      } else {
        entries.delete(message);
      }
      return tokens;
    },
  };
}

function getProviderMessageOriginKey(message: BaseMessage): string | undefined {
  const type = message.getType();
  if (
    message instanceof ToolMessage &&
    typeof message.tool_call_id === 'string' &&
    message.tool_call_id.length > 0
  ) {
    return `tool:call:${message.tool_call_id}`;
  }
  if (typeof message.id === 'string' && message.id.length > 0) {
    return `${type}:id:${message.id}`;
  }
  return undefined;
}

/** Measures repeated provider projections while tokenizing each message object once. */
export function createContextPressureMeter({
  provider,
  tokenCounter,
  tokenCountCache,
  sourceMessages,
  retainedMessages,
  indexTokenCountMap,
  contextUsage,
  instructionTokens,
  calibrationRatio,
}: ContextPressureMeterParams): ContextPressureMeter {
  const tokenCounts = new WeakMap<BaseMessage, number>();
  const origins = new WeakMap<BaseMessage, number>();
  const baselineAttributions = new Map<number, ProviderBaselineAttribution>();
  let baseline: ProviderMessageBaseline[] | undefined;
  let baselineWeights: Record<string, number> | undefined;
  let totalBaselineWeight = 0;

  const count = (message: BaseMessage): number => {
    const cached = tokenCounts.get(message);
    if (cached != null) {
      return cached;
    }
    const tokens =
      tokenCountCache?.count(message) ?? tokenCounter?.(message) ?? 0;
    tokenCounts.set(message, tokens);
    return tokens;
  };

  if (
    contextUsage != null &&
    (tokenCounter != null || tokenCountCache != null)
  ) {
    const sourceIndices = new WeakMap<BaseMessage, number>();
    for (let i = 0; i < sourceMessages.length; i++) {
      sourceIndices.set(sourceMessages[i], i);
    }
    baseline = retainedMessages.map((message, index) => {
      const sourceIndex = sourceIndices.get(message);
      const indexedTokens =
        sourceIndex != null ? indexTokenCountMap[sourceIndex] : undefined;
      const hasIndexedTokens =
        indexedTokens != null &&
        Number.isFinite(indexedTokens) &&
        indexedTokens >= 0;
      const accountingWeight = hasIndexedTokens
        ? indexedTokens
        : count(message);
      if (!origins.has(message)) {
        origins.set(message, index);
      }
      return {
        message,
        /** Exact count is only needed as the subtrahend when a projection changed this message. */
        get rawTokens(): number {
          return count(message);
        },
        accountingWeight,
      };
    });
    baselineWeights = {};
    for (let i = 0; i < baseline.length; i++) {
      baselineWeights[i] = baseline[i].accountingWeight;
      totalBaselineWeight += baseline[i].accountingWeight;
    }
  }

  const trackClone = (source: BaseMessage, clone: BaseMessage): void => {
    const origin = origins.get(source);
    if (origin != null && !origins.has(clone)) {
      origins.set(clone, origin);
    }
  };

  const trackProjection = (
    before: BaseMessage[],
    after: BaseMessage[]
  ): BaseMessage[] => {
    if (baseline == null || before === after) {
      return after;
    }
    if (before.length === after.length) {
      for (let i = 0; i < after.length; i++) {
        if (
          before[i].getType() === after[i].getType() &&
          !isSyntheticProviderContextMessage(after[i])
        ) {
          trackClone(before[i], after[i]);
        }
      }
      return after;
    }

    const keyedOrigins = new Map<string, number | null>();
    for (const message of before) {
      const origin = origins.get(message);
      const key = getProviderMessageOriginKey(message);
      if (origin == null || key == null) {
        continue;
      }
      keyedOrigins.set(key, keyedOrigins.has(key) ? null : origin);
    }
    for (const message of after) {
      if (origins.has(message) || isSyntheticProviderContextMessage(message)) {
        continue;
      }
      const key = getProviderMessageOriginKey(message);
      const origin = key != null ? keyedOrigins.get(key) : undefined;
      if (origin != null) {
        origins.set(message, origin);
      }
    }
    return after;
  };

  const measure = (
    messages: BaseMessage[],
    options?: ProviderPayloadMeasureOptions
  ): ProviderPayloadMeasurement => {
    const contextBudget = options?.contextBudget ?? contextUsage?.contextBudget;
    const forceRawRecount = options?.forceRawRecount === true;
    const effectiveInstructionTokens =
      contextUsage?.effectiveInstructionTokens ??
      (forceRawRecount ? instructionTokens : undefined);
    if (
      (tokenCounter == null && tokenCountCache == null) ||
      contextBudget == null ||
      effectiveInstructionTokens == null
    ) {
      return { fits: true };
    }
    const availableMessageTokens = Math.max(
      0,
      contextBudget - effectiveInstructionTokens
    );
    let usageRatio = calibrationRatio > 0 ? calibrationRatio : 1;
    if (
      contextUsage?.calibrationRatio != null &&
      contextUsage.calibrationRatio > 0
    ) {
      usageRatio = contextUsage.calibrationRatio;
    }
    if (forceRawRecount) {
      usageRatio = Math.max(1, usageRatio);
    }
    const baselineRemaining = contextUsage?.remainingContextTokens;
    const accountedMessageTokens =
      !forceRawRecount &&
      baseline != null &&
      baselineRemaining != null &&
      Number.isFinite(baselineRemaining)
        ? availableMessageTokens -
          Math.min(availableMessageTokens, baselineRemaining)
        : undefined;
    const toolUsage = createToolMessageUsageAccumulator(provider);
    const toolUsageState: { error?: Error } = {};
    const addToolUsage = (
      message: BaseMessage,
      getRawTokens: () => number
    ): void => {
      if (toolUsageState.error != null) {
        return;
      }
      try {
        toolUsage.add(message, getRawTokens);
      } catch (error) {
        toolUsageState.error =
          error instanceof Error ? error : new Error(String(error));
      }
    };

    let projectedMessageTokens: number;
    if (
      accountedMessageTokens != null &&
      baseline != null &&
      baselineWeights != null
    ) {
      let attribution = baselineAttributions.get(availableMessageTokens);
      if (attribution == null) {
        const replyPrimerTokens = Math.round(REPLY_PRIMER_TOKENS * usageRatio);
        const attributableTokens =
          totalBaselineWeight > 0
            ? Math.min(
              Math.max(0, accountedMessageTokens - replyPrimerTokens),
              Math.round(totalBaselineWeight * usageRatio)
            )
            : 0;
        const apportionedTokens =
          totalBaselineWeight > 0
            ? apportionTokenCounts(
              baselineWeights,
              attributableTokens / totalBaselineWeight,
              attributableTokens
            )
            : {};
        attribution = {
          attributedByOrigin: baseline.map(
            (_, origin) => apportionedTokens[origin] || 0
          ),
          projectedBaseTokens: Math.max(
            replyPrimerTokens,
            accountedMessageTokens - attributableTokens
          ),
        };
        baselineAttributions.set(availableMessageTokens, attribution);
      }
      projectedMessageTokens = attribution.projectedBaseTokens;
      let newRawTokens = 0;
      const usedOrigins = new Set<number>();
      for (const message of messages) {
        const origin = origins.get(message);
        let rawTokens: number | undefined;
        const getRawTokens = (): number => (rawTokens ??= count(message));
        addToolUsage(message, getRawTokens);
        if (origin == null || usedOrigins.has(origin)) {
          newRawTokens += getRawTokens();
          continue;
        }
        usedOrigins.add(origin);
        const projectionDelta =
          message === baseline[origin].message
            ? 0
            : getRawTokens() - baseline[origin].rawTokens;
        projectedMessageTokens += Math.max(
          0,
          attribution.attributedByOrigin[origin] +
            Math.round(projectionDelta * usageRatio)
        );
      }
      projectedMessageTokens += Math.round(newRawTokens * usageRatio);
    } else {
      let rawTokens = REPLY_PRIMER_TOKENS;
      for (const message of messages) {
        const messageTokens = count(message);
        addToolUsage(message, () => messageTokens);
        rawTokens += messageTokens;
      }
      projectedMessageTokens = Math.round(rawTokens * usageRatio);
    }
    let measuredToolUsage:
      | Pick<
          t.TokenBudgetBreakdown,
          'toolMessageTokens' | 'toolMessageTokenCounts'
        >
      | undefined;
    try {
      measuredToolUsage =
        toolUsageState.error == null ? toolUsage.finish(usageRatio) : undefined;
    } catch (error) {
      toolUsageState.error =
        error instanceof Error ? error : new Error(String(error));
      measuredToolUsage = undefined;
    }
    return {
      fits: projectedMessageTokens <= availableMessageTokens,
      projectedMessageTokens,
      availableMessageTokens,
      contextBudget,
      effectiveInstructionTokens,
      ...measuredToolUsage,
      ...(toolUsageState.error == null
        ? {}
        : { toolMessageUsageError: toolUsageState.error }),
    };
  };

  return { trackProjection, trackClone, measure };
}

/**
 * Recovery policy for provider context-overflow rejections.
 *
 * Detection (`@/utils/errors`) answers "was this an overflow, and what did
 * the provider disclose?". This module answers the follow-up: "what budget
 * should the retry target?" — deliberately kept as pure functions so the
 * policy can be reasoned about and tested without a graph.
 *
 * On units: `maxContextTokens` is a **provider-space** budget. The pruner
 * converts it into its own raw estimate space by dividing by the
 * `calibrationRatio` it learns from reported usage. A provider-reported
 * ceiling is therefore applied verbatim — converting it here as well would
 * apply the same correction twice and prune toward roughly `limit / ratio²`,
 * silently discarding far more history than the overflow called for.
 *
 * `observedCalibrationRatio` is returned separately so the caller can seed
 * the pruner's conversion without folding the same correction into the
 * provider-space budget.
 */
import type { ContextOverflowInfo } from '@/utils/errors';
import type { ProviderName } from '@/types';
import { getContextOverflowInfo } from '@/utils/errors';

/** Fraction of the previous budget used when the provider named no ceiling. */
const BLIND_SHRINK_RATIO = 0.7;

/** Slack left below a known ceiling so the retry is not sized to the edge. */
const CEILING_HEADROOM_RATIO = 0.95;

/**
 * Fallback floor, used only when the instruction size is unknown. When it is
 * known the floor is derived from it instead, so a genuinely small model — a
 * 4k window, where a 95%-of-ceiling budget lands below this constant — can
 * still recover.
 */
const MIN_RECOVERY_BUDGET_TOKENS = 4_000;

/**
 * Room a corrected budget must leave above the instructions for the messages
 * themselves. Without it, a budget that merely clears the system prompt and
 * tool schemas is not a budget anything can be compacted into.
 */
export const MIN_MESSAGE_HEADROOM_TOKENS = 2_000;

/** Bound on forced-compaction retries per agent, per run. */
export const DEFAULT_MAX_OVERFLOW_RECOVERIES = 2;

export interface OverflowRecoveryPlan {
  /** Budget the retry should target, in provider token units when known. */
  budgetTokens?: number;
  /** What the provider disclosed. Carried through for logging. */
  info: ContextOverflowInfo;
  /**
   * Provider-reported message tokens divided by our own message estimate,
   * when both are known. Greater than 1 means we under-count relative to this
   * provider.
   *
   * Returned separately so the graph can seed the pruner's calibration;
   * applying it to this plan's budget as well would double-count. Fixed
   * instruction overhead is removed before deriving it from
   * `info.promptTokens`, never `info.requestedTokens`, since several providers
   * fold the completion allowance into the latter.
   */
  observedCalibrationRatio?: number;
}

export interface OverflowRecoveryParams {
  error: unknown;
  provider: ProviderName;
  /** Budget in force when the rejected prompt was built. */
  maxContextTokens?: number;
  /** Our own estimate of the prompt we actually sent. */
  estimatedPromptTokens?: number;
  /** Provider/local calibration already applied to the prompt estimate. */
  calibrationRatio?: number;
  /**
   * System prompt plus tool schemas — the part of the budget compaction
   * cannot touch. A corrected budget at or below this leaves no room for
   * messages, and the summarize node refuses to run, so recovery is declined
   * rather than entered.
   */
  instructionTokens?: number;
  /** Whether a model-backed summary can compact messages without a pruner. */
  canSummarize?: boolean;
  /**
   * Completion allowance the caller configured. Providers count it against
   * the same ceiling, so it has to come off the top when the error itself did
   * not break the total down.
   */
  configuredCompletionTokens?: number;
  /** Recoveries already attempted for this agent in this run. */
  attemptsSoFar: number;
  maxAttempts?: number;
}

function isUsable(value: number | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0;
}

/**
 * Converts a provider-space retry budget into the units consumed by a pruner
 * calibrated for another provider while preserving the same raw-token limit.
 */
export function translateRecoveryBudget(
  budgetTokens: number | undefined,
  sourceCalibrationRatio: number | undefined,
  targetCalibrationRatio: number | undefined
): number | undefined {
  if (
    !isUsable(budgetTokens) ||
    !isUsable(sourceCalibrationRatio) ||
    !isUsable(targetCalibrationRatio)
  ) {
    return budgetTokens;
  }
  return Math.floor(
    (budgetTokens * targetCalibrationRatio) / sourceCalibrationRatio
  );
}

/** Applies the conservative shrink used when no provider-space ceiling is usable. */
export function getBlindRecoveryBudget(
  maxContextTokens: number | undefined
): number | undefined {
  return isUsable(maxContextTokens)
    ? Math.floor(maxContextTokens * BLIND_SHRINK_RATIO)
    : undefined;
}

/**
 * The completion allowance the provider counted against the same ceiling,
 * when it reported both the total and the prompt portion. The retry budget
 * governs the prompt only, so this has to come off the ceiling first —
 * otherwise a large `maxTokens` keeps the request over the limit no matter
 * how far the prompt is compacted.
 */
function reservedForCompletion(
  info: ContextOverflowInfo,
  configuredCompletionTokens: number | undefined
): number {
  if (isUsable(info.requestedTokens) && isUsable(info.promptTokens)) {
    const difference = info.requestedTokens - info.promptTokens;
    if (difference > 0) {
      return difference;
    }
  }
  /**
   * No breakdown on offer. Fall back to what the caller configured, because
   * the provider still counts it: targeting the whole ceiling would leave the
   * retry at `prompt + maxTokens` and over the limit however far the prompt
   * is compacted.
   */
  return isUsable(configuredCompletionTokens) ? configuredCompletionTokens : 0;
}

function resolveTargetBudget(
  info: ContextOverflowInfo,
  maxContextTokens: number | undefined,
  estimatedPromptTokens: number | undefined,
  configuredCompletionTokens: number | undefined
): number | null {
  if (isUsable(info.limitTokens)) {
    const promptCeiling =
      info.limitTokens -
      reservedForCompletion(info, configuredCompletionTokens);
    /**
     * A completion allowance at or above the ceiling leaves nothing for the
     * prompt: even an empty one plus the requested output overruns the limit.
     * Compaction cannot fix that, so declining surfaces the real problem
     * instead of burning the recovery budget on retries that must fail.
     */
    if (promptCeiling <= 0) {
      return null;
    }

    return promptCeiling * CEILING_HEADROOM_RATIO;
  }
  if (isUsable(estimatedPromptTokens)) {
    return estimatedPromptTokens * BLIND_SHRINK_RATIO;
  }
  if (isUsable(maxContextTokens)) {
    return maxContextTokens * BLIND_SHRINK_RATIO;
  }
  return null;
}

/**
 * Decides whether a failed model call is a recoverable context overflow and,
 * if so, what budget the retry should be re-pruned against.
 *
 * Returns `null` when the error is something compaction cannot fix, or when
 * the per-run recovery budget is spent — in both cases the caller should let
 * its normal failure handling proceed.
 */
export function planContextOverflowRecovery({
  error,
  provider,
  maxContextTokens,
  estimatedPromptTokens,
  calibrationRatio,
  instructionTokens,
  canSummarize = false,
  configuredCompletionTokens,
  attemptsSoFar,
  maxAttempts = DEFAULT_MAX_OVERFLOW_RECOVERIES,
}: OverflowRecoveryParams): OverflowRecoveryPlan | null {
  if (attemptsSoFar >= maxAttempts) {
    return null;
  }

  const info = getContextOverflowInfo(error, {
    provider,
    estimatedPromptTokens,
    maxContextTokens,
  });
  if (info == null) {
    return null;
  }

  const currentCalibrationRatio = isUsable(calibrationRatio)
    ? calibrationRatio
    : 1;
  const estimatedMessageTokens =
    isUsable(estimatedPromptTokens) && isUsable(instructionTokens)
      ? estimatedPromptTokens - instructionTokens
      : estimatedPromptTokens;
  const observedMessageTokens =
    isUsable(info.promptTokens) && isUsable(instructionTokens)
      ? info.promptTokens - instructionTokens
      : info.promptTokens;
  const observedCalibrationRatio =
    isUsable(observedMessageTokens) && isUsable(estimatedMessageTokens)
      ? (observedMessageTokens / estimatedMessageTokens) *
        currentCalibrationRatio
      : undefined;

  const target = resolveTargetBudget(
    info,
    maxContextTokens,
    estimatedPromptTokens,
    configuredCompletionTokens
  );
  if (target == null) {
    const hasNumericBasis =
      isUsable(info.limitTokens) ||
      isUsable(maxContextTokens) ||
      isUsable(estimatedPromptTokens);
    return hasNumericBasis
      ? null
      : { info, observedCalibrationRatio, budgetTokens: undefined };
  }

  /**
   * A retry that does not actually shrink the prompt would just reproduce the
   * same rejection, so a ceiling that lands at or above the budget we already
   * had is replaced by a blind shrink.
   */
  const bounded =
    isUsable(maxContextTokens) && target >= maxContextTokens
      ? (getBlindRecoveryBudget(maxContextTokens) ?? target)
      : target;

  const budgetTokens = Math.floor(bounded);

  /**
   * Below the floor there is no usable budget left: either nothing survives
   * pruning, or the instructions alone fill the window, in which case the
   * summarize node refuses to run and the detour would bounce between the
   * agent and summarize nodes without ever shrinking the prompt. Declining
   * lets the existing "instructions exceed context budget" guidance surface
   * instead.
   */
  let floorTokens = MIN_RECOVERY_BUDGET_TOKENS;
  if (isUsable(instructionTokens)) {
    floorTokens = instructionTokens + MIN_MESSAGE_HEADROOM_TOKENS;
  } else if (canSummarize) {
    floorTokens = 1;
  }
  if (budgetTokens < floorTokens) {
    return null;
  }

  /**
   * Refuse to report a "recovery" that changes nothing: when the budget in
   * force is already at or below the target, re-pruning cannot free space.
   */
  if (isUsable(maxContextTokens) && budgetTokens >= maxContextTokens) {
    return null;
  }

  return { budgetTokens, info, observedCalibrationRatio };
}

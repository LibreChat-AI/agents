import type { FadingTier } from '@/types/graph';
import {
  calculateMaxToolCallInputChars,
  calculateMaxToolResultChars,
} from '@/utils/truncation';

/**
 * Version 2: exchange width counts only the current turn. Version 1 tiers may
 * have latched on history whose steps storage had merged into one assistant
 * message, so they are discarded and re-derived once.
 */
export const FADING_TIER_VERSION = 2;

/** Context pressure at which observation masking activates. */
export const PRESSURE_THRESHOLD_MASKING = 0.8;

/** Smallest token budget the ladder can shrink to; keeps the emergency floor near 200 chars. */
export const FADING_MIN_BUDGET_TOKENS = 170;

/** Floor for masked (consumed) tool results. */
export const MASKED_RESULT_MIN_CHARS = 300;

/** Fraction of a fresh result's cap that a masked (consumed) result keeps. */
const MASKED_RESULT_CAP_RATIO = 0.1;

/**
 * Pressure bands expressed as extra rungs on top of the fit rung, so the
 * budget factors land at ×0.5 (85 %), ×0.25 (90 %) and ×0.0625 (99 %).
 */
const PRESSURE_BAND_RUNGS: readonly (readonly [number, number])[] = [
  [0.99, 4],
  [0.9, 2],
  [0.85, 1],
];

export type FadingSignals = {
  /** calibratedTotal / pruningBudget, measured before any truncation. */
  contextPressure: number;
  /** (pruningBudget − instruction tokens) ÷ calibrationRatio, in raw token space. */
  effectiveRawTokens: number;
  summarizationEnabled: boolean;
  /** Largest number of parallel calls in one assistant exchange of the current turn. */
  toolExchangeWidth?: number;
  /** Recovery paths force at least this rung on the current window's ladder. */
  minRung?: number;
};

export type FadingCaps = {
  budgetTokens: number;
  /** Cap for tool results the model has not answered yet. */
  resultChars: number;
  /** Cap for consumed tool results; equals `resultChars` until masking activates. */
  consumedChars: number;
  /** Cap for historical tool-call inputs. */
  inputChars: number;
};

function floorBudgetTokens(window: number): number {
  return Math.min(FADING_MIN_BUDGET_TOKENS, window);
}

/** Tier for a conversation that has never faded: the whole window, nothing masked. */
export function createFadingTier(window: number): FadingTier {
  return { v: FADING_TIER_VERSION, budgetTokens: window, masked: false };
}

export function isFadingTier(value: unknown): value is FadingTier {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const { v, budgetTokens, masked, latched } = value as Partial<
    Record<keyof FadingTier, unknown>
  >;
  return (
    v === FADING_TIER_VERSION &&
    typeof budgetTokens === 'number' &&
    Number.isFinite(budgetTokens) &&
    budgetTokens > 0 &&
    typeof masked === 'boolean' &&
    (latched === undefined || latched === true)
  );
}

/** Deepest rung for a window: the point where the budget reaches its floor. */
export function maxFadingRung(window: number): number {
  const floor = floorBudgetTokens(window);
  if (!(floor > 0)) {
    return 0;
  }
  return Math.max(0, Math.ceil(Math.log2(window / floor)));
}

/** Token budget at a rung: the window halved per rung, never below the floor. */
export function fadingBudgetTokens(window: number, rung: number): number {
  return Math.max(floorBudgetTokens(window), Math.floor(window / 2 ** rung));
}

/**
 * Restores a tier persisted by the host. The budget is absolute, so a tier
 * survives a mid-run budget correction and the return to the normal window
 * on the next run; it is only clamped to the current window. Anything
 * invalid starts fresh.
 */
export function seedFadingTier(window: number, seed?: unknown): FadingTier {
  if (!isFadingTier(seed)) {
    return createFadingTier(window);
  }
  /** The ladder never produces a budget below its floor, so a persisted one
   * is corrupt; clamping it up keeps every cap positive. */
  const budgetTokens = Math.max(
    floorBudgetTokens(window),
    Math.min(seed.budgetTokens, window)
  );
  /** Only a tier that was tightened, masked, or clamped carries information.
   * A fresh tier re-seeded through a pruner rebuild must stay fresh, or it
   * would be reported and pin later runs to this window. */
  const informative =
    seed.latched === true ||
    seed.masked ||
    budgetTokens !== seed.budgetTokens ||
    budgetTokens < window;
  if (!informative) {
    return createFadingTier(window);
  }
  return {
    v: FADING_TIER_VERSION,
    budgetTokens,
    masked: seed.masked,
    latched: true,
  };
}

/** Shallowest rung whose fresh-result cap is at most `targetChars`. */
export function fadingRungForResultChars(
  window: number,
  targetChars: number,
  maxToolResultChars?: number
): number {
  const deepest = maxFadingRung(window);
  for (let rung = 0; rung < deepest; rung++) {
    const windowResultChars = calculateMaxToolResultChars(
      fadingBudgetTokens(window, rung)
    );
    const effectiveResultChars =
      maxToolResultChars == null
        ? windowResultChars
        : Math.min(windowResultChars, maxToolResultChars);
    if (effectiveResultChars <= targetChars) {
      return rung;
    }
  }
  return deepest;
}

/**
 * Shallowest rung at which the configured number of parallel tool-call inputs
 * and fresh results fit together within `rawTokens`, the effective budget in
 * raw token space. This prevents orphan repair from dropping results merely
 * because their shared assistant message used the room left by a configured
 * result cap.
 */
export function fadingRungForBudget(
  window: number,
  rawTokens: number,
  maxToolResultChars?: number,
  toolExchangeWidth = 1
): number {
  if (!(rawTokens > 0)) {
    return 0;
  }
  const targetChars = Math.floor(rawTokens) * 4;
  const width = Math.max(1, Math.floor(toolExchangeWidth));
  const deepest = maxFadingRung(window);
  for (let rung = 0; rung < deepest; rung++) {
    const budgetTokens = fadingBudgetTokens(window, rung);
    const windowResultChars = calculateMaxToolResultChars(budgetTokens);
    const resultChars =
      maxToolResultChars == null
        ? windowResultChars
        : Math.min(windowResultChars, maxToolResultChars);
    const inputChars = calculateMaxToolCallInputChars(budgetTokens);
    if (width * (resultChars + inputChars) <= targetChars) {
      return rung;
    }
  }
  return deepest;
}

/**
 * Shallowest rung at which one complete tool exchange (a fresh result plus its
 * call input) fits within `targetChars`. The emergency pass uses it so a small
 * configured result cap cannot satisfy the target while inputs stay uncapped.
 */
export function fadingRungForExchangeChars(
  window: number,
  targetChars: number,
  maxToolResultChars?: number
): number {
  const deepest = maxFadingRung(window);
  for (let rung = 0; rung < deepest; rung++) {
    const budgetTokens = fadingBudgetTokens(window, rung);
    const windowResultChars = calculateMaxToolResultChars(budgetTokens);
    const resultChars =
      maxToolResultChars == null
        ? windowResultChars
        : Math.min(windowResultChars, maxToolResultChars);
    if (
      resultChars + calculateMaxToolCallInputChars(budgetTokens) <=
      targetChars
    ) {
      return rung;
    }
  }
  return deepest;
}

/** Character caps for a tier; a pure function of `(budgetTokens, masked)`. */
export function resolveFadingCaps(
  tier: FadingTier,
  maxToolResultChars?: number
): FadingCaps {
  const { budgetTokens } = tier;
  const windowResultChars = calculateMaxToolResultChars(budgetTokens);
  const resultChars =
    maxToolResultChars != null
      ? Math.min(windowResultChars, maxToolResultChars)
      : windowResultChars;
  const consumedChars = tier.masked
    ? Math.min(
      resultChars,
      Math.max(
        MASKED_RESULT_MIN_CHARS,
        Math.floor(resultChars * MASKED_RESULT_CAP_RATIO)
      )
    )
    : resultChars;
  return {
    budgetTokens,
    resultChars,
    consumedChars,
    inputChars: calculateMaxToolCallInputChars(budgetTokens),
  };
}

/**
 * Advances a tier from this call's signals on the current window's ladder.
 * The budget only ever shrinks and masking only ever activates, which is the
 * hysteresis: drift in calibration, instruction overhead or message count
 * can escalate once at a rung boundary but never oscillate. Returns the same
 * object when nothing changes.
 *
 * - The fit rung guarantees the widest observed parallel tool exchange fits
 *   the effective budget as a complete input/result batch.
 * - Pressure bands add rungs on top of the fit rung when summarization is
 *   off, mirroring the staged budget factors of progressive context fading.
 * - A window larger than the latched budget (model switch) does not loosen
 *   the tier; only compaction, which rewrites the prefix anyway, resets it.
 */
export function resolveFadingTier(
  tier: FadingTier,
  window: number,
  signals: FadingSignals,
  maxToolResultChars?: number
): FadingTier {
  const fitRung = fadingRungForBudget(
    window,
    signals.effectiveRawTokens,
    maxToolResultChars,
    signals.toolExchangeWidth
  );
  const bandRung = signals.summarizationEnabled
    ? 0
    : (PRESSURE_BAND_RUNGS.find(
      ([threshold]) => signals.contextPressure >= threshold
    )?.[1] ?? 0);
  const rung = Math.min(
    maxFadingRung(window),
    Math.max(fitRung + bandRung, signals.minRung ?? 0)
  );
  const budgetTokens = Math.min(
    tier.budgetTokens,
    fadingBudgetTokens(window, rung)
  );
  const masked =
    tier.masked || signals.contextPressure >= PRESSURE_THRESHOLD_MASKING;
  if (budgetTokens === tier.budgetTokens && masked === tier.masked) {
    return tier;
  }
  return { v: FADING_TIER_VERSION, budgetTokens, masked, latched: true };
}

/**
 * Whether a tier carries information a host should persist: masking has
 * activated or the budget sits below the pruner's window. A fresh tier only
 * seeds what the next run derives on its own.
 */
export function isInformativeFadingTier(
  tier: FadingTier | undefined,
  window: number | undefined
): tier is FadingTier {
  if (tier == null) {
    return false;
  }
  return (
    tier.latched === true ||
    tier.masked ||
    (window != null && window > 0 && tier.budgetTokens < window)
  );
}

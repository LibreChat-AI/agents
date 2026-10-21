import type {
  Reranker,
  RerankKind,
  RerankResult,
  RerankEntry,
  RerankUsage,
  RankRequest,
  RerankReason,
  RerankAttempt,
  RerankOutcome,
  OutcomeReranker,
} from './types';
import type { DecisionFailure } from '../decisions/types';
import { DecisionError, isDecisionObject } from '../decisions/types';
import { withDecisionDeadline } from '../decisions/deadline';

/** Longest deadline a step may have; matches `withDecisionDeadline`. */
export const MAX_RERANK_DEADLINE_MS = 3_600_000;

/**
 * Input tokens one document costs a request, rounded up from Codegraph's
 * measurements on file summaries (Jev with its rubric and question ~520,
 * qwen3-reranker-8b ~140). Override per reranker for other document shapes.
 */
export const ESTIMATED_TOKENS_PER_DOCUMENT: Readonly<
  Record<RerankKind, number>
> = Object.freeze({
  systemone: 600,
  'cross-encoder': 200,
  bm25: 0,
  none: 0,
});

export function emptyRerankUsage(): RerankUsage {
  return {
    calls: 0,
    inputTokens: 0,
    outputTokens: 0,
    searchUnits: 0,
    cached: 0,
    estimatedTokens: 0,
    estimatedSearchUnits: 0,
  };
}

const USAGE_KEYS = [
  'calls',
  'inputTokens',
  'outputTokens',
  'searchUnits',
  'cached',
  'estimatedTokens',
  'estimatedSearchUnits',
] as const;

const count = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;

/** Every usage counter present as a finite number of at least zero. */
export function isRerankUsage(value: unknown): value is RerankUsage {
  if (!isDecisionObject(value)) {
    return false;
  }
  for (const key of USAGE_KEYS) {
    const counter = value[key];
    if (
      typeof counter !== 'number' ||
      !Number.isFinite(counter) ||
      counter < 0
    ) {
      return false;
    }
  }
  return true;
}

/** Adds `usage` into `sum`; anything that is not a positive finite count adds nothing. */
export function addRerankUsage(sum: RerankUsage, usage: unknown): RerankUsage {
  if (!isDecisionObject(usage)) {
    return sum;
  }
  for (const key of USAGE_KEYS) {
    sum[key] += count(usage[key]);
  }
  return sum;
}

/**
 * The usage `documents` documents sent and never answered are charged as. A
 * deadline aborts the client, not the host: the host received the request and
 * bills it, so counting only answered calls lets every timeout spend without
 * reaching a budget. A cross-encoder request is one search unit per 100
 * documents, as Cohere bills it. `bm25` and `none` cost nothing.
 */
export function estimatedUsage(
  kind: RerankKind,
  documents: number,
  tokensPerDocument: number = ESTIMATED_TOKENS_PER_DOCUMENT[kind]
): Pick<RerankUsage, 'estimatedTokens' | 'estimatedSearchUnits'> {
  const n = Math.max(0, Math.floor(count(documents)));
  if (n === 0 || kind === 'bm25' || kind === 'none') {
    return { estimatedTokens: 0, estimatedSearchUnits: 0 };
  }
  // Search units follow the document count alone, so a zero token estimate never hides them.
  return {
    estimatedTokens: n * count(tokensPerDocument),
    estimatedSearchUnits: kind === 'cross-encoder' ? Math.ceil(n / 100) : 0,
  };
}

/** An `estimatedTokensPerDocument` option: absent, or a finite count of at least zero. */
export const isTokenEstimate = (value: unknown): boolean =>
  value === undefined ||
  (typeof value === 'number' && Number.isFinite(value) && value >= 0);

/**
 * Prices in USD for one model; the SDK ships none. A rate is a finite number
 * of at least zero: `0` prices a dimension as free, while an absent or invalid
 * rate leaves usage of that dimension unpriced.
 */
export interface RerankPrice {
  inputPerMTok?: number;
  outputPerMTok?: number;
  perSearchUnit?: number;
}

/**
 * A caller-supplied price table: per kind, a model id, a `prefix*` pattern
 * (the longest match wins), or `*` for any model of that kind.
 */
export type RerankPrices = Readonly<
  Partial<Record<RerankKind, Readonly<Record<string, RerankPrice>>>>
>;

export interface RerankCost {
  /** Every attempt, estimates included. */
  usd: number;
  /**
   * Attempts that used something the table has no rate for (no entry, or an
   * entry without a rate for a dimension they used); their spend is not in `usd`.
   */
  unpriced: Array<{ provider: RerankKind; model: string }>;
}

function priceFor(
  table: Readonly<Record<string, RerankPrice>> | undefined,
  model: string
): RerankPrice | undefined {
  if (!table) {
    return undefined;
  }
  if (Object.hasOwn(table, model)) {
    return table[model];
  }
  let best: string | undefined;
  for (const key of Object.keys(table)) {
    if (
      key.endsWith('*') &&
      model.startsWith(key.slice(0, -1)) &&
      (best === undefined || key.length > best.length)
    ) {
      best = key;
    }
  }
  return best === undefined ? undefined : table[best];
}

/** A rate the table sets, or `undefined` when it sets none it can be priced by. */
const rateOf = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : undefined;

/**
 * One attempt's spend under `price`, or `undefined` when part of it has no
 * rate. Each reported dimension needs its own rate. An estimate books tokens
 * and search units together because the SDK cannot tell which one the host
 * bills, so it is priced by the rates the entry has and is unpriced only when
 * the entry has a rate for neither.
 */
function attemptCost(
  usage: RerankAttempt['usage'],
  price: unknown
): number | undefined {
  if (!isDecisionObject(price)) {
    return undefined;
  }
  const input = rateOf(price.inputPerMTok);
  const output = rateOf(price.outputPerMTok);
  const unit = rateOf(price.perSearchUnit);
  const tokens = count(usage.inputTokens);
  const outputTokens = count(usage.outputTokens);
  const units = count(usage.searchUnits);
  const estimatedTokens = count(usage.estimatedTokens);
  const estimatedUnits = count(usage.estimatedSearchUnits);
  if (
    (tokens > 0 && input === undefined) ||
    (outputTokens > 0 && output === undefined) ||
    (units > 0 && unit === undefined)
  ) {
    return undefined;
  }
  if (
    (estimatedTokens > 0 || estimatedUnits > 0) &&
    !(estimatedTokens > 0 && input !== undefined) &&
    !(estimatedUnits > 0 && unit !== undefined)
  ) {
    return undefined;
  }
  return (
    ((tokens + estimatedTokens) * (input ?? 0)) / 1e6 +
    (outputTokens * (output ?? 0)) / 1e6 +
    (units + estimatedUnits) * (unit ?? 0)
  );
}

/**
 * What an outcome's attempts cost under `prices`; reported and estimated
 * usage are priced alike. An attempt with usage the table has no rate for is
 * listed in `unpriced` and adds nothing to `usd`, never counted as free.
 */
export function rerankCost(
  outcome: Pick<RerankOutcome, 'attempts'>,
  prices: RerankPrices
): RerankCost {
  let usd = 0;
  const unpriced: RerankCost['unpriced'] = [];
  for (const attempt of outcome.attempts) {
    const tokens =
      count(attempt.usage.inputTokens) + count(attempt.usage.estimatedTokens);
    const output = count(attempt.usage.outputTokens);
    const units =
      count(attempt.usage.searchUnits) +
      count(attempt.usage.estimatedSearchUnits);
    if (tokens === 0 && output === 0 && units === 0) {
      continue;
    }
    const spent = attemptCost(
      attempt.usage,
      priceFor(prices[attempt.provider], attempt.model)
    );
    if (spent === undefined) {
      if (
        !unpriced.some(
          (seen) =>
            seen.provider === attempt.provider && seen.model === attempt.model
        )
      ) {
        unpriced.push({ provider: attempt.provider, model: attempt.model });
      }
      continue;
    }
    usd += spent;
  }
  return { usd: Math.round(usd * 1e9) / 1e9, unpriced };
}

function byScore(a: RerankEntry, b: RerankEntry): number {
  if (a.score === null) {
    return b.score === null ? a.index - b.index : 1;
  }
  if (b.score === null) {
    return -1;
  }
  return b.score - a.score || a.index - b.index;
}

/** Every input once, score descending, ties in input order, `null` last; a threshold adds `kept`. */
export function rankedEntries(
  scores: ReadonlyArray<number | null>,
  threshold?: number
): RerankEntry[] {
  return scores
    .map(
      (score, index): RerankEntry =>
        threshold === undefined
          ? { index, score }
          : { index, score, kept: score !== null && score >= threshold }
    )
    .sort(byScore);
}

const KINDS: ReadonlySet<string> = new Set<RerankKind>([
  'systemone',
  'cross-encoder',
  'bm25',
  'none',
]);

export const isRerankKind = (value: unknown): value is RerankKind =>
  typeof value === 'string' && KINDS.has(value);

/**
 * Whether a permutation is in the order `rankedEntries` produces: score
 * descending, ties in input order, `null` last, with `kept` a boolean when set.
 */
export function isRankedOrder(results: readonly RerankEntry[]): boolean {
  for (let index = 0; index < results.length; index++) {
    const entry = results[index];
    if (entry.kept !== undefined && typeof entry.kept !== 'boolean') {
      return false;
    }
    if (index > 0 && byScore(results[index - 1], entry) >= 0) {
      return false;
    }
  }
  return true;
}

/** Whether `results` is a full permutation of `count` documents with finite or `null` scores. */
export function isRerankPermutation(
  results: unknown,
  count: number
): results is RerankEntry[] {
  if (!Array.isArray(results) || results.length !== count) {
    return false;
  }
  const seen = new Uint8Array(count);
  for (const entry of results as unknown[]) {
    if (!isDecisionObject(entry)) {
      return false;
    }
    const { index, score } = entry;
    if (
      typeof index !== 'number' ||
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= count ||
      seen[index] === 1 ||
      (score !== null && (typeof score !== 'number' || !Number.isFinite(score)))
    ) {
      return false;
    }
    seen[index] = 1;
  }
  return true;
}

/** Reads the signal now; a function call keeps TypeScript from narrowing a later read away. */
export const isAborted = (signal: AbortSignal | undefined): boolean =>
  signal?.aborted === true;

/** How many documents a request carries, read defensively at the public boundary. */
export const documentCount = (request: unknown): number =>
  isDecisionObject(request) && Array.isArray(request.documents)
    ? request.documents.length
    : 0;

/** Identity order with `null` scores: what a caller serves when nothing answered. */
export function identityOutcome(
  documents: number,
  options: {
    reason: string;
    provider?: RerankKind;
    model?: string;
    durationMs?: number;
  }
): RerankOutcome {
  return {
    results: Array.from({ length: documents }, (_, index) => ({
      index,
      score: null,
    })),
    provider: options.provider ?? 'none',
    model: options.model ?? '',
    status: 'skipped',
    reason: options.reason,
    attempts: [],
    durationMs: options.durationMs ?? 0,
    usage: emptyRerankUsage(),
  };
}

export interface StepOutcome {
  kind: RerankKind;
  documents: number;
  startedAt: number;
  model: string;
  /** One score per document when the step answered. */
  scores?: ReadonlyArray<number | null>;
  /** A failure; identity order is returned. */
  reason?: string;
  usage?: RerankUsage;
  /** False when no request was made, so the attempt is `skipped` rather than `failed`. */
  called?: boolean;
  threshold?: number;
  pinVerified?: boolean;
}

export const elapsedMs = (startedAt: number): number =>
  Math.round(performance.now() - startedAt);

function attemptStatus(
  ok: boolean,
  called: boolean | undefined
): RerankAttempt['status'] {
  if (ok) {
    return 'ok';
  }
  return called === false ? 'skipped' : 'failed';
}

/** One reranker's own outcome: `ok` with its scores, or `skipped` in identity order with a reason. */
export function stepOutcome(step: StepOutcome): RerankOutcome {
  const durationMs = elapsedMs(step.startedAt);
  const usage = step.usage ?? emptyRerankUsage();
  const ok = step.reason === undefined && step.scores !== undefined;
  const reason = ok ? null : (step.reason ?? 'error');
  const attempt: RerankAttempt = {
    provider: step.kind,
    model: step.model,
    status: attemptStatus(ok, step.called),
    reason,
    durationMs,
    usage: { ...usage },
  };
  const results =
    ok && step.scores
      ? rankedEntries(step.scores, step.threshold)
      : identityOutcome(step.documents, { reason: reason ?? 'error' }).results;
  return {
    results,
    provider: step.kind,
    model: step.model,
    status: ok ? 'ok' : 'skipped',
    reason,
    attempts: [attempt],
    durationMs,
    usage,
    ...(step.pinVerified !== undefined
      ? { pinVerified: step.pinVerified }
      : {}),
    ...(ok && step.threshold !== undefined
      ? { threshold: step.threshold }
      : {}),
  };
}

const REASON_FOR_FAILURE: Readonly<
  Partial<Record<DecisionFailure, RerankReason>>
> = Object.freeze({
  unauthorized: 'unauthorized',
  quota: 'quota',
  rate_limited: 'rate_limited',
  timeout: 'timeout',
  aborted: 'aborted',
  malformed_response: 'bad_response',
  unsupported_question: 'unsupported',
  unsupported_mode: 'unsupported',
});

/** A failure as an outcome reason; the caller decides whether a fired deadline means `timeout` or `aborted`. */
export function rerankReasonOf(error: unknown): RerankReason {
  if (!(error instanceof DecisionError)) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      return 'timeout';
    }
    return error instanceof Error && error.name === 'AbortError'
      ? 'aborted'
      : 'error';
  }
  if (error.circuitOpen) {
    return 'circuit_open';
  }
  return REASON_FOR_FAILURE[error.failure] ?? 'error';
}

/** A positive integer deadline no longer than an hour, or `undefined` when it is not one. */
export function validDeadline(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return undefined;
  }
  const ms = Math.ceil(value);
  return ms <= MAX_RERANK_DEADLINE_MS ? ms : undefined;
}

/**
 * `test` holds for every slot of `values`, holes included. `some` and `every`
 * skip the holes of a sparse array, which a spread, an index, or
 * `JSON.stringify` later reads as `undefined` or `null`.
 */
export function everySlot<T>(
  values: readonly T[],
  test: (value: T | undefined) => boolean
): boolean {
  for (let index = 0; index < values.length; index++) {
    if (!test(values[index])) {
      return false;
    }
  }
  return true;
}

/** A non-empty query and string documents, or the reason the request cannot be ranked. */
export function rankRequestProblem(request: unknown): RerankReason | undefined {
  if (
    !isDecisionObject(request) ||
    !Array.isArray(request.documents) ||
    typeof request.query !== 'string' ||
    !request.query.trim() ||
    !everySlot(
      request.documents as unknown[],
      (document) => typeof document === 'string'
    )
  ) {
    return 'invalid_request';
  }
  if (request.documents.length === 0) {
    return 'empty';
  }
  if (
    request.deadlineMs !== undefined &&
    validDeadline(request.deadlineMs) === undefined
  ) {
    return 'invalid_request';
  }
  return undefined;
}

/**
 * The never-throwing layer over a rejecting `Reranker`: its full ranking
 * becomes an `ok` outcome, a rejection becomes a reason, and a ranking that
 * is not a full permutation in descending score order is `invalid_results`.
 * Usage counts one call, plus every retry the reranker reports through
 * `onRequest`, with whatever tokens it reported, or none when an open breaker
 * (`circuitOpen`) refused it before dispatch. The request's `deadlineMs` and
 * `signal` bound the wait even for a reranker that ignores them.
 */
export function toOutcomeReranker(
  reranker: Reranker,
  options: { kind: RerankKind; model?: string }
): OutcomeReranker {
  const { kind } = options;
  const model = options.model ?? '';
  return {
    id: reranker.id,
    kind,
    model,
    async rank(request: RankRequest): Promise<RerankOutcome> {
      const startedAt = performance.now();
      const documents = documentCount(request);
      const problem = rankRequestProblem(request);
      if (problem !== undefined) {
        return stepOutcome({
          kind,
          documents,
          startedAt,
          model,
          reason: problem,
          called: false,
        });
      }
      if (isAborted(request.signal)) {
        return stepOutcome({
          kind,
          documents,
          startedAt,
          model,
          reason: 'aborted',
          called: false,
        });
      }
      const usage = emptyRerankUsage();
      usage.calls = 1;
      /** Requests the reranker reported sending; the first is the call counted above. */
      let requests = 0;
      let settled = false;
      try {
        const deadlineMs = validDeadline(request.deadlineMs);
        const call = (signal: AbortSignal | undefined): Promise<RerankResult> =>
          reranker.rerank({
            query: request.query,
            documents: request.documents,
            ...(signal !== undefined ? { signal } : {}),
            ...(deadlineMs !== undefined ? { timeoutMs: deadlineMs } : {}),
            onRequest: () => {
              requests += 1;
              if (requests > 1 && !settled) {
                usage.calls += 1;
              }
            },
          });
        // Bounded here too, so a reranker that ignores its deadline and signal cannot hold the call.
        let response: RerankResult;
        if (deadlineMs === undefined && request.signal === undefined) {
          response = await call(undefined);
        } else {
          response = await withDecisionDeadline(
            reranker.id,
            deadlineMs ?? MAX_RERANK_DEADLINE_MS,
            request.signal,
            (signal, waitFor) =>
              waitFor(Promise.resolve().then(() => call(signal)))
          );
        }
        settled = true;
        usage.inputTokens = count(response.usage?.inputTokens);
        usage.outputTokens = count(response.usage?.outputTokens);
        const echoed =
          typeof response.model === 'string' && response.model !== ''
            ? response.model
            : model;
        const results: unknown = response.results;
        if (!isRerankPermutation(results, documents)) {
          return stepOutcome({
            kind,
            documents,
            startedAt,
            model: echoed,
            reason: 'invalid_results',
            usage,
          });
        }
        const scores: Array<number | null> = new Array(documents).fill(null);
        let previous = Infinity;
        for (const entry of results) {
          if (entry.score === null || entry.score > previous) {
            return stepOutcome({
              kind,
              documents,
              startedAt,
              model: echoed,
              reason: 'invalid_results',
              usage,
            });
          }
          previous = entry.score;
          scores[entry.index] = entry.score;
        }
        return stepOutcome({
          kind,
          documents,
          startedAt,
          model: echoed,
          scores,
          usage,
        });
      } catch (error) {
        settled = true;
        // An open breaker refuses before dispatch: nothing was sent, so nothing is charged.
        const sent = !(error instanceof DecisionError && error.circuitOpen);
        if (!sent) {
          usage.calls = 0;
        }
        const reason = isAborted(request.signal)
          ? 'aborted'
          : rerankReasonOf(error);
        return stepOutcome({
          kind,
          documents,
          startedAt,
          model,
          reason,
          called: sent,
          usage:
            sent && (reason === 'timeout' || reason === 'aborted')
              ? { ...usage, ...estimatedUsage(kind, documents) }
              : usage,
        });
      }
    },
  };
}

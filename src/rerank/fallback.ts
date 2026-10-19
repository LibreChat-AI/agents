import type {
  RerankKind,
  RerankUsage,
  RankRequest,
  RerankAttempt,
  RerankOutcome,
  OutcomeReranker,
} from './types';
import {
  elapsedMs,
  isAborted,
  documentCount,
  validDeadline,
  addRerankUsage,
  estimatedUsage,
  identityOutcome,
  emptyRerankUsage,
  rankRequestProblem,
  isRerankPermutation,
} from './outcome';
import { DecisionError, isDecisionObject } from '../decisions/types';

/** Identity order, never a request: the last step of a chain, or the whole chain when nothing is configured. */
export function noneReranker(
  options: { reason?: string } = {}
): OutcomeReranker {
  const reason = options.reason ?? 'disabled';
  return {
    id: 'none',
    kind: 'none',
    model: '',
    async rank(request: RankRequest): Promise<RerankOutcome> {
      return identityOutcome(documentCount(request), { reason });
    },
  };
}

export interface FallbackStep {
  reranker: OutcomeReranker;
  /** This step's own deadline, so a hung step cannot starve the next one. */
  deadlineMs?: number;
}

export interface FallbackOptions {
  /**
   * Returns a reason to skip a step without a request (a backoff, a budget,
   * an unverified pin), or nothing to run it. Not consulted for `none` steps.
   * A gate that throws skips its step with `error`.
   */
  gate?: (
    reranker: OutcomeReranker
  ) => string | null | undefined | Promise<string | null | undefined>;
  /** Sees every step that ran, for backoff bookkeeping; a throw is ignored. */
  onAttempt?: (reranker: OutcomeReranker, outcome: RerankOutcome) => void;
}

export interface FallbackReranker extends OutcomeReranker {
  readonly steps: readonly FallbackStep[];
}

const STATUSES: ReadonlySet<string> = new Set(['ok', 'degraded', 'skipped']);

/** Grace after a step's deadline for a reranker that honors its abort to report its own timeout and spend. */
export const STEP_GRACE_MS = 50;

function usageOf(value: unknown): RerankUsage {
  return addRerankUsage(emptyRerankUsage(), value);
}

/** A step that broke its contract or outlived its deadline, charged what it reported or an estimate. */
function failedStep(
  reranker: OutcomeReranker,
  documents: number,
  reason: string,
  durationMs: number,
  usage: RerankUsage = emptyRerankUsage()
): RerankOutcome {
  const attempt: RerankAttempt = {
    provider: reranker.kind,
    model: reranker.model,
    status: 'failed',
    reason,
    durationMs,
    usage: { ...usage },
  };
  return {
    ...identityOutcome(documents, {
      reason,
      provider: reranker.kind,
      model: reranker.model,
      durationMs,
    }),
    attempts: [attempt],
    usage,
  };
}

/** What a step that never returned is charged: every document it was sent, at the estimate. */
function abandonedUsage(kind: RerankKind, documents: number): RerankUsage {
  const estimate = estimatedUsage(kind, documents);
  if (estimate.estimatedTokens === 0) {
    return emptyRerankUsage();
  }
  return {
    ...emptyRerankUsage(),
    calls: kind === 'systemone' ? documents : 1,
    ...estimate,
  };
}

function isOutcome(value: unknown, documents: number): value is RerankOutcome {
  return (
    isDecisionObject(value) &&
    typeof value.status === 'string' &&
    STATUSES.has(value.status) &&
    isRerankPermutation(value.results, documents) &&
    Array.isArray(value.attempts) &&
    value.attempts.every(isDecisionObject)
  );
}

/** The step's deadline reached it as an abort; to the chain's caller it is the step's timeout. */
function asTimeout(outcome: RerankOutcome): RerankOutcome {
  const timeout = (reason: string | null): string | null =>
    reason === 'aborted' ? 'timeout' : reason;
  return {
    ...outcome,
    reason: timeout(outcome.reason),
    attempts: outcome.attempts.map((attempt) => ({
      ...attempt,
      reason: timeout(attempt.reason),
    })),
  };
}

/**
 * One step under its own deadline: an abort signal the reranker should honor,
 * and a timer that ends the wait regardless, so a step that ignores its abort
 * cannot hold the chain.
 */
async function runStep(
  reranker: OutcomeReranker,
  query: string,
  documents: readonly string[],
  deadlineMs: number | undefined,
  outer: AbortSignal | undefined
): Promise<RerankOutcome> {
  const startedAt = performance.now();
  const n = documents.length;
  const deadline = new AbortController();
  let signal = outer;
  if (deadlineMs !== undefined) {
    signal = outer
      ? AbortSignal.any([outer, deadline.signal])
      : deadline.signal;
  }
  const timers: Array<ReturnType<typeof setTimeout>> = [];
  const work = Promise.resolve()
    .then(() =>
      reranker.rank({
        query,
        documents,
        ...(signal !== undefined ? { signal } : {}),
        ...(deadlineMs !== undefined ? { deadlineMs } : {}),
      })
    )
    .then(
      (outcome: unknown): RerankOutcome => {
        if (isOutcome(outcome, n)) {
          return outcome;
        }
        const reason = outcome == null ? 'error' : 'invalid_results';
        const reported = isDecisionObject(outcome) ? outcome.usage : null;
        return failedStep(
          reranker,
          n,
          reason,
          elapsedMs(startedAt),
          usageOf(reported)
        );
      },
      () => failedStep(reranker, n, 'error', elapsedMs(startedAt))
    );
  if (deadlineMs === undefined) {
    return work;
  }
  timers.push(setTimeout(() => deadline.abort(), deadlineMs));
  const late = new Promise<RerankOutcome>((resolve) => {
    timers.push(
      setTimeout(
        () =>
          resolve(
            failedStep(
              reranker,
              n,
              'timeout',
              elapsedMs(startedAt),
              abandonedUsage(reranker.kind, n)
            )
          ),
        deadlineMs + STEP_GRACE_MS
      )
    );
  });
  try {
    const outcome = await Promise.race([work, late]);
    return deadline.signal.aborted && !isAborted(outer)
      ? asTimeout(outcome)
      : outcome;
  } finally {
    for (const timer of timers) {
      clearTimeout(timer);
    }
  }
}

/**
 * A step's own deadline bounded by what is left of the chain's; a step
 * without one gets the rest of the chain's. `undefined` only when neither
 * the step nor the request has a deadline.
 */
function stepDeadline(
  own: number | undefined,
  expiresAt: number
): number | undefined {
  if (expiresAt === Infinity) {
    return own;
  }
  const remaining = Math.max(1, Math.ceil(expiresAt - performance.now()));
  return own === undefined ? remaining : Math.min(own, remaining);
}

/**
 * A chain of steps tried in order; the first that answers wins. Step 0
 * answering is `ok`; a later step is `degraded` with the first failed step's
 * reason; none answering is `skipped` in identity order with that reason. A
 * `none` step never answers. `kept`, `threshold`, and `pinVerified` come from
 * the answering step only, so falling back to a cross-encoder never carries a
 * System One threshold. A step that throws, rejects, returns something other
 * than a full permutation, or outlives its deadline plus `STEP_GRACE_MS` is a
 * failed step. A request's `deadlineMs` bounds the whole chain: each step
 * gets the lesser of its own deadline and what is left, a step without one
 * gets what is left, and once nothing is left the chain stops with
 * `timeout`. Usage sums every attempt: spend counts what was sent, not what
 * answered. An invalid request is identity order over every document. Never
 * throws.
 */
export function withFallback(
  steps: readonly FallbackStep[],
  options: FallbackOptions = {}
): FallbackReranker {
  if (
    !Array.isArray(steps) ||
    steps.length === 0 ||
    steps.some(
      (step) =>
        step === null ||
        typeof step !== 'object' ||
        typeof step.reranker?.rank !== 'function' ||
        (step.deadlineMs !== undefined &&
          (!Number.isSafeInteger(step.deadlineMs) ||
            validDeadline(step.deadlineMs) === undefined))
    )
  ) {
    throw new DecisionError('bad_request', 'invalid reranker chain', {
      provider: 'fallback',
    });
  }
  const chain = steps.map((step) => ({ ...step }));
  const head = chain[0].reranker;
  const { gate, onAttempt } = options;

  return {
    id: `fallback(${chain.map((step) => step.reranker.id).join(',')})`,
    kind: head.kind,
    model: head.model,
    steps: chain,
    async rank(request: RankRequest): Promise<RerankOutcome> {
      const startedAt = performance.now();
      const documents = documentCount(request);
      if (rankRequestProblem(request) === 'invalid_request') {
        return identityOutcome(documents, {
          reason: 'invalid_request',
          durationMs: elapsedMs(startedAt),
        });
      }
      if (documents === 0) {
        return identityOutcome(0, {
          reason: 'empty',
          durationMs: elapsedMs(startedAt),
        });
      }
      const attempts: RerankAttempt[] = [];
      const usage = emptyRerankUsage();
      const budget = validDeadline(request.deadlineMs);
      const expiresAt = budget === undefined ? Infinity : startedAt + budget;
      let first: string | null = null;
      for (let index = 0; index < chain.length; index++) {
        const { reranker } = chain[index];
        if (isAborted(request.signal)) {
          first ??= 'aborted';
          break;
        }
        if (performance.now() >= expiresAt) {
          first ??= 'timeout';
          break;
        }
        if (gate && reranker.kind !== 'none') {
          let why: string | null | undefined;
          try {
            why = await gate(reranker);
          } catch {
            why = 'error';
          }
          if (typeof why === 'string' && why !== '') {
            attempts.push({
              provider: reranker.kind,
              model: reranker.model,
              status: 'skipped',
              reason: why,
              durationMs: 0,
              usage: emptyRerankUsage(),
            });
            first ??= why;
            continue;
          }
        }
        const outcome = await runStep(
          reranker,
          request.query,
          request.documents,
          stepDeadline(chain[index].deadlineMs, expiresAt),
          request.signal
        );
        try {
          onAttempt?.(reranker, outcome);
        } catch {
          // Bookkeeping must not break a ranking.
        }
        attempts.push(...outcome.attempts);
        addRerankUsage(usage, usageOf(outcome.usage));
        if (reranker.kind !== 'none' && outcome.status !== 'skipped') {
          const ok = index === 0 && outcome.status === 'ok';
          return {
            results: outcome.results,
            provider: outcome.provider,
            model: outcome.model,
            status: ok ? 'ok' : 'degraded',
            reason: ok ? null : (first ?? outcome.reason ?? null),
            attempts,
            durationMs: elapsedMs(startedAt),
            usage,
            ...(outcome.pinVerified !== undefined
              ? { pinVerified: outcome.pinVerified }
              : {}),
            ...(outcome.threshold !== undefined
              ? { threshold: outcome.threshold }
              : {}),
          };
        }
        first ??= outcome.reason ?? 'error';
      }
      return {
        ...identityOutcome(documents, {
          reason: first ?? 'error',
          durationMs: elapsedMs(startedAt),
        }),
        attempts,
        usage,
      };
    },
  };
}

import type {
  RerankKind,
  RerankEntry,
  RerankUsage,
  RankRequest,
  RerankAttempt,
  RerankOutcome,
  OutcomeReranker,
} from './types';
import {
  elapsedMs,
  isAborted,
  stepOutcome,
  isRerankKind,
  documentCount,
  validDeadline,
  addRerankUsage,
  estimatedUsage,
  identityOutcome,
  isRankedOrder,
  emptyRerankUsage,
  everySlot,
  isRerankUsage,
  rankRequestProblem,
  isRerankPermutation,
} from './outcome';
import { DecisionError, isDecisionObject } from '../decisions/types';
import { callHook } from '../decisions/hook';

/**
 * Identity order, never a request: the last step of a chain, or the whole
 * chain when nothing is configured. Its one attempt is `skipped` with
 * `reason` and no usage, so a chain's history shows it ran. A malformed
 * request is `invalid_request` and an empty one `empty`, as for every other
 * reranker, so a bad call is never mistaken for a disabled one.
 */
export function noneReranker(
  options: { reason?: string } = {}
): OutcomeReranker {
  const reason =
    typeof options.reason === 'string' && options.reason !== ''
      ? options.reason
      : 'disabled';
  return {
    id: 'none',
    kind: 'none',
    model: '',
    async rank(request: RankRequest): Promise<RerankOutcome> {
      return stepOutcome({
        kind: 'none',
        documents: documentCount(request),
        startedAt: performance.now(),
        model: '',
        reason: rankRequestProblem(request) ?? reason,
        called: false,
      });
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
   * A gate that throws skips its step with `error`. The wait for it is
   * bounded like a step: a gate still pending when the request's deadline
   * passes or its signal aborts is abandoned and the chain stops with
   * `timeout` or `aborted`.
   */
  gate?: (
    reranker: OutcomeReranker
  ) => string | null | undefined | Promise<string | null | undefined>;
  /**
   * Sees every step that ran, for backoff bookkeeping, as a copy: changing
   * it changes nothing the chain returns. Not awaited; a throw or a rejected
   * promise is ignored.
   */
  onAttempt?: (reranker: OutcomeReranker, outcome: RerankOutcome) => void;
}

export interface FallbackReranker extends OutcomeReranker {
  readonly steps: readonly FallbackStep[];
}

const STATUSES: ReadonlySet<string> = new Set(['ok', 'degraded', 'skipped']);
const ATTEMPT_STATUSES: ReadonlySet<string> = new Set([
  'ok',
  'failed',
  'skipped',
]);

/**
 * Grace after a step's deadline or the caller's abort for a reranker that
 * honors its abort to report its own timeout and spend. A full answer that
 * arrives within it is accepted: waiting for it costs nothing the timeout
 * would not.
 */
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
  if (kind === 'bm25' || kind === 'none' || documents === 0) {
    return emptyRerankUsage();
  }
  const estimate = estimatedUsage(kind, documents);
  return {
    ...emptyRerankUsage(),
    calls: kind === 'systemone' ? documents : 1,
    ...estimate,
  };
}

/** `null` exactly when `status` is `ok`; otherwise the non-empty reason it failed or degraded. */
const reasonFits = (status: string, reason: unknown): boolean =>
  status === 'ok'
    ? reason === null
    : typeof reason === 'string' && reason !== '';

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

function isAttempt(value: unknown): value is RerankAttempt {
  return (
    isDecisionObject(value) &&
    isRerankKind(value.provider) &&
    typeof value.model === 'string' &&
    typeof value.status === 'string' &&
    ATTEMPT_STATUSES.has(value.status) &&
    reasonFits(value.status, value.reason) &&
    isFiniteNumber(value.durationMs) &&
    value.durationMs >= 0 &&
    isRerankUsage(value.usage)
  );
}

/** Every entry's `kept` is what `threshold` says of its score, as `rankedEntries` sets it. */
const keptByThreshold = (
  results: readonly RerankEntry[],
  threshold: number
): boolean =>
  results.every(
    (entry) => entry.kept === (entry.score !== null && entry.score >= threshold)
  );

/** A step that did not answer has no `ok` attempt, and an answer that reports attempts has one. */
const attemptsFit = (
  status: string,
  attempts: readonly RerankAttempt[]
): boolean =>
  status === 'skipped'
    ? attempts.every((attempt) => attempt.status !== 'ok')
    : attempts.length === 0 ||
      attempts.some((attempt) => attempt.status === 'ok');

/**
 * The whole `RerankOutcome` contract, since a step's results and attempts are
 * passed through as they are: `reason` is `null` exactly when the status
 * (the outcome's and each attempt's) is `ok`, an answer (`ok` or
 * `degraded`) scores every document while a `skipped` outcome scores none
 * (so it is identity order), the attempts agree with the status, and `kept`
 * comes only with the `threshold` behind it, on every entry and matching it.
 */
function isOutcome(value: unknown, documents: number): value is RerankOutcome {
  return (
    isDecisionObject(value) &&
    typeof value.status === 'string' &&
    STATUSES.has(value.status) &&
    isRerankKind(value.provider) &&
    typeof value.model === 'string' &&
    reasonFits(value.status, value.reason) &&
    isRerankPermutation(value.results, documents) &&
    isRankedOrder(value.results) &&
    // A skipped step scored nothing, so ranked order is then identity order.
    value.results.every(
      (entry) => (value.status === 'skipped') === (entry.score === null)
    ) &&
    (value.threshold === undefined
      ? value.results.every((entry) => entry.kept === undefined)
      : isFiniteNumber(value.threshold) &&
        keptByThreshold(value.results, value.threshold)) &&
    Array.isArray(value.attempts) &&
    everySlot(value.attempts as unknown[], isAttempt) &&
    attemptsFit(value.status, value.attempts as RerankAttempt[]) &&
    (value.pinVerified === undefined || typeof value.pinVerified === 'boolean')
  );
}

/** A copy of a validated outcome for a hook, so nothing it does reaches the chain's results or history. */
function hookCopy(outcome: RerankOutcome): RerankOutcome {
  return {
    ...outcome,
    results: outcome.results.map((entry) => ({ ...entry })),
    attempts: outcome.attempts.map((attempt) => ({
      ...attempt,
      usage: { ...attempt.usage },
    })),
    usage: usageOf(outcome.usage),
  };
}

/** The attempt a step that ran but reported none stands for, so the chain's history keeps every step. */
function ownAttempt(outcome: RerankOutcome): RerankAttempt {
  const usage = usageOf(outcome.usage);
  let status: RerankAttempt['status'] = 'ok';
  if (outcome.status === 'skipped') {
    status = usage.calls > 0 ? 'failed' : 'skipped';
  }
  return {
    provider: outcome.provider,
    model: outcome.model,
    status,
    // A `degraded` step answered, so its own attempt is `ok` and carries no reason.
    reason: status === 'ok' ? null : outcome.reason,
    durationMs: isFiniteNumber(outcome.durationMs)
      ? Math.max(0, outcome.durationMs)
      : 0,
    usage,
  };
}

type GateVerdict = { why: string | null } | { stop: 'aborted' | 'timeout' };

/**
 * The gate's verdict, or why the chain stopped first: a gate that is slow or
 * never settles is abandoned at the chain's deadline or the caller's abort.
 */
async function consult(
  gate: NonNullable<FallbackOptions['gate']>,
  reranker: OutcomeReranker,
  signal: AbortSignal | undefined,
  expiresAt: number
): Promise<GateVerdict> {
  const cleanup: Array<() => void> = [];
  const contenders: Array<Promise<GateVerdict>> = [
    Promise.resolve()
      .then(() => gate(reranker))
      .then(
        (why): GateVerdict => ({
          why: typeof why === 'string' && why !== '' ? why : null,
        }),
        (): GateVerdict => ({ why: 'error' })
      ),
  ];
  if (expiresAt !== Infinity) {
    contenders.push(
      new Promise<GateVerdict>((resolve) => {
        const timer = setTimeout(
          () => resolve({ stop: 'timeout' }),
          Math.max(0, expiresAt - performance.now())
        );
        cleanup.push(() => clearTimeout(timer));
      })
    );
  }
  if (signal !== undefined) {
    contenders.push(
      new Promise<GateVerdict>((resolve) => {
        const onAbort = (): void => resolve({ stop: 'aborted' });
        signal.addEventListener('abort', onAbort, { once: true });
        cleanup.push(() => signal.removeEventListener('abort', onAbort));
      })
    );
  }
  try {
    return await Promise.race(contenders);
  } finally {
    for (const release of cleanup) {
      release();
    }
  }
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
 * and timers that end the wait `STEP_GRACE_MS` after its deadline or the
 * caller's abort regardless, so a step that ignores its abort cannot hold the
 * chain or its caller.
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
  /** A step that gave up without an answer after its signal fired was stopped, not broken: `asTimeout` then names a deadline. */
  const failure = (): 'aborted' | 'error' =>
    isAborted(signal) ? 'aborted' : 'error';
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
        const reason = outcome == null ? failure() : 'invalid_results';
        const reported = isDecisionObject(outcome) ? outcome.usage : null;
        return failedStep(
          reranker,
          n,
          reason,
          elapsedMs(startedAt),
          usageOf(reported)
        );
      },
      () => failedStep(reranker, n, failure(), elapsedMs(startedAt))
    );
  if (deadlineMs === undefined && outer === undefined) {
    return work;
  }
  /** A step still pending this long is abandoned, charged as if every document was sent. */
  const abandon = (
    reason: 'timeout' | 'aborted',
    afterMs: number
  ): Promise<RerankOutcome> =>
    new Promise<RerankOutcome>((resolve) => {
      timers.push(
        setTimeout(
          () =>
            resolve(
              failedStep(
                reranker,
                n,
                reason,
                elapsedMs(startedAt),
                abandonedUsage(reranker.kind, n)
              )
            ),
          afterMs
        )
      );
    });
  const contenders = [work];
  if (deadlineMs !== undefined) {
    timers.push(setTimeout(() => deadline.abort(), deadlineMs));
    contenders.push(abandon('timeout', deadlineMs + STEP_GRACE_MS));
  }
  let release = (): void => {};
  if (outer !== undefined) {
    contenders.push(
      new Promise<RerankOutcome>((resolve) => {
        const onAbort = (): void => {
          resolve(abandon('aborted', STEP_GRACE_MS));
        };
        if (outer.aborted) {
          onAbort();
          return;
        }
        outer.addEventListener('abort', onAbort, { once: true });
        release = (): void => outer.removeEventListener('abort', onAbort);
      })
    );
  }
  try {
    const outcome = await Promise.race(contenders);
    return deadline.signal.aborted && !isAborted(outer)
      ? asTimeout(outcome)
      : outcome;
  } finally {
    release();
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
 * System One threshold. A step that throws, rejects, returns something that
 * breaks the `RerankOutcome` contract (a full permutation in score order,
 * every document scored when it answers, `kept` only with a `threshold` and
 * then on every entry, `true` exactly where the score reaches it, well-formed
 * attempts whose usage counters and durations are finite counts of at least
 * zero, and well-formed fields), or outlives its deadline or the
 * caller's abort by `STEP_GRACE_MS` is a failed step; a step that ran but
 * reported no attempt is recorded as one. A request's `deadlineMs` bounds the whole
 * chain, gates included: each step gets the lesser of its own deadline and
 * what is left, a step without one gets what is left, and once nothing is
 * left the chain stops with `timeout`. A step's deadline reaches it as
 * `RankRequest.deadlineMs`, which caps the reranker's own timeout and never
 * extends it. Usage sums every attempt, from the
 * attempts themselves rather than a step's own total: spend counts what was
 * sent, not what answered. An invalid request is identity order over every
 * document. Never throws.
 */
export function withFallback(
  steps: readonly FallbackStep[],
  options: FallbackOptions = {}
): FallbackReranker {
  if (
    !Array.isArray(steps) ||
    steps.length === 0 ||
    !everySlot(
      steps,
      (step) =>
        step !== null &&
        typeof step === 'object' &&
        typeof step.reranker?.rank === 'function' &&
        isRerankKind(step.reranker.kind) &&
        typeof step.reranker.model === 'string' &&
        (step.deadlineMs === undefined ||
          (Number.isSafeInteger(step.deadlineMs) &&
            validDeadline(step.deadlineMs) !== undefined))
    )
  ) {
    throw new DecisionError('bad_request', 'invalid reranker chain', {
      provider: 'fallback',
    });
  }
  const chain = steps.map((step) => ({ ...step }));
  const head =
    chain.find(({ reranker }) => reranker.kind !== 'none')?.reranker ??
    chain[0].reranker;
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
      /** Read before every step and after every gate: either may have run out the clock. */
      const stopped = (): 'aborted' | 'timeout' | null => {
        if (isAborted(request.signal)) {
          return 'aborted';
        }
        return performance.now() >= expiresAt ? 'timeout' : null;
      };
      let first: string | null = null;
      try {
        for (let index = 0; index < chain.length; index++) {
          const { reranker } = chain[index];
          let stop = stopped();
          if (stop === null && gate && reranker.kind !== 'none') {
            const verdict = await consult(
              gate,
              reranker,
              request.signal,
              expiresAt
            );
            if ('stop' in verdict) {
              stop = verdict.stop;
            } else if (verdict.why !== null) {
              attempts.push({
                provider: reranker.kind,
                model: reranker.model,
                status: 'skipped',
                reason: verdict.why,
                durationMs: 0,
                usage: emptyRerankUsage(),
              });
              first ??= verdict.why;
              continue;
            } else {
              stop = stopped();
            }
          }
          if (stop !== null) {
            first ??= stop;
            break;
          }
          const outcome = await runStep(
            reranker,
            request.query,
            request.documents,
            stepDeadline(chain[index].deadlineMs, expiresAt),
            request.signal
          );
          // Bookkeeping, sync or async, must not break a ranking.
          if (onAttempt !== undefined) {
            const seen = hookCopy(outcome);
            callHook(() => onAttempt(reranker, seen));
          }
          const ran =
            outcome.attempts.length > 0
              ? outcome.attempts
              : [ownAttempt(outcome)];
          attempts.push(...ran);
          // From the validated attempts, never the step's own total, so `usage` always sums `attempts`.
          for (const attempt of ran) {
            addRerankUsage(usage, attempt.usage);
          }
          if (outcome.provider !== 'none' && outcome.status !== 'skipped') {
            const ok = index === 0 && outcome.status === 'ok';
            return {
              results: outcome.results,
              provider: outcome.provider,
              model: outcome.model,
              status: ok ? 'ok' : 'degraded',
              reason: ok ? null : (first ?? outcome.reason ?? 'error'),
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
      } catch {
        // Every helper above is meant never to throw; this keeps `rank` from rejecting if one does.
        first ??= 'error';
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

import type {
  DecisionModel,
  DecisionResult,
  DecisionFailure,
  DecisionRequest,
} from './types';
import { DecisionError } from './types';

/** Failures that belong to the credential, not the request: the next request would be refused the same way. */
export const FATAL_DECISION_FAILURES: ReadonlySet<DecisionFailure> =
  new Set<DecisionFailure>(['unauthorized', 'quota']);

export function isFatalDecisionError(error: unknown): error is DecisionError {
  return (
    error instanceof DecisionError && FATAL_DECISION_FAILURES.has(error.failure)
  );
}

/** A fuse: `open` once `trip` has recorded an error, until `reset`. */
export interface DecisionBreaker {
  readonly open: boolean;
  /** The first error that tripped the breaker, or `null` while it is closed. */
  readonly error: DecisionError | null;
  /** Counts resets; a call captures it before its request so a stale failure cannot reopen a reset breaker. */
  readonly generation: number;
  /**
   * Records `error` unless the breaker is already open; the first error wins.
   * With `generation`, records it only if no `reset` happened since that
   * generation was read.
   */
  trip(error: DecisionError, generation?: number): void;
  reset(): void;
}

export interface BreakerDecisionModel extends DecisionModel {
  readonly breaker: DecisionBreaker;
}

export function createBreaker(): DecisionBreaker {
  let tripped: DecisionError | null = null;
  let generation = 0;
  return {
    get open(): boolean {
      return tripped !== null;
    },
    get error(): DecisionError | null {
      return tripped;
    },
    get generation(): number {
      return generation;
    },
    trip(error: DecisionError, seen?: number): void {
      if (seen !== undefined && seen !== generation) {
        return;
      }
      tripped ??= error;
    },
    reset(): void {
      tripped = null;
      generation += 1;
    },
  };
}

/**
 * The model, fused: after one `unauthorized` or `quota` failure every later
 * call rejects at once, without a request, with the same failure, status, and
 * `circuitOpen: true`, until `breaker.reset()`. Rate limits, server errors,
 * timeouts, and network failures never trip it; the transport retries those.
 * A fatal failure from a call that started before a `reset()` does not reopen
 * it. Share one breaker between models that share a credential.
 */
export function withBreaker(
  model: DecisionModel,
  breaker: DecisionBreaker = createBreaker()
): BreakerDecisionModel {
  return {
    id: model.id,
    model: model.model,
    breaker,
    async decide(request: DecisionRequest): Promise<DecisionResult> {
      const tripped = breaker.error;
      if (tripped !== null) {
        throw new DecisionError(
          tripped.failure,
          `decision model circuit open (${tripped.failure})`,
          {
            provider: tripped.provider,
            status: tripped.status,
            circuitOpen: true,
          }
        );
      }
      const generation = breaker.generation;
      try {
        return await model.decide(request);
      } catch (error) {
        if (isFatalDecisionError(error)) {
          breaker.trip(error, generation);
        }
        throw error;
      }
    },
  };
}

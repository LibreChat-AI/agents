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
  /** Records `error` unless the breaker is already open; the first error wins. */
  trip(error: DecisionError): void;
  reset(): void;
}

export interface BreakerDecisionModel extends DecisionModel {
  readonly breaker: DecisionBreaker;
}

export function createBreaker(): DecisionBreaker {
  let tripped: DecisionError | null = null;
  return {
    get open(): boolean {
      return tripped !== null;
    },
    get error(): DecisionError | null {
      return tripped;
    },
    trip(error: DecisionError): void {
      tripped ??= error;
    },
    reset(): void {
      tripped = null;
    },
  };
}

/**
 * The model, fused: after one `unauthorized` or `quota` failure every later
 * call rejects at once, without a request, with the same failure, status, and
 * `circuitOpen: true`, until `breaker.reset()`. Rate limits, server errors,
 * timeouts, and network failures never trip it; the transport retries those.
 * Share one breaker between models that share a credential.
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
      try {
        return await model.decide(request);
      } catch (error) {
        if (isFatalDecisionError(error)) {
          breaker.trip(error);
        }
        throw error;
      }
    },
  };
}

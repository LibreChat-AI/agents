import type { DecisionModel, DecisionFailure, DecisionRequest } from './types';
import { withDecisionDeadline } from './deadline';
import { booleanQuestion } from './questions';
import { redactExcerpt } from './redact';
import { DecisionError } from './types';

const DETAIL_LIMIT = 200;
const PROBE_FAILED = 'decision model probe failed';
/** How long a probe without `timeoutMs` waits for a model that never settles. */
export const PROBE_MAX_WAIT_MS = 60_000;

export type DecisionProbe =
  | { ok: true; model: string }
  | {
      ok: false;
      failure: DecisionFailure;
      status?: number;
      circuitOpen: boolean;
      /** Redacted, whitespace-collapsed, and truncated; safe for a log line or a status row. */
      detail: string;
    };

export interface DecisionProbeOptions {
  /**
   * Passed to the model and enforced around its call, so a model that ignores
   * it still settles; allow for a slow first token mint. Without it the model
   * keeps its own timeout and the wait is bounded at `PROBE_MAX_WAIT_MS`.
   */
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Credentials the failure detail must never contain. */
  secrets?: ReadonlyArray<string | null | undefined>;
}

export type DecisionProbeRequest = Pick<
  DecisionRequest,
  'state' | 'questions' | 'label'
>;

/** One yes/no question a host answers in a few tokens; the answer itself is irrelevant. */
export function probeRequest(): DecisionProbeRequest {
  return {
    state: 'decision model readiness probe',
    questions: { ok: booleanQuestion('Is this text a readiness probe?') },
    label: 'probe',
  };
}

/**
 * Can this model answer right now? Asks one tiny boolean question and never
 * throws, and settles by its deadline or the caller's abort even when the
 * model ignores both: any reply is `ok`, and any failure, including a non-SDK
 * exception (read as `network`), is returned with a redacted detail. Through
 * `withBreaker`, a fatal failure trips the breaker and an open breaker answers
 * without a request.
 */
export async function probe(
  model: DecisionModel,
  options: DecisionProbeOptions = {}
): Promise<DecisionProbe> {
  try {
    const { timeoutMs } = options;
    const result = await withDecisionDeadline(
      model.id,
      timeoutMs ?? PROBE_MAX_WAIT_MS,
      options.signal,
      (signal, waitFor) =>
        waitFor(
          Promise.resolve().then(() =>
            model.decide({
              ...probeRequest(),
              ...(timeoutMs !== undefined ? { timeoutMs } : {}),
              signal,
            })
          )
        )
    );
    return { ok: true, model: result.model };
  } catch (error) {
    if (!(error instanceof DecisionError)) {
      return {
        ok: false,
        failure: 'network',
        circuitOpen: false,
        detail: PROBE_FAILED,
      };
    }
    return {
      ok: false,
      failure: error.failure,
      ...(error.status !== undefined ? { status: error.status } : {}),
      circuitOpen: error.circuitOpen,
      detail: safeDetail(error.message, options),
    };
  }
}

/**
 * The redacted start of the message, or a fixed detail when `secrets` cannot
 * be read: never the raw message, never a throw. Only the start is read, so a
 * host model's oversized error message cannot make the probe slow or large.
 */
function safeDetail(message: string, options: DecisionProbeOptions): string {
  try {
    const { secrets } = options;
    return redactExcerpt(
      message,
      secrets === undefined ? [] : secrets,
      DETAIL_LIMIT
    );
  } catch {
    return PROBE_FAILED;
  }
}

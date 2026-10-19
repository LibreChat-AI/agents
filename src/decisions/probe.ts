import type { DecisionModel, DecisionFailure, DecisionRequest } from './types';
import { booleanQuestion } from './questions';
import { DecisionError } from './types';
import { redact } from './redact';

const DETAIL_LIMIT = 200;

export type DecisionProbe =
  | { ok: true; model: string }
  | {
      ok: false;
      failure: DecisionFailure;
      status?: number;
      circuitOpen: boolean;
      /** Redacted and truncated; safe for a log line or a status row. */
      detail: string;
    };

export interface DecisionProbeOptions {
  /** Defaults to the model's own timeout; allow for a slow first token mint. */
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
 * throws: any reply is `ok`, and any failure, including a non-SDK exception
 * (read as `network`), is returned with a redacted detail. Through
 * `withBreaker`, a fatal failure trips the breaker and an open breaker answers
 * without a request.
 */
export async function probe(
  model: DecisionModel,
  options: DecisionProbeOptions = {}
): Promise<DecisionProbe> {
  try {
    const result = await model.decide({
      ...probeRequest(),
      ...(options.timeoutMs !== undefined
        ? { timeoutMs: options.timeoutMs }
        : {}),
      ...(options.signal !== undefined ? { signal: options.signal } : {}),
    });
    return { ok: true, model: result.model };
  } catch (error) {
    if (!(error instanceof DecisionError)) {
      return {
        ok: false,
        failure: 'network',
        circuitOpen: false,
        detail: 'decision model probe failed',
      };
    }
    return {
      ok: false,
      failure: error.failure,
      ...(error.status !== undefined ? { status: error.status } : {}),
      circuitOpen: error.circuitOpen,
      detail: redact(error.message, options.secrets).slice(0, DETAIL_LIMIT),
    };
  }
}

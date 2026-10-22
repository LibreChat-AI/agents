import type { DecisionModel } from '../decisions/types';
import type { RerankReason } from './types';
import { rerankReasonOf, validDeadline, isRequestSignal } from './outcome';
import { withDecisionDeadline } from '../decisions/deadline';
import { DecisionError } from '../decisions/types';
import { probeRequest } from '../decisions/probe';

/** A model id no host serves: a host that answers it echoes whatever it is asked, so its echo proves nothing. */
export const PIN_SENTINEL_MODEL = 'jev-0.0.0-pin-sentinel';

export interface PinProbeOptions {
  /** The same host and credential, configured for `model`. */
  modelFor: (model: string) => DecisionModel;
  /** The checkpoint to verify, such as `jev-1.13.0`. */
  pin: string;
  /** Defaults to `PIN_SENTINEL_MODEL`; a blank sentinel, or one that is the pin, is `invalid_request`. */
  sentinel?: string;
  /**
   * Per request, enforced around the model call even when the model ignores
   * it; defaults to 15 seconds to allow a slow first token mint. Anything but
   * an integer from 1 to 3,600,000 is `invalid_request`, with no request sent.
   */
  timeoutMs?: number;
  signal?: AbortSignal;
}

export type PinProbe =
  | { verified: true; echoed: string; sentinelStatus: 404 }
  | {
      verified: false;
      /**
       * `sentinel_accepted`: the host answered a model it cannot serve.
       * `echo_mismatch`: the pin was answered as another model. Otherwise the
       * request's own failure, which says nothing about the pin.
       */
      reason: 'sentinel_accepted' | 'echo_mismatch' | RerankReason;
      echoed?: string;
      /** The HTTP status the sentinel request got, when it got a reply. */
      sentinelStatus?: number;
    };

const echoOf = (value: unknown): string | undefined =>
  value !== null &&
  typeof value === 'object' &&
  typeof (value as { model?: unknown }).model === 'string'
    ? (value as { model: string }).model
    : undefined;

/**
 * Does this host really serve the pinned model? Two answers are needed: a
 * sentinel model no host serves must be refused with HTTP 404 (a host that
 * answers it, as Laya does, echoes whatever it is asked, so its echo proves
 * nothing), and a probe with the pin must echo exactly the pin. Each request
 * is bounded by `timeoutMs` and the caller's signal even when the model ignores
 * them. Never throws.
 */
export async function probePin(options: PinProbeOptions): Promise<PinProbe> {
  const pin = typeof options.pin === 'string' ? options.pin.trim() : '';
  let sentinel = PIN_SENTINEL_MODEL;
  if (options.sentinel !== undefined) {
    sentinel =
      typeof options.sentinel === 'string' ? options.sentinel.trim() : '';
  }
  if (
    pin === '' ||
    sentinel === '' ||
    sentinel === pin ||
    typeof options.modelFor !== 'function' ||
    !isRequestSignal(options.signal)
  ) {
    return { verified: false, reason: 'invalid_request' };
  }
  const timeoutMs = options.timeoutMs ?? 15_000;
  if (
    !Number.isSafeInteger(timeoutMs) ||
    validDeadline(timeoutMs) === undefined
  ) {
    return { verified: false, reason: 'invalid_request' };
  }
  /** Bounded here, so a model that ignores its deadline and signal cannot hold the probe. */
  const ask = (model: string): Promise<unknown> =>
    withDecisionDeadline(
      'pin-probe',
      timeoutMs,
      options.signal,
      (signal, waitFor) =>
        waitFor(
          Promise.resolve().then(() =>
            options.modelFor(model).decide({
              ...probeRequest(),
              timeoutMs,
              signal,
            })
          )
        )
    );
  try {
    const answer = await ask(sentinel);
    const echoed = echoOf(answer);
    return {
      verified: false,
      reason: 'sentinel_accepted',
      ...(echoed !== undefined ? { echoed } : {}),
      sentinelStatus: 200,
    };
  } catch (error) {
    if (!(error instanceof DecisionError) || error.status !== 404) {
      // An open breaker refuses with the status that tripped it, not one the sentinel got.
      const replied =
        error instanceof DecisionError &&
        error.status !== undefined &&
        !error.circuitOpen;
      return {
        verified: false,
        reason: rerankReasonOf(error),
        ...(replied ? { sentinelStatus: error.status } : {}),
      };
    }
  }
  try {
    const echoed = echoOf(await ask(pin));
    if (echoed !== pin) {
      return {
        verified: false,
        reason: 'echo_mismatch',
        ...(echoed !== undefined ? { echoed } : {}),
        sentinelStatus: 404,
      };
    }
    return { verified: true, echoed, sentinelStatus: 404 };
  } catch (error) {
    return {
      verified: false,
      reason: rerankReasonOf(error),
      sentinelStatus: 404,
    };
  }
}

import type { DecisionModel } from '../decisions/types';
import type { RerankReason } from './types';
import { DecisionError } from '../decisions/types';
import { probeRequest } from '../decisions/probe';
import { rerankReasonOf } from './outcome';

/** A model id no host serves: a host that answers it echoes whatever it is asked, so its echo proves nothing. */
export const PIN_SENTINEL_MODEL = 'jev-0.0.0-pin-sentinel';

export interface PinProbeOptions {
  /** The same host and credential, configured for `model`. */
  modelFor: (model: string) => DecisionModel;
  /** The checkpoint to verify, such as `jev-1.13.0`. */
  pin: string;
  sentinel?: string;
  /** Per request; defaults to 15 seconds to allow a slow first token mint. */
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
 * nothing), and a probe with the pin must echo exactly the pin. Never throws.
 */
export async function probePin(options: PinProbeOptions): Promise<PinProbe> {
  const pin = typeof options.pin === 'string' ? options.pin.trim() : '';
  const sentinel = options.sentinel ?? PIN_SENTINEL_MODEL;
  if (
    pin === '' ||
    sentinel === pin ||
    typeof options.modelFor !== 'function'
  ) {
    return { verified: false, reason: 'invalid_request' };
  }
  const ask = async (model: string): Promise<unknown> =>
    options.modelFor(model).decide({
      ...probeRequest(),
      timeoutMs: options.timeoutMs ?? 15_000,
      ...(options.signal !== undefined ? { signal: options.signal } : {}),
    });
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
      return {
        verified: false,
        reason: rerankReasonOf(error),
        ...(error instanceof DecisionError && error.status !== undefined
          ? { sentinelStatus: error.status }
          : {}),
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

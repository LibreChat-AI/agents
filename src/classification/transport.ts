import type { ClassificationCredential } from './types';
import { ClassificationError } from './types';

const DEFAULT_TIMEOUT_MS = 4_000;
const DEFAULT_MAX_RETRIES = 2;
const BACKOFF_MS = [250, 750, 1_500, 3_000, 6_000];
const MAX_RETRY_AFTER_MS = 10_000;

export type ClassificationFetch = (
  input: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body: string;
    signal: AbortSignal;
  }
) => Promise<{
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
}>;

export interface TransportOptions {
  providerId: string;
  apiKey: ClassificationCredential;
  /** Full URL, not a base path. */
  endpoint: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: ClassificationFetch;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Called once per answered request with its label and wall time, for the caller's own logging. */
  onAnswered?: (label: string, ms: number) => void;
}

export type Transport = (
  payload: string,
  signal: AbortSignal | undefined,
  label: string,
  timeoutOverrideMs?: number
) => Promise<string>;

function failureForStatus(
  status: number
): 'unauthorized' | 'rate_limited' | 'server_error' | 'bad_request' {
  if (status === 401 || status === 403) {
    return 'unauthorized' as const;
  }
  if (status === 429) {
    return 'rate_limited' as const;
  }
  if (status >= 500) {
    return 'server_error' as const;
  }
  return 'bad_request' as const;
}

function isRetryable(failure: string): boolean {
  return (
    failure === 'rate_limited' ||
    failure === 'server_error' ||
    failure === 'network'
  );
}

function retryAfterMs(header: string | null): number | undefined {
  if (header == null || header === '') {
    return undefined;
  }
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1_000, MAX_RETRY_AFTER_MS);
  }
  const at = Date.parse(header);
  if (Number.isNaN(at)) {
    return undefined;
  }
  return Math.min(Math.max(at - Date.now(), 0), MAX_RETRY_AFTER_MS);
}

function briefly(body: string): string {
  const flat = body.replace(/\s+/g, ' ').trim();
  return flat.length > 200 ? `${flat.slice(0, 200)}…` : flat;
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted === true) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const done = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', done);
      resolve();
    };
    const timer = setTimeout(done, ms);
    signal?.addEventListener('abort', done, { once: true });
  });
}

/**
 * POST JSON to one endpoint with a bearer key: typed failures, one deadline for the whole call
 * (retries and backoff included), a bounded retry on rate limits, server errors and network
 * failures, honouring `retry-after` up to ten seconds.
 */
export function createTransport(options: TransportOptions): Transport {
  const { providerId } = options;
  const credential = options.apiKey;
  if (typeof credential === 'string' && credential.trim() === '') {
    throw new ClassificationError(
      'unauthorized',
      'classifier requires an API key',
      {
        provider: providerId,
      }
    );
  }

  const endpoint = options.endpoint.trim();
  if (endpoint === '') {
    throw new ClassificationError(
      'bad_request',
      'classifier requires an endpoint',
      {
        provider: providerId,
      }
    );
  }

  const defaultTimeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const sleep = options.sleep ?? defaultSleep;
  const onAnswered = options.onAnswered;
  const candidate =
    options.fetch ??
    (globalThis.fetch as unknown as ClassificationFetch | undefined);
  if (typeof candidate !== 'function') {
    throw new ClassificationError(
      'network',
      'no fetch implementation available',
      {
        provider: providerId,
      }
    );
  }
  const fetchImpl: ClassificationFetch = candidate;

  async function resolveKey(refresh: boolean): Promise<string> {
    if (typeof credential === 'string') {
      return credential.trim();
    }
    const minted = (await credential({ refresh })).trim();
    if (minted === '') {
      throw new ClassificationError(
        'unauthorized',
        'credential function returned no token',
        { provider: providerId }
      );
    }
    return minted;
  }

  async function attempt(
    payload: string,
    signal: AbortSignal | undefined,
    timeoutMs: number,
    refreshKey: boolean
  ): Promise<string> {
    const apiKey = await resolveKey(refreshKey);
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), timeoutMs);
    const combined =
      signal != null
        ? AbortSignal.any([signal, timeout.signal])
        : timeout.signal;

    try {
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: payload,
        signal: combined,
      });

      const text = await response.text();
      if (!response.ok) {
        const error = new ClassificationError(
          failureForStatus(response.status),
          `classifier returned ${response.status}: ${briefly(text)}`,
          { provider: providerId, status: response.status }
        );
        const wait = retryAfterMs(response.headers.get('retry-after'));
        if (wait != null) {
          error.retryAfterMs = wait;
        }
        throw error;
      }
      return text;
    } catch (error) {
      if (error instanceof ClassificationError) {
        throw error;
      }
      if (signal?.aborted === true) {
        throw new ClassificationError('aborted', 'caller aborted the request', {
          provider: providerId,
        });
      }
      if (timeout.signal.aborted) {
        throw new ClassificationError(
          'timeout',
          `no answer within ${timeoutMs}ms`,
          {
            provider: providerId,
          }
        );
      }
      const message = error instanceof Error ? error.message : String(error);
      throw new ClassificationError('network', message, {
        provider: providerId,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  /** `timeoutMs` bounds the whole call, retries and backoff included, not each attempt. */
  return async function send(payload, signal, label, timeoutOverrideMs) {
    const timeoutMs =
      timeoutOverrideMs != null && timeoutOverrideMs > 0
        ? timeoutOverrideMs
        : defaultTimeoutMs;
    const deadline = Date.now() + timeoutMs;
    let lastError: ClassificationError | undefined;
    let refreshKey = false;
    let refreshed = false;
    for (let attemptNo = 0; attemptNo <= maxRetries; attemptNo++) {
      try {
        const started = Date.now();
        const body = await attempt(
          payload,
          signal,
          Math.max(1, deadline - started),
          refreshKey
        );
        onAnswered?.(label, Date.now() - started);
        return body;
      } catch (error) {
        lastError =
          error instanceof ClassificationError
            ? error
            : new ClassificationError('network', String(error), {
              provider: providerId,
            });
        if (
          lastError.status === 401 &&
          typeof credential === 'function' &&
          !refreshed
        ) {
          refreshed = true;
          refreshKey = true;
          attemptNo -= 1;
          continue;
        }
        refreshKey = false;
        if (attemptNo === maxRetries || !isRetryable(lastError.failure)) {
          break;
        }
        const wait =
          lastError.retryAfterMs ??
          BACKOFF_MS[Math.min(attemptNo, BACKOFF_MS.length - 1)];
        if (wait >= deadline - Date.now()) {
          break;
        }
        await sleep(wait, signal);
        if (signal?.aborted === true) {
          lastError = new ClassificationError(
            'aborted',
            'caller aborted the request',
            {
              provider: providerId,
            }
          );
          break;
        }
      }
    }
    throw (
      lastError ??
      new ClassificationError('network', 'request failed', {
        provider: providerId,
      })
    );
  };
}

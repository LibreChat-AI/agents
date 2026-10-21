import type { AwaitWithinDeadline } from './deadline';
import type { DecisionCredential } from './types';
import { withDecisionDeadline } from './deadline';
import { DecisionError } from './types';
import { callHook } from './hook';

const DEFAULT_TIMEOUT_MS = 4_000;
const DEFAULT_MAX_RETRIES = 2;
const MAX_RESPONSE_BYTES = 256 * 1024;
const BACKOFF_MS = [250, 750, 1_500, 3_000, 6_000];
const MAX_RETRY_AFTER_MS = 10_000;

export type DecisionFetch = (
  input: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body: string;
    signal: AbortSignal;
    redirect: 'error';
  }
) => Promise<Pick<Response, 'ok' | 'status' | 'headers' | 'body'>>;

export interface TransportOptions {
  providerId: string;
  apiKey?: DecisionCredential;
  requiresAuth?: boolean;
  /** Full URL, not a base path. */
  endpoint: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: DecisionFetch;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /**
   * Called only after the response parses, with no state, answers, or
   * credentials. Not awaited; a throw or a rejected promise is ignored.
   */
  onAnswered?: (label: string, ms: number) => void;
}

/**
 * `onRequest` is called once per HTTP request, just before it is sent:
 * retries and every re-send after a 401 included. `onResponse` is called once
 * per such request that ended before the deadline or the caller's abort, with
 * its HTTP status (after a successful body is read in full) or `null` when the
 * fetch failed, and a reply's `Retry-After` in milliseconds when it was not OK
 * and sent one; a request interrupted in flight gets none. Neither is awaited;
 * a throw or a rejected promise is ignored.
 */
export type Transport = <T>(
  prepare: () => { payload: string; parse: (body: string) => T },
  signal: AbortSignal | undefined,
  label: string,
  timeoutOverrideMs?: number,
  onRequest?: () => void,
  onResponse?: (status: number | null, retryAfterMs?: number) => void
) => Promise<T>;

export function failureForStatus(
  status: number
): 'unauthorized' | 'quota' | 'rate_limited' | 'server_error' | 'bad_request' {
  if (status === 401 || status === 403) {
    return 'unauthorized';
  }
  if (status === 402) {
    return 'quota';
  }
  if (status === 429) {
    return 'rate_limited';
  }
  if (status >= 500) {
    return 'server_error';
  }
  return 'bad_request';
}

export function retryAfterMs(header: string | null): number | undefined {
  if (header == null || header === '') {
    return undefined;
  }
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1_000, MAX_RETRY_AFTER_MS);
  }
  const at = Date.parse(header);
  return Number.isFinite(at)
    ? Math.min(Math.max(at - Date.now(), 0), MAX_RETRY_AFTER_MS)
    : undefined;
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

/** Stream successful bodies only, so neither errors nor oversized replies are ever buffered. */
async function readResponse(
  body: ReadableStream<Uint8Array> | null,
  waitFor: AwaitWithinDeadline,
  provider: string
): Promise<string> {
  if (!body) {
    return '';
  }
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = '';
  try {
    for (;;) {
      const chunk = await waitFor(reader.read());
      if (chunk.done) {
        return text + decoder.decode();
      }
      bytes += chunk.value.byteLength;
      if (bytes > MAX_RESPONSE_BYTES) {
        throw new DecisionError(
          'malformed_response',
          'decision model response too large',
          {
            provider,
          }
        );
      }
      text += decoder.decode(chunk.value, { stream: true });
    }
  } catch (error) {
    void reader.cancel().catch(() => {});
    throw error;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // A timed-out read still holds the lock until cancellation settles.
    }
  }
}

/** One immutable transport; every call owns its signal, deadline, and credentials. */
export function createTransport(options: TransportOptions): Transport {
  const { providerId } = options;
  const credential = options.apiKey;
  if (
    options.requiresAuth !== false &&
    (credential == null ||
      (typeof credential === 'string' && credential.trim() === ''))
  ) {
    throw new DecisionError(
      'unauthorized',
      'decision model requires an API key',
      {
        provider: providerId,
      }
    );
  }
  const endpoint = options.endpoint.trim();
  if (!endpoint) {
    throw new DecisionError(
      'bad_request',
      'decision model requires an endpoint',
      {
        provider: providerId,
      }
    );
  }
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  const sleep = options.sleep ?? defaultSleep;
  const onAnswered = options.onAnswered;
  if (!Number.isSafeInteger(maxRetries) || maxRetries < 0 || maxRetries > 5) {
    throw new DecisionError(
      'bad_request',
      'invalid decision model retry limit',
      {
        provider: providerId,
      }
    );
  }
  const fetchImpl: DecisionFetch = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new DecisionError('network', 'no fetch implementation available', {
      provider: providerId,
    });
  }

  async function resolveKey(
    refused: string | undefined,
    signal: AbortSignal
  ): Promise<string | null> {
    if (typeof credential === 'string') {
      return credential.trim() || null;
    }
    if (!credential) {
      return null;
    }
    const minted: string = await credential(
      refused === undefined
        ? { refresh: false, signal }
        : { refresh: true, refused, signal }
    );
    const token = typeof minted === 'string' ? minted.trim() : '';
    if (!token) {
      throw new DecisionError(
        'unauthorized',
        'credential function returned no token',
        {
          provider: providerId,
        }
      );
    }
    return token;
  }

  return async function send(
    prepare,
    callerSignal,
    label,
    timeoutOverrideMs,
    onRequest,
    onResponse
  ) {
    const started = performance.now();
    return withDecisionDeadline(
      providerId,
      timeoutOverrideMs ?? timeoutMs,
      callerSignal,
      async (signal, waitFor) => {
        let prepared: ReturnType<typeof prepare>;
        try {
          prepared = prepare();
        } catch (error) {
          if (error instanceof DecisionError) {
            throw error;
          }
          throw new DecisionError(
            'bad_request',
            'invalid decision model request',
            {
              provider: providerId,
            }
          );
        }
        const { payload, parse } = prepared;
        await waitFor(Promise.resolve());
        const wasAborted = (): boolean => signal.aborted;
        const interrupted = (): DecisionError =>
          new DecisionError(
            callerSignal?.aborted === true ? 'aborted' : 'timeout',
            'decision model request interrupted',
            { provider: providerId }
          );
        /**
         * Keys 401s refused in this call. The first earns a refresh; a second
         * earns one recheck with `refresh: false`, since another call may have
         * replaced the key while this request was in flight, and the call
         * re-sends only a key it has not had refused.
         */
        const refused: string[] = [];
        let refusal: DecisionError | undefined;
        let key: string | null | undefined;
        let lastError: DecisionError | undefined;
        for (let attemptNo = 0; attemptNo <= maxRetries; attemptNo++) {
          if (wasAborted()) {
            throw interrupted();
          }
          try {
            if (key === undefined) {
              key = await waitFor(
                resolveKey(
                  refused.length === 1 ? refused[0] : undefined,
                  signal
                )
              );
              if (
                refusal !== undefined &&
                typeof key === 'string' &&
                refused.includes(key)
              ) {
                /** The minter has nothing newer than a key this call had refused: that refusal stands. */
                throw refusal;
              }
            }
            if (onRequest !== undefined) {
              callHook(() => onRequest());
            }
            /** Reports the request's end once, unless the deadline or the caller's abort interrupted it. */
            let open = true;
            const ended = (
              status: number | null,
              retryAfter?: number
            ): void => {
              if (open && !wasAborted() && onResponse !== undefined) {
                callHook(() => onResponse(status, retryAfter));
              }
              open = false;
            };
            let response: Awaited<ReturnType<DecisionFetch>>;
            try {
              response = await waitFor(
                fetchImpl(endpoint, {
                  method: 'POST',
                  headers: {
                    ...(key !== null ? { Authorization: `Bearer ${key}` } : {}),
                    'Content-Type': 'application/json',
                  },
                  body: payload,
                  signal,
                  redirect: 'error',
                })
              );
            } catch (error) {
              ended(null);
              throw error;
            }
            if (!response.ok) {
              const retryAfter = retryAfterMs(
                response.headers.get('retry-after')
              );
              ended(response.status, retryAfter);
              void response.body?.cancel().catch(() => {});
              const error = new DecisionError(
                failureForStatus(response.status),
                `decision model returned HTTP ${response.status}`,
                { provider: providerId, status: response.status }
              );
              error.retryAfterMs = retryAfter;
              throw error;
            }
            let body: string;
            try {
              body = await readResponse(response.body, waitFor, providerId);
            } finally {
              ended(response.status);
            }
            const result = await waitFor(Promise.resolve(parse(body)));
            // Observability, sync or async, must not turn a valid answer into a retried request.
            callHook(() => onAnswered?.(label, performance.now() - started));
            return result;
          } catch (error) {
            if (wasAborted()) {
              throw interrupted();
            }
            if (error instanceof DecisionError) {
              lastError = error;
            } else {
              lastError = new DecisionError(
                'network',
                'decision model request failed',
                { provider: providerId }
              );
            }
            if (
              lastError.status === 401 &&
              typeof credential === 'function' &&
              typeof key === 'string' &&
              refused.length < 2 &&
              !refused.includes(key)
            ) {
              refused.push(key);
              refusal = lastError;
              key = undefined;
              attemptNo -= 1;
              continue;
            }
            if (
              attemptNo === maxRetries ||
              !['rate_limited', 'server_error', 'network'].includes(
                lastError.failure
              )
            ) {
              break;
            }
            const wait =
              lastError.retryAfterMs ??
              BACKOFF_MS[Math.min(attemptNo, BACKOFF_MS.length - 1)];
            try {
              await waitFor(sleep(wait, signal));
            } catch (error) {
              if (error instanceof DecisionError) {
                throw error;
              }
              throw new DecisionError(
                'network',
                'decision model backoff failed',
                {
                  provider: providerId,
                }
              );
            }
          }
        }
        throw (
          lastError ??
          new DecisionError('network', 'decision model request failed', {
            provider: providerId,
          })
        );
      }
    );
  };
}

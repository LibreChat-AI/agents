import type { ClassificationCredential } from './types';
import type { AwaitWithinDeadline } from './deadline';
import { withClassificationDeadline } from './deadline';
import { ClassificationError } from './types';

const DEFAULT_TIMEOUT_MS = 4_000;
const DEFAULT_MAX_RETRIES = 2;
const MAX_RESPONSE_BYTES = 256 * 1024;
const BACKOFF_MS = [250, 750, 1_500, 3_000, 6_000];
const MAX_RETRY_AFTER_MS = 10_000;

export type ClassificationFetch = (
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
  apiKey?: ClassificationCredential;
  requiresAuth?: boolean;
  /** Full URL, not a base path. */
  endpoint: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetch?: ClassificationFetch;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Called only after the response parses, with no state, answers, or credentials. */
  onAnswered?: (label: string, ms: number) => void;
}

export type Transport = <T>(
  prepare: () => { payload: string; parse: (body: string) => T },
  signal: AbortSignal | undefined,
  label: string,
  timeoutOverrideMs?: number
) => Promise<T>;

function failureForStatus(
  status: number
): 'unauthorized' | 'rate_limited' | 'server_error' | 'bad_request' {
  if (status === 401 || status === 403) {
    return 'unauthorized';
  }
  if (status === 429) {
    return 'rate_limited';
  }
  if (status >= 500) {
    return 'server_error';
  }
  return 'bad_request';
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
        throw new ClassificationError(
          'malformed_response',
          'classifier response too large',
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
    throw new ClassificationError(
      'unauthorized',
      'classifier requires an API key',
      {
        provider: providerId,
      }
    );
  }
  const endpoint = options.endpoint.trim();
  if (!endpoint) {
    throw new ClassificationError(
      'bad_request',
      'classifier requires an endpoint',
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
    throw new ClassificationError(
      'bad_request',
      'invalid classifier retry limit',
      {
        provider: providerId,
      }
    );
  }
  const fetchImpl: ClassificationFetch = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new ClassificationError(
      'network',
      'no fetch implementation available',
      {
        provider: providerId,
      }
    );
  }

  async function resolveKey(
    refresh: boolean,
    signal: AbortSignal
  ): Promise<string | null> {
    if (typeof credential === 'string') {
      return credential.trim() || null;
    }
    if (!credential) {
      return null;
    }
    const minted: string = await credential({ refresh, signal });
    const token = typeof minted === 'string' ? minted.trim() : '';
    if (!token) {
      throw new ClassificationError(
        'unauthorized',
        'credential function returned no token',
        {
          provider: providerId,
        }
      );
    }
    return token;
  }

  return async function send(prepare, callerSignal, label, timeoutOverrideMs) {
    const started = performance.now();
    return withClassificationDeadline(
      providerId,
      timeoutOverrideMs ?? timeoutMs,
      callerSignal,
      async (signal, waitFor) => {
        let prepared: ReturnType<typeof prepare>;
        try {
          prepared = prepare();
        } catch (error) {
          if (error instanceof ClassificationError) {
            throw error;
          }
          throw new ClassificationError(
            'bad_request',
            'invalid classifier request',
            {
              provider: providerId,
            }
          );
        }
        const { payload, parse } = prepared;
        await waitFor(Promise.resolve());
        const wasAborted = (): boolean => signal.aborted;
        const interrupted = (): ClassificationError =>
          new ClassificationError(
            callerSignal?.aborted === true ? 'aborted' : 'timeout',
            'classifier request interrupted',
            { provider: providerId }
          );
        let refreshed = false;
        let refreshKey = false;
        let lastError: ClassificationError | undefined;
        for (let attemptNo = 0; attemptNo <= maxRetries; attemptNo++) {
          if (wasAborted()) {
            throw interrupted();
          }
          try {
            const key = await waitFor(resolveKey(refreshKey, signal));
            const response = await waitFor(
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
            if (!response.ok) {
              void response.body?.cancel().catch(() => {});
              const error = new ClassificationError(
                failureForStatus(response.status),
                `classifier returned HTTP ${response.status}`,
                { provider: providerId, status: response.status }
              );
              error.retryAfterMs = retryAfterMs(
                response.headers.get('retry-after')
              );
              throw error;
            }
            const body = await readResponse(response.body, waitFor, providerId);
            const result = await waitFor(Promise.resolve(parse(body)));
            try {
              onAnswered?.(label, performance.now() - started);
            } catch {
              // Observability must not turn a valid answer into a retried request.
            }
            return result;
          } catch (error) {
            if (wasAborted()) {
              throw interrupted();
            }
            lastError =
              error instanceof ClassificationError
                ? error
                : new ClassificationError(
                  'network',
                  'classifier request failed',
                  {
                    provider: providerId,
                  }
                );
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
              if (error instanceof ClassificationError) {
                throw error;
              }
              throw new ClassificationError(
                'network',
                'classifier backoff failed',
                {
                  provider: providerId,
                }
              );
            }
          }
        }
        throw (
          lastError ??
          new ClassificationError('network', 'classifier request failed', {
            provider: providerId,
          })
        );
      }
    );
  };
}

/**
 * Retry policy for provider 429s.
 *
 * LangChain refuses to retry a 429 whose text reads like a spent allowance:
 * `classifyRateLimitError` matches `/insufficient[_ -]?quota/i` and
 * `/exceeded (?:your|the current|the available).+quota/i`, returns `stop`, and
 * its default failed-attempt handler turns that into a thrown
 * `RateLimitQuotaExhaustedError` before p-retry ever gets a second attempt.
 * For OpenAI that is correct - `insufficient_quota` means the account is out
 * of credit and waiting achieves nothing.
 *
 * Scaleway answers all three of its rate limits with the same wording, and
 * nothing else to go on:
 *
 *     HTTP/2 429
 *     content-type: application/json
 *     {"status":429,"error":"INSUFFICIENT QUOTA",
 *      "message":"You exceeded your current limit of concurrent requests."}
 *
 * Measured against api.scaleway.ai on 2026-07-30. The tokens-per-minute and
 * requests-per-minute limits differ from that only in `message` - and `message`
 * is exactly what the OpenAI SDK drops, since `APIError` keeps `body.error`
 * alone. So the classifier only ever sees `429 "INSUFFICIENT QUOTA"`, reads it
 * as a billing wall, and kills the run. There are no `x-ratelimit-*` and no
 * `retry-after` headers on the 429 either, although successful responses do
 * carry the former.
 *
 * All three of those limits clear on their own - in milliseconds for
 * concurrency, inside the minute for the token bucket - so the attempt should
 * wait and try again instead of failing the turn. This handler restores that
 * while keeping every genuinely terminal case terminal: a structured
 * `insufficient_quota`, wording that names billing or credit, and a single
 * request too large for the per-minute allowance, which no amount of waiting
 * can make fit.
 */

import { parseRetryAfterMs } from '@langchain/core/utils/async_caller';
import type { FailedAttemptHandler } from '@langchain/core/utils/async_caller';
import { extractErrorMessage, isContextOverflowError } from '@/utils/errors';

/**
 * Statuses LangChain never retries. Mirrored rather than imported because
 * `defaultFailedAttemptHandler` is not exported, and this handler replaces it
 * wholesale - anything missing here would turn into a retry loop on an error
 * that can only ever fail.
 */
const STATUS_NO_RETRY = new Set([400, 401, 402, 403, 404, 405, 406, 407, 409]);

/**
 * Wording that means the allowance will not come back by itself.
 *
 * `insufficient_quota` is matched with the underscore only: that spelling is
 * OpenAI's error code for an account out of credit, while Scaleway writes its
 * transient rate limit as "INSUFFICIENT QUOTA" with a space.
 */
const BILLING_EXHAUSTED_RE =
  /insufficient_quota|billing|credit balance|out of credits|payment required|purchase|upgrade your plan|exceeded your monthly/i;

/**
 * A `Retry-After` beyond this is capacity planning rather than throttling.
 * Failing fast is better than holding a request open for minutes; the value
 * matches LangChain's own threshold for the same decision.
 */
const MAX_RETRY_AFTER_WAIT_MS = 60_000;

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null
    ? (value as Record<string, unknown>)
    : undefined;
}

function getStatus(error: unknown): number | undefined {
  const record = asRecord(error);
  if (record == null) {
    return undefined;
  }
  if (typeof record.status === 'number') {
    return record.status;
  }
  if (typeof record.statusCode === 'number') {
    return record.statusCode;
  }
  const response = asRecord(record.response);
  return typeof response?.status === 'number' ? response.status : undefined;
}

function getErrorCode(error: unknown): string | undefined {
  const record = asRecord(error);
  if (record == null) {
    return undefined;
  }
  if (typeof record.code === 'string') {
    return record.code;
  }
  const nested = asRecord(record.error);
  return typeof nested?.code === 'string' ? nested.code : undefined;
}

/** Reads `retry-after` from either a `Headers` instance or a plain object. */
function getRetryAfterHeader(error: unknown): string | undefined {
  const record = asRecord(error);
  const containers = [record?.headers, asRecord(record?.response)?.headers];
  for (const container of containers) {
    if (container == null) {
      continue;
    }
    const getter = (container as { get?: unknown }).get;
    if (typeof getter === 'function') {
      const value = (container as Headers).get('retry-after');
      if (value != null) {
        return value;
      }
      continue;
    }
    const plain = asRecord(container);
    const value = plain?.['retry-after'] ?? plain?.['Retry-After'];
    if (typeof value === 'string') {
      return value;
    }
  }
  return undefined;
}

/** Requests the caller cancelled. Retrying one would ignore the cancellation. */
function isAborted(error: unknown): boolean {
  const record = asRecord(error);
  if (record == null) {
    return false;
  }
  const message = typeof record.message === 'string' ? record.message : '';
  return (
    message.startsWith('Cancel') ||
    message.startsWith('AbortError') ||
    record.name === 'AbortError' ||
    record.code === 'ECONNABORTED'
  );
}

function toError(error: unknown, fallbackMessage: string): Error {
  if (error instanceof Error) {
    return error;
  }
  const coerced = new Error(extractErrorMessage(error) || fallbackMessage);
  const record = asRecord(error);
  if (record != null) {
    Object.assign(coerced, record);
  }
  return coerced;
}

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Whether a 429 describes a limit that clears on its own.
 *
 * Exported for the retry decision to be assertable without driving a live
 * provider.
 */
export function isRetryableRateLimit(error: unknown): boolean {
  if (getStatus(error) !== 429) {
    return false;
  }
  if (getErrorCode(error) === 'insufficient_quota') {
    return false;
  }
  if (BILLING_EXHAUSTED_RE.test(extractErrorMessage(error))) {
    return false;
  }
  /**
   * A 429 that is really "this one request is bigger than the whole per-minute
   * allowance" never fits, however long the caller waits. Throwing hands it to
   * the graph's overflow recovery, which shrinks the prompt instead.
   */
  return !isContextOverflowError(error);
}

/**
 * Failed-attempt handler for OpenAI-compatible providers.
 *
 * Returning lets p-retry apply its randomized exponential backoff; throwing
 * ends the attempts and surfaces the error.
 */
export const handleRateLimitedAttempt: FailedAttemptHandler = async (
  error: unknown
): Promise<void> => {
  if (isAborted(error)) {
    throw toError(error, 'Request was aborted');
  }

  const status = getStatus(error);
  if (status != null && STATUS_NO_RETRY.has(status)) {
    throw toError(error, `Request failed with status ${status}`);
  }

  /** Connection errors and 5xx: nothing to classify, the backoff is enough. */
  if (status !== 429) {
    return;
  }

  if (!isRetryableRateLimit(error)) {
    throw toError(error, 'Rate limit exceeded');
  }

  const retryAfterMs = parseRetryAfterMs(getRetryAfterHeader(error));
  if (retryAfterMs == null) {
    return;
  }
  if (retryAfterMs > MAX_RETRY_AFTER_WAIT_MS) {
    throw toError(error, 'Rate limit exceeded');
  }
  await wait(retryAfterMs);
};

/**
 * Adds the handler to constructor fields unless the caller brought its own.
 */
export function withRateLimitRetry<T extends object>(fields?: T): T {
  const next = (fields ?? {}) as T & {
    onFailedAttempt?: FailedAttemptHandler;
  };
  if (next.onFailedAttempt != null) {
    return next;
  }
  return { ...next, onFailedAttempt: handleRateLimitedAttempt };
}

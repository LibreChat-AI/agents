import type { SummarizationErrorCode } from '@/types/summarize';
import {
  extractErrorMessage,
  isContextOverflowError,
  isProviderCancellation,
} from '@/utils/errors';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { StreamLimitExceededError } from '@/llm/streamLimits';

const MAX_ATTEMPTS = 3;
const BACKOFF_MS = 250;
const MAX_ERROR_DEPTH = 4;

export class ReasoningOnlySummaryError extends Error {
  constructor() {
    super('The summarizer returned reasoning without summary text');
    this.name = 'ReasoningOnlySummaryError';
  }
}

interface ProviderError {
  name?: string;
  code?: string | number;
  type?: string;
  status?: number;
  statusCode?: number;
  httpStatusCode?: number;
  cause?: unknown;
  error?: unknown;
  body?: unknown;
  data?: unknown;
  response?: unknown;
  $metadata?: unknown;
}

export interface SummarizationRetryOptions {
  signal?: AbortSignal;
  /** Reports the upcoming attempt (2 or 3), without provider error text. */
  onRetry?: (retry: {
    attempt: number;
    errorCode: SummarizationErrorCode;
  }) => void;
}

function* errorChain(error: unknown, depth = 0): Generator<unknown> {
  if (error == null || depth > MAX_ERROR_DEPTH) {
    return;
  }
  yield error;
  if (typeof error !== 'object') {
    return;
  }
  const record = error as ProviderError;
  for (const nested of [
    record.cause,
    record.error,
    record.body,
    record.data,
    record.response,
    record.$metadata,
  ]) {
    yield* errorChain(nested, depth + 1);
  }
}

function describeFailure(error: unknown): {
  errorCode: SummarizationErrorCode;
  retryable: boolean;
} {
  if (isProviderCancellation(error)) {
    return { errorCode: 'unknown', retryable: false };
  }
  let errorCode: SummarizationErrorCode = isContextOverflowError(error)
    ? 'context_exceeded'
    : 'unknown';
  let retryable = error instanceof ReasoningOnlySummaryError;

  for (const cause of errorChain(error)) {
    const record: ProviderError =
      typeof cause === 'object' && cause !== null ? cause : {};
    const text = `${record.name ?? ''} ${record.code ?? ''} ${record.type ?? ''} ${extractErrorMessage(cause)}`;
    if (
      cause instanceof StreamLimitExceededError ||
      cause instanceof ProviderTextProtectionError ||
      /^(?:StreamLimitExceededError|ProviderTextProtectionError)$/.test(
        record.name ?? ''
      )
    ) {
      return { errorCode: 'unknown', retryable: false };
    }

    if (record.name === 'SummarizationInputTooLargeError') {
      errorCode = 'context_exceeded';
    }
    const status =
      record.status ??
      record.statusCode ??
      record.httpStatusCode ??
      record.code;
    const overloaded =
      Number(status) === 529 || /\boverloaded(?:_error)?\b/i.test(text);
    const rateLimited =
      Number(status) === 429 ||
      /rate[_ -]?limit|too many requests|ThrottlingException/i.test(text);
    const timeout =
      Number(status) === 408 ||
      /timed?[_ -]?out|timeout|ETIMEDOUT|ESOCKETTIMEDOUT|UND_ERR_(?:CONNECT|HEADERS|BODY)_TIMEOUT/i.test(
        text
      );
    retryable ||=
      overloaded ||
      rateLimited ||
      timeout ||
      (Number(status) >= 500 && Number(status) < 600) ||
      /internal[_ ]server[_ ]error|service[_ ]unavailable|\b(?:HTTP|status(?: code)?)\s*[:=]?\s*5\d\d\b|ECONNRESET|ECONNABORTED|EPIPE|UND_ERR_SOCKET|ERR_STREAM_PREMATURE_CLOSE|socket hang up|socket closed|connection (?:reset|closed|terminated)|terminated|premature (?:close|end)|stream (?:disconnected|interrupted|closed)|fetch failed/i.test(
        text
      );
    if (errorCode === 'context_exceeded') {
      continue;
    }
    if (overloaded) {
      errorCode = 'provider_overloaded';
    } else if (rateLimited && errorCode !== 'provider_overloaded') {
      errorCode = 'rate_limited';
    } else if (timeout && errorCode === 'unknown') {
      errorCode = 'timeout';
    }
  }

  return {
    errorCode,
    retryable: retryable && errorCode !== 'context_exceeded',
  };
}

/** Maps provider failures to safe event metadata, never provider text. */
export function classifySummarizationError(
  error: unknown
): SummarizationErrorCode {
  return describeFailure(error).errorCode;
}

function waitForRetry(ms: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Runs at most three attempts. Pass the same signal to the provider callback. */
export async function withSummarizationRetry<T>(
  operation: () => Promise<T>,
  { signal, onRetry }: SummarizationRetryOptions = {}
): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    signal?.throwIfAborted();
    try {
      const result = await operation();
      signal?.throwIfAborted();
      return result;
    } catch (error) {
      signal?.throwIfAborted();
      const failure = describeFailure(error);
      if (!failure.retryable || attempt === MAX_ATTEMPTS) {
        throw error;
      }
      onRetry?.({ attempt: attempt + 1, errorCode: failure.errorCode });
      const delay = BACKOFF_MS * 2 ** (attempt - 1) * (1 + Math.random());
      await waitForRetry(Math.floor(delay), signal);
    }
  }
}

import type { SummarizationErrorCode } from '@/types/summarize';
import type * as t from '@/types';
import {
  extractErrorMessage,
  isContextOverflowError,
  isProviderCancellation,
} from '@/utils/errors';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

const MAX_ATTEMPTS = 3;
const BACKOFF_MS = 250;
const BACKPRESSURE_BACKOFF_MS = 2000;
const MAX_RETRY_DELAY_MS = 60_000;
const MAX_ERROR_DEPTH = 4;

/** Compaction owns retries; model/transport defaults must not multiply attempts. */
export function withSummarizationRetryOwner<T extends object>(
  provider: t.ProviderName,
  clientOptions: T
): T & { maxRetries: number } {
  const options = { ...clientOptions, maxRetries: 0 };
  const family = getProviderFamily(provider);
  // Moonshot keeps generic request shaping but uses the OpenAI transport.
  if (
    provider === Providers.OPENAI ||
    provider === Providers.AZURE ||
    provider === Providers.OPENROUTER ||
    provider === Providers.MOONSHOT ||
    family === 'openai'
  ) {
    const openai = clientOptions as Partial<t.OpenAIClientOptions>;
    return {
      ...options,
      configuration: { ...openai.configuration, maxRetries: 0 },
    };
  }
  if (provider === Providers.ANTHROPIC || family === 'anthropic') {
    const anthropic = clientOptions as Partial<t.AnthropicClientOptions>;
    return {
      ...options,
      clientOptions: { ...anthropic.clientOptions, maxRetries: 0 },
    };
  }
  if (provider === Providers.BEDROCK || family === 'bedrock') {
    const bedrock = clientOptions as Partial<t.BedrockAnthropicInput>;
    return {
      ...options,
      clientOptions: { ...bedrock.clientOptions, maxAttempts: 1 },
    };
  }
  return options;
}

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
  headers?: Headers | Record<string, string | number | undefined>;
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
    delayMs?: number;
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

function parseRetryAfter(
  value: string | number | null | undefined,
  unitMs: number
): number {
  if (typeof value !== 'string' && typeof value !== 'number') {
    return 0;
  }
  const text = String(value).trim();
  if (/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) {
    return Math.min(MAX_RETRY_DELAY_MS, Math.ceil(Number(text) * unitMs));
  }
  if (unitMs !== 1000 || !/^[a-z]{3}[a-z,]*\s/i.test(text)) {
    return 0;
  }
  const date = Date.parse(text);
  return Number.isFinite(date)
    ? Math.min(MAX_RETRY_DELAY_MS, Math.max(0, date - Date.now()))
    : 0;
}

function retryHeaderDelay(headers: ProviderError['headers']): number {
  if (headers == null || typeof headers !== 'object') {
    return 0;
  }
  if (typeof headers.get === 'function') {
    return Math.max(
      parseRetryAfter(headers.get('retry-after'), 1000),
      parseRetryAfter(headers.get('retry-after-ms'), 1)
    );
  }
  let delay = 0;
  for (const [name, value] of Object.entries(headers)) {
    const header = name.toLowerCase();
    if (header === 'retry-after' || header === 'retry-after-ms') {
      delay = Math.max(
        delay,
        parseRetryAfter(value, header === 'retry-after' ? 1000 : 1)
      );
    }
  }
  return delay;
}

function retryDelay(
  error: unknown,
  errorCode: SummarizationErrorCode,
  attempt: number
): number {
  const backpressure =
    errorCode === 'provider_overloaded' || errorCode === 'rate_limited';
  const base = backpressure ? BACKPRESSURE_BACKOFF_MS : BACKOFF_MS;
  const jitter = 1 + Math.random() * (backpressure ? 0.5 : 1);
  let delay = Math.floor(base * 2 ** (attempt - 1) * jitter);
  for (const cause of errorChain(error)) {
    if (typeof cause === 'object' && cause !== null) {
      delay = Math.max(
        delay,
        retryHeaderDelay((cause as ProviderError).headers)
      );
    }
  }
  return Math.min(MAX_RETRY_DELAY_MS, delay);
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
      const delayMs = retryDelay(error, failure.errorCode, attempt);
      onRetry?.({
        attempt: attempt + 1,
        errorCode: failure.errorCode,
        delayMs,
      });
      await waitForRetry(delayMs, signal);
    }
  }
}

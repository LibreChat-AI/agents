import { APIError } from 'openai';
import { describe, expect, it } from '@jest/globals';
import {
  isRetryableRateLimit,
  withRateLimitRetry,
  handleRateLimitedAttempt,
} from '@/utils/rateLimit';
import {
  ChatXAI,
  ChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  AzureChatOpenAI,
} from '@/llm/openai';
import { ChatOpenRouter } from '@/llm/openrouter';

/**
 * Builds the error the OpenAI SDK actually throws for a given response body, so
 * the assertions run against the shape production sees - `message` derived from
 * `body.error` alone, with `body.message` already gone.
 */
function apiError(
  status: number,
  body: object,
  headers: Record<string, string> = {}
): Error {
  return APIError.generate(
    status,
    body,
    undefined,
    new Headers({ 'content-type': 'application/json', ...headers })
  ) as Error;
}

/** Captured from api.scaleway.ai on 2026-07-30; only `message` differs per axis. */
const scalewayBody = (message: string) => ({
  status: 429,
  error: 'INSUFFICIENT QUOTA',
  message,
});

const scalewayConcurrency = () =>
  apiError(
    429,
    scalewayBody('You exceeded your current limit of concurrent requests.')
  );
const scalewayTokensPerMinute = () =>
  apiError(
    429,
    scalewayBody('You exceeded your current quota of tokens per minute.')
  );

/** OpenAI's genuine billing wall: a structured code, not just quota wording. */
const openAIInsufficientQuota = () =>
  apiError(429, {
    error: {
      code: 'insufficient_quota',
      type: 'insufficient_quota',
      message:
        'You exceeded your current quota, please check your plan and billing details.',
    },
  });

/** A 429 that is really an oversized request - waiting can never make it fit. */
const openAIRequestTooLarge = () =>
  apiError(429, {
    error: {
      code: 'rate_limit_exceeded',
      type: 'tokens',
      message:
        'Request too large for gpt-5-nano in organization org-test on tokens per min (TPM): Limit 200000, Requested 480002. The input or output tokens must be reduced in order to run successfully.',
    },
  });

describe('isRetryableRateLimit', () => {
  it('retries every Scaleway rate limit, which all read "INSUFFICIENT QUOTA"', () => {
    expect(isRetryableRateLimit(scalewayConcurrency())).toBe(true);
    expect(isRetryableRateLimit(scalewayTokensPerMinute())).toBe(true);
  });

  it('does not retry a structured insufficient_quota', () => {
    expect(isRetryableRateLimit(openAIInsufficientQuota())).toBe(false);
  });

  it('does not retry wording that names billing or credit', () => {
    expect(
      isRetryableRateLimit(
        apiError(429, {
          error: {
            message: 'Your credit balance is too low to access this model.',
          },
        })
      )
    ).toBe(false);
  });

  it('does not retry a request too large for the per-minute allowance', () => {
    expect(isRetryableRateLimit(openAIRequestTooLarge())).toBe(false);
  });

  it('is only about 429s', () => {
    expect(
      isRetryableRateLimit(apiError(500, { error: { message: 'oops' } }))
    ).toBe(false);
    expect(isRetryableRateLimit(new Error('nothing to do with HTTP'))).toBe(
      false
    );
  });
});

describe('handleRateLimitedAttempt', () => {
  const resolves = (error: unknown) =>
    expect(handleRateLimitedAttempt(error)).resolves.toBeUndefined();

  it('lets the attempt be retried for a Scaleway 429', async () => {
    await resolves(scalewayConcurrency());
    await resolves(scalewayTokensPerMinute());
  });

  it('rethrows a spent quota so the run ends instead of looping', async () => {
    await expect(
      handleRateLimitedAttempt(openAIInsufficientQuota())
    ).rejects.toThrow(/insufficient_quota|quota/i);
  });

  it('rethrows an oversized request, leaving it to overflow recovery', async () => {
    await expect(
      handleRateLimitedAttempt(openAIRequestTooLarge())
    ).rejects.toThrow(/Request too large/i);
  });

  it('keeps the never-retry statuses terminal', async () => {
    for (const status of [400, 401, 402, 403, 404, 405, 406, 407, 409]) {
      await expect(
        handleRateLimitedAttempt(
          apiError(status, { error: { message: 'nope' } })
        )
      ).rejects.toThrow();
    }
  });

  it('retries what is worth retrying: connection errors and 5xx', async () => {
    await resolves(apiError(500, { error: { message: 'internal' } }));
    await resolves(apiError(503, { error: { message: 'unavailable' } }));
    await resolves(
      Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' })
    );
  });

  it('never retries a cancelled request', async () => {
    await expect(
      handleRateLimitedAttempt(
        Object.assign(new Error('boom'), { name: 'AbortError' })
      )
    ).rejects.toThrow();
    await expect(
      handleRateLimitedAttempt(new Error('Cancel: user left'))
    ).rejects.toThrow();
    await expect(
      handleRateLimitedAttempt(
        Object.assign(new Error('gone'), { code: 'ECONNABORTED' })
      )
    ).rejects.toThrow();
  });

  it('waits out a short Retry-After', async () => {
    const started = Date.now();
    await resolves(scalewayBodyWithRetryAfter('1'));
    expect(Date.now() - started).toBeGreaterThanOrEqual(900);
  });

  it('gives up when Retry-After is longer than a minute', async () => {
    await expect(
      handleRateLimitedAttempt(scalewayBodyWithRetryAfter('3600'))
    ).rejects.toThrow();
  });
});

function scalewayBodyWithRetryAfter(seconds: string): Error {
  return apiError(
    429,
    scalewayBody('You exceeded your current quota of tokens per minute.'),
    {
      'retry-after': seconds,
    }
  );
}

describe('withRateLimitRetry', () => {
  it('leaves a caller-supplied handler alone', () => {
    const own = async (): Promise<void> => undefined;
    expect(withRateLimitRetry({ onFailedAttempt: own }).onFailedAttempt).toBe(
      own
    );
  });

  it('handles being called without fields', () => {
    expect(
      withRateLimitRetry<{ onFailedAttempt?: unknown }>().onFailedAttempt
    ).toBe(handleRateLimitedAttempt);
  });
});

describe('OpenAI-compatible clients', () => {
  /** The retry policy is only worth anything if it reaches the model's caller. */
  it.each([
    ['ChatOpenAI', () => new ChatOpenAI({ apiKey: 'test' })],
    [
      'AzureChatOpenAI',
      () =>
        new AzureChatOpenAI({
          azureOpenAIApiKey: 'test',
          azureOpenAIApiDeploymentName: 'd',
          azureOpenAIApiVersion: 'v',
          azureOpenAIApiInstanceName: 'i',
        }),
    ],
    ['ChatDeepSeek', () => new ChatDeepSeek({ apiKey: 'test' })],
    ['ChatXAI', () => new ChatXAI({ apiKey: 'test' })],
    ['ChatMoonshot', () => new ChatMoonshot({ apiKey: 'test' })],
    ['ChatOpenRouter', () => new ChatOpenRouter({ apiKey: 'test' })],
  ])('%s retries transient 429s', (_name, create) => {
    const model = create() as unknown as {
      caller: { onFailedAttempt?: unknown };
    };
    expect(model.caller.onFailedAttempt).toBe(handleRateLimitedAttempt);
  });

  it('still lets a caller override the policy', () => {
    const own = async (): Promise<void> => undefined;
    const model = new ChatOpenAI({
      apiKey: 'test',
      onFailedAttempt: own,
    }) as unknown as { caller: { onFailedAttempt?: unknown } };
    expect(model.caller.onFailedAttempt).toBe(own);
  });
});

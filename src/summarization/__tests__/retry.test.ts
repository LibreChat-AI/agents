import {
  classifySummarizationError,
  ReasoningOnlySummaryError,
  withSummarizationRetry,
} from '@/summarization/retry';
import { ProviderTextProtectionError } from '@/protection/providerText';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { isProviderCancellation } from '@/utils/errors';

function streamLimitError(): StreamLimitExceededError {
  return new StreamLimitExceededError({
    kind: 'delta_events',
    limit: 5,
    observed: 6,
  });
}

const timeoutErrors = [
  [
    'native aborted timeout',
    new DOMException(
      'The operation was aborted due to timeout',
      'TimeoutError'
    ),
  ],
  [
    'provider timeout name',
    { name: 'TimeoutError', message: 'Operation aborted' },
  ],
  [
    'SDK aborted timeout',
    { name: 'APIConnectionTimeoutError', message: 'Request was aborted' },
  ],
  [
    'aborted timeout message',
    new Error('The operation was aborted due to timeout'),
  ],
  [
    'aborted timed-out request',
    new Error('Request was aborted because it timed out'),
  ],
  ['timeout-qualified abort text', 'AbortError: request timed out'],
  [
    'wrapped native timeout',
    new Error('Provider failed', {
      cause: new DOMException(
        'The operation was aborted due to timeout',
        'TimeoutError'
      ),
    }),
  ],
  [
    'wrapped timeout name',
    { error: { name: 'TimeoutError', message: 'Operation aborted' } },
  ],
  [
    'wrapped SDK timeout name',
    {
      error: {
        name: 'APIConnectionTimeoutError',
        message: 'Request was aborted',
      },
    },
  ],
  [
    'nested provider timeout',
    {
      response: {
        data: {
          error: new DOMException(
            'The operation was aborted due to timeout',
            'TimeoutError'
          ),
        },
      },
    },
  ],
] as const;

const transientErrors = [
  ...timeoutErrors,
  ['reasoning-only summary', new ReasoningOnlySummaryError()],
  ['overload status', { status: 529 }],
  ['overload stream event', { error: { type: 'overloaded_error' } }],
  ['overload message', new Error('Overloaded')],
  ['rate limit status', { status: 429 }],
  ['rate limit code', { code: 429 }],
  ['rate limit stream event', { error: { type: 'rate_limit_error' } }],
  ['rate limit message', new Error('Too many requests')],
  ['throttling', { name: 'ThrottlingException' }],
  ['server error', { status: 500 }],
  ['bad gateway', { response: { status: 502 } }],
  ['service unavailable', { statusCode: 503 }],
  ['gateway timeout', { status: 504 }],
  ['other server error', { status: 599 }],
  ['Bedrock metadata', { $metadata: { httpStatusCode: 503 } }],
  ['server stream event', { error: { type: 'internal_server_error' } }],
  ['status message', new Error('HTTP 503')],
  ['request timeout status', { status: 408 }],
  ['SDK timeout', { name: 'APIConnectionTimeoutError' }],
  ['timeout message', new Error('Request timed out')],
  ['timeout code', { code: 'ETIMEDOUT' }],
  ['connection timeout', { cause: { code: 'UND_ERR_CONNECT_TIMEOUT' } }],
  ['body timeout', { cause: { code: 'UND_ERR_BODY_TIMEOUT' } }],
  ['connection reset', { cause: { code: 'ECONNRESET' } }],
  ['connection abort', { code: 'ECONNABORTED' }],
  ['broken pipe', { code: 'EPIPE' }],
  ['closed socket', { cause: { code: 'UND_ERR_SOCKET' } }],
  ['premature close', { code: 'ERR_STREAM_PREMATURE_CLOSE' }],
  ['socket hangup', new Error('socket hang up')],
  ['mid-stream disconnect', new Error('Stream disconnected')],
  ['fetch disconnect', new TypeError('terminated')],
  ['fetch failure', new TypeError('fetch failed')],
] as const;

const cancellationErrors = [
  ['user abort', new DOMException('Stopped', 'AbortError')],
  [
    'abort with timeout text',
    new DOMException('Request timed out', 'AbortError'),
  ],
  ['abort code with timeout', { name: 'TimeoutError', code: 'ABORT_ERR' }],
  ['cancel code with timeout', { name: 'TimeoutError', code: 'ERR_CANCELED' }],
  [
    'user abort with timeout',
    new Error('The user aborted a request after a timeout'),
  ],
  [
    'timeout wrapping cancellation',
    Object.assign(
      new DOMException(
        'The operation was aborted due to timeout',
        'TimeoutError'
      ),
      {
        cause: { cause: { code: 'ERR_CANCELED' } },
      }
    ),
  ],
  ['SDK user abort', { name: 'APIUserAbortError', status: 503 }],
  ['abort name', { name: 'AbortError' }],
  ['SDK abort name', { name: 'APIUserAbortError' }],
  ['canceled name', { name: 'CanceledError' }],
  ['cancelled name', { name: 'CancelledError' }],
  ['abort code only', { code: 'ABORT_ERR' }],
  ['cancellation code only', { code: 'ERR_CANCELED' }],
  ['abort code', { code: 'ABORT_ERR', message: 'connection reset' }],
  ['cancellation code', { code: 'ERR_CANCELED', status: 503 }],
  ['user abort message', new Error('The user aborted a request.')],
  ['user canceled message', new Error('The user canceled the call.')],
  ['user cancelled message', new Error('The user cancelled the call.')],
  ['request abort message', new Error('Request was aborted')],
  ['request canceled message', new Error('Request canceled')],
  ['request cancelled message', new Error('Request was cancelled')],
  ['operation abort message', new Error('Operation aborted')],
  ['operation canceled message', new Error('Operation was canceled')],
  ['operation cancelled message', new Error('Operation cancelled')],
  ['bare abort message', new Error('AbortError')],
  ['abort text', 'AbortError'],
  ['cancellation text', 'REQUEST WAS CANCELED'],
  ['nested cause', { cause: { cause: { code: 'ERR_CANCELED' } } }],
  [
    'rate-limited cancellation',
    { status: 429, cause: { name: 'CancelledError' } },
  ],
  [
    'overloaded cancellation',
    { error: { type: 'overloaded_error' }, cause: { code: 'ERR_CANCELED' } },
  ],
] as const;

const fatalErrors = [
  ['stream limit', streamLimitError()],
  ['protection timeout', new ProviderTextProtectionError('timeout')],
  ['protection unavailable', new ProviderTextProtectionError('unavailable')],
  ['protection blocked', new ProviderTextProtectionError('blocked')],
  ...cancellationErrors,
  ['wrapped breaker', { status: 503, cause: streamLimitError() }],
  [
    'wrapped protection',
    { status: 503, cause: new ProviderTextProtectionError('timeout') },
  ],
  [
    'wrapped user abort',
    { status: 503, cause: new DOMException('Stopped', 'AbortError') },
  ],
  [
    'duplicate protection class',
    { name: 'ProviderTextProtectionError', code: 'timeout' },
  ],
  [
    'duplicate stream limit class',
    { name: 'StreamLimitExceededError', status: 503 },
  ],
  ['context overflow', { status: 400, code: 'context_length_exceeded' }],
  ['local input budget', { name: 'SummarizationInputTooLargeError' }],
  [
    'wrapped input budget',
    { status: 503, cause: { name: 'SummarizationInputTooLargeError' } },
  ],
  [
    'oversized token bucket',
    { status: 429, message: 'Request too large. Limit 100, Requested 101' },
  ],
  ['bad request', { status: 400 }],
  ['unauthorized', { status: 401 }],
  ['forbidden', { status: 403 }],
  ['not found', { status: 404 }],
  ['unknown failure', new Error('Unexpected summary format')],
] as const;

describe('withSummarizationRetry', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.spyOn(Math, 'random').mockReturnValue(0.5);
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it('returns the original result without scheduling retries', async () => {
    const summary = { text: 'summary' };
    const operation = jest.fn().mockResolvedValue(summary);
    const onRetry = jest.fn();

    await expect(withSummarizationRetry(operation, { onRetry })).resolves.toBe(
      summary
    );
    expect(operation).toHaveBeenCalledTimes(1);
    expect(onRetry).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it.each(transientErrors)('retries %s', async (_label, error) => {
    const operation = jest
      .fn()
      .mockRejectedValueOnce(error)
      .mockResolvedValue('summary');
    const pending = withSummarizationRetry(operation);

    await jest.runAllTimersAsync();

    await expect(pending).resolves.toBe('summary');
    expect(operation).toHaveBeenCalledTimes(2);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('retries a real provider AbortSignal.timeout reason without aborting the caller', async () => {
    jest.useRealTimers();
    const providerSignal = AbortSignal.timeout(0);
    await new Promise((resolve) => setTimeout(resolve, 10));
    jest.useFakeTimers();
    const controller = new AbortController();
    const operation = jest
      .fn()
      .mockRejectedValueOnce(providerSignal.reason)
      .mockResolvedValue('summary');
    const onRetry = jest.fn();
    const pending = withSummarizationRetry(operation, {
      signal: controller.signal,
      onRetry,
    });

    await jest.runAllTimersAsync();

    await expect(pending).resolves.toBe('summary');
    expect(providerSignal.aborted).toBe(true);
    expect(providerSignal.reason).toBeInstanceOf(DOMException);
    expect(providerSignal.reason.name).toBe('TimeoutError');
    expect(isProviderCancellation(providerSignal.reason)).toBe(false);
    expect(classifySummarizationError(providerSignal.reason)).toBe('timeout');
    expect(operation).toHaveBeenCalledTimes(2);
    expect(onRetry.mock.calls).toEqual([
      [{ attempt: 2, errorCode: 'timeout' }],
    ]);
    expect(controller.signal.aborted).toBe(false);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('preserves native timeout classification after exhausting retries', async () => {
    const error = new DOMException(
      'The operation was aborted due to timeout',
      'TimeoutError'
    );
    const operation = jest.fn().mockRejectedValue(error);
    const onRetry = jest.fn();
    const pending = withSummarizationRetry(operation, { onRetry });
    const rejected = expect(pending).rejects.toBe(error);

    await jest.runAllTimersAsync();
    await rejected;

    expect(operation).toHaveBeenCalledTimes(3);
    expect(onRetry.mock.calls).toEqual([
      [{ attempt: 2, errorCode: 'timeout' }],
      [{ attempt: 3, errorCode: 'timeout' }],
    ]);
    expect(classifySummarizationError(error)).toBe('timeout');
    expect(isProviderCancellation(error)).toBe(false);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('does not retry a timeout when the caller signal is explicitly aborted', async () => {
    const reason = new DOMException(
      'The operation was aborted due to timeout',
      'TimeoutError'
    );
    const signal = AbortSignal.abort(reason);
    const operation = jest.fn();

    await expect(withSummarizationRetry(operation, { signal })).rejects.toBe(
      reason
    );
    expect(operation).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it.each(fatalErrors)('never retries %s', async (_label, error) => {
    const operation = jest.fn().mockRejectedValue(error);
    const onRetry = jest.fn();

    await expect(withSummarizationRetry(operation, { onRetry })).rejects.toBe(
      error
    );
    expect(operation).toHaveBeenCalledTimes(1);
    expect(onRetry).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it('caps attempts at three with jittered exponential backoff', async () => {
    const first = { status: 529 };
    const second = { status: 429 };
    const last = { status: 503 };
    const operation = jest
      .fn()
      .mockRejectedValueOnce(first)
      .mockRejectedValueOnce(second)
      .mockRejectedValue(last);
    const onRetry = jest.fn();
    const pending = withSummarizationRetry(operation, { onRetry });
    const rejected = expect(pending).rejects.toBe(last);

    await jest.advanceTimersByTimeAsync(0);
    expect(operation).toHaveBeenCalledTimes(1);
    await jest.advanceTimersByTimeAsync(374);
    expect(operation).toHaveBeenCalledTimes(1);
    await jest.advanceTimersByTimeAsync(1);
    expect(operation).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(749);
    expect(operation).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(1);
    await rejected;

    expect(operation).toHaveBeenCalledTimes(3);
    expect(onRetry.mock.calls).toEqual([
      [{ attempt: 2, errorCode: 'provider_overloaded' }],
      [{ attempt: 3, errorCode: 'rate_limited' }],
    ]);
    expect(Math.random).toHaveBeenCalledTimes(2);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('does not start a call for an already aborted signal', async () => {
    const controller = new AbortController();
    const reason = new Error('User stopped');
    controller.abort(reason);
    const operation = jest.fn();

    await expect(
      withSummarizationRetry(operation, { signal: controller.signal })
    ).rejects.toBe(reason);
    expect(operation).not.toHaveBeenCalled();
  });

  it('preserves user cancellation even when the provider reports a transient error', async () => {
    const controller = new AbortController();
    const reason = new Error('User stopped');
    const operation = jest.fn(async () => {
      controller.abort(reason);
      throw new Error('Request timed out');
    });
    const onRetry = jest.fn();

    await expect(
      withSummarizationRetry(operation, { signal: controller.signal, onRetry })
    ).rejects.toBe(reason);
    expect(operation).toHaveBeenCalledTimes(1);
    expect(onRetry).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it.each([
    ['user cancellation', new Error('User stopped')],
    ['tripped breaker', streamLimitError()],
    ['protection cancellation', new ProviderTextProtectionError('blocked')],
  ])(
    'interrupts backoff for %s and releases its timer and listener',
    async (_label, reason) => {
      const controller = new AbortController();
      const removeListener = jest.spyOn(
        controller.signal,
        'removeEventListener'
      );
      const operation = jest.fn().mockRejectedValue({ status: 503 });
      const pending = withSummarizationRetry(operation, {
        signal: controller.signal,
      });
      const rejected = expect(pending).rejects.toBe(reason);
      await jest.advanceTimersByTimeAsync(0);
      expect(jest.getTimerCount()).toBe(1);

      controller.abort(reason);
      await rejected;
      await jest.runAllTimersAsync();

      expect(operation).toHaveBeenCalledTimes(1);
      expect(jest.getTimerCount()).toBe(0);
      expect(removeListener).toHaveBeenCalledWith(
        'abort',
        expect.any(Function)
      );
    }
  );

  it('does not return a result after the run was aborted', async () => {
    const controller = new AbortController();
    const reason = streamLimitError();
    const operation = jest.fn(async () => {
      controller.abort(reason);
      return 'summary';
    });

    await expect(
      withSummarizationRetry(operation, { signal: controller.signal })
    ).rejects.toBe(reason);
    expect(operation).toHaveBeenCalledTimes(1);
  });

  it('checks cancellation from onRetry before scheduling backoff', async () => {
    const controller = new AbortController();
    const reason = streamLimitError();
    const operation = jest.fn().mockRejectedValue({ status: 503 });

    await expect(
      withSummarizationRetry(operation, {
        signal: controller.signal,
        onRetry: () => controller.abort(reason),
      })
    ).rejects.toBe(reason);
    expect(operation).toHaveBeenCalledTimes(1);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('removes the abort listener when backoff finishes normally', async () => {
    const controller = new AbortController();
    const addListener = jest.spyOn(controller.signal, 'addEventListener');
    const removeListener = jest.spyOn(controller.signal, 'removeEventListener');
    const operation = jest
      .fn()
      .mockRejectedValueOnce({ status: 503 })
      .mockResolvedValue('summary');
    const pending = withSummarizationRetry(operation, {
      signal: controller.signal,
    });

    await jest.runAllTimersAsync();
    await expect(pending).resolves.toBe('summary');

    expect(addListener).toHaveBeenCalledTimes(1);
    expect(removeListener).toHaveBeenCalledWith(
      'abort',
      addListener.mock.calls[0][1]
    );
    expect(jest.getTimerCount()).toBe(0);
  });

  it('does not send provider text to onRetry', async () => {
    const operation = jest
      .fn()
      .mockRejectedValueOnce(new Error('Overloaded: secret provider body'))
      .mockResolvedValue('summary');
    const onRetry = jest.fn();
    const pending = withSummarizationRetry(operation, { onRetry });

    await jest.runAllTimersAsync();
    await pending;

    expect(onRetry.mock.calls).toEqual([
      [{ attempt: 2, errorCode: 'provider_overloaded' }],
    ]);
  });
});

describe('isProviderCancellation', () => {
  it.each(cancellationErrors)(
    'recognizes %s despite transient wrappers',
    (_label, error) => {
      for (const failure of [
        error,
        { status: 429, cause: error },
        { error: { type: 'overloaded_error' }, cause: error },
      ]) {
        expect(isProviderCancellation(failure)).toBe(true);
        expect(classifySummarizationError(failure)).toBe('unknown');
      }
    }
  );

  it.each(['cause', 'error', 'body', 'data', 'response', '$metadata'])(
    'recognizes cancellation nested under %s',
    (key) => {
      const error = { status: 503, [key]: { code: 'ERR_CANCELED' } };

      expect(isProviderCancellation(error)).toBe(true);
      expect(classifySummarizationError(error)).toBe('unknown');
    }
  );

  it.each(transientErrors)(
    'does not treat %s as cancellation',
    (_label, error) => {
      expect(isProviderCancellation(error)).toBe(false);
    }
  );

  it.each([
    ['stream limit', streamLimitError()],
    ['protection timeout', new ProviderTextProtectionError('timeout')],
    ['protection unavailable', new ProviderTextProtectionError('unavailable')],
    ['protection blocked', new ProviderTextProtectionError('blocked')],
    ['wrapped stream limit', { cause: streamLimitError() }],
    [
      'wrapped protection',
      { cause: new ProviderTextProtectionError('blocked') },
    ],
    ['duplicate stream limit class', { name: 'StreamLimitExceededError' }],
    ['duplicate protection class', { name: 'ProviderTextProtectionError' }],
    ['provider busy', new Error('Provider is busy')],
    ['deadline timeout', new DOMException('Deadline elapsed', 'TimeoutError')],
    ['context overflow', { code: 'context_length_exceeded' }],
    ['local input budget', { name: 'SummarizationInputTooLargeError' }],
    ['unrelated failure', new Error('Unexpected summary format')],
    ['null', null],
    ['undefined', undefined],
    ['number', 17],
    ['symbol', Symbol('failure')],
  ])('does not treat %s as cancellation', (_label, error) => {
    expect(isProviderCancellation(error)).toBe(false);
  });

  it('bounds inspection of cyclic causes', () => {
    const error = new Error('Provider is busy');
    error.cause = error;

    expect(isProviderCancellation(error)).toBe(false);

    error.cause = { cause: error, code: 'ERR_CANCELED' };
    expect(isProviderCancellation(error)).toBe(true);
  });

  it('recognizes code-only cancellation through depth four, but not deeper', () => {
    const error = {
      cause: { cause: { cause: { cause: { code: 'ERR_CANCELED' } } } },
    };

    expect(isProviderCancellation(error)).toBe(true);
    expect(classifySummarizationError(error)).toBe('unknown');
    expect(isProviderCancellation({ cause: error })).toBe(false);
  });
});

describe('classifySummarizationError', () => {
  it.each(timeoutErrors)('classifies %s as timeout', (_label, error) => {
    expect(classifySummarizationError(error)).toBe('timeout');
  });

  it.each([
    [{ status: 529 }, 'provider_overloaded'],
    [
      { message: 'Gateway failed', error: { type: 'overloaded_error' } },
      'provider_overloaded',
    ],
    [
      new Error(
        '{"error":{"type":"overloaded_error","message":"private body"}}'
      ),
      'provider_overloaded',
    ],
    [
      { response: { data: { error: { type: 'overloaded_error' } } } },
      'provider_overloaded',
    ],
    [{ status: 429 }, 'rate_limited'],
    [{ code: 'rate_limit_exceeded' }, 'rate_limited'],
    [{ name: 'ThrottlingException' }, 'rate_limited'],
    [
      { status: 429, message: 'Request too large. Limit 100, Requested 100' },
      'rate_limited',
    ],
    [
      { status: 429, message: 'Request too large. Limit 100, Requested 101' },
      'context_exceeded',
    ],
    [{ error: { code: 'context_length_exceeded' } }, 'context_exceeded'],
    [
      new Error('prompt is too long: 120 tokens > 100 maximum'),
      'context_exceeded',
    ],
    [
      { name: 'ContextOverflowError', cause: { status: 400 } },
      'context_exceeded',
    ],
    [
      {
        response: {
          data: {
            message:
              'input token count exceeds the maximum number of tokens allowed (100)',
          },
        },
      },
      'context_exceeded',
    ],
    [{ name: 'SummarizationInputTooLargeError' }, 'context_exceeded'],
    [
      { cause: { name: 'SummarizationInputTooLargeError' } },
      'context_exceeded',
    ],
    [{ status: 408 }, 'timeout'],
    [{ name: 'APIConnectionTimeoutError' }, 'timeout'],
    [{ cause: { code: 'ETIMEDOUT' } }, 'timeout'],
    [new DOMException('Deadline elapsed', 'TimeoutError'), 'timeout'],
    [new Error('Request timed out with private provider text'), 'timeout'],
    [{ status: 503 }, 'unknown'],
    [{ cause: { code: 'ECONNRESET' } }, 'unknown'],
    [new Error('secret provider body'), 'unknown'],
    [new ProviderTextProtectionError('timeout'), 'unknown'],
    [streamLimitError(), 'unknown'],
    [new DOMException('Stopped', 'AbortError'), 'unknown'],
    [null, 'unknown'],
    [undefined, 'unknown'],
    [17, 'unknown'],
    [Symbol('failure'), 'unknown'],
  ])('classifies %p as %s', (error, expected) => {
    expect(classifySummarizationError(error)).toBe(expected);
  });

  it('bounds inspection of cyclic causes', () => {
    const error: Error & { status: number } = Object.assign(
      new Error('server failed'),
      { status: 503 }
    );
    error.cause = error;

    expect(classifySummarizationError(error)).toBe('unknown');
  });
});

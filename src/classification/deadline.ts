import { ClassificationError } from './types';

export type AwaitWithinDeadline = <T>(task: Promise<T>) => Promise<T>;

/** Bounds even non-cooperative credential minters, fetches, body reads, and model calls. */
export async function withClassificationDeadline<T>(
  provider: string,
  timeoutMs: number,
  callerSignal: AbortSignal | undefined,
  operation: (signal: AbortSignal, waitFor: AwaitWithinDeadline) => Promise<T>
): Promise<T> {
  if (
    !Number.isSafeInteger(timeoutMs) ||
    timeoutMs <= 0 ||
    timeoutMs > 3_600_000
  ) {
    throw new ClassificationError('bad_request', 'invalid classifier timeout', {
      provider,
    });
  }
  if (callerSignal?.aborted === true) {
    throw new ClassificationError('aborted', 'caller aborted the request', {
      provider,
    });
  }

  const expiresAt = performance.now() + timeoutMs;
  const deadline = new AbortController();
  const signal = callerSignal
    ? AbortSignal.any([callerSignal, deadline.signal])
    : deadline.signal;
  const timer = setTimeout(() => deadline.abort(), timeoutMs);
  const expired = (): boolean =>
    signal.aborted || performance.now() >= expiresAt;
  const abortError = (): ClassificationError =>
    callerSignal?.aborted === true
      ? new ClassificationError('aborted', 'caller aborted the request', {
        provider,
      })
      : new ClassificationError('timeout', 'classifier deadline exceeded', {
        provider,
      });

  const waitFor: AwaitWithinDeadline = <U>(task: Promise<U>): Promise<U> =>
    new Promise<U>((resolve, reject) => {
      if (expired()) {
        reject(abortError());
        return;
      }
      const onAbort = (): void => {
        signal.removeEventListener('abort', onAbort);
        reject(abortError());
      };
      signal.addEventListener('abort', onAbort, { once: true });
      if (expired()) {
        onAbort();
      }
      task.then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          if (expired()) {
            reject(abortError());
          } else {
            resolve(value);
          }
        },
        (error: unknown) => {
          signal.removeEventListener('abort', onAbort);
          reject(expired() ? abortError() : error);
        }
      );
    });

  try {
    return await waitFor(operation(signal, waitFor));
  } finally {
    clearTimeout(timer);
  }
}

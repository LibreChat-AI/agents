type Outcome<T> =
  | { status: 'completed'; value: T }
  | { status: 'cancelled' | 'timed_out' | 'error'; message: string };

export function isAborted(signal: AbortSignal): boolean {
  return signal.aborted;
}

/** A deadline stops admission immediately; settlement still tracks the actual operation. */
export async function withinDeadline<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  parent: AbortSignal,
  timeoutMs: number,
  settled: () => void
): Promise<Outcome<T>> {
  if (isAborted(parent)) {
    settled();
    return { status: 'cancelled', message: 'verification cancelled' };
  }
  const controller = new AbortController();
  const deadline = performance.now() + timeoutMs;
  const timedOut = (): Outcome<T> => ({
    status: 'timed_out',
    message: `verification timed out after ${timeoutMs} ms`,
  });
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort = (): void => undefined;
  const interrupted = new Promise<Outcome<T>>((resolve) => {
    onAbort = (): void => {
      resolve({ status: 'cancelled', message: 'verification cancelled' });
      controller.abort();
    };
    parent.addEventListener('abort', onAbort, { once: true });
    timer = setTimeout(() => {
      resolve(timedOut());
      controller.abort();
    }, timeoutMs);
  });
  const running = Promise.resolve()
    .then(() => {
      controller.signal.throwIfAborted();
      return operation(controller.signal);
    })
    .then(
      (value): Outcome<T> => ({ status: 'completed', value }),
      (error: unknown): Outcome<T> => ({
        status: 'error',
        message: error instanceof Error ? error.message : String(error),
      })
    )
    .finally(settled);
  try {
    const result = await Promise.race([running, interrupted]);
    if (isAborted(parent)) return { status: 'cancelled', message: 'verification cancelled' };
    if (performance.now() >= deadline) {
      controller.abort();
      return timedOut();
    }
    return result;
  } finally {
    clearTimeout(timer);
    parent.removeEventListener('abort', onAbort);
  }
}

/** At most `size` tasks run at once; the rest wait their turn in arrival order. */
export interface RerankSemaphore {
  readonly size: number;
  readonly active: number;
  readonly waiting: number;
  run<T>(task: () => Promise<T>): Promise<T>;
}

/** Share one semaphore between rerankers that call the same host. */
export function createSemaphore(size: number): RerankSemaphore {
  const limit = Number.isFinite(size) && size >= 1 ? Math.floor(size) : 1;
  const queue: Array<() => void> = [];
  let active = 0;
  const next = (): void => {
    if (active >= limit) {
      return;
    }
    const start = queue.shift();
    if (start) {
      active += 1;
      start();
    }
  };
  return {
    size: limit,
    get active(): number {
      return active;
    },
    get waiting(): number {
      return queue.length;
    },
    run<T>(task: () => Promise<T>): Promise<T> {
      return new Promise<T>((resolve, reject) => {
        queue.push(() => {
          Promise.resolve()
            .then(task)
            .then(resolve, reject)
            .finally(() => {
              active -= 1;
              next();
            });
        });
        next();
      });
    },
  };
}

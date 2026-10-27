import { callHook } from './hook';

describe('callHook', () => {
  it('returns a synchronous result and drops a synchronous throw', () => {
    expect(callHook(() => 3)).toBe(3);
    expect(callHook(() => undefined)).toBeUndefined();
    expect(
      callHook(() => {
        throw new Error('sync');
      })
    ).toBeUndefined();
  });

  it('handles a returned rejection without waiting for it', async () => {
    const seen: unknown[] = [];
    const listener = (reason: unknown): void => {
      seen.push(reason);
    };
    process.on('unhandledRejection', listener);
    try {
      expect(
        callHook(() => Promise.reject(new Error('async')))
      ).toBeUndefined();
      expect(
        callHook(async () => {
          throw new Error('async');
        })
      ).toBeUndefined();
      expect(
        callHook(() => ({
          then: (): never => {
            throw new Error('then');
          },
        }))
      ).toBeUndefined();
      await new Promise((resolve) => setTimeout(resolve, 20));
    } finally {
      process.off('unhandledRejection', listener);
    }
    expect(seen).toEqual([]);
  });
});

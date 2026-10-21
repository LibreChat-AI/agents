/**
 * Calls a caller-supplied hook whose failure must not reach the caller's
 * work: a synchronous throw is dropped, and so is the rejection of a returned
 * promise, which a `void`-typed hook may still produce (TypeScript accepts an
 * `async` function there) and which would otherwise be unhandled. Returns the
 * hook's synchronous result, or `undefined` when it threw or returned a
 * promise; nothing waits for that promise.
 */
export function callHook(hook: () => unknown): unknown {
  let result: unknown;
  try {
    result = hook();
  } catch {
    return undefined;
  }
  if (
    result === null ||
    (typeof result !== 'object' && typeof result !== 'function')
  ) {
    return result;
  }
  try {
    const then: unknown = (result as { then?: unknown }).then;
    if (typeof then !== 'function') {
      return result;
    }
    then.call(result, undefined, () => {});
  } catch {
    // A thenable whose `then` throws is as unusable as a rejection.
  }
  return undefined;
}

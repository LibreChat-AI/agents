import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';

/** Correlation only: never an authorization identity or a callback/output capability. */
export interface ModelInvocationContext {
  readonly version: 1;
  readonly runId: string;
}

const contexts = new WeakMap<object, Readonly<ModelInvocationContext>>();

export function getModelInvocationContext(
  options: object,
  manager?: CallbackManagerForLLMRun
): Readonly<ModelInvocationContext> | undefined {
  if (manager != null)
    return Object.freeze({ version: 1, runId: manager.runId });
  return contexts.get(options);
}

/** Internal option identity survives callback suppression without entering provider parameters. */
export function preserveModelInvocationContext<T extends object>(
  source: object,
  target: T,
  manager?: CallbackManagerForLLMRun
): T {
  const context = getModelInvocationContext(source, manager);
  if (context != null) contexts.set(target, context);
  return target;
}

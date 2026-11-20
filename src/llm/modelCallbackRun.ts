import { getCallbackManagerForConfig } from '@langchain/core/runnables';
import {
  CallbackManager,
  CallbackManagerForLLMRun,
} from '@langchain/core/callbacks/manager';
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';

/**
 * Every callbacks source the real model run would compose beyond the per-call
 * config. `model` here is whatever `createCallModel` produced — with tools
 * that is `bindTools(...)`'s `RunnableBinding`, and a system runnable pipes a
 * `RunnableSequence` on top — while `clientOptions.callbacks` lives on the
 * chat model at the BOTTOM of that stack. Walks `bound` (bindings) and
 * `last`/`steps` (sequences), collecting each wrapper's own `callbacks` and
 * any binding-config callbacks along the way, since the binding merges its
 * config into the call before the chat model composes.
 */
function collectModelCallbackSources(model: unknown): Callbacks[] {
  const sources: Callbacks[] = [];
  const seen = new Set<unknown>();
  let current: unknown = model;
  while (current != null && typeof current === 'object' && !seen.has(current)) {
    seen.add(current);
    const wrapper = current as {
      callbacks?: Callbacks;
      config?: { callbacks?: Callbacks };
      bound?: unknown;
      last?: unknown;
      steps?: unknown[];
    };
    if (wrapper.callbacks != null) {
      sources.push(wrapper.callbacks);
    }
    if (wrapper.config?.callbacks != null) {
      sources.push(wrapper.config.callbacks);
    }
    current =
      wrapper.bound ??
      wrapper.last ??
      (Array.isArray(wrapper.steps)
        ? wrapper.steps[wrapper.steps.length - 1]
        : undefined);
  }
  return sources;
}

export interface ModelCallbackRunContext {
  parentRunId?: string;
  tags?: string[];
  metadata?: RunnableConfig['metadata'];
}

/** Reconstructs the actual native run for early closes, including model-level callback sources. */
export async function getModelCallbackRunManager(
  runId: string,
  config?: RunnableConfig,
  model?: unknown,
  observed?: ModelCallbackRunContext
): Promise<CallbackManagerForLLMRun | undefined> {
  let manager = await getCallbackManagerForConfig(config);
  for (const source of collectModelCallbackSources(model)) {
    manager =
      CallbackManager.configure(manager ?? undefined, source) ?? manager;
  }
  if (manager == null) return undefined;
  return new CallbackManagerForLLMRun(
    runId,
    manager.handlers,
    manager.inheritableHandlers,
    observed?.tags ?? manager.tags,
    manager.inheritableTags,
    observed?.metadata ?? manager.metadata,
    manager.inheritableMetadata,
    observed ? observed.parentRunId : manager.getParentRunId()
  );
}

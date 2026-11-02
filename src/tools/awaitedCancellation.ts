import { DynamicStructuredTool } from '@langchain/core/tools';
import { patchConfig, pickRunnableConfigKeys } from '@langchain/core/runnables';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type { DynamicStructuredToolInput } from '@langchain/core/tools';
import type { GenericTool } from '@/types';

const AWAITED_CANCELLATION = Symbol('librechat.tool.awaitedCancellation');

/** Only opt in when invoke settles after all owned work and cleanup have finished. */
export function withAwaitedCancellation<T extends GenericTool>(tool: T): T {
  Object.defineProperty(tool, AWAITED_CANCELLATION, { value: true });
  return tool;
}

export function supportsAwaitedCancellation(tool: GenericTool): boolean {
  return (
    (tool as GenericTool & { [AWAITED_CANCELLATION]?: boolean })[
      AWAITED_CANCELLATION
    ] === true
  );
}

/** Preserves tool() callback scoping without its early-reject abort race. */
export function createAwaitedTool(
  fields: DynamicStructuredToolInput
): DynamicStructuredTool {
  return withAwaitedCancellation(
    new DynamicStructuredTool({
      ...fields,
      func: (input, manager, config): ReturnType<typeof fields.func> => {
        const childConfig = patchConfig(config, {
          callbacks: manager?.getChild(),
        });
        return AsyncLocalStorageProviderSingleton.runWithConfig(
          pickRunnableConfigKeys(childConfig),
          () => fields.func(input, manager, childConfig)
        );
      },
    })
  );
}

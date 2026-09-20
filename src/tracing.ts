import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { LLMResult } from '@langchain/core/outputs';
import {
  createLangfuseHandler,
  disposeLangfuseHandler,
  withLangfuseAttributes,
} from '@/langfuse';
import { initializeLangfuseTracing } from '@/instrumentation';

export type ModelInvocationTrace = Pick<
  Parameters<typeof createLangfuseHandler>[0],
  | 'langfuse'
  | 'runId'
  | 'traceIdSeed'
  | 'sessionId'
  | 'userId'
  | 'tags'
  | 'traceName'
  | 'traceMetadata'
> & { runId: string; provider: string; model: string };

const safely = async (work: () => void | Promise<void>): Promise<void> => {
  try {
    await work();
  } catch {
    /* Observability cannot replace an inference result or expose provider errors. */
  }
};

/** Owns tracing lifecycle for a model call outside a Run; an optional projection traces raw clients. */
export async function traceModelInvocation<T>(
  params: ModelInvocationTrace,
  work: (callbacks?: Callbacks) => Promise<T>,
  project?: (result: T) => LLMResult
): Promise<T> {
  let handler: ReturnType<typeof createLangfuseHandler>;
  try {
    initializeLangfuseTracing(params.langfuse);
    handler = createLangfuseHandler(params);
  } catch {
    return work();
  }
  if (!handler) return work();
  const trace = handler;
  const state = { invoked: false };
  try {
    return await withLangfuseAttributes(params, async () => {
      state.invoked = true;
      if (project) {
        await safely(() =>
          trace.handleLLMStart(
            {
              lc: 1,
              type: 'constructor',
              id: ['model', params.provider],
              kwargs: {},
            },
            ['[Model input omitted]'],
            params.runId,
            undefined,
            { invocation_params: { model_name: params.model } },
            [],
            { model: params.model }
          )
        );
      }
      try {
        const result = await work(project ? undefined : [trace]);
        if (project) {
          await safely(() => {
            let output: LLMResult;
            try {
              output = project(result);
            } catch {
              output = { generations: [[{ text: '[Model output omitted]' }]] };
            }
            return trace.handleLLMEnd(output, params.runId);
          });
        }
        return result;
      } catch (error) {
        if (project) {
          await safely(() =>
            trace.handleLLMError(
              new Error('Model invocation failed.'),
              params.runId
            )
          );
        }
        throw error;
      }
    });
  } catch (error) {
    if (!state.invoked) return work();
    throw error;
  } finally {
    void safely(() => disposeLangfuseHandler(trace));
  }
}

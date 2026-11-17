import { AIMessageChunk, mergeUsageMetadata } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ChatGeneration } from '@langchain/core/outputs';
import type { ModelCallbackRunContext } from '@/llm/modelCallbackRun';
import type { HostModelErrorCode } from './model';
import { getModelCallbackRunManager } from '@/llm/modelCallbackRun';
import { HostModelServiceError } from './model';
import { appendCallbacks } from '@/utils/callbacks';

/** Observes native completion; only synthesizes an end when iterator return bypasses it. */
export function modelStreamCallbacks(
  model: object,
  config: RunnableConfig
): {
  config: RunnableConfig;
  observe(chunk: AIMessageChunk): void;
  observeUsage(usage: UsageMetadata | undefined): void;
  finish(reason: HostModelErrorCode): Promise<void>;
} {
  let runId: string | undefined;
  let observed: ModelCallbackRunContext | undefined;
  let terminal = false;
  const isTerminal = (): boolean => terminal;
  let finishing: Promise<void> | undefined;
  let usage: UsageMetadata | undefined;
  const chunks: AIMessageChunk[] = [];
  const capture = BaseCallbackHandler.fromMethods({
    handleChatModelStart: (
      _model,
      _messages,
      id,
      parentRunId,
      _extra,
      tags,
      metadata
    ): void => {
      if (runId !== undefined) return;
      runId = id;
      observed = {
        parentRunId,
        tags: tags?.slice(),
        metadata: { ...metadata },
      };
    },
    handleLLMEnd: (_output, id): void => {
      if (id === runId) {
        terminal = true;
        chunks.length = 0;
        usage = undefined;
      }
    },
    handleLLMError: (_error, id): void => {
      if (id === runId) {
        terminal = true;
        chunks.length = 0;
        usage = undefined;
      }
    },
  });
  capture.name = 'host-model-stream-lifecycle';
  capture.awaitHandlers = true;
  const callConfig = {
    ...config,
    callbacks: appendCallbacks(config.callbacks, [capture]),
  };
  return {
    config: callConfig,
    observeUsage: (reported): void => {
      if (!terminal && reported !== undefined) {
        usage = mergeUsageMetadata(usage, reported);
      }
    },
    observe: (chunk): void => {
      if (!terminal) chunks.push(chunk);
    },
    finish: (reason): Promise<void> => {
      finishing ??= (async (): Promise<void> => {
        if (terminal || runId === undefined) return;
        const manager = await getModelCallbackRunManager(
          runId,
          callConfig,
          model,
          observed
        );
        if (isTerminal() || manager == null) return;
        let aggregate: AIMessageChunk;
        try {
          aggregate = chunks.reduce(
            (current, chunk) => current.concat(chunk),
            new AIMessageChunk('')
          );
        } catch {
          await manager.handleLLMError(
            new HostModelServiceError('incompatible')
          );
          return;
        } finally {
          chunks.length = 0;
        }
        aggregate.usage_metadata = usage;
        aggregate.lc_kwargs.usage_metadata = usage;
        const incomplete = usage === undefined;
        aggregate.response_metadata = {
          ...aggregate.response_metadata,
          host_model_terminated: reason,
          ...(incomplete ? { usage_incomplete: true } : {}),
        };
        aggregate.lc_kwargs.response_metadata = aggregate.response_metadata;
        // No invented token counts: native usage is preserved when the provider supplied it.
        const generation: ChatGeneration = {
          text: aggregate.text,
          message: aggregate,
        };
        await manager.handleLLMEnd({
          generations: [[generation]],
          llmOutput: {
            controlFlow: 'HostModelTerminated',
            reason,
            usage_incomplete: incomplete,
            ...(usage
              ? {
                tokenUsage: {
                  promptTokens: usage.input_tokens,
                  completionTokens: usage.output_tokens,
                  totalTokens: usage.total_tokens,
                },
              }
              : {}),
          },
        });
      })();
      return finishing;
    },
  };
}

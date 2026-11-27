import { z } from 'zod';
import { AsyncLocalStorage } from 'node:async_hooks';
import { ensureHandler } from '@langchain/core/callbacks/manager';
import type {
  CallbackManagerForLLMRun,
  Callbacks,
} from '@langchain/core/callbacks/manager';
import type { ChatResult, ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { BaseMessage } from '@langchain/core/messages';
import type { SdkRpc } from './peer';
import { HostChatModel, HostModelServiceError } from '@/index';

export interface RemoteModelAttempt {
  nodeId: string;
  modelId: string;
}
export const modelAttemptSchema: z.ZodType<RemoteModelAttempt> = z
  .object({ nodeId: z.string().uuid(), modelId: z.string().uuid() })
  .strict();
const invocationContextSchema = z
  .object({ version: z.literal(1), runId: z.string().uuid() })
  .strict();
/** Additive SDK capability: older SDKs keep the input-only model adapter during rollout. */
export const supportsModelInvocationContext: boolean =
  Number(HostChatModel.invocationContextVersion) === 1;
const blocked = (): HostModelServiceError =>
  new HostModelServiceError('blocked');

/** Run each iterator operation inside its scope; constructing an async generator does not enter it. */
async function* scopedStream<T, C>(
  storage: AsyncLocalStorage<C>,
  context: C,
  source: () => AsyncIterable<T>
): AsyncGenerator<T> {
  const iterator = storage.run(context, () => source()[Symbol.asyncIterator]());
  try {
    for (;;) {
      const next = await storage.run(context, () => iterator.next());
      if (next.done === true) return;
      yield next.value;
    }
  } finally {
    await storage.run(context, () => iterator.return?.());
  }
}

export interface HostModelLifecycle {
  attach(callbacks: Callbacks | undefined): boolean;
  readonly maxEntries: number;
  failure(): Error | undefined;
  invoke<T>(
    context: RemoteModelAttempt | undefined,
    operation: () => Promise<T>
  ): Promise<T>;
  stream<T>(
    context: RemoteModelAttempt | undefined,
    operation: () => AsyncIterable<T>
  ): AsyncIterable<T>;
  end(data: unknown): boolean;
  dispose(): void;
}

export interface WorkerModelLifecycle {
  callback: BaseCallbackHandler;
  current(context?: unknown): RemoteModelAttempt | undefined;
  model(options: ConstructorParameters<typeof HostChatModel>[0]): HostChatModel;
}

/** Only callbacks installed on this root execution can enroll a model attempt. No message text is relayed. */
export function createWorkerModelLifecycle(
  agentId: string,
  maxEntries: number,
  peer: { call: SdkRpc }
): WorkerModelLifecycle {
  const parents = new Map<string, string>();
  const nodes = new Set<string>();
  const models = new Map<string, RemoteModelAttempt>();
  const attemptedNodes = new Set<string>();
  const storage = new AsyncLocalStorage<RemoteModelAttempt | undefined>();
  let admitted = false;
  const budget = (): void => {
    if (parents.size + nodes.size + models.size >= maxEntries) throw blocked();
  };
  const context = (
    manager?: CallbackManagerForLLMRun
  ): RemoteModelAttempt | undefined => {
    if (admitted) return undefined;
    const value = manager && models.get(manager.runId);
    if (!value) throw blocked();
    attemptedNodes.add(value.nodeId);
    return value;
  };
  const end = async (nodeId: string, completed: boolean): Promise<void> => {
    parents.delete(nodeId);
    nodes.delete(nodeId);
    for (const [modelId, attempt] of models)
      if (attempt.nodeId === nodeId) models.delete(modelId);
    if (!attemptedNodes.delete(nodeId)) return;
    const reply = z
      .object({
        version: z.literal(1),
        ok: z.literal(true),
        admitted: z.boolean(),
      })
      .parse(await peer.call('model.node.end', { nodeId, completed }));
    if (reply.admitted) {
      admitted = true;
      parents.clear();
      nodes.clear();
      models.clear();
      attemptedNodes.clear();
    }
  };
  class CorrelatedHostModel extends HostChatModel {
    override _generate(
      messages: BaseMessage[],
      options: this['ParsedCallOptions'],
      manager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
      return storage.run(context(manager), () =>
        super._generate(messages, options)
      );
    }

    override async *_streamResponseChunks(
      messages: BaseMessage[],
      options: this['ParsedCallOptions'],
      manager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
      yield* scopedStream(storage, context(manager), () =>
        super._streamResponseChunks(messages, options, manager)
      );
    }
  }
  return {
    current(context): RemoteModelAttempt | undefined {
      if (admitted) return undefined;
      if (supportsModelInvocationContext) {
        const { runId } = invocationContextSchema.parse(context);
        const attempt = models.get(runId);
        if (!attempt) throw blocked();
        attemptedNodes.add(attempt.nodeId);
        return attempt;
      }
      const value = storage.getStore();
      if (!value) throw blocked();
      return value;
    },
    model: (options) =>
      supportsModelInvocationContext
        ? new HostChatModel(options)
        : new CorrelatedHostModel(options),
    callback: ensureHandler({
      name: 'librechat-remote-model-admission',
      raiseError: true,
      awaitHandlers: true,
      handleChainStart: (
        _chain,
        _inputs,
        runId,
        parentRunId,
        _tags,
        metadata,
        _type,
        name
      ) => {
        if (admitted) return;
        budget();
        if (parentRunId != null && parentRunId.length > 0)
          parents.set(runId, parentRunId);
        if (name === `agent=${agentId}` && metadata?.langgraph_node === name)
          nodes.add(runId);
      },
      handleChatModelStart: (
        _model,
        _messages,
        runId,
        parentRunId,
        _extra,
        _tags,
        metadata
      ) => {
        if (admitted) return;
        if (
          metadata?.summarization === true ||
          metadata?.agentId !== agentId ||
          metadata.langgraph_node !== `agent=${agentId}`
        )
          return;
        const seen = new Set<string>();
        let parent = parentRunId;
        while (parent != null && parent.length > 0 && !seen.has(parent)) {
          if (nodes.has(parent)) {
            budget();
            models.set(runId, { nodeId: parent, modelId: runId });
            return;
          }
          seen.add(parent);
          parent = parents.get(parent);
        }
      },
      handleChainEnd: async (output, runId) => {
        if (admitted) return;
        if (!nodes.has(runId)) {
          parents.delete(runId);
          return;
        }
        const result = output as {
          messages?: unknown;
          summarizationRequest?: unknown;
        } | null;
        await end(
          runId,
          Array.isArray(result?.messages) &&
            result.messages.length > 0 &&
            result.summarizationRequest == null
        );
      },
      handleChainError: (_error, runId) =>
        admitted ? undefined : end(runId, false),
    }),
  };
}

import { z } from 'zod';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  serialize as serializeNativeValue,
  deserialize as deserializeNativeValue,
} from 'node:v8';
import {
  isAIMessage,
  isAIMessageChunk,
  isBaseMessage,
  mapStoredMessagesToChatMessages,
} from '@langchain/core/messages';
import type {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  StoredMessage,
} from '@langchain/core/messages';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  HostModelDescriptor,
  HostModelRequest,
  HostModelService,
  HostModelErrorCode,
} from './model';
import type { ExecutionIdentity } from './identity';
import {
  executionIdentitySchema,
  snapshotExecutionIdentity,
  matchesExecutionIdentity,
} from './identity';
import {
  hostModelDescriptorSchema,
  HostModelServiceError,
  normalizeHostModelError,
} from './model';
import { modelStreamCallbacks } from './modelStreamCallbacks';
import { getProviderFamily } from '@/llm/providers';
import { toStoredMessage } from '@/messages/stored';

interface ModelLease {
  charge(bytes: number): void;
  retain(): void;
  release(): void;
}

/** Shared host scope; admission remains charged until provider work and iterator cleanup settle. */
export class HostModelAdmission {
  private calls = 0;
  private bytes = 0;
  private retainedCalls = 0;
  private retainedBytes = 0;
  private readonly maxConcurrentCalls: number;
  private readonly maxRequestBytes: number;

  constructor(limits: { maxConcurrentCalls: number; maxRequestBytes: number }) {
    positive(limits.maxConcurrentCalls);
    positive(limits.maxRequestBytes);
    this.maxConcurrentCalls = limits.maxConcurrentCalls;
    this.maxRequestBytes = limits.maxRequestBytes;
  }

  snapshot(): Readonly<{
    calls: number;
    requestBytes: number;
    retainedCalls: number;
    retainedRequestBytes: number;
  }> {
    return Object.freeze({
      calls: this.calls,
      requestBytes: this.bytes,
      retainedCalls: this.retainedCalls,
      retainedRequestBytes: this.retainedBytes,
    });
  }

  /** @internal The binding owns this lease; transports cannot release outstanding provider work. */
  acquire(bytes: number): ModelLease {
    positive(bytes);
    if (
      this.calls >= this.maxConcurrentCalls ||
      bytes > this.maxRequestBytes - this.bytes
    )
      throw new HostModelServiceError('overflow');
    this.calls++;
    this.bytes += bytes;
    let released = false;
    let retained = false;
    return {
      charge: (amount): void => {
        positive(amount);
        if (released) throw new HostModelServiceError('incompatible');
        if (amount > this.maxRequestBytes - this.bytes)
          throw new HostModelServiceError('overflow');
        bytes += amount;
        this.bytes += amount;
        if (retained) this.retainedBytes += amount;
      },
      retain: (): void => {
        if (released || retained) return;
        retained = true;
        this.retainedCalls++;
        this.retainedBytes += bytes;
      },
      release: (): void => {
        if (released) return;
        released = true;
        this.calls--;
        this.bytes -= bytes;
        if (!retained) return;
        this.retainedCalls--;
        this.retainedBytes -= bytes;
      },
    };
  }
}

function positive(value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new HostModelServiceError('incompatible');
}

export interface HostModelLimits {
  readonly timeoutMs: number;
  readonly maxRequestBytes: number;
  /** Cumulative serialized native-message bytes, including metadata and tool chunks. */
  readonly maxResponseBytes: number;
}

const requestSchema = z.object({
  version: z.literal(1),
  identity: executionIdentitySchema,
  reference: z.string(),
  messages: z.array(z.custom<BaseMessage>(isBaseMessage)),
  toolNames: z.array(z.string()),
  toolChoice: z
    .union([
      z.object({ mode: z.enum(['auto', 'none', 'required']) }),
      z.object({ mode: z.literal('tool'), name: z.string() }),
    ])
    .optional(),
});

function messageBytes(
  messages: readonly BaseMessage[],
  maxBytes: number
): number {
  let bytes = 2;
  try {
    for (const message of messages) {
      bytes += serializeNativeValue(toStoredMessage(message)).byteLength + 1;
      if (bytes > maxBytes) throw new HostModelServiceError('overflow');
    }
    return bytes;
  } catch (error) {
    if (error instanceof HostModelServiceError) throw error;
    throw new HostModelServiceError('incompatible');
  }
}

/** Tracks actual settlement separately from a caller's cancelled/timed-out wait. */
class ModelAttempt {
  readonly signal: AbortSignal;
  private readonly controller = new AbortController();
  private readonly timer: ReturnType<typeof setTimeout>;
  private pending = 0;
  private finished = false;
  private stopCleanup = (): void => {};
  private readonly onAbort = (): void => {
    this.finish();
  };

  constructor(
    private readonly lease: ModelLease,
    timeoutMs: number,
    signal?: AbortSignal
  ) {
    this.signal = signal
      ? AbortSignal.any([signal, this.controller.signal])
      : this.controller.signal;
    this.timer = setTimeout(
      () => this.controller.abort(new HostModelServiceError('timeout')),
      timeoutMs
    );
    this.timer.unref();
    this.signal.addEventListener('abort', this.onAbort, { once: true });
  }

  setCleanup(cleanup: () => void): void {
    this.stopCleanup = cleanup;
  }

  track<T>(promise: Promise<T>): Promise<T> {
    this.pending++;
    const tracked = promise.finally(() => {
      this.pending--;
      if (this.finished && this.pending === 0) this.lease.release();
    });
    void tracked.catch(() => undefined);
    return tracked;
  }

  async wait<T>(promise: Promise<T>): Promise<T> {
    this.signal.throwIfAborted();
    return new Promise<T>((resolve, reject) => {
      const abort = (): void => {
        this.signal.removeEventListener('abort', abort);
        reject(this.signal.reason);
      };
      this.signal.addEventListener('abort', abort, { once: true });
      promise.then(
        (value) => {
          this.signal.removeEventListener('abort', abort);
          if (this.signal.aborted) reject(this.signal.reason);
          else resolve(value);
        },
        (error: Error) => {
          this.signal.removeEventListener('abort', abort);
          reject(error);
        }
      );
      if (this.signal.aborted) abort();
    });
  }

  error(error: unknown): Error {
    if (
      this.signal.reason instanceof HostModelServiceError &&
      this.signal.reason.code === 'timeout'
    )
      return new HostModelServiceError('timeout');
    return normalizeHostModelError(error, this.signal);
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    clearTimeout(this.timer);
    this.signal.removeEventListener('abort', this.onAbort);
    this.controller.abort();
    this.stopCleanup();
    if (this.pending > 0) this.lease.retain();
    else this.lease.release();
  }
}

/** Binds native model execution to one authenticated execution; transport authentication remains external. */
export function createHostModelService(options: {
  identity: Readonly<ExecutionIdentity>;
  descriptor: HostModelDescriptor;
  model: BaseChatModel;
  admission: HostModelAdmission;
  limits: HostModelLimits;
  /** Trusted host configuration only. Worker config is never merged into it. */
  config?: RunnableConfig;
}): { descriptor: HostModelDescriptor; service: HostModelService } {
  const identity = snapshotExecutionIdentity(options.identity);
  const parsed = hostModelDescriptorSchema.safeParse(options.descriptor);
  if (!parsed.success) throw new HostModelServiceError('incompatible');
  const descriptor = structuredClone(parsed.data);
  const limits = Object.freeze({ ...options.limits });
  positive(limits.timeoutMs);
  positive(limits.maxRequestBytes);
  positive(limits.maxResponseBytes);
  if (limits.timeoutMs > 2147483647)
    throw new HostModelServiceError('incompatible');
  const tools = new Map(
    descriptor.tools.map((tool) => [tool.function.name, tool])
  );
  if (tools.size !== descriptor.tools.length)
    throw new HostModelServiceError('incompatible');
  const config = { ...options.config };
  const hostStorage = AsyncLocalStorageProviderSingleton.getInstance();
  const hostStore: unknown = hostStorage.run(undefined, () =>
    AsyncLocalStorageProviderSingleton.runWithConfig(
      config,
      () => hostStorage.getStore(),
      true
    )
  );
  function runOnHost<T>(operation: () => T): T {
    return hostStorage.run(hostStore, operation);
  }
  const model = options.model;
  const admission = options.admission;

  function prepare(
    request: HostModelRequest,
    signal?: AbortSignal
  ): { request: HostModelRequest; attempt: ModelAttempt } {
    const checked = requestSchema.safeParse(request);
    if (!checked.success) throw new HostModelServiceError('incompatible');
    const value = checked.data;
    if (
      !matchesExecutionIdentity(identity, value.identity) ||
      value.reference !== descriptor.reference
    )
      throw new HostModelServiceError('forbidden');
    const names = new Set(value.toolNames);
    const choice = value.toolChoice;
    if (
      names.size !== value.toolNames.length ||
      value.toolNames.some((name) => !tools.has(name)) ||
      (choice?.mode === 'tool' && !names.has(choice.name)) ||
      (choice?.mode === 'required' && names.size === 0)
    )
      throw new HostModelServiceError('forbidden');
    let boundSignal = signal;
    if (config.signal)
      boundSignal = signal
        ? AbortSignal.any([config.signal, signal])
        : config.signal;
    if (boundSignal?.aborted === true)
      throw new HostModelServiceError('cancelled');
    const lease = admission.acquire(2);
    try {
      const stored: StoredMessage[] = [];
      let bytes = 2;
      if (bytes > limits.maxRequestBytes)
        throw new HostModelServiceError('overflow');
      for (const message of value.messages) {
        const serialized = serializeNativeValue(toStoredMessage(message));
        const added = serialized.byteLength + 1;
        bytes += added;
        if (bytes > limits.maxRequestBytes)
          throw new HostModelServiceError('overflow');
        lease.charge(added);
        stored.push(deserializeNativeValue(serialized) as StoredMessage);
      }
      value.messages = mapStoredMessagesToChatMessages(stored);
      return {
        request: value,
        attempt: new ModelAttempt(lease, limits.timeoutMs, boundSignal),
      };
    } catch (error) {
      lease.release();
      if (error instanceof HostModelServiceError) throw error;
      throw new HostModelServiceError('incompatible');
    }
  }

  function selected(
    request: HostModelRequest
  ): ReturnType<BaseChatModel['withConfig']> {
    if (descriptor.tools.length === 0) return model.withConfig(config);
    if (!model.bindTools) throw new HostModelServiceError('incompatible');
    const choice = request.toolChoice;
    const family = getProviderFamily(descriptor.provider);
    const definitions = request.toolNames.map((name) =>
      structuredClone(tools.get(name)!)
    );
    if (choice?.mode === 'none' && family === 'bedrock')
      return model.bindTools([]).withConfig(config);
    if (choice?.mode !== 'tool')
      return model
        .bindTools(definitions, {
          ...(choice
            ? { tool_choice: choice.mode === 'required' ? 'any' : choice.mode }
            : {}),
        })
        .withConfig(config);
    if (family === 'openai' || family === 'mistral')
      return model
        .bindTools(definitions, {
          tool_choice: { type: 'function', function: { name: choice.name } },
        })
        .withConfig(config);
    if (family === 'anthropic')
      return model
        .bindTools(definitions, {
          tool_choice: { type: 'tool', name: choice.name },
        })
        .withConfig(config);
    if (family === 'bedrock')
      return model
        .bindTools(definitions, {
          tool_choice: { tool: { name: choice.name } },
        })
        .withConfig(config);
    if (family === 'google') {
      const googleChoice = {
        tool_choice: 'any',
        allowedFunctionNames: [choice.name],
        allowed_function_names: [choice.name],
      };
      return model.bindTools(definitions, googleChoice).withConfig(config);
    }
    // Unregistered custom models may accept names, but reserved words are ambiguous.
    if (['auto', 'none', 'any', 'required'].includes(choice.name))
      throw new HostModelServiceError('incompatible');
    return model
      .bindTools(definitions, { tool_choice: choice.name })
      .withConfig(config);
  }

  return {
    descriptor: structuredClone(descriptor),
    service: {
      version: 1,
      invoke: async (request, signal): Promise<AIMessage> => {
        const prepared = prepare(request, signal);
        const { attempt } = prepared;
        try {
          const operation = attempt.track(
            Promise.resolve().then(() => {
              attempt.signal.throwIfAborted();
              return runOnHost(() =>
                selected(prepared.request).invoke(
                  [...prepared.request.messages],
                  { signal: attempt.signal }
                )
              );
            })
          );
          const message = await attempt.wait(operation);
          if (!isAIMessage(message))
            throw new HostModelServiceError('incompatible');
          messageBytes([message], limits.maxResponseBytes);
          return message;
        } catch (error) {
          throw attempt.error(error);
        } finally {
          attempt.finish();
        }
      },
      stream: async function* (
        request,
        signal
      ): AsyncGenerator<AIMessageChunk> {
        if (!descriptor.streaming)
          throw new HostModelServiceError('incompatible');
        const prepared = prepare(request, signal);
        const { attempt } = prepared;
        let iterator: AsyncIterator<AIMessageChunk> | undefined;
        let callbacks: ReturnType<typeof modelStreamCallbacks> | undefined;
        let termination: HostModelErrorCode = 'cancelled';
        let closing = false;
        const close = (): void => {
          if (!iterator || closing) return;
          closing = true;
          const active = iterator;
          void attempt
            .track(
              Promise.resolve().then(() =>
                runOnHost(async () => {
                  try {
                    await active.return?.();
                  } finally {
                    const reason = attempt.signal.reason;
                    await callbacks?.finish(
                      reason instanceof HostModelServiceError
                        ? reason.code
                        : termination
                    );
                  }
                })
              )
            )
            .catch(() => undefined);
        };
        attempt.setCleanup(close);
        let responseBytes = 0;
        try {
          await attempt.wait(
            attempt.track(
              Promise.resolve().then(async () => {
                attempt.signal.throwIfAborted();
                // Use the native iterator/callback path: ReadableStream cancellation can
                // settle before the provider generator's asynchronous cleanup drains.
                const boundModel = runOnHost(() => selected(prepared.request));
                callbacks = modelStreamCallbacks(boundModel, {
                  signal: attempt.signal,
                });
                const stream = runOnHost(() =>
                  boundModel._streamIterator(
                    [...prepared.request.messages],
                    callbacks!.config
                  )
                );
                iterator = stream[Symbol.asyncIterator]();
                if (attempt.signal.aborted) close();
                attempt.signal.throwIfAborted();
              })
            )
          );
          for (;;) {
            const next = await attempt.wait(
              attempt.track(
                Promise.resolve().then(() => {
                  attempt.signal.throwIfAborted();
                  return runOnHost(() => iterator!.next());
                })
              )
            );
            if (next.done === true) {
              closing = true;
              return;
            }
            if (!isAIMessageChunk(next.value))
              throw new HostModelServiceError('incompatible');
            callbacks?.observeUsage(next.value.usage_metadata);
            responseBytes += messageBytes(
              [next.value],
              limits.maxResponseBytes
            );
            if (responseBytes > limits.maxResponseBytes)
              throw new HostModelServiceError('overflow');
            callbacks?.observe(next.value);
            yield next.value;
          }
        } catch (error) {
          const normalized = attempt.error(error);
          termination =
            normalized instanceof HostModelServiceError
              ? normalized.code
              : 'cancelled';
          throw normalized;
        } finally {
          attempt.finish();
        }
      },
    },
  };
}

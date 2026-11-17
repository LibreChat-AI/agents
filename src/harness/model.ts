import { z } from 'zod';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { isAIMessage, isAIMessageChunk } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import type {
  BaseChatModelCallOptions,
  BindToolsInput,
} from '@langchain/core/language_models/chat_models';
import type {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
} from '@langchain/core/messages';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { Runnable } from '@langchain/core/runnables';
import type { ChatResult } from '@langchain/core/outputs';
import type { ModelInvocationContext } from '@/llm/invocation';
import type { ExecutionIdentity } from './identity';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { stableStringify } from '@/tools/eagerEventExecution';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { getModelInvocationContext } from '@/llm/invocation';
import { snapshotExecutionIdentity } from './identity';

type FunctionTool = ReturnType<typeof convertToOpenAITool>;

/** Public configuration only. The authenticated host owns the actual model, credentials and callbacks. */
export interface HostModelDescriptor {
  readonly version: 1;
  readonly reference: string;
  readonly provider: string;
  readonly model: string;
  readonly streaming: boolean;
  readonly tools: readonly FunctionTool[];
}

const nonempty = z.string().refine((value) => value.trim().length > 0);
export const hostModelDescriptorSchema = z.object({
  version: z.literal(1),
  reference: nonempty,
  provider: nonempty,
  model: nonempty,
  streaming: z.boolean(),
  tools: z.array(
    z
      .object({
        type: z.literal('function'),
        function: z
          .object({
            name: nonempty,
            description: z.string().optional(),
            parameters: z.record(z.string(), z.unknown()),
            strict: z.boolean().nullable().optional(),
          })
          .loose(),
      })
      .loose()
  ),
});
const choiceSchema = z.union([
  nonempty,
  z.object({
    type: z.literal('function'),
    function: z.object({ name: nonempty }),
  }),
  z.object({ type: z.literal('tool'), name: nonempty }),
]);

export type HostModelToolChoice =
  | { readonly mode: 'auto' | 'none' | 'required' }
  | { readonly mode: 'tool'; readonly name: string };

export interface HostModelRequest {
  readonly version: 1;
  readonly identity: Readonly<ExecutionIdentity>;
  readonly reference: string;
  readonly messages: readonly BaseMessage[];
  readonly toolNames: readonly string[];
  readonly toolChoice?: HostModelToolChoice;
}

/**
 * Host implementations authenticate identity/reference and authorize every selected tool.
 * They invoke an already configured model through its public invoke/stream APIs, keeping
 * credentials, model parameters, input/output protection and callbacks host-side. Streams are
 * pull-based: transports must preserve backpressure, cancellation and native message metadata.
 */
export interface HostModelService {
  readonly version: 1;
  invoke(
    request: HostModelRequest,
    signal?: AbortSignal,
    context?: Readonly<ModelInvocationContext>
  ): Promise<AIMessage>;
  stream(
    request: HostModelRequest,
    signal?: AbortSignal,
    context?: Readonly<ModelInvocationContext>
  ): AsyncIterable<AIMessageChunk>;
}

export type HostModelErrorCode =
  | 'incompatible'
  | 'forbidden'
  | 'blocked'
  | 'overflow'
  | 'timeout'
  | 'cancelled'
  | 'unavailable';
const errorCodes: ReadonlySet<string> = new Set<HostModelErrorCode>([
  'incompatible',
  'forbidden',
  'blocked',
  'overflow',
  'timeout',
  'cancelled',
  'unavailable',
]);

export class HostModelServiceError extends Error {
  constructor(readonly code: HostModelErrorCode) {
    const safeCode = errorCodes.has(code) ? code : 'incompatible';
    super(`Host model service: ${safeCode}`);
    this.code = safeCode;
    this.name = 'HostModelServiceError';
  }
}

export function normalizeHostModelError(
  error: unknown,
  signal?: AbortSignal
): HostModelServiceError | StreamLimitExceededError | PreparedSubagentError {
  if (
    signal?.aborted === true &&
    (signal.reason instanceof StreamLimitExceededError ||
      signal.reason instanceof PreparedSubagentError)
  )
    return signal.reason;
  if (signal?.aborted === true) return new HostModelServiceError('cancelled');
  try {
    if (
      error != null &&
      typeof error === 'object' &&
      'name' in error &&
      error.name === 'HostModelServiceError' &&
      'code' in error &&
      typeof error.code === 'string' &&
      errorCodes.has(error.code)
    ) {
      return new HostModelServiceError(error.code as HostModelErrorCode);
    }
  } catch {
    /* Error getters cannot disclose host data. */
  }
  return new HostModelServiceError('unavailable');
}

interface HostModelCallOptions extends BaseChatModelCallOptions {
  hostToolNames?: string[];
  tool_choice?:
    | string
    | { type: 'function'; function: { name: string } }
    | { type: 'tool'; name: string };
}

function toolChoice(input: unknown): HostModelToolChoice | undefined {
  if (input === undefined) return undefined;
  const parsed = choiceSchema.safeParse(input);
  if (!parsed.success) throw new HostModelServiceError('incompatible');
  const value = parsed.data;
  if (value === 'auto' || value === 'none') return { mode: value };
  if (value === 'required' || value === 'any') return { mode: 'required' };
  if (typeof value === 'string') return { mode: 'tool', name: value };
  return {
    mode: 'tool',
    name: value.type === 'function' ? value.function.name : value.name,
  };
}

/** Observe late settlement without allowing it to release output after caller cancellation. */
function waitForService<T>(
  operation: () => Promise<T>,
  signal?: AbortSignal
): Promise<T> {
  signal?.throwIfAborted();
  return new Promise<T>((resolve, reject) => {
    const abort = (): void => {
      signal?.removeEventListener('abort', abort);
      reject(normalizeHostModelError(undefined, signal));
    };
    signal?.addEventListener('abort', abort, { once: true });
    try {
      operation().then(
        (value) => {
          signal?.removeEventListener('abort', abort);
          resolve(value);
        },
        (error: Error) => {
          signal?.removeEventListener('abort', abort);
          reject(error);
        }
      );
      if (signal?.aborted === true) abort();
    } catch (error) {
      signal?.removeEventListener('abort', abort);
      reject(error);
    }
  });
}

/** Native LangChain model facade; no provider/model options or credentials cross to the host. */
export class HostChatModel extends BaseChatModel<HostModelCallOptions> {
  static readonly invocationContextVersion = 1;
  private readonly identity: Readonly<ExecutionIdentity>;
  private readonly descriptor: HostModelDescriptor;
  private readonly service: HostModelService;
  private readonly schemas = new Map<string, string>();

  constructor(options: {
    identity: Readonly<ExecutionIdentity>;
    descriptor: HostModelDescriptor;
    service: HostModelService;
  }) {
    super({});
    const parsed = hostModelDescriptorSchema.safeParse(options.descriptor);
    const serviceVersion: number = options.service.version;
    if (
      !parsed.success ||
      serviceVersion !== 1 ||
      typeof options.service.invoke !== 'function' ||
      typeof options.service.stream !== 'function'
    )
      throw new HostModelServiceError('incompatible');
    try {
      this.identity = snapshotExecutionIdentity(options.identity);
      this.descriptor = structuredClone(parsed.data);
    } catch {
      throw new HostModelServiceError('incompatible');
    }
    this.disableStreaming = !this.descriptor.streaming;
    this.service = options.service;
    for (const tool of this.descriptor.tools) {
      const name = tool.function.name;
      if (this.schemas.has(name))
        throw new HostModelServiceError('incompatible');
      this.schemas.set(name, stableStringify(tool));
    }
  }

  _llmType(): string {
    return 'librechat-host';
  }

  override get callKeys(): string[] {
    return [...super.callKeys, 'hostToolNames', 'tool_choice'];
  }

  override getLsParams(): ReturnType<BaseChatModel['getLsParams']> {
    return {
      ls_model_type: 'chat',
      ls_provider: this.descriptor.provider,
      ls_model_name: this.descriptor.model,
    };
  }

  override _identifyingParams(): Pick<
    HostModelDescriptor,
    'provider' | 'model' | 'reference'
    > {
    return {
      model: this.descriptor.model,
      provider: this.descriptor.provider,
      reference: this.descriptor.reference,
    };
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<HostModelCallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, HostModelCallOptions> {
    const names = tools.map((tool) => {
      const definition = convertToOpenAITool(tool);
      const name = definition.function.name;
      if (!name || this.schemas.get(name) !== stableStringify(definition))
        throw new HostModelServiceError('forbidden');
      return name;
    });
    const choice = toolChoice(kwargs?.tool_choice);
    this.validateTools(names, choice);
    return this.withConfig({ ...kwargs, hostToolNames: names });
  }

  private validateTools(
    names: readonly string[],
    choice?: HostModelToolChoice
  ): void {
    if (
      !Array.isArray(names) ||
      new Set(names).size !== names.length ||
      names.some((name) => !this.schemas.has(name))
    )
      throw new HostModelServiceError('forbidden');
    if (choice === undefined) return;
    if (
      !['auto', 'none', 'required', 'tool'].includes(choice.mode) ||
      (choice.mode === 'tool' && !names.includes(choice.name)) ||
      (choice.mode === 'required' && names.length === 0)
    )
      throw new HostModelServiceError('forbidden');
  }

  private request(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): HostModelRequest {
    const names = options.hostToolNames?.slice() ?? [];
    const choice = toolChoice(options.tool_choice);
    this.validateTools(names, choice);
    return {
      version: 1,
      identity: this.identity,
      reference: this.descriptor.reference,
      messages: messages.slice(),
      toolNames: names,
      ...(choice ? { toolChoice: { ...choice } } : {}),
    };
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const request = this.request(messages, options);
    try {
      options.signal?.throwIfAborted();
      const message = await waitForService(
        () =>
          this.service.invoke(
            request,
            options.signal,
            getModelInvocationContext(options, runManager)
          ),
        options.signal
      );
      options.signal?.throwIfAborted();
      if (!isAIMessage(message))
        throw new HostModelServiceError('incompatible');
      return { generations: [{ text: message.text, message }] };
    } catch (error) {
      throw normalizeHostModelError(error, options.signal);
    }
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const request = this.request(messages, options);
    const controller = new AbortController();
    const signal = options.signal
      ? AbortSignal.any([options.signal, controller.signal])
      : controller.signal;
    let iterator: AsyncIterator<AIMessageChunk> | undefined;
    try {
      signal.throwIfAborted();
      const context = getModelInvocationContext(options, runManager);
      iterator = this.service.stream(request, signal, context)[Symbol.asyncIterator]();
      for (;;) {
        const next = await waitForService(() => iterator!.next(), signal);
        signal.throwIfAborted();
        if (next.done === true) break;
        const message = next.value;
        if (!isAIMessageChunk(message))
          throw new HostModelServiceError('incompatible');
        const chunk = new ChatGenerationChunk({
          text: message.text,
          message,
          generationInfo: message.response_metadata,
        });
        yield chunk;
        signal.throwIfAborted();
        await runManager?.handleLLMNewToken(
          chunk.text,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk }
        );
      }
    } catch (error) {
      throw normalizeHostModelError(error, options.signal);
    } finally {
      controller.abort();
      // The host/transport retains execution ownership until this cleanup actually settles.
      try {
        void iterator?.return?.().catch(() => undefined);
      } catch {
        /* Cleanup errors must not disclose host data. */
      }
    }
  }
}

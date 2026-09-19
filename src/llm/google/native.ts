import { v4 } from 'uuid';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type {
  GenerateContentResponse,
  GenerateContentCandidate,
} from '@google/generative-ai';
import type {
  BaseMessage,
  MessageContent,
  UsageMetadata,
} from '@langchain/core/messages';

type NativeResponse = Omit<GenerateContentResponse, 'candidates'> & {
  candidates?: Array<
    Pick<GenerateContentCandidate, 'finishReason'> & {
      content?: Partial<GenerateContentCandidate['content']>;
    }
  >;
};

export type NativeMediaPart =
  | { kind: 'text'; text: string; thoughtSignature?: string }
  | {
      kind: 'image';
      mimeType: string;
      data: string;
      thoughtSignature?: string;
    };
export type NativeMediaReference = { continuationRef: string };
export type NativeMediaRestoreInput = {
  file_id?: string;
  continuationRef?: string;
};
export type NativeMediaProviderOutcome = {
  kind: 'blocked' | 'invalid';
  code: string;
};

/** Provider consumption survives a failed persistence or cancellation outcome. */
export class NativeMediaError extends Error {
  constructor(
    cause: Error,
    readonly usage?: UsageMetadata,
    readonly providerOutcome?: NativeMediaProviderOutcome
  ) {
    super(cause.message, { cause });
    this.name = cause.name;
  }
}
export type NativeMediaContent =
  | { type: 'text'; text: string; native_media?: NativeMediaReference }
  | {
      type: 'image_file';
      image_file: {
        file_id: string;
        filepath: string;
        filename: string;
        type: string;
        bytes: number;
        width?: number;
        height?: number;
      };
      native_media?: NativeMediaReference;
    };
export interface NativeMediaPort {
  /** Authorize the invocation before the provider request and select its modalities. */
  start(input: {
    modelRunId: string;
    model: string;
    signal?: AbortSignal;
  }): Promise<void | { responseModalities: string[] }>;
  /** Persist the original part before returning content safe to stream and serialize. */
  part(input: {
    modelRunId: string;
    chunkIndex: number;
    partIndex: number;
    part: NativeMediaPart;
  }): Promise<NativeMediaContent>;
  complete(input: { modelRunId: string }): Promise<void>;
  fail(input: {
    modelRunId: string;
    reason: 'aborted' | 'provider' | 'storage';
    /** Failure-only usage; successful calls use the normal model-end callback. */
    usage?: UsageMetadata;
    providerOutcome?: NativeMediaProviderOutcome;
  }): Promise<void>;
  /** Authorize and restore the exact signed provider part for a continuation. */
  restore(input: NativeMediaRestoreInput): Promise<NativeMediaPart>;
  /**
   * Restore all references in order, or reject the entire invocation. Hosts must
   * bound database batches and concurrent asset reads using their own limits.
   */
  restoreBatch?(input: {
    parts: readonly NativeMediaRestoreInput[];
    signal?: AbortSignal;
  }): Promise<NativeMediaPart[]>;
}

/** All durable storage and authorization belongs to the injected host port. */
export class NativeMediaSession {
  private finished = false;
  private storageFailure = false;
  private receivedContent = false;
  private usage?: UsageMetadata;
  private providerOutcome?: NativeMediaProviderOutcome;
  private readonly modelRunId: string;
  constructor(
    private readonly port: NativeMediaPort | undefined,
    private readonly model: string,
    modelRunId?: string,
    private readonly signal?: AbortSignal
  ) {
    this.modelRunId = modelRunId ?? v4();
  }
  async start(): Promise<void | { responseModalities: string[] }> {
    return this.port?.start({
      modelRunId: this.modelRunId,
      model: this.model,
      signal: this.signal,
    });
  }
  /** Added once to the usage chunk, so stream aggregation preserves call identity. */
  usageIdentity(): { native_media_model_run_id: string } | undefined {
    return this.port == null
      ? undefined
      : { native_media_model_run_id: this.modelRunId };
  }
  async complete(): Promise<void> {
    if (this.port != null && !this.receivedContent) {
      this.rejectResponse({ kind: 'invalid', code: 'EMPTY_RESPONSE' });
    }
    this.signal?.throwIfAborted();
    try {
      await this.port?.complete({ modelRunId: this.modelRunId });
    } catch (error) {
      this.storageFailure = true;
      throw error;
    }
    this.finished = true;
  }
  observeResponse(response: NativeResponse, usage?: UsageMetadata): void {
    if (usage != null) this.usage = usage;
    if (this.port == null) return;
    const blockReason: string | undefined =
      response.promptFeedback?.blockReason;
    if (blockReason != null && blockReason !== 'BLOCK_REASON_UNSPECIFIED') {
      this.rejectResponse({
        kind: 'blocked',
        code: /^[A-Z_]{1,64}$/.test(blockReason)
          ? blockReason
          : 'BLOCKED_RESPONSE',
      });
    }
    const candidate = response.candidates?.[0];
    const finishReason: string | undefined = candidate?.finishReason;
    if (
      finishReason != null &&
      finishReason !== 'STOP' &&
      finishReason !== 'MAX_TOKENS' &&
      finishReason !== 'FINISH_REASON_UNSPECIFIED'
    ) {
      this.rejectResponse({
        kind: /SAFETY|RECITATION|BLOCKLIST|PROHIBITED|SPII/.test(finishReason)
          ? 'blocked'
          : 'invalid',
        code: /^[A-Z_]{1,64}$/.test(finishReason)
          ? finishReason
          : 'INVALID_RESPONSE',
      });
    }
    this.receivedContent ||= (candidate?.content?.parts?.length ?? 0) > 0;
  }
  private rejectResponse(outcome: NativeMediaProviderOutcome): never {
    this.providerOutcome = outcome;
    throw new Error(`Native media provider ${outcome.kind}: ${outcome.code}`);
  }
  error(cause: unknown): Error {
    const error = cause instanceof Error ? cause : new Error(String(cause));
    return this.usage == null && this.providerOutcome == null
      ? error
      : new NativeMediaError(error, this.usage, this.providerOutcome);
  }
  async reportFailure(cause: unknown): Promise<Error> {
    try {
      await this.fail();
    } catch (recordingError) {
      const error = cause instanceof Error ? cause : new Error(String(cause));
      return this.error(
        new AggregateError([error, recordingError], error.message)
      );
    }
    return this.error(cause);
  }
  async fail(): Promise<void> {
    if (this.finished) return;
    this.finished = true;
    const failure = this.storageFailure ? 'storage' : 'provider';
    await this.port?.fail({
      modelRunId: this.modelRunId,
      reason: this.signal?.aborted === true ? 'aborted' : failure,
      ...(this.usage == null ? {} : { usage: this.usage }),
      ...(this.providerOutcome == null
        ? {}
        : { providerOutcome: this.providerOutcome }),
    });
  }
  async content(
    content: MessageContent,
    chunkIndex: number
  ): Promise<MessageContent> {
    if (!this.port) {
      if (
        Array.isArray(content) &&
        content.some((part) => 'inlineData' in part)
      ) {
        throw new Error('Native media requires configured storage');
      }
      return Array.isArray(content)
        ? content.map((part) => {
          if (part.type !== 'text') return part;
          const { thoughtSignature: _signature, ...visible } = part;
          return visible;
        })
        : content;
    }
    const parts =
      typeof content === 'string' ? [{ type: 'text', text: content }] : content;
    const output: Exclude<MessageContent, string> = [];
    for (let partIndex = 0; partIndex < parts.length; partIndex++) {
      const part = parts[partIndex];
      const thoughtSignature =
        typeof part.thoughtSignature === 'string'
          ? part.thoughtSignature
          : undefined;
      let native: NativeMediaPart | undefined;
      if (part.type === 'text' && typeof part.text === 'string') {
        if (part.text === '' && thoughtSignature == null) continue;
        native = { kind: 'text', text: part.text, thoughtSignature };
      } else if (
        'inlineData' in part &&
        typeof part.inlineData === 'object' &&
        part.inlineData !== null
      ) {
        const inline = part.inlineData;
        if (
          'mimeType' in inline &&
          typeof inline.mimeType === 'string' &&
          'data' in inline &&
          typeof inline.data === 'string'
        ) {
          if (!inline.mimeType.startsWith('image/'))
            throw new Error('Unsupported native media MIME type');
          native = {
            kind: 'image',
            mimeType: inline.mimeType,
            data: inline.data,
            thoughtSignature,
          };
        }
      }
      if ('inlineData' in part && !native)
        throw new Error('Malformed native media content');
      if (!native) {
        output.push(part);
        continue;
      }
      try {
        const stored = await this.port.part({
          modelRunId: this.modelRunId,
          chunkIndex,
          partIndex,
          part: native,
        });
        if (this.signal?.aborted === true)
          throw this.signal.reason ?? new Error('Native media request aborted');
        if (native.kind === 'text' && stored.type === 'text') {
          const { thoughtSignature: _signature, ...visible } = part;
          output.push({ ...visible, ...stored });
        } else if (native.kind === 'image' && stored.type === 'image_file') {
          output.push(stored);
        } else
          throw new Error('Native media port returned incompatible content');
      } catch (error) {
        this.storageFailure = true;
        throw error;
      }
    }
    return typeof content === 'string' &&
      output.length === 1 &&
      output[0].type === 'text' &&
      output[0].native_media == null
      ? String(output[0].text)
      : output;
  }
  async messages(messages: BaseMessage[]): Promise<BaseMessage[]> {
    if (!this.port) {
      if (
        messages.some(
          (message) =>
            Array.isArray(message.content) &&
            message.content.some((part) => part.native_media != null)
        )
      ) {
        throw new Error('Native continuation requires configured storage');
      }
      return messages;
    }
    const result: BaseMessage[] = [];
    const pending: Array<{
      input: NativeMediaRestoreInput;
      content: Exclude<MessageContent, string>;
      index: number;
    }> = [];
    for (const message of messages) {
      if (!Array.isArray(message.content)) {
        result.push(message);
        continue;
      }
      let changed = false;
      const content: Exclude<MessageContent, string> = [];
      for (const part of message.content) {
        const image =
          part.type === 'image_file' &&
          typeof part.image_file === 'object' &&
          part.image_file !== null
            ? part.image_file
            : undefined;
        const file_id =
          image && 'file_id' in image && typeof image.file_id === 'string'
            ? image.file_id
            : undefined;
        const reference =
          typeof part.native_media === 'object' && part.native_media !== null
            ? part.native_media
            : undefined;
        const continuationRef =
          reference &&
          'continuationRef' in reference &&
          typeof reference.continuationRef === 'string'
            ? reference.continuationRef
            : undefined;
        if (continuationRef == null || continuationRef === '') {
          content.push(part);
          continue;
        }
        changed = true;
        pending.push({
          input: { file_id, continuationRef },
          content,
          index: content.length,
        });
        content.push(part);
      }
      if (!changed) result.push(message);
      else if (message._getType() === 'ai')
        result.push(new AIMessage({ ...message, content }));
      else if (message._getType() === 'human')
        result.push(new HumanMessage({ ...message, content }));
      else
        throw new Error('Native media reference in unsupported message role');
    }
    if (pending.length === 0) return result;
    this.signal?.throwIfAborted();
    const restored =
      this.port.restoreBatch == null
        ? await this.restoreLegacy(pending.map(({ input }) => input))
        : await this.port.restoreBatch({
          parts: pending.map(({ input }) => input),
          signal: this.signal,
        });
    this.signal?.throwIfAborted();
    if (restored.length !== pending.length) {
      throw new Error(
        'Native media port returned incomplete continuation batch'
      );
    }
    for (let index = 0; index < pending.length; index++) {
      const target = pending[index];
      const part = restored.at(index);
      const visible = target.content[target.index];
      if (part?.kind === 'text' && visible.type === 'text') {
        const {
          native_media: _reference,
          thoughtSignature: _signature,
          ...text
        } = visible;
        target.content[target.index] =
          part.text === visible.text
            ? { ...text, thoughtSignature: part.thoughtSignature }
            : text;
      } else if (part?.kind === 'image' && visible.type === 'image_file') {
        target.content[target.index] = {
          type: 'image_url',
          image_url: { url: `data:${part.mimeType};base64,${part.data}` },
          thoughtSignature: part.thoughtSignature,
        };
      } else {
        throw new Error('Native media port returned incompatible continuation');
      }
    }
    return result;
  }
  private async restoreLegacy(
    parts: NativeMediaRestoreInput[]
  ): Promise<NativeMediaPart[]> {
    const result: NativeMediaPart[] = [];
    for (const part of parts) {
      this.signal?.throwIfAborted();
      result.push(await this.port!.restore(part));
    }
    return result;
  }
}

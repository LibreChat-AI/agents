import { v4 } from 'uuid';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, MessageContent } from '@langchain/core/messages';

export type NativeMediaPart =
  | { kind: 'text'; text: string; thoughtSignature?: string }
  | {
      kind: 'image';
      mimeType: string;
      data: string;
      thoughtSignature?: string;
    };
export type NativeMediaReference = { continuationRef: string };
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
  }): Promise<void>;
  /** Authorize and restore the exact signed provider part for a continuation. */
  restore(input: {
    file_id?: string;
    continuationRef?: string;
  }): Promise<NativeMediaPart>;
}

/** All durable storage and authorization belongs to the injected host port. */
export class NativeMediaSession {
  private finished = false;
  private storageFailure = false;
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
  async complete(): Promise<void> {
    await this.port?.complete({ modelRunId: this.modelRunId });
    this.finished = true;
  }
  async fail(): Promise<void> {
    if (this.finished) return;
    this.finished = true;
    const failure = this.storageFailure ? 'storage' : 'provider';
    await this.port?.fail({
      modelRunId: this.modelRunId,
      reason: this.signal?.aborted === true ? 'aborted' : failure,
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
        ? content.map(({ thoughtSignature: _signature, ...part }) => part)
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
        const restored = await this.port.restore({ file_id, continuationRef });
        changed = true;
        if (restored.kind === 'text')
          content.push({
            type: 'text',
            text: restored.text,
            thoughtSignature: restored.thoughtSignature,
          });
        else
          content.push({
            type: 'image_url',
            image_url: {
              url: `data:${restored.mimeType};base64,${restored.data}`,
            },
            thoughtSignature: restored.thoughtSignature,
          });
      }
      if (!changed) result.push(message);
      else if (message._getType() === 'ai')
        result.push(new AIMessage({ ...message, content }));
      else if (message._getType() === 'human')
        result.push(new HumanMessage({ ...message, content }));
      else
        throw new Error('Native media reference in unsupported message role');
    }
    return result;
  }
}

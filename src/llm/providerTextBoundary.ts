import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { RunnableBinding, RunnableSequence } from '@langchain/core/runnables';
import type { MessageContentComplex } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import type { ProviderTextProtection } from '@/protection/providerText';
import type { ChatModel } from '@/types';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
} from '@/protection/providerText';

const additionalKeys = new Set([
  'tool_calls',
  'function_call',
  'reasoning',
  'reasoning_content',
  'reasoning_details',
  'thinking',
  'signature',
]);
const metadataKeys = new Set([
  'finish_reason', 'stop_reason', 'stop_sequence', 'model_name', 'model', 'model_provider',
  'system_fingerprint', 'service_tier', 'usage', 'tokenUsage', 'input_tokens', 'output_tokens',
  'total_tokens', 'index', 'prompt', 'completion', 'output_version',
]);
const controlBlocks = new Set([
  'thinking',
  'redacted_thinking',
  'reasoning',
  'reasoning_content',
  'tool_use',
]);

type TextBlock = MessageContentComplex & {
  type: 'text';
  text: string;
  index?: number;
};

function copyMessage(
  message: AIMessageChunk,
  content: AIMessageChunk['content']
): AIMessageChunk {
  return new AIMessageChunk({
    content,
    id: message.id,
    name: message.name,
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
    tool_calls: message.tool_calls,
    tool_call_chunks: message.tool_call_chunks,
    invalid_tool_calls: message.invalid_tool_calls,
    usage_metadata: message.usage_metadata,
  });
}

function numericMetadata(value: unknown, depth = 4): boolean {
  if (value == null) return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || Array.isArray(value) || depth === 0) return false;
  const values = Object.values(value);
  return values.length <= 64 && values.every((entry) => numericMetadata(entry, depth - 1));
}

class Candidate {
  private block?: TextBlock;
  private stringContent = false;

  constructor(private readonly attempt: ProviderTextAttempt) {}

  strip(generation: ChatGenerationChunk): ChatGenerationChunk {
    const message = generation.message;
    if (!(message instanceof AIMessageChunk))
      throw new ProviderTextProtectionError('unsupported');
    for (const key of Object.keys(message.additional_kwargs)) {
      if (!additionalKeys.has(key))
        throw new ProviderTextProtectionError('unsupported');
    }
    for (const metadata of [message.response_metadata, generation.generationInfo ?? {}]) {
      if (Object.keys(metadata).some((key) => !metadataKeys.has(key))) {
        throw new ProviderTextProtectionError('unsupported');
      }
    }
    for (const metadata of [message.response_metadata, generation.generationInfo ?? {}]) {
      for (const key of ['usage', 'tokenUsage']) {
        const value = metadata[key];
        if (value != null && !numericMetadata(value)) {
          throw new ProviderTextProtectionError('unsupported');
        }
      }
    }
    let content: AIMessageChunk['content'];
    if (typeof message.content === 'string') {
      if (message.content && this.block != null)
        throw new ProviderTextProtectionError('unsupported');
      this.stringContent ||= message.content.length > 0;
      this.attempt.append(message.content);
      content = '';
    } else {
      content = [];
      for (const block of message.content) {
        if (typeof block === 'string')
          throw new ProviderTextProtectionError('unsupported');
        if (block.type !== 'text') {
          if (!controlBlocks.has(block.type))
            throw new ProviderTextProtectionError('unsupported');
          content.push(block);
          continue;
        }
        if (
          typeof block.text !== 'string' ||
          this.stringContent ||
          Object.keys(block).some(
            (key) => !['type', 'text', 'index'].includes(key)
          )
        ) {
          throw new ProviderTextProtectionError('unsupported');
        }
        if (this.block != null && this.block.index !== block.index) {
          throw new ProviderTextProtectionError('unsupported');
        }
        this.block ??= { ...block, text: '' } as TextBlock;
        this.attempt.append(block.text);
      }
    }
    return new ChatGenerationChunk({
      text: '',
      message: copyMessage(message, content),
      generationInfo: generation.generationInfo,
    });
  }

  async canonical(): Promise<ChatGenerationChunk | undefined> {
    const content = await this.attempt.release();
    if (!content) return undefined;
    return new ChatGenerationChunk({
      text: content,
      message: new AIMessageChunk({
        content:
          this.block == null ? content : [{ ...this.block, text: content }],
      }),
    });
  }
}

function clone<T extends object>(value: T): T {
  return Object.create(
    Object.getPrototypeOf(value),
    Object.getOwnPropertyDescriptors(value)
  ) as T;
}

/** Clone only known runnable shells. Never mutate a shared provider or its callback configuration. */
export function withProviderTextBoundary(
  model: ChatModel,
  policy: ProviderTextProtection
): ChatModel {
  const protectedModel = clone(model);
  if (model instanceof RunnableBinding) {
    Object.defineProperty(protectedModel, 'bound', {
      value: withProviderTextBoundary(model.bound as ChatModel, policy),
    });
    return protectedModel;
  }
  if (model instanceof RunnableSequence) {
    const steps = model.steps;
    Object.defineProperty(protectedModel, 'last', {
      value: withProviderTextBoundary(
        steps[steps.length - 1] as ChatModel,
        policy
      ),
    });
    return protectedModel;
  }
  if (!(model instanceof BaseChatModel) || model.cache != null) {
    throw new ProviderTextProtectionError('unsupported');
  }
  const protectedChat = clone(model);
  protectedChat._streamResponseChunks = async function* (
    messages,
    options,
    runManager
  ): AsyncGenerator<ChatGenerationChunk> {
    const attempt = new ProviderTextAttempt(policy, options.signal);
    const candidate = new Candidate(attempt);
    const source = model._streamResponseChunks(
      messages,
      { ...options, signal: attempt.signal },
      undefined
    );
    try {
      for (;;) {
        const next = await attempt.wait(source.next());
        attempt.check();
        if (next.done === true) break;
        const chunk = candidate.strip(next.value);
        yield chunk;
        await runManager?.handleLLMNewToken(
          '',
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk }
        );
      }
      const canonical = await candidate.canonical();
      attempt.check();
      if (canonical != null) {
        yield canonical;
        attempt.check();
        await runManager?.handleLLMNewToken(
          canonical.text,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: canonical }
        );
      }
    } finally {
      const closing = source.return(undefined);
      void attempt.wait(closing).catch(() => {});
      attempt.finish();
    }
  };
  protectedChat._streamChatModelEvents =
    BaseChatModel.prototype._streamChatModelEvents;
  protectedChat._generate = async function (messages, options): Promise<ChatResult> {
    const attempt = new ProviderTextAttempt(policy, options.signal);
    try {
      const result = await attempt.wait(
        model._generate(
          messages,
          { ...options, signal: attempt.signal },
          undefined
        )
      );
      attempt.check();
      if (result.llmOutput != null && !numericMetadata(result.llmOutput)) {
        throw new ProviderTextProtectionError('unsupported');
      }
      if (result.generations.length !== 1)
        throw new ProviderTextProtectionError('unsupported');
      const generation = result.generations[0];
      const message = generation.message instanceof AIMessageChunk ? generation.message : new AIMessageChunk(generation.message);
      const candidate = new Candidate(attempt);
      const stripped = candidate.strip(
        new ChatGenerationChunk({ ...generation, message })
      );
      const canonical = await candidate.canonical();
      attempt.check();
      return {
        ...result,
        generations: [
          canonical == null ? stripped : stripped.concat(canonical),
        ],
      };
    } finally {
      attempt.finish();
    }
  };
  return protectedChat;
}

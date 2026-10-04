import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { RunnableBinding, RunnableSequence } from '@langchain/core/runnables';
import type { MessageContentComplex } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import type { ProviderTextProtection } from '@/protection/providerText';
import type { ChatModel } from '@/types';
import {
  getStreamedToolCallAdapter,
  getStreamedToolCallSeal,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
} from '@/tools/streamedToolCallSeals';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
} from '@/protection/providerText';
import { isProviderTextInput } from '@/protection/providerTextInput';
import { getChatModelClass } from '@/llm/providers';
import { Providers } from '@/common';

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
const additionalControls: Readonly<Partial<Record<string, (value: unknown) => boolean>>> = {
  id: (value) => typeof value === 'string',
  type: (value) => value === 'message',
  role: (value) => value === 'assistant',
  model: (value) => typeof value === 'string',
  stop_reason: (value) => value == null || typeof value === 'string',
  stop_sequence: (value) => value == null || typeof value === 'string',
  usage: (value) => numericMetadata(value),
};

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

function hasKeys(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).every((key) => keys.includes(key));
}

function validateMetadata(metadata: AIMessageChunk['response_metadata']): void {
  for (const key of Object.keys(metadata)) {
    const value: unknown = metadata[key];
    let valid: boolean;
    if (additionalControls[key] != null) {
      valid = additionalControls[key](value);
    } else if (key === 'contentBlockIndex') {
      valid = typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
    } else if (key === STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY) {
      valid = getStreamedToolCallAdapter(metadata) != null;
    } else if (key === STREAMED_TOOL_CALL_SEAL_METADATA_KEY) {
      valid = hasKeys(value, ['kind', 'id', 'index']) && getStreamedToolCallSeal(metadata) != null;
    } else if (key === 'messageStart') {
      valid = hasKeys(value, ['role']) && value.role === 'assistant';
    } else if (key === 'messageStop') {
      valid = hasKeys(value, ['stopReason']) && typeof value.stopReason === 'string';
    } else if (key === 'metadata') {
      valid = hasKeys(value, ['usage', 'metrics']) && numericMetadata(value);
    } else if (key === 'usage' || key === 'tokenUsage' || key === 'estimatedTokenUsage') {
      valid = numericMetadata(value);
    } else {
      valid = metadataKeys.has(key) && (value == null || typeof value === 'string' ||
        (typeof value === 'number' && Number.isFinite(value)));
    }
    if (!valid) throw new ProviderTextProtectionError('unsupported');
  }
}

class Candidate {
  private block?: TextBlock;
  private stringContent = false;

  constructor(private readonly attempt: ProviderTextAttempt) {}

  strip(generation: ChatGenerationChunk): ChatGenerationChunk {
    const message = generation.message;
    if (!(message instanceof AIMessageChunk))
      throw new ProviderTextProtectionError('unsupported');
    this.attempt.observeChunk();
    for (const key of Object.keys(message.additional_kwargs)) {
      if (additionalKeys.has(key)) continue;
      const validate = additionalControls[key];
      if (validate == null || !validate(message.additional_kwargs[key])) {
        throw new ProviderTextProtectionError('unsupported');
      }
    }
    for (const metadata of [message.response_metadata, generation.generationInfo ?? {}]) {
      validateMetadata(metadata);
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

function assertBindingShell(model: RunnableBinding<never, unknown>): void {
  if (model.constructor !== RunnableBinding || (model.configFactories?.length ?? 0) > 0 ||
      model.invoke !== RunnableBinding.prototype.invoke || model.batch !== RunnableBinding.prototype.batch ||
      model.stream !== RunnableBinding.prototype.stream || model.transform !== RunnableBinding.prototype.transform ||
      model._streamIterator !== RunnableBinding.prototype._streamIterator) {
    throw new ProviderTextProtectionError('unsupported');
  }
}

function assertInputStep(input: object): void {
  if (isProviderTextInput(input)) return;
  if (input instanceof RunnableBinding) {
    assertBindingShell(input);
    assertInputStep(input.bound);
    return;
  }
  throw new ProviderTextProtectionError('unsupported');
}

function usesInternalStreaming(model: BaseChatModel): boolean {
  if ('streaming' in model && model.streaming === true) return true;
  for (const key of ['completions', 'responses'] as const) {
    const delegate: unknown = Reflect.get(model, key);
    if (delegate instanceof BaseChatModel && 'streaming' in delegate && delegate.streaming === true) return true;
  }
  return false;
}

/** Clone only known runnable shells. Never mutate a shared provider or its callback configuration. */
export function withProviderTextBoundary(
  model: ChatModel,
  policy: ProviderTextProtection
): ChatModel {
  const protectedModel = clone(model);
  if (model instanceof RunnableBinding) {
    assertBindingShell(model);
    Object.defineProperty(protectedModel, 'bound', {
      value: withProviderTextBoundary(model.bound as ChatModel, policy),
    });
    return protectedModel;
  }
  if (model instanceof RunnableSequence) {
    if (model.constructor !== RunnableSequence ||
        model.invoke !== RunnableSequence.prototype.invoke || model.batch !== RunnableSequence.prototype.batch ||
        model.stream !== RunnableSequence.prototype.stream || model.transform !== RunnableSequence.prototype.transform ||
        model._streamIterator !== RunnableSequence.prototype._streamIterator) {
      throw new ProviderTextProtectionError('unsupported');
    }
    const steps = model.steps;
    for (let index = 0; index < steps.length - 1; index++) assertInputStep(steps[index]);
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
  if (model._streamIterator !== BaseChatModel.prototype._streamIterator ||
      model.generate !== BaseChatModel.prototype.generate ||
      model.generatePrompt !== BaseChatModel.prototype.generatePrompt ||
      model._generateUncached !== BaseChatModel.prototype._generateUncached) {
    throw new ProviderTextProtectionError('unsupported');
  }
  if (model.stream !== BaseChatModel.prototype.stream || model.invoke !== BaseChatModel.prototype.invoke) {
    const openAI = getChatModelClass(Providers.OPENAI);
    if (model.stream !== openAI.prototype.stream || model.invoke !== openAI.prototype.invoke) {
      throw new ProviderTextProtectionError('unsupported');
    }
  }
  if (model.disableStreaming && usesInternalStreaming(model)) {
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
    let source: AsyncGenerator<ChatGenerationChunk> | undefined;
    try {
      source = model._streamResponseChunks(
        messages,
        { ...options, signal: attempt.signal },
        undefined
      );
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
      try {
        if (source != null) void attempt.wait(source.return(undefined)).catch(() => {});
      } finally {
        attempt.finish();
      }
    }
  };
  protectedChat._streamChatModelEvents =
    BaseChatModel.prototype._streamChatModelEvents;
  protectedChat._generate = async function (messages, options): Promise<ChatResult> {
    if (usesInternalStreaming(model)) throw new ProviderTextProtectionError('unsupported');
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
      if (result.llmOutput != null) validateMetadata(result.llmOutput);
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

import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents } from '@/common';

export interface OpenAICompatibleWriter {
  write(data: string): void | Promise<void>;
}

export interface OpenAIResponseContext {
  requestId: string;
  model: string;
  created: number;
}

export interface OpenAICompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAIChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: 'assistant';
    content?: string | null;
    reasoning?: string | null;
    tool_calls?: Array<{
      index: number;
      id?: string;
      type?: 'function';
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
}

export interface OpenAIChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: OpenAIChatCompletionChunkChoice[];
  usage?: OpenAICompletionUsage;
}

export interface OpenAIStreamTracker {
  hasRole: boolean;
  hasText: boolean;
  hasReasoning: boolean;
  lastChunkKind?: 'text' | 'reasoning' | 'tool_call';
  toolCalls: Map<number, OpenAIToolCall>;
  toolCallsByStep?: Map<string, Map<number, OpenAIToolCall>>;
  usage: {
    promptTokens: number;
    completionTokens: number;
    reasoningTokens: number;
  };
}

export interface OpenAIHandlerConfig {
  writer: OpenAICompatibleWriter;
  context: OpenAIResponseContext;
  tracker: OpenAIStreamTracker;
}

interface OpenAIToolCallFragment {
  index?: number;
  id?: string;
  name?: string;
  args?: string | object;
  function?: {
    name?: string;
    arguments?: string | object;
  };
}

export function createOpenAIStreamTracker(): OpenAIStreamTracker {
  return {
    hasRole: false,
    hasText: false,
    hasReasoning: false,
    toolCalls: new Map(),
    toolCallsByStep: new Map(),
    usage: {
      promptTokens: 0,
      completionTokens: 0,
      reasoningTokens: 0,
    },
  };
}

function getTokenCount(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function getReasoningTokenCount(usage: Partial<UsageMetadata>): number {
  return getTokenCount(usage.output_token_details?.reasoning);
}

function getToolCallIndex(
  toolCall: OpenAIToolCallFragment,
  fallbackIndex: number
): number {
  return typeof toolCall.index === 'number' ? toolCall.index : fallbackIndex;
}

function getToolCallName(toolCall: OpenAIToolCallFragment): string {
  return toolCall.name ?? toolCall.function?.name ?? '';
}

function getToolCallArguments(toolCall: OpenAIToolCallFragment): string {
  const args = toolCall.args ?? toolCall.function?.arguments;
  if (args == null) {
    return '';
  }
  if (typeof args === 'string') {
    return args;
  }
  return JSON.stringify(args);
}

function getStepToolCalls(
  tracker: OpenAIStreamTracker,
  stepId: string
): Map<number, OpenAIToolCall> {
  const toolCallsByStep = tracker.toolCallsByStep ?? new Map();
  tracker.toolCallsByStep = toolCallsByStep;
  const existing = toolCallsByStep.get(stepId);
  if (existing != null) {
    return existing;
  }
  const stepToolCalls = new Map<number, OpenAIToolCall>();
  toolCallsByStep.set(stepId, stepToolCalls);
  return stepToolCalls;
}

export function createChatCompletionChunk(
  context: OpenAIResponseContext,
  delta: OpenAIChatCompletionChunkChoice['delta'],
  finishReason: OpenAIChatCompletionChunkChoice['finish_reason'] = null,
  usage?: OpenAICompletionUsage
): OpenAIChatCompletionChunk {
  return {
    id: context.requestId,
    object: 'chat.completion.chunk',
    created: context.created,
    model: context.model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage ? { usage } : {}),
  };
}

export function createChatCompletionUsageChunk(
  context: OpenAIResponseContext,
  usage: OpenAICompletionUsage
): OpenAIChatCompletionChunk {
  return {
    id: context.requestId,
    object: 'chat.completion.chunk',
    created: context.created,
    model: context.model,
    choices: [],
    usage,
  };
}

export async function writeOpenAISSE(
  writer: OpenAICompatibleWriter,
  data: OpenAIChatCompletionChunk | '[DONE]'
): Promise<void> {
  await writer.write(
    `data: ${data === '[DONE]' ? data : JSON.stringify(data)}\n\n`
  );
}

function getTextParts(
  data: t.MessageDeltaEvent | t.ReasoningDeltaEvent
): string[] {
  const parts = data.delta.content ?? [];
  const text: string[] = [];
  for (const part of parts) {
    if ('text' in part && typeof part.text === 'string') {
      text.push(part.text);
      continue;
    }
    if ('think' in part && typeof part.think === 'string') {
      text.push(part.think);
    }
  }
  return text;
}

async function emitAssistantRoleChunk(
  config: OpenAIHandlerConfig
): Promise<void> {
  if (config.tracker.hasRole) {
    return;
  }
  config.tracker.hasRole = true;
  await writeOpenAISSE(
    config.writer,
    createChatCompletionChunk(config.context, { role: 'assistant' })
  );
}

async function emitToolCallChunk(params: {
  config: OpenAIHandlerConfig;
  stepId: string;
  toolCall: OpenAIToolCallFragment;
  fallbackIndex: number;
  completed: boolean;
}): Promise<void> {
  const { config, stepId, toolCall, fallbackIndex, completed } = params;
  const index = getToolCallIndex(toolCall, fallbackIndex);
  const stepToolCalls = getStepToolCalls(config.tracker, stepId);
  const existing = stepToolCalls.get(index);
  const current = existing ?? {
    id: toolCall.id ?? '',
    type: 'function' as const,
    function: { name: '', arguments: '' },
  };
  const name = getToolCallName(toolCall);
  const args = getToolCallArguments(toolCall);
  const idChanged =
    toolCall.id != null && toolCall.id !== '' && current.id !== toolCall.id;
  const nameChanged = name !== '' && current.function.name !== name;
  let argumentDelta = '';
  if (completed) {
    if (args !== '' && args !== current.function.arguments) {
      argumentDelta = args.startsWith(current.function.arguments)
        ? args.slice(current.function.arguments.length)
        : args;
      current.function.arguments = args;
    }
  } else if (args !== '') {
    argumentDelta = args;
    current.function.arguments += args;
  }
  if (toolCall.id != null && toolCall.id !== '') {
    current.id = toolCall.id;
  }
  if (name !== '') {
    current.function.name = name;
  }
  stepToolCalls.set(index, current);
  config.tracker.toolCalls.set(index, current);
  if (!idChanged && !nameChanged && argumentDelta === '' && existing) {
    return;
  }
  config.tracker.lastChunkKind = 'tool_call';
  const functionDelta: { name?: string; arguments?: string } = {};
  if (nameChanged || (!existing && current.function.name !== '')) {
    functionDelta.name = current.function.name;
  }
  if (argumentDelta !== '') {
    functionDelta.arguments = argumentDelta;
  }
  await emitAssistantRoleChunk(config);
  await writeOpenAISSE(
    config.writer,
    createChatCompletionChunk(config.context, {
      tool_calls: [
        {
          index,
          ...(current.id !== '' ? { id: current.id } : {}),
          type: 'function',
          ...(functionDelta.name != null || functionDelta.arguments != null
            ? { function: functionDelta }
            : {}),
        },
      ],
    })
  );
}

export function createOpenAIHandlers(
  config: OpenAIHandlerConfig
): Record<string, t.EventHandler> {
  return {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        for (const text of getTextParts(data as t.MessageDeltaEvent)) {
          config.tracker.hasText = true;
          config.tracker.lastChunkKind = 'text';
          await emitAssistantRoleChunk(config);
          await writeOpenAISSE(
            config.writer,
            createChatCompletionChunk(config.context, { content: text })
          );
        }
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        for (const text of getTextParts(data as t.ReasoningDeltaEvent)) {
          config.tracker.hasReasoning = true;
          config.tracker.lastChunkKind = 'reasoning';
          await emitAssistantRoleChunk(config);
          await writeOpenAISSE(
            config.writer,
            createChatCompletionChunk(config.context, { reasoning: text })
          );
        }
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        const runStepDelta = data as t.RunStepDeltaEvent;
        const delta = runStepDelta.delta;
        if (delta.type !== 'tool_calls') {
          return;
        }
        const toolCalls = delta.tool_calls ?? [];
        for (let index = 0; index < toolCalls.length; index++) {
          await emitToolCallChunk({
            config,
            stepId: runStepDelta.id,
            toolCall: toolCalls[index],
            fallbackIndex: index,
            completed: false,
          });
        }
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: async (_event, data): Promise<void> => {
        const runStep = data as t.RunStep;
        if (runStep.stepDetails.type !== 'tool_calls') {
          return;
        }
        const toolCalls = runStep.stepDetails.tool_calls ?? [];
        for (let index = 0; index < toolCalls.length; index++) {
          await emitToolCallChunk({
            config,
            stepId: runStep.id,
            toolCall: toolCalls[index],
            fallbackIndex: index,
            completed: true,
          });
        }
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event, data): void => {
        const usage = (data as t.ModelEndData)?.output?.usage_metadata as
          | Partial<UsageMetadata>
          | undefined;
        if (!usage) {
          return;
        }
        config.tracker.usage.promptTokens += getTokenCount(usage.input_tokens);
        config.tracker.usage.completionTokens += getTokenCount(
          usage.output_tokens
        );
        config.tracker.usage.reasoningTokens += getReasoningTokenCount(usage);
      },
    },
  };
}

export async function sendOpenAIFinalChunk(
  config: OpenAIHandlerConfig,
  finishReason?: OpenAIChatCompletionChunkChoice['finish_reason']
): Promise<void> {
  const resolvedFinishReason =
    finishReason ??
    (config.tracker.lastChunkKind === 'tool_call' ? 'tool_calls' : 'stop');
  const usage: OpenAICompletionUsage = {
    prompt_tokens: config.tracker.usage.promptTokens,
    completion_tokens: config.tracker.usage.completionTokens,
    total_tokens:
      config.tracker.usage.promptTokens + config.tracker.usage.completionTokens,
  };
  if (config.tracker.usage.reasoningTokens > 0) {
    usage.completion_tokens_details = {
      reasoning_tokens: config.tracker.usage.reasoningTokens,
    };
  }
  await emitAssistantRoleChunk(config);
  await writeOpenAISSE(
    config.writer,
    createChatCompletionChunk(config.context, {}, resolvedFinishReason)
  );
  await writeOpenAISSE(
    config.writer,
    createChatCompletionUsageChunk(config.context, usage)
  );
  await writeOpenAISSE(config.writer, '[DONE]');
}

export { createOpenAIToolCallStream } from './toolProjection';
export type {
  OpenAIToolCallStream,
  OpenAIToolCallStreamConfig,
  OpenAIRunStep,
  OpenAIRunStepDelta,
} from './toolProjection';

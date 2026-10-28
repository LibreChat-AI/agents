/* eslint-disable no-process-env */
import { config } from 'dotenv';
config();
import { expect, test } from '@jest/globals';
import * as fs from 'fs/promises';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { ChatPromptValue } from '@langchain/core/prompt_values';
import {
  PromptTemplate,
  ChatPromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { CallbackManager } from '@langchain/core/callbacks/manager';
import { concat } from '@langchain/core/utils/stream';
import type Anthropic from '@anthropic-ai/sdk';
import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import type {
  BaseMessage,
  ContentBlock,
  MessageContentComplex,
} from '@langchain/core/messages';
import { toLangChainContent } from '@/messages/langchain';
import { formatAgentMessages } from '@/messages/format';
import { Constants, ContentTypes, GraphEvents, Providers } from '@/common';
import { _documentsInParams, CustomAnthropic as ChatAnthropic } from './index';
import { partitionAndMarkAnthropicToolCache } from '@/messages/anthropicToolCache';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ModelEndHandler, ToolEndHandler } from '@/events';
import { Run } from '@/run';
import type { CustomAnthropicCallOptions } from './index';
import type {
  AnthropicContextManagementConfigParam,
  AnthropicMessageCreateParams,
  AnthropicMessageStreamEvent,
  AnthropicMessageResponse,
  AnthropicOutputConfig,
  AnthropicRequestOptions,
  AnthropicStreamingMessageCreateParams,
  AnthropicThinkingConfigParam,
  ChatAnthropicContentBlock,
} from './types';
import type {
  AnthropicClientOptions,
  IState,
  MessageContentComplex as LibreChatContentBlock,
  MessageDeltaEvent,
  ReasoningDeltaEvent,
  RunConfig,
  RunStep,
  RunStepDeltaEvent,
  SharedLLMConfig,
  StreamEventData,
  ToolEndEvent,
  TPayload,
} from '@/types';
import {
  _convertMessagesToAnthropicPayload,
  modelDisallowsAssistantPrefill,
  stripUnsupportedAssistantPrefill,
} from './utils/message_inputs';
import {
  _makeMessageChunkFromAnthropicEvent,
  getAnthropicUsageMetadata,
} from './utils/message_outputs';
jest.setTimeout(120000);

async function invoke(
  chat: ChatAnthropic,
  invocationType: string,
  input: BaseLanguageModelInput
): Promise<AIMessageChunk | AIMessage> {
  if (invocationType === 'stream') {
    let output: AIMessageChunk | undefined;

    const stream = await chat.stream(input);

    for await (const chunk of stream) {
      if (!output) {
        output = chunk;
      } else {
        output = output.concat(chunk);
      }
    }

    return output!;
  }

  return chat.invoke(input);
}

// use this for tests involving "extended thinking"
const extendedThinkingModelName = 'claude-sonnet-4-5-20250929';

// use this for tests involving citations
const citationsModelName = 'claude-sonnet-4-5-20250929';

// use this for tests involving PDF documents
const pdfModelName = 'claude-haiku-4-5-20251001';

const remoteImageUrl =
  'https://raw.githubusercontent.com/langchain-ai/langchainjs/main/libs/providers/langchain-google-genai/src/tests/data/hotdog.jpg';

// Use this model for all other tests
const modelName = 'claude-haiku-4-5-20251001';
const webSearchModelName =
  process.env.ANTHROPIC_WEB_SEARCH_MODEL ?? 'claude-opus-4-7';

type AnthropicThinkingResponseBlock = Anthropic.Messages.ThinkingBlock & {
  index?: number;
};

type AnthropicRedactedThinkingResponseBlock =
  Anthropic.Messages.RedactedThinkingBlock & {
    index?: number;
  };

type AnthropicOutputConfigWithTaskBudget = AnthropicOutputConfig & {
  task_budget: {
    type: 'tokens';
    total: number;
  };
};

type LangChainErrorWithCode = {
  lc_error_code?: string;
};

type CitationContentBlock = ContentBlock & {
  citations: Array<{
    type?: string;
    source?: unknown;
  }>;
};

type CompactionContentBlock = ContentBlock & {
  type: 'compaction';
  content: string;
};

type AnthropicContentBlockWithId = ContentBlock & {
  id?: unknown;
  input?: unknown;
  name?: unknown;
};

function getLangChainErrorCode(error: unknown): string | undefined {
  if (typeof error !== 'object' || error == null) {
    return undefined;
  }

  if (!('lc_error_code' in error)) {
    return undefined;
  }

  const { lc_error_code } = error as LangChainErrorWithCode;
  return typeof lc_error_code === 'string' ? lc_error_code : undefined;
}

function expectContentArray<T>(content: string | T[]): T[] {
  expect(Array.isArray(content)).toBe(true);
  if (!Array.isArray(content)) {
    throw new Error('Expected array content');
  }
  return content;
}

function expectDefined<T>(value: T | undefined): T {
  expect(value).toBeDefined();
  if (value === undefined) {
    throw new Error('Expected defined value');
  }
  return value;
}

function isThinkingBlock(
  block: unknown
): block is AnthropicThinkingResponseBlock {
  return (
    typeof block === 'object' &&
    block !== null &&
    'type' in block &&
    block.type === 'thinking'
  );
}

function isRedactedThinkingBlock(
  block: unknown
): block is AnthropicRedactedThinkingResponseBlock {
  return (
    typeof block === 'object' &&
    block !== null &&
    'type' in block &&
    block.type === 'redacted_thinking'
  );
}

function isReasoningContentBlock(
  block: ContentBlock
): block is ContentBlock.Reasoning {
  return block.type === 'reasoning';
}

function isTextContentBlock(block: ContentBlock): block is ContentBlock.Text {
  return block.type === 'text';
}

function hasCitations(block: ContentBlock): block is CitationContentBlock {
  if (!('citations' in block)) {
    return false;
  }

  const { citations } = block as { citations?: unknown };
  return Array.isArray(citations) && citations.length > 0;
}

function isCompactionBlock(
  block: ContentBlock
): block is CompactionContentBlock {
  if (block.type !== 'compaction') {
    return false;
  }

  const { content } = block as { content?: unknown };
  return typeof content === 'string';
}

function isServerToolUseBlock(
  block: ContentBlock
): block is AnthropicContentBlockWithId {
  return (
    block.type === 'server_tool_use' &&
    typeof (block as AnthropicContentBlockWithId).id === 'string' &&
    ((block as AnthropicContentBlockWithId).id as string).startsWith(
      Constants.ANTHROPIC_SERVER_TOOL_PREFIX
    )
  );
}

function expectAnthropicPayloadContentIsNonEmpty(
  payload: AnthropicMessageCreateParams
): void {
  for (const message of payload.messages) {
    if (typeof message.content === 'string') {
      expect(message.content.trim().length).toBeGreaterThan(0);
      continue;
    }

    expect(message.content.length).toBeGreaterThan(0);
    for (const block of message.content) {
      if (block.type !== 'text') {
        continue;
      }
      expect(block.text.trim().length).toBeGreaterThan(0);
    }
  }
}

function expectNoDanglingServerToolUses(
  payload: AnthropicMessageCreateParams
): void {
  for (const message of payload.messages) {
    if (typeof message.content === 'string') {
      continue;
    }

    const serverToolResultIds = new Set(
      message.content
        .map((block) =>
          'tool_use_id' in block &&
          typeof block.tool_use_id === 'string' &&
          block.tool_use_id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
            ? block.tool_use_id
            : undefined
        )
        .filter((id): id is string => id != null)
    );

    for (const block of message.content) {
      if (block.type !== 'server_tool_use') {
        continue;
      }
      expect(serverToolResultIds.has(block.id)).toBe(true);
    }
  }
}

function getPromptCachedWebSearchTools(): Parameters<
  ChatAnthropic['bindTools']
>[0] {
  const tools = partitionAndMarkAnthropicToolCache(
    [
      {
        type: 'web_search_20250305',
        name: 'web_search',
        max_uses: 3,
      },
    ] as never,
    () => false
  );
  return tools as Parameters<ChatAnthropic['bindTools']>[0];
}

function getWebSearchTool(): {
  type: 'web_search_20250305';
  name: 'web_search';
  max_uses: number;
} {
  return {
    type: 'web_search_20250305',
    name: 'web_search',
    max_uses: 3,
  };
}

function getWebSearchLLMConfig(): AnthropicClientOptions & SharedLLMConfig {
  return {
    provider: Providers.ANTHROPIC,
    model: webSearchModelName,
    maxTokens: 1024,
    promptCache: true,
    streaming: true,
    streamUsage: true,
    thinking: { type: 'adaptive' },
  } as AnthropicClientOptions & SharedLLMConfig;
}

async function createWebSearchRun({
  runId,
  customHandlers,
}: {
  runId: string;
  customHandlers?: RunConfig['customHandlers'];
}): Promise<Run<IState>> {
  return await Run.create<IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig: getWebSearchLLMConfig(),
      tools: [getWebSearchTool()],
      instructions:
        'You are a concise assistant. Use web search when current facts are needed.',
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  });
}

function createLibreChatContentHandlers(): {
  aggregateContent: ReturnType<
    typeof createContentAggregator
  >['aggregateContent'];
  contentParts: Array<LibreChatContentBlock | undefined>;
  customHandlers: NonNullable<RunConfig['customHandlers']>;
} {
  const { contentParts, aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: StreamEventData): void => {
        aggregateContent({ event, data: data as RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: StreamEventData
      ): void => {
        aggregateContent({ event, data: data as RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: StreamEventData
      ): void => {
        aggregateContent({ event, data: data as MessageDeltaEvent });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: StreamEventData
      ): void => {
        aggregateContent({ event, data: data as ReasoningDeltaEvent });
      },
    },
  };
  return { aggregateContent, contentParts, customHandlers };
}

test('Test ChatAnthropic', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.invoke([message]);
  expect(res.response_metadata.usage).toBeDefined();
});

test('Test ChatAnthropic with a bad API key throws appropriate error', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    apiKey: 'bad',
  });
  let error: unknown;
  try {
    const message = new HumanMessage('Hello!');
    await chat.invoke([message]);
  } catch (e) {
    error = e;
  }
  expect(error).toBeDefined();
  expect(getLangChainErrorCode(error)).toEqual('MODEL_AUTHENTICATION');
});

test('Test ChatAnthropic with unknown model throws appropriate error', async () => {
  const chat = new ChatAnthropic({
    modelName: 'badbad',
    maxRetries: 0,
  });
  let error: unknown;
  try {
    const message = new HumanMessage('Hello!');
    await chat.invoke([message]);
  } catch (e) {
    error = e;
  }
  expect(error).toBeDefined();
  expect(getLangChainErrorCode(error)).toEqual('MODEL_NOT_FOUND');
});

test('Test ChatAnthropic Generate', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.generate([[message], [message]]);
  expect(res.generations.length).toBe(2);
  for (const generation of res.generations) {
    expect(generation.length).toBe(1);
    for (const message of generation) {
      // console.log(message.text);
    }
  }
  // console.log({ res });
});

test.skip('Test ChatAnthropic Generate w/ ClientOptions', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    clientOptions: {
      defaultHeaders: {
        'Helicone-Auth': 'HELICONE_API_KEY',
      },
    },
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.generate([[message], [message]]);
  expect(res.generations.length).toBe(2);
  for (const generation of res.generations) {
    expect(generation.length).toBe(1);
    for (const message of generation) {
      // console.log(message.text);
    }
  }
  // console.log({ res });
});

test('Test ChatAnthropic Generate with a signal in call options', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
  });
  const controller = new AbortController();
  const message = new HumanMessage(
    'How is your day going? Be extremely verbose!'
  );
  await expect(() => {
    const res = chat.generate([[message], [message]], {
      signal: controller.signal,
    });
    setTimeout(() => {
      controller.abort();
    }, 1000);
    return res;
  }).rejects.toThrow();
}, 10000);

test('Test ChatAnthropic tokenUsage with a batch', async () => {
  const model = new ChatAnthropic({
    temperature: 0,
    maxRetries: 0,
    modelName,
  });
  const res = await model.generate([
    [new HumanMessage(`Hello!`)],
    [new HumanMessage(`Hi!`)],
  ]);
  // console.log({ res });
});

test('Test ChatAnthropic in streaming mode', async () => {
  let nrNewTokens = 0;
  let streamedCompletion = '';

  const model = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    streaming: true,
    callbacks: CallbackManager.fromHandlers({
      async handleLLMNewToken(token: string) {
        nrNewTokens += 1;
        streamedCompletion += token;
      },
    }),
  });
  const message = new HumanMessage('Hello!');
  const res = await model.invoke([message]);
  // console.log({ res });

  expect(nrNewTokens > 0).toBe(true);
  expect(res.content).toBe(streamedCompletion);
});

test('Test ChatAnthropic in streaming mode with a signal', async () => {
  let nrNewTokens = 0;
  let streamedCompletion = '';

  const model = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    streaming: true,
    callbacks: CallbackManager.fromHandlers({
      async handleLLMNewToken(token: string) {
        nrNewTokens += 1;
        streamedCompletion += token;
      },
    }),
  });
  const controller = new AbortController();
  const message = new HumanMessage(
    'Hello! Give me an extremely verbose response'
  );
  await expect(() => {
    const res = model.invoke([message], {
      signal: controller.signal,
    });
    setTimeout(() => {
      controller.abort();
    }, 500);
    return res;
  }).rejects.toThrow();

  // console.log({ nrNewTokens, streamedCompletion });
}, 5000);

test.skip('Test ChatAnthropic prompt value', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.generatePrompt([new ChatPromptValue([message])]);
  expect(res.generations.length).toBe(1);
  for (const generation of res.generations) {
    for (const g of generation) {
      // console.log(g.text);
    }
  }
  // console.log({ res });
});

test.skip('ChatAnthropic, docs, prompt templates', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    temperature: 0,
  });

  const systemPrompt = PromptTemplate.fromTemplate(
    'You are a helpful assistant that translates {input_language} to {output_language}.'
  );

  const chatPrompt = ChatPromptTemplate.fromMessages([
    new SystemMessagePromptTemplate(systemPrompt),
    HumanMessagePromptTemplate.fromTemplate('{text}'),
  ]);

  const responseA = await chat.generatePrompt([
    await chatPrompt.formatPromptValue({
      input_language: 'English',
      output_language: 'French',
      text: 'I love programming.',
    }),
  ]);

  // console.log(responseA.generations);
});

test.skip('ChatAnthropic, longer chain of messages', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    temperature: 0,
  });

  const chatPrompt = ChatPromptTemplate.fromMessages([
    HumanMessagePromptTemplate.fromTemplate(`Hi, my name is Joe!`),
    AIMessagePromptTemplate.fromTemplate(`Nice to meet you, Joe!`),
    HumanMessagePromptTemplate.fromTemplate('{text}'),
  ]);

  const responseA = await chat.generatePrompt([
    await chatPrompt.formatPromptValue({
      text: 'What did I just say my name was?',
    }),
  ]);

  // console.log(responseA.generations);
});

test.skip('ChatAnthropic, Anthropic apiUrl set manually via constructor', async () => {
  // Pass the default URL through (should use this, and work as normal)
  const anthropicApiUrl = 'https://api.anthropic.com';
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    anthropicApiUrl,
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.invoke([message]);
  // console.log({ res });
});

test('Test ChatAnthropic stream method', async () => {
  const model = new ChatAnthropic({
    maxTokens: 50,
    maxRetries: 0,
    modelName,
  });
  const stream = await model.stream('Print hello world.');
  const chunks: AIMessageChunk[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  expect(chunks.length).toBeGreaterThan(1);
});

test('Test ChatAnthropic stream method with abort', async () => {
  await expect(async () => {
    const model = new ChatAnthropic({
      maxTokens: 500,
      maxRetries: 0,
      modelName,
    });
    const stream = await model.stream(
      'How is your day going? Be extremely verbose.',
      {
        signal: AbortSignal.timeout(1000),
      }
    );
    for await (const chunk of stream) {
      // console.log(chunk);
    }
  }).rejects.toThrow();
});

test('Test ChatAnthropic stream method with early break', async () => {
  const model = new ChatAnthropic({
    maxTokens: 50,
    maxRetries: 0,
    modelName,
  });
  const stream = await model.stream(
    'How is your day going? Be extremely verbose.'
  );
  let i = 0;
  for await (const chunk of stream) {
    // console.log(chunk);
    i += 1;
    if (i > 10) {
      break;
    }
  }
});

test('Test ChatAnthropic headers passed through', async () => {
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    apiKey: 'NOT_REAL',
    clientOptions: {
      defaultHeaders: {
        'X-Api-Key': process.env.ANTHROPIC_API_KEY,
      },
    },
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.invoke([message]);
  // console.log({ res });
});

describe('ChatAnthropic image inputs', () => {
  test.each(['invoke', 'stream'])(
    'Test ChatAnthropic image_url, %s',
    async (invocationType: string) => {
      const chat = new ChatAnthropic({
        modelName,
        maxRetries: 0,
      });

      const dataUrlRes = invoke(chat, invocationType, [
        new HumanMessage({
          content: [
            {
              type: 'image_url',
              image_url: {
                url: 'data:image/jpeg;base64,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',
              },
            },
            { type: 'text', text: 'Describe this image.' },
          ],
        }),
      ]);

      await expect(dataUrlRes).resolves.toBeDefined();

      const urlRes = invoke(chat, invocationType, [
        new HumanMessage({
          content: [
            {
              type: 'image_url',
              image_url: remoteImageUrl,
            },
            { type: 'text', text: 'Describe this image.' },
          ],
        }),
      ]);

      await expect(urlRes).resolves.toBeDefined();

      const invalidSchemeRes = invoke(chat, invocationType, [
        new HumanMessage({
          content: [
            {
              type: 'image_url',
              image_url: 'file:///path/to/image.png',
            },
            { type: 'text', text: 'Describe this image.' },
          ],
        }),
      ]);

      await expect(invalidSchemeRes).rejects.toThrow(
        [
          'Invalid image URL protocol: "file:". Anthropic only supports images as http, https, or base64-encoded data URLs on \'image_url\' content blocks.',
          'Example: data:image/png;base64,/9j/4AAQSk...',
          'Example: https://example.com/image.jpg',
        ].join('\n\n')
      );

      const invalidUrlRes = invoke(chat, invocationType, [
        new HumanMessage({
          content: [
            {
              type: 'image_url',
              image_url: "this isn't a valid URL",
            },
            { type: 'text', text: 'Describe this image.' },
          ],
        }),
      ]);

      await expect(invalidUrlRes).rejects.toThrow(
        [
          `Malformed image URL: "this isn't a valid URL". Content blocks of type 'image_url' must be a valid http, https, or base64-encoded data URL.`,
          'Example: data:image/png;base64,/9j/4AAQSk...',
          'Example: https://example.com/image.jpg',
        ].join('\n\n')
      );
    }
  );

  test.each(['invoke', 'stream'])(
    'Test ChatAnthropic Anthropic Image Block, %s',
    async (invocationType: string) => {
      const chat = new ChatAnthropic({
        modelName,
        maxRetries: 0,
      });
      const base64Res = invoke(chat, invocationType, [
        new HumanMessage({
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: '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',
              },
            },
            {
              type: 'text',
              text: 'Describe this image',
            },
          ],
        }),
      ]);

      await expect(base64Res).resolves.toBeDefined();

      const urlRes = chat.invoke([
        new HumanMessage({
          content: [
            {
              type: 'image',
              source: {
                type: 'url',
                url: remoteImageUrl,
              },
            },
          ],
        }),
      ]);

      await expect(urlRes).resolves.toBeDefined();
    }
  );
});

test('Stream tokens', async () => {
  const model = new ChatAnthropic({
    modelName,
    temperature: 0,
    maxTokens: 10,
  });
  let res: AIMessageChunk | null = null;
  for await (const chunk of await model.stream(
    'Why is the sky blue? Be concise.'
  )) {
    if (!res) {
      res = chunk;
    } else {
      res = res.concat(chunk);
    }
  }
  // console.log(res);
  expect(res?.usage_metadata).toBeDefined();
  if (!res?.usage_metadata) {
    return;
  }
  expect(res.usage_metadata.input_tokens).toBeGreaterThan(1);
  expect(res.usage_metadata.output_tokens).toBeGreaterThan(1);
  expect(res.usage_metadata.total_tokens).toBe(
    res.usage_metadata.input_tokens + res.usage_metadata.output_tokens
  );
});

test('Anthropic usage metadata includes cache input token buckets', () => {
  const usageMetadata = getAnthropicUsageMetadata({
    input_tokens: 10,
    output_tokens: 5,
    cache_creation_input_tokens: 20,
    cache_read_input_tokens: 30,
  });

  expect(usageMetadata).toEqual({
    input_tokens: 60,
    output_tokens: 5,
    total_tokens: 65,
    input_token_details: {
      cache_creation: 20,
      cache_read: 30,
    },
  });
});

type AnthropicStreamEvent = Anthropic.Beta.Messages.BetaRawMessageStreamEvent;

function createMockAnthropicStream(events: AnthropicStreamEvent[]) {
  return {
    controller: { abort: jest.fn() },
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
  };
}

class MockStreamingAnthropic extends ChatAnthropic {
  constructor(private readonly mockEvents: AnthropicStreamEvent[]) {
    super({
      modelName,
      apiKey: 'test-key',
      maxTokens: 10,
      streamUsage: true,
    });
  }

  protected override async createStreamWithRetry() {
    return createMockAnthropicStream(this.mockEvents) as never;
  }
}

class RecordingStreamingAnthropic extends ChatAnthropic {
  messageStartOutputTokens = 0;
  readonly messageDeltaOutputTokens: number[] = [];

  protected override async createStreamWithRetry(
    request: AnthropicStreamingMessageCreateParams,
    options?: AnthropicRequestOptions
  ) {
    const stream = await super.createStreamWithRetry(request, options);
    const recorder = this;

    return {
      controller: stream.controller,
      async *[Symbol.asyncIterator](): AsyncGenerator<AnthropicMessageStreamEvent> {
        for await (const event of stream) {
          if (event.type === 'message_start') {
            recorder.messageStartOutputTokens =
              event.message.usage.output_tokens ??
              recorder.messageStartOutputTokens;
          } else if (event.type === 'message_delta') {
            recorder.messageDeltaOutputTokens.push(event.usage.output_tokens);
          }
          yield event;
        }
      },
    } as unknown as typeof stream;
  }
}

test('Anthropic message_delta usage preserves cumulative input totals', () => {
  const event: AnthropicStreamEvent = {
    type: 'message_delta',
    context_management: null,
    delta: {
      container: null,
      stop_details: null,
      stop_reason: 'end_turn',
      stop_sequence: null,
    },
    usage: {
      input_tokens: 243,
      output_tokens: 375,
      cache_creation_input_tokens: 11,
      cache_read_input_tokens: 13,
      fallback_credit: null,
      server_tool_use: null,
      iterations: null,
      output_tokens_details: null,
    },
  };

  const result = _makeMessageChunkFromAnthropicEvent(event, {
    streamUsage: true,
    coerceContentToString: true,
  });

  expect(result?.chunk.usage_metadata).toEqual({
    input_tokens: 267,
    output_tokens: 375,
    total_tokens: 642,
    input_token_details: {
      cache_creation: 11,
      cache_read: 13,
    },
  });
});

test('Anthropic stream usage does not double-count cumulative input tokens', async () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'message_start',
      message: {
        id: 'msg_token_accounting',
        container: null,
        context_management: null,
        content: [],
        diagnostics: null,
        model: modelName,
        role: 'assistant',
        stop_details: null,
        stop_reason: null,
        stop_sequence: null,
        type: 'message',
        usage: {
          cache_creation: null,
          cache_creation_input_tokens: 20,
          cache_read_input_tokens: 30,
          fallback_credit: null,
          inference_geo: null,
          input_tokens: 243,
          iterations: null,
          output_tokens: 7,
          server_tool_use: null,
          service_tier: null,
          speed: null,
          output_tokens_details: null,
        },
      },
    },
    {
      type: 'message_delta',
      context_management: null,
      delta: {
        container: null,
        stop_details: null,
        stop_reason: 'end_turn',
        stop_sequence: null,
      },
      usage: {
        input_tokens: 243,
        output_tokens: 375,
        cache_creation_input_tokens: 20,
        cache_read_input_tokens: 30,
        fallback_credit: null,
        server_tool_use: null,
        iterations: null,
        output_tokens_details: null,
      },
    },
    { type: 'message_stop' },
  ];
  const model = new MockStreamingAnthropic(events);

  let full: AIMessageChunk | undefined;
  for await (const chunk of await model.stream('hello')) {
    full = !full ? chunk : concat(full, chunk);
  }

  expect(full?.usage_metadata).toEqual({
    input_tokens: 293,
    output_tokens: 375,
    total_tokens: 668,
    input_token_details: {
      cache_creation: 20,
      cache_read: 30,
    },
    output_token_details: {},
  });
});

test('Anthropic stream usage accepts input first reported at message_delta', async () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'message_start',
      message: {
        id: 'msg_late_input_usage',
        container: null,
        context_management: null,
        content: [],
        diagnostics: null,
        model: modelName,
        role: 'assistant',
        stop_details: null,
        stop_reason: null,
        stop_sequence: null,
        type: 'message',
        usage: {
          cache_creation: null,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
          fallback_credit: null,
          inference_geo: null,
          input_tokens: 0,
          iterations: null,
          output_tokens: 0,
          server_tool_use: null,
          service_tier: null,
          speed: null,
          output_tokens_details: null,
        },
      },
    },
    {
      type: 'message_delta',
      context_management: null,
      delta: {
        container: null,
        stop_details: null,
        stop_reason: 'end_turn',
        stop_sequence: null,
      },
      usage: {
        input_tokens: 100,
        output_tokens: 42,
        cache_creation_input_tokens: 500,
        cache_read_input_tokens: 200,
        fallback_credit: null,
        server_tool_use: null,
        iterations: null,
        output_tokens_details: null,
      },
    },
    { type: 'message_stop' },
  ];
  const model = new MockStreamingAnthropic(events);

  let full: AIMessageChunk | undefined;
  for await (const chunk of await model.stream('hello')) {
    full = !full ? chunk : concat(full, chunk);
  }

  expect(full?.usage_metadata).toEqual({
    input_tokens: 800,
    output_tokens: 42,
    total_tokens: 842,
    input_token_details: {
      cache_creation: 500,
      cache_read: 200,
    },
    output_token_details: {},
  });
});

test('Anthropic stream usage handles multiple cumulative message_delta events', async () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'message_start',
      message: {
        id: 'msg_token_accounting_multi_delta',
        container: null,
        context_management: null,
        content: [],
        diagnostics: null,
        model: modelName,
        role: 'assistant',
        stop_details: null,
        stop_reason: null,
        stop_sequence: null,
        type: 'message',
        usage: {
          cache_creation: null,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
          fallback_credit: null,
          inference_geo: null,
          input_tokens: 243,
          iterations: null,
          output_tokens: 7,
          server_tool_use: null,
          service_tier: null,
          speed: null,
          output_tokens_details: null,
        },
      },
    },
    {
      type: 'message_delta',
      context_management: null,
      delta: {
        container: null,
        stop_details: null,
        stop_reason: null,
        stop_sequence: null,
      },
      usage: {
        input_tokens: 243,
        output_tokens: 100,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        fallback_credit: null,
        server_tool_use: null,
        iterations: null,
        output_tokens_details: null,
      },
    },
    {
      type: 'message_delta',
      context_management: null,
      delta: {
        container: null,
        stop_details: null,
        stop_reason: 'end_turn',
        stop_sequence: null,
      },
      usage: {
        input_tokens: 243,
        output_tokens: 375,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        fallback_credit: null,
        server_tool_use: null,
        iterations: null,
        output_tokens_details: null,
      },
    },
    { type: 'message_stop' },
  ];
  const model = new MockStreamingAnthropic(events);

  let full: AIMessageChunk | undefined;
  for await (const chunk of await model.stream('hello')) {
    full = !full ? chunk : concat(full, chunk);
  }

  expect(full?.usage_metadata).toEqual({
    input_tokens: 243,
    output_tokens: 375,
    total_tokens: 618,
    input_token_details: {
      cache_creation: 0,
      cache_read: 0,
    },
    output_token_details: {},
  });
});

test('Anthropic stream smoothing skips empty text block starts', async () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'text',
        text: '',
        citations: null,
      },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'text_delta',
        text: 'hello',
      },
    },
    { type: 'message_stop' },
  ];
  const model = new MockStreamingAnthropic(events);
  const contents: string[] = [];

  for await (const chunk of await model.stream('hello')) {
    if (typeof chunk.content === 'string') {
      contents.push(chunk.content);
    }
  }

  expect(contents).toEqual(['hello']);
});

test('Anthropic stream smoothing closes a queue-full producer after early break', async () => {
  const events: AnthropicStreamEvent[] = [
    {
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'text_delta',
        text: 'word '.repeat(3000),
      },
    },
    { type: 'message_stop' },
  ];
  const model = new MockStreamingAnthropic(events);
  const readOneChunk = async (): Promise<void> => {
    for await (const chunk of await model.stream('hello')) {
      if (typeof chunk.content === 'string' && chunk.content.length > 0) {
        break;
      }
    }
  };
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => {
      reject(new Error('stream close timed out'));
    }, 1000);
  });

  await expect(
    Promise.race([readOneChunk(), timeout])
  ).resolves.toBeUndefined();
});

test('Anthropic live stream usage matches raw cumulative output snapshots', async () => {
  const model = new RecordingStreamingAnthropic({
    modelName,
    temperature: 0,
    maxTokens: 500,
    _lc_stream_delay: 0,
  });

  let full: AIMessageChunk | undefined;
  const stream = await model.stream(
    'Write exactly 18 numbered lines about reliable software telemetry. Each line should contain exactly seven words. Do not add an intro or outro.'
  );
  for await (const chunk of stream) {
    full = !full ? chunk : concat(full, chunk);
  }

  expect(model.messageDeltaOutputTokens.length).toBeGreaterThan(0);
  const rawOutputTokens =
    model.messageDeltaOutputTokens[model.messageDeltaOutputTokens.length - 1];
  expect(rawOutputTokens).toBeGreaterThanOrEqual(
    model.messageStartOutputTokens
  );
  expect(full?.usage_metadata?.output_tokens).toBe(rawOutputTokens);
  expect(full?.usage_metadata?.total_tokens).toBe(
    (full?.usage_metadata?.input_tokens ?? 0) +
      (full?.usage_metadata?.output_tokens ?? 0)
  );

  if (model.messageDeltaOutputTokens.length > 1) {
    const summedOutputTokens = model.messageDeltaOutputTokens.reduce(
      (sum, tokens) => sum + tokens,
      0
    );
    expect(full?.usage_metadata?.output_tokens).toBeLessThan(
      summedOutputTokens
    );
  }
});

test('document detection ignores null content placeholders', () => {
  const params: AnthropicMessageCreateParams = {
    model: modelName,
    max_tokens: 16,
    messages: [
      {
        role: 'user',
        content: [null as never, { type: 'text', text: 'hello' }],
      },
    ],
  };

  expect(_documentsInParams(params)).toBe(false);
});

test('id is supplied when invoking', async () => {
  const model = new ChatAnthropic({ modelName });
  const result = await model.invoke('Hello');
  expect(result.id).toBeDefined();
  expect(result.id).not.toEqual('');
});

test('id is supplied when streaming', async () => {
  const model = new ChatAnthropic({ modelName });
  let finalChunk: AIMessageChunk | undefined;
  for await (const chunk of await model.stream('Hello')) {
    finalChunk = !finalChunk ? chunk : concat(finalChunk, chunk);
  }
  expect(finalChunk).toBeDefined();
  if (!finalChunk) return;
  expect(finalChunk.id).toBeDefined();
  expect(finalChunk.id).not.toEqual('');
});

const CACHED_TEXT = `## Components

LangChain provides standard, extendable interfaces and external integrations for various components useful for building with LLMs.
Some components LangChain implements, some components we rely on third-party integrations for, and others are a mix.

### Chat models

<span data-heading-keywords="chat model,chat models"></span>

Language models that use a sequence of messages as inputs and return chat messages as outputs (as opposed to using plain text).
These are generally newer models (older models are generally \`LLMs\`, see below).
Chat models support the assignment of distinct roles to conversation messages, helping to distinguish messages from the AI, users, and instructions such as system messages.

Although the underlying models are messages in, message out, the LangChain wrappers also allow these models to take a string as input.
This gives them the same interface as LLMs (and simpler to use).
When a string is passed in as input, it will be converted to a \`HumanMessage\` under the hood before being passed to the underlying model.

LangChain does not host any Chat Models, rather we rely on third party integrations.

We have some standardized parameters when constructing ChatModels:

- \`model\`: the name of the model

Chat Models also accept other parameters that are specific to that integration.

:::important
Some chat models have been fine-tuned for **tool calling** and provide a dedicated API for it.
Generally, such models are better at tool calling than non-fine-tuned models, and are recommended for use cases that require tool calling.
Please see the [tool calling section](/docs/concepts/#functiontool-calling) for more information.
:::

For specifics on how to use chat models, see the [relevant how-to guides here](/docs/how_to/#chat-models).

#### Multimodality

Some chat models are multimodal, accepting images, audio and even video as inputs.
These are still less common, meaning model providers haven't standardized on the "best" way to define the API.
Multimodal outputs are even less common. As such, we've kept our multimodal abstractions fairly light weight
and plan to further solidify the multimodal APIs and interaction patterns as the field matures.

In LangChain, most chat models that support multimodal inputs also accept those values in OpenAI's content blocks format.
So far this is restricted to image inputs. For models like Gemini which support video and other bytes input, the APIs also support the native, model-specific representations.

For specifics on how to use multimodal models, see the [relevant how-to guides here](/docs/how_to/#multimodal).

### LLMs

<span data-heading-keywords="llm,llms"></span>

:::caution
Pure text-in/text-out LLMs tend to be older or lower-level. Many popular models are best used as [chat completion models](/docs/concepts/#chat-models),
even for non-chat use cases.

You are probably looking for [the section above instead](/docs/concepts/#chat-models).
:::

Language models that takes a string as input and returns a string.
These are traditionally older models (newer models generally are [Chat Models](/docs/concepts/#chat-models), see above).

Although the underlying models are string in, string out, the LangChain wrappers also allow these models to take messages as input.
This gives them the same interface as [Chat Models](/docs/concepts/#chat-models).
When messages are passed in as input, they will be formatted into a string under the hood before being passed to the underlying model.

LangChain does not host any LLMs, rather we rely on third party integrations.

For specifics on how to use LLMs, see the [relevant how-to guides here](/docs/how_to/#llms).

### Message types

Some language models take an array of messages as input and return a message.
There are a few different types of messages.
All messages have a \`role\`, \`content\`, and \`response_metadata\` property.

The \`role\` describes WHO is saying the message.
LangChain has different message classes for different roles.

The \`content\` property describes the content of the message.
This can be a few different things:

- A string (most models deal this type of content)
- A List of objects (this is used for multi-modal input, where the object contains information about that input type and that input location)

#### HumanMessage

This represents a message from the user.

#### AIMessage

This represents a message from the model. In addition to the \`content\` property, these messages also have:

**\`response_metadata\`**

The \`response_metadata\` property contains additional metadata about the response. The data here is often specific to each model provider.
This is where information like log-probs and token usage may be stored.

**\`tool_calls\`**

These represent a decision from an language model to call a tool. They are included as part of an \`AIMessage\` output.
They can be accessed from there with the \`.tool_calls\` property.

This property returns a list of \`ToolCall\`s. A \`ToolCall\` is an object with the following arguments:

- \`name\`: The name of the tool that should be called.
- \`args\`: The arguments to that tool.
- \`id\`: The id of that tool call.

#### SystemMessage

This represents a system message, which tells the model how to behave. Not every model provider supports this.

#### ToolMessage

This represents the result of a tool call. In addition to \`role\` and \`content\`, this message has:

- a \`tool_call_id\` field which conveys the id of the call to the tool that was called to produce this result.
- an \`artifact\` field which can be used to pass along arbitrary artifacts of the tool execution which are useful to track but which should not be sent to the model.

#### (Legacy) FunctionMessage

This is a legacy message type, corresponding to OpenAI's legacy function-calling API. \`ToolMessage\` should be used instead to correspond to the updated tool-calling API.

This represents the result of a function call. In addition to \`role\` and \`content\`, this message has a \`name\` parameter which conveys the name of the function that was called to produce this result.

### Prompt templates

<span data-heading-keywords="prompt,prompttemplate,chatprompttemplate"></span>

Prompt templates help to translate user input and parameters into instructions for a language model.
This can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based output.

Prompt Templates take as input an object, where each key represents a variable in the prompt template to fill in.

Prompt Templates output a PromptValue. This PromptValue can be passed to an LLM or a ChatModel, and can also be cast to a string or an array of messages.
The reason this PromptValue exists is to make it easy to switch between strings and messages.

There are a few different types of prompt templates:

#### String PromptTemplates

These prompt templates are used to format a single string, and generally are used for simpler inputs.
For example, a common way to construct and use a PromptTemplate is as follows:

\`\`\`typescript
import { PromptTemplate } from "@langchain/core/prompts";

const promptTemplate = PromptTemplate.fromTemplate(
  "Tell me a joke about {topic}"
);

await promptTemplate.invoke({ topic: "cats" });
\`\`\`

#### ChatPromptTemplates

These prompt templates are used to format an array of messages. These "templates" consist of an array of templates themselves.
For example, a common way to construct and use a ChatPromptTemplate is as follows:

\`\`\`typescript
import { ChatPromptTemplate } from "@langchain/core/prompts";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  ["user", "Tell me a joke about {topic}"],
]);

await promptTemplate.invoke({ topic: "cats" });
\`\`\`

In the above example, this ChatPromptTemplate will construct two messages when called.
The first is a system message, that has no variables to format.
The second is a HumanMessage, and will be formatted by the \`topic\` variable the user passes in.

#### MessagesPlaceholder

<span data-heading-keywords="messagesplaceholder"></span>

This prompt template is responsible for adding an array of messages in a particular place.
In the above ChatPromptTemplate, we saw how we could format two messages, each one a string.
But what if we wanted the user to pass in an array of messages that we would slot into a particular spot?
This is how you use MessagesPlaceholder.

\`\`\`typescript
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { HumanMessage } from "@langchain/core/messages";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  new MessagesPlaceholder("msgs"),
]);

promptTemplate.invoke({ msgs: [new HumanMessage({ content: "hi!" })] });
\`\`\`

This will produce an array of two messages, the first one being a system message, and the second one being the HumanMessage we passed in.
If we had passed in 5 messages, then it would have produced 6 messages in total (the system message plus the 5 passed in).
This is useful for letting an array of messages be slotted into a particular spot.

An alternative way to accomplish the same thing without using the \`MessagesPlaceholder\` class explicitly is:

\`\`\`typescript
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant"],
  ["placeholder", "{msgs}"], // <-- This is the changed part
]);
\`\`\`

For specifics on how to use prompt templates, see the [relevant how-to guides here](/docs/how_to/#prompt-templates).

### Example Selectors

One common prompting technique for achieving better performance is to include examples as part of the prompt.
This gives the language model concrete examples of how it should behave.
Sometimes these examples are hardcoded into the prompt, but for more advanced situations it may be nice to dynamically select them.
Example Selectors are classes responsible for selecting and then formatting examples into prompts.

For specifics on how to use example selectors, see the [relevant how-to guides here](/docs/how_to/#example-selectors).

### Output parsers

<span data-heading-keywords="output parser"></span>

:::note

The information here refers to parsers that take a text output from a model try to parse it into a more structured representation.
More and more models are supporting function (or tool) calling, which handles this automatically.
It is recommended to use function/tool calling rather than output parsing.
See documentation for that [here](/docs/concepts/#function-tool-calling).

:::

Responsible for taking the output of a model and transforming it to a more suitable format for downstream tasks.
Useful when you are using LLMs to generate structured data, or to normalize output from chat models and LLMs.

There are two main methods an output parser must implement:

- "Get format instructions": A method which returns a string containing instructions for how the output of a language model should be formatted.
- "Parse": A method which takes in a string (assumed to be the response from a language model) and parses it into some structure.

And then one optional one:

- "Parse with prompt": A method which takes in a string (assumed to be the response from a language model) and a prompt (assumed to be the prompt that generated such a response) and parses it into some structure. The prompt is largely provided in the event the OutputParser wants to retry or fix the output in some way, and needs information from the prompt to do so.

Output parsers accept a string or \`BaseMessage\` as input and can return an arbitrary type.

LangChain has many different types of output parsers. This is a list of output parsers LangChain supports. The table below has various pieces of information:

**Name**: The name of the output parser

**Supports Streaming**: Whether the output parser supports streaming.

**Input Type**: Expected input type. Most output parsers work on both strings and messages, but some (like OpenAI Functions) need a message with specific arguments.

**Output Type**: The output type of the object returned by the parser.

**Description**: Our commentary on this output parser and when to use it.

The current date is ${new Date().toISOString()}`;

test('system prompt caching', async () => {
  const model = new ChatAnthropic({
    modelName: citationsModelName,
    clientOptions: {
      defaultHeaders: {
        'anthropic-beta': 'prompt-caching-2024-07-31',
      },
    },
  });
  const messages = [
    new SystemMessage({
      content: [
        {
          type: 'text',
          text: `${new Date().toISOString()} (Now)`,
        },
        {
          type: 'text',
          text: `You are a pirate. Always respond in pirate dialect.\nUse the following as context when answering questions: ${CACHED_TEXT}`,
          cache_control: { type: 'ephemeral' },
        },
      ],
    }),
    new HumanMessage({
      content: 'What types of messages are supported in LangChain?',
    }),
  ];
  const res = await model.invoke(messages);
  expect(
    res.usage_metadata?.input_token_details?.cache_creation
  ).toBeGreaterThan(0);
  expect(res.usage_metadata?.input_token_details?.cache_read).toBe(0);
  expect(res.usage_metadata?.input_tokens).toBeGreaterThan(
    res.usage_metadata?.input_token_details?.cache_creation ?? 0
  );
  const res2 = await model.invoke(messages);
  expect(res2.usage_metadata?.input_token_details?.cache_creation).toBe(0);
  expect(res2.usage_metadata?.input_token_details?.cache_read).toBeGreaterThan(
    0
  );
  expect(res2.usage_metadata?.input_tokens).toBeGreaterThan(
    res2.usage_metadata?.input_token_details?.cache_read ?? 0
  );
  const stream = await model.stream(messages);
  let agg;
  for await (const chunk of stream) {
    agg = agg === undefined ? chunk : concat(agg, chunk);
  }
  expect(agg).toBeDefined();
  expect(agg!.usage_metadata?.input_token_details?.cache_creation).toBe(0);
  expect(agg!.usage_metadata?.input_token_details?.cache_read).toBeGreaterThan(
    0
  );
  expect(agg!.usage_metadata?.input_tokens).toBeGreaterThan(
    agg!.usage_metadata?.input_token_details?.cache_read ?? 0
  );
});

// TODO: Add proper test with long tool content
test.skip('tool caching', async () => {
  const model = new ChatAnthropic({
    modelName,
    clientOptions: {
      defaultHeaders: {
        'anthropic-beta': 'prompt-caching-2024-07-31',
      },
    },
  }).bindTools([
    {
      name: 'get_weather',
      description: 'Get the weather for a specific location',
      input_schema: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'Location to get the weather for',
          },
          unit: {
            type: 'string',
            description: 'Temperature unit to return',
          },
        },
        required: ['location'],
      },
      cache_control: { type: 'ephemeral' },
    },
  ]);
  const messages = [
    new HumanMessage({
      content: 'What is the weather in Regensburg?',
    }),
  ];
  const res = await model.invoke(messages);
  console.log(res);
  expect(
    res.usage_metadata?.input_token_details?.cache_creation
  ).toBeGreaterThan(0);
  expect(res.usage_metadata?.input_token_details?.cache_read).toBe(0);
  const res2 = await model.invoke(messages);
  expect(res2.usage_metadata?.input_token_details?.cache_creation).toBe(0);
  expect(res2.usage_metadata?.input_token_details?.cache_read).toBeGreaterThan(
    0
  );
});

test.skip('Test ChatAnthropic with custom client', async () => {
  const client = new AnthropicVertex();
  const chat = new ChatAnthropic({
    modelName,
    maxRetries: 0,
    createClient: () => client,
  });
  const message = new HumanMessage('Hello!');
  const res = await chat.invoke([message]);
  // console.log({ res });
  expect(res.usage_metadata?.input_token_details).toBeDefined();
});

test('human message caching', async () => {
  const model = new ChatAnthropic({
    modelName: citationsModelName,
  });

  const messages = [
    new SystemMessage({
      content: [
        {
          type: 'text',
          text: `${new Date().toISOString()} (Now)`,
        },
        {
          type: 'text',
          text: `You are a pirate. Always respond in pirate dialect.\nUse the following as context when answering questions: ${CACHED_TEXT}`,
        },
      ],
    }),
    new HumanMessage({
      content: [
        {
          type: 'text',
          text: 'What types of messages are supported in LangChain?',
          cache_control: { type: 'ephemeral' },
        },
      ],
    }),
  ];

  const res = await model.invoke(messages);
  expect(
    res.usage_metadata?.input_token_details?.cache_creation
  ).toBeGreaterThan(0);
  expect(res.usage_metadata?.input_token_details?.cache_read).toBe(0);
  expect(res.usage_metadata?.input_tokens).toBeGreaterThan(
    res.usage_metadata?.input_token_details?.cache_creation ?? 0
  );
  const res2 = await model.invoke(messages);
  expect(res2.usage_metadata?.input_token_details?.cache_creation).toBe(0);
  expect(res2.usage_metadata?.input_token_details?.cache_read).toBeGreaterThan(
    0
  );
  expect(res2.usage_metadata?.input_tokens).toBeGreaterThan(
    res2.usage_metadata?.input_token_details?.cache_read ?? 0
  );

  const stream = await model.stream(messages);
  let agg;
  for await (const chunk of stream) {
    agg = agg === undefined ? chunk : concat(agg, chunk);
  }
  expect(agg).toBeDefined();
  expect(agg!.usage_metadata?.input_token_details?.cache_creation).toBe(0);
  expect(agg!.usage_metadata?.input_token_details?.cache_read).toBeGreaterThan(
    0
  );
  expect(agg!.usage_metadata?.input_tokens).toBeGreaterThan(
    agg!.usage_metadata?.input_token_details?.cache_read ?? 0
  );
});

describe('Anthropic web search live regressions', () => {
  test('accepts prompt-cache markers on built-in web search tools', async () => {
    const model = new ChatAnthropic({
      model: webSearchModelName,
      maxTokens: 1024,
      thinking: { type: 'adaptive' },
    });
    const tools = getPromptCachedWebSearchTools();
    const formattedTools = model.formatStructuredToolToAnthropic(tools);

    expect(formattedTools?.[0]).toMatchObject({
      type: 'web_search_20250305',
      name: 'web_search',
      cache_control: { type: 'ephemeral' },
    });
    expect(formattedTools?.[0]).not.toHaveProperty('extras');

    const response = await model
      .bindTools(tools)
      .invoke([
        new HumanMessage(
          'Use web search once and answer with only the word: ok'
        ),
      ]);

    expect(response.content.length).toBeGreaterThan(0);
  });

  test('replays LibreChat-persisted web search content across runs', async () => {
    const threadId = `web-search-e2e-${Date.now()}`;
    const firstPrompt =
      'Use web search. Who is the lowest seed survived in 2026 NBA playoffs? Answer with only the team name.';
    const followUpPrompt = "Who are 76ers' opponents in current series?";
    const { contentParts: firstContentParts, customHandlers: firstHandlers } =
      createLibreChatContentHandlers();
    const firstRun = await createWebSearchRun({
      runId: `${threadId}-turn-1`,
      customHandlers: firstHandlers,
    });
    const runConfig = {
      configurable: { provider: Providers.ANTHROPIC, thread_id: threadId },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const firstRunContent = await firstRun.processStream(
      { messages: [new HumanMessage(firstPrompt)] },
      runConfig
    );
    const persistedAssistantContent = firstContentParts.filter(
      (part): part is LibreChatContentBlock => part != null
    );
    const hasPersistedServerToolCall = persistedAssistantContent.some(
      (part) =>
        part.type === ContentTypes.TOOL_CALL &&
        typeof part.tool_call?.id === 'string' &&
        part.tool_call.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
    );
    const hasPersistedAnswerText = persistedAssistantContent.some(
      (part) =>
        part.type === ContentTypes.TEXT &&
        typeof part.text === 'string' &&
        part.text.trim().length > 0
    );

    expect(firstRunContent).toBeDefined();
    expect(persistedAssistantContent.length).toBeGreaterThan(0);
    expect(hasPersistedServerToolCall).toBe(true);
    expect(hasPersistedAnswerText).toBe(true);

    const persistedPayload: TPayload = [
      { role: 'user', content: firstPrompt },
      { role: 'assistant', content: persistedAssistantContent },
      { role: 'user', content: followUpPrompt },
    ];
    const { messages } = formatAgentMessages(
      persistedPayload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const secondRun = await createWebSearchRun({
      runId: `${threadId}-turn-2`,
    });

    expectAnthropicPayloadContentIsNonEmpty(anthropicPayload);
    expectNoDanglingServerToolUses(anthropicPayload);
    await expect(
      secondRun.processStream({ messages }, runConfig)
    ).resolves.toBeDefined();
  });
});

test('Can accept PDF documents', async () => {
  const model = new ChatAnthropic({
    modelName: pdfModelName,
  });

  const pdfPath = './src/llm/anthropic/Jacob_Lee_Resume_2023.pdf';
  const pdfBase64 = await fs.readFile(pdfPath, 'base64');

  const response = await model.invoke([
    ['system', 'Use the provided documents to answer the question'],
    [
      'user',
      [
        {
          type: 'document',
          source: {
            type: 'base64',
            media_type: 'application/pdf',
            data: pdfBase64,
          },
        },
        {
          type: 'text',
          text: 'Summarize the contents of this PDF',
        },
      ],
    ],
  ]);

  expect(response.content.length).toBeGreaterThan(10);
});

describe('Citations', () => {
  test('document blocks', async () => {
    const citationsModel = new ChatAnthropic({
      model: citationsModelName,
    });
    const messages = [
      {
        role: 'user',
        content: [
          {
            type: 'document',
            source: {
              type: 'text',
              media_type: 'text/plain',
              data: "The grass the user is asking about is bluegrass. The sky is orange because it's night.",
            },
            title: 'My Document',
            context: 'This is a trustworthy document.',
            citations: {
              enabled: true,
            },
          },
          {
            type: 'text',
            text: 'What color is the grass and sky?',
          },
        ],
      },
    ];

    const response = await citationsModel.invoke(messages);

    const responseBlocks = expectContentArray<ContentBlock>(response.content);
    expect(responseBlocks.length).toBeGreaterThan(2);
    const blocksWithCitations = responseBlocks.filter(hasCitations);
    expect(blocksWithCitations.length).toEqual(2);
    expect(typeof blocksWithCitations[0].citations[0]).toEqual('object');

    const stream = await citationsModel.stream(messages);
    let aggregated;
    let chunkHasCitation = false;
    for await (const chunk of stream) {
      aggregated = aggregated === undefined ? chunk : concat(aggregated, chunk);
      if (
        !chunkHasCitation &&
        Array.isArray(chunk.content) &&
        chunk.content.some(hasCitations)
      ) {
        chunkHasCitation = true;
      }
    }
    expect(chunkHasCitation).toBe(true);
    const aggregatedBlocks = expectContentArray<ContentBlock>(
      aggregated?.content ?? []
    );
    expect(aggregatedBlocks.length).toBeGreaterThan(2);
    expect(aggregatedBlocks.some(hasCitations)).toBe(true);
  });
  describe('search result blocks', () => {
    const citationsModel = new ChatAnthropic({
      model: citationsModelName,
      clientOptions: {
        defaultHeaders: {
          'anthropic-beta': 'search-results-2025-06-09',
        },
      },
    });

    const messages = [
      {
        role: 'user',
        content: [
          {
            type: 'search_result',
            title: 'History of France',
            source: 'https://example.com/france-history',
            citations: { enabled: true },
            content: [
              {
                type: 'text',
                text: 'The capital of France is Paris.',
              },
              {
                type: 'text',
                text: 'The old capital of France was Lyon.',
              },
            ],
          },
          {
            type: 'search_result',
            title: 'Geography of France',
            source: 'https://example.com/france-geography',
            citations: { enabled: true },
            content: [
              {
                type: 'text',
                text: 'France is a country in Europe.',
              },
              {
                type: 'text',
                text: 'France borders Spain to the south.',
              },
            ],
          },
          {
            type: 'text',
            text: 'What is the capital of France and where is it located? You must cite your sources.',
          },
        ],
      },
    ];

    test('without streaming', async () => {
      const response = await citationsModel.invoke(messages);

      const responseBlocks = expectContentArray<ContentBlock>(response.content);
      expect(responseBlocks.length).toBeGreaterThan(0);

      // Check that we have cited content
      const blocksWithCitations = responseBlocks.filter(hasCitations);
      expect(blocksWithCitations.length).toBeGreaterThan(0);

      // Verify citation structure
      const citation = blocksWithCitations[0].citations[0];
      expect(typeof citation).toBe('object');
      expect(citation.type).toBe('search_result_location');
      expect(citation.source).toBeDefined();
    });
    test('with streaming', async () => {
      // Test streaming
      const stream = await citationsModel.stream(messages);
      let aggregated;
      let chunkHasCitation = false;
      for await (const chunk of stream) {
        aggregated =
          aggregated === undefined ? chunk : concat(aggregated, chunk);
        if (
          !chunkHasCitation &&
          Array.isArray(chunk.content) &&
          chunk.content.some(hasCitations)
        ) {
          chunkHasCitation = true;
        }
      }
      expect(chunkHasCitation).toBe(true);
      const aggregatedBlocks = expectContentArray<ContentBlock>(
        aggregated?.content ?? []
      );
      expect(aggregatedBlocks.some(hasCitations)).toBe(true);
    });
  });

  test('search result blocks from tool', async () => {
    const ragTool = tool(
      (): ChatAnthropicContentBlock[] => [
        {
          type: 'search_result',
          title: 'History of France',
          source: 'https://example.com/france-history',
          citations: { enabled: true },
          content: [
            {
              type: 'text',
              text: 'The capital of France is Paris.',
            },
            {
              type: 'text',
              text: 'France was established as a republic in 1792.',
            },
          ],
        },
        {
          type: 'search_result',
          title: 'Geography of France',
          source: 'https://example.com/france-geography',
          citations: { enabled: true },
          content: [
            {
              type: 'text',
              text: 'France is located in Western Europe.',
            },
            {
              type: 'text',
              text: 'France has a population of approximately 67 million people.',
            },
          ],
        },
      ],
      {
        name: 'search_knowledge_base',
        description: 'Search the knowledge base for information about France',
        schema: z.object({
          query: z.string().describe('The search query'),
        }),
      }
    );

    const citationsModel = new ChatAnthropic({
      model: citationsModelName,
      clientOptions: {
        defaultHeaders: {
          'anthropic-beta': 'search-results-2025-06-09',
        },
      },
    }).bindTools([ragTool]);

    const messages: BaseMessage[] = [
      new HumanMessage(
        'Search for information about France and tell me what you find with proper citations.'
      ),
    ];

    const response = await citationsModel.invoke(messages);
    messages.push(response);

    expect(Array.isArray(response.content)).toBe(true);
    expect(response.content.length).toBeGreaterThan(0);

    // Check that the model called the tool
    expect(response.tool_calls?.length).toBeGreaterThan(0);
    expect(response.tool_calls?.[0].name).toBe('search_knowledge_base');

    const toolResponse = await ragTool.invoke(response.tool_calls![0]);
    messages.push(toolResponse);

    const response2 = await citationsModel.invoke(messages);

    const response2Blocks = expectContentArray<ContentBlock>(response2.content);
    expect(response2Blocks.length).toBeGreaterThan(0);
    // Make sure that a citation exists somewhere in the content list
    const citationBlock = expectDefined(response2Blocks.find(hasCitations));
    expect(citationBlock.citations[0].type).toBe('search_result_location');
    expect(citationBlock.citations[0].source).toBeDefined();
  });
});

// Inherited from @langchain/anthropic@1.5.1 tests/chat_models.test.ts — verifies
// the fork honors upstream's `thinkingExplicitlySet` gating in invocationParams.
describe('invocationParams thinking gating', () => {
  test('omits thinking when not explicitly configured', () => {
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      apiKey: 'testing',
    });

    expect(model.invocationParams({}).thinking).toBeUndefined();
  });

  test('includes thinking when explicitly disabled', () => {
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      apiKey: 'testing',
      thinking: { type: 'disabled' },
    });

    expect(model.invocationParams({}).thinking).toEqual({ type: 'disabled' });
  });

  test('includes thinking when explicitly enabled', () => {
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      temperature: 1,
      apiKey: 'testing',
      thinking: { type: 'enabled', budget_tokens: 1000 },
    });

    expect(model.invocationParams({}).thinking).toEqual({
      type: 'enabled',
      budget_tokens: 1000,
    });
  });
});

describe('invocationParams thinking block binding', () => {
  const binding = 'thinking-binding-controls-2026-08-01';
  const paramsFor = (fields: Record<string, unknown>) =>
    new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      apiKey: 'testing',
      ...fields,
    }).invocationParams({}) as { thinking?: unknown; betas?: string[] };

  test('sends drop_block with its beta when enabled thinking is replayed', () => {
    const params = paramsFor({
      thinking: { type: 'adaptive' },
      dropMismatchedThinkingBlocks: true,
    });

    expect(params.thinking).toEqual({
      type: 'adaptive',
      block_binding: { prefix_mismatch_behavior: 'drop_block' },
    });
    expect(params.betas).toContain(binding);
  });

  test('sends neither without the option, without thinking, or with thinking disabled', () => {
    for (const fields of [
      { thinking: { type: 'adaptive' } },
      { dropMismatchedThinkingBlocks: true },
      { thinking: { type: 'disabled' }, dropMismatchedThinkingBlocks: true },
    ]) {
      const params = paramsFor(fields);
      expect(JSON.stringify(params.thinking ?? null)).not.toContain('block_binding');
      expect(params.betas ?? []).not.toContain(binding);
    }
  });
});

// Inherited from @langchain/anthropic@1.5.1 — top-level request cache_control.
describe('invocationParams cache_control', () => {
  const newModel = () =>
    new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      apiKey: 'testing',
    });

  test('includes cache_control when provided in call options', () => {
    expect(
      newModel().invocationParams({ cache_control: { type: 'ephemeral' } })
        .cache_control
    ).toEqual({ type: 'ephemeral' });
  });

  test('includes cache_control with 1h ttl', () => {
    expect(
      newModel().invocationParams({
        cache_control: { type: 'ephemeral', ttl: '1h' },
      }).cache_control
    ).toEqual({ type: 'ephemeral', ttl: '1h' });
  });

  test('omits cache_control when not provided', () => {
    expect(newModel().invocationParams({}).cache_control).toBeUndefined();
  });
});

describe('Opus 4.7', () => {
  test('default max_tokens for claude-opus-4-7 is 16384', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
    });

    const params = model.invocationParams({});

    expect(params.max_tokens).toBe(16384);
  });

  test('does not apply Opus 4.7 rules to longer prefix matches', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-70',
      apiKey: 'testing',
      topK: 40,
    });

    const params = model.invocationParams({});

    expect(params.top_k).toBe(40);
  });

  test('rejects thinking.type=enabled for claude-opus-4-7', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      thinking: { type: 'enabled', budget_tokens: 2048 },
    });

    expect(() => model.invocationParams({})).toThrow(
      'thinking.type="enabled" is not supported for claude-opus-4-7; use thinking.type="adaptive" instead'
    );
  });

  test('rejects thinking.budget_tokens for claude-opus-4-7', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      thinking: {
        type: 'adaptive',
        budget_tokens: 2048,
      } as AnthropicThinkingConfigParam & { budget_tokens: number },
    });

    expect(() => model.invocationParams({})).toThrow(
      'thinking.budget_tokens is not supported for claude-opus-4-7; use outputConfig.effort instead'
    );
  });

  test('rejects non-default sampling params for claude-opus-4-7', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      temperature: 0.1,
    });

    expect(() => model.invocationParams({})).toThrow(
      'temperature is not supported for claude-opus-4-7 when set to non-default values'
    );
  });

  test('does not include sampling params for claude-opus-4-7 even if set to defaults', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      temperature: 1,
      topP: 1,
    });

    const params = model.invocationParams({});

    expect(params.temperature).toBeUndefined();
    expect(params.top_p).toBeUndefined();
    expect(params.top_k).toBeUndefined();
  });

  test('passes thinking.display through for claude-opus-4-7', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      thinking: { type: 'adaptive', display: 'summarized' },
    });

    const params = model.invocationParams({});

    expect(params.thinking).toEqual({
      type: 'adaptive',
      display: 'summarized',
    });
  });

  test('auto-adds task budget beta when outputConfig.task_budget is provided', () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-7',
      apiKey: 'testing',
      outputConfig: {
        effort: 'high',
        task_budget: { type: 'tokens', total: 128000 },
      } as AnthropicOutputConfigWithTaskBudget,
    });

    const params = model.invocationParams({});

    expect(params.betas).toContain('task-budgets-2026-03-13');
    expect(params.output_config).toEqual({
      effort: 'high',
      task_budget: {
        type: 'tokens',
        total: 128000,
      },
    });
  });
});

test('Test thinking blocks multiturn invoke', async () => {
  const model = new ChatAnthropic({
    model: extendedThinkingModelName,
    maxTokens: 5000,
    thinking: { type: 'enabled', budget_tokens: 2000 },
  });

  async function doInvoke(messages: BaseMessage[]) {
    const response = await model.invoke(messages);

    expect(Array.isArray(response.content)).toBe(true);
    const content = response.content as AnthropicMessageResponse[];
    expect(content.some(isThinkingBlock)).toBe(true);

    for (const block of content) {
      expect(typeof block).toBe('object');
      if (isThinkingBlock(block)) {
        expect(Object.keys(block).sort()).toEqual(
          ['type', 'thinking', 'signature'].sort()
        );
        expect(block.thinking).toBeTruthy();
        expect(typeof block.thinking).toBe('string');
        expect(block.signature).toBeTruthy();
        expect(typeof block.signature).toBe('string');
      }
    }
    return response;
  }

  const invokeMessages: BaseMessage[] = [new HumanMessage('Hello')];

  invokeMessages.push(await doInvoke(invokeMessages));
  invokeMessages.push(new HumanMessage('What is 42+7?'));

  // test a second time to make sure that we've got input translation working correctly
  await model.invoke(invokeMessages);
});

test('Test thinking blocks multiturn streaming', async () => {
  const model = new ChatAnthropic({
    model: extendedThinkingModelName,
    maxTokens: 5000,
    thinking: { type: 'enabled', budget_tokens: 2000 },
  });

  async function doStreaming(messages: BaseMessage[]) {
    let full: AIMessageChunk | null = null;
    for await (const chunk of await model.stream(messages)) {
      full = full ? concat(full, chunk) : chunk;
    }
    expect(full).toBeInstanceOf(AIMessageChunk);
    expect(Array.isArray(full?.content)).toBe(true);
    const content3 = full?.content as AnthropicMessageResponse[];
    expect(content3.some(isThinkingBlock)).toBe(true);

    for (const block of content3) {
      expect(typeof block).toBe('object');
      if (isThinkingBlock(block)) {
        expect(Object.keys(block).sort()).toEqual(
          ['type', 'thinking', 'signature', 'index'].sort()
        );
        expect(block.thinking).toBeTruthy();
        expect(typeof block.thinking).toBe('string');
        expect(block.signature).toBeTruthy();
        expect(typeof block.signature).toBe('string');
      }
    }
    return full as AIMessageChunk;
  }

  const streamingMessages: BaseMessage[] = [new HumanMessage('Hello')];

  streamingMessages.push(await doStreaming(streamingMessages));
  streamingMessages.push(new HumanMessage('What is 42+7?'));

  // test a second time to make sure that we've got input translation working correctly
  await doStreaming(streamingMessages);
});

test('Test redacted thinking blocks multiturn invoke', async () => {
  const model = new ChatAnthropic({
    model: extendedThinkingModelName,
    maxTokens: 5000,
    thinking: { type: 'enabled', budget_tokens: 2000 },
  });

  async function doInvoke(messages: BaseMessage[]) {
    const response = await model.invoke(messages);
    let hasReasoning = false;
    const content = response.content as AnthropicMessageResponse[];

    for (const block of content) {
      expect(typeof block).toBe('object');
      if (isRedactedThinkingBlock(block)) {
        hasReasoning = true;
        expect(Object.keys(block).sort()).toEqual(['type', 'data'].sort());
        expect(block.data).toBeTruthy();
        expect(typeof block.data).toBe('string');
      }
    }
    if (!hasReasoning) {
      expect(response.content.length).toBeGreaterThan(0);
    }
    return response;
  }

  const invokeMessages: BaseMessage[] = [
    new HumanMessage(
      'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    ),
  ];

  invokeMessages.push(await doInvoke(invokeMessages));
  invokeMessages.push(new HumanMessage('What is 42+7?'));

  // test a second time to make sure that we've got input translation working correctly
  await doInvoke(invokeMessages);
});

test('Test redacted thinking blocks multiturn streaming', async () => {
  const model = new ChatAnthropic({
    model: extendedThinkingModelName,
    maxTokens: 5000,
    thinking: { type: 'enabled', budget_tokens: 2000 },
  });

  async function doStreaming(messages: BaseMessage[]) {
    let full: AIMessageChunk | null = null;
    for await (const chunk of await model.stream(messages)) {
      full = full ? concat(full, chunk) : chunk;
    }
    expect(full).toBeInstanceOf(AIMessageChunk);
    expect(Array.isArray(full?.content)).toBe(true);
    let streamHasReasoning = false;
    const content = full?.content as AnthropicMessageResponse[];

    for (const block of content) {
      expect(typeof block).toBe('object');
      if (isRedactedThinkingBlock(block)) {
        streamHasReasoning = true;
        expect(Object.keys(block).sort()).toEqual(
          ['type', 'data', 'index'].sort()
        );
        expect(block.data).toBeTruthy();
        expect(typeof block.data).toBe('string');
      }
    }
    if (!streamHasReasoning) {
      expect(full?.content.length).toBeGreaterThan(0);
    }
    return full as AIMessageChunk;
  }

  const streamingMessages: BaseMessage[] = [
    new HumanMessage(
      'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    ),
  ];

  streamingMessages.push(await doStreaming(streamingMessages));
  streamingMessages.push(new HumanMessage('What is 42+7?'));

  // test a second time to make sure that we've got input translation working correctly
  await doStreaming(streamingMessages);
});

test('Can properly format messages with redacted thinking blocks', () => {
  const messageHistory = [
    new AIMessage({
      content: toLangChainContent([
        {
          type: 'redacted_thinking',
          data: 'encrypted-redacted-thinking-data',
        },
        {
          type: 'text',
          text: 'Continuing after redacted thinking.',
        },
      ] satisfies ChatAnthropicContentBlock[]),
    }),
  ];

  const formattedMessages = _convertMessagesToAnthropicPayload(messageHistory);

  expect(formattedMessages.messages).toHaveLength(1);
  expect(formattedMessages.messages[0].role).toBe('assistant');
  expect(formattedMessages.messages[0].content).toEqual([
    {
      type: 'redacted_thinking',
      data: 'encrypted-redacted-thinking-data',
    },
    {
      type: 'text',
      text: 'Continuing after redacted thinking.',
    },
  ]);
});

test('Can convert redacted thinking stream blocks', () => {
  const event: Anthropic.Beta.Messages.BetaRawMessageStreamEvent = {
    type: 'content_block_start',
    index: 0,
    content_block: {
      type: 'redacted_thinking',
      data: 'encrypted-redacted-thinking-data',
    },
  };
  const result = _makeMessageChunkFromAnthropicEvent(event, {
    streamUsage: true,
    coerceContentToString: false,
  });

  expect(result?.chunk.content).toEqual([
    {
      index: 0,
      type: 'redacted_thinking',
      data: 'encrypted-redacted-thinking-data',
    },
  ]);
  expect(result?.chunk.contentBlocks).toEqual([
    {
      type: 'non_standard',
      value: {
        index: 0,
        type: 'redacted_thinking',
        data: 'encrypted-redacted-thinking-data',
      },
    },
  ]);
});

test('Can handle google function calling blocks in content', async () => {
  const chat = new ChatAnthropic({
    modelName: 'claude-sonnet-4-5-20250929',
    maxRetries: 0,
  });
  const toolCallId = 'tool_call_id';
  const messages = [
    new SystemMessage("You're a helpful assistant"),
    new HumanMessage('What is the weather like in San Francisco?'),
    new AIMessage({
      content: toLangChainContent([
        {
          // Pass a content block with the `functionCall` object that Google returns.
          functionCall: {
            args: {
              location: 'san francisco',
            },
            name: 'get_weather',
          },
        } as MessageContentComplex,
      ]),
      tool_calls: [
        {
          id: toolCallId,
          name: 'get_weather',
          args: {
            location: 'san francisco',
          },
        },
      ],
    }),
    new ToolMessage({
      tool_call_id: toolCallId,
      content: 'The weather is sunny',
    }),
    new HumanMessage(
      'Give me a one sentence description of what the sky looks like.'
    ),
  ];
  const res = await chat.invoke(messages);
  expect(res.content.length).toBeGreaterThan(1);
});

describe('Opus 4.1', () => {
  test('works without passing any args', async () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-1',
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });

  test('works with streaming and thinking', async () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-1',
      thinking: {
        type: 'enabled',
        budget_tokens: 1024,
      },
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});

describe('Sonnet 4.5', () => {
  test('works without passing any args', async () => {
    const model = new ChatAnthropic({
      model: 'claude-sonnet-4-5-20250929',
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });

  test('works with streaming and thinking', async () => {
    const model = new ChatAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      thinking: {
        type: 'enabled',
        budget_tokens: 1024,
      },
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });

  test('works when passing topP arg', async () => {
    const model = new ChatAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      topP: 0.99,
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});

describe('Opus 4.5', () => {
  test('works without passing any args', async () => {
    const model = new ChatAnthropic({
      model: 'claude-opus-4-5',
    });

    const response = await model.invoke(
      'Please respond to this message simply with: Hello'
    );

    expect(response.content.length).toBeGreaterThan(0);
  });
});

test("won't modify structured output content if outputVersion is set", async () => {
  const schema = z.object({ name: z.string() });
  const model = new ChatAnthropic({
    model: 'claude-opus-4-1',
    outputVersion: 'v1',
  });

  const response = await model
    .withStructuredOutput(schema)
    .invoke("respond with the name 'John'");

  expect(response.name).toBeDefined();
});

describe('will work with native structured output', () => {
  const schema = z.object({ name: z.string() });

  test.each(['claude-opus-4-1', 'claude-sonnet-4-5-20250929'])(
    'works with %s',
    async (structuredOutputModelName) => {
      const model = new ChatAnthropic({
        model: structuredOutputModelName,
      });

      const response = await model
        .withStructuredOutput(schema, { method: 'jsonSchema' })
        .invoke("respond with the name 'John'");

      expect(response.name).toBeDefined();
    }
  );
});

describe('Anthropic Reasoning with contentBlocks', () => {
  test('invoke returns thinking as reasoning in contentBlocks', async () => {
    const model = new ChatAnthropic({
      model: extendedThinkingModelName,
      maxTokens: 5000,
      thinking: { type: 'enabled', budget_tokens: 2000 },
    });

    const result = await model.invoke('What is 2 + 2?');
    const blocks = result.contentBlocks;

    expect(blocks.length).toBeGreaterThan(0);

    const reasoningBlocks = blocks.filter(isReasoningContentBlock);
    expect(reasoningBlocks.length).toBeGreaterThan(0);
    expect(reasoningBlocks[0].reasoning.length).toBeGreaterThan(10);

    const textBlocks = blocks.filter((block) => block.type === 'text');
    expect(textBlocks.length).toBeGreaterThan(0);
  });

  test('stream returns thinking as reasoning in contentBlocks', async () => {
    const model = new ChatAnthropic({
      model: extendedThinkingModelName,
      maxTokens: 5000,
      thinking: { type: 'enabled', budget_tokens: 2000 },
    });

    let fullMessage: AIMessageChunk | null = null;
    for await (const chunk of await model.stream('What is 3 + 3?')) {
      fullMessage = fullMessage ? concat(fullMessage, chunk) : chunk;
    }

    expect(fullMessage).toBeDefined();

    const blocks = fullMessage!.contentBlocks;
    expect(blocks.length).toBeGreaterThan(0);

    const reasoningBlocks = blocks.filter(isReasoningContentBlock);
    expect(reasoningBlocks.length).toBeGreaterThan(0);
    expect(reasoningBlocks[0].reasoning.length).toBeGreaterThan(10);
  });
});

describe('Claude assistant prefill compatibility', () => {
  test.each([
    'claude-sonnet-4-6',
    'claude-sonnet-4-6@20260217',
    'claude-opus-4-7',
    'claude-opus-4-10',
    'global.anthropic.claude-opus-4-6-v1:0',
    'anthropic/claude-sonnet-4.6',
    'anthropic/claude-sonnet-4.12',
    'claude-sonnet-4-6[1m]',
    'claude-opus-5',
    'claude-sonnet-5',
    'claude-sonnet-5-5',
    'claude-opus-5-5',
    'claude-haiku-5-5',
    'claude-haiku-5.5',
    'claude-fable-5',
    'claude-fable-5-1',
    'claude-mythos-5',
    'claude-opus-6',
    'claude-opus-5-20260301',
    'claude-sonnet-5-5@20260801',
    'anthropic.claude-haiku-5-5',
    'global.anthropic.claude-opus-5-5-v1:0',
    'us.anthropic.claude-sonnet-5-v1:0',
  ])('detects %s as not supporting assistant prefill', (model) => {
    expect(modelDisallowsAssistantPrefill(model)).toBe(true);
  });

  test.each([
    'claude-sonnet-4-5-20250929',
    'claude-opus-4-20250514',
    'anthropic.claude-opus-4-20250514-v1:0',
    'claude-sonnet-4@20250514',
    'claude-opus-4-1-20250805',
    'claude-haiku-4-5',
    'claude-haiku-4-5-20251001',
    'claude-3-7-sonnet-20250219',
    'claude-3-5-haiku-latest',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'gpt-5.4',
    '',
    undefined,
  ])('leaves %s prefill support unchanged', (model) => {
    expect(modelDisallowsAssistantPrefill(model)).toBe(false);
  });

  test.each(['claude-opus-4-6', 'claude-haiku-5-5', 'claude-fable-5-1'])(
    'strips trailing assistant messages for %s requests',
    (model) => {
      const request = {
        model,
        max_tokens: 100,
        messages: [
          { role: 'user' as const, content: 'What changed?' },
          { role: 'assistant' as const, content: 'Draft prefill' },
          { role: 'assistant' as const, content: 'Another prefill' },
        ],
      };

      const sanitized = stripUnsupportedAssistantPrefill(request);

      expect(sanitized).not.toBe(request);
      expect(sanitized.messages).toEqual([
        { role: 'user', content: 'What changed?' },
      ]);
    }
  );

  test('does not strip assistant messages for older Claude models', () => {
    const request = {
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 100,
      messages: [
        { role: 'user' as const, content: 'Write JSON only.' },
        { role: 'assistant' as const, content: '{' },
      ],
    };

    expect(stripUnsupportedAssistantPrefill(request)).toBe(request);
  });
});

const opus46Model = 'claude-opus-4-6';

describe('Opus 4.6', () => {
  describe('Adaptive thinking', () => {
    test('invocationParams accepts adaptive thinking type', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      const params = model.invocationParams({});

      expect(params.thinking).toEqual({ type: 'adaptive' });
    });

    test('adaptive thinking disables temperature/topK/topP', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      const params = model.invocationParams({});

      expect(params.temperature).toBeUndefined();
      expect(params.top_k).toBeUndefined();
      expect(params.top_p).toBeUndefined();
    });

    test('adaptive thinking treats sampling sentinels as unset', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });
      model.temperature = -1;
      model.topP = -1;
      model.top_k = -1;

      const params = model.invocationParams({});

      expect(params.temperature).toBeUndefined();
      expect(params.top_k).toBeUndefined();
      expect(params.top_p).toBeUndefined();
    });

    test('adaptive thinking throws on non-default temperature', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        temperature: 0.5,
        apiKey: 'testing',
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      expect(() => model.invocationParams({})).toThrow(
        'temperature is not supported when thinking is enabled'
      );
    });

    test('adaptive thinking invoke', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      const result = await model.invoke('What is 15 * 23?');
      expect(result.content).toBeDefined();

      if (Array.isArray(result.content)) {
        const textBlocks = result.content.filter(isTextContentBlock);
        expect(textBlocks.length).toBeGreaterThan(0);
      } else {
        expect(typeof result.content).toBe('string');
        expect((result.content as string).length).toBeGreaterThan(0);
      }
    });

    test('adaptive thinking multiturn round-trip', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      const messages: BaseMessage[] = [new HumanMessage('Hello')];
      const response1 = await model.invoke(messages);
      messages.push(response1);
      messages.push(new HumanMessage('What is 42 + 7?'));

      const response2 = await model.invoke(messages);
      expect(response2.content).toBeDefined();
    });

    test('withStructuredOutput jsonSchema', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
      });

      const schema = z.object({
        answer: z.number().describe('The numeric answer'),
      });

      const structured = model.withStructuredOutput(schema, {
        method: 'jsonSchema',
      });

      const result = await structured.invoke('What is 2 + 2?');
      expect(result).toBeDefined();
      expect(typeof result.answer).toBe('number');
    });
  });

  describe('Effort parameter (outputConfig)', () => {
    test('invocationParams passes outputConfig from constructor', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        outputConfig: { effort: 'medium' },
      });

      const params = model.invocationParams({});

      expect(params.output_config).toEqual({ effort: 'medium' });
    });

    test('invocationParams passes outputConfig from call options', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({
        outputConfig: { effort: 'low' },
      });

      expect(params.output_config).toEqual({ effort: 'low' });
    });

    test('call-option outputConfig overrides constructor outputConfig', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        outputConfig: { effort: 'high' },
      });

      const params = model.invocationParams({
        outputConfig: { effort: 'low' },
      });

      expect(params.output_config).toEqual({ effort: 'low' });
    });

    test('effort max is accepted', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        outputConfig: { effort: 'max' },
      });

      const params = model.invocationParams({});

      expect(params.output_config).toEqual({ effort: 'max' });
    });

    test('output_config is undefined when not set', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({});

      expect(params.output_config).toBeUndefined();
    });

    test.each(['low', 'medium', 'high'] as const)(
      'invoke with %s effort',
      async (effort) => {
        const model = new ChatAnthropic({
          model: opus46Model,
          maxTokens: 4096,
          thinking: { type: 'adaptive' },
          outputConfig: { effort },
        });

        const result = await model.invoke('Say hello.');
        expect(result.content).toBeDefined();
      }
    );

    test.each(['low', 'medium', 'high'] as const)(
      'stream with %s effort',
      async (effort) => {
        const model = new ChatAnthropic({
          model: opus46Model,
          maxTokens: 4096,
          thinking: { type: 'adaptive' },
          outputConfig: { effort },
        });

        let full: AIMessageChunk | undefined;
        for await (const chunk of await model.stream('Say hello.')) {
          full = full ? concat(full, chunk) : chunk;
        }
        expect(full).toBeInstanceOf(AIMessageChunk);
        expect(full!.content).toBeDefined();
      }
    );

    test('via call options', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        thinking: { type: 'adaptive' },
      });

      const options: CustomAnthropicCallOptions = {
        outputConfig: { effort: 'low' },
      };
      const result = await model.invoke('Say hello.', options);
      expect(result.content).toBeDefined();
    });
  });

  describe('outputFormat to outputConfig.format migration', () => {
    test('deprecated outputFormat on call options maps to output_config.format', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({
        outputFormat: {
          type: 'json_schema',
          schema: { type: 'object' },
        },
      });

      expect(params.output_config).toEqual({
        format: {
          type: 'json_schema',
          schema: { type: 'object' },
        },
      });
    });

    test('outputConfig.format takes precedence over deprecated outputFormat', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({
        outputConfig: {
          format: {
            type: 'json_schema',
            schema: { type: 'object', properties: { a: { type: 'string' } } },
          },
        },
        outputFormat: {
          type: 'json_schema',
          schema: { type: 'object', properties: { b: { type: 'number' } } },
        },
      });

      expect(params.output_config?.format).toEqual({
        type: 'json_schema',
        schema: { type: 'object', properties: { a: { type: 'string' } } },
      });
    });

    test('effort and format can be combined in outputConfig', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        outputConfig: { effort: 'medium' },
      });

      const params = model.invocationParams({
        outputFormat: {
          type: 'json_schema',
          schema: { type: 'object' },
        },
      });

      expect(params.output_config).toEqual({
        effort: 'medium',
        format: {
          type: 'json_schema',
          schema: { type: 'object' },
        },
      });
    });
  });

  describe('Data residency (inferenceGeo)', () => {
    test('invocationParams passes inferenceGeo from constructor', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        inferenceGeo: 'us',
      });

      const params = model.invocationParams({});

      expect(params.inference_geo).toBe('us');
    });

    test('invocationParams passes inferenceGeo from call options', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({
        inferenceGeo: 'us',
      });

      expect(params.inference_geo).toBe('us');
    });

    test('call-option inferenceGeo overrides constructor inferenceGeo', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        inferenceGeo: 'eu',
      });

      const params = model.invocationParams({
        inferenceGeo: 'us',
      });

      expect(params.inference_geo).toBe('us');
    });

    test('inferenceGeo is undefined when not set', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
      });

      const params = model.invocationParams({});

      expect(params.inference_geo).toBeUndefined();
    });

    test('inferenceGeo accepted by API', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 256,
        inferenceGeo: 'us',
      });

      const result = await model.invoke('Say hello.');
      expect(result.content).toBeDefined();
    });
  });

  describe('Compaction API', () => {
    const compactionConfig: AnthropicContextManagementConfigParam = {
      edits: [
        {
          type: 'compact_20260112' as const,
          trigger: { type: 'input_tokens' as const, value: 50000 },
        },
      ],
    };

    function buildLongConversation(): BaseMessage[] {
      const padding =
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' +
        'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ' +
        'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris ' +
        'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in ' +
        'reprehenderit in voluptate velit esse cillum dolore eu fugiat. ';

      const messages: BaseMessage[] = [];
      for (let i = 0; i < 300; i++) {
        messages.push(new HumanMessage(`${padding} (message ${i})`));
        messages.push(new AIMessage(`Acknowledged message ${i}. ${padding}`));
      }
      messages.push(new HumanMessage('Summarize our conversation.'));
      return messages;
    }

    test('context_management passed through invocationParams', () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        apiKey: 'testing',
        maxTokens: 4096,
        contextManagement: compactionConfig,
      });

      const params = model.invocationParams({});

      const contextManagement = expectDefined(params.context_management);
      const edits = expectDefined(contextManagement.edits);
      expect(edits[0].type).toBe('compact_20260112');
    });

    test('accepted by API without triggering', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        contextManagement: compactionConfig,
      });

      const response = await model.invoke('Say hello.');
      expect(response.content).toBeDefined();
    });

    test('triggers compaction block (invoke)', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        contextManagement: compactionConfig,
      });

      const result = await model.invoke(buildLongConversation());

      const blocks = expectContentArray<ContentBlock>(result.content);
      const compactionBlock = expectDefined(blocks.find(isCompactionBlock));
      expect(typeof compactionBlock.content).toBe('string');
      expect(compactionBlock.content.length).toBeGreaterThan(0);
    });

    test('triggers compaction block (stream)', async () => {
      const model = new ChatAnthropic({
        model: opus46Model,
        maxTokens: 4096,
        contextManagement: compactionConfig,
      });

      let full: AIMessageChunk | undefined;
      for await (const chunk of await model.stream(buildLongConversation())) {
        full = full ? concat(full, chunk) : chunk;
      }

      expect(full).toBeInstanceOf(AIMessageChunk);
      const blocks = expectContentArray<ContentBlock>(full!.content);
      const compactionBlock = expectDefined(blocks.find(isCompactionBlock));
      expect(typeof compactionBlock.content).toBe('string');
      expect(compactionBlock.content.length).toBeGreaterThan(0);
    });

    test('Can properly format messages with compaction blocks', () => {
      const messageHistory = [
        new AIMessage({
          content: [
            {
              type: 'compaction',
              content:
                'Summary: The user asked about building a web scraper...',
            },
            {
              type: 'text',
              text: 'Based on our conversation so far, let me continue...',
            },
          ],
        }),
      ];

      const formattedMessages =
        _convertMessagesToAnthropicPayload(messageHistory);

      expect(formattedMessages.messages).toHaveLength(1);
      expect(formattedMessages.messages[0].role).toBe('assistant');
      expect(formattedMessages.messages[0].content).toHaveLength(2);

      const [compactionBlock, textBlock] =
        expectContentArray<Anthropic.Messages.ContentBlockParam>(
          formattedMessages.messages[0].content
        );
      expect(compactionBlock).toEqual({
        type: 'compaction',
        content: 'Summary: The user asked about building a web scraper...',
      });
      expect(textBlock).toEqual({
        type: 'text',
        text: 'Based on our conversation so far, let me continue...',
      });
    });
  });
});

describe('Tool input survives message conversion', () => {
  // Adapted from @langchain/anthropic's "converting messages doesn't drop tool input".
  // Guards the core >= 1.1.46 streaming-aggregation regression where the tool_use
  // content block's input was emptied (and re-serialization dropped it).
  test('converting messages does not drop tool input (live)', async () => {
    const jokeTool = {
      name: 'generate_random_joke',
      description: 'Generate a random joke.',
      schema: z.object({
        prompt: z.string().describe('The prompt to generate the joke for.'),
      }),
    };
    const model = new ChatAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      temperature: 0,
    }).bindTools([jokeTool]);

    const result = await model.invoke([
      new HumanMessage(
        'Generate three (3) random jokes. Use the generate_random_joke tool and call it three times before responding. This is very important.'
      ),
    ]);
    expect(result.tool_calls?.length ?? 0).toBeGreaterThan(0);

    const converted = _convertMessagesToAnthropicPayload([result]);
    const toolUseBlocks = (
      converted.messages[0].content as unknown as Array<Record<string, unknown>>
    ).filter((block) => block.type === 'tool_use');
    expect(toolUseBlocks.length).toBeGreaterThan(0);
    for (const block of toolUseBlocks) {
      expect(block.input).toBeDefined();
      expect((block.input as Record<string, unknown>).prompt).toBeDefined();
    }
  });
});

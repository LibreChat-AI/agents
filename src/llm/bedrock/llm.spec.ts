/* eslint-disable no-process-env */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import { expect, test, describe, jest } from '@jest/globals';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  SystemMessage,
  AIMessageChunk,
  type MessageContentComplex,
} from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type { ConverseResponse, Tool } from '@aws-sdk/client-bedrock-runtime';
import {
  convertConverseMessageToLangChainMessage,
  handleConverseStreamMetadata,
  convertToConverseMessages,
} from './utils';
import type { GraphTools } from '@/types';
import { toLangChainContent } from '@/messages/langchain';
import { CustomChatBedrockConverse, ServiceTierType } from './index';
import { partitionAndMarkBedrockToolCache } from './toolCache';

jest.setTimeout(120000);

// Base constructor args for tests
const baseConstructorArgs = {
  region: 'us-east-1',
  credentials: {
    secretAccessKey: 'test-secret-key',
    accessKeyId: 'test-access-key',
  },
};

type ConverseContentBlock = NonNullable<
  ReturnType<
    typeof convertToConverseMessages
  >['converseMessages'][number]['content']
>[number];
type BedrockVideoBlock = ConverseContentBlock & {
  video: {
    format?: string;
    source?: { bytes?: Uint8Array; s3Location?: { uri?: string } };
  };
};
type BedrockAudioBlock = ConverseContentBlock & {
  audio: {
    format?: string;
    source?: { bytes?: Uint8Array; s3Location?: { uri?: string } };
  };
};
type BedrockDocumentBlock = ConverseContentBlock & {
  document: {
    format?: string;
    name?: string;
  };
};

function expectVideoBlock(block: ConverseContentBlock): BedrockVideoBlock {
  expect(block).toHaveProperty('video');
  return block as BedrockVideoBlock;
}

function expectAudioBlock(block: ConverseContentBlock): BedrockAudioBlock {
  expect(block).toHaveProperty('audio');
  return block as BedrockAudioBlock;
}

function expectDocumentBlock(
  block: ConverseContentBlock
): BedrockDocumentBlock {
  expect(block).toHaveProperty('document');
  return block as BedrockDocumentBlock;
}

function humanMessageWithContent(
  content: MessageContentComplex[]
): HumanMessage {
  return new HumanMessage({ content: toLangChainContent(content) });
}

function getBedrockToolName(entry: Tool): string {
  if ('cachePoint' in entry) {
    return 'cachePoint';
  }
  return entry.toolSpec?.name ?? 'missing';
}

function getBedrockToolNames(tools: Tool[] | undefined): string[] {
  return (tools ?? []).map(getBedrockToolName);
}

describe('CustomChatBedrockConverse', () => {
  describe('applicationInferenceProfile parameter', () => {
    test('should initialize applicationInferenceProfile from constructor', () => {
      const testArn =
        'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        applicationInferenceProfile: testArn,
      });
      expect(model.model).toBe('anthropic.claude-3-haiku-20240307-v1:0');
      expect(model.applicationInferenceProfile).toBe(testArn);
    });

    test('should be undefined when not provided in constructor', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
      });
      expect(model.model).toBe('anthropic.claude-3-haiku-20240307-v1:0');
      expect(model.applicationInferenceProfile).toBeUndefined();
    });

    test('should send applicationInferenceProfile as modelId in ConverseCommand when provided', async () => {
      const testArn =
        'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
      const mockSend = jest.fn<any>().mockResolvedValue({
        output: {
          message: {
            role: 'assistant',
            content: [{ text: 'Test response' }],
          },
        },
        stopReason: 'end_turn',
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          totalTokens: 15,
        },
      });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        applicationInferenceProfile: testArn,
        client: mockClient,
      });

      await model.invoke([new HumanMessage('Hello')]);

      expect(mockSend).toHaveBeenCalledTimes(1);
      const commandArg = mockSend.mock.calls[0][0] as {
        input: { modelId: string };
      };
      expect(commandArg.input.modelId).toBe(testArn);
      expect(commandArg.input.modelId).not.toBe(
        'anthropic.claude-3-haiku-20240307-v1:0'
      );
    });

    test('should send model as modelId in ConverseCommand when applicationInferenceProfile is not provided', async () => {
      const mockSend = jest.fn<any>().mockResolvedValue({
        output: {
          message: {
            role: 'assistant',
            content: [{ text: 'Test response' }],
          },
        },
        stopReason: 'end_turn',
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          totalTokens: 15,
        },
      });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        client: mockClient,
      });

      await model.invoke([new HumanMessage('Hello')]);

      expect(mockSend).toHaveBeenCalledTimes(1);
      const commandArg = mockSend.mock.calls[0][0] as {
        input: { modelId: string };
      };
      expect(commandArg.input.modelId).toBe(
        'anthropic.claude-3-haiku-20240307-v1:0'
      );
    });

    test('should send applicationInferenceProfile as modelId in ConverseStreamCommand when provided', async () => {
      const testArn =
        'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
      const mockSend = jest
        .fn<
          (command: ConverseStreamCommand) => Promise<{
            stream: AsyncIterable<{
              contentBlockDelta: {
                contentBlockIndex: number;
                delta: { text: string };
              };
            }>;
          }>
        >()
        .mockResolvedValue({
          stream: (async function* streamChunks() {
            yield {
              contentBlockDelta: {
                contentBlockIndex: 0,
                delta: { text: 'Test response' },
              },
            };
          })(),
        });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        applicationInferenceProfile: testArn,
        client: mockClient,
        _lc_stream_delay: 0,
      });

      let chunks = 0;
      for await (const _chunk of await model.stream([
        new HumanMessage('Hello'),
      ])) {
        chunks += 1;
      }

      expect(mockSend).toHaveBeenCalledTimes(1);
      expect(chunks).toBe(1);
      const commandArg = mockSend.mock.calls[0][0] as ConverseStreamCommand;
      expect(commandArg).toBeInstanceOf(ConverseStreamCommand);
      expect(commandArg.input.modelId).toBe(testArn);
      expect(commandArg.input.modelId).not.toBe(
        'anthropic.claude-3-haiku-20240307-v1:0'
      );
    });
  });

  describe('guardrailConfig configuration', () => {
    test('should pass guardrailConfig through to ConverseStreamCommand', async () => {
      const guardrailConfig = {
        guardrailIdentifier: 'test-guardrail-id',
        guardrailVersion: 'DRAFT',
        trace: 'enabled_full' as const,
        streamProcessingMode: 'sync' as const,
      };
      const mockSend = jest.fn<any>().mockResolvedValue({
        stream: (async function* streamChunks() {
          yield {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { text: 'Guardrail response' },
            },
          };
        })(),
      });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        guardrailConfig,
        client: mockClient,
        _lc_stream_delay: 0,
      });

      let chunks = 0;
      for await (const _chunk of await model.stream([
        new HumanMessage('Hello'),
      ])) {
        chunks += 1;
      }

      expect(mockSend).toHaveBeenCalledTimes(1);
      expect(chunks).toBe(1);
      const commandArg = mockSend.mock.calls[0][0] as ConverseStreamCommand;
      expect(commandArg).toBeInstanceOf(ConverseStreamCommand);
      expect(commandArg.input.guardrailConfig).toEqual(guardrailConfig);
    });
  });

  describe('serviceTier configuration', () => {
    test('should set serviceTier in constructor', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'priority',
      });
      expect(model.serviceTier).toBe('priority');
    });

    test('should set serviceTier as undefined when not provided', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
      });
      expect(model.serviceTier).toBeUndefined();
    });

    test.each(['priority', 'default', 'flex', 'reserved'])(
      'should include serviceTier in invocationParams when set to %s',
      (serviceTier) => {
        const model = new CustomChatBedrockConverse({
          ...baseConstructorArgs,
          serviceTier: serviceTier as ServiceTierType,
        });
        const params = model.invocationParams({});
        expect(params.serviceTier).toEqual({ type: serviceTier });
      }
    );

    test('should not include serviceTier in invocationParams when not set', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
      });
      const params = model.invocationParams({});
      expect(params.serviceTier).toBeUndefined();
    });

    test('should override serviceTier from call options in invocationParams', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'default',
      });
      const params = model.invocationParams({
        serviceTier: 'priority',
      });
      expect(params.serviceTier).toEqual({ type: 'priority' });
    });

    test('should use class-level serviceTier when call options do not override it', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'flex',
      });
      const params = model.invocationParams({});
      expect(params.serviceTier).toEqual({ type: 'flex' });
    });

    test('should handle serviceTier in invocationParams with other config options', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'reserved',
        temperature: 0.5,
        maxTokens: 100,
      });
      const params = model.invocationParams({
        stop: ['stop_sequence'],
      });
      expect(params.serviceTier).toEqual({ type: 'reserved' });
      expect(params.inferenceConfig?.temperature).toBe(0.5);
      expect(params.inferenceConfig?.maxTokens).toBe(100);
      expect(params.inferenceConfig?.stopSequences).toEqual(['stop_sequence']);
    });
  });

  describe('promptCache tool configuration', () => {
    test('adds a Bedrock cache point for directly-bound tools', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        promptCache: true,
      });

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'direct_tool',
        'cachePoint',
      ]);
    });

    test('defaults the tool cache point to the 1h extended TTL', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        promptCache: true,
      });

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      const toolList = (params.toolConfig?.tools ?? []) as unknown as Array<
        Record<string, unknown>
      >;
      const cachePoints = toolList.filter((t) => 'cachePoint' in t);
      expect(cachePoints).toHaveLength(1);
      expect((cachePoints[0] as { cachePoint: unknown }).cachePoint).toEqual({
        type: 'default',
        ttl: '1h',
      });
    });

    test('honors an explicit 5m promptCacheTtl on the tool cache point', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        promptCache: true,
        promptCacheTtl: '5m',
      });

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      const toolList = (params.toolConfig?.tools ?? []) as unknown as Array<
        Record<string, unknown>
      >;
      const cachePoints = toolList.filter((t) => 'cachePoint' in t);
      expect(cachePoints).toHaveLength(1);
      // 5m omits the ttl field (provider default).
      expect((cachePoints[0] as { cachePoint: unknown }).cachePoint).toEqual({
        type: 'default',
      });
    });

    test('adds the Bedrock cache point before deferred tools', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        promptCache: true,
      });
      const tools = partitionAndMarkBedrockToolCache(
        [
          {
            type: 'function',
            function: {
              name: 'static_tool',
              description: 'Static tool',
              parameters: { type: 'object', properties: {} },
            },
          },
          {
            type: 'function',
            function: {
              name: 'deferred_tool',
              description: 'Deferred tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ] as GraphTools,
        (name) => name === 'deferred_tool'
      );

      const params = model.invocationParams({ tools });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'static_tool',
        'cachePoint',
        'deferred_tool',
      ]);
      expect(JSON.stringify(params.toolConfig?.tools)).not.toContain(
        '__lc_bedrock_cache_point_after'
      );
    });

    test('does not fall back to caching when Graph marks all tools deferred', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        promptCache: true,
      });
      const tools = partitionAndMarkBedrockToolCache(
        [
          {
            type: 'function',
            function: {
              name: 'deferred_tool',
              description: 'Deferred tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ] as GraphTools,
        () => true
      );

      const params = model.invocationParams({ tools });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'deferred_tool',
      ]);
      expect(JSON.stringify(params.toolConfig?.tools)).not.toContain(
        '__lc_bedrock_skip_tool_cache'
      );
    });

    test('does not add a tool cache point for Amazon Nova models', () => {
      // Regression for danny-avila/LibreChat#13838: Nova rejects a cachePoint in
      // toolConfig.tools ("Malformed input request: #/toolConfig/tools/0:
      // extraneous key [cachePoint] is not permitted"). Only the tool checkpoint
      // is Claude-only — Nova still caches system/messages — so promptCache: true
      // on Nova must omit just the tool cache point.
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'us.amazon.nova-pro-v1:0',
        promptCache: true,
      });

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'direct_tool',
      ]);
      expect(JSON.stringify(params.toolConfig?.tools ?? [])).not.toContain(
        'cachePoint'
      );
    });

    test('still adds a tool cache point for a Claude model behind an inference profile', () => {
      // The configured Claude model id is captured at construction, so the gate
      // survives the application-inference-profile ARN swap during generation.
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
        applicationInferenceProfile:
          'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test',
        promptCache: true,
      });

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'direct_tool',
        'cachePoint',
      ]);
    });

    test('still adds a tool cache point when model is omitted (default is Claude)', () => {
      // When no model is provided, LangChain initializes this.model to a default
      // Claude model, so the tool cache point must still apply.
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        promptCache: true,
      });

      expect(model.model).toMatch(/claude/i);

      const params = model.invocationParams({
        tools: [
          {
            type: 'function',
            function: {
              name: 'direct_tool',
              description: 'Direct tool',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
      });

      expect(getBedrockToolNames(params.toolConfig?.tools)).toEqual([
        'direct_tool',
        'cachePoint',
      ]);
    });
  });

  describe('contentBlockIndex cleanup', () => {
    // Access private methods for testing via any cast
    function getModelWithCleanMethods() {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
      });
      return model as any;
    }

    test('should remove contentBlockIndex from top level', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        contentBlockIndex: 0,
        text: 'hello',
        other: 'data',
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({ text: 'hello', other: 'data' });
      expect(cleaned.contentBlockIndex).toBeUndefined();
    });

    test('should remove contentBlockIndex from nested objects', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        outer: {
          contentBlockIndex: 1,
          inner: {
            contentBlockIndex: 2,
            data: 'test',
          },
        },
        topLevel: 'value',
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({
        outer: {
          inner: {
            data: 'test',
          },
        },
        topLevel: 'value',
      });
    });

    test('should handle arrays when removing contentBlockIndex', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        items: [
          { contentBlockIndex: 0, text: 'first' },
          { contentBlockIndex: 1, text: 'second' },
        ],
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({
        items: [{ text: 'first' }, { text: 'second' }],
      });
    });

    test('should preserve null and undefined values', () => {
      const model = getModelWithCleanMethods();

      expect(model.removeContentBlockIndex(null)).toBeNull();
      expect(model.removeContentBlockIndex(undefined)).toBeUndefined();
    });

    test('enrichChunk should strip contentBlockIndex from response_metadata', () => {
      const model = getModelWithCleanMethods();

      const chunkWithIndex = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            contentBlockIndex: 0,
            stopReason: null,
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithIndex, new Set([0]));

      expect(enriched.message.response_metadata).toEqual({
        stopReason: null,
      });
      expect(
        (enriched.message.response_metadata as any).contentBlockIndex
      ).toBeUndefined();
      expect(enriched.text).toBe('Hello');
    });

    test('enrichChunk should pass through chunks without contentBlockIndex unchanged', () => {
      const model = getModelWithCleanMethods();

      const chunkWithoutIndex = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            stopReason: 'end_turn',
            usage: { inputTokens: 10, outputTokens: 5 },
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithoutIndex, new Set());

      expect(enriched.message.response_metadata).toEqual({
        stopReason: 'end_turn',
        usage: { inputTokens: 10, outputTokens: 5 },
      });
    });

    test('enrichChunk should inject index on array content blocks', () => {
      const model = getModelWithCleanMethods();

      const chunkWithArrayContent = new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: [
            {
              type: 'reasoning_content',
              reasoningText: { text: 'thinking...' },
            },
          ],
          response_metadata: {
            contentBlockIndex: 0,
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithArrayContent, new Set([0]));

      expect(Array.isArray(enriched.message.content)).toBe(true);
      const blocks = enriched.message.content as any[];
      expect(blocks[0].index).toBe(0);
      expect(blocks[0].type).toBe('reasoning_content');
      expect(
        (enriched.message.response_metadata as any).contentBlockIndex
      ).toBeUndefined();
    });

    test('enrichChunk should promote text to array when multiple block indices seen', () => {
      const model = getModelWithCleanMethods();

      const textChunk = new ChatGenerationChunk({
        text: 'Hello world',
        message: new AIMessageChunk({
          content: 'Hello world',
          response_metadata: {
            contentBlockIndex: 1,
          },
        }),
      });

      const enriched = model.enrichChunk(textChunk, new Set([0, 1]));

      expect(Array.isArray(enriched.message.content)).toBe(true);
      const blocks = enriched.message.content as any[];
      expect(blocks).toHaveLength(1);
      expect(blocks[0]).toEqual({
        type: 'text',
        text: 'Hello world',
        index: 1,
      });
    });

    test('enrichChunk should keep text as string when only one block index seen', () => {
      const model = getModelWithCleanMethods();

      const textChunk = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            contentBlockIndex: 0,
            stopReason: null,
          },
        }),
      });

      const enriched = model.enrichChunk(textChunk, new Set([0]));

      expect(typeof enriched.message.content).toBe('string');
      expect(enriched.message.content).toBe('Hello');
    });

    test('enrichChunk should strip deeply nested contentBlockIndex from response_metadata', () => {
      const model = getModelWithCleanMethods();

      const chunkWithNestedIndex = new ChatGenerationChunk({
        text: 'Test',
        message: new AIMessageChunk({
          content: 'Test',
          response_metadata: {
            contentBlockIndex: 0,
            amazon: {
              bedrock: {
                contentBlockIndex: 0,
                trace: { something: 'value' },
              },
            },
            otherData: 'preserved',
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithNestedIndex, new Set([0]));

      expect(enriched.message.response_metadata).toEqual({
        amazon: {
          bedrock: {
            trace: { something: 'value' },
          },
        },
        otherData: 'preserved',
      });
    });
  });
});

describe('handleConverseStreamMetadata - cache token extraction', () => {
  test('should extract cacheReadInputTokens and cacheWriteInputTokens into input_token_details', () => {
    const metadata = {
      usage: {
        inputTokens: 13,
        outputTokens: 5,
        totalTokens: 10849,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      },
      metrics: { latencyMs: 1000 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toEqual({
      input_tokens: 13,
      output_tokens: 5,
      total_tokens: 10849,
      input_token_details: {
        cache_read: 10831,
        cache_creation: 0,
      },
    });
  });

  test('should not include input_token_details when no cache tokens present', () => {
    const metadata = {
      usage: {
        inputTokens: 100,
        outputTokens: 50,
        totalTokens: 150,
      },
      metrics: { latencyMs: 500 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(msg.usage_metadata?.input_token_details).toBeUndefined();
  });

  test('should include input_token_details when only cacheWriteInputTokens is present', () => {
    const metadata = {
      usage: {
        inputTokens: 50,
        outputTokens: 10,
        totalTokens: 10060,
        cacheWriteInputTokens: 10000,
      },
      metrics: { latencyMs: 800 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata?.input_token_details).toEqual({
      cache_read: 0,
      cache_creation: 10000,
    });
  });

  test('should return undefined usage_metadata when streamUsage is false', () => {
    const metadata = {
      usage: {
        inputTokens: 13,
        outputTokens: 5,
        totalTokens: 10849,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      },
      metrics: { latencyMs: 1000 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: false,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toBeUndefined();
  });
});

describe('convertConverseMessageToLangChainMessage - cache token extraction', () => {
  const makeResponseMetadata = (
    usage: Record<string, number>
  ): Omit<ConverseResponse, 'output'> =>
    ({
      usage,
      stopReason: 'end_turn',
      metrics: undefined,
      $metadata: { requestId: 'test-id' },
    }) as unknown as Omit<ConverseResponse, 'output'>;

  test('should extract cache tokens in non-streaming response', () => {
    const message = {
      role: 'assistant' as const,
      content: [{ text: 'Hello!' }],
    };

    const result = convertConverseMessageToLangChainMessage(
      message,
      makeResponseMetadata({
        inputTokens: 20,
        outputTokens: 5,
        totalTokens: 10856,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      })
    );

    expect(result.usage_metadata).toEqual({
      input_tokens: 10851,
      output_tokens: 5,
      total_tokens: 10856,
      input_token_details: {
        cache_read: 10831,
        cache_creation: 0,
      },
    });
  });

  test('should not include input_token_details when no cache tokens in non-streaming response', () => {
    const message = {
      role: 'assistant' as const,
      content: [{ text: 'Hello!' }],
    };

    const result = convertConverseMessageToLangChainMessage(
      message,
      makeResponseMetadata({
        inputTokens: 100,
        outputTokens: 50,
        totalTokens: 150,
      })
    );

    expect(result.usage_metadata).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(result.usage_metadata?.input_token_details).toBeUndefined();
  });
});

describe('convertToConverseMessages', () => {
  test('should convert basic messages', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an AI assistant."),
      new HumanMessage('Hello!'),
    ]);

    expect(converseSystem).toEqual([{ text: "You're an AI assistant." }]);
    expect(converseMessages).toHaveLength(1);
    expect(converseMessages[0].role).toBe('user');
    expect(converseMessages[0].content).toEqual([{ text: 'Hello!' }]);
  });

  test('should handle standard v1 format with tool_call blocks (e.g., from Anthropic provider)', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an advanced AI assistant."),
      new HumanMessage("What's the weather in SF?"),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me check the weather for you.' },
          {
            type: 'tool_call',
            id: 'call_123',
            name: 'get_weather',
            args: { location: 'San Francisco' },
          },
        ],
        response_metadata: {
          output_version: 'v1',
          model_provider: 'anthropic',
        },
      }),
      new ToolMessage({
        tool_call_id: 'call_123',
        content: '72°F and sunny',
      }),
    ]);

    expect(converseSystem).toEqual([
      { text: "You're an advanced AI assistant." },
    ]);
    expect(converseMessages).toHaveLength(3);

    // Check user message
    expect(converseMessages[0].role).toBe('user');
    expect(converseMessages[0].content).toEqual([
      { text: "What's the weather in SF?" },
    ]);

    // Check AI message with tool use
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toHaveLength(2);
    expect(converseMessages[1].content?.[0]).toEqual({
      text: 'Let me check the weather for you.',
    });
    expect(converseMessages[1].content?.[1]).toEqual({
      toolUse: {
        toolUseId: 'call_123',
        name: 'get_weather',
        input: { location: 'San Francisco' },
      },
    });

    // Check tool result
    expect(converseMessages[2].role).toBe('user');
    expect(converseMessages[2].content).toHaveLength(1);
    expect((converseMessages[2].content?.[0] as any).toolResult).toBeDefined();
    expect((converseMessages[2].content?.[0] as any).toolResult.toolUseId).toBe(
      'call_123'
    );
  });

  test('should handle standard v1 format with reasoning blocks (e.g., from Anthropic provider)', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an advanced AI assistant."),
      new HumanMessage('What is 2+2?'),
      new AIMessage({
        content: [
          {
            type: 'reasoning',
            reasoning: 'I need to add 2 and 2 together.',
          },
          { type: 'text', text: 'The answer is 4.' },
        ],
        response_metadata: {
          output_version: 'v1',
          model_provider: 'anthropic',
        },
      }),
      new HumanMessage('Thanks! What about 3+3?'),
    ]);

    expect(converseSystem).toEqual([
      { text: "You're an advanced AI assistant." },
    ]);
    expect(converseMessages).toHaveLength(3);

    // Check AI message with reasoning
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toHaveLength(2);
    expect(
      (converseMessages[1].content?.[0] as any).reasoningContent
    ).toBeDefined();
    expect(
      (converseMessages[1].content?.[0] as any).reasoningContent.reasoningText
        .text
    ).toBe('I need to add 2 and 2 together.');
    expect(converseMessages[1].content?.[1]).toEqual({
      text: 'The answer is 4.',
    });
  });

  test('should handle messages without v1 format', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('Hello'),
      new AIMessage({
        content: 'Hi there!',
        tool_calls: [],
      }),
    ]);

    expect(converseMessages).toHaveLength(2);
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toEqual([{ text: 'Hi there!' }]);
  });

  test('should combine consecutive tool result messages', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('Get weather for SF and NYC'),
      new AIMessage({
        content: 'I will check both cities.',
        tool_calls: [
          { id: 'call_1', name: 'get_weather', args: { city: 'SF' } },
          { id: 'call_2', name: 'get_weather', args: { city: 'NYC' } },
        ],
      }),
      new ToolMessage({
        tool_call_id: 'call_1',
        content: 'SF: 72°F',
      }),
      new ToolMessage({
        tool_call_id: 'call_2',
        content: 'NYC: 65°F',
      }),
    ]);

    // Tool messages should be combined into one user message
    expect(converseMessages).toHaveLength(3);
    const toolResultMessage = converseMessages[2];
    expect(toolResultMessage.role).toBe('user');
    expect(toolResultMessage.content).toHaveLength(2);
    expect((toolResultMessage.content?.[0] as any).toolResult.toolUseId).toBe(
      'call_1'
    );
    expect((toolResultMessage.content?.[1] as any).toolResult.toolUseId).toBe(
      'call_2'
    );
  });

  describe('video content block conversion', () => {
    test('converts multimodal video block with base64 data', () => {
      const videoData = btoa('fake-video-bytes');
      const result = convertToConverseMessages([
        humanMessageWithContent([
          { type: 'text', text: 'Describe this video' },
          {
            type: 'video',
            mimeType: 'video/mp4',
            data: videoData,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      expect(content).toHaveLength(2);
      expect(content[0]).toEqual({ text: 'Describe this video' });
      const videoBlock = expectVideoBlock(content[1]);
      expect(videoBlock.video.format).toBe('mp4');
      expect(videoBlock.video.source?.bytes).toBeInstanceOf(Uint8Array);
    });

    test('converts multimodal video block with Uint8Array data', () => {
      const videoBytes = new Uint8Array([1, 2, 3, 4]);
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'video',
            mimeType: 'video/webm',
            data: videoBytes,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const videoBlock = expectVideoBlock(content[0]);
      expect(videoBlock.video.format).toBe('webm');
      expect(videoBlock.video.source?.bytes).toBe(videoBytes);
    });

    test('passes through native Bedrock video block', () => {
      const videoSource = {
        bytes: new Uint8Array([1, 2, 3]),
      };
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'video',
            video: {
              format: 'mp4',
              source: videoSource,
            },
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const videoBlock = expectVideoBlock(content[0]);
      expect(videoBlock.video.format).toBe('mp4');
      expect(videoBlock.video.source).toBe(videoSource);
    });

    test('converts video block with S3 fileId', () => {
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'video',
            mimeType: 'video/mp4',
            fileId: 's3://my-bucket/my-video.mp4',
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const videoBlock = expectVideoBlock(content[0]);
      expect(videoBlock.video.format).toBe('mp4');
      expect(videoBlock.video.source?.s3Location?.uri).toBe(
        's3://my-bucket/my-video.mp4'
      );
    });
  });

  describe('audio content block conversion', () => {
    test('converts multimodal audio block with base64 data', () => {
      const audioData = btoa('fake-audio-bytes');
      const result = convertToConverseMessages([
        humanMessageWithContent([
          { type: 'text', text: 'Transcribe this audio' },
          {
            type: 'audio',
            mimeType: 'audio/mp3',
            data: audioData,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      expect(content).toHaveLength(2);
      expect(content[0]).toEqual({ text: 'Transcribe this audio' });
      const audioBlock = expectAudioBlock(content[1]);
      expect(audioBlock.audio.format).toBe('mp3');
      expect(audioBlock.audio.source?.bytes).toBeInstanceOf(Uint8Array);
    });

    test('converts multimodal audio block with Uint8Array data', () => {
      const audioBytes = new Uint8Array([1, 2, 3, 4]);
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'audio',
            mimeType: 'audio/wav',
            data: audioBytes,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const audioBlock = expectAudioBlock(content[0]);
      expect(audioBlock.audio.format).toBe('wav');
      expect(audioBlock.audio.source?.bytes).toBe(audioBytes);
    });

    test('passes through native Bedrock audio block', () => {
      const audioSource = {
        bytes: new Uint8Array([1, 2, 3]),
      };
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'audio',
            audio: {
              format: 'flac',
              source: audioSource,
            },
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const audioBlock = expectAudioBlock(content[0]);
      expect(audioBlock.audio.format).toBe('flac');
      expect(audioBlock.audio.source).toBe(audioSource);
    });

    test('converts audio block with S3 fileId', () => {
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'audio',
            mimeType: 'audio/mp3',
            fileId: 's3://my-bucket/my-audio.mp3',
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      const audioBlock = expectAudioBlock(content[0]);
      expect(audioBlock.audio.format).toBe('mp3');
      expect(audioBlock.audio.source?.s3Location?.uri).toBe(
        's3://my-bucket/my-audio.mp3'
      );
    });
  });

  describe('document content block conversion', () => {
    test('imputes placeholder filename when no name is provided', () => {
      const pdfData = btoa('fake-pdf-bytes');
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: pdfData,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      expect(content).toHaveLength(2);
      // Converse requires a text block alongside document blocks; it is appended last.
      expect(content[1]).toEqual({ text: '_' });
      const documentBlock = expectDocumentBlock(content[0]);
      expect(documentBlock.document.format).toBe('pdf');
      expect(documentBlock.document.name).toBeDefined();
      expect(typeof documentBlock.document.name).toBe('string');
      expect(documentBlock.document.name?.length).toBe(12);
    });

    test('uses provided filename from metadata', () => {
      const pdfData = btoa('fake-pdf-bytes');
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: pdfData,
            metadata: { filename: 'my-report.pdf' },
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      expect(expectDocumentBlock(content[0]).document.name).toBe(
        'my-report.pdf'
      );
    });

    test('generates unique placeholder filenames for multiple files', () => {
      const pdfData = btoa('fake-pdf-bytes');
      const result = convertToConverseMessages([
        humanMessageWithContent([
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: pdfData,
          } as MessageContentComplex,
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: pdfData,
          } as MessageContentComplex,
        ]),
      ]);

      const content = result.converseMessages[0].content!;
      expect(content).toHaveLength(3);
      // Converse requires a text block alongside document blocks; it is appended last.
      expect(content[2]).toEqual({ text: '_' });
      const firstDocument = expectDocumentBlock(content[0]);
      const secondDocument = expectDocumentBlock(content[1]);
      expect(firstDocument.document.name).toBeDefined();
      expect(secondDocument.document.name).toBeDefined();
      expect(firstDocument.document.name).not.toBe(
        secondDocument.document.name
      );
    });
  });
});

// Integration tests (require AWS credentials)
describe.skip('Integration tests', () => {
  const integrationArgs = {
    region: process.env.BEDROCK_AWS_REGION ?? 'us-east-1',
    credentials: {
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
    },
  };

  test('basic invoke', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxRetries: 0,
    });
    const message = new HumanMessage('Hello!');
    const res = await model.invoke([message]);
    expect(res.response_metadata.usage).toBeDefined();
  });

  test('basic streaming', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxRetries: 0,
    });

    let fullMessage: AIMessageChunk | undefined;
    for await (const chunk of await model.stream('Hello!')) {
      fullMessage = fullMessage ? concat(fullMessage, chunk) : chunk;
    }

    expect(fullMessage).toBeDefined();
    expect(fullMessage?.content).toBeDefined();
  });

  test('with thinking/reasoning enabled', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
      maxTokens: 5000,
      additionalModelRequestFields: {
        thinking: { type: 'enabled', budget_tokens: 2000 },
      },
    });

    const result = await model.invoke('What is 2 + 2?');
    expect(result.content).toBeDefined();

    // Should have reasoning content if the model supports it
    if (Array.isArray(result.content)) {
      const reasoningBlocks = result.content.filter(
        (b: any) => b.type === 'reasoning_content' || b.type === 'reasoning'
      );
      expect(reasoningBlocks.length).toBeGreaterThanOrEqual(0);
    }
  });

  test('cache tokens should populate input_token_details', async () => {
    const client = new BedrockRuntimeClient({
      region: integrationArgs.region,
      credentials: integrationArgs.credentials,
    });

    // Large system prompt (>1024 tokens) to meet Bedrock's minimum cache threshold
    const largeSystemPrompt = [
      'You are an expert assistant.',
      ...Array(200).fill(
        'This is padding content to exceed the minimum token threshold for Bedrock prompt caching. '
      ),
      'When answering, be brief and direct.',
    ].join(' ');

    const systemBlocks = [
      { text: largeSystemPrompt },
      { cachePoint: { type: 'default' as const } },
    ];

    const converseArgs = {
      modelId: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
      system: systemBlocks,
      inferenceConfig: { maxTokens: 50 },
    };

    // Call 1: populate the cache (may be a write or read if already warm)
    await client.send(
      new ConverseCommand({
        ...converseArgs,
        messages: [{ role: 'user', content: [{ text: 'Say hello.' }] }],
      })
    );

    // Call 2: should read from cache — this is the one we assert on
    const response = await client.send(
      new ConverseCommand({
        ...converseArgs,
        messages: [
          { role: 'user', content: [{ text: 'Say hello.' }] },
          { role: 'assistant', content: [{ text: 'Hello!' }] },
          { role: 'user', content: [{ text: 'Say goodbye.' }] },
        ],
      })
    );

    // Feed raw response through convertConverseMessageToLangChainMessage
    const result = convertConverseMessageToLangChainMessage(
      response.output!.message!,
      response
    );

    expect(result.usage_metadata).toBeDefined();
    expect(result.usage_metadata!.input_tokens).toBeGreaterThan(0);
    expect(result.usage_metadata!.output_tokens).toBeGreaterThan(0);

    // Cache should have been populated by call 1, so call 2 should show cache reads
    expect(result.usage_metadata!.input_token_details).toBeDefined();
    expect(
      result.usage_metadata!.input_token_details!.cache_read
    ).toBeGreaterThan(0);
  });
});

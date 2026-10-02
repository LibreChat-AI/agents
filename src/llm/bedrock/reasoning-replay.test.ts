import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type { ConverseCommandOutput } from '@aws-sdk/client-bedrock-runtime';
import { CustomChatBedrockConverse } from './index';

const claudeHistory = () => [
  new HumanMessage('Hello'),
  new AIMessage({
    content: [
      {
        type: 'reasoning_content',
        reasoningText: {
          text: 'Claude reasoning',
          signature: 'claude-signature',
        },
      },
      { type: 'text', text: 'Hello back' },
    ],
  }),
  new HumanMessage('Continue'),
];

const profileArn =
  'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123';
const nonStreamingResponse: ConverseCommandOutput = {
  $metadata: { requestId: 'non-stream-request' },
  output: {
    message: {
      role: 'assistant',
      content: [{ text: 'Done.' }],
    },
  },
  stopReason: 'end_turn',
  usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
  metrics: { latencyMs: 0 },
};

const structuredResponse: ConverseCommandOutput = {
  $metadata: { requestId: 'structured-response-request' },
  output: {
    message: {
      role: 'assistant',
      content: [
        {
          reasoningContent: {
            reasoningText: {
              text: 'Returned reasoning',
              signature: 'returned-signature',
            },
          },
        },
        {
          citationsContent: {
            content: [{ text: 'Cited answer.' }],
            citations: [
              {
                sourceContent: [{ text: 'Source passage' }],
                location: {
                  documentChar: {
                    documentIndex: 0,
                    start: 0,
                    end: 13,
                  },
                },
              },
            ],
          },
        },
        { cachePoint: { type: 'default', ttl: '1h' } },
        {
          document: {
            format: 'txt',
            name: 'Reference document',
            source: { text: 'Document body' },
          },
        },
        {
          guardContent: { text: { text: 'Guarded text' } },
        },
        {
          image: {
            format: 'png',
            source: { bytes: new Uint8Array([1, 2, 3]) },
          },
        },
        {
          toolResult: {
            toolUseId: 'previous-tool-call',
            content: [{ text: 'Tool result' }],
          },
        },
        {
          video: {
            format: 'mp4',
            source: { bytes: new Uint8Array([4, 5, 6]) },
          },
        },
        { text: 'Visible answer' },
        {
          toolUse: {
            toolUseId: 'response-tool-use',
            name: 'lookup',
            input: { query: 'weather' },
          },
        },
        { $unknown: ['futureBlock', { value: 'preserve me' }] },
      ],
    },
  },
  stopReason: 'tool_use',
  usage: {
    inputTokens: 100,
    outputTokens: 9,
    totalTokens: 130,
    cacheReadInputTokens: 17,
    cacheWriteInputTokens: 4,
  },
  metrics: { latencyMs: 42 },
  additionalModelResponseFields: { provider: 'bedrock' },
};

function requireConverseCommand(command: unknown): ConverseCommand {
  if (!(command instanceof ConverseCommand)) {
    throw new Error('Expected invoke to send a ConverseCommand');
  }
  return command;
}

function createDeferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  } {
  let resolvePromise!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve: (value: T) => resolvePromise(value),
  };
}

describe('Bedrock reasoning replay in streaming requests', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([
    {
      model:
        'arn:aws:bedrock:us-west-2::foundation-model/openai.gpt-oss-120b-1:0',
      replayReasoning: false,
    },
    {
      model:
        'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-luna',
      replayReasoning: false,
    },
    {
      model:
        'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0',
      replayReasoning: true,
    },
    {
      model: 'deepseek.v3.2',
      replayReasoning: true,
    },
    {
      model: 'moonshot.kimi-k2-thinking',
      replayReasoning: true,
    },
    {
      model: 'qwen.qwen3-32b-v1:0',
      replayReasoning: true,
    },
    {
      model: profileArn,
      replayReasoning: true,
    },
    {
      model: 'openai.gpt-oss-120b-1:0',
      applicationInferenceProfile: profileArn,
      replayReasoning: false,
    },
    {
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      applicationInferenceProfile: profileArn,
      replayReasoning: true,
    },
  ])(
    'sets reasoning replay to $replayReasoning for $model',
    async ({ model, applicationInferenceProfile, replayReasoning }) => {
      const client = new BedrockRuntimeClient({ region: 'us-east-1' });
      const send = jest.spyOn(client, 'send').mockImplementation(async () => ({
        $metadata: {},
        stream: (async function* () {
          yield {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { text: 'OK' },
            },
          };
        })(),
      }));
      const chat = new CustomChatBedrockConverse({
        model,
        applicationInferenceProfile,
        region: 'us-east-1',
        client,
        _lc_stream_delay: 0,
      });

      try {
        const stream = await chat.stream(claudeHistory());
        for await (const chunk of stream) {
          expect(chunk.content).toBe('OK');
        }

        expect(send).toHaveBeenCalledTimes(1);
        const command = send.mock.calls[0][0] as ConverseStreamCommand;
        expect(command).toBeInstanceOf(ConverseStreamCommand);
        expect(command.input.modelId).toBe(
          applicationInferenceProfile ?? model
        );
        expect(
          (command.input.messages ?? []).some((message) =>
            (message.content ?? []).some(
              (block) => block.reasoningContent != null
            )
          )
        ).toBe(replayReasoning);
        expect(command.input.messages?.[1].content).toContainEqual({
          text: 'Hello back',
        });
      } finally {
        client.destroy();
      }
    }
  );
});
describe('Bedrock reasoning replay in non-streaming requests', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([
    {
      model: 'openai.gpt-oss-120b-1:0',
      applicationInferenceProfile: profileArn,
      replayReasoning: false,
    },
    {
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      applicationInferenceProfile: profileArn,
      replayReasoning: true,
    },
    { model: profileArn, replayReasoning: true },
  ])(
    'uses configured model $model for replay while sending to its wire target',
    async ({ model, applicationInferenceProfile, replayReasoning }) => {
      const client = new BedrockRuntimeClient({ region: 'us-east-1' });
      const send = jest
        .spyOn(client, 'send')
        .mockImplementation(async () => nonStreamingResponse);
      const chat = new CustomChatBedrockConverse({
        model,
        applicationInferenceProfile,
        region: 'us-east-1',
        client,
      });

      try {
        const result = await chat.invoke(claudeHistory());
        expect(result.content).toBe('Done.');
        expect(result.response_metadata.model_provider).toBe(
          'bedrock-converse'
        );

        expect(send).toHaveBeenCalledTimes(1);
        const command = requireConverseCommand(send.mock.calls[0]?.[0]);
        expect(command.input.modelId).toBe(
          applicationInferenceProfile ?? model
        );
        expect(
          command.input.messages?.some((message) =>
            (message.content ?? []).some(
              (block) => block.reasoningContent != null
            )
          )
        ).toBe(replayReasoning);
        expect(command.input.messages?.[1].content).toContainEqual({
          text: 'Hello back',
        });
        expect(chat.model).toBe(model);
      } finally {
        client.destroy();
      }
    }
  );

  it('replaces reasoning-only history with a placeholder for OpenAI v1 output', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const send = jest
      .spyOn(client, 'send')
      .mockImplementation(async () => nonStreamingResponse);
    const chat = new CustomChatBedrockConverse({
      model: 'openai.gpt-oss-120b-1:0',
      outputVersion: 'v1',
      region: 'us-east-1',
      client,
    });
    const reasoningOnly = new AIMessage({
      content: [{ type: 'reasoning', reasoning: 'Private reasoning' }],
      response_metadata: { output_version: 'v1' },
    });

    try {
      await chat.invoke([
        new HumanMessage('First'),
        reasoningOnly,
        new HumanMessage('Second'),
      ]);

      const command = requireConverseCommand(send.mock.calls[0]?.[0]);
      expect(command.input.messages?.[1].content).toEqual([{ text: '_' }]);
      expect(reasoningOnly.content).toEqual([
        { type: 'reasoning', reasoning: 'Private reasoning' },
      ]);
    } finally {
      client.destroy();
    }
  });

  it('preserves request options and normalizes structured Converse output', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const send = jest
      .spyOn(client, 'send')
      .mockImplementation(async () => structuredResponse);
    const chat = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      applicationInferenceProfile: profileArn,
      region: 'us-east-1',
      client,
      temperature: 0.25,
      maxTokens: 256,
      serviceTier: 'default',
      additionalModelRequestFields: { top_k: 3 },
    });

    try {
      const result = await chat.invoke(
        [
          new SystemMessage('Use the supplied source.'),
          new HumanMessage('Answer.'),
        ],
        {
          cache_control: { type: 'ephemeral', ttl: '1h' },
          stop: ['HALT'],
          requestMetadata: { traceId: 'request-trace' },
          serviceTier: 'priority',
          tools: [
            {
              toolSpec: {
                name: 'lookup',
                description: 'Look up information',
                inputSchema: {
                  json: {
                    type: 'object',
                    properties: { query: { type: 'string' } },
                  },
                },
              },
            },
          ],
        }
      );
      const command = requireConverseCommand(send.mock.calls[0]?.[0]);

      expect(command.input.modelId).toBe(profileArn);
      expect(command.input.system).toEqual([
        { text: 'Use the supplied source.' },
        { cachePoint: { type: 'default', ttl: '1h' } },
      ]);
      expect(command.input.messages?.[0]).toEqual({
        role: 'user',
        content: [
          { text: 'Answer.' },
          { cachePoint: { type: 'default', ttl: '1h' } },
        ],
      });
      expect(command.input.toolConfig?.tools).toEqual([
        {
          toolSpec: {
            name: 'lookup',
            description: 'Look up information',
            inputSchema: {
              json: {
                type: 'object',
                properties: { query: { type: 'string' } },
              },
            },
          },
        },
        { cachePoint: { type: 'default', ttl: '1h' } },
      ]);
      expect(command.input.inferenceConfig).toEqual({
        maxTokens: 256,
        temperature: 0.25,
        stopSequences: ['HALT'],
      });
      expect(command.input.additionalModelRequestFields).toEqual({ top_k: 3 });
      expect(command.input.serviceTier).toEqual({ type: 'priority' });
      expect(command.input.requestMetadata).toEqual({
        traceId: 'request-trace',
      });

      expect(result.id).toBe('structured-response-request');
      expect(result.response_metadata).toMatchObject({
        model_provider: 'bedrock-converse',
        stopReason: 'tool_use',
        additionalModelResponseFields: { provider: 'bedrock' },
      });
      expect(result.usage_metadata).toEqual({
        input_tokens: 121,
        output_tokens: 9,
        total_tokens: 130,
        input_token_details: { cache_read: 17, cache_creation: 4 },
      });
      expect(result.tool_calls).toEqual([
        {
          id: 'response-tool-use',
          name: 'lookup',
          args: { query: 'weather' },
          type: 'tool_call',
        },
      ]);
      expect(result.content).toEqual(
        expect.arrayContaining([
          {
            type: 'reasoning_content',
            reasoningText: {
              text: 'Returned reasoning',
              signature: 'returned-signature',
            },
          },
          {
            type: 'citations_content',
            citationsContent: {
              content: [{ text: 'Cited answer.' }],
              citations: [
                {
                  sourceContent: [{ text: 'Source passage' }],
                  location: {
                    documentChar: {
                      documentIndex: 0,
                      start: 0,
                      end: 13,
                    },
                  },
                },
              ],
            },
          },
          {
            type: 'document',
            document: {
              format: 'txt',
              name: 'Reference document',
              source: { text: 'Document body' },
            },
          },
          {
            type: 'image',
            image: {
              format: 'png',
              source: { bytes: new Uint8Array([1, 2, 3]) },
            },
          },
          {
            type: 'video',
            video: {
              format: 'mp4',
              source: { bytes: new Uint8Array([4, 5, 6]) },
            },
          },
          {
            type: 'non_standard',
            value: { $unknown: ['futureBlock', { value: 'preserve me' }] },
          },
          { type: 'cache_point', cachePoint: { type: 'default', ttl: '1h' } },
          {
            type: 'guard_content',
            guardContent: { text: { text: 'Guarded text' } },
          },
          { type: 'text', text: 'Visible answer' },
          {
            type: 'tool_result',
            toolResult: {
              toolUseId: 'previous-tool-call',
              content: [{ text: 'Tool result' }],
            },
          },
        ])
      );
      expect(result.contentBlocks).toEqual(
        expect.arrayContaining([
          {
            type: 'reasoning',
            reasoning: 'Returned reasoning',
            signature: 'returned-signature',
          },
          {
            type: 'text',
            text: 'Cited answer.',
            annotations: [
              {
                type: 'citation',
                citedText: 'Source passage',
                source: '0',
                startIndex: 0,
                endIndex: 13,
              },
            ],
          },
          {
            type: 'image',
            mimeType: 'image/png',
            data: new Uint8Array([1, 2, 3]),
          },
          {
            type: 'video',
            mimeType: 'video/mp4',
            data: new Uint8Array([4, 5, 6]),
          },
        ])
      );
    } finally {
      client.destroy();
    }
  });

  it('exposes v1 reasoning and citations through the Bedrock content translator', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    jest.spyOn(client, 'send').mockImplementation(async () => structuredResponse);
    const chat = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      outputVersion: 'v1',
      region: 'us-east-1',
      client,
    });

    try {
      const result = await chat.invoke('Answer from the source.');
      expect(result.content).toEqual(
        expect.arrayContaining([
          {
            type: 'reasoning',
            reasoning: 'Returned reasoning',
            signature: 'returned-signature',
          },
          expect.objectContaining({
            type: 'text',
            text: 'Cited answer.',
            annotations: expect.arrayContaining([
              expect.objectContaining({
                type: 'citation',
                citedText: 'Source passage',
              }),
            ]),
          }),
        ])
      );
      expect(result.response_metadata.output_version).toBe('v1');
      expect(result.tool_calls).toEqual([
        {
          id: 'response-tool-use',
          name: 'lookup',
          args: { query: 'weather' },
          type: 'tool_call',
        },
      ]);
    } finally {
      client.destroy();
    }
  });

  it('folds cache usage into input and fallback total tokens', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const response: ConverseCommandOutput = {
      $metadata: {},
      output: {
        message: { role: 'assistant', content: [{ text: 'Done.' }] },
      },
      stopReason: undefined,
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: undefined,
        cacheReadInputTokens: 7,
        cacheWriteInputTokens: 3,
      },
      metrics: undefined,
    };
    jest.spyOn(client, 'send').mockImplementation(async () => response);
    const chat = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      region: 'us-east-1',
      client,
    });

    try {
      const result = await chat.invoke('Count tokens.');
      expect(result.usage_metadata).toEqual({
        input_tokens: 20,
        output_tokens: 5,
        total_tokens: 25,
        input_token_details: { cache_read: 7, cache_creation: 3 },
      });
    } finally {
      client.destroy();
    }
  });

  it('preserves native errors and normalizes AWS-shaped errors and missing outputs', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const send = jest.spyOn(client, 'send');
    const chat = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      region: 'us-east-1',
      client,
    });

    try {
      send.mockImplementationOnce(async () => ({
        $metadata: {},
        output: undefined,
        stopReason: undefined,
        usage: undefined,
        metrics: undefined,
      }));
      await expect(chat.invoke('Missing output.')).rejects.toThrow(
        'No message found in Bedrock response.'
      );

      const nativeError = new Error('Native service error');
      send.mockImplementationOnce(async () => {
        throw nativeError;
      });
      await expect(chat.invoke('Native failure.')).rejects.toBe(nativeError);

      const awsError = {
        errors: [{ message: 'First failure' }, 'Second failure'],
      };
      send.mockImplementationOnce(async () => {
        throw awsError;
      });
      await expect(chat.invoke('AWS-shaped failure.')).rejects.toMatchObject({
        message: 'First failure; Second failure',
        cause: awsError,
      });
    } finally {
      client.destroy();
    }
  });

  it('forwards an in-flight abort signal to Converse', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const send = jest.spyOn(client, 'send');
    const sendSignal = createDeferred<unknown>();
    const abortError = new Error('Request aborted');
    send.mockImplementation(async (_command, options) => {
      const signal = options.abortSignal;
      if (
        signal == null ||
        !('addEventListener' in signal) ||
        typeof signal.addEventListener !== 'function'
      ) {
        sendSignal.resolve(signal);
        throw new Error('Converse did not receive a native abort signal');
      }
      sendSignal.resolve(signal);
      return new Promise<ConverseCommandOutput>((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(abortError), {
          once: true,
        });
      });
    });
    const chat = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
      region: 'us-east-1',
      client,
    });
    const controller = new AbortController();

    try {
      const invocation = chat.invoke('Abort this request.', {
        signal: controller.signal,
      });
      expect(await sendSignal.promise).toBe(controller.signal);
      controller.abort();
      await expect(invocation).rejects.toBe(abortError);
    } finally {
      client.destroy();
    }
  });

  it('keeps concurrent invoke and stream on the configured OpenAI family', async () => {
    const client = new BedrockRuntimeClient({ region: 'us-east-1' });
    const invokeResponse = createDeferred<ConverseCommandOutput>();
    const invokeStarted = createDeferred<void>();
    const send = jest
      .spyOn(client, 'send')
      .mockImplementationOnce(async () => {
        invokeStarted.resolve(undefined);
        return invokeResponse.promise;
      })
      .mockImplementationOnce(async () => ({
        $metadata: {},
        stream: (async function* () {
          yield {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { text: 'Streamed.' },
            },
          };
        })(),
      }));
    const chat = new CustomChatBedrockConverse({
      model: 'openai.gpt-oss-120b-1:0',
      applicationInferenceProfile: profileArn,
      region: 'us-east-1',
      client,
      _lc_stream_delay: 0,
    });

    try {
      const invoke = chat.invoke(claudeHistory());
      await invokeStarted.promise;
      expect(chat.model).toBe('openai.gpt-oss-120b-1:0');

      for await (const chunk of await chat.stream(claudeHistory())) {
        expect(chunk.content).toBe('Streamed.');
      }
      expect(send).toHaveBeenCalledTimes(2);

      const invokeCommand = requireConverseCommand(send.mock.calls[0]?.[0]);
      expect(invokeCommand.input.modelId).toBe(profileArn);
      expect(
        invokeCommand.input.messages?.some((message) =>
          (message.content ?? []).some(
            (block) => block.reasoningContent != null
          )
        )
      ).toBe(false);

      const streamCommand = send.mock.calls[1]?.[0];
      if (!(streamCommand instanceof ConverseStreamCommand)) {
        throw new Error('Expected stream to send a ConverseStreamCommand');
      }
      expect(streamCommand.input.modelId).toBe(profileArn);
      expect(
        streamCommand.input.messages?.some((message) =>
          (message.content ?? []).some(
            (block) => block.reasoningContent != null
          )
        )
      ).toBe(false);
      expect(chat.model).toBe('openai.gpt-oss-120b-1:0');

      invokeResponse.resolve(nonStreamingResponse);
      await invoke;
    } finally {
      invokeResponse.resolve(nonStreamingResponse);
      client.destroy();
    }
  });
});

import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  ConverseCommand,
  BedrockRuntimeClient,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
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

describe('Bedrock Grok reasoning in tool continuations', () => {
  it.each([
    { model: 'us.xai.grok-4.7' },
    {
      model:
        'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.xai.grok-4.7',
    },
    { model: 'xai.grok-4.7', applicationInferenceProfile: profileArn },
  ])(
    'omits replay but preserves tool pairing for $model',
    async ({ model, applicationInferenceProfile }) => {
      const client = new BedrockRuntimeClient({ region: 'us-east-1' });
      const send = jest.spyOn(client, 'send');
      const toolUseId = 'tooluse_lookup';
      const assistant = new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: 'Search first' },
          },
          { type: 'text', text: 'Looking it up.' },
        ],
        tool_calls: [{ id: toolUseId, name: 'lookup', args: {} }],
      });
      const messages = [
        new HumanMessage('Find the answer.'),
        assistant,
        new ToolMessage({ tool_call_id: toolUseId, content: 'Found it.' }),
      ];
      const chat = new CustomChatBedrockConverse({
        model,
        applicationInferenceProfile,
        region: 'us-east-1',
        client,
        _lc_stream_delay: 0,
      });

      try {
        send.mockImplementationOnce(async () => ({
          $metadata: {},
          output: { message: { role: 'assistant', content: [{ text: 'OK' }] } },
        }));
        expect((await chat.invoke(messages)).content).toBe('OK');
        send.mockImplementationOnce(async () => ({
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
        for await (const chunk of await chat.stream(messages)) {
          expect(chunk.content).toBe('OK');
        }

        expect(send).toHaveBeenCalledTimes(2);
        const invoke = send.mock.calls[0][0] as ConverseCommand;
        const stream = send.mock.calls[1][0] as ConverseStreamCommand;
        expect(invoke).toBeInstanceOf(ConverseCommand);
        expect(stream).toBeInstanceOf(ConverseStreamCommand);
        for (const { input } of [invoke, stream]) {
          expect(input.modelId).toBe(applicationInferenceProfile ?? model);
          expect(input.messages?.[1].content).toEqual([
            { text: 'Looking it up.' },
            { toolUse: { toolUseId, name: 'lookup', input: {} } },
          ]);
          expect(input.messages?.[2].content?.[0].toolResult?.toolUseId).toBe(
            toolUseId
          );
        }
        expect(assistant.content).toContainEqual({
          type: 'reasoning_content',
          reasoningText: { text: 'Search first' },
        });
      } finally {
        client.destroy();
      }
    }
  );
});

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

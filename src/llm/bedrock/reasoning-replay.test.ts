import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
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

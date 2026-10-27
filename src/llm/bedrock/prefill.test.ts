import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import {
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  BedrockRuntimeClient,
  Message,
} from '@aws-sdk/client-bedrock-runtime';
import { stripUnsupportedConversePrefill } from './cachePoints';
import { CustomChatBedrockConverse } from './index';

const credentials = {
  secretAccessKey: 'test-secret-key',
  accessKeyId: 'test-access-key',
};

const user = (text: string): Message => ({ role: 'user', content: [{ text }] });
const assistant = (text: string): Message => ({
  role: 'assistant',
  content: [{ text }],
});

describe('stripUnsupportedConversePrefill', () => {
  test.each([
    'anthropic.claude-sonnet-4-6',
    'us.anthropic.claude-opus-4-7-v1:0',
    'global.anthropic.claude-haiku-5-5',
    'eu.anthropic.claude-sonnet-5',
    'anthropic.claude-opus-5-5-v1:0',
    'anthropic.claude-fable-5-1',
    'arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-5',
  ])('strips trailing assistant turns for %s', (model) => {
    const messages = [user('hi'), assistant('a'), assistant('b')];
    expect(stripUnsupportedConversePrefill(messages, model)).toEqual([
      user('hi'),
    ]);
  });

  test.each([
    'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'anthropic.claude-sonnet-4-20250514-v1:0',
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'anthropic.claude-haiku-4-5-20251001-v1:0',
    'amazon.nova-pro-v1:0',
    'meta.llama3-70b-instruct-v1:0',
    'mistral.mistral-large-2402-v1:0',
    undefined,
  ])('keeps the prefill for %s', (model) => {
    const messages = [user('hi'), assistant('prefill')];
    expect(stripUnsupportedConversePrefill(messages, model)).toBe(messages);
  });

  test('keeps a conversation that ends with a user turn', () => {
    const messages = [user('hi'), assistant('a'), user('next')];
    expect(
      stripUnsupportedConversePrefill(messages, 'anthropic.claude-haiku-5-5')
    ).toBe(messages);
  });

  test('never strips the only message', () => {
    const messages = [assistant('only')];
    expect(
      stripUnsupportedConversePrefill(messages, 'anthropic.claude-haiku-5-5')
    ).toBe(messages);
  });

  test('re-anchors a cache point that rode the stripped prefill', () => {
    const cachePoint = {
      cachePoint: { type: 'default' as const, ttl: '1h' as const },
    };
    const messages: Message[] = [
      user('hi'),
      { role: 'assistant', content: [{ text: 'prefill' }, cachePoint] },
    ];
    const result = stripUnsupportedConversePrefill(
      messages,
      'anthropic.claude-haiku-5-5'
    );
    expect(result).toEqual([
      { role: 'user', content: [{ text: 'hi' }, cachePoint] },
    ]);
    expect(messages[0].content).toEqual([{ text: 'hi' }]);
  });

  test('leaves surviving cache points alone', () => {
    const cachePoint = { cachePoint: { type: 'default' as const } };
    const messages: Message[] = [
      { role: 'user', content: [{ text: 'hi' }, cachePoint] },
      { role: 'assistant', content: [{ text: 'prefill' }, cachePoint] },
    ];
    expect(
      stripUnsupportedConversePrefill(messages, 'anthropic.claude-haiku-5-5')
    ).toEqual([messages[0]]);
  });
});

describe('CustomChatBedrockConverse assistant prefill', () => {
  const history = [new HumanMessage('Hello'), new AIMessage('Sure, ')];

  function createModel(
    model: string,
    send: jest.Mock
  ): CustomChatBedrockConverse {
    return new CustomChatBedrockConverse({
      region: 'us-east-1',
      credentials,
      model,
      client: { send } as unknown as BedrockRuntimeClient,
      _lc_stream_delay: 0,
    });
  }

  const converseResponse = {
    output: { message: { role: 'assistant', content: [{ text: 'Hi' }] } },
    stopReason: 'end_turn',
    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
  };

  const streamResponse = (): { stream: AsyncIterable<unknown> } => ({
    stream: (async function* streamChunks() {
      yield {
        contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'Hi' } },
      };
    })(),
  });

  test('strips the prefill on invoke for Claude Haiku 5.5', async () => {
    const send = jest.fn().mockResolvedValue(converseResponse);
    await createModel('global.anthropic.claude-haiku-5-5', send).invoke(
      history
    );
    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command).toBeInstanceOf(ConverseCommand);
    expect(command.input.messages?.map((m) => m.role)).toEqual(['user']);
  });

  test('strips a prefill followed by a system message on invoke', async () => {
    const send = jest.fn().mockResolvedValue(converseResponse);
    await createModel('anthropic.claude-sonnet-4-6', send).invoke([
      ...history,
      new SystemMessage('Be brief.'),
    ]);
    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command.input.messages?.map((m) => m.role)).toEqual(['user']);
  });

  test('strips the prefill on stream for Claude Haiku 5.5', async () => {
    const send = jest.fn().mockResolvedValue(streamResponse());
    const model = createModel('global.anthropic.claude-haiku-5-5', send);
    for await (const _chunk of await model.stream(history)) {
      // drain
    }
    const command = send.mock.calls[0][0] as ConverseStreamCommand;
    expect(command).toBeInstanceOf(ConverseStreamCommand);
    expect(command.input.messages?.map((m) => m.role)).toEqual(['user']);
  });

  test('keeps the prefill on invoke for Claude Haiku 4.5', async () => {
    const send = jest.fn().mockResolvedValue(converseResponse);
    await createModel('anthropic.claude-haiku-4-5-20251001-v1:0', send).invoke(
      history
    );
    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command.input.messages?.map((m) => m.role)).toEqual([
      'user',
      'assistant',
    ]);
  });

  test('keeps the prefill on stream for Claude Haiku 4.5', async () => {
    const send = jest.fn().mockResolvedValue(streamResponse());
    const model = createModel('anthropic.claude-haiku-4-5-20251001-v1:0', send);
    for await (const _chunk of await model.stream(history)) {
      // drain
    }
    const command = send.mock.calls[0][0] as ConverseStreamCommand;
    expect(command.input.messages?.map((m) => m.role)).toEqual([
      'user',
      'assistant',
    ]);
  });

  describe('retry without signed reasoning', () => {
    const signedHistory = [
      new HumanMessage('Hello'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: 'Thinking.', signature: 'SIG' },
          },
          { type: 'text', text: 'Hello back' },
        ],
      }),
      new HumanMessage('Continue'),
    ];
    const rejection = (message = 'thinking signature is not valid'): Error =>
      Object.assign(new Error(message), { name: 'ValidationException' });
    const retrying = (model: string, send: jest.Mock, retry = true) =>
      new CustomChatBedrockConverse({
        region: 'us-east-1',
        credentials,
        model,
        client: { send } as unknown as BedrockRuntimeClient,
        retryWithoutReasoningOnRejection: retry,
        _lc_stream_delay: 0,
      });
    const sentMessages = (send: jest.Mock): string[] =>
      send.mock.calls.map(([command]) =>
        JSON.stringify(
          (command as ConverseCommand | ConverseStreamCommand).input.messages
        )
      );

    test.each([
      ['the local converter', 'global.anthropic.claude-haiku-5-5'],
      ['the parent converter', 'anthropic.claude-haiku-4-5-20251001-v1:0'],
    ])('retries once on invoke through %s', async (_path, model) => {
      const send = jest
        .fn()
        .mockRejectedValueOnce(rejection())
        .mockResolvedValueOnce(converseResponse);

      await retrying(model, send).invoke(signedHistory);

      const sent = sentMessages(send);
      expect(sent).toHaveLength(2);
      expect(sent[0]).toContain('SIG');
      expect(sent[1]).not.toContain('SIG');
    });

    test('retries once on stream when nothing was emitted yet', async () => {
      const send = jest
        .fn()
        .mockRejectedValueOnce(rejection())
        .mockResolvedValueOnce(streamResponse());

      const model = retrying('anthropic.claude-haiku-4-5-20251001-v1:0', send);
      for await (const _chunk of await model.stream(signedHistory)) {
        // drain
      }

      const sent = sentMessages(send);
      expect(sent).toHaveLength(2);
      expect(sent[1]).not.toContain('SIG');
    });

    test('does not retry without the option or for other errors', async () => {
      const model = 'anthropic.claude-haiku-4-5-20251001-v1:0';
      const optedOut = jest.fn().mockRejectedValue(rejection());
      const unrelated = jest.fn().mockRejectedValue(rejection('throttled'));

      await expect(
        retrying(model, optedOut, false).invoke(signedHistory)
      ).rejects.toThrow('signature');
      await expect(
        retrying(model, unrelated).invoke(signedHistory)
      ).rejects.toThrow('throttled');
      expect(optedOut).toHaveBeenCalledTimes(1);
      expect(unrelated).toHaveBeenCalledTimes(1);
    });
  });
});

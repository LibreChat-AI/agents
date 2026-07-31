import type { TMessage } from '@/types';
import { ContentTypes, Providers } from '@/common';
import { formatAgentMessages } from './format';

const model = 'claude-sonnet-5-20260701';

describe('formatAgentMessages Anthropic reasoning replay', () => {
  const persistedMessage: TMessage = {
    role: 'assistant',
    content: [
      {
        type: ContentTypes.THINK,
        think: '',
        provider_replay: {
          anthropic: {
            model,
            blocks: [
              {
                type: ContentTypes.THINKING,
                thinking: '',
                signature: 'signed-encrypted-thinking',
              },
              {
                type: 'redacted_thinking',
                data: 'redacted-encrypted-thinking',
              },
            ],
          },
        },
      },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'toolu_123',
          name: 'lookup',
          args: '{"value":42}',
          output: 'done',
        },
      },
    ],
  };

  it('reconstructs signed and redacted blocks before Anthropic tool use', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      [persistedMessage],
      { 0: 700 },
      undefined,
      undefined,
      { provider: Providers.ANTHROPIC, model }
    );

    expect(messages[0].content).toEqual([
      {
        type: ContentTypes.THINKING,
        thinking: '',
        signature: 'signed-encrypted-thinking',
      },
      {
        type: 'redacted_thinking',
        data: 'redacted-encrypted-thinking',
      },
      {
        type: 'tool_use',
        id: 'toolu_123',
        name: 'lookup',
        input: { value: 42 },
      },
    ]);
    expect(
      Object.values(indexTokenCountMap ?? {}).reduce(
        (total, count) => total + count,
        0
      )
    ).toBe(700);
  });

  it('does not replay Anthropic blocks to another provider and invalidates their token weight', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      [persistedMessage],
      { 0: 700 },
      undefined,
      undefined,
      { provider: Providers.BEDROCK, model: 'anthropic.claude-sonnet-5-v1:0' }
    );

    expect(JSON.stringify(messages)).not.toContain('signed-encrypted-thinking');
    expect(JSON.stringify(messages)).not.toContain(
      'redacted-encrypted-thinking'
    );
    expect(indexTokenCountMap).toEqual({});
  });

  it('does not replay thinking from another Anthropic model', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      [persistedMessage],
      { 0: 700 },
      undefined,
      undefined,
      { provider: Providers.ANTHROPIC, model: 'claude-opus-4-8' }
    );

    expect(JSON.stringify(messages)).not.toContain('signed-encrypted-thinking');
    expect(JSON.stringify(messages)).not.toContain(
      'redacted-encrypted-thinking'
    );
    expect(indexTokenCountMap).toEqual({});
  });

  it('rejects unsigned persisted thinking blocks', () => {
    const message: TMessage = {
      role: 'assistant',
      content: [
        {
          type: ContentTypes.THINK,
          think: 'visible summary',
          provider_replay: {
            anthropic: {
              model,
              blocks: [
                {
                  type: ContentTypes.THINKING,
                  thinking: 'visible summary',
                },
              ],
            },
          },
        },
        { type: ContentTypes.TEXT, text: 'answer' },
      ],
    };

    const { messages } = formatAgentMessages(
      [message],
      undefined,
      undefined,
      undefined,
      { provider: Providers.ANTHROPIC, model }
    );

    expect(JSON.stringify(messages)).not.toContain('"signature"');
    expect(messages[0].content).toBe('answer');
  });
});

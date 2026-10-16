import type { AIMessage } from '@langchain/core/messages';
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
      {
        replayProviderReasoning: { provider: Providers.ANTHROPIC, model },
      }
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
    ]);
    expect((messages[0] as AIMessage).tool_calls?.[0]).toMatchObject({
      id: 'toolu_123',
      name: 'lookup',
      args: { value: 42 },
    });
    // Replayed blocks are not part of the stored weight, so it is recounted.
    expect(indexTokenCountMap).toEqual({});
  });

  it('does not replay Anthropic blocks to another provider and keeps their token weight', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      [persistedMessage],
      { 0: 700 },
      undefined,
      undefined,
      {
        replayProviderReasoning: { provider: Providers.BEDROCK, model: 'anthropic.claude-sonnet-5-v1:0' },
      }
    );

    expect(JSON.stringify(messages)).not.toContain('signed-encrypted-thinking');
    expect(JSON.stringify(messages)).not.toContain(
      'redacted-encrypted-thinking'
    );
    expect(
      Object.values(indexTokenCountMap ?? {}).reduce((total, count) => total + count, 0)
    ).toBe(700);
  });

  it('does not replay thinking from another Anthropic model', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      [persistedMessage],
      { 0: 700 },
      undefined,
      undefined,
      {
        replayProviderReasoning: { provider: Providers.ANTHROPIC, model: 'claude-opus-4-8' },
      }
    );

    expect(JSON.stringify(messages)).not.toContain('signed-encrypted-thinking');
    expect(JSON.stringify(messages)).not.toContain(
      'redacted-encrypted-thinking'
    );
    expect(
      Object.values(indexTokenCountMap ?? {}).reduce((total, count) => total + count, 0)
    ).toBe(700);
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
      {
        replayProviderReasoning: { provider: Providers.ANTHROPIC, model },
      }
    );

    expect(JSON.stringify(messages)).not.toContain('"signature"');
    expect(messages[0].content).toBe('answer');
  });

  it('leaves OpenAI Responses reasoning to the tool round items, whatever a part stores', () => {
    const message: TMessage = {
      role: 'assistant',
      content: [
        {
          type: ContentTypes.THINK,
          think: 'summary',
          provider_replay: {
            openai_responses: {
              provider: Providers.OPENAI,
              model: 'gpt-test',
              items: [{ type: 'reasoning', id: 'rs_1', encrypted_content: 'opaque' }],
            },
          },
        } as never,
        { type: ContentTypes.TEXT, text: 'answer' },
      ],
    };

    const { messages } = formatAgentMessages([message], undefined, undefined, undefined, {
      replayProviderReasoning: { provider: Providers.OPENAI, model: 'gpt-test' },
    });

    expect(JSON.stringify(messages)).not.toContain('opaque');
    expect(messages[0].content).toBe('answer');
  });

  it('ignores stored replay unless the option is set', () => {
    const { messages } = formatAgentMessages([persistedMessage]);

    expect(JSON.stringify(messages)).not.toContain('signed-encrypted-thinking');
  });

  describe('position', () => {
    const replayPart = (signature: string): TMessage['content'] extends infer C
      ? C extends Array<infer P> ? P : never : never =>
      ({
        type: ContentTypes.THINK,
        think: '',
        provider_replay: {
          anthropic: {
            model,
            blocks: [{ type: ContentTypes.THINKING, thinking: '', signature }],
          },
        },
      }) as never;
    const toolCall = (id: string, roundId: string): never =>
      ({
        type: ContentTypes.TOOL_CALL,
        tool_call: { id, name: 'lookup', args: '{}', output: 'done', roundId },
      }) as never;
    const options = {
      replayProviderReasoning: { provider: Providers.ANTHROPIC, model },
    };

    it('keeps each round blocks with the round they came from', () => {
      const { messages } = formatAgentMessages(
        [
          {
            role: 'assistant',
            content: [
              replayPart('sig-1'),
              toolCall('toolu_1', 'r1'),
              replayPart('sig-2'),
              toolCall('toolu_2', 'r2'),
              replayPart('sig-3'),
              { type: ContentTypes.TEXT, text: 'final' },
            ],
          } as TMessage,
        ],
        undefined,
        undefined,
        undefined,
        options
      );
      const signatures = messages
        .filter((message) => message.getType() === 'ai')
        .map((message) =>
          (message.content as Array<{ signature?: string }>)
            .filter((part) => typeof part === 'object' && part.signature != null)
            .map((part) => part.signature)
        );
      expect(signatures).toEqual([['sig-1'], ['sig-2'], ['sig-3']]);
    });

    it('keeps a block that follows the text of its own response', () => {
      const { messages } = formatAgentMessages(
        [
          {
            role: 'assistant',
            content: [
              replayPart('sig-before'),
              { type: ContentTypes.TEXT, text: 'checking' },
              replayPart('sig-after'),
              toolCall('toolu_1', 'r1'),
            ],
          } as TMessage,
        ],
        undefined,
        undefined,
        undefined,
        options
      );
      const ai = messages.find((message) => message.getType() === 'ai');
      expect(
        (ai?.content as Array<{ type: string; text?: string; signature?: string }>).map(
          (part) => part.signature ?? part.text
        )
      ).toEqual(['sig-before', 'checking', 'sig-after']);
    });
  });

  describe('holes in the chain', () => {
    const options = {
      replayProviderReasoning: { provider: Providers.ANTHROPIC, model },
    };
    const turn = (signature: string | undefined, think = 'plan'): TMessage =>
      ({
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            think,
            ...(signature != null && {
              provider_replay: {
                anthropic: {
                  model,
                  blocks: [{ type: ContentTypes.THINKING, thinking: '', signature }],
                },
              },
            }),
          },
          { type: ContentTypes.TEXT, text: 'answer' },
        ],
      }) as TMessage;
    const sent = (payload: TMessage[]): string[] => {
      const { messages } = formatAgentMessages(
        payload,
        undefined,
        undefined,
        undefined,
        options
      );
      const signatures = JSON.stringify(messages).match(/sig-\d/g) ?? [];
      return [...new Set(signatures)];
    };

    it('replays every turn when none is missing', () => {
      expect(sent([turn('sig-1'), turn('sig-2'), turn('sig-3'), turn('sig-4')])).toEqual([
        'sig-1',
        'sig-2',
        'sig-3',
        'sig-4',
      ]);
    });

    it('drops the turns before a hole, never the ones after it', () => {
      /** Turn 2 has no replay (not stored, over a cap, or edited away). */
      expect(sent([turn('sig-1'), turn(undefined), turn('sig-3'), turn('sig-4')])).toEqual([
        'sig-3',
        'sig-4',
      ]);
    });

    it('does not treat a turn whose blocks were withdrawn as a hole', () => {
      /** It was sent without them, so the blocks around it were signed for that. */
      const withdrawn = turn(undefined);
      (withdrawn.content as Array<Record<string, unknown>>)[0].provider_replay = {
        invalidated: true,
      };
      expect(sent([turn('sig-1'), withdrawn, turn('sig-3')])).toEqual(['sig-1', 'sig-3']);
    });

    it('treats a turn of another model as a hole', () => {
      const other = turn('sig-2');
      (other.content as Array<Record<string, unknown>>)[0].provider_replay = {
        anthropic: {
          model: 'claude-other',
          blocks: [{ type: ContentTypes.THINKING, thinking: '', signature: 'sig-2' }],
        },
      };
      expect(sent([turn('sig-1'), other, turn('sig-3')])).toEqual(['sig-3']);
    });

    it('leaves the stored weight of a turn that is not replayed alone', () => {
      const { indexTokenCountMap } = formatAgentMessages(
        [turn('sig-1'), turn(undefined), turn('sig-3')],
        { 0: 10, 1: 20, 2: 30 },
        undefined,
        undefined,
        options
      );
      expect(Object.keys(indexTokenCountMap ?? {}).sort()).toEqual(['0', '1']);
    });

    describe('the message that carries a summary', () => {
      const part = (signature: string): Record<string, unknown> =>
        (turn(signature).content as Array<Record<string, unknown>>)[0];
      const summary = { type: ContentTypes.SUMMARY, text: 'summary', tokenCount: 5 };
      const text = { type: ContentTypes.TEXT, text: 'answer' };
      const toolCall = {
        type: ContentTypes.TOOL_CALL,
        tool_call: { id: 'toolu_1', name: 'lookup', args: '{}', output: 'done' },
      };

      it('replays nothing from before the summary and everything after it', () => {
        /** [summary, think, text]: the summary opens the turn that produced the block. */
        const carrier = {
          role: 'assistant',
          content: [summary, part('sig-2'), text],
        } as TMessage;
        expect(sent([turn('sig-1'), carrier, turn('sig-3'), turn('sig-4')])).toEqual([
          'sig-2',
          'sig-3',
          'sig-4',
        ]);
      });

      it('replays only the parts after a summary in the middle of the turn', () => {
        /** [think, tool_call, summary, think, text]: the first block precedes the summary. */
        const carrier = {
          role: 'assistant',
          content: [part('sig-1'), toolCall, summary, part('sig-2'), text],
        } as TMessage;
        expect(sent([turn('sig-0'), carrier, turn('sig-3')])).toEqual(['sig-2', 'sig-3']);
      });

      it('does not count the parts before the summary as a hole', () => {
        const unsigned = { type: ContentTypes.THINK, think: 'plan' };
        const carrier = {
          role: 'assistant',
          content: [unsigned, toolCall, summary, part('sig-2'), text],
        } as TMessage;
        expect(sent([carrier, turn('sig-3')])).toEqual(['sig-2', 'sig-3']);
      });

      it('treats a part after the summary without a replay as a hole', () => {
        const unsigned = { type: ContentTypes.THINK, think: 'plan' };
        const carrier = {
          role: 'assistant',
          content: [part('sig-1'), summary, unsigned, text],
        } as TMessage;
        expect(sent([carrier, turn('sig-3')])).toEqual(['sig-3']);
      });

      it('keeps the messages a coverage summary retained out of the replay', () => {
        const retained = { ...turn('sig-1'), messageId: 'retained' } as TMessage;
        const carrier = {
          role: 'assistant',
          messageId: 'carrier',
          content: [
            { ...summary, coverage: { retainedFromMessageId: 'retained' } },
            part('sig-2'),
            text,
          ],
        } as TMessage;
        expect(sent([retained, carrier, turn('sig-3')])).toEqual(['sig-2', 'sig-3']);
      });
    });
  });
});

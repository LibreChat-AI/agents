import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { ContentTypes, Providers } from '@/common';
import { formatAgentMessages } from './format';

const model = 'anthropic.claude-sonnet-5-20260701-v1:0';
const payload: TPayload = [
  {
    role: 'assistant',
    content: [
      {
        type: ContentTypes.THINK,
        think: 'reasoned answer',
        provider_replay: {
          bedrock: {
            model,
            blocks: [
              {
                type: ContentTypes.REASONING_CONTENT,
                reasoningText: {
                  text: 'reasoned answer',
                  signature: 'sig-bedrock',
                },
              },
            ],
          },
        },
      },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'tool-1',
          name: 'calculator',
          args: { expression: '6 * 7' },
          output: '42',
        },
      },
      {
        type: ContentTypes.TEXT,
        text: '42',
      },
    ],
  },
];

describe('formatAgentMessages Bedrock reasoning replay', () => {
  it('reconstructs signed native reasoning before its Bedrock tool call', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      payload,
      { 0: 500 },
      new Set(['calculator']),
      undefined,
      { provider: Providers.BEDROCK, model }
    );

    expect(messages).toHaveLength(3);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[0].content).toEqual([
      {
        type: ContentTypes.REASONING_CONTENT,
        reasoningText: {
          text: 'reasoned answer',
          signature: 'sig-bedrock',
        },
      },
    ]);
    expect((messages[0] as AIMessage).tool_calls).toEqual([
      {
        id: 'tool-1',
        name: 'calculator',
        args: { expression: '6 * 7' },
      },
    ]);
    expect(messages[1]).toBeInstanceOf(ToolMessage);
    expect(messages[2].content).toBe('42');
    expect(
      Object.values(indexTokenCountMap ?? {}).reduce(
        (total, count) => total + count,
        0
      )
    ).toBe(500);
  });

  it('does not replay a Bedrock signature to another provider and invalidates its token weight', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      payload,
      { 0: 500 },
      new Set(['calculator']),
      undefined,
      { provider: Providers.ANTHROPIC, model: 'claude-sonnet-5' }
    );

    expect(JSON.stringify(messages)).not.toContain('sig-bedrock');
    expect(JSON.stringify(messages)).not.toContain('reasoning_content');
    expect(indexTokenCountMap).toEqual({});
  });

  it('does not replay signed reasoning to another Bedrock model', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      payload,
      { 0: 500 },
      new Set(['calculator']),
      undefined,
      { provider: Providers.BEDROCK, model: 'amazon.nova-2-lite-v1:0' }
    );

    expect(JSON.stringify(messages)).not.toContain('sig-bedrock');
    expect(JSON.stringify(messages)).not.toContain('reasoning_content');
    expect(indexTokenCountMap).toEqual({});
  });

  it('drops incomplete persisted reasoning instead of sending unsigned native content', () => {
    const incompletePayload = structuredClone(payload);
    const think = incompletePayload[0].content?.[0];
    if (typeof think === 'object' && 'provider_replay' in think) {
      const block = think.provider_replay.bedrock?.blocks[0];
      if (block?.reasoningText != null) {
        block.reasoningText.signature = '';
      }
    }

    const { messages } = formatAgentMessages(
      incompletePayload,
      undefined,
      new Set(['calculator']),
      undefined,
      { provider: Providers.BEDROCK, model }
    );

    expect(JSON.stringify(messages)).not.toContain('reasoning_content');
    expect(JSON.stringify(messages)).not.toContain('sig-bedrock');
  });
});

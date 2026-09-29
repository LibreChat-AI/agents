import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  replaysBedrockReasoning,
  convertToConverseMessages,
} from './message_inputs';

/**
 * Companion to the cross-provider reasoning fix, for a handoff that stays on Bedrock but
 * changes model family (Claude → GPT). The Claude turn's reasoning is Bedrock-native
 * `reasoning_content`, so the cross-provider drop does not apply, and Bedrock's OpenAI
 * models reject an assistant turn that carries it: "This model doesn't support the
 * reasoningContent.reasoningText.text field for assistant messages".
 */
type ConverseResult = ReturnType<typeof convertToConverseMessages>;

/** Minimal view of a converted Bedrock Converse content block the assertions read. */
interface ConverseBlock {
  text?: string;
  reasoningContent?: { reasoningText?: { text?: string; signature?: string } };
  toolUse?: { toolUseId?: string; name?: string };
}

const assistantTurns = (result: ConverseResult): ConverseBlock[][] =>
  result.converseMessages
    .filter((m) => m.role === 'assistant')
    .map((m) => (m.content ?? []) as ConverseBlock[]);

const claudeReasoning = {
  type: 'reasoning_content',
  reasoningText: {
    text: 'The user wants PA coverage; search first.',
    signature: 'claude-signature',
  },
};

/** A Claude turn as a Bedrock Converse response leaves it: reasoning, text, then a tool call. */
const claudeHandoffHistory = (): BaseMessage[] => [
  new HumanMessage(
    'Search for the top story, then hand over to the wire agent.'
  ),
  new AIMessage({
    content: [claudeReasoning, { type: 'text', text: 'Searching.' }],
    tool_calls: [
      {
        id: 'tooluse_search',
        name: 'search',
        args: { q: 'politics' },
        type: 'tool_call',
      },
    ],
  }),
  new ToolMessage({ tool_call_id: 'tooluse_search', content: 'three items' }),
  new AIMessage({
    content: [claudeReasoning],
    tool_calls: [
      {
        id: 'tooluse_transfer',
        name: 'lc_transfer_to_wire_agent',
        args: {},
        type: 'tool_call',
      },
    ],
  }),
  new ToolMessage({ tool_call_id: 'tooluse_transfer', content: 'Transferred' }),
];

describe('convertToConverseMessages — cross-model reasoning (Bedrock Claude → Bedrock GPT)', () => {
  it.each([
    'global.openai.gpt-6-luna',
    'arn:aws:bedrock:us-west-2::foundation-model/openai.gpt-oss-120b-1:0',
    'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-luna',
  ])('leaves Claude reasoning out of a request for %s', (model) => {
    const turns = assistantTurns(
      convertToConverseMessages(claudeHandoffHistory(), { model })
    );

    expect(
      turns.flat().find((b) => b.reasoningContent != null)
    ).toBeUndefined();
    expect(JSON.stringify(turns)).not.toContain('claude-signature');
    expect(turns[0].some((b) => b.text === 'Searching.')).toBe(true);
    expect(turns[0].find((b) => b.toolUse != null)?.toolUse).toMatchObject({
      toolUseId: 'tooluse_search',
      name: 'search',
    });
    expect(turns[1].find((b) => b.toolUse != null)?.toolUse).toMatchObject({
      toolUseId: 'tooluse_transfer',
    });
  });

  it.each([
    'eu.anthropic.claude-sonnet-5',
    'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0',
    'arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-5',
  ])('keeps the reasoning across a tool loop for %s', (model) => {
    const turns = assistantTurns(
      convertToConverseMessages(claudeHandoffHistory(), { model })
    );

    expect(turns[0][0].reasoningContent?.reasoningText).toEqual({
      text: 'The user wants PA coverage; search first.',
      signature: 'claude-signature',
    });
    expect(turns[1][0].reasoningContent).toBeDefined();
  });

  it('keeps the reasoning when the model is unknown, as before', () => {
    for (const options of [
      {},
      {
        model:
          'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/abc123',
      },
    ]) {
      const turns = assistantTurns(
        convertToConverseMessages(claudeHandoffHistory(), options)
      );
      expect(turns[0][0].reasoningContent).toBeDefined();
    }
  });

  it('emits a placeholder when a reasoning-only turn is left empty', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({ content: [claudeReasoning] }),
      new HumanMessage('and?'),
    ];

    const turns = assistantTurns(
      convertToConverseMessages(messages, { model: 'global.openai.gpt-6-luna' })
    );

    expect(turns[0]).toEqual([{ text: '_' }]);
  });

  it.each([
    'global.openai.gpt-6-sol',
    'arn:aws:bedrock:us-west-2::foundation-model/openai.gpt-oss-120b-1:0',
    'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.openai.gpt-6-sol',
  ])('drops v1 reasoning blocks for %s', (model) => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'Thinking it through.' },
          { type: 'text', text: 'Hello.' },
        ],
        response_metadata: { output_version: 'v1' },
      }),
    ];

    const [gptTurn] = assistantTurns(
      convertToConverseMessages(messages, { model })
    );
    const [claudeTurn] = assistantTurns(
      convertToConverseMessages(messages, { model: 'anthropic.claude-opus-5' })
    );

    expect(gptTurn).toEqual([{ text: 'Hello.' }]);
    expect(claudeTurn[0].reasoningContent?.reasoningText?.text).toBe(
      'Thinking it through.'
    );
  });
});

describe('replaysBedrockReasoning', () => {
  it('replays reasoning only to Claude, or when the model is unknown', () => {
    expect(replaysBedrockReasoning('eu.anthropic.claude-sonnet-5')).toBe(true);
    expect(
      replaysBedrockReasoning('anthropic.claude-3-7-sonnet-20250219-v1:0')
    ).toBe(true);
    expect(replaysBedrockReasoning(undefined)).toBe(true);
    expect(replaysBedrockReasoning('')).toBe(true);
    expect(
      replaysBedrockReasoning(
        'arn:aws:bedrock:us-east-1:123:application-inference-profile/x'
      )
    ).toBe(true);
    expect(replaysBedrockReasoning('global.openai.gpt-6-luna')).toBe(false);
    expect(replaysBedrockReasoning('openai.gpt-oss-120b-1:0')).toBe(false);
    expect(replaysBedrockReasoning('global.xai.grok-4.6')).toBe(false);
  });

  it.each([
    'arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/openai.gpt-oss-120b-1:0',
    'arn:aws-cn:bedrock:cn-north-1:123456789012:inference-profile/openai.gpt-oss-120b-1:0',
  ])('identifies a non-Claude model in %s', (model) => {
    expect(replaysBedrockReasoning(model)).toBe(false);
  });

  it.each([
    'arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123',
    'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/openai-profile',
    'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/foundation-model/openai.gpt-oss-120b-1:0',
  ])('preserves the fallback for an opaque resource: %s', (model) => {
    expect(replaysBedrockReasoning(model)).toBe(true);
  });
});

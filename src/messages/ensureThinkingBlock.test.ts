import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ExtendedMessageContent } from '@/types';
import {
  getProviderMessageProvenance,
  setProviderMessageProvenance,
} from './provenance';
import { ensureThinkingBlockInMessages } from './format';
import { Providers, ContentTypes } from '@/common';

/** Helper: extract concatenated text from a message's content (string or structured array). */
function getTextContent(msg: {
  content: string | ExtendedMessageContent[];
}): string {
  if (typeof msg.content === 'string') {
    return msg.content;
  }
  if (Array.isArray(msg.content)) {
    return (msg.content as ExtendedMessageContent[])
      .filter((b) => b.type === 'text')
      .map((b) => String(b.text ?? ''))
      .join('\n');
  }
  return '';
}

describe('ensureThinkingBlockInMessages', () => {
  describe('messages with thinking blocks (should not be modified)', () => {
    test('should not modify AI message that already has thinking block', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({
          content: [
            { type: ContentTypes.THINKING, thinking: 'Let me think...' },
            { type: 'text', text: 'Hi there!' },
          ],
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.THINKING
      );
    });

    test('should not modify AI message that has redacted_thinking block', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({
          content: [
            { type: 'redacted_thinking', data: 'redacted' },
            { type: 'text', text: 'Hi there!' },
          ],
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        'redacted_thinking'
      );
    });

    test('should not modify AI message with reasoning_content block and tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Calculate something' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'I need to use a calculator' },
            },
          ],
          tool_calls: [
            {
              id: 'call_456',
              name: 'calculator',
              args: { input: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          tool_call_id: 'call_456',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.REASONING_CONTENT
      );
    });

    test('should not convert follow-up tool calls in a thinking-enabled chain (Bedrock multi-step)', () => {
      // Bedrock reasoning models produce reasoning on the first AI response,
      // then subsequent tool calls in the same chain have content: "" with no
      // reasoning block. These should NOT be converted because the chain
      // already has a thinking block upstream.
      const messages = [
        new HumanMessage({ content: 'show me something cool' }),
        new AIMessage({
          content: [
            { type: 'text', text: '\n\n' },
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'Let me navigate to a page' },
            },
            { type: 'text', text: 'Let me whip up something fun!' },
          ],
          tool_calls: [
            {
              id: 'call_nav',
              name: 'navigate_page',
              args: { url: 'about:blank' },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'Navigated to about:blank',
          tool_call_id: 'call_nav',
        }),
        // Follow-up: content: "", tool calls, NO reasoning block
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_eval',
              name: 'evaluate_script',
              args: { script: 'document.title = "test"' },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'Script executed',
          tool_call_id: 'call_eval',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      // All 5 messages preserved — the follow-up AI message at index 3 is NOT converted
      expect(result).toHaveLength(5);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect(result[3]).toBeInstanceOf(AIMessage);
      expect(result[3].content).toBe('');
      expect((result[3] as AIMessage).tool_calls).toHaveLength(1);
      expect(result[4]).toBeInstanceOf(ToolMessage);
    });

    test('should not convert multiple follow-up tool calls in a long chain', () => {
      // Three AI→Tool rounds: only the first has reasoning
      const messages = [
        new HumanMessage({ content: 'do stuff' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'Planning...' },
            },
          ],
          tool_calls: [
            { id: 'c1', name: 'step1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c2', name: 'step2', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r2', tool_call_id: 'c2' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c3', name: 'step3', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r3', tool_call_id: 'c3' }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(7);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[3]).toBeInstanceOf(AIMessage);
      expect(result[5]).toBeInstanceOf(AIMessage);
    });

    test('should still convert non-thinking agent tool calls after a human message boundary', () => {
      // A chain with thinking, then a new human message, then a chain WITHOUT thinking
      const messages = [
        new HumanMessage({ content: 'first request' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'Thinking...' },
            },
          ],
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new HumanMessage({ content: 'second request' }),
        // This chain has NO thinking blocks — should be converted
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            { id: 'c2', name: 'tool2', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r2', tool_call_id: 'c2' }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      // First chain preserved (3 msgs), human preserved, second chain converted (1 HumanMessage)
      expect(result).toHaveLength(5);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage); // reasoning chain — kept
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect(result[3]).toBeInstanceOf(HumanMessage); // user message
      expect(result[4]).toBeInstanceOf(HumanMessage); // converted — no thinking in this chain
      expect(getTextContent(result[4])).toContain('[Previous agent context]');
    });

    test('preserves model and tool attribution when folding prior agent context', () => {
      const assistant = new AIMessage({
        content: '',
        tool_calls: [
          { id: 'c1', name: 'lookup', args: {}, type: 'tool_call' as const },
        ],
      });
      setProviderMessageProvenance(assistant, [
        { attribution: 'model', sourceMessageId: 'assistant-row' },
      ]);
      const toolResult = new ToolMessage({
        content: 'trusted tool bytes',
        tool_call_id: 'c1',
      });
      setProviderMessageProvenance(toolResult, [
        { attribution: 'tool', sourceMessageId: 'tool-row' },
      ]);

      const result = ensureThinkingBlockInMessages(
        [new HumanMessage('run lookup'), assistant, toolResult],
        Providers.BEDROCK
      );

      expect(getProviderMessageProvenance(result[1])?.parts).toEqual([
        { attribution: 'synthetic' },
        { attribution: 'model', sourceMessageId: 'assistant-row' },
        { attribution: 'tool', sourceMessageId: 'tool-row' },
      ]);
    });

    test('should detect thinking via additional_kwargs.reasoning_content in chain', () => {
      const messages = [
        new HumanMessage({ content: 'hello' }),
        new AIMessage({
          content: '',
          additional_kwargs: {
            reasoning_content: 'Some reasoning...',
          },
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c2', name: 'tool2', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r2', tool_call_id: 'c2' }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      // Index 3 should NOT be converted — index 1 has reasoning in additional_kwargs
      expect(result).toHaveLength(5);
      expect(result[3]).toBeInstanceOf(AIMessage);
    });

    test('should not modify AI message when reasoning_content is not the first block (Bedrock whitespace artifact)', () => {
      // Bedrock emits a "\n\n" text chunk before the thinking block,
      // pushing reasoning_content to content[1] instead of content[0].
      const messages = [
        new HumanMessage({ content: 'Do something' }),
        new AIMessage({
          content: [
            { type: 'text', text: '\n\n' },
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'Let me think about this' },
            },
            { type: 'text', text: 'Let me help!' },
          ],
          tool_calls: [
            {
              id: 'call_bedrock',
              name: 'some_tool',
              args: { x: 1 },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'tool result',
          tool_call_id: 'call_bedrock',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      // The AI message should be preserved, not converted to a HumanMessage
      expect(result[1].content).toEqual(messages[1].content);
    });

    test('should not modify AI message with reasoning block and tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Calculate something' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING,
              reasoning: 'I need to use a calculator',
            },
          ],
          tool_calls: [
            {
              id: 'call_789',
              name: 'calculator',
              args: { input: '3+3' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '6',
          tool_call_id: 'call_789',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.VERTEXAI
      );

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.REASONING
      );
    });
  });

  describe('messages with tool_calls (should be converted)', () => {
    test('should convert AI message with tool_calls to HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'What is the weather?' }),
        new AIMessage({
          content: 'Let me check the weather.',
          tool_calls: [
            {
              id: 'call_123',
              name: 'get_weather',
              args: { location: 'NYC' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Sunny, 75°F',
          tool_call_id: 'call_123',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should have 2 messages: HumanMessage + converted HumanMessage
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('What is the weather?');
      expect(result[1]).toBeInstanceOf(HumanMessage);

      // Check that the converted message includes the context prefix
      const text = getTextContent(result[1]);
      expect(text).toContain('[Previous agent context]');
      expect(text).toContain('Let me check the weather');
      expect(text).toContain('Sunny, 75°F');
    });

    test('should convert AI message with tool_use in content to HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'Search for something' }),
        new AIMessage({
          content: [
            { type: 'text', text: 'Searching...' },
            {
              type: 'tool_use',
              id: 'call_456',
              name: 'search',
              input: { query: 'test' },
            },
          ],
        }),
        new ToolMessage({
          content: 'Found results',
          tool_call_id: 'call_456',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      const text = getTextContent(result[1]);
      expect(text).toContain('[Previous agent context]');
      expect(text).toContain('Searching...');
      expect(text).toContain('Found results');
    });

    test('should handle multiple tool messages in sequence', () => {
      const messages = [
        new HumanMessage({ content: 'Do multiple things' }),
        new AIMessage({
          content: 'I will perform multiple actions.',
          tool_calls: [
            {
              id: 'call_1',
              name: 'action1',
              args: { param: 'a' },
              type: 'tool_call',
            },
            {
              id: 'call_2',
              name: 'action2',
              args: { param: 'b' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 1',
          tool_call_id: 'call_1',
        }),
        new ToolMessage({
          content: 'Result 2',
          tool_call_id: 'call_2',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should combine all tool messages into one HumanMessage
      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      const text = getTextContent(result[1]);
      expect(text).toContain('Result 1');
      expect(text).toContain('Result 2');
    });
  });

  describe('messages without tool calls (should pass through)', () => {
    test('should not modify AI message without tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({ content: 'Hi there, how can I help?' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('Hello');
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[1].content).toBe('Hi there, how can I help?');
    });

    test('should preserve HumanMessages and other message types', () => {
      const messages = [
        new HumanMessage({ content: 'Question 1' }),
        new AIMessage({ content: 'Answer 1' }),
        new HumanMessage({ content: 'Question 2' }),
        new AIMessage({ content: 'Answer 2' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(4);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(HumanMessage);
      expect(result[3]).toBeInstanceOf(AIMessage);
    });
  });

  describe('mixed scenarios', () => {
    test('should handle mix of normal and tool-using messages', () => {
      const messages = [
        new HumanMessage({ content: 'First question' }),
        new AIMessage({ content: 'First answer without tools' }),
        new HumanMessage({ content: 'Second question' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            {
              id: 'call_abc',
              name: 'some_tool',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Tool result',
          tool_call_id: 'call_abc',
        }),
        new HumanMessage({ content: 'Third question' }),
        new AIMessage({ content: 'Third answer without tools' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Only the trailing sequence after the last HumanMessage is processed.
      // The AI+Tool at indices 3-4 is history — preserved as-is.
      // Last HumanMessage is at index 5 ("Third question").
      // Index 6 (AIMessage without tools) is in the trailing sequence but has
      // no tool calls, so it passes through.
      expect(result).toHaveLength(7);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(HumanMessage);
      expect(result[3]).toBeInstanceOf(AIMessage); // History — preserved
      expect(result[4]).toBeInstanceOf(ToolMessage); // History — preserved
      expect(result[5]).toBeInstanceOf(HumanMessage);
      expect(result[6]).toBeInstanceOf(AIMessage);
    });

    test('should handle multiple tool-using sequences', () => {
      const messages = [
        new HumanMessage({ content: 'Do task 1' }),
        new AIMessage({
          content: 'Doing task 1',
          tool_calls: [
            {
              id: 'call_1',
              name: 'tool1',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 1',
          tool_call_id: 'call_1',
        }),
        new HumanMessage({ content: 'Do task 2' }),
        new AIMessage({
          content: 'Doing task 2',
          tool_calls: [
            {
              id: 'call_2',
              name: 'tool2',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 2',
          tool_call_id: 'call_2',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Only the trailing sequence after the last HumanMessage is converted.
      // First tool sequence (indices 1-2) is history — preserved.
      // Last HumanMessage is at index 3 ("Do task 2").
      // Trailing sequence (indices 4-5) is converted to 1 HumanMessage.
      expect(result).toHaveLength(5);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('Do task 1');
      expect(result[1]).toBeInstanceOf(AIMessage); // History — preserved
      expect(result[2]).toBeInstanceOf(ToolMessage); // History — preserved
      expect(result[3]).toBeInstanceOf(HumanMessage);
      expect(result[3].content).toBe('Do task 2');
      expect(result[4]).toBeInstanceOf(HumanMessage); // Converted trailing sequence
      expect(getTextContent(result[4])).toContain('Doing task 2');
    });
  });

  describe('fast exit when last message is HumanMessage', () => {
    test('should return messages as-is when last message is a HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'first request' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new HumanMessage({ content: 'second request' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should return the exact same array reference — no processing done
      expect(result).toBe(messages);
      expect(result).toHaveLength(4);
    });

    test('should return messages as-is when only message is a HumanMessage', () => {
      const messages = [new HumanMessage({ content: 'hello' })];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toBe(messages);
    });

    test('should still process when last message is not a HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'do something' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should process — last message is ToolMessage, not HumanMessage
      expect(result).not.toBe(messages);
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('do something');
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(getTextContent(result[1])).toContain('[Previous agent context]');
    });
  });

  describe('edge cases', () => {
    test('should handle empty messages array', () => {
      const messages: never[] = [];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(0);
    });

    test('should handle AI message with empty content array', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({ content: [] }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(AIMessage);
    });

    test('should work with different providers', () => {
      const messages = [
        new AIMessage({
          content: 'Using tool',
          tool_calls: [
            {
              id: 'call_x',
              name: 'test',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result',
          tool_call_id: 'call_x',
        }),
      ];

      // Test with Anthropic
      const resultAnthropic = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );
      expect(resultAnthropic).toHaveLength(1);
      expect(resultAnthropic[0]).toBeInstanceOf(HumanMessage);

      // Test with Bedrock
      const resultBedrock = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK
      );
      expect(resultBedrock).toHaveLength(1);
      expect(resultBedrock[0]).toBeInstanceOf(HumanMessage);
    });

    test('should handle tool message without preceding AI message', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new ToolMessage({
          content: 'Unexpected tool result',
          tool_call_id: 'call_orphan',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should preserve both messages as-is since tool message has no preceding AI message with tools
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(ToolMessage);
    });

    test('should not emit an empty user message after the shared fold budget is exhausted', () => {
      const messages = [
        new HumanMessage({ content: 'Run both tools' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'first', name: 'query', args: {}, type: 'tool_call' },
          ],
        }),
        new ToolMessage({
          content: Array.from({ length: 60 }, () => ({
            type: 'text',
            text: 'x'.repeat(8_000),
          })),
          tool_call_id: 'first',
        }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'second', name: 'query', args: {}, type: 'tool_call' },
          ],
        }),
        new ToolMessage({
          content: 'second result',
          tool_call_id: 'second',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(getTextContent(result[1])).toContain(
        'additional folded context omitted'
      );
      expect(
        result
          .filter((message) => message instanceof HumanMessage)
          .every((message) =>
            typeof message.content === 'string'
              ? message.content.length > 0
              : message.content.length > 0
          )
      ).toBe(true);
    });
  });

  describe('image content preservation (token amplification fix)', () => {
    const FAKE_BASE64 =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk';

    /**
     * Reproduces the reported bug: a base64 image from an MCP tool, when
     * serialized by the old getBufferString() path, would become text tokens.
     * With the fix, the base64 data stays in a structured image block.
     */
    test('should not serialize base64 image as text (reported 174x token amplification)', () => {
      const LARGE_BASE64 = 'A'.repeat(10_000);

      const messages = [
        new HumanMessage({ content: 'Take a screenshot' }),
        new AIMessage({
          content: 'Taking a screenshot.',
          tool_calls: [
            {
              id: 'call_mcp',
              name: 'screenshot',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Screenshot captured (1280x720)' },
            {
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${LARGE_BASE64}` },
            },
          ],
          tool_call_id: 'call_mcp',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);

      const content = result[1].content as ExtendedMessageContent[];
      const textBlocks = content.filter((b) => b.type === 'text');
      const imageBlocks = content.filter((b) => b.type === 'image_url');

      expect(imageBlocks).toHaveLength(1);

      const allText = textBlocks.map((b) => String(b.text ?? '')).join('\n');
      expect(allText).toContain('[Previous agent context]');
      expect(allText).toContain('Screenshot captured');
      expect(allText).not.toContain(LARGE_BASE64);
      // Text must be orders of magnitude smaller than the image data
      expect(allText.length).toBeLessThan(LARGE_BASE64.length / 10);
    });

    test('should preserve image_url blocks from ToolMessage instead of serializing as text', () => {
      const messages = [
        new HumanMessage({ content: 'Take a screenshot' }),
        new AIMessage({
          content: 'Taking screenshot now.',
          tool_calls: [
            {
              id: 'call_ss',
              name: 'screenshot',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Screenshot captured' },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${FAKE_BASE64}`,
              },
            },
          ],
          tool_call_id: 'call_ss',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);

      // Content should be an array with structured blocks
      const content = result[1].content as ExtendedMessageContent[];
      expect(Array.isArray(content)).toBe(true);

      // Should have text block(s) and an image_url block
      const textBlocks = content.filter((b) => b.type === 'text');
      const imageBlocks = content.filter((b) => b.type === 'image_url');

      expect(textBlocks.length).toBeGreaterThanOrEqual(1);
      expect(imageBlocks).toHaveLength(1);

      // The image block should be preserved as-is (not serialized to text)
      const imageBlock = imageBlocks[0] as {
        type: string;
        image_url: { url: string };
      };
      expect(imageBlock.image_url.url).toContain(FAKE_BASE64);

      // The text should contain context info but NOT the base64 data
      const allText = textBlocks.map((b) => String(b.text ?? '')).join('\n');
      expect(allText).toContain('[Previous agent context]');
      expect(allText).toContain('Screenshot captured');
      expect(allText).not.toContain(FAKE_BASE64);
    });

    test('should preserve Anthropic-style image blocks from ToolMessage', () => {
      const messages = [
        new HumanMessage({ content: 'Take a screenshot' }),
        new AIMessage({
          content: 'Let me capture that.',
          tool_calls: [
            {
              id: 'call_ss2',
              name: 'screenshot',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Here is the screenshot' },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: FAKE_BASE64,
              },
            },
          ],
          tool_call_id: 'call_ss2',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const content = result[1].content as ExtendedMessageContent[];
      expect(Array.isArray(content)).toBe(true);

      const imageBlocks = content.filter((b) => b.type === 'image');
      expect(imageBlocks).toHaveLength(1);
      const imageBlock = imageBlocks[0] as {
        type: string;
        source: { data: string };
      };
      expect(imageBlock.source.data).toBe(FAKE_BASE64);

      // Text should not contain base64
      const allText = content
        .filter((b) => b.type === 'text')
        .map((b) => String(b.text ?? ''))
        .join('\n');
      expect(allText).not.toContain(FAKE_BASE64);
    });

    test('should handle multiple images across multiple ToolMessages', () => {
      const messages = [
        new HumanMessage({ content: 'Compare two pages' }),
        new AIMessage({
          content: 'Taking screenshots of both pages.',
          tool_calls: [
            {
              id: 'call_a',
              name: 'screenshot',
              args: { page: 'A' },
              type: 'tool_call' as const,
            },
            {
              id: 'call_b',
              name: 'screenshot',
              args: { page: 'B' },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Page A screenshot' },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,PAGE_A_DATA' },
            },
          ],
          tool_call_id: 'call_a',
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Page B screenshot' },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,PAGE_B_DATA' },
            },
          ],
          tool_call_id: 'call_b',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const content = result[1].content as ExtendedMessageContent[];
      const imageBlocks = content.filter((b) => b.type === 'image_url');
      expect(imageBlocks).toHaveLength(2);

      const allText = content
        .filter((b) => b.type === 'text')
        .map((b) => String(b.text ?? ''))
        .join('\n');
      expect(allText).toContain('Page A screenshot');
      expect(allText).toContain('Page B screenshot');
    });

    test('should still produce text-only content when no images are present', () => {
      const messages = [
        new HumanMessage({ content: 'Do something' }),
        new AIMessage({
          content: 'Doing it.',
          tool_calls: [
            {
              id: 'call_t',
              name: 'tool',
              args: { x: 1 },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'plain text result',
          tool_call_id: 'call_t',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const content = result[1].content as ExtendedMessageContent[];
      // When no images, should still be an array with a single text block
      expect(Array.isArray(content)).toBe(true);
      expect(content).toHaveLength(1);
      expect(content[0].type).toBe('text');
      expect(content[0].text).toContain('[Previous agent context]');
      expect(content[0].text).toContain('plain text result');
    });

    test('should not double-serialize when AIMessage has both content tool_use and tool_calls', () => {
      const messages = [
        new HumanMessage({ content: 'Search for something' }),
        new AIMessage({
          content: [
            { type: 'text', text: 'Searching...' },
            {
              type: 'tool_use',
              id: 'call_dual',
              name: 'search',
              input: { query: 'test' },
            },
          ],
          tool_calls: [
            {
              id: 'call_dual',
              name: 'search',
              args: { query: 'test' },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'Found 5 results',
          tool_call_id: 'call_dual',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const allText = getTextContent(result[1]);
      // Array content path serializes tool_use blocks but skips appendToolCalls
      expect(allText).not.toContain('[tool_call]');
      expect(allText).toContain('[tool_use]');
    });

    test('should serialize tool_calls when content is empty array (no tool_use blocks)', () => {
      const messages = [
        new HumanMessage({ content: 'Do something' }),
        new AIMessage({
          content: [],
          tool_calls: [
            {
              id: 'call_empty',
              name: 'some_tool',
              args: { x: 1 },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'tool result',
          tool_call_id: 'call_empty',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const allText = getTextContent(result[1]);
      // With empty content array, should fall back to tool_calls
      expect(allText).toContain('[tool_call]');
      expect(allText).toContain('some_tool');
    });

    test('should serialize unrecognized block types instead of dropping them', () => {
      const messages = [
        new HumanMessage({ content: 'Fetch resource' }),
        new AIMessage({
          content: 'Fetching.',
          tool_calls: [
            {
              id: 'call_res',
              name: 'fetch_resource',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Resource fetched' },
            {
              type: 'resource',
              resource: { uri: 'file:///data.csv', text: 'a,b,c' },
            },
          ],
          tool_call_id: 'call_res',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      const allText = getTextContent(result[1]);
      // The resource block should be serialized as text, not silently dropped.
      expect(allText).toContain('[resource]');
      expect(allText).toContain('data.csv');
    });

    test('should preserve image blocks when provider is Bedrock', () => {
      const messages = [
        new HumanMessage({ content: 'Screenshot' }),
        new AIMessage({
          content: 'Taking screenshot.',
          tool_calls: [
            {
              id: 'call_br',
              name: 'screenshot',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [
            { type: 'text', text: 'Captured' },
            {
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${FAKE_BASE64}` },
            },
          ],
          tool_call_id: 'call_br',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      const content = result[1].content as ExtendedMessageContent[];
      const imageBlocks = content.filter((b) => b.type === 'image_url');
      expect(imageBlocks).toHaveLength(1);
      const allText = getTextContent(result[1]);
      expect(allText).not.toContain(FAKE_BASE64);
    });

    test('should shallow-copy image blocks to prevent aliasing', () => {
      const originalImageBlock = {
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${FAKE_BASE64}` },
      };
      const messages = [
        new HumanMessage({ content: 'Screenshot' }),
        new AIMessage({
          content: 'Taking screenshot.',
          tool_calls: [
            {
              id: 'call_alias',
              name: 'screenshot',
              args: {},
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: [{ type: 'text', text: 'Captured' }, originalImageBlock],
          tool_call_id: 'call_alias',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      const content = result[1].content as ExtendedMessageContent[];
      const outputImageBlock = content.find((b) => b.type === 'image_url');
      // Should be a different object reference (shallow copy)
      expect(outputImageBlock).not.toBe(originalImageBlock);
    });
  });

  describe('runStartIndex (current-run boundary)', () => {
    /**
     * Claude is allowed to skip a thinking block before a tool_use (cf.
     * PR #116). When the agent's own first iteration produces an
     * `AI(tool_use, no thinking)`, the function would otherwise convert
     * it to a `[Previous agent context]` HumanMessage — polluting the
     * next iteration's prompt with text the model treats as suspicious
     * injected content. The model then ignores its own real prior tool
     * result and re-runs the tool to verify, often failing because the
     * subsequent sandbox doesn't have the file.
     *
     * The `runStartIndex` parameter tells the function which messages
     * are the agent's own in-run work: those at or after it must NEVER
     * be converted, even if no thinking block appears in the chain.
     */

    test('preserves the agent first-iteration AI(tool_use) when its index is at runStartIndex', () => {
      const messages = [
        new HumanMessage({ content: 'fetch the data' }),
        // No thinking block — Claude validly skipped it before tool_use
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c1', name: 'fetch', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'data', tool_call_id: 'c1' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK,
        undefined,
        /* runStartIndex */ 1
      );

      // All 3 preserved — the AI at index 1 is the agent's own work
      expect(result).toHaveLength(3);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect((result[1] as AIMessage).tool_calls).toHaveLength(1);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      // No placeholder leaked in
      expect(getTextContent(result[1])).not.toContain(
        '[Previous agent context]'
      );
    });

    test('preserves multiple in-run AI(tool_use) iterations without thinking blocks', () => {
      const messages = [
        new HumanMessage({ content: 'do work' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c1', name: 'step1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c2', name: 'step2', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r2', tool_call_id: 'c2' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK,
        undefined,
        /* runStartIndex */ 1
      );

      expect(result).toHaveLength(5);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[3]).toBeInstanceOf(AIMessage);
      // Neither AI was converted
      expect(getTextContent(result[1])).not.toContain(
        '[Previous agent context]'
      );
      expect(getTextContent(result[3])).not.toContain(
        '[Previous agent context]'
      );
    });

    test('still converts pre-runStartIndex history that lacks thinking blocks', () => {
      // Real handoff scenario: a prior non-thinking agent's tool calls
      // appear before this run started. They genuinely need the
      // placeholder (the legacy reason this function exists).
      const messages = [
        new HumanMessage({ content: 'first request' }),
        new AIMessage({
          content: 'using tool',
          tool_calls: [
            { id: 'old', name: 'legacy', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'old result', tool_call_id: 'old' }),
        // Current run starts here — say after a handoff. Index >= 3 is
        // the new agent's own work.
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK,
        undefined,
        /* runStartIndex */ 3
      );

      // The pre-run AI(tool_use)+Tool got converted to a placeholder
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(getTextContent(result[1])).toContain('[Previous agent context]');
    });

    test('falls back to chainHasThinkingBlock heuristic when runStartIndex is undefined (backward compat)', () => {
      const messages = [
        new HumanMessage({ content: 'do work' }),
        // No reasoning + no runStartIndex hint → still gets converted
        // (preserves the prior behavior for callers that haven't been
        // updated to pass the boundary).
        new AIMessage({
          content: 'using tool',
          tool_calls: [
            { id: 'c1', name: 'tool', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r', tool_call_id: 'c1' }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(getTextContent(result[1])).toContain('[Previous agent context]');
    });

    test('runStartIndex of 0 is honored (whole array is the current run)', () => {
      // Edge: a fresh run with no prior history at all. Everything is
      // in-run and must be preserved even without thinking blocks.
      const messages = [
        new HumanMessage({ content: 'do work' }),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'c1', name: 'tool', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r', tool_call_id: 'c1' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK,
        undefined,
        /* runStartIndex */ 0
      );

      expect(result).toHaveLength(3);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(getTextContent(result[1])).not.toContain(
        '[Previous agent context]'
      );
    });
  });
});

import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  serializeStructuredValueBounded,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  applyFadingCaps,
  projectToolCallInputs,
  serializeToolCallInput,
} from '@/messages/prune';
import {
  truncateToolInput,
  truncateToolResultContent,
} from '@/utils/truncation';
import { applyContextPruning } from '@/messages/contextPruning';

const emojiText = 'a🧠🍱🗳🪜'.repeat(600);
const caps = [
  0, 1, 2, 4, 13, 14, 63, 64, 99, 100, 101, 257, 258, 259, 300, 301, 400, 4095,
  4096, 4097,
];

function charCounter(message: BaseMessage): number {
  return typeof message.content === 'string'
    ? message.content.length
    : JSON.stringify(message.content).length;
}

function customToolMessage(input: string): AIMessage {
  return new AIMessage({
    content: '',
    response_metadata: {
      output: [
        { type: 'custom_tool_call', call_id: 'custom', name: 'shell', input },
      ],
    },
  });
}

function customToolInput(message: BaseMessage): string {
  return (
    (message as AIMessage).response_metadata.output as Array<{ input: string }>
  )[0].input;
}

describe('Unicode-safe tool truncation', () => {
  it.each(caps)(
    'keeps tool-result head/tail cuts well-formed at cap %i',
    (cap) => {
      const result = truncateToolResultContent(emojiText, cap);
      expect(result.isWellFormed()).toBe(true);
      expect(result.length).toBeLessThanOrEqual(cap);
    }
  );

  it.each(caps)('keeps legacy tool-input cuts well-formed at cap %i', (cap) => {
    const result = truncateToolInput(emojiText, cap);
    expect(result._truncated.isWellFormed()).toBe(true);
    expect(result._originalChars).toBe(emojiText.length);
  });

  it.each(caps)(
    'keeps bounded structured previews and registry prefixes well-formed at cap %i',
    (cap) => {
      const value = { payload: emojiText, end: emojiText };
      const exact = JSON.stringify(value);
      const result = serializeStructuredValueBounded(value, cap, cap);
      expect(result.content.isWellFormed()).toBe(true);
      expect(result.prefix.isWellFormed()).toBe(true);
      expect(result.content.length).toBeLessThanOrEqual(cap);
      expect(result.prefix.length).toBeLessThanOrEqual(cap);
      expect(exact.startsWith(result.prefix)).toBe(true);
      expect(result.originalChars).toBe(exact.length);
    }
  );

  it.each(caps)(
    'keeps dense text-block previews well-formed at cap %i',
    (cap) => {
      const content = [
        { type: 'text', text: emojiText },
        { type: 'text', text: emojiText },
      ];
      const result = serializeToolContentBounded(content, cap);
      expect(result.isWellFormed()).toBe(true);
      expect(result.length).toBeLessThanOrEqual(cap);
    }
  );

  it('preserves the exact serialized prefix when a collector cut lands inside an emoji', () => {
    const value = {
      payload: '🧠'.repeat(5_000),
      end: 'must not enter the prefix',
    };
    const exact = JSON.stringify(value);
    const result = serializeStructuredValueBounded(value, 100, 13);
    expect(result.prefix).toBe(exact.slice(0, 12));
    expect(result.prefix.isWellFormed()).toBe(true);
  });

  it('keeps structured input envelopes well-formed through every small budget and re-projection', () => {
    const input = { payload: emojiText };
    for (let cap = 4; cap <= 200; cap++) {
      const serialized = serializeToolCallInput(input, cap);
      const projected = JSON.parse(serialized) as {
        _inputPrefix?: string;
        _originalChars?: number;
      };
      expect(serialized.length).toBeLessThanOrEqual(cap);
      expect(projected._inputPrefix?.isWellFormed() ?? true).toBe(true);
      expect(serializeToolCallInput(projected, cap)).toBe(serialized);
      if (projected._inputPrefix != null) {
        expect(projected._originalChars).toBe(JSON.stringify(input).length);
      }
    }
  });

  it('keeps custom-tool input marker and markerless cuts well-formed without mutating history', () => {
    const message = customToolMessage(emojiText);
    for (let cap = 4; cap <= 100; cap++) {
      const projected = projectToolCallInputs([message], cap);
      const input = customToolInput(projected[0]);
      expect(input.isWellFormed()).toBe(true);
      expect(input.length).toBeLessThanOrEqual(cap);
      if (cap > 50) {
        expect(input).toContain(
          `\n… [shortened; call completed: ${emojiText.length} chars]`
        );
        const shrunk = projectToolCallInputs(projected, 50);
        const direct = projectToolCallInputs([message], 50);
        expect(customToolInput(shrunk[0])).toBe(customToolInput(direct[0]));
      }
      expect(projectToolCallInputs(projected, cap)).toBe(projected);
      const shrunk = projectToolCallInputs(projected, 4);
      expect(customToolInput(shrunk[0]).isWellFormed()).toBe(true);
    }
    expect(customToolInput(message)).toBe(emojiText);
  });

  it.each(['fresh', 'consumed'] as const)(
    'keeps %s fading projections well-formed, stable, and separate from canonical history',
    (state) => {
      const original = new ToolMessage({
        content: emojiText,
        tool_call_id: 'query',
      });
      const canonicalMessages: BaseMessage[] = [
        original,
        new AIMessage('Read the result.'),
      ];
      const messages = [...canonicalMessages];
      const indexTokenCountMap: Record<string, number | undefined> = {};
      const params = {
        messages,
        canonicalMessages,
        indexTokenCountMap,
        tokenCounter: charCounter,
        caps: {
          resultChars: 101,
          consumedChars: 101,
          inputChars: Number.POSITIVE_INFINITY,
        },
        masked: state === 'consumed',
      };
      applyFadingCaps(params);
      const content = messages[0].content as string;
      expect(content.isWellFormed()).toBe(true);
      expect(content.length).toBeLessThanOrEqual(101);
      expect(indexTokenCountMap[0]).toBe(content.length);
      expect(original.content).toBe(emojiText);
      applyFadingCaps(params);
      expect(messages[0].content).toBe(content);
    }
  );

  it('keeps both position-based soft-trim boundaries well-formed', () => {
    const content = `a🧠${'x'.repeat(500)}🧠z`;
    const original = new ToolMessage({ content, tool_call_id: 'query' });
    const messages: BaseMessage[] = [
      new HumanMessage('query'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'query', name: 'query', args: {} }],
      }),
      original,
      new AIMessage('Read it.'),
      new HumanMessage('continue'),
      new AIMessage('Ready.'),
      new HumanMessage('next'),
    ];
    const indexTokenCountMap: Record<string, number | undefined> = {};
    const result = applyContextPruning({
      messages,
      indexTokenCountMap,
      tokenCounter: charCounter,
      config: {
        enabled: true,
        keepLastAssistants: 1,
        softTrimRatio: 0,
        minPrunableToolChars: 1,
        softTrim: { maxChars: 100, headChars: 2, tailChars: 2 },
        hardClear: { enabled: false },
      },
    });
    const trimmed = messages[2].content as string;
    expect(result.softTrimmed).toBe(1);
    expect(trimmed.isWellFormed()).toBe(true);
    expect(trimmed.startsWith('a\n\n')).toBe(true);
    expect(trimmed.endsWith('\n\nz')).toBe(true);
    expect(indexTokenCountMap[2]).toBe(trimmed.length);
    expect(original.content).toBe(content);
  });
});

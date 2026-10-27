import {
  isReasoningContentBlock,
  hasMeaningfulReasoningPayload,
  hasMeaningfulReasoningContentBlock,
} from './reasoningTypes';

describe('isReasoningContentBlock', () => {
  it.each([
    'think',
    'thinking',
    'thinking_delta',
    'redacted_thinking',
    'reasoning',
    'reasoning-delta',
    'reasoning_content',
  ])('recognizes %s blocks', (type) => {
    expect(isReasoningContentBlock({ type })).toBe(true);
  });

  it.each(['text', 'thinker', 'reasoning_label', 'reasoning.text', undefined])(
    'rejects %s blocks',
    (type) => {
      expect(isReasoningContentBlock({ type })).toBe(false);
    }
  );
});

describe('hasMeaningfulReasoningContentBlock', () => {
  it.each([
    { type: 'think' },
    { type: 'think', think: '' },
    { type: 'think', think: ' \n\t' },
    { type: 'thinking', thinking: '' },
    { type: 'thinking', thinking: ' \n', signature: '' },
    { type: 'thinking_delta', thinking: '\t' },
    { type: 'redacted_thinking', data: '' },
    { type: 'redacted_thinking', data: ' ' },
    { type: 'reasoning', reasoning: '' },
    { type: 'reasoning', text: ' \n' },
    { type: 'reasoning-delta', reasoning: '\t' },
    { type: 'reasoning_content', reasoningText: { text: '' } },
    { type: 'reasoning_content', reasoningText: { text: ' \n' } },
    { type: 'reasoning_content', reasoningText: {} },
    { type: 'reasoning_content', reasoningText: { signature: ' ' } },
    { type: 'reasoning_content', redactedContent: '' },
    { type: 'reasoning_content', redactedContent: new Uint8Array() },
    { type: 'reasoning', summary: [] },
    { type: 'reasoning', summary: [{ type: 'summary_text', text: ' ' }] },
    { type: 'reasoning', content: [{ type: 'reasoning_text', text: '' }] },
    { type: 'reasoning', summary: [null, {}, { type: 'summary_text' }] },
    { type: 'reasoning', encrypted_content: ' ' },
    { type: 'reasoning', id: 'rs_1', status: 'completed' },
    { type: 'reasoning', metadata: { text: 'not a reasoning payload' } },
  ])('rejects empty evidence without changing classification: %j', (block) => {
    expect(isReasoningContentBlock(block)).toBe(true);
    expect(hasMeaningfulReasoningContentBlock(block)).toBe(false);
  });

  it.each([
    { type: 'think', think: 'thought' },
    { type: 'thinking', thinking: 'thought' },
    { type: 'thinking', thinking: '', signature: 'signature' },
    { type: 'thinking_delta', thinking: 'thought' },
    { type: 'redacted_thinking', data: 'opaque' },
    { type: 'reasoning', reasoning: 'thought' },
    { type: 'reasoning', text: 'standardized thought' },
    { type: 'reasoning-delta', reasoning: 'thought' },
    { type: 'reasoning_content', reasoningText: { text: 'thought' } },
    { type: 'reasoning_content', reasoningText: { signature: 'signature' } },
    { type: 'reasoning_content', redactedContent: 'YQ==' },
    { type: 'reasoning_content', redactedContent: new Uint8Array([0]) },
    { type: 'reasoning', summary: [{ type: 'summary_text', text: 'thought' }] },
    { type: 'reasoning', summary: [{ text: '' }, { text: 'later thought' }] },
    {
      type: 'reasoning',
      content: [{ type: 'reasoning_text', text: 'thought' }],
    },
    { type: 'reasoning', summary: [], encrypted_content: 'opaque' },
    { type: 'reasoning', reasoning: '', signature: 'signature' },
  ])('accepts fulfilled provider payloads: %j', (block) => {
    expect(hasMeaningfulReasoningContentBlock(block)).toBe(true);
  });

  it.each(['text', 'tool_call', 'reasoning.text', undefined])(
    'still requires a reasoning content kind: %s',
    (type) => {
      expect(
        hasMeaningfulReasoningContentBlock(
          Object.assign({ type }, { text: 'not reasoning' })
        )
      ).toBe(false);
    }
  );
});

describe('hasMeaningfulReasoningPayload', () => {
  it.each([
    null,
    undefined,
    false,
    0,
    [],
    {},
    '',
    ' \n',
    { text: {} },
    { summary: 'text' },
  ])('rejects empty or malformed payloads: %j', (payload) => {
    expect(hasMeaningfulReasoningPayload(payload)).toBe(false);
  });
});

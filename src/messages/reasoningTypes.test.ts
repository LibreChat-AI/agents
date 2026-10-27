import { isReasoningContentBlock } from './reasoningTypes';

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

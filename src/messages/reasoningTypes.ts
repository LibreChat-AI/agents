const REASONING_CONTENT_BLOCK_TYPES: ReadonlySet<string> = new Set([
  'think',
  'thinking',
  'thinking_delta',
  'redacted_thinking',
  'reasoning',
  'reasoning-delta',
  'reasoning_content',
]);

/** Identifies every persisted or streamed reasoning content-block shape. */
export function isReasoningContentBlock(block: { type?: unknown }): boolean {
  return (
    typeof block.type === 'string' &&
    REASONING_CONTENT_BLOCK_TYPES.has(block.type)
  );
}

export function isAnthropicThinkingContentBlock(block: {
  type?: unknown;
}): boolean {
  return block.type === 'thinking';
}

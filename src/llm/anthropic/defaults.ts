const CLAUDE_GENERATION_PATTERN = /^claude-[a-z]+-(\d+)(?:-|$)/;
const CURRENT_CLAUDE_DEFAULT_MAX_TOKENS = 16384;

/**
 * The `max_tokens` floor for current-generation Claude models
 * (`claude-<family>-<N>`, N >= 4) when the caller sets none.
 *
 * Upstream's default table is keyed by model prefix and falls back to 4096 for
 * any Claude name it does not list yet, which caps newer models (and thinking,
 * which shares that budget) far below the 16384 it gives every other Claude 4+
 * model. Raising the default further would trip the SDK's non-streaming guard,
 * which rejects `max_tokens` above ~21K.
 */
export function getCurrentClaudeDefaultMaxTokens(
  model: string | undefined
): number | undefined {
  const generation = CLAUDE_GENERATION_PATTERN.exec(model ?? '')?.[1];
  if (generation == null || Number(generation) < 4) {
    return undefined;
  }
  return CURRENT_CLAUDE_DEFAULT_MAX_TOKENS;
}

/**
 * Ingestion-time and pre-flight truncation utilities for tool results.
 *
 * Prevents oversized tool outputs from entering the message array and
 * consuming the entire context window.
 */

/** Slices by UTF-16 code units, dropping orphaned surrogate halves at either edge. */
export function sliceWithoutSplittingSurrogates(
  value: string,
  start: number,
  end?: number
): string {
  const sliced = value.slice(start, end);
  const first = sliced.charCodeAt(0);
  const last = sliced.charCodeAt(sliced.length - 1);
  const trimStart = first >= 0xdc00 && first <= 0xdfff;
  const trimEnd = last >= 0xd800 && last <= 0xdbff;
  return trimStart || trimEnd
    ? sliced.slice(trimStart ? 1 : 0, sliced.length - (trimEnd ? 1 : 0))
    : sliced;
}

/**
 * Absolute hard cap on tool result length (characters).
 * Even if the model has a 1M-token context, a single tool result
 * larger than this is almost certainly a bug (e.g., dumping a binary file).
 */
export const HARD_MAX_TOOL_RESULT_CHARS = 400_000;

/**
 * Absolute hard cap on the aggregate size (characters) of all registered
 * tool outputs kept for `{{tool<i>turn<n>}}` substitution. Set at 5 MB
 * because the registry stores *raw, untruncated* tool output — full
 * fidelity for piping into downstream bash/jq — so the budget needs
 * enough headroom to keep a handful of large responses without
 * ballooning unbounded.
 */
export const HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE = 5_000_000;

/**
 * Computes the dynamic max tool result size based on the model's context window.
 * Uses 30% of the context window (in estimated characters, ~4 chars/token)
 * capped at HARD_MAX_TOOL_RESULT_CHARS.
 *
 * @param contextWindowTokens - The model's max context tokens (optional).
 * @returns Maximum allowed characters for a single tool result.
 */
export function calculateMaxToolResultChars(
  contextWindowTokens?: number
): number {
  if (contextWindowTokens == null || contextWindowTokens <= 0) {
    return HARD_MAX_TOOL_RESULT_CHARS;
  }
  return Math.min(
    Math.floor(contextWindowTokens * 0.3) * 4,
    HARD_MAX_TOOL_RESULT_CHARS
  );
}

/**
 * Computes the default aggregate size (characters) for the tool output
 * reference registry based on the per-output budget. Mirrors
 * `calculateMaxToolResultChars`'s shape: a multiple of the per-output
 * cap, clamped to `HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE`.
 *
 * @param maxOutputSize - Per-output maximum characters (e.g., the
 *   ToolNode's `maxToolResultChars`). When omitted or non-positive,
 *   falls back to the absolute total cap.
 * @returns Maximum total characters retained across the registry.
 */
export function calculateMaxTotalToolOutputSize(
  maxOutputSize?: number
): number {
  if (maxOutputSize == null || maxOutputSize <= 0) {
    return HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE;
  }
  return Math.min(maxOutputSize * 2, HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE);
}

/**
 * Truncates a tool-call input (the arguments/payload of a tool_use block)
 * using head+tail strategy. Returns an object with `_truncated` (the
 * truncated string) and `_originalChars` (for diagnostics).
 *
 * Accepts any type — objects are JSON-serialized before truncation.
 *
 * @param input - The tool input (string, object, etc.).
 * @param maxChars - Maximum allowed characters.
 */
export function truncateToolInput(
  input: unknown,
  maxChars: number
): { _truncated: string; _originalChars: number } {
  const serialized = typeof input === 'string' ? input : JSON.stringify(input);
  if (serialized.length <= maxChars) {
    return { _truncated: serialized, _originalChars: serialized.length };
  }
  const indicator = `\n… [truncated: ${serialized.length} chars exceeded ${maxChars} limit] …\n`;
  const available = maxChars - indicator.length;

  if (available < 100) {
    return {
      _truncated:
        sliceWithoutSplittingSurrogates(serialized, 0, maxChars) +
        indicator.trimEnd(),
      _originalChars: serialized.length,
    };
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;

  return {
    _truncated:
      sliceWithoutSplittingSurrogates(serialized, 0, headSize) +
      indicator +
      sliceWithoutSplittingSurrogates(serialized, serialized.length - tailSize),
    _originalChars: serialized.length,
  };
}

/**
 * Truncates tool result content that exceeds `maxChars` using a head+tail
 * strategy. Keeps the beginning (structure/headers) and end (return value /
 * conclusion) of the content so the model retains both the opening context
 * and the final outcome.
 *
 * Head gets ~70% of the budget, tail gets ~30%. Falls back to head-only
 * when the budget is too small for a meaningful tail.
 *
 * @param content - The tool result string content.
 * @param maxChars - Maximum allowed characters.
 * @returns The (possibly truncated) content string.
 */
export function truncateToolResultContent(
  content: string,
  maxChars: number
): string {
  if (content.length <= maxChars) {
    return content;
  }

  const indicator = `\n\n… [truncated: ${content.length} chars exceeded ${maxChars} limit] …\n\n`;
  const available = maxChars - indicator.length;
  if (available <= 0) {
    return sliceWithoutSplittingSurrogates(content, 0, maxChars);
  }

  // When budget is too small for a meaningful tail, fall back to head-only
  if (available < 200) {
    return (
      sliceWithoutSplittingSurrogates(content, 0, available) +
      indicator.trimEnd()
    );
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;

  // Try to break at newline boundaries for cleaner output
  let headEnd = headSize;
  const headNewline = content.lastIndexOf('\n', headSize);
  if (headNewline > headSize - 200 && headNewline > 0) {
    headEnd = headNewline;
  }

  let tailStart = content.length - tailSize;
  const tailNewline = content.indexOf('\n', tailStart);
  if (tailNewline > 0 && tailNewline < tailStart + 200) {
    tailStart = tailNewline + 1;
  }

  return (
    sliceWithoutSplittingSurrogates(content, 0, headEnd) +
    indicator +
    sliceWithoutSplittingSurrogates(content, tailStart)
  );
}

/** Absolute hard cap on a single tool-call input (characters). */
export const HARD_MAX_TOOL_CALL_INPUT_CHARS = 200_000;

/** Smallest JSON value a truncated tool-call input can shrink to. */
export const MIN_JSON_VALUE_CHARS = 4;

/**
 * Computes the max tool-call input size for a context window: 15 % of the
 * window in estimated characters (~4 chars/token), capped at 200K.
 */
export function calculateMaxToolCallInputChars(
  maxContextTokens?: number
): number {
  if (maxContextTokens == null || maxContextTokens <= 0) {
    return HARD_MAX_TOOL_CALL_INPUT_CHARS;
  }
  return Math.max(
    MIN_JSON_VALUE_CHARS,
    Math.min(
      Math.floor(maxContextTokens * 0.15) * 4,
      HARD_MAX_TOOL_CALL_INPUT_CHARS
    )
  );
}

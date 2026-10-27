/**
 * Context overflow error detection.
 *
 * Providers disagree on how they report "your input is bigger than I can
 * take" — the class thrown, the HTTP status, whether numbers are reported,
 * and even whether it arrives as an HTTP error at all. Every pattern below
 * was captured from a live over-limit request (see
 * `src/scripts/context-overflow-probe.ts` and
 * `docs/context-overflow-signatures.md`); nothing here is guessed.
 *
 * Consumed by the graph's overflow recovery loop, which converts a detection
 * into a forced summarization pass instead of surfacing the error.
 */
import { ContextOverflowError } from '@langchain/core/errors';
import type { ProviderName } from '@/types';

/**
 * Why the request was rejected. Both kinds are fixed by shrinking the
 * prompt, which is what makes them recoverable; they are distinguished
 * because only `context_window` tells us the model's true window.
 */
export type ContextOverflowKind =
  /** Input exceeded the model's context window. */
  | 'context_window'
  /**
   * A single request exceeded a per-minute token allowance. Waiting cannot
   * help — the request can never fit the bucket — so this is a payload
   * problem wearing a 429, not throttling.
   */
  | 'request_too_large';

export interface ContextOverflowInfo {
  kind: ContextOverflowKind;
  /** Ceiling the provider reported, when it named one. */
  limitTokens?: number;
  /**
   * Token count the provider attributed to the whole request. Several
   * providers fold the requested completion allowance into this number, so it
   * is not interchangeable with the prompt size.
   */
  requestedTokens?: number;
  /**
   * The prompt alone, counted by the provider — set only when the provider
   * distinguished input from output, either by reporting an input-only figure
   * or by breaking the total down. Callers comparing provider counts against
   * their own prompt estimate must use this and not `requestedTokens`, whose
   * completion component would inflate the comparison.
   */
  promptTokens?: number;
  /** Which layer produced the verdict. Surfaced in logs and asserted in tests. */
  source: 'langchain' | 'pattern';
  provider?: ProviderName;
}

interface OverflowPattern {
  readonly kind: ContextOverflowKind;
  readonly re: RegExp;
  readonly limitGroup?: number;
  readonly requestedGroup?: number;
  /**
   * Whether `requestedGroup` counts the prompt alone. Providers that report a
   * combined input+completion total leave this false, and their number is
   * never used as a prompt measurement.
   */
  readonly requestedIsPromptOnly?: boolean;
  /**
   * Marks a signature that is consistent with overflow but not exclusive to
   * it, so it only counts when the caller can corroborate that the prompt was
   * actually near the budget. Without corroboration the error propagates
   * untouched rather than triggering a needless compaction.
   */
  readonly requiresContextPressure?: boolean;
}

/**
 * Fraction of the believed budget the prompt must reach before an ambiguous
 * provider error is read as overflow. Well above normal traffic, low enough to
 * catch the case the budget itself was miscalibrated.
 */
const CONTEXT_PRESSURE_RATIO = 0.8;

export interface ContextOverflowContext {
  provider?: ProviderName;
  /** Our own estimate of the prompt size for the call that failed. */
  estimatedPromptTokens?: number;
  /** The budget we believed applied when we built that prompt. */
  maxContextTokens?: number;
}

/**
 * Ordered most-specific first; the first match wins. Patterns that capture
 * both numbers come before the bare-phrase fallbacks for the same provider so
 * a recoverable error still yields the numbers when the provider reported
 * them.
 */
const OVERFLOW_PATTERNS: readonly OverflowPattern[] = [
  /** Anthropic, and Bedrock's passthrough of the same upstream. */
  {
    kind: 'context_window',
    re: /prompt is too long:\s*(\d+)\s*tokens\s*>\s*(\d+)\s*maximum/i,
    requestedGroup: 1,
    limitGroup: 2,
    requestedIsPromptOnly: true,
  },
  /**
   * OpenAI's own wording, which measures the messages and nothing else.
   * Ordered ahead of the shared sentence below so the prompt-only reading is
   * preferred when OpenAI is the one answering.
   */
  {
    kind: 'context_window',
    re: /maximum context length is\s*(\d+)\s*tokens\.\s*however,\s*your messages resulted in\s*(\d+)/i,
    limitGroup: 1,
    requestedGroup: 2,
    requestedIsPromptOnly: true,
  },
  /**
   * OpenRouter (`you requested about`) and DeepSeek (`you requested`). Their
   * total folds in the completion allowance — both then break it down in
   * parentheses, which `PROMPT_ONLY_BREAKDOWN_RE` recovers.
   */
  {
    kind: 'context_window',
    re: /maximum context length is\s*(\d+)\s*tokens\.\s*however,\s*you requested(?:\s*about)?\s*(\d+)/i,
    limitGroup: 1,
    requestedGroup: 2,
  },
  /**
   * xAI. Says "prompt length" rather than "context length", and does not say
   * whether the count it quotes includes the completion allowance — so it is
   * not trusted as a prompt measurement.
   */
  {
    kind: 'context_window',
    re: /maximum prompt length is\s*(\d+)\s*(?:tokens\s*)?but the request contains\s*(\d+)\s*tokens/i,
    limitGroup: 1,
    requestedGroup: 2,
  },
  /** Mistral. */
  {
    kind: 'context_window',
    re: /prompt contains\s*(\d+)\s*tokens[^.]*?too large for model with\s*(\d+)\s*maximum context length/i,
    requestedGroup: 1,
    limitGroup: 2,
    requestedIsPromptOnly: true,
  },
  /** Google Gemini / Vertex. Reports the ceiling only. */
  {
    kind: 'context_window',
    re: /input token count exceeds the maximum number of tokens allowed\s*\((\d+)\)/i,
    limitGroup: 1,
  },
  /**
   * OpenAI's token-bucket rejection. Numbers are validated by the caller:
   * `Requested >= Limit` means no retry can ever succeed.
   */
  {
    kind: 'request_too_large',
    re: /request too large[\s\S]*?limit\s*(\d+),\s*requested\s*(\d+)/i,
    limitGroup: 1,
    requestedGroup: 2,
  },
  /** Bedrock (Llama upstream) and any provider naming the ceiling alone. */
  {
    kind: 'context_window',
    re: /(?:maximum context length|maximum prompt length) is\s*(\d+)\s*tokens/i,
    limitGroup: 1,
  },
  /** Bedrock (Nova upstream). */
  {
    kind: 'context_window',
    re: /number of input tokens exceeds maximum length|input tokens exceeded/i,
  },
  /** Bedrock (Claude Sonnet upstream). */
  {
    kind: 'context_window',
    re: /\binput (?:is )?too long(?: for requested model)?\b/i,
  },
  /** OpenAI-compatible error code, and the phrases LangChain itself keys on. */
  {
    kind: 'context_window',
    re: /context[_ ]length[_ ]exceeded|input tokens exceed the configured limit|exceeds the context window|exceeds model context window/i,
  },
  /** Generic long-tail phrasings observed across OpenAI-compatible gateways. */
  {
    kind: 'context_window',
    re: /prompt is too long|too large for model|reduce the length of (?:the |your )?(?:messages|prompt)/i,
  },
  /**
   * Vertex AI. Its gaxios path discards the API error document, so an
   * over-limit prompt arrives as a bare status line with no reason —
   * identical to every other 400 from the same endpoint. Only a corroborated
   * near-budget prompt makes this readable as overflow.
   */
  {
    kind: 'context_window',
    re: /google request failed with status code 400(?!\s*:)/i,
    requiresContextPressure: true,
  },
] as const;

/**
 * Errors that mention size or limits but are NOT recoverable by compaction.
 * Checked before the positive patterns.
 *
 * Note the deliberate precision: OpenAI's *recoverable* "Request too large"
 * body links to `platform.openai.com/account/rate-limits`, so a loose
 * /rate.?limit/ test would discard the very case this module exists to catch.
 * URLs are stripped from the haystack before matching, and genuine throttling
 * is matched on its own distinct phrasing.
 */
const NON_RECOVERABLE_RE =
  /rate limit reached|requests per (?:min|day)|\brpm\b|too many requests|insufficient[_ ]quota|quota exceeded|billing|payment required|invalid[_ ]api[_ ]key|authentication|unauthorized|permission denied|forbidden/i;

/**
 * Output-cap complaints. `max_tokens` appears in both families, so these are
 * matched on the surrounding grammar rather than the bare parameter name.
 */
const OUTPUT_LIMIT_RE =
  /max_?(?:completion_?)?tokens\s*(?:must be|is too|cannot|exceeds|too large|greater than)|maximum number of output tokens|max_tokens.*less than or equal/i;

/**
 * A gateway can report this disjunction without saying whether the prompt or
 * requested output caused the rejection. Compaction is unsafe in that case:
 * shrinking a healthy prompt cannot fix an invalid output allowance.
 */
const AMBIGUOUS_CONTEXT_OR_OUTPUT_RE =
  /(?:context window[\s\S]{0,80}\bor\b[\s\S]{0,80}max(?:imum)? output|max(?:imum)? output[\s\S]{0,80}\bor\b[\s\S]{0,80}context window)/i;

/**
 * Recovers the input-only figure from providers that quote a combined total
 * and then break it down — OpenRouter's "(56811 of text input, 16 in the
 * output)" and DeepSeek's "(1179652 in the messages, 16 in the completion)".
 */
const PROMPT_ONLY_BREAKDOWN_RE =
  /\(\s*(\d+)\s*(?:of\s+text\s+input|in\s+the\s+messages|of\s+input|input\s+tokens)\b/i;

/** Broader hints for the deliberately fuzzy `isLikelyContextOverflowError`. */
const CONTEXT_OVERFLOW_HINT_RE =
  /413|payload too large|content_too_large|request entity too large|too many tokens|token count.*exceed|exceed.*token count/i;

const MAX_CAUSE_DEPTH = 4;

interface NestedErrorShape {
  name?: string;
  message?: unknown;
  code?: unknown;
  type?: unknown;
  status?: unknown;
  reason?: unknown;
  error?: unknown;
  cause?: unknown;
  $metadata?: unknown;
  body?: unknown;
  response?: unknown;
  /** gaxios-style clients (Vertex AI) put the API error document here. */
  data?: unknown;
}

function asRecord(value: unknown): NestedErrorShape | undefined {
  return typeof value === 'object' && value !== null
    ? (value as NestedErrorShape)
    : undefined;
}

/**
 * Flattens an error into a single searchable string.
 *
 * Necessary because providers bury the useful sentence at different depths:
 * Anthropic puts a JSON document in `message`, Mistral puts one in `body`,
 * LangChain's `ContextOverflowError` keeps the original API error under
 * `cause`, and the OpenAI SDK nests the body under `error`.
 */
function collectErrorText(error: unknown, depth = 0): string {
  if (error == null || depth > MAX_CAUSE_DEPTH) {
    return '';
  }
  if (typeof error === 'string') {
    return error;
  }
  const record = asRecord(error);
  if (record == null) {
    return String(error);
  }

  const parts: string[] = [];
  if (typeof record.message === 'string') {
    parts.push(record.message);
  }
  for (const reason of [
    record.code,
    record.type,
    record.status,
    record.reason,
  ]) {
    if (typeof reason === 'string') {
      parts.push(reason);
    }
  }
  for (const nested of [
    record.error,
    record.cause,
    record.body,
    record.data,
    record.response,
  ]) {
    if (nested == null) {
      continue;
    }
    parts.push(
      typeof nested === 'string' ? nested : collectErrorText(nested, depth + 1)
    );
  }
  if (parts.length === 0) {
    try {
      /** Non-objects returned above, so this always yields a string. */
      return JSON.stringify(error);
    } catch {
      return String(error);
    }
  }
  return parts.join(' ');
}

/** Strips URLs so their path segments cannot trip the negative matchers. */
function stripUrls(text: string): string {
  return text.replace(/https?:\/\/\S+/gi, ' ');
}

function readNumber(
  match: RegExpMatchArray,
  group?: number
): number | undefined {
  if (group == null) {
    return undefined;
  }
  const parsed = Number(match[group]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

/**
 * The provider's count of the prompt alone: its own breakdown when it gave
 * one, otherwise the quoted total but only for providers that quote the
 * prompt rather than the whole request. Returns undefined when the number on
 * offer includes the completion allowance, since treating that as a prompt
 * measurement would overstate how much the prompt has to shrink.
 */
function resolvePromptTokens(
  haystack: string,
  pattern: OverflowPattern,
  requestedTokens: number | undefined
): number | undefined {
  const breakdown = haystack.match(PROMPT_ONLY_BREAKDOWN_RE);
  if (breakdown != null) {
    const parsed = Number(breakdown[1]);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return pattern.requestedIsPromptOnly === true ? requestedTokens : undefined;
}

/**
 * True when the caller's own accounting says the failed prompt was close
 * enough to the budget that an otherwise ambiguous provider error is best
 * explained by overflow.
 */
function hasContextPressure(context?: ContextOverflowContext): boolean {
  const estimated = context?.estimatedPromptTokens;
  const budget = context?.maxContextTokens;
  if (
    estimated == null ||
    budget == null ||
    !Number.isFinite(estimated) ||
    !Number.isFinite(budget) ||
    budget <= 0
  ) {
    return false;
  }
  return estimated / budget >= CONTEXT_PRESSURE_RATIO;
}

function isLangChainOverflowError(error: unknown): boolean {
  if (ContextOverflowError.isInstance(error)) {
    return true;
  }
  /** Duplicate `@langchain/core` copies break branding; the name survives. */
  const record = asRecord(error) as
    | { name?: unknown; lc_error_code?: unknown }
    | undefined;
  return (
    record?.name === 'ContextOverflowError' ||
    record?.lc_error_code === 'CONTEXT_OVERFLOW'
  );
}

/**
 * Extracts a human-readable error message from an unknown error value.
 */
export function extractErrorMessage(error: unknown): string {
  if (error == null) {
    return '';
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error instanceof Error) {
    return error.message;
  }
  const record = asRecord(error);
  if (record == null) {
    /** Functions and symbols serialize to nothing; describe them instead. */
    return String(error);
  }
  if (typeof record.message === 'string') {
    return record.message;
  }
  if (typeof record.error === 'string') {
    return record.error;
  }
  const nested = asRecord(record.error);
  if (typeof nested?.message === 'string') {
    return nested.message;
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function hasProviderCancellation(error: unknown, depth: number): boolean {
  if (error == null || depth > MAX_CAUSE_DEPTH) {
    return false;
  }
  const record = asRecord(error);
  const text = `${record?.name ?? ''} ${record?.code ?? ''} ${record?.type ?? ''} ${extractErrorMessage(error)}`;
  if (
    /^(?:AbortError|APIUserAbortError|CanceledError|CancelledError)$/.test(
      record?.name ?? ''
    ) ||
    /^(?:ABORT_ERR|ERR_CANCELED)$/.test(String(record?.code ?? '')) ||
    /\bAbortError\b|\buser (?:aborted|cancelled|canceled)\b|\b(?:request|operation) (?:was )?(?:aborted|cancelled|canceled)\b/i.test(
      text
    )
  ) {
    return true;
  }
  if (record == null) {
    return false;
  }
  for (const nested of [
    record.cause,
    record.error,
    record.body,
    record.data,
    record.response,
    record.$metadata,
  ]) {
    if (hasProviderCancellation(nested, depth + 1)) {
      return true;
    }
  }
  return false;
}

/** Recognizes cancellation names, codes, and messages in bounded error chains. */
export function isProviderCancellation(error: unknown): boolean {
  return hasProviderCancellation(error, 0);
}

/**
 * Classifies a provider error as a recoverable context overflow, returning
 * whatever the provider disclosed about the limit and the request size.
 *
 * Returns `null` for anything that compaction cannot fix — genuine
 * throttling, auth, quota, and output-token-cap errors all mention limits
 * and must not be mistaken for overflow.
 *
 * The reported numbers are the point of the return value: they let the
 * caller retarget the token budget to the provider's real ceiling instead of
 * retrying blindly against a configured value that was evidently wrong.
 */
export function getContextOverflowInfo(
  error: unknown,
  context?: ContextOverflowContext
): ContextOverflowInfo | null {
  const provider = context?.provider;
  const haystack = stripUrls(collectErrorText(error));
  if (haystack === '') {
    return null;
  }

  if (AMBIGUOUS_CONTEXT_OR_OUTPUT_RE.test(haystack)) {
    return null;
  }

  if (OUTPUT_LIMIT_RE.test(haystack)) {
    return null;
  }

  const langChainFlagged = isLangChainOverflowError(error);
  if (!langChainFlagged && NON_RECOVERABLE_RE.test(haystack)) {
    return null;
  }

  const underContextPressure = hasContextPressure(context);

  for (const pattern of OVERFLOW_PATTERNS) {
    const match = haystack.match(pattern.re);
    if (match == null) {
      continue;
    }
    if (pattern.requiresContextPressure === true && !underContextPressure) {
      continue;
    }
    const limitTokens = readNumber(match, pattern.limitGroup);
    const requestedTokens = readNumber(match, pattern.requestedGroup);

    /**
     * A token-bucket rejection is only unrecoverable-by-waiting when the
     * request alone overruns the bucket. When it merely fills it, the account
     * was busy and the request will fit once the window drains — so equality
     * belongs on the retry side, not the compaction side. Losing conversation
     * history to a temporarily busy account is the worse error.
     */
    if (
      pattern.kind === 'request_too_large' &&
      limitTokens != null &&
      requestedTokens != null &&
      requestedTokens <= limitTokens
    ) {
      return null;
    }

    return {
      kind: pattern.kind,
      limitTokens,
      requestedTokens,
      promptTokens: resolvePromptTokens(haystack, pattern, requestedTokens),
      source: 'pattern',
      provider,
    };
  }

  if (langChainFlagged) {
    return { kind: 'context_window', source: 'langchain', provider };
  }

  return null;
}

/**
 * Returns true if the error definitively indicates a context overflow.
 *
 * Accepts either a raw error or a pre-extracted message; passing the error
 * itself is preferred, since several providers report the decisive detail in
 * a nested body rather than in `message`.
 */
export function isContextOverflowError(
  error?: unknown,
  context?: ContextOverflowContext
): boolean {
  return getContextOverflowInfo(error, context) != null;
}

/**
 * Returns true if the error likely indicates a context overflow, adding
 * body-size and token-count heuristics on top of the definitive patterns.
 *
 * May produce false positives on unusual messages. Use when the cost of
 * being wrong is one extra compaction pass.
 */
export function isLikelyContextOverflowError(
  error?: unknown,
  context?: ContextOverflowContext
): boolean {
  if (isContextOverflowError(error, context)) {
    return true;
  }
  const haystack = stripUrls(collectErrorText(error));
  if (
    haystack === '' ||
    AMBIGUOUS_CONTEXT_OR_OUTPUT_RE.test(haystack) ||
    OUTPUT_LIMIT_RE.test(haystack)
  ) {
    return false;
  }
  if (NON_RECOVERABLE_RE.test(haystack)) {
    return false;
  }
  return CONTEXT_OVERFLOW_HINT_RE.test(haystack);
}

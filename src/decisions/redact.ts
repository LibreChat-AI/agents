const REDACTED = '[REDACTED]';
const BEARER = /\b(bearer\s+)([^\s"',;]+)/gi;
const JWT = /\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}/g;
/** Long opaque runs (API keys, session tokens); also catches 40-hex digests, which is acceptable in a log line. */
const OPAQUE = /[A-Za-z0-9_-]{40,}/g;

type Span = [start: number, end: number];

function literalSpans(text: string, secret: string, spans: Span[]): void {
  for (
    let at = text.indexOf(secret);
    at !== -1;
    at = text.indexOf(secret, at + 1)
  ) {
    spans.push([at, at + secret.length]);
  }
}

function patternSpans(
  text: string,
  pattern: RegExp,
  spans: Span[],
  keepGroup = 0
): void {
  for (const match of text.matchAll(pattern)) {
    const kept = keepGroup > 0 ? match[keepGroup].length : 0;
    spans.push([match.index + kept, match.index + match[0].length]);
  }
}

/**
 * Removes credentials from text bound for a log line or a status row: every
 * listed secret verbatim, plus anything shaped like a bearer value, a JWT, or
 * a long opaque token. Every recognizer reads the original text and the
 * union of their matches is replaced at once, so a listed secret that
 * overlaps a `Bearer` keyword or splits a long token cannot hide a credential
 * from the shape checks. Unlisted short secrets are not recognizable by
 * shape, so pass every credential the text could contain.
 */
export function redact(
  text: string,
  secrets: ReadonlyArray<string | null | undefined> = []
): string {
  const source = String(text);
  const spans: Span[] = [];
  for (const secret of secrets) {
    const listed = typeof secret === 'string' ? secret.trim() : '';
    if (listed !== '') {
      literalSpans(source, listed, spans);
    }
  }
  patternSpans(source, BEARER, spans, 1);
  patternSpans(source, JWT, spans);
  patternSpans(source, OPAQUE, spans);
  if (spans.length === 0) {
    return source;
  }

  spans.sort((a, b) => a[0] - b[0]);
  let result = '';
  let cursor = 0;
  let [start, end] = spans[0];
  for (const [nextStart, nextEnd] of spans) {
    if (nextStart <= end) {
      end = Math.max(end, nextEnd);
      continue;
    }
    result += source.slice(cursor, start) + REDACTED;
    cursor = end;
    [start, end] = [nextStart, nextEnd];
  }
  return result + source.slice(cursor, start) + REDACTED + source.slice(end);
}

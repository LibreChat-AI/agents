const REDACTED = '[REDACTED]';
const BEARER = /\b(bearer\s+)([^\s"',;]+)/gi;
/** Where a JWT can start; its three segments are measured by `jwtSpans`. */
const JWT_START = /\beyJ/g;
/** A character of one JWT segment. */
const SEGMENT_CHAR = /[\w-]/;
/** The shortest segment a JWT is recognized with. */
const SEGMENT_MIN = 8;
/** Long opaque runs (API keys, session tokens); also catches 40-hex digests, which is acceptable in a log line. */
const OPAQUE = /[A-Za-z0-9_-]{40,}/g;
/** A character of a token-shaped run (opaque, JWT, dotted) a cut at the end of an excerpt window may split. */
const TOKEN_CHAR = /[\w.-]/;
/** How much of the text an excerpt reads, so its cost does not grow with the text. */
const EXCERPT_WINDOW = 4096;
/** How much raw text an excerpt collapses to fill its window; whitespace past it is never scanned. */
const RAW_WINDOW = EXCERPT_WINDOW * 4;

/**
 * Coverage of `source` as a difference array: `depth[i]` adds one where a
 * span starts and subtracts one where it ends, so memory stays one slot per
 * character however many matches there are.
 */
type Coverage = { depth: Int32Array; marked: boolean };

function mark(coverage: Coverage, start: number, end: number): void {
  coverage.depth[start] += 1;
  coverage.depth[end] -= 1;
  coverage.marked = true;
}

/** Every occurrence of `secret`, overlapping ones included, merged into runs as they are found. */
function literalSpans(
  source: string,
  secret: string,
  coverage: Coverage
): void {
  let start = -1;
  let end = -1;
  for (
    let at = source.indexOf(secret);
    at !== -1;
    at = source.indexOf(secret, at + 1)
  ) {
    if (at > end) {
      if (start !== -1) {
        mark(coverage, start, end);
      }
      start = at;
    }
    end = at + secret.length;
  }
  if (start !== -1) {
    mark(coverage, start, end);
  }
}

function patternSpans(
  source: string,
  pattern: RegExp,
  coverage: Coverage,
  keepGroup = 0
): void {
  for (const match of source.matchAll(pattern)) {
    const kept = keepGroup > 0 ? match[keepGroup].length : 0;
    mark(coverage, match.index + kept, match.index + match[0].length);
  }
}

/** Where the run of segment characters starting at `from` ends. */
function segmentEnd(source: string, from: number): number {
  let at = from;
  while (at < source.length && SEGMENT_CHAR.test(source[at])) {
    at += 1;
  }
  return at;
}

/**
 * Every JWT shape (`eyJ` and three dot-separated segments of at least
 * `SEGMENT_MIN` characters each) in one pass. A segment cannot contain the
 * dot that ends it, so each segment's end is fixed by where its run of
 * segment characters stops, and a start that fails fails for every later
 * start inside its first segment too: the scan resumes past that segment
 * instead of rescanning the run from each word boundary, which a
 * backtracking pattern does in quadratic time.
 */
function jwtSpans(source: string, coverage: Coverage): void {
  JWT_START.lastIndex = 0;
  for (
    let match = JWT_START.exec(source);
    match !== null;
    match = JWT_START.exec(source)
  ) {
    const start = match.index;
    const firstEnd = segmentEnd(source, start + 3);
    let from = start + 3;
    let stop = firstEnd;
    let segments = 0;
    while (
      stop - from >= SEGMENT_MIN &&
      ++segments < 3 &&
      source[stop] === '.'
    ) {
      from = stop + 1;
      stop = segmentEnd(source, from);
    }
    if (segments === 3) {
      mark(coverage, start, stop);
    }
    JWT_START.lastIndex = Math.max(segments === 3 ? stop : firstEnd, start + 1);
  }
}

/**
 * The distinct listed secrets, each normalized by `normalize`; iterates
 * `secrets` with `for...of`, so an unreadable list throws instead of
 * silently redacting nothing.
 */
function listed(
  secrets: ReadonlyArray<string | null | undefined>,
  normalize: (secret: string) => string
): Set<string> {
  const distinct = new Set<string>();
  for (const secret of secrets) {
    const value = typeof secret === 'string' ? normalize(secret) : '';
    if (value !== '') {
      distinct.add(value);
    }
  }
  return distinct;
}

/** Marks every recognizer's matches, then replaces each covered run with one marker. */
function cover(source: string, secrets: Set<string>, cutAt?: number): string {
  const coverage: Coverage = {
    depth: new Int32Array(source.length + 1),
    marked: false,
  };
  for (const secret of secrets) {
    literalSpans(source, secret, coverage);
  }
  patternSpans(source, BEARER, coverage, 1);
  jwtSpans(source, coverage);
  patternSpans(source, OPAQUE, coverage);
  if (cutAt !== undefined && cutAt < source.length) {
    mark(coverage, cutAt, source.length);
  }
  if (!coverage.marked) {
    return source;
  }

  let result = '';
  let cursor = 0;
  let depth = 0;
  for (let at = 0; at <= source.length; at++) {
    const covered = depth > 0;
    depth += coverage.depth[at];
    if (!covered && depth > 0) {
      result += source.slice(cursor, at) + REDACTED;
    } else if (covered && depth === 0) {
      cursor = at;
    }
  }
  return result + source.slice(cursor);
}

/**
 * Removes credentials from text bound for a log line or a status row: every
 * listed secret verbatim, plus anything shaped like a bearer value, a JWT, or
 * a long opaque token. Every recognizer reads the original text and the
 * union of their matches is replaced at once, so a listed secret that
 * overlaps a `Bearer` keyword or splits a long token cannot hide a credential
 * from the shape checks. Unlisted short secrets are not recognizable by
 * shape, so pass every credential the text could contain. Memory stays
 * proportional to the text however often a secret repeats in it; time grows
 * with the text times the number of distinct secrets.
 */
export function redact(
  text: string,
  secrets: ReadonlyArray<string | null | undefined> = []
): string {
  return cover(
    String(text),
    listed(secrets, (secret) => secret.trim())
  );
}

const collapse = (text: string): string => text.replace(/\s+/g, ' ').trim();

/**
 * The first `limit` characters of `text` redacted, with whitespace
 * collapsed. Only the first 4096 characters after collapsing, drawn from at
 * most the first 16384 of the text, are read, so the cost stays bounded
 * however long the text and however many secrets there are. When the text runs past that window, its tail is redacted as far
 * back as the longest listed secret could reach and through any token-shaped
 * run the cut splits, so a credential the window cuts off cannot show its
 * first characters.
 */
export function redactExcerpt(
  text: string,
  secrets: ReadonlyArray<string | null | undefined>,
  limit: number
): string {
  const raw = String(text);
  const head = raw.slice(0, RAW_WINDOW);
  const whole = collapse(head);
  const source = whole.slice(0, EXCERPT_WINDOW);
  const distinct = listed(secrets, collapse);
  let cutAt: number | undefined;
  if (raw.length > head.length || whole.length > source.length) {
    let longest = 0;
    for (const secret of distinct) {
      longest = Math.max(longest, secret.length);
    }
    let run = source.length;
    while (run > 0 && TOKEN_CHAR.test(source[run - 1])) {
      run -= 1;
    }
    cutAt = Math.max(0, Math.min(source.length - longest + 1, run));
  }
  return cover(source, distinct, cutAt).slice(0, limit);
}

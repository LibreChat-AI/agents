import { tmpdir } from 'os';
import { resolve } from 'path';
import { writeFileSync } from 'fs';
import { randomUUID } from 'crypto';

/**
 * Writes text to a fresh file under the host temp dir and returns its path.
 * The file is readable by its owner only, since tool output may hold secrets.
 */
export function spillToFile(text: string): string {
  const path = resolve(tmpdir(), `lc-output-${randomUUID()}.txt`);
  writeFileSync(path, text, { encoding: 'utf8', mode: 0o600 });
  return path;
}

/** Keeps the last `maxChars` characters, starting at a line boundary when one is near. */
export function tailTruncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  const tail = text.slice(text.length - maxChars);
  const newline = tail.indexOf('\n');
  return newline >= 0 && newline < 200 ? tail.slice(newline + 1) : tail;
}

/** Longest grep match line kept when results are capped. */
export const CAPPED_GREP_LINE_CHARS = 300;

/**
 * Caps line-oriented results (search matches): each line is cut to
 * `CAPPED_GREP_LINE_CHARS`, and lines stop at `maxChars` with a note naming
 * how many were left out.
 */
export function capLines(text: string, maxChars: number): string {
  const lines = text.split('\n');
  const kept: string[] = [];
  let used = 0;
  for (const raw of lines) {
    const line =
      raw.length > CAPPED_GREP_LINE_CHARS
        ? `${raw.slice(0, CAPPED_GREP_LINE_CHARS)} [line truncated]`
        : raw;
    if (used + line.length + 1 > maxChars && kept.length > 0) break;
    kept.push(line);
    used += line.length + 1;
  }
  const omitted = lines.length - kept.length;
  return omitted > 0
    ? `${kept.join('\n')}\n[${omitted} more matching lines not shown; narrow the pattern or path]`
    : kept.join('\n');
}

/**
 * Caps free-form output (shell results): keeps the first 20% and the last 80%
 * of `maxChars`, where errors and test summaries are, and names a file that
 * holds the full text.
 */
export function capHeadTail(
  text: string,
  maxChars: number,
  spill: (text: string) => string = spillToFile
): string {
  if (text.length <= maxChars) {
    return text;
  }
  const head = text.slice(0, Math.floor(maxChars * 0.2));
  const tail = tailTruncate(text, Math.floor(maxChars * 0.8));
  const omitted = text.length - head.length - tail.length;
  return `${head}\n[... ${omitted} characters omitted; full output at ${spill(text)} ...]\n${tail}`;
}

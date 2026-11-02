import { createHash } from 'crypto';
import type { ReadFileRecovery } from '@/types/tools';
import { sliceWithoutSplittingSurrogates } from '@/utils/truncation';

export interface ReadFilePageInput {
  cursor?: string;
  offset?: number;
  limit?: number;
  line_numbers?: boolean;
}

interface ReadFilePage {
  text: string;
  recovery: ReadFileRecovery;
}

function cursorPosition(
  cursor: string,
  sha256: string,
  content: string
): number {
  const match = /^v1:([a-f0-9]{64}):(0|[1-9]\d{0,15})$/.exec(cursor);
  if (match == null) throw new Error('Invalid read_file cursor.');
  if (match[1] !== sha256) {
    throw new Error('File content changed; start a new read without cursor.');
  }
  const position = Number(match[2]);
  if (!Number.isSafeInteger(position) || position > content.length) {
    throw new Error('read_file cursor is outside the file.');
  }
  const code = content.charCodeAt(position);
  if (code >= 0xdc00 && code <= 0xdfff) {
    throw new Error('read_file cursor splits a Unicode character.');
  }
  return position;
}

function renderPage(
  content: string,
  start: number,
  line: number,
  maxLines: number,
  maxChars: number,
  numbered: boolean
): { text: string; end: number } {
  const parts: string[] = [];
  let position = start;
  let used = 0;
  while (position < content.length && parts.length < maxLines) {
    const newline = content.indexOf('\n', position);
    const end = newline === -1 ? content.length : newline + 1;
    const prefix = numbered
      ? `${String(line + parts.length).padStart(6, ' ')}\t`
      : '';
    const available = maxChars - used - prefix.length;
    if (parts.length > 0 && end - position > available) break;
    const text = sliceWithoutSplittingSurrogates(
      content,
      position,
      Math.min(end, position + Math.max(0, available))
    );
    if (text.length === 0) {
      if (parts.length > 0) break;
      throw new Error(
        'read_file result limit is too small for one source character.'
      );
    }
    parts.push(prefix + text);
    position += text.length;
    used += prefix.length + text.length;
    if (position < end) break;
  }
  return { text: parts.join(''), end: position };
}

/** Cursors bind to captured bytes, not to a workspace revision or a stored snapshot. */
export function readFilePage(
  bytes: Buffer,
  input: ReadFilePageInput,
  maxBodyChars: number,
  maxResultChars = Infinity
): ReadFilePage {
  for (const value of [input.offset, input.limit]) {
    if (value != null && (!Number.isSafeInteger(value) || value < 1)) {
      throw new Error(
        'read_file offset and limit must be positive safe integers.'
      );
    }
  }
  if (input.cursor != null && input.offset != null) {
    throw new Error('Use either cursor or offset, not both.');
  }
  const content = new TextDecoder('utf-8', {
    fatal: true,
    ignoreBOM: true,
  }).decode(bytes);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  let start =
    input.cursor == null ? 0 : cursorPosition(input.cursor, sha256, content);
  let line = 1;
  let position = 0;
  while (input.cursor == null ? line < (input.offset ?? 1) : position < start) {
    const newline = content.indexOf('\n', position);
    if (newline === -1) {
      if (input.cursor == null) start = content.length;
      break;
    }
    if (input.cursor != null && newline >= start) break;
    position = newline + 1;
    line++;
    if (input.cursor == null) start = position;
  }
  const maxLines = input.limit ?? Infinity;
  const numbered = input.line_numbers !== false;
  const budget = Math.min(maxBodyChars, maxResultChars);
  let page = renderPage(content, start, line, maxLines, budget, numbered);
  const footer = (end: number): string =>
    `\n[continue with cursor=v1:${sha256}:${end}]`;
  if (
    page.end < content.length &&
    page.text.length + footer(page.end).length > maxResultChars
  ) {
    const available = Math.min(
      maxBodyChars,
      maxResultChars - footer(content.length).length
    );
    if (available < 1) {
      throw new Error(
        'read_file result limit is too small for a continuation cursor.'
      );
    }
    page = renderPage(content, start, line, maxLines, available, numbered);
  }
  const nextCursor =
    page.end < content.length ? `v1:${sha256}:${page.end}` : undefined;
  return {
    text: page.text + (nextCursor == null ? '' : footer(page.end)),
    recovery: {
      sha256,
      start,
      end: page.end,
      unit: 'utf16',
      bodyChars: page.text.length,
      nextCursor,
    },
  };
}

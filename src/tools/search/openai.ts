import type {
  NativeSearchResults,
  CitationSpan,
  UrlCitation,
} from './citations';
import type { ProcessedOrganic } from './types';
import { getAttribution, getDomainName } from './utils';
import { codePointToCharOffset } from './citations';

/** A markdown link, optionally wrapped in the parentheses OpenAI puts around citations. */
const MARKDOWN_LINK = /^\(?\[[^\]\n]+\]\((https?:\/\/\S+?)\)\)?$/;

function linksTo(span: string, url: string): boolean {
  const target = MARKDOWN_LINK.exec(span)?.[1];
  if (target == null) {
    return false;
  }
  return target === url || getDomainName(target) === getDomainName(url);
}

/**
 * Finds the inline markdown link a `url_citation` covers. Its indices count
 * characters of one output text block; they are read as UTF-16 first, then as
 * code points, and the end is tried as exclusive and inclusive. A span that is
 * not a link to the cited host is left alone.
 */
function locateLink(
  text: string,
  blockStart: number,
  citation: UrlCitation
): { start: number; end: number } | undefined {
  if (citation.start == null || citation.end == null) {
    return undefined;
  }
  const candidates = [
    {
      start: blockStart + citation.start,
      end: blockStart + citation.end,
    },
    {
      start: codePointToCharOffset(text, blockStart, citation.start),
      end: codePointToCharOffset(text, blockStart, citation.end),
    },
  ];
  for (const { start, end } of candidates) {
    if (start == null || end == null) {
      continue;
    }
    for (const close of [end, end + 1]) {
      if (!linksTo(text.slice(start, close), citation.url)) {
        continue;
      }
      const wrapped = text[start - 1] === '(' && text[close] === ')';
      return wrapped
        ? { start: start - 1, end: close + 1 }
        : { start, end: close };
    }
  }
  return undefined;
}

/** Turns OpenAI `url_citation` annotations into organic results and the links they replace. */
export function coerceOpenAICitationResults(
  citations: readonly UrlCitation[],
  text: string,
  blocks: ReadonlyMap<number, number>
): NativeSearchResults | undefined {
  const organic: ProcessedOrganic[] = [];
  const sourceByUrl = new Map<string, number>();
  const spans: CitationSpan[] = [];
  for (const citation of citations) {
    let source = sourceByUrl.get(citation.url);
    if (source == null) {
      source = organic.length;
      sourceByUrl.set(citation.url, source);
      organic.push({
        link: citation.url,
        position: source + 1,
        title: citation.title,
        attribution: getAttribution(citation.url),
      });
    }
    const blockStart =
      citation.block == null ? undefined : blocks.get(citation.block);
    const span =
      blockStart == null ? undefined : locateLink(text, blockStart, citation);
    if (span != null) {
      spans.push({ ...span, sources: [source] });
    }
  }
  return organic.length > 0 ? { organic, spans } : undefined;
}

import type {
  NativeSearchResults,
  CitationSpan,
  UrlCitation,
} from './citations';
import type { ProcessedOrganic } from './types';
import { getAttribution, getDomainName } from './utils';

/** A markdown link, optionally wrapped in the parentheses OpenAI puts around citations. */
const MARKDOWN_LINK = /^\(?\[[^\]\n]+\]\((https?:\/\/\S+?)\)\)?$/;

function linksTo(span: string, url: string): boolean {
  const target = MARKDOWN_LINK.exec(span)?.[1];
  if (target == null) {
    return false;
  }
  return target === url || getDomainName(target) === getDomainName(url);
}

function matchLink(
  text: string,
  start: number,
  end: number,
  url: string
): { start: number; end: number } | undefined {
  for (const close of [end, end + 1]) {
    if (!linksTo(text.slice(start, close), url)) {
      continue;
    }
    const wrapped = text[start - 1] === '(' && text[close] === ')';
    return wrapped
      ? { start: start - 1, end: close + 1 }
      : { start, end: close };
  }
  return undefined;
}

/** UTF-16 index of every code point from `from` to the end of `text`, then the end itself. */
function getCodePointOffsets(text: string, from: number): number[] {
  const offsets: number[] = [];
  let index = from;
  while (index < text.length) {
    offsets.push(index);
    index += (text.codePointAt(index) ?? 0) > 0xffff ? 2 : 1;
  }
  offsets.push(index);
  return offsets;
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
  citation: UrlCitation,
  getOffsets: () => number[]
): { start: number; end: number } | undefined {
  const { start, end, url } = citation;
  if (start == null || end == null) {
    return undefined;
  }
  const direct = matchLink(text, blockStart + start, blockStart + end, url);
  if (direct != null) {
    return direct;
  }
  const offsets = getOffsets();
  const from = offsets.at(start);
  const to = offsets.at(end);
  return from == null || to == null
    ? undefined
    : matchLink(text, from, to, url);
}

/** Turns OpenAI `url_citation` annotations into organic results and the links they replace. */
export function coerceOpenAICitationResults(
  citations: readonly UrlCitation[],
  text: string,
  blocks: ReadonlyMap<number, number>
): NativeSearchResults | undefined {
  const organic: ProcessedOrganic[] = [];
  const sourceByUrl = new Map<string, number>();
  const offsetsByBlock = new Map<number, number[]>();
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
      blockStart == null
        ? undefined
        : locateLink(text, blockStart, citation, () => {
          let offsets = offsetsByBlock.get(blockStart);
          if (offsets == null) {
            offsets = getCodePointOffsets(text, blockStart);
            offsetsByBlock.set(blockStart, offsets);
          }
          return offsets;
        });
    if (span != null) {
      spans.push({ ...span, sources: [source] });
    }
  }
  return organic.length > 0 ? { organic, spans } : undefined;
}

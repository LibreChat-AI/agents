import * as cheerio from 'cheerio';
import type { References, MediaReference } from './types';

type ReferenceType = 'image' | 'video' | 'iframe' | 'link';

interface ReferenceTarget {
  type: ReferenceType;
  idx: number;
}

interface ReferenceEntry extends ReferenceTarget {
  url: string;
}

/**
 * Aho-Corasick automaton over `(` followed by every referenced URL, held in
 * typed arrays: 18 bytes per node and no object per node. Nodes are numbered
 * breadth first, so the children of `node` are the nodes from
 * `firstChild[node]` up to `firstChild[node + 1]`, sorted by `code`.
 */
export interface ReferenceMatcher {
  firstChild: Int32Array;
  code: Uint16Array;
  depth: Int32Array;
  fail: Int32Array;
  /** Nearest node on the fail chain, itself included, that ends a URL, or -1. */
  output: Int32Array;
  targets: Map<number, ReferenceTarget>;
}

interface ReferenceClose {
  end: number;
  title?: string;
}

/**
 * Largest automaton built for one page. Nodes are the distinct URL prefixes, at
 * most the URL characters on the page: 2^21 of them measured 36 MiB and 0.15 s
 * to build, about 100 times a long Wikipedia article. A page above the cap
 * keeps its markdown as is.
 */
export const MAX_MATCHER_NODES = 1 << 21;
const OPEN_PAREN = 0x28;
const CLOSE_PAREN = 0x29;
const QUOTE = 0x22;
const WHITESPACE = /\s/;

function compareUrls(a: ReferenceEntry, b: ReferenceEntry): number {
  if (a.url === b.url) {
    return 0;
  }
  return a.url < b.url ? -1 : 1;
}

function sharedPrefixLength(a: string, b: string): number {
  const limit = Math.min(a.length, b.length);
  let length = 0;
  while (length < limit && a.charCodeAt(length) === b.charCodeAt(length)) {
    length++;
  }
  return length;
}

/**
 * Indexes every referenced URL once, or returns `undefined` when the automaton
 * would exceed `MAX_MATCHER_NODES`. A URL listed under several types keeps the
 * first, in image, video, iframe, link order, which is the order the replacement
 * has always preferred. Sorting the URLs gives the exact node count before
 * anything is allocated, and lets each node take its children from one
 * contiguous run of URLs.
 */
export function buildMatcher(
  groups: ReadonlyArray<[ReferenceType, Map<string, MediaReference>]>
): ReferenceMatcher | undefined {
  const entries: ReferenceEntry[] = [];
  for (const [type, references] of groups) {
    let idx = 0;
    for (const url of references.keys()) {
      entries.push({ url, type, idx: ++idx });
    }
  }
  entries.sort(compareUrls);
  let size = 2;
  for (let i = 0; i < entries.length; i++) {
    const { url } = entries[i];
    size +=
      url.length - (i === 0 ? 0 : sharedPrefixLength(entries[i - 1].url, url));
  }
  if (size > MAX_MATCHER_NODES) {
    return undefined;
  }

  const matcher: ReferenceMatcher = {
    firstChild: new Int32Array(size + 1),
    code: new Uint16Array(size),
    depth: new Int32Array(size),
    fail: new Int32Array(size),
    output: new Int32Array(size).fill(-1),
    targets: new Map<number, ReferenceTarget>(),
  };
  const { firstChild, code, depth, fail, output, targets } = matcher;
  /** Nodes awaiting their children span two trie levels, each at most one node per URL. */
  const pending = 2 * entries.length + 2;
  const from = new Int32Array(pending);
  const to = new Int32Array(pending);
  firstChild[0] = 1;
  code[1] = OPEN_PAREN;
  depth[1] = 1;
  to[1] = entries.length;
  let next = 2;
  for (let node = 1; node < next; node++) {
    const consumed = depth[node] - 1;
    const end = to[node % pending];
    let i = from[node % pending];
    const endsUrl = i < end && entries[i].url.length === consumed;
    if (endsUrl) {
      targets.set(node, entries[i]);
    }
    while (i < end && entries[i].url.length === consumed) {
      i++;
    }
    output[node] = endsUrl ? node : output[fail[node]];
    firstChild[node] = next;
    while (i < end) {
      const charCode = entries[i].url.charCodeAt(consumed);
      code[next] = charCode;
      depth[next] = consumed + 2;
      fail[next] = transition(matcher, fail[node], charCode);
      from[next % pending] = i;
      do {
        i++;
      } while (i < end && entries[i].url.charCodeAt(consumed) === charCode);
      to[next % pending] = i;
      next++;
    }
  }
  firstChild[next] = next;
  return matcher;
}

function childOf(
  matcher: ReferenceMatcher,
  node: number,
  charCode: number
): number {
  const { firstChild, code } = matcher;
  const end = firstChild[node + 1];
  let low = firstChild[node];
  let high = end;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (code[middle] < charCode) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < end && code[low] === charCode ? low : -1;
}

function transition(
  matcher: ReferenceMatcher,
  from: number,
  charCode: number
): number {
  let state = from;
  let next = childOf(matcher, state, charCode);
  while (next === -1 && state !== 0) {
    state = matcher.fail[state];
    next = childOf(matcher, state, charCode);
  }
  return next === -1 ? 0 : next;
}

/**
 * Closes a reference whose URL ends just before `urlEnd` with `)` or
 * ` "title")`. Whitespace runs and quote pairs are resolved once each and
 * cached, so checking many URL ends never rescans the same text.
 */
function createCloser(
  markdown: string
): (urlEnd: number) => ReferenceClose | undefined {
  const closingQuote = new Map<number, number>();
  let nonSpaceFrom: Int32Array | undefined;
  const firstNonSpace = (index: number): number => {
    if (nonSpaceFrom == null) {
      nonSpaceFrom = new Int32Array(markdown.length + 1);
      nonSpaceFrom[markdown.length] = markdown.length;
      for (let i = markdown.length - 1; i >= 0; i--) {
        nonSpaceFrom[i] = WHITESPACE.test(markdown[i])
          ? nonSpaceFrom[i + 1]
          : i;
      }
    }
    return nonSpaceFrom[index];
  };
  return (urlEnd) => {
    const next = markdown.charCodeAt(urlEnd);
    if (next === CLOSE_PAREN) {
      return { end: urlEnd + 1 };
    }
    if (!WHITESPACE.test(markdown[urlEnd] ?? '')) {
      return undefined;
    }
    const open = firstNonSpace(urlEnd);
    if (markdown.charCodeAt(open) !== QUOTE) {
      return undefined;
    }
    let close = closingQuote.get(open);
    if (close == null) {
      close = markdown.indexOf('"', open + 1);
      closingQuote.set(open, close);
    }
    if (close === -1 || markdown.charCodeAt(close + 1) !== CLOSE_PAREN) {
      return undefined;
    }
    return { end: close + 2, title: markdown.slice(open + 1, close) };
  };
}

/**
 * Rewrites every markdown reference to a known URL. One scan finds each
 * `(url` occurrence that is closed as a reference; the longest at each opening
 * parenthesis is applied left to right. The work is linear in the markdown
 * plus the occurrences found, where one regex per URL over the whole page grew
 * with links times page size and blocked the event loop for seconds.
 * Occurrences exceed the markdown length only when URLs are suffixes of one
 * another; n such URLs need about n * n characters of HTML, so the worst case
 * stays near the input size to the power 1.5.
 */
function replaceReferences(
  markdown: string,
  matcher: ReferenceMatcher
): string {
  if (matcher.targets.size === 0) {
    return markdown;
  }
  const { depth, fail, output, targets } = matcher;
  const closeAt = createCloser(markdown);
  const longest = new Map<number, { node: number; close: ReferenceClose }>();
  let state = 0;
  for (let i = 0; i < markdown.length; i++) {
    state = transition(matcher, state, markdown.charCodeAt(i));
    let node = output[state];
    if (node === -1) {
      continue;
    }
    const close = closeAt(i + 1);
    if (close == null) {
      continue;
    }
    for (; node !== -1; node = output[fail[node]]) {
      const start = i - depth[node] + 1;
      const current = longest.get(start);
      if (current == null || depth[current.node] < depth[node]) {
        longest.set(start, { node, close });
      }
    }
  }

  let result = '';
  let copied = 0;
  const ordered = Array.from(longest.entries()).sort((a, b) => a[0] - b[0]);
  for (const [start, { node, close }] of ordered) {
    const target = targets.get(node);
    if (start < copied || target == null) {
      continue;
    }
    const label = `${target.type}#${target.idx}`;
    const text =
      close.title == null ? `(${label})` : `(${label} "${close.title}")`;
    result += markdown.slice(copied, start) + text;
    copied = close.end;
  }
  return result + markdown.slice(copied);
}

export function processContent(
  html: string,
  markdown: string
): {
  markdown: string;
} & References {
  const linkMap = new Map<string, MediaReference>();
  const imageMap = new Map<string, MediaReference>();
  const videoMap = new Map<string, MediaReference>();
  const iframeMap = new Map<string, MediaReference>();

  const $ = cheerio.load(html, {
    xmlMode: false,
  });

  // Extract all media references
  $('a[href]').each((_, el) => {
    const href = $(el).attr('href');
    if (href != null && href) {
      linkMap.set(href, {
        originalUrl: href,
        title: $(el).attr('title'),
        text: $(el).text().trim(),
      });
    }
  });

  $('img[src]').each((_, el) => {
    const src = $(el).attr('src');
    if (src != null && src) {
      imageMap.set(src, {
        originalUrl: src,
        title: $(el).attr('alt') ?? $(el).attr('title'),
      });
    }
  });

  // Handle videos (dedicated video elements and video platforms in iframes)
  $('video[src], iframe[src*="youtube"], iframe[src*="vimeo"]').each(
    (_, el) => {
      const src = $(el).attr('src');
      if (src != null && src) {
        videoMap.set(src, {
          originalUrl: src,
          title: $(el).attr('title'),
        });
      }
    }
  );

  // Handle all other generic iframes that aren't already captured as videos
  $('iframe').each((_, el) => {
    const src = $(el).attr('src');
    if (
      src != null &&
      src &&
      !src.includes('youtube') &&
      !src.includes('vimeo')
    ) {
      iframeMap.set(src, {
        originalUrl: src,
        title: $(el).attr('title'),
      });
    }
  });

  const matcher = buildMatcher([
    ['image', imageMap],
    ['video', videoMap],
    ['iframe', iframeMap],
    ['link', linkMap],
  ]);
  const result =
    matcher == null ? markdown : replaceReferences(markdown, matcher);

  iframeMap.clear();
  const links = Array.from(linkMap.values());
  linkMap.clear();
  const images = Array.from(imageMap.values());
  imageMap.clear();
  const videos = Array.from(videoMap.values());
  videoMap.clear();

  return {
    markdown: result,
    links,
    images,
    videos,
  };
}

import * as cheerio from 'cheerio';
import type { References, MediaReference } from './types';

type ReferenceType = 'image' | 'video' | 'iframe' | 'link';

interface ReferenceTarget {
  type: ReferenceType;
  idx: number;
}

/**
 * Aho-Corasick automaton over `(` followed by every referenced URL. Edges live in
 * one map keyed by `node * EDGE_STRIDE + charCode`, so memory stays proportional
 * to the URL text, and a single scan of the markdown finds every occurrence.
 */
interface ReferenceMatcher {
  edges: Map<number, number>;
  depth: number[];
  fail: number[];
  /** Nearest node on the fail chain, itself included, that ends a URL, or -1. */
  output: number[];
  targets: Map<number, ReferenceTarget>;
}

interface ReferenceClose {
  end: number;
  title?: string;
}

const EDGE_STRIDE = 0x10000;
const OPEN_PAREN = 0x28;
const CLOSE_PAREN = 0x29;
const QUOTE = 0x22;
const WHITESPACE = /\s/;

/**
 * Indexes every referenced URL once. A URL listed under several types keeps the
 * first, in image, video, iframe, link order, which is the order the replacement
 * has always preferred.
 */
function buildMatcher(
  groups: ReadonlyArray<[ReferenceType, Map<string, MediaReference>]>
): ReferenceMatcher {
  const edges = new Map<number, number>();
  const depth = [0];
  const charOf = [0];
  const children: number[][] = [[]];
  const targets = new Map<number, ReferenceTarget>();
  const advance = (node: number, code: number): number => {
    const key = node * EDGE_STRIDE + code;
    const existing = edges.get(key);
    if (existing != null) {
      return existing;
    }
    const next = depth.length;
    edges.set(key, next);
    depth.push(depth[node] + 1);
    charOf.push(code);
    children.push([]);
    children[node].push(next);
    return next;
  };
  for (const [type, references] of groups) {
    let idx = 0;
    for (const url of references.keys()) {
      idx++;
      let node = advance(0, OPEN_PAREN);
      for (let i = 0; i < url.length; i++) {
        node = advance(node, url.charCodeAt(i));
      }
      if (!targets.has(node)) {
        targets.set(node, { type, idx });
      }
    }
  }

  const fail = new Array<number>(depth.length).fill(0);
  const output = new Array<number>(depth.length).fill(-1);
  const queue = [...children[0]];
  for (let head = 0; head < queue.length; head++) {
    const node = queue[head];
    output[node] = targets.has(node) ? node : output[fail[node]];
    for (const child of children[node]) {
      fail[child] = transition(edges, fail, fail[node], charOf[child]);
      queue.push(child);
    }
  }
  return { edges, depth, fail, output, targets };
}

function transition(
  edges: Map<number, number>,
  fail: number[],
  from: number,
  code: number
): number {
  let state = from;
  while (state !== 0 && !edges.has(state * EDGE_STRIDE + code)) {
    state = fail[state];
  }
  return edges.get(state * EDGE_STRIDE + code) ?? 0;
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
 */
function replaceReferences(
  markdown: string,
  matcher: ReferenceMatcher
): string {
  if (matcher.targets.size === 0) {
    return markdown;
  }
  const { edges, depth, fail, output, targets } = matcher;
  const closeAt = createCloser(markdown);
  const longest = new Map<number, { node: number; close: ReferenceClose }>();
  let state = 0;
  for (let i = 0; i < markdown.length; i++) {
    state = transition(edges, fail, state, markdown.charCodeAt(i));
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

  const result = replaceReferences(
    markdown,
    buildMatcher([
      ['image', imageMap],
      ['video', videoMap],
      ['iframe', iframeMap],
      ['link', linkMap],
    ])
  );

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

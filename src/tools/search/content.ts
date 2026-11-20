import * as cheerio from 'cheerio';
import type { References, MediaReference } from './types';

type ReferenceType = 'image' | 'video' | 'iframe' | 'link';

interface ReferenceTarget {
  type: ReferenceType;
  idx: number;
}

/**
 * Character trie over every referenced URL. Edges live in one map keyed by
 * `node * EDGE_STRIDE + charCode`, so memory stays proportional to the URL text.
 */
interface ReferenceTrie {
  edges: Map<number, number>;
  terminals: Map<number, ReferenceTarget>;
}

const EDGE_STRIDE = 0x10000;
const TITLE_SUFFIX = /\s+"([^"]*)"\)/y;

/**
 * Indexes every referenced URL once. A URL listed under several types keeps the
 * first, in image, video, iframe, link order, which is the order the replacement
 * has always preferred.
 */
function buildTrie(
  groups: ReadonlyArray<[ReferenceType, Map<string, MediaReference>]>
): ReferenceTrie {
  const edges = new Map<number, number>();
  const terminals = new Map<number, ReferenceTarget>();
  let nodes = 1;
  for (const [type, references] of groups) {
    let idx = 0;
    for (const url of references.keys()) {
      idx++;
      let node = 0;
      for (let i = 0; i < url.length; i++) {
        const key = node * EDGE_STRIDE + url.charCodeAt(i);
        let next = edges.get(key);
        if (next == null) {
          next = nodes++;
          edges.set(key, next);
        }
        node = next;
      }
      if (!terminals.has(node)) {
        terminals.set(node, { type, idx });
      }
    }
  }
  return { edges, terminals };
}

/** Where `(url)` or `(url "title")` ends when `url` ends just before `urlEnd`. */
function referenceEnd(
  markdown: string,
  urlEnd: number
): { end: number; title?: string } | undefined {
  if (markdown[urlEnd] === ')') {
    return { end: urlEnd + 1 };
  }
  TITLE_SUFFIX.lastIndex = urlEnd;
  const title = TITLE_SUFFIX.exec(markdown);
  return title == null
    ? undefined
    : { end: TITLE_SUFFIX.lastIndex, title: title[1] };
}

/**
 * Matches the longest known URL starting at `start` that is closed as a
 * reference. The walk stops at the first character no URL continues with, so
 * a position costs at most the URL text actually present there.
 */
function matchReference(
  markdown: string,
  start: number,
  { edges, terminals }: ReferenceTrie
): { text: string; end: number } | undefined {
  let best: { text: string; end: number } | undefined;
  let node = 0;
  for (let i = start; i < markdown.length; i++) {
    const next = edges.get(node * EDGE_STRIDE + markdown.charCodeAt(i));
    if (next == null) {
      break;
    }
    node = next;
    const target = terminals.get(node);
    if (target == null) {
      continue;
    }
    const close = referenceEnd(markdown, i + 1);
    if (close == null) {
      continue;
    }
    const label = `${target.type}#${target.idx}`;
    best = {
      text: close.title == null ? `(${label})` : `(${label} "${close.title}")`,
      end: close.end,
    };
  }
  return best;
}

/**
 * Rewrites every markdown reference to a known URL in one left-to-right pass.
 * One regex per URL over the whole page grew with links times page size and
 * blocked the event loop for seconds on a link-heavy page.
 */
function replaceReferences(markdown: string, trie: ReferenceTrie): string {
  if (trie.terminals.size === 0) {
    return markdown;
  }
  let result = '';
  let copied = 0;
  let open = markdown.indexOf('(');
  while (open !== -1) {
    const match = matchReference(markdown, open + 1, trie);
    if (match == null) {
      open = markdown.indexOf('(', open + 1);
      continue;
    }
    result += markdown.slice(copied, open) + match.text;
    copied = match.end;
    open = markdown.indexOf('(', match.end);
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
    buildTrie([
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

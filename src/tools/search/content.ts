import * as cheerio from 'cheerio';
import type { References, MediaReference } from './types';

type ReferenceType = 'image' | 'video' | 'iframe' | 'link';

interface ReferenceTarget {
  type: ReferenceType;
  idx: number;
}

interface ReferenceTargets {
  byUrl: Map<string, ReferenceTarget>;
  /** Distinct URL lengths, longest first, so a longer URL wins at the same position. */
  lengths: number[];
}

const TITLE_SUFFIX = /\s+"([^"]*)"\)/y;

/**
 * Indexes every referenced URL once. A URL listed under several types keeps the
 * first, in image, video, iframe, link order, which is the order the replacement
 * has always preferred.
 */
function collectTargets(
  groups: ReadonlyArray<[ReferenceType, Map<string, MediaReference>]>
): ReferenceTargets {
  const byUrl = new Map<string, ReferenceTarget>();
  const lengths = new Set<number>();
  for (const [type, references] of groups) {
    let idx = 0;
    for (const url of references.keys()) {
      idx++;
      if (byUrl.has(url)) {
        continue;
      }
      byUrl.set(url, { type, idx });
      lengths.add(url.length);
    }
  }
  return { byUrl, lengths: Array.from(lengths).sort((a, b) => b - a) };
}

/** Matches `(url)` or `(url "title")` for a known URL starting at `start`. */
function matchReference(
  markdown: string,
  start: number,
  { byUrl, lengths }: ReferenceTargets
): { text: string; end: number } | undefined {
  for (const length of lengths) {
    const urlEnd = start + length;
    if (urlEnd >= markdown.length) {
      continue;
    }
    const target = byUrl.get(markdown.slice(start, urlEnd));
    if (target == null) {
      continue;
    }
    const label = `${target.type}#${target.idx}`;
    if (markdown[urlEnd] === ')') {
      return { text: `(${label})`, end: urlEnd + 1 };
    }
    TITLE_SUFFIX.lastIndex = urlEnd;
    const title = TITLE_SUFFIX.exec(markdown);
    if (title != null) {
      return {
        text: `(${label} "${title[1]}")`,
        end: TITLE_SUFFIX.lastIndex,
      };
    }
  }
  return undefined;
}

/**
 * Rewrites every markdown reference to a known URL in one left-to-right pass.
 * Looking each candidate up by length keeps the work proportional to the
 * markdown, where one regex per URL over the whole page grew with links times
 * page size and blocked the event loop for seconds on a link-heavy page.
 */
function replaceReferences(
  markdown: string,
  targets: ReferenceTargets
): string {
  if (targets.byUrl.size === 0) {
    return markdown;
  }
  let result = '';
  let copied = 0;
  let open = markdown.indexOf('(');
  while (open !== -1) {
    const match = matchReference(markdown, open + 1, targets);
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
    collectTargets([
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

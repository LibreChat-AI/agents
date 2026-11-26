import type {
  GoogleGroundingMetadata,
  SearchResultData,
  ProcessedOrganic,
} from './types';
import type * as t from '@/types';
import { ContentTypes } from '@/common/enum';

/**
 * LibreChat's citation markers. They are private-use code points, while a
 * model following the web search prompt writes the escaped `\ue202` text, so
 * markers in these forms were inserted here.
 */
const ANCHOR = '\ue202';
const GROUP_START = '\ue200';
const GROUP_END = '\ue201';
const INSERTED_MARKERS = /\ue200|\ue201|\ue202turn\d+search\d+/g;

export type SearchRef = { turn: number; index: number };

/** A span of a model call's text that cites `sources` (organic indices). */
export type CitationSpan = { start: number; end: number; sources: number[] };

export type NativeSearchResults = {
  organic: ProcessedOrganic[];
  spans: CitationSpan[];
  queries?: string[];
};

export type UrlCitation = {
  block?: number;
  url: string;
  title?: string;
  start?: number;
  end?: number;
};

export type NativeSearchCall = {
  text: string;
  /** Where each step's text starts, in `text` and in the step's content part. */
  steps: { id: string; start: number; base: number }[];
  /** Offset in `text` where each provider text block starts. */
  blocks: Map<number, number>;
  citations: UrlCitation[];
  grounding?: GoogleGroundingMetadata;
  metadata?: Record<string, unknown>;
};

type PendingAnchors = {
  block?: number;
  refs: SearchRef[];
  metadata?: Record<string, unknown>;
};

type CitedPart = {
  type?: string;
  text?: unknown;
  index?: unknown;
  citations?: unknown;
  annotations?: unknown;
};

export function formatSearchAnchors(refs: readonly SearchRef[]): string {
  const anchors = refs
    .map(({ turn, index }) => `${ANCHOR}turn${turn}search${index}`)
    .join('');
  return refs.length > 1 ? `${GROUP_START}${anchors}${GROUP_END}` : anchors;
}

/** Removes the markers {@link formatSearchAnchors} writes, so replayed text matches what the provider sent. */
export function stripSearchAnchors(text: string): string {
  return text.replace(INSERTED_MARKERS, '');
}

/** Applies non-overlapping edits by position; an edit outside the text or over an earlier one is skipped. */
export function applyTextEdits(
  text: string,
  edits: readonly t.TextEdit[]
): string {
  const valid = edits
    .filter(
      ({ start, end }) =>
        Number.isSafeInteger(start) &&
        Number.isSafeInteger(end) &&
        start >= 0 &&
        start <= end &&
        end <= text.length
    )
    .sort((a, b) => a.start - b.start || a.end - b.end);
  let result = '';
  let cursor = 0;
  for (const edit of valid) {
    if (edit.start < cursor) {
      continue;
    }
    result += text.slice(cursor, edit.start) + edit.text;
    cursor = edit.end;
  }
  return result + text.slice(cursor);
}

function getUtf8Length(code: number): number {
  if (code < 0x80) {
    return 1;
  }
  if (code < 0x800) {
    return 2;
  }
  return code < 0x10000 ? 3 : 4;
}

/** Converts a UTF-8 byte offset into a UTF-16 index of `text`; undefined when it falls inside a character. */
export function byteToCharOffset(
  text: string,
  bytes: number
): number | undefined {
  let consumed = 0;
  let index = 0;
  while (consumed < bytes && index < text.length) {
    const code = text.codePointAt(index) ?? 0;
    consumed += getUtf8Length(code);
    index += code > 0xffff ? 2 : 1;
  }
  return consumed === bytes ? index : undefined;
}

/** Converts a count of code points after `from` into a UTF-16 index of `text`. */
export function codePointToCharOffset(
  text: string,
  from: number,
  codePoints: number
): number | undefined {
  let index = from;
  for (let count = 0; count < codePoints; count++) {
    if (index >= text.length) {
      return undefined;
    }
    index += (text.codePointAt(index) ?? 0) > 0xffff ? 2 : 1;
  }
  return index;
}

function getUrlCitations(part: CitedPart): UrlCitation[] {
  if (!Array.isArray(part.annotations)) {
    return [];
  }
  const block = typeof part.index === 'number' ? part.index : undefined;
  const citations: UrlCitation[] = [];
  for (const annotation of part.annotations as Record<string, unknown>[]) {
    const isUrlCitation =
      annotation.source === 'url_citation' ||
      annotation.type === 'url_citation';
    if (!isUrlCitation || typeof annotation.url !== 'string') {
      continue;
    }
    const start = annotation.startIndex ?? annotation.start_index;
    const end = annotation.endIndex ?? annotation.end_index;
    citations.push({
      block,
      url: annotation.url,
      title:
        typeof annotation.title === 'string' ? annotation.title : undefined,
      start: typeof start === 'number' ? start : undefined,
      end: typeof end === 'number' ? end : undefined,
    });
  }
  return citations;
}

/**
 * Run-scoped state that turns provider-native web search citations into the
 * `turnNsearchM` anchors hosts already render for the SDK's own search tool.
 */
export class SearchCitations {
  /** Organic results by link, so a cited URL resolves to its turn and index. */
  private sources = new Map<string, SearchRef>();
  /** Anchors for a cited block, written once the block ends. */
  private pending = new Map<string, PendingAnchors>();
  /** Text and citation data of a model call still streaming, by agent lane. */
  private calls = new Map<string, NativeSearchCall>();
  /** Text length dispatched to each tracked step, to place late edits. */
  private lengths = new Map<string, number>();
  /** Turns claimed by searches that never produce a tool call id. */
  private claimed = 0;

  register(data: SearchResultData): void {
    const turn = data.turn ?? 0;
    data.organic?.forEach((result, index) => {
      if (result.link) {
        this.sources.set(result.link, { turn, index });
      }
    });
  }

  /** The turn for a search identified by a tool call id that joins `invokedToolIds`. */
  turn(invoked: number): number {
    return invoked + this.claimed;
  }

  /** Claims a turn for a search with no tool call id of its own. */
  claimTurn(invoked: number): number {
    const turn = this.turn(invoked);
    this.claimed += 1;
    return turn;
  }

  /**
   * Adds the anchors of a cited block before the first delta of the next
   * block. Anthropic sends a block's citations before or after its text, and
   * the anchor belongs after the text, so it waits for the block to end.
   * Returns undefined when the content needs no change.
   */
  anchor(
    stepId: string,
    content: t.MessageContentComplex[],
    metadata?: Record<string, unknown>
  ): t.MessageContentComplex[] | undefined {
    let result: t.MessageContentComplex[] | undefined;
    for (let i = 0; i < content.length; i++) {
      const part = content[i] as CitedPart;
      const pending = this.pending.get(stepId);
      if (
        pending != null &&
        part.type === ContentTypes.TEXT &&
        part.index !== pending.block
      ) {
        result ??= content.slice(0, i);
        result.push({
          type: ContentTypes.TEXT,
          text: formatSearchAnchors(pending.refs),
        });
        this.pending.delete(stepId);
      }
      result?.push(content[i]);
      const refs = this.resolve(part);
      if (refs.length === 0) {
        continue;
      }
      const current = this.pending.get(stepId) ?? {
        block: typeof part.index === 'number' ? part.index : undefined,
        refs: [],
        metadata,
      };
      for (const ref of refs) {
        if (
          !current.refs.some(
            ({ turn, index }) => turn === ref.turn && index === ref.index
          )
        ) {
          current.refs.push(ref);
        }
      }
      this.pending.set(stepId, current);
    }
    return result;
  }

  /** Anchors still waiting on a step that is closing. */
  release(
    stepId: string
  ): { text: string; metadata?: Record<string, unknown> } | undefined {
    const pending = this.pending.get(stepId);
    if (pending == null) {
      return undefined;
    }
    this.pending.delete(stepId);
    return {
      text: formatSearchAnchors(pending.refs),
      metadata: pending.metadata,
    };
  }

  /** Records a step's streamed text and url citations for edits made when the model call ends. */
  track(
    lane: string,
    stepId: string,
    content: t.MessageContentComplex[],
    metadata?: Record<string, unknown>
  ): void {
    for (const item of content) {
      const part = item as CitedPart;
      if (part.type !== ContentTypes.TEXT) {
        continue;
      }
      const text = typeof part.text === 'string' ? part.text : '';
      const citations = getUrlCitations(part);
      if (text === '' && citations.length === 0) {
        continue;
      }
      const call = this.open(lane, metadata);
      call.citations.push(...citations);
      if (text === '') {
        continue;
      }
      const base = this.lengths.get(stepId) ?? 0;
      if (call.steps.at(-1)?.id !== stepId) {
        call.steps.push({ id: stepId, start: call.text.length, base });
      }
      if (typeof part.index === 'number' && !call.blocks.has(part.index)) {
        call.blocks.set(part.index, call.text.length);
      }
      call.text += text;
      this.lengths.set(stepId, base + text.length);
    }
  }

  /** Keeps the latest grounding payload of a model call. */
  ground(
    lane: string,
    grounding: GoogleGroundingMetadata,
    metadata?: Record<string, unknown>
  ): void {
    this.open(lane, metadata).grounding = grounding;
  }

  /** Hands over the finished model call of a lane. */
  take(lane: string): NativeSearchCall | undefined {
    const call = this.calls.get(lane);
    this.calls.delete(lane);
    return call;
  }

  /** Splits call-relative edits by the step that holds their text, in that step's offsets. */
  locate(
    call: NativeSearchCall,
    edits: readonly t.TextEdit[]
  ): Map<string, t.TextEdit[]> {
    const byStep = new Map<string, t.TextEdit[]>();
    for (const edit of edits) {
      const step = call.steps.findLast(
        ({ start }) => start < edit.end || (start === 0 && edit.end === 0)
      );
      if (step == null || edit.start < step.start) {
        continue;
      }
      const offset = step.base - step.start;
      const stepEdits = byStep.get(step.id) ?? [];
      stepEdits.push({
        start: edit.start + offset,
        end: edit.end + offset,
        text: edit.text,
      });
      byStep.set(step.id, stepEdits);
      this.lengths.set(
        step.id,
        (this.lengths.get(step.id) ?? 0) +
          edit.text.length -
          (edit.end - edit.start)
      );
    }
    return byStep;
  }

  private open(
    lane: string,
    metadata?: Record<string, unknown>
  ): NativeSearchCall {
    let call = this.calls.get(lane);
    if (call == null) {
      call = {
        text: '',
        steps: [],
        blocks: new Map(),
        citations: [],
        metadata,
      };
      this.calls.set(lane, call);
    }
    return call;
  }

  private resolve(part: CitedPart): SearchRef[] {
    if (part.type !== ContentTypes.TEXT || !Array.isArray(part.citations)) {
      return [];
    }
    const refs: SearchRef[] = [];
    for (const citation of part.citations as Record<string, unknown>[]) {
      if (
        citation.type !== 'web_search_result_location' ||
        typeof citation.url !== 'string'
      ) {
        continue;
      }
      const ref = this.sources.get(citation.url);
      if (ref != null) {
        refs.push(ref);
      }
    }
    return refs;
  }
}

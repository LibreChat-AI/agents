import type { GoogleGroundingMetadata, ProcessedOrganic } from './types';
import type { NativeSearchResults, CitationSpan } from './citations';
import { byteToCharOffset } from './citations';
import { getAttribution } from './utils';

type GroundedChunk = {
  additional_kwargs?: Record<string, unknown>;
  response_metadata?: Record<string, unknown>;
};

function isGroundingMetadata(value: unknown): value is GoogleGroundingMetadata {
  return (
    value != null &&
    typeof value === 'object' &&
    Array.isArray((value as GoogleGroundingMetadata).groundingChunks)
  );
}

/** Gemini API chunks carry it in `additional_kwargs`, Vertex AI chunks in `response_metadata`. */
export function getGoogleGroundingMetadata(
  chunk: GroundedChunk
): GoogleGroundingMetadata | undefined {
  const candidates = [
    chunk.additional_kwargs?.groundingMetadata,
    chunk.response_metadata?.groundingMetadata,
    chunk.response_metadata?.grounding_metadata,
  ];
  return candidates.find(isGroundingMetadata);
}

/**
 * Finds where a grounded segment ends in the call text. The API measures
 * offsets in UTF-8 bytes of one response part, so the byte offset is trusted
 * only for the first part and when the segment text sits right before it;
 * otherwise the segment text is searched from the previous segment's end.
 */
function locateSegmentEnd(
  text: string,
  segment: { partIndex?: number; endIndex?: number; text?: string },
  cursor: number
): number | undefined {
  const value = segment.text;
  const end =
    (segment.partIndex ?? 0) === 0 && typeof segment.endIndex === 'number'
      ? byteToCharOffset(text, segment.endIndex)
      : undefined;
  if (value == null || value === '') {
    return end;
  }
  if (end != null && text.slice(end - value.length, end) === value) {
    return end;
  }
  let index = text.indexOf(value, cursor);
  if (index === -1) {
    index = text.indexOf(value);
  }
  return index === -1 ? undefined : index + value.length;
}

/** Turns Gemini grounding into organic results and the spans each support cites. */
export function coerceGoogleGroundingResults(
  grounding: GoogleGroundingMetadata,
  text: string
): NativeSearchResults | undefined {
  const organic: ProcessedOrganic[] = [];
  const sourceByChunk = new Map<number, number>();
  grounding.groundingChunks?.forEach((chunk, index) => {
    const link = chunk.web?.uri;
    if (link == null || link === '') {
      return;
    }
    sourceByChunk.set(index, organic.length);
    organic.push({
      link,
      position: organic.length + 1,
      title: chunk.web?.title,
      attribution:
        chunk.web?.domain ?? chunk.web?.title ?? getAttribution(link),
    });
  });
  if (organic.length === 0) {
    return undefined;
  }

  const sourcesByEnd = new Map<number, number[]>();
  let cursor = 0;
  for (const support of grounding.groundingSupports ?? []) {
    const sources = [
      ...new Set(
        (support.groundingChunkIndices ?? []).flatMap(
          (index) => sourceByChunk.get(index) ?? []
        )
      ),
    ];
    if (sources.length === 0 || support.segment == null) {
      continue;
    }
    const end = locateSegmentEnd(text, support.segment, cursor);
    if (end == null) {
      continue;
    }
    cursor = end;
    const cited = sourcesByEnd.get(end) ?? [];
    sourcesByEnd.set(end, [...new Set([...cited, ...sources])]);
  }

  const spans: CitationSpan[] = [...sourcesByEnd].map(([end, sources]) => ({
    start: end,
    end,
    sources,
  }));
  return { organic, spans, queries: grounding.webSearchQueries };
}

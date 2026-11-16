import type { ChatGenerationChunk } from '@langchain/core/outputs';

/**
 * `@langchain/openai` stamps these scalar fields together onto every chunk
 * whose `choice.finish_reason` is set. Providers that emit `finish_reason` on
 * more than one streamed chunk (e.g. OpenRouter) otherwise make core's
 * `_mergeDicts` concatenate them into `stopstop` / duplicated model names — in
 * both the aggregated graph message and the Langfuse trace. Core keeps
 * `id`/`name`/`model_provider` last but not these, so keep the first occurrence
 * and drop later repeats at the source, before either aggregation runs.
 */
const REPEATED_SCALAR_METADATA_FIELDS = [
  'model_name',
  'finish_reason',
  'service_tier',
  'system_fingerprint',
  'provider',
  'usage',
] as const;

/** Per-completion-index set of scalar fields already emitted this stream. */
export type SeenScalarMetadata = Map<number, Set<string>>;

function completionIndex(
  generationInfo: Record<string, unknown> | undefined
): number {
  const completion = generationInfo?.completion;
  return typeof completion === 'number' ? completion : 0;
}

/**
 * Strip scalar `response_metadata`/`generationInfo` fields that have already
 * appeared on an earlier chunk of the same completion (tracked in `seen`),
 * keeping the first occurrence (first nonempty snapshot for usage) so the
 * aggregated message still carries the value once. Scoped per completion index
 * so multi-choice (`n > 1`) responses keep each choice's own metadata.
 *
 * Deletes from a shallow clone rather than in place: providers such as
 * `ChatDeepSeek` split one raw chunk into several synthetic pieces that share
 * the same `response_metadata`/`generationInfo` objects, so mutating in place
 * would strip fields from sibling chunks that were already emitted.
 */
export function dropRepeatedScalarMetadata(
  chunk: ChatGenerationChunk,
  seen: SeenScalarMetadata
): void {
  const generationInfo = chunk.generationInfo as
    | Record<string, unknown>
    | undefined;
  const responseMetadata = chunk.message.response_metadata as Record<
    string,
    unknown
  >;

  const index = completionIndex(generationInfo);
  let seenFields = seen.get(index);
  if (seenFields == null) {
    seenFields = new Set<string>();
    seen.set(index, seenFields);
  }

  const dropFromGenerationInfo: string[] = [];
  const dropFromResponseMetadata: string[] = [];
  for (const field of REPEATED_SCALAR_METADATA_FIELDS) {
    const inGenerationInfo =
      generationInfo != null && generationInfo[field] != null;
    const inResponseMetadata = responseMetadata[field] != null;
    if (!inGenerationInfo && !inResponseMetadata) {
      continue;
    }
    if (
      field === 'usage' &&
      (!inGenerationInfo ||
        Object.keys(generationInfo.usage ?? {}).length === 0) &&
      (!inResponseMetadata ||
        Object.keys(responseMetadata.usage ?? {}).length === 0)
    ) {
      continue;
    }
    if (!seenFields.has(field)) {
      seenFields.add(field);
      continue;
    }
    if (inGenerationInfo) {
      dropFromGenerationInfo.push(field);
    }
    if (inResponseMetadata) {
      dropFromResponseMetadata.push(field);
    }
  }

  if (dropFromGenerationInfo.length > 0 && generationInfo != null) {
    const cloned = { ...generationInfo };
    for (const field of dropFromGenerationInfo) {
      delete cloned[field];
    }
    chunk.generationInfo = cloned;
  }
  if (dropFromResponseMetadata.length > 0) {
    const cloned = { ...responseMetadata };
    for (const field of dropFromResponseMetadata) {
      delete cloned[field];
    }
    chunk.message.response_metadata = cloned;
  }
}

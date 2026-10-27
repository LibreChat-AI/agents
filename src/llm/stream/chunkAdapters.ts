import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { SmoothItem, SmoothPiece } from '@/llm/stream/smoother';
import { smoothStream } from '@/llm/stream/smoother';

/**
 * Rebuilds a generation chunk carrying one piece of a split plain-text chunk.
 * Unsplit pieces return the original chunk untouched, so disabled smoothing is
 * byte-identical to no smoothing. Usage metadata, additional kwargs and
 * response metadata survive only on the first piece: the aggregator merges
 * dicts by concatenating string fields, so replicating them across pieces
 * would duplicate reasoning text and scalar metadata once per piece.
 */
export function cloneGenerationChunkPiece(
  chunk: ChatGenerationChunk,
  piece: SmoothPiece
): ChatGenerationChunk {
  if (piece.isFirst && piece.isLast) {
    return chunk;
  }
  const message = chunk.message as AIMessageChunk;
  return new ChatGenerationChunk({
    text: piece.text,
    generationInfo: piece.isFirst ? chunk.generationInfo : undefined,
    message: new AIMessageChunk(
      Object.assign({}, message, {
        content: piece.text,
        usage_metadata: piece.isFirst ? message.usage_metadata : undefined,
        additional_kwargs: piece.isFirst ? message.additional_kwargs : {},
        response_metadata: piece.isFirst ? message.response_metadata : {},
      })
    ),
  });
}

const SPLITTABLE_TEXT_PART_KEYS: ReadonlySet<string> = new Set([
  'type',
  'text',
  'index',
]);

/**
 * Returns the sole plain text part of a single-element complex content array
 * (the shape google-common emits for text deltas), or null when the part
 * carries anything beyond `type`/`text`/`index` (thought signatures, media,
 * function calls) and must not be sliced. The part must carry a numeric
 * `index`: it is the aggregator's list-merge key, and without it each split
 * piece would append as a separate content part instead of merging back
 * into one — index-less parts pace whole instead.
 */
function getSplittableTextPart(
  message: AIMessageChunk,
  chunkText: string
): Record<string, unknown> | null {
  const content = message.content;
  if (!Array.isArray(content) || content.length !== 1) {
    return null;
  }
  const part: unknown = content[0];
  if (part == null || typeof part !== 'object') {
    return null;
  }
  const record = part as Record<string, unknown>;
  if (typeof record.text !== 'string' || record.text !== chunkText) {
    return null;
  }
  if (record.type != null && record.type !== 'text') {
    return null;
  }
  if (typeof record.index !== 'number') {
    return null;
  }
  if (!Object.keys(record).every((key) => SPLITTABLE_TEXT_PART_KEYS.has(key))) {
    return null;
  }
  return record;
}

function clonePartGenerationChunkPiece(
  chunk: ChatGenerationChunk,
  part: Record<string, unknown>,
  piece: SmoothPiece
): ChatGenerationChunk {
  if (piece.isFirst && piece.isLast) {
    return chunk;
  }
  const message = chunk.message as AIMessageChunk;
  return new ChatGenerationChunk({
    text: piece.text,
    generationInfo: piece.isFirst ? chunk.generationInfo : undefined,
    message: new AIMessageChunk(
      Object.assign({}, message, {
        content: [Object.assign({}, part, { text: piece.text })],
        usage_metadata: piece.isFirst ? message.usage_metadata : undefined,
        additional_kwargs: piece.isFirst ? message.additional_kwargs : {},
        response_metadata: piece.isFirst ? message.response_metadata : {},
      })
    ),
  });
}

/**
 * Provider-agnostic classification of a `ChatGenerationChunk` for the
 * smoothing engine:
 * - splittable: plain visible text — string content equal to `chunk.text`, or
 *   a single plain text part (google-common's delta shape) — with no
 *   logprobs / finish_reason; sliced adaptively at the pacing cadence.
 * - atomic: any other text-bearing chunk (complex content arrays, logprobs,
 *   finish_reason, provider-specific reasoning payloads surfaced via
 *   `getAtomicText`) — paced as one piece, never split.
 * - passthrough: tool-call deltas, usage-only and metadata chunks — strict
 *   FIFO, zero delay.
 */
/**
 * google-common stamps `logprobs: { content: [] }` on every chunk, so only
 * logprobs that actually carry data may block splitting here. This is looser
 * than the OpenAI-family adapter in `llm/openai/index.ts`, which treats ANY
 * logprobs as atomic (its providers only attach logprobs when requested, and
 * the DeepSeek suite pins that contract) — do not unify the two predicates.
 */
function hasMeaningfulLogprobs(
  generationInfo: ChatGenerationChunk['generationInfo']
): boolean {
  const logprobs = generationInfo?.logprobs;
  if (logprobs == null) {
    return false;
  }
  const content = (logprobs as { content?: unknown }).content;
  if (Array.isArray(content)) {
    return content.length > 0;
  }
  return true;
}

/** google-common reports the terminal reason camelCase (`finishReason`). */
function hasFinishReason(
  generationInfo: ChatGenerationChunk['generationInfo']
): boolean {
  return (
    generationInfo?.finish_reason != null || generationInfo?.finishReason != null
  );
}

/**
 * Chunks that pair visible text with reasoning payloads in
 * `additional_kwargs` (Gemini thought summaries, reasoning_content deltas,
 * OpenRouter reasoning_details) must pace whole: split pieces would each
 * carry the same kwargs and downstream merging — the aggregator's dict merge
 * or OpenRouter's reasoning_details accumulation — duplicates them once per
 * piece.
 */
/**
 * Mixed text/tool-call deltas must never split: cloned pieces would each
 * carry the same tool_call_chunks and downstream accumulation would corrupt
 * the assembled tool arguments.
 */
export function hasToolCallChunks(message: AIMessageChunk): boolean {
  return (message.tool_call_chunks?.length ?? 0) > 0;
}

export function hasReasoningKwargs(message: Pick<AIMessageChunk, 'additional_kwargs'>): boolean {
  const kwargs = message.additional_kwargs;
  if (
    typeof kwargs.reasoning_content === 'string' &&
    kwargs.reasoning_content.trim() !== ''
  ) {
    return true;
  }
  if (
    Array.isArray(kwargs.reasoning_details) &&
    kwargs.reasoning_details.length > 0
  ) {
    return true;
  }
  return typeof kwargs.reasoning === 'string'
    ? kwargs.reasoning.trim() !== ''
    : kwargs.reasoning != null;
}

/**
 * Extracts the visible reasoning text a kwargs-borne delta contributes, so
 * reasoning-only chunks (Gemini thoughts, DeepSeek reasoning_content,
 * OpenAI reasoning summaries, OpenRouter reasoning_details) pace atomically
 * at the cadence instead of passing through unsmoothed.
 */
export function getReasoningKwargsText(message: AIMessageChunk): string {
  const kwargs = message.additional_kwargs;
  if (
    typeof kwargs.reasoning_content === 'string' &&
    kwargs.reasoning_content !== ''
  ) {
    return kwargs.reasoning_content;
  }
  const reasoning = kwargs.reasoning;
  if (typeof reasoning === 'string' && reasoning !== '') {
    return reasoning;
  }
  if (reasoning != null && typeof reasoning === 'object') {
    const summaryText = (
      reasoning as { summary?: { text?: unknown }[] }
    ).summary?.[0]?.text;
    if (typeof summaryText === 'string' && summaryText !== '') {
      return summaryText;
    }
  }
  const details = kwargs.reasoning_details;
  if (Array.isArray(details)) {
    let text = '';
    for (const detail of details) {
      if (detail != null && typeof detail === 'object') {
        const detailText = (detail as { text?: unknown }).text;
        if (typeof detailText === 'string') {
          text += detailText;
        }
      }
    }
    if (text !== '') {
      return text;
    }
  }
  return '';
}

export function toGenerationSmoothItem(
  chunk: ChatGenerationChunk,
  getAtomicText?: (message: AIMessageChunk) => string
): SmoothItem<ChatGenerationChunk> {
  const { message } = chunk;
  const isMessageChunk = message instanceof AIMessageChunk;
  const cleanGenerationInfo =
    !hasMeaningfulLogprobs(chunk.generationInfo) &&
    !hasFinishReason(chunk.generationInfo);
  const splittable =
    Boolean(chunk.text) &&
    isMessageChunk &&
    typeof message.content === 'string' &&
    message.content === chunk.text &&
    cleanGenerationInfo &&
    !hasReasoningKwargs(message) &&
    !hasToolCallChunks(message);

  if (splittable) {
    return {
      text: chunk.text,
      smooth: true,
      emit: (piece) => cloneGenerationChunkPiece(chunk, piece),
    };
  }

  const splittablePart =
    Boolean(chunk.text) &&
    isMessageChunk &&
    cleanGenerationInfo &&
    !hasReasoningKwargs(message) &&
    !hasToolCallChunks(message)
      ? getSplittableTextPart(message, chunk.text)
      : null;
  if (splittablePart != null) {
    return {
      text: chunk.text,
      smooth: true,
      emit: (piece) =>
        clonePartGenerationChunkPiece(chunk, splittablePart, piece),
    };
  }

  const pacedText =
    chunk.text ||
    (isMessageChunk && getAtomicText != null ? getAtomicText(message) : '');
  if (pacedText !== '') {
    return {
      text: pacedText,
      smooth: true,
      atomic: true,
      emit: () => chunk,
    };
  }

  return { text: '', smooth: false, emit: () => chunk };
}

/**
 * Wraps a provider's raw chunk stream with adaptive smoothing and per-piece
 * `handleLLMNewToken` dispatch. The raw stream must NOT dispatch runManager
 * callbacks itself — callback-echo consumers would otherwise observe the
 * unsmoothed deltas.
 */
export async function* smoothGenerationChunks({
  chunks,
  delayMs,
  signal,
  runManager,
}: {
  chunks: AsyncGenerator<ChatGenerationChunk>;
  delayMs: number;
  signal?: AbortSignal;
  runManager?: CallbackManagerForLLMRun;
}): AsyncGenerator<ChatGenerationChunk> {
  const source = (async function* (): AsyncGenerator<
    SmoothItem<ChatGenerationChunk>
    > {
    for await (const chunk of chunks) {
      yield toGenerationSmoothItem(chunk, getReasoningKwargsText);
    }
  })();

  for await (const outputChunk of smoothStream({ source, delayMs, signal })) {
    yield outputChunk;
    await runManager?.handleLLMNewToken(
      outputChunk.text || '',
      undefined,
      undefined,
      undefined,
      undefined,
      { chunk: outputChunk }
    );
  }
}

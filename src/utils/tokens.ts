import { Tokenizer } from 'ai-tokenizer';
import { isProxy } from 'node:util/types';
import type { AIMessage, BaseMessage } from '@langchain/core/messages';
import {
  isComputerCallOutputContent,
  isAtomicToolContentBlock,
  serializeStructuredValueBounded,
} from './toolContent';
import { markTokenCounterCacheCompatible } from '@/llm/tokenCounterCacheCompatibility';
import { ContentTypes } from '@/common/enum';

export type EncodingName = 'o200k_base' | 'claude';

export type UnsafeTokenMeasurementReason =
  | 'message_proxy'
  | 'content_proxy'
  | 'metadata_proxy'
  | 'metadata_accessor'
  | 'metadata_limit'
  | 'invalid_count';

export class UnsafeTokenMeasurementError extends Error {
  readonly type = 'unsafe_token_measurement';
  readonly reason: UnsafeTokenMeasurementReason;
  readonly path: string;

  constructor({
    reason,
    path,
  }: {
    reason: UnsafeTokenMeasurementReason;
    path: string;
  }) {
    super(
      JSON.stringify({
        type: 'unsafe_token_measurement',
        reason,
        path,
      })
    );
    this.name = 'UnsafeTokenMeasurementError';
    this.reason = reason;
    this.path = path;
  }
}

function ensureSafeTokenMeasurement(value: number, path: string): number {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value >= Number.MAX_SAFE_INTEGER
  ) {
    throw new UnsafeTokenMeasurementError({
      reason: 'invalid_count',
      path,
    });
  }
  return value;
}

/** Anthropic minimum image token cost. */
const ANTHROPIC_IMAGE_MIN_TOKENS = 1024;
/** Anthropic divisor: tokens = width × height / 750. */
const ANTHROPIC_IMAGE_DIVISOR = 750;
/** OpenAI low-detail fixed cost. */
const OPENAI_IMAGE_LOW_TOKENS = 85;
/** OpenAI high-detail tile size. */
const OPENAI_IMAGE_TILE_SIZE = 512;
/** OpenAI high-detail tokens per tile. */
const OPENAI_IMAGE_TOKENS_PER_TILE = 170;
/** Google Gemini fixed per-image cost. */
const _GEMINI_IMAGE_TOKENS = 258;
/** Safety margin for image and document token estimates (5% overestimate). */
export const IMAGE_TOKEN_SAFETY_MARGIN = 1.05;

/**
 * Anthropic PDF: each page costs image tokens + text tokens.
 * Typical range is 1500-3000 tokens/page. Using 2000 as midpoint.
 */
const ANTHROPIC_PDF_TOKENS_PER_PAGE = 2000;
/** OpenAI PDF: each page rendered as high-detail image. ~1500 tokens typical. */
const OPENAI_PDF_TOKENS_PER_PAGE = 1500;
/** Gemini PDF: fixed 258 tokens per page. */
const _GEMINI_PDF_TOKENS_PER_PAGE = 258;
/** Approximate base64 bytes per PDF page for page count estimation. */
const BASE64_BYTES_PER_PDF_PAGE = 75_000;
/** Fallback token cost for URL-referenced documents without local data. */
const URL_DOCUMENT_FALLBACK_TOKENS = 2000;
/** Approximate decoded bytes per token for a binary (Office) document. */
const BINARY_DOCUMENT_BYTES_PER_TOKEN = 4;

/**
 * Timed media (video/audio) is priced by DURATION, not size, and the content
 * carries no duration — so duration is estimated from encoded size at
 * representative bitrates and priced at Gemini's rates (the dominant provider
 * for native video/audio). Deliberately rough; superseded by real provider
 * usage after the first turn.
 */
/** Gemini video ≈ 258 tokens/frame (1 fps) + 32 tokens/s audio ≈ 300 tokens/s. */
const VIDEO_TOKENS_PER_SECOND = 300;
/** Gemini audio: 32 tokens/s (single channel). */
const AUDIO_TOKENS_PER_SECOND = 32;
/** Representative encoded byte rates for duration-from-size estimation. */
const VIDEO_BYTES_PER_SECOND = 250_000; // ~2 Mbps
const AUDIO_BYTES_PER_SECOND = 16_000; // ~128 kbps
/** Flat fallback when only a URL is present (no size to estimate from) — ~30s. */
const VIDEO_URL_FALLBACK_TOKENS = 9000;
const AUDIO_URL_FALLBACK_TOKENS = 960;
/** Content block types that can carry timed media (video/audio). `media` is
 *  generic (Google) so it is classified by MIME; the rest are unambiguous. */
const TIMED_MEDIA_TYPES = new Set([
  'media',
  'video',
  'audio',
  'video_url',
  'input_audio',
]);
/**
 * Structured content is tokenized exactly up to this size. Larger values are
 * traversed by the bounded serializer and priced conservatively for the omitted
 * suffix, avoiding a full JSON string allocation in the context guard.
 */
const MAX_STRUCTURED_TOKENIZATION_CHARS = 200_000;
/** One UTF-16 character can encode to several tokenizer pieces. */
const MAX_TOKENS_PER_OMITTED_STRUCTURED_CHAR = 4;
const MAX_STRUCTURED_SAFETY_INSPECTION_WORK = 10_000;
const MAX_RAW_TOOL_CALLS = MAX_STRUCTURED_SAFETY_INSPECTION_WORK;
/** Bounds BPE work per plain-text segment without omitting any text. */
const MAX_TEXT_TOKENIZATION_CHARS = 8_192;
/** Covers tokenizer segmentation changes on both sides of a chunk boundary. */
const TEXT_TOKEN_CHUNK_BOUNDARY_SAFETY_TOKENS = 4;

/**
 * Extracts image dimensions from the first bytes of a base64-encoded
 * PNG, JPEG, GIF, or WebP without decoding the full image.
 * Returns null if the format is unrecognized or data is too short.
 */
export function extractImageDimensions(
  base64Data: string
): { width: number; height: number } | null {
  const raw = base64Data.startsWith('data:')
    ? base64Data.slice(base64Data.indexOf(',') + 1)
    : base64Data;

  if (raw.length < 32) {
    return null;
  }

  const bytes = new Uint8Array(Buffer.from(raw.slice(0, 80), 'base64'));

  if (bytes[0] === 0x89 && bytes[1] === 0x50) {
    // PNG: width at bytes 16-19, height at 20-23 (big-endian)
    const width =
      (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19];
    const height =
      (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23];
    return { width, height };
  }

  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    // JPEG: scan for SOF0 (0xFFC0) or SOF2 (0xFFC2) marker
    for (let i = 2; i < bytes.length - 9; i++) {
      if (
        bytes[i] === 0xff &&
        (bytes[i + 1] === 0xc0 || bytes[i + 1] === 0xc2)
      ) {
        const height = (bytes[i + 5] << 8) | bytes[i + 6];
        const width = (bytes[i + 7] << 8) | bytes[i + 8];
        return { width, height };
      }
    }
    return null;
  }

  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) {
    // GIF: width at bytes 6-7, height at 8-9 (little-endian)
    const width = bytes[6] | (bytes[7] << 8);
    const height = bytes[8] | (bytes[9] << 8);
    return { width, height };
  }

  if (
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    // WebP VP8: width at bytes 26-27, height at 28-29
    if (bytes.length > 29) {
      const width = (bytes[26] | (bytes[27] << 8)) & 0x3fff;
      const height = (bytes[28] | (bytes[29] << 8)) & 0x3fff;
      return { width, height };
    }
    return null;
  }

  return null;
}

/** Estimates image token cost for Anthropic/Bedrock (Claude). */
export function estimateAnthropicImageTokens(
  width: number,
  height: number
): number {
  return Math.max(
    ANTHROPIC_IMAGE_MIN_TOKENS,
    Math.ceil((width * height) / ANTHROPIC_IMAGE_DIVISOR)
  );
}

/** Estimates image token cost for OpenAI (high detail). */
export function estimateOpenAIImageTokens(
  width: number,
  height: number,
  detail: string = 'high'
): number {
  if (detail === 'low') {
    return OPENAI_IMAGE_LOW_TOKENS;
  }
  const tiles =
    Math.ceil(width / OPENAI_IMAGE_TILE_SIZE) *
    Math.ceil(height / OPENAI_IMAGE_TILE_SIZE);
  return OPENAI_IMAGE_LOW_TOKENS + tiles * OPENAI_IMAGE_TOKENS_PER_TILE;
}

/**
 * Estimates token cost for an image content block.
 * Extracts dimensions from base64 header when available.
 * Falls back to Anthropic minimum (1024) when dimensions can't be determined.
 */
export function estimateImageBlockTokens(
  block: Record<string, unknown>,
  encoding: EncodingName
): number {
  let base64Data: string | undefined;

  if (
    block.type === ContentTypes.IMAGE_URL ||
    block.type === 'image_url' ||
    block.type === 'computer_screenshot' ||
    block.type === 'input_image'
  ) {
    const imageUrl = block.image_url as string | { url?: string } | undefined;
    const url = typeof imageUrl === 'string' ? imageUrl : imageUrl?.url;
    if (typeof url === 'string' && url.startsWith('data:')) {
      base64Data = url;
    } else if (typeof block.file_id === 'string' && block.file_id !== '') {
      return block.detail === 'low'
        ? OPENAI_IMAGE_LOW_TOKENS
        : ANTHROPIC_IMAGE_MIN_TOKENS;
    } else {
      return ANTHROPIC_IMAGE_MIN_TOKENS;
    }
  } else if (block.type === 'image') {
    const source = block.source as { type?: string; data?: string } | undefined;
    if (source?.type === 'base64' && typeof source.data === 'string') {
      base64Data = source.data;
    } else {
      return ANTHROPIC_IMAGE_MIN_TOKENS;
    }
  } else {
    return ANTHROPIC_IMAGE_MIN_TOKENS;
  }

  const dims = extractImageDimensions(base64Data);
  if (dims == null) {
    return ANTHROPIC_IMAGE_MIN_TOKENS;
  }

  if (encoding === 'claude') {
    return estimateAnthropicImageTokens(dims.width, dims.height);
  }
  return estimateOpenAIImageTokens(dims.width, dims.height);
}

/**
 * A base64 document that is neither a PDF nor an image reaches the model as
 * its contents, not as base64: a textual one is counted by its decoded text,
 * a binary one (Office formats) by its decoded size. Tokenizing the base64
 * itself roughly doubles a text file's count.
 */
function estimateBase64DocumentTokens(
  data: string,
  mimeType: string,
  getTokenCount: (text: string) => number
): number {
  if (!mimeType.startsWith('text/')) {
    return Math.ceil(
      Buffer.byteLength(data, 'base64') / BINARY_DOCUMENT_BYTES_PER_TOKEN
    );
  }
  return getTokenCount(Buffer.from(data, 'base64').toString('utf8'));
}

/**
 * Estimates token cost for a document/file content block.
 * Handles both LangChain standard format (`type: 'file'` with `source_type`)
 * and Anthropic format (`type: 'document'` with `source`).
 *
 * - Plain text: tokenized directly via `getTokenCount`.
 * - Audio and video: priced by duration, as {@link estimateTimedMediaBlockTokens}.
 * - Base64 PDF: page count estimated from base64 length × per-page cost.
 * - Other base64 documents: decoded text, or decoded size when binary.
 * - URL reference: conservative flat estimate.
 */
export function estimateDocumentBlockTokens(
  block: Record<string, unknown>,
  encoding: EncodingName,
  getTokenCount: (text: string) => number
): number {
  const pdfTokensPerPage =
    encoding === 'claude'
      ? ANTHROPIC_PDF_TOKENS_PER_PAGE
      : OPENAI_PDF_TOKENS_PER_PAGE;

  // LangChain standard format: type='file', source_type, data/text/url, mime_type
  const sourceType = block.source_type as string | undefined;
  if (typeof sourceType === 'string') {
    const mimeType = ((block.mime_type as string | undefined) ?? '').split(
      ';'
    )[0];

    if (sourceType === 'text' && typeof block.text === 'string') {
      return getTokenCount(block.text as string);
    }

    const mediaMimeType = normalizeMediaMimeType(mimeType);
    if (
      mediaMimeType.startsWith('audio/') ||
      mediaMimeType.startsWith('video/')
    ) {
      return estimateTimedMediaBlockTokens({ ...block, type: 'media' });
    }

    if (sourceType === 'base64' && typeof block.data === 'string') {
      if (mimeType === 'application/pdf' || mimeType === '') {
        const pageEstimate = Math.max(
          1,
          Math.ceil((block.data as string).length / BASE64_BYTES_PER_PDF_PAGE)
        );
        return pageEstimate * pdfTokensPerPage;
      }
      // Image inside a file block — delegate to image estimation
      if (mimeType.startsWith('image/')) {
        return estimateImageBlockTokens(
          {
            ...block,
            type: 'image',
            source: { type: 'base64', data: block.data },
          },
          encoding
        );
      }
      return estimateBase64DocumentTokens(
        block.data as string,
        mimeType,
        getTokenCount
      );
    }

    if (sourceType === 'url') {
      return URL_DOCUMENT_FALLBACK_TOKENS;
    }

    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  // Anthropic format: type='document', source: { type, data, media_type }
  const source = block.source as
    | {
        type?: string;
        data?: string;
        media_type?: string;
        content?: unknown[];
      }
    | undefined;

  if (source == null) {
    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  if (source.type === 'text' && typeof source.data === 'string') {
    return getTokenCount(source.data);
  }

  if (source.type === 'base64' && typeof source.data === 'string') {
    const mediaType = (source.media_type ?? '').split(';')[0];
    if (mediaType === 'application/pdf' || mediaType === '') {
      const pageEstimate = Math.max(
        1,
        Math.ceil(source.data.length / BASE64_BYTES_PER_PDF_PAGE)
      );
      return pageEstimate * pdfTokensPerPage;
    }
    if (mediaType.startsWith('image/')) {
      return estimateImageBlockTokens(
        { type: 'image', source: { type: 'base64', data: source.data } },
        encoding
      );
    }
    return estimateBase64DocumentTokens(source.data, mediaType, getTokenCount);
  }

  if (source.type === 'url') {
    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  // content-type source (wraps other blocks like images)
  if (source.type === 'content' && Array.isArray(source.content)) {
    let total = 0;
    for (const inner of source.content) {
      if (inner != null && typeof inner === 'object' && 'type' in inner) {
        const innerBlock = inner as Record<string, unknown>;
        if (innerBlock.type === 'image') {
          total += estimateImageBlockTokens(innerBlock, encoding);
        }
      }
    }
    return total > 0 ? total : URL_DOCUMENT_FALLBACK_TOKENS;
  }

  return URL_DOCUMENT_FALLBACK_TOKENS;
}

/** Decoded byte length of base64 or a `data:` URL; 0 for any remote URI (http,
 *  gs, s3, file, …) or empty. Base64 never contains `:`, so any scheme prefix
 *  marks a remote reference with no local size. */
function base64ByteLength(value: string | undefined): number {
  if (typeof value !== 'string' || value.length === 0) {
    return 0;
  }
  if (value.startsWith('data:')) {
    const comma = value.indexOf(',');
    return comma < 0 ? 0 : Math.floor(((value.length - comma - 1) * 3) / 4);
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) {
    return 0;
  }
  return Math.floor((value.length * 3) / 4);
}

function timedMediaTokens(
  bytes: number,
  bytesPerSecond: number,
  tokensPerSecond: number,
  urlFallback: number
): number {
  if (bytes <= 0) {
    return urlFallback;
  }
  return Math.max(
    tokensPerSecond,
    Math.ceil((bytes / bytesPerSecond) * tokensPerSecond)
  );
}

/** Decoded byte length of a media block payload — top-level `data` (base64
 *  string or `Uint8Array`) or base64 `url`, or the nested `video|audio.{data,
 *  url, source.bytes}` shapes (`formatMessage` media arrays / native Bedrock);
 *  0 for a bare remote URL, `fileId`/`fileUri`, S3 location, or empty. */
function mediaBlockByteLength(block: Record<string, unknown>): number {
  const data = block.data;
  if (typeof data === 'string') {
    return base64ByteLength(data);
  }
  if (data instanceof Uint8Array) {
    return data.length;
  }
  if (typeof block.url === 'string') {
    return base64ByteLength(block.url);
  }
  const nested = (block.video ?? block.audio) as
    | { data?: unknown; url?: unknown; source?: { bytes?: unknown } }
    | undefined;
  if (nested != null) {
    if (typeof nested.data === 'string') {
      return base64ByteLength(nested.data);
    }
    if (typeof nested.url === 'string') {
      return base64ByteLength(nested.url);
    }
    if (nested.source?.bytes instanceof Uint8Array) {
      return nested.source.bytes.length;
    }
  }
  return 0;
}

/** Classifies a block as timed media, or null when it is not video/audio — e.g.
 *  a generic Google `media` block carrying an image/document MIME, which must
 *  NOT be priced as video. Also handles the Google shape where `type` IS the
 *  MIME string (e.g. `{ type: 'audio/wav', data }`). */
function timedMediaKind(
  block: Record<string, unknown>
): 'video' | 'audio' | null {
  const type =
    typeof block.type === 'string' ? normalizeMediaMimeType(block.type) : '';
  if (type === 'input_audio' || type === 'audio') {
    return 'audio';
  }
  if (type === 'video_url' || type === 'video') {
    return 'video';
  }
  const mediaMime = getMediaMimeType(block);
  const mime = type === 'media' && mediaMime != null ? mediaMime : type;
  if (mime.startsWith('audio/')) {
    return 'audio';
  }
  if (mime.startsWith('video/')) {
    return 'video';
  }
  return null;
}

/** The normalized MIME type a media or file content block declares, if any. */
export function getMediaMimeType(
  block: Record<string, unknown>
): string | undefined {
  if (typeof block.mimeType === 'string') {
    return normalizeMediaMimeType(block.mimeType);
  }
  if (typeof block.mime_type === 'string') {
    return normalizeMediaMimeType(block.mime_type);
  }
  return undefined;
}

function normalizeMediaMimeType(mimeType: string): string {
  return mimeType.split(';')[0].trim().toLowerCase();
}

/** Whether a content-block `type` can carry timed media — the fixed set plus the
 *  Google MIME-as-type shape (`audio/*` / `video/*`). Gates callers before they
 *  hand the block to {@link estimateTimedMediaBlockTokens}. */
function isTimedMediaType(type: string): boolean {
  const normalizedType = normalizeMediaMimeType(type);
  return (
    TIMED_MEDIA_TYPES.has(normalizedType) ||
    normalizedType.startsWith('audio/') ||
    normalizedType.startsWith('video/')
  );
}

/** Conservatively prices unknown inline media without materializing JSON. */
function estimateUnknownMediaBlockTokens(data: unknown): number {
  if (typeof data === 'string') {
    if (base64ByteLength(data) <= 0) {
      return URL_DOCUMENT_FALLBACK_TOKENS;
    }
    return Math.max(URL_DOCUMENT_FALLBACK_TOKENS, data.length);
  }
  if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
    return Math.max(
      URL_DOCUMENT_FALLBACK_TOKENS,
      Math.ceil((data.byteLength * 4) / 3)
    );
  }
  return URL_DOCUMENT_FALLBACK_TOKENS;
}

/**
 * Estimates Google `type: "media"` blocks whose MIME describes static content.
 * Timed audio/video continues through {@link estimateTimedMediaBlockTokens}.
 */
function estimateStaticMediaBlockTokens(
  block: Record<string, unknown>,
  encoding: EncodingName,
  getTokenCount: (text: string) => number
): number | undefined {
  const mediaMime = getMediaMimeType(block);
  if (block.type !== 'media' || mediaMime == null) {
    return undefined;
  }

  const mimeType = mediaMime;
  const nestedMedia =
    block.media != null && typeof block.media === 'object'
      ? (block.media as Record<string, unknown>)
      : undefined;
  const data =
    block.data ?? block.bytes ?? nestedMedia?.data ?? nestedMedia?.bytes;
  const reference =
    block.url ??
    block.fileUri ??
    block.fileId ??
    nestedMedia?.url ??
    nestedMedia?.fileUri ??
    nestedMedia?.fileId;

  if (mimeType.startsWith('image/')) {
    let encodedImage: string | undefined;
    if (typeof data === 'string') {
      encodedImage = data;
    } else if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      const view =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      encodedImage = Buffer.from(view.subarray(0, 80)).toString('base64');
    }

    if (encodedImage != null && encodedImage.length > 0) {
      return estimateImageBlockTokens(
        {
          type: 'image',
          source: { type: 'base64', data: encodedImage },
        },
        encoding
      );
    }
    return estimateImageBlockTokens(
      {
        type: 'image_url',
        image_url: { url: typeof reference === 'string' ? reference : '' },
      },
      encoding
    );
  }

  if (mimeType === 'application/pdf') {
    if (typeof data === 'string' && data.length > 0) {
      const comma = data.startsWith('data:') ? data.indexOf(',') : -1;
      const base64Data = comma >= 0 ? data.slice(comma + 1) : data;
      return estimateDocumentBlockTokens(
        {
          type: 'file',
          source_type: 'base64',
          mime_type: mimeType,
          data: base64Data,
        },
        encoding,
        getTokenCount
      );
    }
    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      const estimatedBase64Chars = Math.ceil((data.byteLength * 4) / 3);
      const pdfTokensPerPage =
        encoding === 'claude'
          ? ANTHROPIC_PDF_TOKENS_PER_PAGE
          : OPENAI_PDF_TOKENS_PER_PAGE;
      return (
        Math.max(
          1,
          Math.ceil(estimatedBase64Chars / BASE64_BYTES_PER_PDF_PAGE)
        ) * pdfTokensPerPage
      );
    }
    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  if (mimeType.startsWith('audio/') || mimeType.startsWith('video/')) {
    return undefined;
  }
  return estimateUnknownMediaBlockTokens(data);
}

/**
 * Estimates token cost for a timed-media block (video/audio). Handles Google
 * `{ type: 'media', mimeType, data|url|fileUri }`, OpenRouter
 * `{ type: 'video_url' }` / `{ type: 'input_audio' }`, and standard
 * `{ type: 'video' }` / `{ type: 'audio' }` blocks (payload as `data` base64 or
 * `Uint8Array`, base64 `url`, or `fileId`). Duration is inferred from encoded
 * size (providers price by duration, which the block does not carry) at Gemini's
 * rates; a payload with no size (bare URL / file id) falls back to a ~30s
 * estimate. Returns 0 for non-timed media (e.g. an image/document `media` block)
 * so it is never mispriced as video.
 */
export function estimateTimedMediaBlockTokens(
  block: Record<string, unknown>
): number {
  const kind = timedMediaKind(block);
  if (kind == null) {
    return 0;
  }
  let bytes: number;
  if (block.type === 'input_audio') {
    const audio = block.input_audio as { data?: string } | undefined;
    bytes = base64ByteLength(audio?.data);
  } else if (block.type === 'video_url') {
    const video = block.video_url as string | { url?: string } | undefined;
    bytes = base64ByteLength(typeof video === 'string' ? video : video?.url);
  } else {
    bytes = mediaBlockByteLength(block);
  }
  if (kind === 'audio') {
    return timedMediaTokens(
      bytes,
      AUDIO_BYTES_PER_SECOND,
      AUDIO_TOKENS_PER_SECOND,
      AUDIO_URL_FALLBACK_TOKENS
    );
  }
  return timedMediaTokens(
    bytes,
    VIDEO_BYTES_PER_SECOND,
    VIDEO_TOKENS_PER_SECOND,
    VIDEO_URL_FALLBACK_TOKENS
  );
}

const tokenizers: Partial<Record<EncodingName, Tokenizer>> = {};

async function getTokenizer(
  encoding: EncodingName = 'o200k_base'
): Promise<Tokenizer> {
  const cached = tokenizers[encoding];
  if (cached) {
    return cached;
  }
  const data =
    encoding === 'claude'
      ? await import('ai-tokenizer/encoding/claude')
      : await import('ai-tokenizer/encoding/o200k_base');
  const instance = new Tokenizer(data);
  tokenizers[encoding] = instance;
  return instance;
}

export function encodingForModel(model: string): EncodingName {
  if (model.toLowerCase().includes('claude')) {
    return 'claude';
  }
  return 'o200k_base';
}

type StructuredSafetyInspection = {
  remaining: number;
};

function consumeStructuredSafetyWork(
  inspection: StructuredSafetyInspection
): boolean {
  if (inspection.remaining <= 0) {
    return false;
  }
  inspection.remaining--;
  return true;
}

function hasUnsafeToJSON(
  value: object,
  inspection: StructuredSafetyInspection
): boolean {
  let current: object | null = value;
  const seen = new Set<object>();
  try {
    while (current != null) {
      if (isProxy(current)) {
        return true;
      }
      if (seen.has(current) || !consumeStructuredSafetyWork(inspection)) {
        return true;
      }
      seen.add(current);
      const descriptor = Object.getOwnPropertyDescriptor(current, 'toJSON');
      if (descriptor != null) {
        return (
          ('value' in descriptor && typeof descriptor.value === 'function') ||
          !('value' in descriptor)
        );
      }
      current = Object.getPrototypeOf(current) as object | null;
    }
  } catch {
    return true;
  }
  return false;
}

const arrayBufferByteLengthGetter = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength'
)?.get;

/**
 * Tests native binary values without `instanceof`, whose prototype walk can
 * recurse or throw for adversarial proxies.
 */
function isNativeBinaryValue(value: object): boolean {
  try {
    if (ArrayBuffer.isView(value)) {
      return true;
    }
  } catch {
    return true;
  }
  if (arrayBufferByteLengthGetter == null) {
    return false;
  }
  try {
    arrayBufferByteLengthGetter.call(value);
    return true;
  } catch {
    return false;
  }
}

/**
 * Detects an inherited accessor without reading it. Any proxy/prototype trap or
 * cyclic prototype chain is unsafe rather than observable to the caller.
 */
function hasUnsafeInheritedProperty(
  value: object,
  key: string,
  inspection: StructuredSafetyInspection
): boolean {
  const seen = new Set<object>([value]);
  let current: object | null;
  try {
    current = Object.getPrototypeOf(value) as object | null;
  } catch {
    return true;
  }

  while (current != null) {
    if (isProxy(current)) {
      return true;
    }
    if (seen.has(current) || !consumeStructuredSafetyWork(inspection)) {
      return true;
    }
    seen.add(current);
    let descriptor: PropertyDescriptor | undefined;
    try {
      descriptor = Object.getOwnPropertyDescriptor(current, key);
    } catch {
      return true;
    }
    if (descriptor != null) {
      return !('value' in descriptor);
    }
    try {
      current = Object.getPrototypeOf(current) as object | null;
    } catch {
      return true;
    }
  }
  return false;
}

/**
 * Detects values whose native JSON representation can differ after local
 * measurement. Accessor properties and `toJSON` hooks are deliberately not
 * invoked; values too large to inspect within bounded work are treated as
 * unsafe so the caller can normalize or reject them conservatively.
 */
export function hasUnsafeStructuredSerialization(value: unknown): boolean {
  if (typeof value === 'bigint') {
    return true;
  }
  if (value == null || typeof value !== 'object') {
    return false;
  }
  if (isProxy(value)) {
    return true;
  }
  if (isNativeBinaryValue(value)) {
    return true;
  }

  const stack: object[] = [value];
  const seen = new Set<object>();
  const inspection: StructuredSafetyInspection = {
    remaining: MAX_STRUCTURED_SAFETY_INSPECTION_WORK,
  };
  while (stack.length > 0) {
    const current = stack.pop() as object;
    if (isProxy(current)) {
      return true;
    }
    if (seen.has(current)) {
      return true;
    }
    seen.add(current);
    if (!consumeStructuredSafetyWork(inspection)) {
      return true;
    }

    try {
      if (hasUnsafeToJSON(current, inspection)) {
        return true;
      }
      for (const key in current) {
        if (!consumeStructuredSafetyWork(inspection)) {
          return true;
        }
        const descriptor = Object.getOwnPropertyDescriptor(current, key);
        if (descriptor == null) {
          if (hasUnsafeInheritedProperty(current, key, inspection)) {
            return true;
          }
          continue;
        }
        if (descriptor.enumerable !== true) {
          continue;
        }
        if (!('value' in descriptor)) {
          return true;
        }
        if ('value' in descriptor) {
          if (typeof descriptor.value === 'bigint') {
            return true;
          }
          if (
            descriptor.value != null &&
            typeof descriptor.value === 'object' &&
            isNativeBinaryValue(descriptor.value)
          ) {
            return true;
          }
          if (
            descriptor.value != null &&
            typeof descriptor.value === 'object'
          ) {
            stack.push(descriptor.value);
          }
        }
      }
    } catch {
      return true;
    }
  }
  return false;
}

function getBoundedTextTokenCount(
  value: string,
  getTokenCount: (text: string) => number
): number {
  let numTokens = 0;
  for (
    let offset = 0;
    offset < value.length;
    offset += MAX_TEXT_TOKENIZATION_CHARS
  ) {
    const chunkTokens = ensureSafeTokenMeasurement(
      getTokenCount(value.slice(offset, offset + MAX_TEXT_TOKENIZATION_CHARS)),
      'tokenizer'
    );
    const boundaryTokens =
      offset > 0 ? TEXT_TOKEN_CHUNK_BOUNDARY_SAFETY_TOKENS : 0;
    numTokens = Math.min(
      Number.MAX_SAFE_INTEGER,
      numTokens + chunkTokens + boundaryTokens
    );
  }
  return numTokens;
}

function getBoundedStructuredTokenCount(
  value: unknown,
  getTokenCount: (text: string) => number
): number {
  const serialized = serializeStructuredValueBounded(
    value,
    MAX_STRUCTURED_TOKENIZATION_CHARS
  );
  const previewTokens = ensureSafeTokenMeasurement(
    getTokenCount(serialized.content),
    'tokenizer'
  );
  if (!serialized.truncated) {
    return previewTokens;
  }
  // The bounded serializer uses MAX_SAFE_INTEGER as its unknown-size sentinel.
  // Do not extrapolate from it: use its bounded preview and let the provider
  // input projection preserve the same bounded behavior before invocation.
  if (
    !Number.isSafeInteger(serialized.originalChars) ||
    serialized.originalChars >= Number.MAX_SAFE_INTEGER
  ) {
    return previewTokens;
  }

  const omittedChars = Math.max(
    0,
    serialized.originalChars - serialized.content.length
  );
  return Math.min(
    Number.MAX_SAFE_INTEGER,
    previewTokens + omittedChars * MAX_TOKENS_PER_OMITTED_STRUCTURED_CHAR
  );
}

export function readTokenMetadataProperty(
  value: object,
  key: PropertyKey,
  path: string
): unknown {
  const seen = new Set<object>();
  let current: object | null = value;
  for (let depth = 0; current != null && depth < 100; depth++) {
    if (isProxy(current)) {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_proxy',
        path,
      });
    }
    if (seen.has(current)) {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_accessor',
        path,
      });
    }
    seen.add(current);
    let descriptor: PropertyDescriptor | undefined;
    try {
      descriptor = Object.getOwnPropertyDescriptor(current, key);
    } catch {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_accessor',
        path,
      });
    }
    if (descriptor != null) {
      if (!('value' in descriptor)) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_accessor',
          path,
        });
      }
      return descriptor.value;
    }
    try {
      current = Object.getPrototypeOf(current) as object | null;
    } catch {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_accessor',
        path,
      });
    }
  }
  if (current != null) {
    throw new UnsafeTokenMeasurementError({
      reason: 'metadata_accessor',
      path,
    });
  }
  return undefined;
}

export function getTokenCountForMessage(
  message: BaseMessage,
  getTokenCount: (text: string) => number,
  encoding: EncodingName = 'o200k_base'
): number {
  const tokensPerMessage = 3;
  const countText = (text: string): number =>
    getBoundedTextTokenCount(text, getTokenCount);
  if (isProxy(message)) {
    throw new UnsafeTokenMeasurementError({
      reason: 'message_proxy',
      path: 'message',
    });
  }

  type ContentBlock = Record<string, unknown> & {
    type?: string;
    tool_call?: {
      id?: string;
      name?: string;
      args?: unknown;
      output?: unknown;
    };
  };
  const representedToolCallIds = new Set<string>();

  const processValue = (value: unknown, path: string): void => {
    if (value != null && typeof value === 'object' && isProxy(value)) {
      throw new UnsafeTokenMeasurementError({
        reason: 'content_proxy',
        path,
      });
    }
    if (Array.isArray(value)) {
      for (let index = 0; index < value.length; index++) {
        const raw = value[index];
        if (
          typeof raw === 'string' ||
          typeof raw === 'number' ||
          typeof raw === 'boolean'
        ) {
          processValue(raw, `${path}[${index}]`);
          continue;
        }
        const item = raw as ContentBlock | null | undefined;
        if (item == null || typeof item !== 'object') {
          continue;
        }
        if (isProxy(item)) {
          throw new UnsafeTokenMeasurementError({
            reason: 'content_proxy',
            path: `${path}[${index}]`,
          });
        }
        if (typeof item.type !== 'string') {
          numTokens += getBoundedStructuredTokenCount(item, countText);
          continue;
        }
        if (item.type === ContentTypes.TOOL_CALL || item.type === 'tool_use') {
          const inlineId =
            typeof item.id === 'string' ? item.id : item.tool_call?.id;
          if (typeof inlineId === 'string' && inlineId.length > 0) {
            representedToolCallIds.add(inlineId);
          }
        }
        if (item.type === ContentTypes.ERROR) {
          continue;
        }

        if (
          isAtomicToolContentBlock(item) &&
          (item.type === ContentTypes.IMAGE_URL ||
            item.type === 'image_url' ||
            item.type === 'image' ||
            item.type === 'computer_screenshot' ||
            item.type === 'input_image')
        ) {
          numTokens += Math.ceil(
            estimateImageBlockTokens(item, encoding) * IMAGE_TOKEN_SAFETY_MARGIN
          );
          continue;
        }

        if (
          isAtomicToolContentBlock(item) &&
          (item.type === 'document' ||
            item.type === 'file' ||
            item.type === ContentTypes.IMAGE_FILE)
        ) {
          numTokens += Math.ceil(
            estimateDocumentBlockTokens(item, encoding, countText) *
              IMAGE_TOKEN_SAFETY_MARGIN
          );
          continue;
        }

        if (isAtomicToolContentBlock(item) && item.type === 'media') {
          const staticMediaTokens = estimateStaticMediaBlockTokens(
            item,
            encoding,
            countText
          );
          if (staticMediaTokens != null) {
            numTokens += Math.ceil(
              staticMediaTokens * IMAGE_TOKEN_SAFETY_MARGIN
            );
            continue;
          }
        }

        if (isAtomicToolContentBlock(item) && isTimedMediaType(item.type)) {
          const timedMediaTokens = estimateTimedMediaBlockTokens(item);
          if (timedMediaTokens > 0) {
            numTokens += Math.ceil(
              timedMediaTokens * IMAGE_TOKEN_SAFETY_MARGIN
            );
            continue;
          }
        }

        if (item.type === ContentTypes.TOOL_CALL && item.tool_call != null) {
          const toolName = item.tool_call.name;
          if (typeof toolName === 'string' && toolName.length > 0) {
            numTokens += countText(toolName);
          }
          const args = item.tool_call.args;
          if (args != null) {
            numTokens +=
              typeof args === 'string'
                ? countText(args)
                : getBoundedStructuredTokenCount(args, countText);
          }
          const output = item.tool_call.output;
          if (output != null) {
            processValue(output, `${path}[${index}].tool_call.output`);
          }
          continue;
        }

        const nestedValue = item[item.type];
        if (nestedValue == null) {
          numTokens += getBoundedStructuredTokenCount(item, countText);
          continue;
        }

        processValue(nestedValue, `${path}[${index}].${item.type}`);
      }
    } else if (typeof value === 'string') {
      numTokens += countText(value);
    } else if (typeof value === 'number') {
      numTokens += countText(value.toString());
    } else if (typeof value === 'boolean') {
      numTokens += countText(value.toString());
    } else if (value != null && typeof value === 'object') {
      numTokens += getBoundedStructuredTokenCount(value, countText);
    }
  };

  const rawAdditionalKwargs = message.additional_kwargs as unknown;
  const additionalKwargs =
    rawAdditionalKwargs != null && typeof rawAdditionalKwargs === 'object'
      ? rawAdditionalKwargs
      : undefined;
  if (additionalKwargs != null && isProxy(additionalKwargs)) {
    throw new UnsafeTokenMeasurementError({
      reason: 'metadata_proxy',
      path: 'additional_kwargs',
    });
  }
  let additionalType: PropertyDescriptor | undefined;
  try {
    additionalType =
      additionalKwargs != null
        ? Object.getOwnPropertyDescriptor(additionalKwargs, 'type')
        : undefined;
  } catch {
    throw new UnsafeTokenMeasurementError({
      reason: 'metadata_accessor',
      path: 'additional_kwargs.type',
    });
  }
  if (additionalType != null && !('value' in additionalType)) {
    throw new UnsafeTokenMeasurementError({
      reason: 'metadata_accessor',
      path: 'additional_kwargs.type',
    });
  }

  let numTokens = tokensPerMessage;
  const messageType = message.getType();
  const isComputerCallOutput =
    messageType === 'tool' &&
    additionalType?.value === 'computer_call_output' &&
    isComputerCallOutputContent(message.content);
  if (isComputerCallOutput && typeof message.content === 'string') {
    numTokens += Math.ceil(
      estimateImageBlockTokens(
        {
          type: 'computer_screenshot',
          image_url: message.content,
        },
        encoding
      ) * IMAGE_TOKEN_SAFETY_MARGIN
    );
  } else {
    processValue(message.content, 'content');
  }
  ensureSafeTokenMeasurement(numTokens, 'content');
  const messageRole = (message as BaseMessage & { role?: unknown }).role;
  if (messageType === 'ai' || messageRole === 'assistant') {
    const toolCalls = (message as AIMessage).tool_calls ?? [];
    if (isProxy(toolCalls)) {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_proxy',
        path: 'tool_calls',
      });
    }
    for (const toolCall of toolCalls) {
      if (isProxy(toolCall)) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_proxy',
          path: 'tool_calls',
        });
      }
      if (
        typeof toolCall.id === 'string' &&
        representedToolCallIds.has(toolCall.id)
      ) {
        continue;
      }
      if (typeof toolCall.name === 'string' && toolCall.name.length > 0) {
        numTokens += countText(toolCall.name);
      }
      const args: unknown = toolCall.args;
      if (args != null) {
        numTokens +=
          typeof args === 'string'
            ? countText(args)
            : getBoundedStructuredTokenCount(args, countText);
      }
    }
    const rawCalls =
      toolCalls.length > 0 || additionalKwargs == null
        ? undefined
        : readTokenMetadataProperty(
          additionalKwargs,
          'tool_calls',
          'additional_kwargs.tool_calls'
        );
    if (rawCalls != null) {
      if (isProxy(rawCalls)) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_proxy',
          path: 'additional_kwargs.tool_calls',
        });
      }
      if (!Array.isArray(rawCalls)) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_accessor',
          path: 'additional_kwargs.tool_calls',
        });
      }
      if (rawCalls.length > MAX_RAW_TOOL_CALLS) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_limit',
          path: 'additional_kwargs.tool_calls',
        });
      }
      for (let i = 0; i < rawCalls.length; i++) {
        const rawCall = readTokenMetadataProperty(
          rawCalls,
          String(i),
          `additional_kwargs.tool_calls[${i}]`
        );
        if (
          rawCall == null ||
          typeof rawCall !== 'object' ||
          isProxy(rawCall)
        ) {
          throw new UnsafeTokenMeasurementError({
            reason: 'metadata_proxy',
            path: `additional_kwargs.tool_calls[${i}]`,
          });
        }
        const callPath = `additional_kwargs.tool_calls[${i}]`;
        if (hasUnsafeStructuredSerialization(rawCall)) {
          throw new UnsafeTokenMeasurementError({
            reason: 'metadata_accessor',
            path: callPath,
          });
        }
        const callId = readTokenMetadataProperty(
          rawCall,
          'id',
          `${callPath}.id`
        );
        const callType = readTokenMetadataProperty(
          rawCall,
          'type',
          `${callPath}.type`
        );
        const rawFunction = readTokenMetadataProperty(
          rawCall,
          'function',
          `${callPath}.function`
        );
        const rawCustom = readTokenMetadataProperty(
          rawCall,
          'custom',
          `${callPath}.custom`
        );
        let projectedFunction = rawFunction;
        if (rawFunction != null && typeof rawFunction === 'object') {
          if (hasUnsafeStructuredSerialization(rawFunction)) {
            throw new UnsafeTokenMeasurementError({
              reason: 'metadata_accessor',
              path: `${callPath}.function`,
            });
          }
          const name = readTokenMetadataProperty(
            rawFunction,
            'name',
            `${callPath}.function.name`
          );
          const args = readTokenMetadataProperty(
            rawFunction,
            'arguments',
            `${callPath}.function.arguments`
          );
          projectedFunction = { name, arguments: args };
        }
        let projectedCustom = rawCustom;
        if (rawCustom != null && typeof rawCustom === 'object') {
          if (hasUnsafeStructuredSerialization(rawCustom)) {
            throw new UnsafeTokenMeasurementError({
              reason: 'metadata_accessor',
              path: `${callPath}.custom`,
            });
          }
          const name = readTokenMetadataProperty(
            rawCustom,
            'name',
            `${callPath}.custom.name`
          );
          const input = readTokenMetadataProperty(
            rawCustom,
            'input',
            `${callPath}.custom.input`
          );
          projectedCustom = { name, input };
        }
        if (typeof callId === 'string' && representedToolCallIds.has(callId)) {
          continue;
        }
        numTokens += getBoundedStructuredTokenCount(
          {
            id: callId,
            type: callType,
            custom: projectedCustom,
            function: projectedFunction,
          },
          countText
        );
      }
    }
    let legacyFunctionCall: PropertyDescriptor | undefined;
    try {
      legacyFunctionCall =
        additionalKwargs != null
          ? Object.getOwnPropertyDescriptor(additionalKwargs, 'function_call')
          : undefined;
    } catch {
      throw new UnsafeTokenMeasurementError({
        reason: 'metadata_accessor',
        path: 'additional_kwargs.function_call',
      });
    }
    if (legacyFunctionCall != null) {
      if (!('value' in legacyFunctionCall)) {
        throw new UnsafeTokenMeasurementError({
          reason: 'metadata_accessor',
          path: 'additional_kwargs.function_call',
        });
      }
      if (legacyFunctionCall.value != null) {
        numTokens += getBoundedStructuredTokenCount(
          legacyFunctionCall.value,
          countText
        );
      }
    }
  }
  return ensureSafeTokenMeasurement(numTokens, 'message');
}

/**
 * Largest-remainder apportionment: scales each count by `multiplier` and
 * distributes the rounding remainder so the results sum exactly to
 * `targetTotal`. Keeps per-item breakdowns reconciled with an aggregate
 * computed as a single rounded product of the summed raw counts.
 */
export function apportionTokenCounts(
  rawCounts: Record<string, number>,
  multiplier: number,
  targetTotal: number
): Record<string, number> {
  const result: Record<string, number> = Object.create(null);
  const remainders: Array<{ name: string; remainder: number }> = [];
  let floorSum = 0;
  for (const [name, rawCount] of Object.entries(rawCounts)) {
    const scaled = rawCount * multiplier;
    const floored = Math.floor(scaled);
    result[name] = floored;
    floorSum += floored;
    remainders.push({ name, remainder: scaled - floored });
  }
  let leftover = targetTotal - floorSum;
  if (leftover <= 0 || remainders.length === 0) {
    return result;
  }
  remainders.sort((a, b) => b.remainder - a.remainder);
  for (let i = 0; leftover > 0; i = (i + 1) % remainders.length) {
    result[remainders[i].name] += 1;
    leftover--;
  }
  return result;
}

/**
 * Anthropic's API consistently reports ~10% more tokens than the local
 * claude tokenizer due to internal message framing and content encoding.
 * Verified empirically across content types via the count_tokens endpoint.
 */
const CLAUDE_TOKEN_CORRECTION = 1.1;

const tokenCounterEncodings = new WeakMap<
  (message: BaseMessage) => number,
  EncodingName
>();

/**
 * Encoding a counter measures in, for counters built here.
 *
 * `undefined` for a counter the host supplied itself: unknown, not wrong. A
 * caller that needs a count in a specific encoding can therefore tell "counts
 * in the encoding I need" from "counts in a different one" without treating
 * every foreign counter as suspect.
 */
export function encodingOfTokenCounter(
  tokenCounter: (message: BaseMessage) => number
): EncodingName | undefined {
  return tokenCounterEncodings.get(tokenCounter);
}

/**
 * Creates a token counter function using the specified encoding.
 * Lazily loads the encoding data on first use via dynamic import.
 */
export const createTokenCounter = async (
  encoding: EncodingName = 'o200k_base'
): Promise<(message: BaseMessage) => number> => {
  const tok = await getTokenizer(encoding);
  const countTokens = (text: string): number => tok.count(text);
  const isClaude = encoding === 'claude';
  const counter = markTokenCounterCacheCompatible(
    (message: BaseMessage): number => {
      const count = getTokenCountForMessage(message, countTokens, encoding);
      const correctedCount = isClaude
        ? Math.ceil(count * CLAUDE_TOKEN_CORRECTION)
        : count;
      return ensureSafeTokenMeasurement(correctedCount, 'message');
    }
  );
  tokenCounterEncodings.set(counter, encoding);
  return counter;
};

/** Utility to manage the token encoder lifecycle explicitly. */
export const TokenEncoderManager = {
  async initialize(): Promise<void> {
    // No-op: ai-tokenizer is synchronously initialized from bundled data.
  },

  reset(): void {
    for (const key of Object.keys(tokenizers)) {
      delete tokenizers[key as EncodingName];
    }
  },

  isInitialized(): boolean {
    return Object.keys(tokenizers).length > 0;
  },
};

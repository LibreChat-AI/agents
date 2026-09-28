import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';

/** Payloads shorter than this cost less to export than a descriptor saves. */
const MIN_INLINE_MEDIA_LENGTH = 4096;
const MIN_INLINE_MEDIA_BYTES = (MIN_INLINE_MEDIA_LENGTH * 3) / 4;
const BASE64_SAMPLE_LENGTH = 256;
const BASE64_BODY_PATTERN = /^[A-Za-z0-9+/]+$/;
const BASE64_TAIL_PATTERN = /^[A-Za-z0-9+/]*={0,2}$/;
const DATA_URI_PATTERN = /^data:([^;,]{1,127});base64,/;
const MIME_TYPE_KEYS = ['mimeType', 'mime_type', 'media_type', 'mediaType'];
const INLINE_MEDIA_ATTRIBUTES = [
  LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
  LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
];

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

type JsonObject = { [key: string]: JsonValue };

type SerializedBuffer = { type: 'Buffer'; data: number[] };

export type LangfuseInlineMediaPolicy = {
  /** Also omit `data:` URIs. False while the Langfuse SDK uploads them as media. */
  omitDataUris: boolean;
};

function isJsonObject(value: JsonValue): value is JsonObject {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isSerializedBuffer(value: JsonValue): value is SerializedBuffer {
  return (
    isJsonObject(value) &&
    value.type === 'Buffer' &&
    Array.isArray(value.data) &&
    value.data.length >= MIN_INLINE_MEDIA_BYTES &&
    typeof value.data[0] === 'number'
  );
}

/**
 * Samples the head and tail instead of testing the whole string: a full scan
 * of a multi-megabyte payload costs ~100ms per span on the export path.
 */
function isBase64Payload(value: string): boolean {
  return (
    BASE64_BODY_PATTERN.test(value.slice(0, BASE64_SAMPLE_LENGTH)) &&
    BASE64_TAIL_PATTERN.test(value.slice(-BASE64_SAMPLE_LENGTH))
  );
}

function getBase64ByteLength(value: string, payloadLength: number): number {
  let padding = 0;
  if (value.endsWith('==')) {
    padding = 2;
  } else if (value.endsWith('=')) {
    padding = 1;
  }
  return Math.floor((payloadLength * 3) / 4) - padding;
}

function findMimeType(parent?: JsonObject): string | undefined {
  if (parent == null) {
    return undefined;
  }
  for (const key of MIME_TYPE_KEYS) {
    const mimeType = parent[key];
    if (typeof mimeType === 'string' && mimeType !== '') {
      return mimeType;
    }
  }
  return undefined;
}

function describeOmittedMedia(bytes: number, mimeType?: string): string {
  return `[inline media omitted from trace: ${mimeType ?? 'unknown type'}, ${bytes} bytes]`;
}

function describeInlineString(
  value: string,
  policy: LangfuseInlineMediaPolicy,
  parent?: JsonObject
): string | undefined {
  if (value.length < MIN_INLINE_MEDIA_LENGTH) {
    return undefined;
  }

  const dataUri = DATA_URI_PATTERN.exec(value);
  if (dataUri != null) {
    return policy.omitDataUris
      ? describeOmittedMedia(
        getBase64ByteLength(value, value.length - dataUri[0].length),
        dataUri[1]
      )
      : undefined;
  }

  return isBase64Payload(value)
    ? describeOmittedMedia(
      getBase64ByteLength(value, value.length),
      findMimeType(parent)
    )
    : undefined;
}

function describeInlineValue(
  value: JsonValue,
  policy: LangfuseInlineMediaPolicy,
  parent?: JsonObject
): string | undefined {
  if (typeof value === 'string') {
    return describeInlineString(value, policy, parent);
  }
  if (isSerializedBuffer(value)) {
    return describeOmittedMedia(value.data.length, findMimeType(parent));
  }
  return undefined;
}

/**
 * Replaces inline media inside a freshly parsed JSON value. Mutates in place:
 * the value is a private parse of the span attribute, and copying a tree that
 * holds tens of megabytes would double the cost this exists to remove.
 */
function omitInlineMediaInPlace(
  value: JsonValue,
  policy: LangfuseInlineMediaPolicy
): boolean {
  if (Array.isArray(value)) {
    let changed = false;
    for (let i = 0; i < value.length; i++) {
      const descriptor = describeInlineValue(value[i], policy);
      if (descriptor != null) {
        value[i] = descriptor;
        changed = true;
      } else if (omitInlineMediaInPlace(value[i], policy)) {
        changed = true;
      }
    }
    return changed;
  }

  if (!isJsonObject(value)) {
    return false;
  }

  let changed = false;
  for (const key in value) {
    const descriptor = describeInlineValue(value[key], policy, value);
    if (descriptor != null) {
      value[key] = descriptor;
      changed = true;
    } else if (omitInlineMediaInPlace(value[key], policy)) {
      changed = true;
    }
  }
  return changed;
}

function omitInlineMediaFromAttribute(
  value: string,
  policy: LangfuseInlineMediaPolicy
): string | undefined {
  if (value.length < MIN_INLINE_MEDIA_LENGTH) {
    return undefined;
  }
  if (value[0] !== '{' && value[0] !== '[') {
    return describeInlineString(value, policy);
  }

  let parsed: JsonValue;
  try {
    parsed = JSON.parse(value) as JsonValue;
  } catch {
    return undefined;
  }
  return omitInlineMediaInPlace(parsed, policy)
    ? JSON.stringify(parsed)
    : undefined;
}

/**
 * Replaces inline base64 media in a span's input and output with a short
 * type-and-size descriptor. Every span that serializes the conversation (the
 * agent, its model node, the prompt, and the generation) otherwise carries its
 * own full copy of each attached file.
 */
export function omitLangfuseSpanInlineMedia(
  span: ReadableSpan,
  policy: LangfuseInlineMediaPolicy
): void {
  for (const key of INLINE_MEDIA_ATTRIBUTES) {
    const value = span.attributes[key];
    if (typeof value !== 'string') {
      continue;
    }
    const next = omitInlineMediaFromAttribute(value, policy);
    if (next != null) {
      span.attributes[key] = next;
    }
  }
}

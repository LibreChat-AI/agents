import { isArrayBuffer, isDate, isProxy } from 'node:util/types';
import { ToolMessage, type BaseMessage } from '@langchain/core/messages';
import {
  HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE,
  truncateToolResultContent,
  sliceWithoutSplittingSurrogates,
} from './truncation';

type ToolContent = BaseMessage['content'];
type ToolContentBlock = Exclude<ToolContent, string>[number];
type TextToolContentBlock = ToolContentBlock & { type: 'text'; text: string };

const ATOMIC_CONTENT_TYPES = new Set([
  'audio',
  'computer_screenshot',
  'document',
  'file',
  'image',
  'image_file',
  'image_url',
  'input_audio',
  'input_image',
  'media',
  'resource',
  'resource_link',
  'video',
  'video_url',
]);
const MAX_TOOL_CONTENT_TYPE_CHARS = 256;
const MAX_PROVIDER_IMAGE_URL_CHARS = 16_384;
const MAX_PROVIDER_FILE_ID_CHARS = 4_096;

const PROVIDER_NATIVE_TOOL_RESULT_TYPES = new Set([
  'search_result',
  'server_tool_call_result',
  'tool_result',
  'toolResponse',
  'web_search_result',
  'web_search_tool_result',
]);

/**
 * Serializes values that providers render as JSON without throwing on values
 * such as bigint or circular diagnostic metadata.
 */
export function serializeStructuredValue(value: unknown): string {
  return serializeStructuredValueBounded(value, Number.MAX_SAFE_INTEGER)
    .content;
}

const SERIALIZATION_CHUNK_SIZE = 4_096;
const MAX_STRUCTURED_SERIALIZATION_DEPTH = 200;
const MAX_STRUCTURED_SERIALIZATION_WORK = 1_000_000;
const MAX_STRUCTURED_CHARACTER_WORK = 1_000_000;
const MAX_STRUCTURED_PROPERTY_CACHE_ENTRIES = 10_000;
const PROXY_VALUE_PLACEHOLDER = '[Proxy value omitted]';
const CHARACTER_WORK_PLACEHOLDER =
  '[truncated: character traversal limit exceeded]';

type StructuredWorkState = {
  remaining: number;
  characterRemaining: number;
  exceeded: boolean;
  failurePlaceholder?: string;
  propertyCache: WeakMap<object, Map<string, unknown>>;
  propertyCacheEntries: number;
};

function createStructuredWorkState(): StructuredWorkState {
  return {
    remaining: MAX_STRUCTURED_SERIALIZATION_WORK,
    characterRemaining: MAX_STRUCTURED_CHARACTER_WORK,
    exceeded: false,
    propertyCache: new WeakMap<object, Map<string, unknown>>(),
    propertyCacheEntries: 0,
  };
}

function consumeStructuredWork(state: StructuredWorkState): boolean {
  if (state.remaining <= 0) {
    state.exceeded = true;
    state.failurePlaceholder ??= '[Traversal limit exceeded]';
    return false;
  }
  state.remaining--;
  return true;
}

function consumeStructuredCharacterWork(state: StructuredWorkState): boolean {
  if (state.characterRemaining <= 0) {
    state.exceeded = true;
    state.failurePlaceholder ??= CHARACTER_WORK_PLACEHOLDER;
    return false;
  }
  state.characterRemaining--;
  return true;
}

function hasExceededStructuredWork(state: StructuredWorkState): boolean {
  return state.exceeded;
}

function conservativeStructuredLength(limit: number): number {
  return limit >= Number.MAX_SAFE_INTEGER ? Number.MAX_SAFE_INTEGER : limit + 1;
}

function jsonStringLength(
  value: string,
  limit: number,
  work: StructuredWorkState
): number {
  let length = 2;
  for (let i = 0; i < value.length; i++) {
    if (!consumeStructuredCharacterWork(work)) {
      return conservativeStructuredLength(limit);
    }
    const code = value.charCodeAt(i);
    if (
      code === 0x08 ||
      code === 0x09 ||
      code === 0x0a ||
      code === 0x0c ||
      code === 0x0d ||
      code === 0x22 ||
      code === 0x5c
    ) {
      length += 2;
    } else if (code < 0x20) {
      length += 6;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      if (!consumeStructuredCharacterWork(work)) {
        return conservativeStructuredLength(limit);
      }
      const next = value.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        length += 2;
        i++;
      } else {
        length += 6;
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      length += 6;
    } else {
      length++;
    }
    if (length > limit) {
      return conservativeStructuredLength(limit);
    }
  }
  return length;
}

type NativeDateRead = { matched: true; time: number } | { matched: false };

/**
 * `isArrayBuffer` reads the internal slot natively (no user code, cross-realm safe) and
 * costs nothing for the plain objects that make up almost all content; the getter's
 * throw path was the expensive part, so it only runs once the brand is already known.
 */
function readNativeArrayBufferByteLength(value: unknown): number | undefined {
  if (
    value == null ||
    typeof value !== 'object' ||
    NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER == null ||
    !isArrayBuffer(value)
  ) {
    return undefined;
  }
  try {
    return NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(value) as number;
  } catch {
    return undefined;
  }
}

function readNativeDate(value: unknown): NativeDateRead {
  if (value == null || typeof value !== 'object' || !isDate(value)) {
    return { matched: false };
  }
  try {
    return { matched: true, time: NATIVE_DATE_GET_TIME.call(value) };
  } catch {
    return { matched: false };
  }
}

function readNativeBoxedString(value: object): string | undefined {
  try {
    return String.prototype.valueOf.call(value);
  } catch {
    return undefined;
  }
}

function readNativeBoxedNumber(value: object): number | undefined {
  try {
    return Number.prototype.valueOf.call(value);
  } catch {
    return undefined;
  }
}

function readNativeBoxedBoolean(value: object): boolean | undefined {
  try {
    return Boolean.prototype.valueOf.call(value);
  } catch {
    return undefined;
  }
}

function estimateStructuredChars(
  value: unknown,
  limit = Number.MAX_SAFE_INTEGER,
  ancestors: object[] = [],
  work = createStructuredWorkState(),
  depth = 0
): number {
  if (hasExceededStructuredWork(work)) {
    return conservativeStructuredLength(limit);
  }
  if (typeof value === 'string') {
    return jsonStringLength(value, limit, work);
  }
  if (typeof value === 'bigint') {
    return jsonStringLength(value.toString(), limit, work);
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value === 0 ? 0 : value).length : 4;
  }
  if (typeof value === 'boolean' || value == null) {
    return String(value).length;
  }
  if (isProxy(value)) {
    work.exceeded = true;
    work.failurePlaceholder = PROXY_VALUE_PLACEHOLDER;
    return conservativeStructuredLength(limit);
  }
  if (typeof value === 'undefined' || typeof value === 'function') {
    return 4;
  }
  if (typeof value !== 'object') {
    return jsonStringLength(String(value), limit, work);
  }

  const valueIsArray = isArray(value);
  if (!valueIsArray) {
    const arrayBufferByteLength = readNativeArrayBufferByteLength(value);
    if (arrayBufferByteLength != null) {
      if (!isSafeNativeArrayBuffer(value as ArrayBuffer)) {
        work.exceeded = true;
        return conservativeStructuredLength(limit);
      }
      return Math.min(
        limit + 1,
        Math.ceil((arrayBufferByteLength * 4) / 3) + 32
      );
    }
    if (ArrayBuffer.isView(value)) {
      const view = readSafeNativeArrayBufferView(
        value,
        undefined,
        undefined,
        false
      );
      if (view == null) {
        work.exceeded = true;
        return conservativeStructuredLength(limit);
      }
      return Math.min(limit + 1, Math.ceil((view.byteLength * 4) / 3) + 32);
    }
    const date = readNativeDate(value);
    if (date.matched) {
      if (!isSafeNativeDate(value as Date)) {
        work.exceeded = true;
        return conservativeStructuredLength(limit);
      }
      return Number.isFinite(date.time)
        ? jsonStringLength(Date.prototype.toISOString.call(value), limit, work)
        : 4;
    }
    const boxedString = readNativeBoxedString(value);
    if (boxedString != null) {
      return jsonStringLength(boxedString, limit, work);
    }
    const boxedNumber = readNativeBoxedNumber(value);
    if (boxedNumber != null) {
      return Number.isFinite(boxedNumber)
        ? String(boxedNumber === 0 ? 0 : boxedNumber).length
        : 4;
    }
    const boxedBoolean = readNativeBoxedBoolean(value);
    if (boxedBoolean != null) {
      return String(boxedBoolean).length;
    }
    let prototype: object | null;
    try {
      prototype = Object.getPrototypeOf(value) as object | null;
    } catch {
      work.exceeded = true;
      return conservativeStructuredLength(limit);
    }
    if (
      isProxy(prototype) ||
      (prototype !== Object.prototype && prototype !== null)
    ) {
      work.exceeded = true;
      return conservativeStructuredLength(limit);
    }
  } else {
    let prototype: object | null;
    try {
      prototype = Object.getPrototypeOf(value) as object | null;
    } catch {
      work.exceeded = true;
      return conservativeStructuredLength(limit);
    }
    if (
      isProxy(prototype) ||
      (prototype !== Array.prototype && prototype !== null)
    ) {
      work.exceeded = true;
      return conservativeStructuredLength(limit);
    }
  }
  if (depth >= MAX_STRUCTURED_SERIALIZATION_DEPTH) {
    return jsonStringLength('[Max serialization depth]', limit, work);
  }
  if (ancestors.includes(value)) {
    return 12;
  }

  ancestors.push(value);
  let length = 2;
  try {
    if (valueIsArray) {
      const arrayLength = readStructuredArrayLength(value, work);
      if (arrayLength == null) {
        return conservativeStructuredLength(limit);
      }
      for (let i = 0; i < arrayLength && length <= limit; i++) {
        if (!consumeStructuredWork(work)) {
          return conservativeStructuredLength(limit);
        }
        if (i > 0) {
          length++;
        }
        length += estimateStructuredChars(
          readStructuredProperty(value, i, work),
          Math.max(0, limit - length),
          ancestors,
          work,
          depth + 1
        );
      }
    } else {
      let emitted = 0;
      for (const key in value) {
        if (!consumeStructuredWork(work)) {
          return conservativeStructuredLength(limit);
        }
        if (!Object.prototype.propertyIsEnumerable.call(value, key)) {
          continue;
        }
        if (length > limit) {
          break;
        }
        const nested = readStructuredProperty(
          value as Record<string, unknown>,
          key,
          work
        );
        if (
          typeof nested === 'undefined' ||
          typeof nested === 'function' ||
          typeof nested === 'symbol'
        ) {
          continue;
        }
        if (emitted > 0) {
          length++;
        }
        length += jsonStringLength(key, Math.max(0, limit - length), work) + 1;
        length += estimateStructuredChars(
          nested,
          Math.max(0, limit - length),
          ancestors,
          work,
          depth + 1
        );
        emitted++;
      }
    }
  } catch {
    work.exceeded = true;
    return conservativeStructuredLength(limit);
  } finally {
    ancestors.pop();
  }
  if (hasExceededStructuredWork(work)) {
    return conservativeStructuredLength(limit);
  }
  return Math.min(limit + 1, length);
}

type StructuredSerializationCollector = {
  totalChars: number;
  prefix: SegmentedStringBuffer;
  suffix: SegmentedStringBuffer;
  prefixLimit: number;
  suffixLimit: number;
  work: StructuredWorkState;
};

type SegmentedStringBuffer = {
  segments: string[];
  head: number;
  pending: string[];
  pendingChars: number;
  length: number;
};

export type BoundedStructuredSerialization = {
  /** Provider-facing head/tail preview, bounded by `maxChars`. */
  content: string;
  /** Code-point-aligned serialized prefix up to the traversal-work ceiling. */
  prefix: string;
  /** Exact length, or `Number.MAX_SAFE_INTEGER` when traversal was capped. */
  originalChars: number;
  truncated: boolean;
};

function normalizeCharLimit(limit: number): number {
  return Number.isFinite(limit)
    ? Math.max(0, Math.floor(limit))
    : Number.MAX_SAFE_INTEGER;
}

function createSegmentedStringBuffer(): SegmentedStringBuffer {
  return {
    segments: [],
    head: 0,
    pending: [],
    pendingChars: 0,
    length: 0,
  };
}

function flushSegmentedStringBuffer(buffer: SegmentedStringBuffer): void {
  if (buffer.pendingChars === 0) {
    return;
  }
  buffer.segments.push(buffer.pending.join(''));
  buffer.pending = [];
  buffer.pendingChars = 0;
}

function appendToSegmentedStringBuffer(
  buffer: SegmentedStringBuffer,
  chunk: string
): void {
  if (chunk.length === 0) {
    return;
  }
  buffer.pending.push(chunk);
  buffer.pendingChars += chunk.length;
  buffer.length += chunk.length;
  if (buffer.pendingChars >= SERIALIZATION_CHUNK_SIZE) {
    flushSegmentedStringBuffer(buffer);
  }
}

function trimSegmentedStringBufferStart(
  buffer: SegmentedStringBuffer,
  chars: number
): void {
  let remaining = Math.min(chars, buffer.length);
  if (remaining <= 0) {
    return;
  }
  flushSegmentedStringBuffer(buffer);
  while (remaining > 0 && buffer.head < buffer.segments.length) {
    const segment = buffer.segments[buffer.head];
    if (segment.length <= remaining) {
      remaining -= segment.length;
      buffer.length -= segment.length;
      buffer.head++;
    } else {
      buffer.segments[buffer.head] = segment.slice(remaining);
      buffer.length -= remaining;
      remaining = 0;
    }
  }
  if (buffer.head >= 1_024 && buffer.head * 2 >= buffer.segments.length) {
    buffer.segments = buffer.segments.slice(buffer.head);
    buffer.head = 0;
  }
}

function materializeSegmentedStringBuffer(
  buffer: SegmentedStringBuffer
): string {
  const segments = buffer.segments.slice(buffer.head);
  if (buffer.pendingChars > 0) {
    segments.push(buffer.pending.join(''));
  }
  return segments.join('');
}

/**
 * Retains exact code-unit prefixes internally. Trimming surrogate edges here
 * would let later chunks fill the gap and corrupt the prefix; trim on exposure.
 */
function appendSerializedChunk(
  collector: StructuredSerializationCollector,
  chunk: string
): void {
  if (chunk.length === 0) {
    return;
  }

  collector.totalChars += chunk.length;

  const prefixRemaining = collector.prefixLimit - collector.prefix.length;
  if (prefixRemaining > 0) {
    appendToSegmentedStringBuffer(
      collector.prefix,
      chunk.slice(0, prefixRemaining)
    );
  }

  if (collector.suffixLimit <= 0) {
    return;
  }
  appendToSegmentedStringBuffer(collector.suffix, chunk);
  if (
    collector.suffix.length >=
    collector.suffixLimit + SERIALIZATION_CHUNK_SIZE
  ) {
    trimSegmentedStringBufferStart(
      collector.suffix,
      collector.suffix.length - collector.suffixLimit
    );
  }
}

function appendJsonSafeRange(
  collector: StructuredSerializationCollector,
  value: string,
  start: number,
  end: number
): void {
  let offset = start;
  while (offset < end) {
    let chunkEnd = Math.min(end, offset + SERIALIZATION_CHUNK_SIZE);
    if (
      chunkEnd < end &&
      chunkEnd > offset &&
      value.charCodeAt(chunkEnd - 1) >= 0xd800 &&
      value.charCodeAt(chunkEnd - 1) <= 0xdbff &&
      value.charCodeAt(chunkEnd) >= 0xdc00 &&
      value.charCodeAt(chunkEnd) <= 0xdfff
    ) {
      chunkEnd--;
    }
    appendSerializedChunk(collector, value.slice(offset, chunkEnd));
    offset = chunkEnd;
  }
}

/**
 * Emits one JSON string incrementally. Calling JSON.stringify on the complete
 * string would allocate an escaped copy before the output cap could apply.
 */
function appendJsonString(
  collector: StructuredSerializationCollector,
  value: string
): void {
  if (collector.work.exceeded) {
    appendSerializedChunk(
      collector,
      `"${collector.work.failurePlaceholder ?? CHARACTER_WORK_PLACEHOLDER}"`
    );
    return;
  }
  appendSerializedChunk(collector, '"');
  let safeStart = 0;

  for (let i = 0; i < value.length; i++) {
    if (!consumeStructuredCharacterWork(collector.work)) {
      appendJsonSafeRange(collector, value, safeStart, i);
      appendSerializedChunk(collector, `${CHARACTER_WORK_PLACEHOLDER}"`);
      return;
    }
    const code = value.charCodeAt(i);
    let escaped: string | undefined;
    switch (code) {
    case 0x08:
      escaped = '\\b';
      break;
    case 0x09:
      escaped = '\\t';
      break;
    case 0x0a:
      escaped = '\\n';
      break;
    case 0x0c:
      escaped = '\\f';
      break;
    case 0x0d:
      escaped = '\\r';
      break;
    case 0x22:
      escaped = '\\"';
      break;
    case 0x5c:
      escaped = '\\\\';
      break;
    default:
      if (code < 0x20) {
        escaped = `\\u${code.toString(16).padStart(4, '0')}`;
      } else if (code >= 0xd800 && code <= 0xdbff) {
        if (!consumeStructuredCharacterWork(collector.work)) {
          appendJsonSafeRange(collector, value, safeStart, i);
          appendSerializedChunk(collector, `${CHARACTER_WORK_PLACEHOLDER}"`);
          return;
        }
        const next = value.charCodeAt(i + 1);
        if (next >= 0xdc00 && next <= 0xdfff) {
          i++;
        } else {
          escaped = `\\u${code.toString(16).padStart(4, '0')}`;
        }
      } else if (code >= 0xdc00 && code <= 0xdfff) {
        escaped = `\\u${code.toString(16).padStart(4, '0')}`;
      }
    }

    if (escaped != null) {
      appendJsonSafeRange(collector, value, safeStart, i);
      appendSerializedChunk(collector, escaped);
      safeStart = i + 1;
    }
  }
  appendJsonSafeRange(collector, value, safeStart, value.length);
  appendSerializedChunk(collector, '"');
}

function readStructuredProperty(
  value: Record<string, unknown> | unknown[],
  key: string | number,
  work: StructuredWorkState
): unknown {
  const normalizedKey = String(key);
  // Callers still consume traversal work for every occurrence. This bounded
  // cache only avoids re-reading the same descriptor through shared references.
  const cachedProperties = work.propertyCache.get(value);
  if (cachedProperties != null && cachedProperties.has(normalizedKey)) {
    return cachedProperties.get(normalizedKey);
  }

  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, normalizedKey);
    let propertyValue: unknown;
    if (descriptor == null) {
      propertyValue = undefined;
    } else if ('value' in descriptor) {
      propertyValue = descriptor.value;
    } else {
      propertyValue = '[Property accessor omitted]';
    }
    if (work.propertyCacheEntries < MAX_STRUCTURED_PROPERTY_CACHE_ENTRIES) {
      const properties = cachedProperties ?? new Map<string, unknown>();
      if (cachedProperties == null) {
        work.propertyCache.set(value, properties);
      }
      properties.set(normalizedKey, propertyValue);
      work.propertyCacheEntries++;
    }
    return propertyValue;
  } catch {
    work.exceeded = true;
    work.remaining = 0;
    return '[Unreadable property omitted]';
  }
}

function readStructuredArrayLength(
  value: unknown[],
  work: StructuredWorkState
): number | undefined {
  const length = readStructuredProperty(value, 'length', work);
  if (
    typeof length === 'number' &&
    Number.isSafeInteger(length) &&
    length >= 0
  ) {
    return length;
  }
  work.exceeded = true;
  work.remaining = 0;
  return undefined;
}

/**
 * Provider-neutral JSON writer which deliberately does not invoke `toJSON` or
 * property accessors. Native `JSON.stringify` can invoke both before a replacer
 * can bound their output.
 */
function appendStructuredValue(
  collector: StructuredSerializationCollector,
  value: unknown,
  ancestors: Set<object>,
  depth: number,
  position: 'top' | 'array' | 'object'
): boolean {
  if (collector.work.exceeded) {
    appendSerializedChunk(
      collector,
      `"${collector.work.failurePlaceholder ?? CHARACTER_WORK_PLACEHOLDER}"`
    );
    return true;
  }
  if (typeof value === 'string') {
    appendJsonString(collector, value);
    return true;
  }
  if (typeof value === 'bigint') {
    appendJsonString(collector, value.toString());
    return true;
  }
  if (typeof value === 'number') {
    appendSerializedChunk(
      collector,
      Number.isFinite(value) ? String(value === 0 ? 0 : value) : 'null'
    );
    return true;
  }
  if (typeof value === 'boolean' || value === null) {
    appendSerializedChunk(collector, String(value));
    return true;
  }
  if (isProxy(value)) {
    appendSerializedChunk(collector, `"${PROXY_VALUE_PLACEHOLDER}"`);
    collector.work.exceeded = true;
    collector.work.failurePlaceholder = PROXY_VALUE_PLACEHOLDER;
    return true;
  }
  if (
    typeof value === 'undefined' ||
    typeof value === 'function' ||
    typeof value === 'symbol'
  ) {
    if (position === 'object') {
      return false;
    }
    if (position === 'array') {
      appendSerializedChunk(collector, 'null');
    } else if (typeof value === 'undefined') {
      appendSerializedChunk(collector, 'undefined');
    } else {
      appendJsonString(
        collector,
        typeof value === 'function' ? '[Function]' : String(value)
      );
    }
    return true;
  }
  const valueIsArray = isArray(value);
  if (!valueIsArray) {
    const arrayBufferByteLength = readNativeArrayBufferByteLength(value);
    if (arrayBufferByteLength != null) {
      if (!isSafeNativeArrayBuffer(value as ArrayBuffer)) {
        appendSerializedChunk(collector, '"[Unsafe ArrayBuffer omitted]"');
        collector.work.exceeded = true;
        return true;
      }
      appendJsonString(
        collector,
        `[ArrayBuffer: ${arrayBufferByteLength} bytes]`
      );
      return true;
    }
    if (ArrayBuffer.isView(value)) {
      const view = readSafeNativeArrayBufferView(
        value,
        undefined,
        undefined,
        false
      );
      if (view == null) {
        appendSerializedChunk(collector, '"[Unsafe ArrayBufferView omitted]"');
        collector.work.exceeded = true;
        return true;
      }
      appendJsonString(collector, `[${view.name}: ${view.byteLength} bytes]`);
      return true;
    }
    const date = readNativeDate(value);
    if (date.matched) {
      if (!isSafeNativeDate(value as Date)) {
        appendSerializedChunk(collector, '"[Unsafe Date omitted]"');
        collector.work.exceeded = true;
      } else if (!Number.isFinite(date.time)) {
        appendSerializedChunk(collector, 'null');
      } else {
        appendJsonString(collector, Date.prototype.toISOString.call(value));
      }
      return true;
    }
    const boxedString = readNativeBoxedString(value);
    if (boxedString != null) {
      appendJsonString(collector, boxedString);
      return true;
    }
    const boxedNumber = readNativeBoxedNumber(value);
    if (boxedNumber != null) {
      appendSerializedChunk(
        collector,
        Number.isFinite(boxedNumber)
          ? String(boxedNumber === 0 ? 0 : boxedNumber)
          : 'null'
      );
      return true;
    }
    const boxedBoolean = readNativeBoxedBoolean(value);
    if (boxedBoolean != null) {
      appendSerializedChunk(collector, String(boxedBoolean));
      return true;
    }
    let prototype: object | null;
    try {
      prototype = Object.getPrototypeOf(value) as object | null;
    } catch {
      appendSerializedChunk(collector, '"[Unreadable object omitted]"');
      collector.work.exceeded = true;
      return true;
    }
    if (
      isProxy(prototype) ||
      (prototype !== Object.prototype && prototype !== null)
    ) {
      appendSerializedChunk(collector, '"[Unsupported object omitted]"');
      collector.work.exceeded = true;
      return true;
    }
  } else {
    let prototype: object | null;
    try {
      prototype = Object.getPrototypeOf(value) as object | null;
    } catch {
      appendSerializedChunk(collector, '"[Unreadable array omitted]"');
      collector.work.exceeded = true;
      return true;
    }
    if (
      isProxy(prototype) ||
      (prototype !== Array.prototype && prototype !== null)
    ) {
      appendSerializedChunk(collector, '"[Unsafe array omitted]"');
      collector.work.exceeded = true;
      return true;
    }
  }
  if (depth >= MAX_STRUCTURED_SERIALIZATION_DEPTH) {
    appendJsonString(collector, '[Max serialization depth]');
    return true;
  }
  if (ancestors.has(value)) {
    appendJsonString(collector, '[Circular]');
    return true;
  }

  ancestors.add(value);
  try {
    if (valueIsArray) {
      const arrayLength = readStructuredArrayLength(value, collector.work);
      if (arrayLength == null) {
        appendJsonString(collector, '[Unreadable array omitted]');
        return true;
      }
      appendSerializedChunk(collector, '[');
      for (let i = 0; i < arrayLength; i++) {
        if (!consumeStructuredWork(collector.work)) {
          if (i > 0) {
            appendSerializedChunk(collector, ',');
          }
          appendJsonString(
            collector,
            `[Traversal limit exceeded: ${arrayLength - i} array entries omitted]`
          );
          break;
        }
        if (i > 0) {
          appendSerializedChunk(collector, ',');
        }
        appendStructuredValue(
          collector,
          readStructuredProperty(value, i, collector.work),
          ancestors,
          depth + 1,
          'array'
        );
      }
      appendSerializedChunk(collector, ']');
      return true;
    }

    appendSerializedChunk(collector, '{');
    let emitted = 0;
    // `Object.keys()` allocates an array proportional to the complete object
    // before either output limit applies. Filter a streaming `for…in` walk to
    // retain JSON's own-enumerable-key semantics without that extra array.
    for (const key in value) {
      if (!consumeStructuredWork(collector.work)) {
        if (emitted > 0) {
          appendSerializedChunk(collector, ',');
        }
        appendSerializedChunk(
          collector,
          '"_truncated":"[Traversal limit exceeded]"'
        );
        break;
      }
      if (!Object.prototype.propertyIsEnumerable.call(value, key)) {
        continue;
      }
      const nested = readStructuredProperty(
        value as Record<string, unknown>,
        key,
        collector.work
      );
      if (
        typeof nested === 'undefined' ||
        typeof nested === 'function' ||
        typeof nested === 'symbol'
      ) {
        continue;
      }
      if (emitted > 0) {
        appendSerializedChunk(collector, ',');
      }
      appendJsonString(collector, key);
      appendSerializedChunk(collector, ':');
      appendStructuredValue(collector, nested, ancestors, depth + 1, 'object');
      emitted++;
    }
    appendSerializedChunk(collector, '}');
    return true;
  } finally {
    ancestors.delete(value);
  }
}

function formatBoundedStructuredContent(
  collector: StructuredSerializationCollector,
  maxChars: number,
  prefix: string,
  suffix: string
): string {
  if (collector.totalChars <= maxChars) {
    return sliceWithoutSplittingSurrogates(prefix, 0, collector.totalChars);
  }

  const indicator = collector.work.exceeded
    ? '\n\n… [truncated: safe traversal limit exceeded] …\n\n'
    : `\n\n… [truncated: ${collector.totalChars} chars exceeded ` +
      `${maxChars} limit] …\n\n`;
  const available = maxChars - indicator.length;
  if (available <= 0) {
    return sliceWithoutSplittingSurrogates(prefix, 0, maxChars);
  }
  if (available < 200) {
    return (
      sliceWithoutSplittingSurrogates(prefix, 0, available) +
      indicator.trimEnd()
    );
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;
  let headEnd = headSize;
  const headNewline = prefix.lastIndexOf('\n', headSize);
  if (headNewline > headSize - 200 && headNewline > 0) {
    headEnd = headNewline;
  }

  let tailStart = Math.max(0, suffix.length - tailSize);
  const tailNewline = suffix.indexOf('\n', tailStart);
  if (tailNewline > 0 && tailNewline < tailStart + 200) {
    tailStart = tailNewline + 1;
  }

  return (
    sliceWithoutSplittingSurrogates(prefix, 0, headEnd) +
    indicator +
    sliceWithoutSplittingSurrogates(suffix, tailStart)
  );
}

/**
 * Serializes structured output while retaining at most the requested provider
 * preview and registry prefix. The traversal never calls `toJSON` and never
 * materializes the complete serialized string when it exceeds those limits.
 *
 * `prefixChars` is useful for the tool-output registry: it receives the exact
 * serialized prefix up to `registry.perOutputLimit`, while `content` retains
 * the provider-facing head/tail truncation behavior at `maxChars`. Property
 * accessors are represented by a fixed placeholder instead of being invoked.
 */
export function serializeStructuredValueBounded(
  value: unknown,
  maxChars: number,
  prefixChars = 0,
  work = createStructuredWorkState()
): BoundedStructuredSerialization {
  const normalizedMaxChars = normalizeCharLimit(maxChars);
  const normalizedPrefixChars = normalizeCharLimit(prefixChars);
  const prefixLimit = Math.max(normalizedMaxChars, normalizedPrefixChars);
  const collector: StructuredSerializationCollector = {
    totalChars: 0,
    prefix: createSegmentedStringBuffer(),
    suffix: createSegmentedStringBuffer(),
    prefixLimit,
    suffixLimit:
      normalizedMaxChars === Number.MAX_SAFE_INTEGER ? 0 : normalizedMaxChars,
    work,
  };

  try {
    appendStructuredValue(collector, value, new Set<object>(), 0, 'top');
  } catch {
    collector.totalChars = 0;
    collector.prefix = createSegmentedStringBuffer();
    collector.suffix = createSegmentedStringBuffer();
    collector.work.exceeded = true;
    appendSerializedChunk(collector, '"[Unserializable structured value]"');
  }

  const prefix = materializeSegmentedStringBuffer(collector.prefix);
  const suffix = materializeSegmentedStringBuffer(collector.suffix);
  const originalChars = collector.work.exceeded
    ? Number.MAX_SAFE_INTEGER
    : collector.totalChars;
  return {
    content: formatBoundedStructuredContent(
      collector,
      normalizedMaxChars,
      prefix,
      suffix
    ),
    prefix: sliceWithoutSplittingSurrogates(prefix, 0, normalizedPrefixChars),
    originalChars,
    truncated:
      collector.work.exceeded || collector.totalChars > normalizedMaxChars,
  };
}

function serializeStructuredValueWithinLimit(
  value: unknown,
  maxChars: number,
  work = createStructuredWorkState()
): string {
  return serializeStructuredValueBounded(value, maxChars, 0, work).content;
}

function isTextBlockUnchecked(value: unknown): value is TextToolContentBlock {
  if (!isRecord(value) || isProxy(value)) {
    return false;
  }
  try {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      return false;
    }
    const keys = Object.keys(value);
    if (keys.length !== 2 || !keys.includes('type') || !keys.includes('text')) {
      return false;
    }
    const type = readOwnEnumerableDataProperty(value, 'type');
    const text = readOwnEnumerableDataProperty(value, 'text');
    return (
      type.found &&
      type.value === 'text' &&
      text.found &&
      typeof text.value === 'string'
    );
  } catch {
    return false;
  }
}

function isArray(value: unknown): value is unknown[] {
  try {
    return Array.isArray(value);
  } catch {
    return false;
  }
}

function getDenseTextBlockArrayLength(content: unknown[]): number | undefined {
  if (isProxy(content)) {
    return undefined;
  }
  const validation = createToolBlockClassificationContext();
  const containerWork = { value: MAX_ATOMIC_VALIDATION_WORK };
  let contentLength: number;
  try {
    const prototype = Object.getPrototypeOf(content);
    if (
      (prototype !== Array.prototype && prototype !== null) ||
      hasPotentiallyCallableToJSON(
        content,
        containerWork,
        validation.totalRemaining
      )
    ) {
      return undefined;
    }
    contentLength = content.length;
  } catch {
    return undefined;
  }
  if (
    contentLength === 0 ||
    contentLength > MAX_STRUCTURED_SERIALIZATION_WORK
  ) {
    return undefined;
  }

  let length = contentLength - 1;
  try {
    for (let i = 0; i < contentLength; i++) {
      const descriptor = Object.getOwnPropertyDescriptor(content, String(i));
      if (descriptor == null || !('value' in descriptor)) {
        return undefined;
      }
      const block = descriptor.value;
      if (
        classifyToolContentBlock(block, validation) !== TOOL_BLOCK_TEXT ||
        validation.totalRemaining.value <= 0
      ) {
        return undefined;
      }
      const text = readOwnEnumerableDataProperty(block, 'text');
      length += typeof text.value === 'string' ? text.value.length : 0;
      if (length >= Number.MAX_SAFE_INTEGER) {
        return Number.MAX_SAFE_INTEGER;
      }
    }
  } catch {
    return undefined;
  }
  return length;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null;
}

type OwnDataProperty =
  | { found: true; value: unknown }
  | { found: false; value?: never };

function readOwnEnumerableDataProperty(
  record: Record<string, unknown>,
  key: string
): OwnDataProperty {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(record, key);
    if (
      descriptor == null ||
      descriptor.enumerable !== true ||
      !('value' in descriptor)
    ) {
      return { found: false };
    }
    return { found: true, value: descriptor.value };
  } catch {
    return { found: false };
  }
}

function hasString(record: Record<string, unknown>, key: string): boolean {
  const property = readOwnEnumerableDataProperty(record, key);
  return (
    property.found &&
    typeof property.value === 'string' &&
    property.value !== ''
  );
}

function hasValue(record: Record<string, unknown>, key: string): boolean {
  const property = readOwnEnumerableDataProperty(record, key);
  return property.found && property.value !== undefined;
}

function isStringPayload(value: unknown): boolean {
  return typeof value === 'string' && value.length > 0;
}

function isBinaryPayload(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.length > 0;
  }
  if (isArray(value)) {
    return false;
  }
  const arrayBufferByteLength = readNativeArrayBufferByteLength(value);
  if (arrayBufferByteLength != null) {
    return (
      isSafeNativeArrayBuffer(value as ArrayBuffer) && arrayBufferByteLength > 0
    );
  }
  if (ArrayBuffer.isView(value)) {
    const view = readSafeNativeArrayBufferView(value);
    return view != null && view.byteLength > 0;
  }
  return false;
}

function isContentPayload(value: unknown): boolean {
  return isArray(value) && value.length > 0;
}

function hasPayloadPath(
  record: Record<string, unknown>,
  path: string[],
  validator: (value: unknown) => boolean = isBinaryPayload
): boolean {
  let value: unknown = record;
  for (const key of path) {
    if (!isRecord(value)) {
      return false;
    }
    const property = readOwnEnumerableDataProperty(value, key);
    if (!property.found) {
      return false;
    }
    value = property.value;
  }
  return validator(value);
}

function hasAnyPayloadPath(
  record: Record<string, unknown>,
  paths: string[][],
  validator: (value: unknown) => boolean = isBinaryPayload
): boolean {
  return paths.some((path) => hasPayloadPath(record, path, validator));
}

function isProviderNativeToolResultBlock(
  record: Record<string, unknown>,
  type: string
): boolean {
  if (type === 'search_result') {
    return (
      hasString(record, 'title') &&
      hasString(record, 'source') &&
      hasPayloadPath(record, ['content'], isArray)
    );
  }
  if (type === 'web_search_result') {
    return hasString(record, 'url');
  }
  if (type === 'web_search_tool_result') {
    const content = readOwnEnumerableDataProperty(record, 'content');
    return (
      hasString(record, 'tool_use_id') &&
      content.found &&
      (isArray(content.value) || isRecord(content.value))
    );
  }
  if (type === 'tool_result') {
    return hasString(record, 'tool_use_id');
  }
  if (type === 'server_tool_call_result') {
    const status = readOwnEnumerableDataProperty(record, 'status');
    return (
      hasString(record, 'toolCallId') &&
      status.found &&
      (status.value === 'success' || status.value === 'error') &&
      hasValue(record, 'output')
    );
  }
  if (type === 'toolResponse') {
    const response = readOwnEnumerableDataProperty(record, 'toolResponse');
    return response.found && isRecord(response.value);
  }
  return false;
}

const MAX_ATOMIC_VALIDATION_WORK = 10_000;
const MAX_ATOMIC_VALIDATION_DEPTH = 50;
type AtomicValidationWork = { value: number };
const NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength'
)?.get;
const NATIVE_TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(
  Uint8Array.prototype
) as object;
const NATIVE_TYPED_ARRAY_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_TYPED_ARRAY_PROTOTYPE,
  'byteLength'
)?.get;
const NATIVE_DATA_VIEW_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  DataView.prototype,
  'byteLength'
)?.get;
const NATIVE_ARRAY_BUFFER_VIEW_NAMES = new Map<object, string>([
  [Buffer.prototype, 'Buffer'],
  [DataView.prototype, 'DataView'],
  [Int8Array.prototype, 'Int8Array'],
  [Uint8Array.prototype, 'Uint8Array'],
  [Uint8ClampedArray.prototype, 'Uint8ClampedArray'],
  [Int16Array.prototype, 'Int16Array'],
  [Uint16Array.prototype, 'Uint16Array'],
  [Int32Array.prototype, 'Int32Array'],
  [Uint32Array.prototype, 'Uint32Array'],
  [Float32Array.prototype, 'Float32Array'],
  [Float64Array.prototype, 'Float64Array'],
  [BigInt64Array.prototype, 'BigInt64Array'],
  [BigUint64Array.prototype, 'BigUint64Array'],
]);
const NATIVE_DATE_GET_TIME = Date.prototype.getTime;
const NATIVE_DATE_TO_JSON = Date.prototype.toJSON;
const NATIVE_DATE_TO_ISO_STRING = Date.prototype.toISOString;
const NATIVE_DATE_VALUE_OF = Date.prototype.valueOf;
const NATIVE_DATE_TO_STRING = Date.prototype.toString;
const NATIVE_DATE_TO_PRIMITIVE = Date.prototype[Symbol.toPrimitive];
const NATIVE_DATE_PROTOTYPE_METHODS: ReadonlyArray<
  readonly [PropertyKey, unknown]
> = [
  ['getTime', NATIVE_DATE_GET_TIME],
  ['toJSON', NATIVE_DATE_TO_JSON],
  ['toISOString', NATIVE_DATE_TO_ISO_STRING],
  ['valueOf', NATIVE_DATE_VALUE_OF],
  ['toString', NATIVE_DATE_TO_STRING],
  [Symbol.toPrimitive, NATIVE_DATE_TO_PRIMITIVE],
];

function consumeAtomicValidationWork(
  remaining: AtomicValidationWork,
  totalRemaining: AtomicValidationWork | undefined,
  amount = 1
): boolean {
  if (
    amount > remaining.value ||
    (totalRemaining != null && amount > totalRemaining.value)
  ) {
    remaining.value = 0;
    if (totalRemaining != null) {
      totalRemaining.value = 0;
    }
    return false;
  }
  remaining.value -= amount;
  if (totalRemaining != null) {
    totalRemaining.value -= amount;
  }
  return true;
}

function hasPotentiallyCallableToJSON(
  value: object,
  remaining?: AtomicValidationWork,
  totalRemaining?: AtomicValidationWork
): boolean {
  let current: object | null = value;
  const seen = new Set<object>();
  for (let depth = 0; current != null && depth < 100; depth++) {
    if (isProxy(current)) {
      return true;
    }
    if (
      remaining != null &&
      !consumeAtomicValidationWork(remaining, totalRemaining)
    ) {
      return true;
    }
    if (seen.has(current)) {
      return true;
    }
    seen.add(current);
    const descriptor = Object.getOwnPropertyDescriptor(current, 'toJSON');
    if (descriptor != null) {
      return (
        typeof descriptor.value === 'function' ||
        typeof descriptor.get === 'function'
      );
    }
    current = Object.getPrototypeOf(current) as object | null;
  }
  return current != null;
}

type NativeArrayBufferViewRead = {
  byteLength: number;
  name: string;
};

function readSafeNativeArrayBufferView(
  value: ArrayBufferView,
  remaining?: AtomicValidationWork,
  totalRemaining?: AtomicValidationWork,
  rejectCallableToJSON = true
): NativeArrayBufferViewRead | undefined {
  try {
    const prototype = Object.getPrototypeOf(value) as object | null;
    if (prototype == null || isProxy(prototype)) {
      return undefined;
    }
    const name = NATIVE_ARRAY_BUFFER_VIEW_NAMES.get(prototype);
    if (
      name == null ||
      Object.getOwnPropertyDescriptor(value, 'byteLength') != null ||
      Object.getOwnPropertyDescriptor(value, 'constructor') != null ||
      Object.getOwnPropertyDescriptor(value, 'toJSON') != null ||
      (rejectCallableToJSON &&
        hasPotentiallyCallableToJSON(value, remaining, totalRemaining))
    ) {
      return undefined;
    }
    const getter =
      prototype === DataView.prototype
        ? NATIVE_DATA_VIEW_BYTE_LENGTH_GETTER
        : NATIVE_TYPED_ARRAY_BYTE_LENGTH_GETTER;
    if (getter == null) {
      return undefined;
    }
    const byteLength = getter.call(value) as number;
    return Number.isSafeInteger(byteLength) && byteLength >= 0
      ? { byteLength, name }
      : undefined;
  } catch {
    return undefined;
  }
}

function isSafeNativeArrayBuffer(
  value: ArrayBuffer,
  remaining?: AtomicValidationWork,
  totalRemaining?: AtomicValidationWork
): boolean {
  try {
    if (
      Object.getPrototypeOf(value) !== ArrayBuffer.prototype ||
      NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER == null ||
      Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength')
        ?.get !== NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER ||
      hasPotentiallyCallableToJSON(value, remaining, totalRemaining)
    ) {
      return false;
    }
    NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(value);
    for (const key in value) {
      if (
        remaining != null &&
        !consumeAtomicValidationWork(remaining, totalRemaining)
      ) {
        return false;
      }
      if (Object.prototype.propertyIsEnumerable.call(value, key)) {
        return false;
      }
    }
    return Object.getOwnPropertyDescriptor(value, 'byteLength') == null;
  } catch {
    return false;
  }
}

function isSafeNativeDate(value: Date): boolean {
  try {
    if (Object.getPrototypeOf(value) !== Date.prototype) {
      return false;
    }
    NATIVE_DATE_GET_TIME.call(value);
    for (const [key, nativeMethod] of NATIVE_DATE_PROTOTYPE_METHODS) {
      if (
        Object.getOwnPropertyDescriptor(value, key) != null ||
        Object.getOwnPropertyDescriptor(Date.prototype, key)?.value !==
          nativeMethod
      ) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

function hasUnsafeCallableToJSON(
  value: unknown,
  seen = new Set<object>(),
  remaining = { value: MAX_ATOMIC_VALIDATION_WORK },
  depth = 0,
  totalRemaining?: AtomicValidationWork
): boolean {
  if (isProxy(value)) {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  if (
    seen.has(value) ||
    depth >= MAX_ATOMIC_VALIDATION_DEPTH ||
    !consumeAtomicValidationWork(remaining, totalRemaining)
  ) {
    return true;
  }
  const valueIsArray = isArray(value);
  if (!valueIsArray) {
    const arrayBufferByteLength = readNativeArrayBufferByteLength(value);
    if (arrayBufferByteLength != null) {
      return !isSafeNativeArrayBuffer(
        value as unknown as ArrayBuffer,
        remaining,
        totalRemaining
      );
    }
    const date = readNativeDate(value);
    if (date.matched) {
      return !isSafeNativeDate(value as unknown as Date);
    }
    if (ArrayBuffer.isView(value)) {
      return (
        readSafeNativeArrayBufferView(value, remaining, totalRemaining) == null
      );
    }
  }

  seen.add(value);
  try {
    if (hasPotentiallyCallableToJSON(value, remaining, totalRemaining)) {
      return true;
    }
    if (valueIsArray) {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Array.prototype && prototype !== null) {
        return true;
      }
      const lengthDescriptor = Object.getOwnPropertyDescriptor(value, 'length');
      const length = lengthDescriptor?.value;
      if (
        typeof length !== 'number' ||
        !Number.isSafeInteger(length) ||
        length < 0 ||
        !consumeAtomicValidationWork(remaining, totalRemaining, length)
      ) {
        return true;
      }
      for (let i = 0; i < length; i++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(i));
        if (
          descriptor == null ||
          typeof descriptor.get === 'function' ||
          typeof descriptor.set === 'function'
        ) {
          return true;
        }
        if (
          hasUnsafeCallableToJSON(
            descriptor.value,
            seen,
            remaining,
            depth + 1,
            totalRemaining
          )
        ) {
          return true;
        }
      }
      return false;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      return true;
    }
    for (const key in value) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (descriptor == null) {
        return true;
      }
      if (descriptor.enumerable !== true) {
        continue;
      }
      if (
        typeof descriptor.get === 'function' ||
        typeof descriptor.set === 'function' ||
        !consumeAtomicValidationWork(remaining, totalRemaining)
      ) {
        return true;
      }
      if (
        hasUnsafeCallableToJSON(
          descriptor.value,
          seen,
          remaining,
          depth + 1,
          totalRemaining
        )
      ) {
        return true;
      }
    }
    return false;
  } catch {
    return true;
  } finally {
    seen.delete(value);
  }
}

export function isAtomicToolContentBlock(value: unknown): boolean {
  try {
    return (
      !hasUnsafeCallableToJSON(value) &&
      isAtomicToolContentBlockUnchecked(value)
    );
  } catch {
    return false;
  }
}

function isAtomicToolContentBlockUnchecked(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  const record = value;
  const typeProperty = readOwnEnumerableDataProperty(record, 'type');
  if (!typeProperty.found) {
    const cachePointProperty = readOwnEnumerableDataProperty(
      record,
      'cachePoint'
    );
    if (!cachePointProperty.found || !isRecord(cachePointProperty.value)) {
      return false;
    }
    const cachePointType = readOwnEnumerableDataProperty(
      cachePointProperty.value,
      'type'
    );
    return cachePointType.found && cachePointType.value === 'default';
  }
  const type = typeof typeProperty.value === 'string' ? typeProperty.value : '';
  if (type.length > MAX_TOOL_CONTENT_TYPE_CHARS) {
    return false;
  }
  const slashIndex = type.indexOf('/');
  const isMimeType =
    slashIndex > 0 &&
    slashIndex < type.length - 1 &&
    slashIndex === type.lastIndexOf('/');
  if (
    !ATOMIC_CONTENT_TYPES.has(type) &&
    !PROVIDER_NATIVE_TOOL_RESULT_TYPES.has(type) &&
    !isMimeType
  ) {
    return false;
  }

  if (PROVIDER_NATIVE_TOOL_RESULT_TYPES.has(type)) {
    return isProviderNativeToolResultBlock(record, type);
  }

  if (type === 'computer_screenshot') {
    return hasAnyPayloadPath(
      record,
      [['image_url'], ['file_id']],
      isStringPayload
    );
  }
  if (type === 'input_image') {
    return hasAnyPayloadPath(
      record,
      [['image_url'], ['file_id']],
      isStringPayload
    );
  }
  if (type === 'image_url') {
    return hasAnyPayloadPath(
      record,
      [['image_url'], ['image_url', 'url']],
      isStringPayload
    );
  }
  if (type === 'image_file') {
    return hasAnyPayloadPath(
      record,
      [
        ['image_file', 'file_id'],
        ['image_file', 'fileId'],
      ],
      isStringPayload
    );
  }
  if (type === 'input_audio') {
    return hasAnyPayloadPath(record, [
      ['input_audio', 'data'],
      ['input_audio', 'bytes'],
    ]);
  }
  if (type === 'resource_link') {
    return hasString(record, 'uri');
  }
  if (type === 'video_url') {
    return hasAnyPayloadPath(
      record,
      [['video_url'], ['video_url', 'url']],
      isStringPayload
    );
  }
  if (type === 'resource') {
    return (
      hasAnyPayloadPath(record, [
        ['resource', 'blob'],
        ['resource', 'bytes'],
        ['resource', 'data'],
        ['resource', 'text'],
      ]) || hasPayloadPath(record, ['resource', 'content'], isContentPayload)
    );
  }

  if (isMimeType) {
    return hasPayloadPath(record, ['data'], isStringPayload);
  }

  if (type === 'image' || type === 'audio' || type === 'video') {
    return hasAnyPayloadPath(record, [
      ['data'],
      ['bytes'],
      ['url'],
      ['source', 'data'],
      ['source', 'bytes'],
      ['source', 'url'],
      [type, 'data'],
      [type, 'bytes'],
      [type, 'url'],
      [type, 'source', 'data'],
      [type, 'source', 'bytes'],
      [type, 'source', 'url'],
    ]);
  }

  if (type === 'media') {
    const hasMimeType =
      hasString(record, 'mimeType') || hasString(record, 'mime_type');
    return (
      hasMimeType &&
      hasAnyPayloadPath(record, [
        ['data'],
        ['bytes'],
        ['fileUri'],
        ['media', 'data'],
        ['media', 'bytes'],
        ['media', 'fileUri'],
        ['media', 'source', 'bytes'],
      ])
    );
  }

  if (type === 'document' || type === 'file') {
    const nestedType = type;
    return (
      hasAnyPayloadPath(record, [
        ['data'],
        ['bytes'],
        ['url'],
        ['file_data'],
        ['file_id'],
        ['fileId'],
        ['fileUri'],
        ['source', 'data'],
        ['source', 'bytes'],
        ['source', 'url'],
        ['source', 'file_data'],
        ['source', 'file_id'],
        ['source', 'fileId'],
        ['source', 'fileUri'],
        [nestedType, 'data'],
        [nestedType, 'bytes'],
        [nestedType, 'url'],
        [nestedType, 'file_data'],
        [nestedType, 'file_id'],
        [nestedType, 'fileId'],
        [nestedType, 'fileUri'],
        [nestedType, 'source', 'data'],
        [nestedType, 'source', 'bytes'],
        [nestedType, 'source', 'url'],
      ]) ||
      hasPayloadPath(record, ['source', 'content'], isContentPayload) ||
      hasPayloadPath(
        record,
        [nestedType, 'source', 'content'],
        isContentPayload
      ) ||
      (readOwnEnumerableDataProperty(record, 'source_type').value === 'text' &&
        hasString(record, 'text')) ||
      (readOwnEnumerableDataProperty(record, 'source_type').value === 'id' &&
        hasString(record, 'id'))
    );
  }

  return false;
}

const TOOL_BLOCK_OPAQUE = 0;
const TOOL_BLOCK_TEXT = 1;
const TOOL_BLOCK_ATOMIC = 2;

type ToolBlockClassificationContext = {
  totalRemaining: AtomicValidationWork;
  cache: WeakMap<object, number>;
};

function createToolBlockClassificationContext(): ToolBlockClassificationContext {
  return {
    totalRemaining: { value: MAX_STRUCTURED_SERIALIZATION_WORK },
    cache: new WeakMap<object, number>(),
  };
}

function classifyToolContentBlock(
  value: unknown,
  context: ToolBlockClassificationContext
): number {
  if (isRecord(value)) {
    const cached = context.cache.get(value);
    if (cached != null) {
      return cached;
    }
  }
  const remaining = { value: MAX_ATOMIC_VALIDATION_WORK };
  let kind: number;
  try {
    if (
      hasUnsafeCallableToJSON(
        value,
        new Set<object>(),
        remaining,
        0,
        context.totalRemaining
      )
    ) {
      kind = TOOL_BLOCK_OPAQUE;
    } else if (isAtomicToolContentBlockUnchecked(value)) {
      kind = TOOL_BLOCK_ATOMIC;
    } else {
      kind = isTextBlockUnchecked(value) ? TOOL_BLOCK_TEXT : TOOL_BLOCK_OPAQUE;
    }
  } catch {
    kind = TOOL_BLOCK_OPAQUE;
  }
  if (isRecord(value) && context.totalRemaining.value > 0) {
    context.cache.set(value, kind);
  }
  return kind;
}

/**
 * Produces the text representation providers use for structured tool results.
 * Text-only block arrays stay readable; opaque blocks use canonical JSON.
 */
export function serializeToolContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  if (!isArray(content)) {
    return serializeStructuredValue(content);
  }
  if (getDenseTextBlockArrayLength(content) != null) {
    return (content as TextToolContentBlock[])
      .map((block) => block.text)
      .join('\n');
  }
  return serializeStructuredValue(content);
}

export function getToolContentCharLength(content: unknown): number {
  if (typeof content === 'string') {
    return content.length;
  }
  if (isArray(content)) {
    const denseTextLength = getDenseTextBlockArrayLength(content);
    if (denseTextLength != null) {
      return denseTextLength;
    }
  }
  try {
    return estimateStructuredChars(content);
  } catch {
    return Number.MAX_SAFE_INTEGER;
  }
}

/**
 * Produces provider-facing tool-result text without ever exceeding `maxChars`.
 */
export function serializeToolContentBounded(
  content: unknown,
  maxChars: number
): string {
  const normalizedMaxChars = normalizeCharLimit(maxChars);
  if (typeof content === 'string') {
    return truncateToolResultContent(content, normalizedMaxChars);
  }
  if (!isArray(content)) {
    return serializeStructuredValueWithinLimit(content, normalizedMaxChars);
  }

  const originalChars = getDenseTextBlockArrayLength(content);
  if (originalChars == null) {
    return serializeStructuredValueWithinLimit(content, normalizedMaxChars);
  }
  return serializeDenseTextBlocksWithinLimit(
    content as TextToolContentBlock[],
    originalChars,
    normalizedMaxChars
  );
}

function serializeDenseTextBlocksWithinLimit(
  textBlocks: readonly TextToolContentBlock[],
  originalChars: number,
  normalizedMaxChars: number
): string {
  if (originalChars <= normalizedMaxChars) {
    return textBlocks.map((block) => block.text).join('\n');
  }
  const indicator =
    `\n\n… [truncated: ${originalChars} chars exceeded ` +
    `${normalizedMaxChars} limit] …`;
  const available = Math.max(0, normalizedMaxChars - indicator.length);
  let preview = '';
  for (let i = 0; i < textBlocks.length && preview.length < available; i++) {
    if (i > 0) {
      preview += '\n';
    }
    preview += textBlocks[i].text.slice(0, available - preview.length);
  }
  return sliceWithoutSplittingSurrogates(
    sliceWithoutSplittingSurrogates(preview, 0) + indicator,
    0,
    normalizedMaxChars
  );
}

export type CompactToolContentResult = {
  content: ToolContent;
  changed: boolean;
  originalChars: number;
};

function isProviderImageUrl(value: unknown): value is string {
  if (typeof value !== 'string' || value === '') {
    return false;
  }
  const isDataImage = value.startsWith('data:image/');
  if (
    value.length >
    (isDataImage
      ? HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      : MAX_PROVIDER_IMAGE_URL_CHARS)
  ) {
    return false;
  }
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code <= 0x20 || code === 0x7f) {
      return false;
    }
  }
  if (isDataImage) {
    const separator = value.indexOf(',');
    if (
      separator < 0 ||
      !value.slice(0, separator).toLowerCase().endsWith(';base64')
    ) {
      return false;
    }
    const payload = value.slice(separator + 1);
    return (
      payload.length > 0 &&
      payload.length % 4 === 0 &&
      /^[a-z0-9+/]+={0,2}$/i.test(payload)
    );
  }
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

function isProviderFileId(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length <= MAX_PROVIDER_FILE_ID_CHARS &&
    /^file[-_][a-z0-9_-]+$/i.test(value)
  );
}

function hasOnlyEnumerableKeys(
  record: Record<string, unknown>,
  allowed: ReadonlySet<string>
): boolean {
  try {
    const keys = Object.keys(record);
    return keys.every((key) => allowed.has(key));
  } catch {
    return false;
  }
}

export type CacheControlledTextToolContent = [
  {
    type: 'text';
    text: string;
    cache_control: { type: 'ephemeral'; ttl?: '1h' };
  },
];

const SINGLE_TEXT_TOOL_CONTENT_KEYS = new Set(['type', 'text']);
const CACHE_CONTROLLED_TEXT_TOOL_CONTENT_KEYS = new Set([
  'type',
  'text',
  'cache_control',
]);
const CACHE_CONTROL_KEYS = new Set(['type', 'ttl']);

function hasSafePlainPrototype(value: object): boolean {
  try {
    const prototype = Object.getPrototypeOf(value) as object | null;
    return (
      (prototype === Object.prototype || prototype === null) &&
      !hasPotentiallyCallableToJSON(value)
    );
  } catch {
    return false;
  }
}

function readSafeArrayItem(
  content: unknown,
  expectedLength: number,
  index: number
): OwnDataProperty {
  if (!isArray(content) || isProxy(content)) {
    return { found: false };
  }
  try {
    const length = Object.getOwnPropertyDescriptor(content, 'length');
    if (length?.value !== expectedLength) {
      return { found: false };
    }
    const prototype = Object.getPrototypeOf(content) as object | null;
    if (
      (prototype !== Array.prototype && prototype !== null) ||
      hasPotentiallyCallableToJSON(content)
    ) {
      return { found: false };
    }
    const item = Object.getOwnPropertyDescriptor(content, String(index));
    if (item?.enumerable !== true || !('value' in item)) {
      return { found: false };
    }
    return { found: true, value: item.value };
  } catch {
    return { found: false };
  }
}

/** Reads one safe text block and returns its bounded text payload. */
export function getBoundedSingleTextToolContent(
  content: unknown,
  maxChars: number
): string | undefined {
  const block = readSafeArrayItem(content, 1, 0);
  if (
    !block.found ||
    !isRecord(block.value) ||
    isProxy(block.value) ||
    !hasSafePlainPrototype(block.value) ||
    !hasOnlyEnumerableKeys(block.value, SINGLE_TEXT_TOOL_CONTENT_KEYS)
  ) {
    return undefined;
  }
  const type = readOwnEnumerableDataProperty(block.value, 'type');
  const text = readOwnEnumerableDataProperty(block.value, 'text');
  if (
    !type.found ||
    type.value !== 'text' ||
    !text.found ||
    typeof text.value !== 'string'
  ) {
    return undefined;
  }
  const normalizedMaxChars = Number.isFinite(maxChars)
    ? Math.max(0, Math.floor(maxChars))
    : Number.MAX_SAFE_INTEGER;
  return truncateToolResultContent(text.value, normalizedMaxChars);
}

/**
 * Canonicalizes the single cache-decorated text block produced by tail prompt
 * caching without invoking accessors or custom serialization hooks.
 */
export function getBoundedCacheControlledTextToolContent(
  content: unknown,
  maxChars: number
): CacheControlledTextToolContent | undefined {
  const block = readSafeArrayItem(content, 1, 0);
  if (
    !block.found ||
    !isRecord(block.value) ||
    isProxy(block.value) ||
    !hasSafePlainPrototype(block.value) ||
    !hasOnlyEnumerableKeys(block.value, CACHE_CONTROLLED_TEXT_TOOL_CONTENT_KEYS)
  ) {
    return undefined;
  }
  const type = readOwnEnumerableDataProperty(block.value, 'type');
  const text = readOwnEnumerableDataProperty(block.value, 'text');
  const cacheControl = readOwnEnumerableDataProperty(
    block.value,
    'cache_control'
  );
  if (
    !type.found ||
    type.value !== 'text' ||
    !text.found ||
    typeof text.value !== 'string' ||
    !cacheControl.found ||
    !isRecord(cacheControl.value) ||
    isProxy(cacheControl.value) ||
    !hasSafePlainPrototype(cacheControl.value) ||
    !hasOnlyEnumerableKeys(cacheControl.value, CACHE_CONTROL_KEYS)
  ) {
    return undefined;
  }

  const cacheType = readOwnEnumerableDataProperty(cacheControl.value, 'type');
  const ttl = readOwnEnumerableDataProperty(cacheControl.value, 'ttl');
  let hasOwnTtl: boolean;
  try {
    hasOwnTtl =
      Object.getOwnPropertyDescriptor(cacheControl.value, 'ttl') != null;
  } catch {
    return undefined;
  }
  if (
    !cacheType.found ||
    cacheType.value !== 'ephemeral' ||
    (hasOwnTtl && (!ttl.found || ttl.value !== '1h'))
  ) {
    return undefined;
  }

  const normalizedMaxChars = Number.isFinite(maxChars)
    ? Math.max(0, Math.floor(maxChars))
    : Number.MAX_SAFE_INTEGER;
  return [
    {
      type: 'text',
      text: truncateToolResultContent(text.value, normalizedMaxChars),
      cache_control: {
        type: 'ephemeral',
        ...(ttl.found ? { ttl: '1h' as const } : {}),
      },
    },
  ];
}

export type ComputerCallOutputScreenshot =
  | { type: 'input_image'; image_url: string; detail?: InputImageDetail }
  | { type: 'input_image'; file_id: string; detail?: InputImageDetail }
  | { type: 'computer_screenshot'; image_url: string }
  | { type: 'computer_screenshot'; file_id: string };

type InputImageDetail = 'low' | 'high' | 'auto' | 'original';

const INPUT_IMAGE_KEYS = new Set(['type', 'image_url', 'file_id', 'detail']);
const COMPUTER_SCREENSHOT_KEYS = new Set(['type', 'image_url', 'file_id']);
const IMAGE_URL_KEYS = new Set(['type', 'image_url', 'detail']);
const NESTED_IMAGE_URL_KEYS = new Set(['url', 'detail']);
const INPUT_IMAGE_DETAILS = new Set(['low', 'high', 'auto', 'original']);

/**
 * Selects the exact screenshot shape accepted by LangChain's Responses
 * converter without invoking accessors or user-defined serialization hooks.
 */
export function getComputerCallOutputScreenshot(
  content: unknown
): ComputerCallOutputScreenshot | undefined {
  if (isProviderImageUrl(content)) {
    return { type: 'input_image', image_url: content };
  }
  if (
    !isArray(content) ||
    hasUnsafeCallableToJSON(content) ||
    content.length !== 1
  ) {
    return undefined;
  }

  const descriptor = Object.getOwnPropertyDescriptor(content, '0');
  if (descriptor == null || !('value' in descriptor)) {
    return undefined;
  }
  const block = descriptor.value;
  if (!isRecord(block)) {
    return undefined;
  }
  const type = readOwnEnumerableDataProperty(block, 'type');
  if (!type.found) {
    return undefined;
  }
  const imageUrl = readOwnEnumerableDataProperty(block, 'image_url');
  const fileId = readOwnEnumerableDataProperty(block, 'file_id');
  const validImageUrl =
    imageUrl.found && isProviderImageUrl(imageUrl.value)
      ? imageUrl.value
      : undefined;
  const validFileId =
    fileId.found && isProviderFileId(fileId.value) ? fileId.value : undefined;
  const hasImageUrl = validImageUrl != null;
  const hasFileId = validFileId != null;

  if (
    type.value === 'input_image' &&
    hasOnlyEnumerableKeys(block, INPUT_IMAGE_KEYS)
  ) {
    if (
      (imageUrl.found && !hasImageUrl) ||
      (fileId.found && !hasFileId) ||
      hasImageUrl === hasFileId
    ) {
      return undefined;
    }
    const detail = readOwnEnumerableDataProperty(block, 'detail');
    if (
      detail.found &&
      (typeof detail.value !== 'string' ||
        !INPUT_IMAGE_DETAILS.has(detail.value))
    ) {
      return undefined;
    }
    return hasImageUrl
      ? {
        type: 'input_image',
        image_url: validImageUrl,
        ...(detail.found ? { detail: detail.value as InputImageDetail } : {}),
      }
      : {
        type: 'input_image',
        file_id: validFileId as string,
        ...(detail.found ? { detail: detail.value as InputImageDetail } : {}),
      };
  }
  if (
    type.value === 'computer_screenshot' &&
    hasOnlyEnumerableKeys(block, COMPUTER_SCREENSHOT_KEYS)
  ) {
    if (
      (imageUrl.found && !hasImageUrl) ||
      (fileId.found && !hasFileId) ||
      hasImageUrl === hasFileId
    ) {
      return undefined;
    }
    return hasImageUrl
      ? { type: 'computer_screenshot', image_url: validImageUrl }
      : {
        type: 'computer_screenshot',
        file_id: validFileId as string,
      };
  }
  if (
    type.value === 'image_url' &&
    !fileId.found &&
    hasOnlyEnumerableKeys(block, IMAGE_URL_KEYS)
  ) {
    if (hasImageUrl) {
      return { type: 'input_image', image_url: validImageUrl };
    }
    if (
      imageUrl.found &&
      isRecord(imageUrl.value) &&
      hasOnlyEnumerableKeys(imageUrl.value, NESTED_IMAGE_URL_KEYS)
    ) {
      const url = readOwnEnumerableDataProperty(imageUrl.value, 'url');
      if (url.found && isProviderImageUrl(url.value)) {
        return { type: 'input_image', image_url: url.value };
      }
    }
  }
  return undefined;
}

/** Returns the provider image URL when a screenshot is URL-backed. */
export function getComputerCallOutputImageUrl(
  content: unknown
): string | undefined {
  const screenshot = getComputerCallOutputScreenshot(content);
  return screenshot != null && 'image_url' in screenshot
    ? screenshot.image_url
    : undefined;
}

/** Returns whether content can be sent as a native computer screenshot. */
export function isComputerCallOutputContent(
  content: unknown
): content is ToolContent {
  return getComputerCallOutputScreenshot(content) != null;
}

/** Returns whether a ToolMessage carries the native computer-output marker. */
export function hasComputerCallOutputMarker(message: BaseMessage): boolean {
  try {
    if (message.getType() !== 'tool') {
      return false;
    }
    const metadata = message.additional_kwargs as unknown;
    if (!isRecord(metadata)) {
      return false;
    }
    const type = readOwnEnumerableDataProperty(metadata, 'type');
    return type.found && type.value === 'computer_call_output';
  } catch {
    return false;
  }
}

/** Native Responses computer screenshots are indivisible provider payloads. */
export function isComputerCallOutputMessage(message: BaseMessage): boolean {
  return (
    hasComputerCallOutputMarker(message) &&
    isComputerCallOutputContent(message.content)
  );
}

/**
 * Bounds structured tool output without slicing binary/media payloads.
 * Purely textual/structured arrays become a universally valid string tool
 * result. For mixed-media arrays, compactable blocks become one text preview
 * while atomic blocks remain intact.
 */
export function compactToolContent(
  content: unknown,
  maxChars: number
): CompactToolContentResult {
  const normalizedMaxChars = Number.isFinite(maxChars)
    ? Math.max(0, Math.floor(maxChars))
    : Number.MAX_SAFE_INTEGER;
  if (typeof content === 'string') {
    return {
      content: truncateToolResultContent(content, normalizedMaxChars),
      changed: content.length > normalizedMaxChars,
      originalChars: content.length,
    };
  }
  if (!isArray(content)) {
    const serialized = serializeStructuredValueBounded(
      content,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
  if (isProxy(content)) {
    const serialized = serializeStructuredValueBounded(
      content,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
  const contentBlocks = content as ToolContentBlock[];

  let contentLength: number;
  try {
    contentLength = contentBlocks.length;
  } catch {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: Number.MAX_SAFE_INTEGER,
    };
  }
  if (contentLength > MAX_STRUCTURED_SERIALIZATION_WORK) {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }

  try {
    const blockKinds = new Uint8Array(contentLength);
    const validation = createToolBlockClassificationContext();
    const structuredWork = createStructuredWorkState();
    const containerValidationWork = {
      value: MAX_ATOMIC_VALIDATION_WORK,
    };
    const containerPrototype = Object.getPrototypeOf(contentBlocks);
    if (
      (containerPrototype !== Array.prototype && containerPrototype !== null) ||
      hasPotentiallyCallableToJSON(
        contentBlocks,
        containerValidationWork,
        validation.totalRemaining
      )
    ) {
      const serialized = serializeStructuredValueBounded(
        contentBlocks,
        normalizedMaxChars
      );
      return {
        content: serialized.content,
        changed: true,
        originalChars: serialized.originalChars,
      };
    }
    let atomicBlockCount = 0;
    let denseTextLength = contentLength > 0 ? contentLength - 1 : undefined;
    for (let i = 0; i < contentLength; i++) {
      const descriptor = Object.getOwnPropertyDescriptor(
        contentBlocks,
        String(i)
      );
      if (
        descriptor == null ||
        typeof descriptor.get === 'function' ||
        typeof descriptor.set === 'function'
      ) {
        const serialized = serializeStructuredValueBounded(
          contentBlocks,
          normalizedMaxChars
        );
        return {
          content: serialized.content,
          changed: true,
          originalChars: serialized.originalChars,
        };
      }
      const block = descriptor.value as ToolContentBlock;
      const kind = classifyToolContentBlock(block, validation);
      if (validation.totalRemaining.value <= 0) {
        const serialized = serializeStructuredValueBounded(
          contentBlocks,
          normalizedMaxChars
        );
        return {
          content: serialized.content,
          changed: true,
          originalChars: serialized.originalChars,
        };
      }
      blockKinds[i] = kind;
      if (kind === TOOL_BLOCK_ATOMIC) {
        atomicBlockCount++;
      }
      if (denseTextLength == null) {
        continue;
      }
      if (kind !== TOOL_BLOCK_TEXT) {
        denseTextLength = undefined;
        continue;
      }
      const text = readOwnEnumerableDataProperty(block, 'text').value;
      denseTextLength += typeof text === 'string' ? text.length : 0;
      if (denseTextLength >= Number.MAX_SAFE_INTEGER) {
        denseTextLength = Number.MAX_SAFE_INTEGER;
      }
    }
    const originalChars =
      denseTextLength ??
      estimateStructuredChars(
        contentBlocks,
        Number.MAX_SAFE_INTEGER,
        [],
        structuredWork
      );
    if (atomicBlockCount === 0) {
      if (denseTextLength != null && originalChars <= normalizedMaxChars) {
        return {
          content: contentBlocks,
          changed: false,
          originalChars,
        };
      }
      const serialized =
        denseTextLength != null
          ? serializeDenseTextBlocksWithinLimit(
              contentBlocks as TextToolContentBlock[],
              denseTextLength,
              normalizedMaxChars
          )
          : serializeStructuredValueWithinLimit(
            contentBlocks,
            normalizedMaxChars,
            structuredWork
          );
      return {
        content: truncateToolResultContent(serialized, normalizedMaxChars),
        // Opaque arrays are normalized even when they are small so every
        // provider receives a universally valid tool-result string.
        changed: true,
        originalChars,
      };
    }

    let hasOpaqueBlocks = false;
    const normalizedBlocks: ToolContentBlock[] = [];
    const normalizedKinds: number[] = [];
    let opaqueRun: unknown[] = [];
    const flushOpaqueRun = (): void => {
      if (opaqueRun.length === 0) {
        return;
      }
      normalizedBlocks.push({
        type: 'text',
        text: serializeStructuredValueWithinLimit(
          opaqueRun,
          normalizedMaxChars,
          structuredWork
        ),
      });
      normalizedKinds.push(TOOL_BLOCK_TEXT);
      opaqueRun = [];
    };
    for (let i = 0; i < contentLength; i++) {
      const block = contentBlocks[i];
      const kind = blockKinds[i];
      if (kind !== TOOL_BLOCK_OPAQUE) {
        flushOpaqueRun();
        normalizedBlocks.push(block);
        normalizedKinds.push(kind);
      } else {
        hasOpaqueBlocks = true;
        opaqueRun.push(block);
      }
    }
    flushOpaqueRun();

    const normalizedLength = estimateStructuredChars(
      normalizedBlocks,
      Number.MAX_SAFE_INTEGER,
      [],
      structuredWork
    );
    if (normalizedLength <= normalizedMaxChars) {
      return {
        content: hasOpaqueBlocks ? normalizedBlocks : contentBlocks,
        changed: hasOpaqueBlocks,
        originalChars,
      };
    }

    // Keep small media/resource blocks intact, but never let a single inline
    // base64/blob block defeat the cap. Half of the budget is reserved for
    // compactable text so the model still receives an explanation.
    const atomicBudget = Math.floor(normalizedMaxChars / 2);
    let atomicChars = 0;
    let preservedAtomicCount = 0;
    const preservedAtomicFlags = new Uint8Array(normalizedBlocks.length);
    let omittedAtomicCount = 0;
    const omittedAtomicTypes: string[] = [];
    const compactableBlocks: ToolContentBlock[] = [];
    let compactableChars = 0;
    for (let i = 0; i < normalizedBlocks.length; i++) {
      const block = normalizedBlocks[i];
      if (normalizedKinds[i] !== TOOL_BLOCK_ATOMIC) {
        if (compactableBlocks.length > 0) {
          compactableChars++;
        }
        compactableBlocks.push(block);
        compactableChars += (block as TextToolContentBlock).text.length;
        continue;
      }
      const blockChars = estimateStructuredChars(
        block,
        atomicBudget - atomicChars,
        [],
        structuredWork
      );
      if (atomicChars + blockChars <= atomicBudget) {
        preservedAtomicFlags[i] = 1;
        preservedAtomicCount++;
        atomicChars += blockChars;
      } else {
        omittedAtomicCount++;
        const typeProperty = isRecord(block)
          ? readOwnEnumerableDataProperty(block, 'type')
          : { found: false as const };
        if (omittedAtomicTypes.length < 8) {
          const rawType =
            typeProperty.found && typeof typeProperty.value === 'string'
              ? typeProperty.value
              : 'media';
          omittedAtomicTypes.push(truncateToolResultContent(rawType, 80));
        }
      }
    }

    let previewSource = serializeDenseTextBlocksWithinLimit(
      compactableBlocks as TextToolContentBlock[],
      compactableChars,
      normalizedMaxChars
    );
    if (omittedAtomicCount > 0) {
      const undisplayedCount = omittedAtomicCount - omittedAtomicTypes.length;
      const omittedNotice =
        `[omitted ${omittedAtomicCount} oversized atomic tool-content ` +
        `block${omittedAtomicCount === 1 ? '' : 's'}: ` +
        `${omittedAtomicTypes.join(', ')}` +
        `${undisplayedCount > 0 ? `, +${undisplayedCount} more` : ''}]`;
      previewSource =
        previewSource.length > 0
          ? `${previewSource}\n${omittedNotice}`
          : omittedNotice;
    }

    const structuralAllowance =
      preservedAtomicCount * 4 + (previewSource.length > 0 ? 32 : 2);
    const previewBudget = Math.max(
      0,
      normalizedMaxChars - atomicChars - structuralAllowance
    );
    const preview = truncateToolResultContent(previewSource, previewBudget);
    const compacted: ToolContentBlock[] = [];
    let previewInserted = false;
    for (let i = 0; i < normalizedBlocks.length; i++) {
      const block = normalizedBlocks[i];
      if (normalizedKinds[i] === TOOL_BLOCK_ATOMIC) {
        if (preservedAtomicFlags[i] === 1) {
          compacted.push(block);
        }
      } else if (!previewInserted) {
        if (preview.length > 0) {
          compacted.push({ type: 'text', text: preview });
        }
        previewInserted = true;
      }
    }
    if (!previewInserted && preview.length > 0) {
      compacted.push({ type: 'text', text: preview });
    }

    // JSON framing can make the block array a few bytes larger than the
    // character budget. Fall back to bounded provider-neutral text rather than
    // leaking an oversized media payload.
    const compactedChars =
      preservedAtomicCount > 0 || compacted.length === 0
        ? estimateStructuredChars(
          compacted,
          Number.MAX_SAFE_INTEGER,
          [],
          structuredWork
        )
        : preview.length;
    if (compactedChars > normalizedMaxChars) {
      return {
        content: serializeStructuredValueWithinLimit(
          normalizedBlocks,
          normalizedMaxChars,
          structuredWork
        ),
        changed: true,
        originalChars,
      };
    }

    return {
      content: compacted,
      changed: true,
      originalChars,
    };
  } catch {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
}

/** Clones a ToolMessage while retaining fields omitted by ad-hoc clones. */
export function cloneToolMessageWithContent(
  message: ToolMessage,
  content: ToolContent,
  artifact: unknown = message.artifact
): ToolMessage {
  return new ToolMessage({
    content,
    tool_call_id: message.tool_call_id,
    name: message.name,
    id: message.id,
    status: message.status,
    artifact,
    metadata: message.metadata,
    additional_kwargs: { ...message.additional_kwargs },
    response_metadata: message.response_metadata,
  });
}

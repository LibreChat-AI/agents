import { types } from 'node:util';

const MAX_ARGUMENT_DEPTH = 64;
const INVALID = 'Accepted tool call arguments are not JSON serializable';
const LIMIT = 'Tool projection buffer limit exceeded';

function invalid(): never {
  throw new Error(INVALID);
}

/** structuredClone and VM providers may return objects from another realm. */
function hasJSONPrototype(value: object, array: boolean): boolean {
  const proto: unknown = Object.getPrototypeOf(value);
  if (proto === null) return true;
  if (typeof proto !== 'object' || types.isProxy(proto)) return false;
  const ctor: unknown = Object.getOwnPropertyDescriptor(
    proto,
    'constructor'
  )?.value;
  if (typeof ctor !== 'function' || types.isProxy(ctor)) return false;
  return (
    Object.getOwnPropertyDescriptor(ctor, 'prototype')?.value === proto &&
    Function.prototype.toString.call(ctor) ===
      Function.prototype.toString.call(array ? Array : Object)
  );
}

/** Encode JSON data, not arbitrary JS values. No getters, proxies or toJSON hooks run.
 * Repeated references expand like JSON (and count against the budget); cycles fail.
 * Limits bound output and traversal, including deeply nested/alias-heavy input.
 */
export function serializeToolArguments(
  value: unknown,
  maxBytes: number
): string {
  if (maxBytes < 2) throw new Error(LIMIT);
  const parts: string[] = [];
  const ancestors = new Set<object>();
  let remaining = maxBytes;
  const append = (text: string): void => {
    remaining -= Buffer.byteLength(text, 'utf8');
    if (remaining < 0) throw new Error(LIMIT);
    parts.push(text);
  };
  const string = (text: string): void => {
    // Every UTF-16 unit needs at least one encoded byte. Check before allocating escapes.
    if (text.length + 2 > remaining) throw new Error(LIMIT);
    append(JSON.stringify(text));
  };
  const encode = (input: unknown, depth: number): void => {
    if (depth > MAX_ARGUMENT_DEPTH) invalid();
    if (input === null) {
      append('null');
      return;
    }
    if (typeof input === 'string') {
      string(input);
      return;
    }
    if (typeof input === 'boolean') {
      append(input ? 'true' : 'false');
      return;
    }
    if (typeof input === 'number') {
      if (!Number.isFinite(input)) invalid();
      append(Object.is(input, -0) ? '-0' : String(input));
      return;
    }
    if (typeof input !== 'object') invalid();
    if (types.isProxy(input) || ancestors.has(input)) invalid();
    const array = Array.isArray(input);
    if (!hasJSONPrototype(input, array)) invalid();
    const keys = Reflect.ownKeys(input);
    // At least one byte per entry, plus punctuation. Bound descriptor work up front.
    if (keys.length > remaining + (array ? 1 : 0)) throw new Error(LIMIT);
    ancestors.add(input);
    if (array) {
      // Dense arrays only, with no symbols, hidden entries or extra named properties.
      if (keys.length !== input.length + 1) invalid();
      append('[');
      for (let i = 0; i < input.length; i++) {
        const descriptor = Object.getOwnPropertyDescriptor(input, String(i));
        if (
          descriptor == null ||
          descriptor.enumerable !== true ||
          !('value' in descriptor)
        )
          invalid();
        if (i !== 0) append(',');
        encode(descriptor.value, depth + 1);
      }
      append(']');
    } else {
      append('{');
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        if (typeof key !== 'string') invalid();
        const descriptor = Object.getOwnPropertyDescriptor(input, key);
        if (
          descriptor == null ||
          descriptor.enumerable !== true ||
          !('value' in descriptor)
        )
          invalid();
        if (i !== 0) append(',');
        string(key);
        append(':');
        encode(descriptor.value, depth + 1);
      }
      append('}');
    }
    ancestors.delete(input);
  };
  if (value != null && typeof value === 'object' && types.isProxy(value))
    invalid();
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Accepted tool call arguments must be an object');
  }
  encode(value, 0);
  return parts.join('');
}

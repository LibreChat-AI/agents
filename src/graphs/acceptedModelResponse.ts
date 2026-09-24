import { types } from 'node:util';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ModelResponseEvent } from '@/types';
import {
  cloneToolArguments,
  serializeToolArguments,
} from '@/utils/acceptedToolArguments';
import { linkStreamLimitCanonical } from '@/llm/streamLimits';

const MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024;
const MAX_SNAPSHOT_CALLS = 1024;

export class InvalidModelToolCallError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidModelToolCallError';
  }
}

/** Detach executable calls before dispatch. Invalid diagnostics remain available
 * for ToolNode to synthesize paired error results; accepted projection rejects them.
 */
export function detachValidatedModelToolCalls(
  message: AIMessageChunk,
  partial = false
): void {
  try {
    // Inspect every collection before replacing any provider-owned data. Raw
    // fragments are read by accounting and handlers even when tool_calls is empty.
    const toolCalls = snapshotToolCalls(message, true, partial);
    const chunks = snapshotToolRecords(message, 'tool_call_chunks');
    const invalid = snapshotToolRecords(message, 'invalid_tool_calls');
    Object.defineProperties(message, {
      tool_calls: {
        value: toolCalls,
        enumerable: true,
        writable: true,
        configurable: true,
      },
      tool_call_chunks: {
        value: chunks,
        enumerable: true,
        writable: true,
        configurable: true,
      },
      invalid_tool_calls: {
        value: invalid,
        enumerable: true,
        writable: true,
        configurable: true,
      },
    });
  } catch (error) {
    throw new InvalidModelToolCallError(
      error instanceof Error
        ? error.message
        : 'Accepted model response contains non-serializable tool calls'
    );
  }
}

/** Inspect original descriptors before a getter, proxy, or custom instance can
 * be normalized away. The bound applies to each model message independently.
 */
function snapshotToolCalls(
  finalResponse: AIMessageChunk,
  allowInvalidDiagnostics: boolean,
  allowFragmentArgs = false
): ToolCall[] {
  // Read own data descriptors, not accessors supplied by a custom model. Invalid
  // diagnostics are rejected in O(1); cloning them can run getters and bypass
  // the valid-call snapshot's byte/count limits.
  if (types.isProxy(finalResponse)) {
    throw new Error(
      'Accepted model response contains non-serializable tool calls'
    );
  }
  const calls = Object.getOwnPropertyDescriptor(finalResponse, 'tool_calls');
  const diagnostics = Object.getOwnPropertyDescriptor(
    finalResponse,
    'invalid_tool_calls'
  );
  const readArray = (descriptor: PropertyDescriptor | undefined): unknown[] => {
    if (descriptor === undefined) return [];
    if (!('value' in descriptor)) {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    const value: unknown = descriptor.value;
    if (value === undefined) return [];
    if (types.isProxy(value) || !Array.isArray(value)) {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    return value;
  };
  if (readArray(diagnostics).length > 0 && !allowInvalidDiagnostics) {
    throw new Error('Accepted model response contains invalid tool calls');
  }
  const source = readArray(calls);
  if (!allowInvalidDiagnostics && source.length > MAX_SNAPSHOT_CALLS) {
    throw new Error('Accepted model response exceeds snapshot limits');
  }
  const toolCalls: ToolCall[] = [];
  let remaining = allowInvalidDiagnostics ? Infinity : MAX_SNAPSHOT_BYTES;
  for (let index = 0; index < source.length; index++) {
    const entry = Object.getOwnPropertyDescriptor(source, String(index));
    if (entry == null || !('value' in entry) || entry.enumerable !== true) {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    const call: unknown = entry.value;
    if (call == null || typeof call !== 'object' || types.isProxy(call)) {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    const name = Object.getOwnPropertyDescriptor(call, 'name');
    const originalId = Object.getOwnPropertyDescriptor(call, 'id');
    const originalArgs = Object.getOwnPropertyDescriptor(call, 'args');
    if (
      name == null ||
      !('value' in name) ||
      typeof name.value !== 'string' ||
      originalArgs == null ||
      !('value' in originalArgs) ||
      (originalId != null && !('value' in originalId))
    ) {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    const providerId: unknown = originalId?.value;
    if (providerId !== undefined && typeof providerId !== 'string') {
      throw new Error(
        'Accepted model response contains non-serializable tool calls'
      );
    }
    let args: ToolCall['args'];
    if (allowInvalidDiagnostics) {
      // Callback streams may carry a not-yet-plannable argument string. It is
      // safe scalar data, but must not become an accepted executable call.
      args =
        allowFragmentArgs && typeof originalArgs.value === 'string'
          ? (originalArgs.value as unknown as ToolCall['args'])
          : cloneToolArguments(originalArgs.value);
    } else {
      remaining -=
        Buffer.byteLength(name.value, 'utf8') +
        (typeof providerId === 'string'
          ? Buffer.byteLength(providerId, 'utf8')
          : 0);
      const encoded = serializeToolArguments(originalArgs.value, remaining);
      remaining -= Buffer.byteLength(encoded, 'utf8');
      args = JSON.parse(encoded);
    }
    toolCalls.push({
      name: name.value,
      id: providerId,
      args,
      type: 'tool_call',
    });
  }
  return toolCalls;
}

/** Only the accepted final response becomes a host-visible model result. */
export function snapshotAcceptedModelResponse(
  finalResponse: AIMessageChunk,
  id: string,
  agentId: string
): ModelResponseEvent {
  return {
    type: 'model_response',
    id,
    agentId,
    ...(finalResponse.id != null ? { messageId: finalResponse.id } : {}),
    toolCalls: snapshotToolCalls(finalResponse, false),
    invalidToolCalls: [],
  };
}

/** These records contain only scalar fields, unlike parsed tool arguments. Never
 * spread or iterate provider records until their original descriptors pass.
 */
function snapshotToolRecords(
  message: AIMessageChunk,
  field: 'tool_call_chunks' | 'invalid_tool_calls'
): Record<string, unknown>[] {
  function invalid(): never {
    throw new Error(
      'Accepted model response contains non-serializable tool calls'
    );
  }
  const descriptor = Object.getOwnPropertyDescriptor(message, field);
  if (descriptor == null) return [];
  if (!('value' in descriptor)) invalid();
  const source: unknown = descriptor.value;
  if (source === undefined) return [];
  if (source == null || types.isProxy(source) || !Array.isArray(source))
    invalid();
  const result: Record<string, unknown>[] = [];
  for (let index = 0; index < source.length; index++) {
    const entry = Object.getOwnPropertyDescriptor(source, String(index));
    if (entry == null || !('value' in entry)) invalid();
    const record: unknown = entry.value;
    if (record == null || typeof record !== 'object' || types.isProxy(record))
      invalid();
    const copy: Record<string, unknown> = {};
    for (const key of Reflect.ownKeys(record)) {
      if (typeof key !== 'string') invalid();
      const property = Object.getOwnPropertyDescriptor(record, key);
      if (property == null || !('value' in property)) invalid();
      const value: unknown = property.value;
      if (
        value != null &&
        (key === 'index'
          ? typeof value !== 'number' ||
            !Number.isSafeInteger(value) ||
            value < 0
          : typeof value !== 'string')
      )
        invalid();
      Object.defineProperty(copy, key, {
        value,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    result.push(copy);
  }
  return result;
}

/** Providers can mutate and re-yield records. Inspect a per-emission snapshot,
 * leaving their originals intact, but keep producer/consumer charge identity.
 */
export function snapshotValidatedModelChunk(
  message: AIMessageChunk
): AIMessageChunk {
  if (types.isProxy(message)) {
    throw new InvalidModelToolCallError(
      'Accepted model response contains non-serializable tool calls'
    );
  }
  const descriptors = Object.getOwnPropertyDescriptors(message);
  for (const field of [
    'tool_calls',
    'tool_call_chunks',
    'invalid_tool_calls',
  ]) {
    if (Object.hasOwn(descriptors, field)) descriptors[field].configurable = true;
  }
  const copy = Object.create(
    Object.getPrototypeOf(message),
    descriptors
  ) as AIMessageChunk;
  detachValidatedModelToolCalls(copy, true);
  linkStreamLimitCanonical(copy, message);
  return copy;
}

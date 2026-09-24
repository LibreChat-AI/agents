import { types } from 'node:util';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ModelResponseEvent } from '@/types';
import { serializeToolArguments } from '@/utils/acceptedToolArguments';

const MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024;
const MAX_SNAPSHOT_CALLS = 1024;

/** Inspect the original descriptors before structuredClone can run a getter,
 * flatten an instance, or drop a symbol. Only graph-accepted calls are copied. */
export function snapshotAcceptedModelResponse(
  finalResponse: AIMessageChunk,
  id: string,
  agentId: string
): ModelResponseEvent {
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
  if (readArray(diagnostics).length > 0) {
    throw new Error('Accepted model response contains invalid tool calls');
  }
  const source = readArray(calls);
  if (source.length > MAX_SNAPSHOT_CALLS) {
    throw new Error('Accepted model response exceeds snapshot limits');
  }
  const toolCalls: ToolCall[] = [];
  let remaining = MAX_SNAPSHOT_BYTES;
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
    const identityBytes =
      Buffer.byteLength(name.value, 'utf8') +
      (typeof providerId === 'string'
        ? Buffer.byteLength(providerId, 'utf8')
        : 0);
    remaining -= identityBytes;
    const encoded = serializeToolArguments(originalArgs.value, remaining);
    remaining -= Buffer.byteLength(encoded, 'utf8');
    const args: ToolCall['args'] = JSON.parse(encoded);
    toolCalls.push({
      name: name.value,
      id: providerId,
      args,
      type: 'tool_call',
    });
  }
  return {
    type: 'model_response',
    id,
    agentId,
    toolCalls,
    invalidToolCalls: [],
  };
}

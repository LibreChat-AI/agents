import { AIMessageChunk } from '@langchain/core/messages';
import type { AIMessageChunkFields } from '@langchain/core/messages';

type ChunkConstructor = new (fields: AIMessageChunkFields) => AIMessageChunk;
const constructors = new WeakMap<object, ChunkConstructor>();

/** Internal certification for SDK-owned aggregation semantics, not arbitrary provider subclasses. */
export function registerProviderTextChunkConstructor(constructor: ChunkConstructor): void {
  constructors.set(constructor.prototype, constructor);
}

export function createProviderTextChunk(source: object, fields: AIMessageChunkFields): AIMessageChunk {
  const constructor = constructors.get(Object.getPrototypeOf(source)) ?? AIMessageChunk;
  return new constructor(fields);
}

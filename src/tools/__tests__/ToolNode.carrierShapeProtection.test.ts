import { z } from 'zod';
import { Command } from '@langchain/langgraph';
import { RunnableLambda } from '@langchain/core/runnables';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { ToolResultProtection } from '@/protection/toolResult';
import type { GenericTool } from '@/types';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { ToolNode } from '@/tools/ToolNode';

const raw = 'shape-carrier@example.invalid';
const selected = () => new ToolMessage({ name: 'lookup', tool_call_id: 'selected-call', content: raw });
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 1000, maxAttemptBytes: 65536, maxBufferedBytes: 65536, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const input = () => ({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'shape-call', name: 'other', args: {} }] })] });
const source = (func: () => Promise<unknown>) => new DynamicStructuredTool({ name: 'other', description: 'Allowed shape control.', schema: z.object({}), responseFormat: 'content_and_artifact', func });
async function invoke(tool: GenericTool, callback?: BaseCallbackHandler, configured = true) {
  const node = new ToolNode({ tools: [tool], trace: true, ...(configured ? { toolResultProtection: policy() } : {}) });
  return await node.invoke(input(), { callbacks: callback == null ? [] : [callback] }).catch((error: Error) => error) as { messages: ToolMessage[] } | Error;
}

const nativePrototypes = [
  { name: 'Buffer', prototype: Buffer.prototype, make: () => Buffer.from([1, 2]) },
  { name: 'Uint8Array', prototype: Uint8Array.prototype, make: () => new Uint8Array([1, 2]) },
  { name: 'TypedArray', prototype: Object.getPrototypeOf(Uint8Array.prototype) as object, make: () => new Uint8Array([1, 2]) },
  { name: 'ArrayBuffer', prototype: ArrayBuffer.prototype, make: () => new ArrayBuffer(2) },
  { name: 'DataView', prototype: DataView.prototype, make: () => new DataView(new ArrayBuffer(2)) },
  { name: 'URL', prototype: URL.prototype, make: () => new URL('https://example.invalid') },
  { name: 'URLSearchParams', prototype: URLSearchParams.prototype, make: () => new URL('https://example.invalid') },
  { name: 'Date', prototype: Date.prototype, make: () => new Date(0) },
];
describe.each(nativePrototypes)('$name prototype shape', ({ prototype, make }) => {
  it.each(['before', 'callback', 'getter', 'symbol'] as const)('rejects a %s prototype carrier', async (mode) => {
    const artifact = make(), message = selected(); let reads = 0; let observations = 0;
    const key = mode === 'symbol' ? Symbol('selected') : '_c1ArtifactCarrier';
    const expand = (): void => { Object.defineProperty(prototype, key, mode === 'getter' ? { configurable: true, get: () => { reads++; throw new Error(raw); } } : { configurable: true, value: message }); };
    const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (): void => { observations++; if (mode === 'callback') expand(); } }); callback.awaitHandlers = true;
    if (mode !== 'callback') expand();
    try {
      const outcome = await invoke(source(async () => ['Allowed result', artifact]), callback);
      expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(String(outcome)).not.toContain(raw); expect(reads).toBe(0); expect(observations).toBe(mode === 'callback' ? 1 : 0);
    } finally { Reflect.deleteProperty(prototype, key); }
  });
});

it.each(['message', 'command', 'plain', 'default-off'] as const)('gates non-native runnable tool callbacks (%s)', async (mode) => {
  let executions = 0; let observations = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleChainEnd: (): void => { observations++; } }); callback.awaitHandlers = true;
  const runnable = RunnableLambda.from(async () => { executions++; if (mode === 'plain') return 'Allowed unselected prose'; return mode === 'message' ? selected() : new Command({ update: { custom: { selected: selected() } } }); }).withConfig({ callbacks: [callback] });
  const tool = runnable.asTool({ name: 'other', description: 'Allowed runnable control.', schema: z.object({}) });
  const outcome = await invoke(tool, undefined, mode !== 'default-off');
  if (mode === 'default-off') { expect(executions).toBe(1); expect(observations).toBeGreaterThan(0); expect(outcome).not.toBeInstanceOf(Error); return; }
  expect(executions).toBe(0); expect(observations).toBe(0); expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(String(outcome)).not.toContain(raw);
});

it.each((['artifact', 'command', 'array'] as const).flatMap((carrier) => (['producer', 'callback', 'empty'] as const).map((mode) => [carrier, mode] as const)))('rejects function-valued %s carriers (%s)', async (carrier, mode) => {
  let observations = 0;
  const fn = (): string => 'Allowed function';
  if (mode === 'producer') Object.defineProperty(fn, 'selected', { value: selected() });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed function control.', schema: z.object({}), responseFormat: carrier === 'artifact' ? 'content_and_artifact' : 'content', func: async () => {
    if (carrier === 'artifact') return ['Allowed result', { fn }];
    return new Command({ update: { custom: carrier === 'array' ? [fn] : { fn } } });
  } });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (): void => { observations++; if (mode === 'callback') Object.defineProperty(fn, 'selected', { value: selected() }); } }); callback.awaitHandlers = true;
  const outcome = await invoke(tool, callback);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observations).toBe(0); expect(String(outcome)).not.toContain(raw);
});

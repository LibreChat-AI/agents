import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { ToolResultProtection } from '@/protection/toolResult';
import type { ToolExecuteBatchRequest } from '@/types';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { ToolNode } from '@/tools/ToolNode';
import { GraphEvents } from '@/common';

const raw = 'unreleased-artifact@example.invalid';
const opaqueArtifacts = [
  { name: 'Buffer', make: () => Buffer.from([1, 2, 3]) },
  { name: 'Uint8Array', make: () => new Uint8Array([1, 2, 3]) },
  { name: 'Uint16Array', make: () => new Uint16Array([1, 2, 3]) },
  { name: 'Int32Array', make: () => new Int32Array([1, 2, 3]) },
  { name: 'Float64Array', make: () => new Float64Array([1, 2, 3]) },
  { name: 'BigInt64Array', make: () => new BigInt64Array([1n, 2n, 3n]) },
  { name: 'ArrayBuffer', make: () => new ArrayBuffer(3) },
  { name: 'SharedArrayBuffer', make: () => new SharedArrayBuffer(3) },
  { name: 'DataView', make: () => new DataView(new ArrayBuffer(3)) },
  { name: 'URL', make: () => new URL('https://example.invalid/allowed') },
  { name: 'Date', make: () => new Date(0) },
];

type Route = 'direct' | 'host';
function policy(): ToolResultProtection {
  return { version: 1, toolNames: ['lookup'], timeoutMs: 1000, maxAttemptBytes: 65536, maxBufferedBytes: 65536, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) };
}
function input(name = 'other') {
  return { messages: [new AIMessage({ content: '', tool_calls: [{ id: 'artifact-call', name, args: {}, type: 'tool_call' }] })] };
}
function selectedMessage(): ToolMessage {
  return new ToolMessage({ name: 'lookup', tool_call_id: 'selected-call', content: raw });
}
async function invoke(artifact: object, route: Route, options: { name?: string; configured?: boolean; mutate?: () => void } = {}) {
  const { name = 'other', configured = true, mutate } = options;
  let executions = 0;
  const native: object[] = [];
  const completions: object[] = [];
  const node = new ToolNode({ trace: true, toolCallStepIds: new Map([['artifact-call', 'artifact-step']]), tools: [new DynamicStructuredTool({ name, description: 'Allowed artifact control.', schema: z.object({}), responseFormat: 'content_and_artifact', func: async () => { executions++; return ['Allowed unselected text', artifact]; } })], eventDrivenMode: route === 'host', ...(configured ? { toolResultProtection: policy() } : {}) });
  const callbacks = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { native.push((output as ToolMessage).artifact as object); mutate?.(); }, handleCustomEvent: (event, data): void => {
    if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(data as object);
    if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
    executions++;
    const request = data as ToolExecuteBatchRequest;
    request.resolve([{ toolCallId: request.toolCalls[0].id, status: 'success', content: 'Allowed unselected text', artifact }]);
  } }); callbacks.awaitHandlers = true;
  const result = await node.invoke(input(name), { callbacks: [callbacks] }).catch((error: Error) => error) as { messages: ToolMessage[] } | Error;
  return { result, native, completions, executions };
}

describe.each(opaqueArtifacts)('$name opaque artifacts', ({ make }) => {
  it.each((['direct', 'host'] as const).flatMap((route) => [false, true].map((nested) => [route, nested] as const)))('preserves unselected %s artifacts (nested=%s)', async (route, nested) => {
    const value = make(); const artifact = nested ? { data: value } : value;
    const { result, native, completions, executions } = await invoke(artifact, route);
    expect(result).not.toBeInstanceOf(Error); expect(executions).toBe(1); expect(completions).toHaveLength(1);
    const message = (result as { messages: ToolMessage[] }).messages[0];
    expect(message.content).toBe('Allowed unselected text'); expect(message.status).toBe('success'); expect(message.artifact).toBe(artifact);
    expect(message.lc_kwargs.artifact).toBe(artifact);
    if (route === 'direct') expect(native).toEqual([artifact]);
  });

  it.each(['direct', 'host'] as const)('keeps selected %s artifacts gated', async (route) => {
    const { result, native, completions } = await invoke(make(), route, { name: 'lookup' });
    expect(result).toBeInstanceOf(ToolResultProtectionError); expect(result).toMatchObject({ code: 'unsupported' }); expect(native).toEqual([]); expect(completions).toEqual([]);
  });

  it('keeps adjacent selected messages gated', async () => {
    const { result, native } = await invoke({ data: make(), selected: selectedMessage() }, 'direct');
    expect(result).toBeInstanceOf(ToolResultProtectionError); expect(native).toEqual([]); expect(String(result)).not.toContain(raw);
  });
});

it.each(['Buffer', 'Uint8Array', 'ArrayBuffer', 'DataView', 'URL', 'Date'] as const)('rejects %s carrier expandos before native release', async (name) => {
  const artifact = opaqueArtifacts.find((entry) => entry.name === name)!.make();
  Object.defineProperty(artifact, 'selected', { value: selectedMessage(), enumerable: true });
  const { result, native } = await invoke(artifact, 'direct');
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(native).toEqual([]); expect(String(result)).not.toContain(raw);
});

it.each(['own-getter', 'symbol', 'subclass', 'proxy', 'backing-buffer', 'late-property', 'late-prototype'] as const)('rejects binary %s carriers without invoking untrusted code', async (mode) => {
  let reads = 0; let artifact: object = Buffer.from([1, 2, 3]);
  if (mode === 'own-getter') Object.defineProperty(artifact, 'selected', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'symbol') Object.defineProperty(artifact, Symbol('selected'), { value: selectedMessage() });
  if (mode === 'subclass') { class CustomBytes extends Uint8Array {} artifact = new CustomBytes([1, 2, 3]); }
  if (mode === 'proxy') artifact = new Proxy(artifact, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } });
  if (mode === 'backing-buffer') { artifact = new Uint8Array(3); Object.defineProperty((artifact as Uint8Array).buffer, 'selected', { value: selectedMessage() }); }
  const mutate = (): void => {
    if (mode === 'late-property') Object.defineProperty(artifact, 'selected', { value: selectedMessage() });
    if (mode === 'late-prototype') Object.setPrototypeOf(artifact, { selected: selectedMessage() });
  };
  const { result, native } = await invoke(artifact, 'direct', { mutate });
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(String(result)).not.toContain(raw); expect(reads).toBe(0);
  expect(native).toHaveLength(mode.startsWith('late-') ? 1 : 0);
});

it('preserves default-off binary artifact behavior', async () => {
  const artifact = Buffer.from([1, 2, 3]); const { result } = await invoke(artifact, 'direct', { configured: false });
  expect((result as { messages: ToolMessage[] }).messages[0].artifact).toBe(artifact);
});

it.each(['message', 'getter', 'symbol', 'prototype', 'late', 'control'] as const)('validates reachable URL parameters before release (%s)', async (mode) => {
  let reads = 0;
  const artifact = new URL('https://example.invalid/allowed?a=1');
  const params = artifact.searchParams;
  if (mode === 'message') Object.defineProperty(params, 'selected', { value: selectedMessage() });
  if (mode === 'getter') Object.defineProperty(params, 'selected', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'symbol') Object.defineProperty(params, Symbol('selected'), { value: selectedMessage() });
  if (mode === 'prototype') Object.setPrototypeOf(params, { selected: selectedMessage() });
  const { result, native } = await invoke(artifact, 'direct', { mutate: () => { if (mode === 'late') Object.defineProperty(params, 'selected', { value: selectedMessage() }); } });
  expect(reads).toBe(0);
  if (mode === 'control') { expect((result as { messages: ToolMessage[] }).messages[0].artifact).toBe(artifact); expect(artifact.search).toBe('?a=1'); return; }
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(String(result)).not.toContain(raw); expect(native).toHaveLength(mode === 'late' ? 1 : 0);
});

it.each(['Buffer', 'Uint8Array'] as const)('rejects oversized %s before enumerating its indices', async (kind) => {
  const artifact = kind === 'Buffer' ? Buffer.alloc(4 * 1024 * 1024) : new Uint8Array(4 * 1024 * 1024);
  const keys = jest.spyOn(Reflect, 'ownKeys');
  try {
    const { result, native } = await invoke(artifact, 'direct');
    const inspected = keys.mock.calls.some(([value]) => value === artifact);
    expect(inspected).toBe(false); expect(result).toBeInstanceOf(ToolResultProtectionError); expect(result).toMatchObject({ code: 'overflow' }); expect(native).toEqual([]);
  } finally { keys.mockRestore(); }
});

it('bounds aggregate binary traversal across repeated artifact branches', async () => {
  const artifact = Array.from({ length: 33 }, () => new Uint8Array(1024));
  const { result, native } = await invoke(artifact, 'direct');
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(result).toMatchObject({ code: 'overflow' }); expect(native).toEqual([]);
});

it('does not enumerate the same binary twice while collecting and admitting a carrier', async () => {
  const value = new Uint8Array(1024); const artifact = { first: value, second: value };
  let enumerationCount = 0; let beforeNative = 0;
  const keys = Reflect.ownKeys;
  const spy = jest.spyOn(Reflect, 'ownKeys').mockImplementation((object) => { if (object === value) enumerationCount++; return keys(object); });
  try {
    const { result } = await invoke(artifact, 'direct', { mutate: () => { beforeNative = enumerationCount; } });
    expect(result).not.toBeInstanceOf(Error); expect(beforeNative).toBe(1);
  } finally { spy.mockRestore(); }
});

it.each(['ArrayBuffer', 'DataView'] as const)('preserves large index-free %s artifacts', async (kind) => {
  const buffer = new ArrayBuffer(4 * 1024 * 1024); const artifact = kind === 'DataView' ? new DataView(buffer) : buffer;
  const { result } = await invoke(artifact, 'direct'); expect((result as { messages: ToolMessage[] }).messages[0].artifact).toBe(artifact);
});

it('preserves the supported binary index limit without changing bytes', async () => {
  const artifact = new Uint8Array(32_768); artifact[0] = 42; artifact[32_767] = 7;
  const { result } = await invoke(artifact, 'direct');
  expect((result as { messages: ToolMessage[] }).messages[0].artifact).toBe(artifact); expect(artifact[0]).toBe(42); expect(artifact[32_767]).toBe(7);
});

it('does not apply the binary traversal limit when protection is off', async () => {
  const artifact = Buffer.alloc(4 * 1024 * 1024);
  const { result } = await invoke(artifact, 'direct', { configured: false });
  expect((result as { messages: ToolMessage[] }).messages[0].artifact).toBe(artifact);
});

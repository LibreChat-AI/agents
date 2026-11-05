import { z } from 'zod';
import { Command, Send } from '@langchain/langgraph';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, ToolMessage, HumanMessage, SystemMessage, ToolMessageChunk } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '@/protection/toolResult';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { Providers } from '@/common';

const raw = 'nested-carrier@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 1000, maxAttemptBytes: 65536, maxBufferedBytes: 65536, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = (): ToolMessage => new ToolMessage({ name: 'lookup', tool_call_id: 'selected-call', content: raw });
async function invoke(fn: () => unknown, callback?: (output: unknown) => void, configured = true, artifact = false) {
  let executions = 0; const observed: string[] = [];
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed nested carrier.', schema: z.object({}), responseFormat: artifact ? 'content_and_artifact' : 'content', func: async () => { executions++; const value = fn(); return artifact ? ['Allowed result', value] : value; } });
  const handler = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { observed.push(JSON.stringify(output)); callback?.(output); } }); handler.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, ...(configured ? { toolResultProtection: policy() } : {}) });
  const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [handler] }).catch((error: Error) => error) as { messages: ToolMessage[] } | Error;
  return { result, observed, executions };
}
const assertSafe = (value: unknown): void => { expect(JSON.stringify(value)).not.toContain(raw); expect(String(value)).not.toContain(raw); };

it.each(['__pregel_push', '__pregel_tasks'] as const)('rejects nested reserved Send field %s before native release', async (field) => {
  const packet = new Send('default', { messages: [], [field]: selected() });
  const graph = new StandardGraph({ runId: 'nested-push', agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
  await expect(graph.validateToolResultSend(packet)).rejects.toBeInstanceOf(ToolResultProtectionError);
  const { result, observed } = await invoke(() => new Command({ graph: Command.PARENT, goto: [packet] }));
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toEqual([]); assertSafe(result);
});

const fields = ['content', 'additional_kwargs', 'response_metadata', 'metadata', 'serialization', 'call-extra'] as const;
it.each(fields.flatMap((field) => ['producer', 'callback', 'late', 'control'].map((mode) => [field, mode] as const)))('guards native message %s descendants (%s)', async (field, mode) => {
  const container = { type: 'text', text: 'Allowed prose', nested: mode === 'producer' ? selected() : 'Allowed nested value' };
  const message = new ToolMessage({ name: 'other', tool_call_id: 'outer-call', content: field === 'content' ? [container] : 'Allowed prose' });
  if (field === 'serialization') message.lc_kwargs.custom = container;
  else if (field === 'call-extra') Object.defineProperty(message, 'tool_calls', { value: [{ id: 'other-call', name: 'other', args: { opaque: 42 }, extra: container }], enumerable: true, writable: true });
  else if (field !== 'content') Object.defineProperty(message, field, { value: { custom: container }, enumerable: true, configurable: true, writable: true });
  const { result, observed, executions } = await invoke(() => message, (output) => {
    if (mode !== 'callback') return;
    const current = output as ToolMessage & { tool_calls?: Array<{ extra: typeof container }> };
    let visible: typeof container;
    if (field === 'content') visible = (current.content as typeof container[])[0];
    else if (field === 'serialization') visible = current.lc_kwargs.custom as typeof container;
    else if (field === 'call-extra') visible = current.tool_calls![0].extra;
    else visible = (current[field] as { custom: typeof container }).custom;
    visible.nested = selected();
  });
  expect(executions).toBe(1); assertSafe(observed);
  if (mode === 'late') container.nested = selected(); assertSafe(result);
  if (mode === 'producer' || mode === 'callback') expect(result).toBeInstanceOf(ToolResultProtectionError); else expect(result).not.toBeInstanceOf(Error);
  if (mode === 'producer') expect(observed).toEqual([]);
});

it.each(['HumanMessage', 'AIMessage', 'SystemMessage', 'ToolMessageChunk'] as const)('guards nested content from native %s implementations', async (kind) => {
  const Message = { HumanMessage, AIMessage, SystemMessage, ToolMessageChunk }[kind];
  const message: BaseMessage = new Message({ content: [{ type: 'text', text: 'Allowed prose', nested: selected() }], tool_call_id: 'outer-call' });
  const { result, observed } = await invoke(() => message);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toEqual([]); assertSafe(result);
});

it('preserves opaque tool arguments while guarding non-argument message fields', async () => {
  const args = { opaque: selected() };
  const message = new AIMessage({ content: 'Allowed prose', tool_calls: [{ id: 'allowed-call', name: 'other', args }] });
  const { result } = await invoke(() => message);
  expect(result).not.toBeInstanceOf(Error); expect(message.tool_calls![0].args).toBe(args);
});

it.each(['inert', 'selected', 'callback', 'control-off'] as const)('bounds shared container traversal (%s)', async (mode) => {
  const leaf: { value: string } | { selected: ToolMessage } = mode === 'selected' ? { selected: selected() } : { value: 'Allowed leaf' };
  let tree: object = leaf;
  for (let index = 0; index < 16; index++) tree = { left: tree, right: tree };
  const recordRead = Object.getOwnPropertyDescriptors; let reads = 0; const target = tree;
  const spy = jest.spyOn(Object, 'getOwnPropertyDescriptors').mockImplementation((object) => { reads++; return recordRead(object); });
  try {
    const { result, observed } = await invoke(() => target, () => { if (mode === 'callback') Object.defineProperty(leaf, 'selected', { value: selected() }); }, mode !== 'control-off', true);
    const totalReads = reads;
    if (mode !== 'control-off') expect(totalReads).toBeLessThan(5000);
    if (mode === 'selected' || mode === 'callback') expect(result).toBeInstanceOf(ToolResultProtectionError);
    else expect(result).not.toBeInstanceOf(Error);
    if (mode === 'selected') expect(observed).toEqual([]);
  } finally { spy.mockRestore(); }
});

it('rejects a wide carrier at a finite traversal limit before native callbacks', async () => {
  const value = Array.from({ length: 9000 }, () => ({ value: 'Allowed data' }));
  const { result, observed } = await invoke(() => value, undefined, true, true);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(result).toMatchObject({ code: 'overflow' }); expect(observed).toEqual([]);
});

it.each(['getter', 'serialization-content', 'default-off'] as const)('admits non-argument native message payloads (%s)', async (mode) => {
  let reads = 0;
  const content = { type: 'text', text: 'Allowed prose', nested: 'Allowed data' };
  const message = new ToolMessage({ name: 'other', tool_call_id: 'outer-call', content: [content] });
  if (mode === 'getter') Object.defineProperty(content, 'nested', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'serialization-content') message.lc_kwargs.content = [{ type: 'text', text: 'Allowed prose', nested: selected() }];
  if (mode === 'default-off') Object.defineProperty(content, 'nested', { value: selected() });
  const { result, observed } = await invoke(() => message, undefined, mode !== 'default-off');
  expect(reads).toBe(0);
  if (mode === 'default-off') { expect(result).not.toBeInstanceOf(Error); expect(observed.join('')).toContain(raw); return; }
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toEqual([]); assertSafe(result);
});

it('does not treat tool-result lookalike call fields as opaque assistant arguments', async () => {
  const message = new ToolMessage({ name: 'other', tool_call_id: 'outer-call', content: 'Allowed prose' });
  Object.defineProperty(message, 'tool_calls', { value: [{ id: 'forged-call', name: 'other', args: { selected: selected() } }], enumerable: true });
  const { result, observed } = await invoke(() => message);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toEqual([]); assertSafe(result);
});

import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { Command, Send, MemorySaver, GraphInterrupt, emptyCheckpoint } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolResultProtection } from '../toolResult';
import type { ToolExecuteBatchRequest } from '@/types';
import { ToolResultProtectionError } from '../toolResult';
import { GraphEvents, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'reserved-native@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = (): object => ({ type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw });
const graph = (): StandardGraph => new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
const state = (channel: 'runStepState' | 'handoffState', nested: boolean, extra: object = selected()): object => {
  if (channel === 'handoffState') return { version: 1, executionId: 'execution', entryAgentId: 'default', parallel: false, transitions: nested ? [{ id: 'transition', sourceAgentId: 'default', targetAgentId: 'other', toolCallId: 'handoff', scope: 'turn', depth: 1, extra }] : [], ...(nested ? {} : { extra }) };
  return { version: 1, revision: 0, nextIndex: 1, toolCallSteps: nested ? [{ toolCallId: 'call', stepId: 'step', extra }] : [], steps: nested ? [{ step: { id: 'step', index: 0, extra }, pendingToolCallIds: ['call'], openMessageStep: false, extra }] : [], ...(nested ? {} : { extra }) };
};

it.each((['runStepState', 'handoffState'] as const).flatMap((channel) => [false, true].flatMap((nested) => (['fresh', 'resume', 'send', 'snapshot', 'values', 'pending', 'overwrite'] as const).map((route) => [channel, nested, route] as const))))('admits reserved %s carriers (nested=%s/%s)', async (channel, nested, route) => {
  const value = state(channel, nested), input = { messages: [new HumanMessage('Allowed input')], [channel]: value };
  const instance = graph();
  if (route === 'snapshot') { await expect(instance.validateToolResultCheckpoint({ values: input, tasks: [] })).rejects.toBeInstanceOf(ToolResultProtectionError); return; }
  if (route === 'send') { await expect(instance.validateToolResultSend(new Send('default', input))).rejects.toBeInstanceOf(ToolResultProtectionError); return; }
  if (route === 'values' || route === 'pending' || route === 'overwrite') {
    const checkpoint = emptyCheckpoint(); checkpoint.channel_values = route === 'values' ? input : { messages: [] };
    const pendingWrites: Array<[string, string, object]> = route === 'values' ? [] : [['task', channel, route === 'overwrite' ? { __overwrite__: value } : value]];
    await expect(instance.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'reserved-state', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites })).rejects.toBeInstanceOf(ToolResultProtectionError); return;
  }
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const runId = `reserved-${channel}-${nested}-${route}`;
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, ...(route === 'resume' ? { compileOptions: { checkpointer: new MemorySaver() } } : {}), tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed' })] }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed final'], toolCalls: [{ id: 'approval', name: 'lookup', args: {} }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  if (route === 'resume') { await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config); expect(run.getInterrupt()).toBeDefined(); }
  let observed = 0; const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const options = { ...config, callbacks: [callback] };
  const result = await (route === 'resume' ? run.resume([{ type: 'approve' }], options, undefined, { update: [[channel, value]] }) : run.processStream(input, options)).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); expect(String(result)).not.toContain(raw);
});

const natives = [
  { name: 'Buffer', make: () => Buffer.from(new Uint8Array([1, 2, 3]).buffer) },
  ...[Uint8Array, Uint8ClampedArray, Int8Array, Uint16Array, Int16Array, Uint32Array, Int32Array, Float32Array, Float64Array].map((Type) => ({ name: Type.name, make: () => new Type([1, 2, 3]) })),
  ...[BigInt64Array, BigUint64Array].map((Type) => ({ name: Type.name, make: () => new Type([1n, 2n, 3n]) })),
  { name: 'ArrayBuffer', make: () => new ArrayBuffer(3) },
  { name: 'SharedArrayBuffer', make: () => new SharedArrayBuffer(3) },
  { name: 'DataView', make: () => new DataView(new ArrayBuffer(8), 2, 3) },
  { name: 'URL', make: () => new URL('https://example.invalid/allowed?a=1') },
  { name: 'Date', make: () => new Date(123) },
];

it.each(natives.flatMap(({ name, make }) => (['native', 'host', 'history', 'command'] as const).map((route) => [name, route, make] as const)))('detaches %s native slots and callback-held leaves (%s)', async (_name, route, make) => {
  const source = make(); let retained: object | undefined;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), responseFormat: route === 'command' ? 'content' : 'content_and_artifact', func: async () => route === 'command' ? new Command({ update: { custom: { artifact: source } } }) : ['Allowed', source] });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (route !== 'command') retained = (output as ToolMessage).artifact as object; }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) { const request = data as ToolExecuteBatchRequest; request.resolve([{ toolCallId: 'outer', status: 'success', content: 'Allowed', artifact: source }]); } } }); callback.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, eventDrivenMode: route === 'host', toolResultProtection: policy() });
  const result = route === 'history' ? await graph().validateToolResultHistory([new ToolMessage({ name: 'other', tool_call_id: 'outer', content: 'Allowed', artifact: source })]) : await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] });
  const artifact = route === 'command' ? ((result as Command[])[0].update as Array<[string, { artifact: object }]>)[0][1].artifact : (Array.isArray(result) ? result : (result as { messages: ToolMessage[] }).messages)[0].artifact as object;
  expect(artifact).toEqual(source); expect(Object.getPrototypeOf(artifact)).toBe(Object.getPrototypeOf(source));
  const serialize = (): string => JSON.stringify(artifact, (_key, value: unknown) => typeof value === 'bigint' ? String(value) : value);
  const before = serialize();
  for (const alias of new Set([source, retained])) {
    if (alias == null) continue;
    Object.defineProperty(alias, 'selected', { value: selected(), enumerable: true });
    if (ArrayBuffer.isView(alias)) { Object.defineProperty(alias.buffer, 'selected', { value: selected(), enumerable: true, configurable: true }); new Uint8Array(alias.buffer).fill(42); }
    if (alias instanceof ArrayBuffer || alias instanceof SharedArrayBuffer) new Uint8Array(alias).fill(42);
    if (alias instanceof Date) alias.setTime(0);
    if (alias instanceof URL) { alias.searchParams.set('changed', raw); Object.defineProperty(alias.searchParams, 'selected', { value: selected(), enumerable: true }); }
  }
  expect(serialize()).toBe(before); expect(artifact).not.toBe(source); if (retained != null) expect(artifact).not.toBe(retained);
  expect(Object.hasOwn(artifact, 'selected')).toBe(false);
  if (ArrayBuffer.isView(artifact)) { expect(artifact.buffer).not.toBe((source as ArrayBufferView).buffer); expect(Object.hasOwn(artifact.buffer, 'selected')).toBe(false); expect([...new Uint8Array(artifact.buffer, artifact.byteOffset, artifact.byteLength)]).not.toContain(42); }
  if (artifact instanceof ArrayBuffer || artifact instanceof SharedArrayBuffer) expect([...new Uint8Array(artifact)]).toEqual([0, 0, 0]);
  expect(serialize()).not.toContain(raw);
});

it('retains native view offsets and shared detached backing topology', async () => {
  const buffer = new ArrayBuffer(12), bytes = new Uint8Array(buffer, 2, 4), view = new DataView(buffer, 4, 3); bytes.set([1, 2, 3, 4]);
  const source = { buffer, bytes, view, repeated: bytes };
  const [message] = await graph().validateToolResultHistory([new ToolMessage({ name: 'other', tool_call_id: 'outer', content: 'Allowed', artifact: source })]);
  const artifact = (message as ToolMessage).artifact as typeof source;
  expect(artifact.buffer).not.toBe(buffer); expect(artifact.bytes.buffer).toBe(artifact.buffer); expect(artifact.view.buffer).toBe(artifact.buffer); expect(artifact.bytes).toBe(artifact.repeated);
  expect(artifact.bytes.byteOffset).toBe(2); expect(artifact.view.byteOffset).toBe(4); expect(artifact.view.byteLength).toBe(3); expect([...artifact.bytes]).toEqual([1, 2, 3, 4]);
});

it.each(['runStepState', 'handoffState'] as const)('retains ordinary %s extras with isolated containers', async (channel) => {
  const extra = { principal: 'Allowed principal', resource: { id: 'Allowed resource' } }, value = state(channel, true, extra);
  const result = await graph().validateToolResultCheckpoint({ values: { messages: [], [channel]: value }, tasks: [] });
  expect(result.values[channel]).toEqual(value); expect(result.values[channel]).not.toBe(value); extra.resource.id = raw; expect(JSON.stringify(result)).not.toContain(raw);
});

it.each(['one', 'aggregate', 'capacity'] as const)('bounds native copy allocation before release (%s)', async (mode) => {
  const BufferType: new (length: number, options?: { maxByteLength: number }) => ArrayBuffer = ArrayBuffer;
  const artifact = mode === 'aggregate' ? [new ArrayBuffer(9 * 1024 * 1024), new ArrayBuffer(9 * 1024 * 1024)] : new BufferType(mode === 'one' ? 17 * 1024 * 1024 : 0, mode === 'capacity' ? { maxByteLength: 17 * 1024 * 1024 } : undefined);
  await expect(graph().validateToolResultHistory([new ToolMessage({ name: 'other', tool_call_id: 'outer', content: 'Allowed', artifact })])).rejects.toMatchObject({ code: 'overflow' });
});

it.each([ArrayBuffer, SharedArrayBuffer])('isolates resizable native buffer capacity (%s)', async (Type) => {
  const BufferType: new (length: number, options?: { maxByteLength: number }) => ArrayBuffer | SharedArrayBuffer = Type;
  const source = new BufferType(3, { maxByteLength: 10 }); new Uint8Array(source).set([1, 2, 3]);
  const [message] = await graph().validateToolResultHistory([new ToolMessage({ name: 'other', tool_call_id: 'outer', content: 'Allowed', artifact: source })]);
  const returned = (message as ToolMessage).artifact as ArrayBuffer & { maxByteLength: number };
  expect(returned).not.toBe(source); expect(returned.maxByteLength).toBe(10); expect([...new Uint8Array(returned)]).toEqual([1, 2, 3]);
});

it.each((['runStepState', 'handoffState'] as const).flatMap((channel) => (['values', 'pending'] as const).map((route) => [channel, route] as const)))('admits real saver %s payloads before native observers (%s)', async (channel, route) => {
  let tamper = false, observed = 0;
  class Saver extends MemorySaver { override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
    const tuple = await super.getTuple(config); if (tuple == null || !tamper) return tuple;
    if (route === 'values') tuple.checkpoint.channel_values[channel] = state(channel, true); else tuple.pendingWrites = [['task', channel, state(channel, true)]];
    return tuple;
  } }
  const saver = new Saver(), runId = `reserved-saver-${channel}-${route}`;
  const create = () => Run.create({ runId, graphConfig: { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } }; await first.processStream({ messages: [new HumanMessage('Allowed input')] }, config);
  tamper = true; const next = await create(); next.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await next.processStream({ messages: [new HumanMessage('Allowed next input')] }, { ...config, callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); expect(String(result)).not.toContain(raw);
});

it.each(['runStepState', 'handoffState'] as const)('leaves default-off fresh %s behavior unchanged', async (channel) => {
  const value = state(channel, false), input = { messages: [new HumanMessage('Allowed input')], [channel]: value };
  const run = await Run.create({ runId: 'default-off-reserved', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] }); const stream = jest.spyOn(run.graphRunnable!, 'streamEvents');
  await run.processStream(input, { version: 'v2' }); expect((stream.mock.calls[0][0] as typeof input)[channel]).toBe(value);
});

it.each((['runStepState', 'handoffState'] as const).flatMap((channel) => (['native', 'wide', 'getter', 'proxy', 'function'] as const).map((mode) => [channel, mode] as const)))('fails closed on unsupported %s extras (%s)', async (channel, mode) => {
  let reads = 0; let extra: object = new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw });
  if (mode === 'wide') extra = Array.from({ length: 9000 }, () => ({ allowed: 42 }));
  if (mode === 'getter') extra = Object.defineProperty({}, 'selected', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'proxy') extra = new Proxy({}, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } });
  if (mode === 'function') extra = { selected };
  const result = await graph().validateToolResultCheckpoint({ values: { messages: [], [channel]: state(channel, false, extra) }, tasks: [] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); expect(String(result)).not.toContain(raw);
});

it('admits reserved replay wrappers before native error observers', async () => {
  const payload = { __librechat_run_step_resume_wrapper: 1, __librechat_run_step_resume_state: state('runStepState', true), __librechat_run_step_resume_payload: 'Allowed' };
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw new GraphInterrupt([{ value: payload, id: 'interrupt' }]); } });
  const observed: Error[] = []; const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { observed.push(error); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [tool], toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toHaveLength(1); expect(observed[0]).toBeInstanceOf(ToolResultProtectionError); expect(JSON.stringify(observed)).not.toContain(raw); expect(String(result)).not.toContain(raw);
});

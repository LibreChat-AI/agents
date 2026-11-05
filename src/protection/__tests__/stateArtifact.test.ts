import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { Command, Send, MemorySaver, emptyCheckpoint } from '@langchain/langgraph';
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

const raw = 'state-artifact@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = (native = false): object => native ? new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }) : { type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw };
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };

it.each(['fresh', 'resume', 'send', 'values', 'pending', 'snapshot'] as const)('admits custom state descendants before observers (%s)', async (carrier) => {
  let observed = 0;
  const custom = { nested: { kwargs: selected() } };
  const instance = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
  if (carrier === 'snapshot') { await expect(instance.validateToolResultCheckpoint({ values: { messages: [], custom }, tasks: [] })).rejects.toBeInstanceOf(ToolResultProtectionError); return; }
  if (carrier === 'values' || carrier === 'pending') {
    const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [], ...(carrier === 'values' ? { custom } : {}) };
    await expect(instance.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'custom-state', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: carrier === 'pending' ? [['task', 'custom', custom]] : [] })).rejects.toBeInstanceOf(ToolResultProtectionError); return;
  }
  const runId = 'custom-state-' + carrier, hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: new MemorySaver() }, tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed control', schema: z.object({}), func: async () => 'Allowed output' })] }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed final'], toolCalls: [{ id: 'approval', name: 'lookup', args: {} }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  if (carrier !== 'fresh') { await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config); expect(run.getInterrupt()).toBeDefined(); }
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const options = { ...config, callbacks: [callback] };
  const invoke = async (): Promise<unknown> => {
    if (carrier === 'resume') return run.resume([{ type: 'approve' }], options, undefined, { update: [['custom', custom]] });
    if (carrier === 'send') return run.resume([{ type: 'approve' }], options, undefined, { goto: [new Send('default', { messages: [new HumanMessage('Allowed input')], custom })] });
    const state = { messages: [new HumanMessage('Allowed input')], custom };
    return run.processStream(state, options);
  };
  const result = await invoke().catch((error: Error) => error); expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); safe(result);
});

it.each(['custom', 'authority'] as const)('preserves ordinary %s data while detaching its containers', async (field) => {
  const original = { principal: 'Allowed principal', resources: [{ id: 'Allowed resource' }] };
  const graph = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
  const result = await graph.validateToolResultCheckpoint({ values: { messages: [], [field]: original }, tasks: [] });
  expect(result.values[field]).toEqual(original); expect(result.values[field]).not.toBe(original); original.resources[0].id = raw; safe(result);
});

it.each((['native', 'host', 'command', 'history'] as const).flatMap((route) => (['record', 'array', 'shared'] as const).flatMap((shape) => [false, true].map((native) => [route, shape, native] as const))))('detaches mutable artifact containers after release (%s/%s/native=%s)', async (route, shape, native) => {
  const leaf: { value: string; selected?: object } = { value: 'Allowed artifact' };
  let source: object = { nested: leaf };
  if (shape === 'array') source = [leaf];
  if (shape === 'shared') source = { first: leaf, second: leaf };
  let retained: typeof source | undefined, executions = 0;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed artifact', schema: z.object({}), responseFormat: route === 'command' ? 'content' : 'content_and_artifact', func: async () => { executions++; return route === 'command' ? new Command({ update: { custom: { artifact: source } } }) : ['Allowed output', source]; } });
  const node = new ToolNode({ tools: [tool], trace: true, eventDrivenMode: route === 'host', toolResultProtection: policy() });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { if (route !== 'command') retained = (output as ToolMessage).artifact as typeof source; }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) { executions++; const request = data as ToolExecuteBatchRequest; request.resolve([{ toolCallId: 'outer', status: 'success', content: 'Allowed output', artifact: source }]); } } }); callback.awaitHandlers = true;
  let result: unknown;
  if (route === 'history') { const graph = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() }); result = await graph.validateToolResultHistory([new ToolMessage({ name: 'other', tool_call_id: 'outer', content: 'Allowed output', artifact: source })]); }
  else result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] });
  const before = JSON.stringify(result); leaf.selected = selected(native); leaf.value = raw;
  if (retained != null) Object.defineProperty(retained, 'lateSelected', { value: selected(native), enumerable: true });
  expect(JSON.stringify(result)).toBe(before); safe(result); if (route !== 'history') expect(executions).toBe(1);
});

it('keeps default-off late artifact alias behavior unchanged', async () => {
  const artifact = { nested: { value: 'Allowed artifact' } };
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), responseFormat: 'content_and_artifact', func: async () => ['Allowed output', artifact] });
  const result = await new ToolNode({ tools: [tool] }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }) as { messages: ToolMessage[] };
  expect(result.messages[0].artifact).toBe(artifact); artifact.nested.value = raw; expect(JSON.stringify(result)).toContain(raw);
});

it.each(['values', 'pending'] as const)('rejects serialized custom channels in a real saver before callbacks (%s)', async (route) => {
  let tamper = false, observed = 0;
  class Saver extends MemorySaver { override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
    const tuple = await super.getTuple(config); if (tuple == null || !tamper) return tuple;
    const custom = { nested: selected() };
    if (route === 'values') tuple.checkpoint.channel_values.custom = custom; else tuple.pendingWrites = [['task', 'custom', custom]];
    return tuple;
  } }
  const saver = new Saver(), runId = 'custom-saver-' + route;
  const create = () => Run.create({ runId, graphConfig: { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } }; await first.processStream({ messages: [new HumanMessage('Allowed input')] }, config);
  tamper = true; const next = await create(); next.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await next.processStream({ messages: [new HumanMessage('Allowed next input')] }, { ...config, callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); safe(result);
});

it('retains shared-data topology and inert binary slots without retaining plain containers', async () => {
  const data = { value: 'Allowed' }, bytes = Buffer.from([1, 2, 3]); const artifact = { first: data, second: data, bytes };
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), responseFormat: 'content_and_artifact', func: async () => ['Allowed output', artifact] });
  const result = await new ToolNode({ tools: [tool], toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }) as { messages: ToolMessage[] };
  const returned = result.messages[0].artifact as typeof artifact;
  expect(returned).toEqual(artifact); expect(returned).not.toBe(artifact); expect(returned.first).not.toBe(data); expect(returned.first).toBe(returned.second); expect(returned.bytes).toEqual(bytes); expect(returned.bytes).not.toBe(bytes); expect(result.messages[0].lc_kwargs.artifact).toBe(returned);
});

it.each(['native-message', 'getter', 'proxy', 'function', 'wide'] as const)('rejects unsupported custom-state %s without reading hostile behavior', async (mode) => {
  let reads = 0;
  let custom: object = { nested: selected(true) };
  if (mode === 'getter') custom = Object.defineProperty({}, 'nested', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'proxy') custom = new Proxy({}, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } });
  if (mode === 'function') custom = { nested: (): object => selected() };
  if (mode === 'wide') custom = Array.from({ length: 9000 }, () => ({ allowed: 42 }));
  const graph = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
  const result = await graph.validateToolResultCheckpoint({ values: { messages: [], custom }, tasks: [] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); safe(result);
  if (mode === 'wide') expect(result).toMatchObject({ code: 'overflow' });
});

it('leaves default-off fresh custom state unchanged', async () => {
  const custom = { nested: selected() };
  const run = await Run.create({ runId: 'default-off-custom', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] });
  const stream = jest.spyOn(run.graphRunnable!, 'streamEvents');
  const state = { messages: [new HumanMessage('Allowed input')], custom };
  await run.processStream(state, { version: 'v2' });
  expect((stream.mock.calls[0][0] as typeof state).custom).toBe(custom);
});

import { z } from 'zod';
import { writer as writeCustom } from '@langchain/langgraph';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '../toolResult';
import type { ToolExecuteBatchRequest } from '@/types';
import { Constants, GraphEvents, Providers } from '@/common';
import { ToolResultProtectionError } from '../toolResult';
import { ChatModelStreamHandler } from '@/stream';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'release-sinks@example.invalid';
const selected = (): ToolMessage => new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw });
const policy = (block = false): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => block ? { version: 1, ok: false, error: { code: 'blocked' } } : { version: 1, ok: true, value: { content, replacements: 0, categories: [] } } });
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };
type WriterConfig = RunnableConfig & { writer?: (data: unknown) => void; config?: WriterConfig };

it.each([false, true].flatMap((configured) => [false, true].flatMap((selectedTool) => (['runtime', 'nested', 'fallback', 'ambient'] as const).map((route) => [configured, selectedTool, route] as const))))('quarantines tool writers (configured=%s/selected=%s/%s)', async (configured, selectedTool, route) => {
  const observed: unknown[] = [], authority = { principal: 'Allowed', resource: 'Allowed' }; let executions = 0;
  const writer = (data: unknown): void => { observed.push(data); };
  const name = selectedTool ? 'lookup' : 'other';
  const tool = new DynamicStructuredTool({ name, description: 'Allowed', schema: z.object({}), func: async (_args, _manager, options) => {
    executions++; const runtime = options as WriterConfig; expect(runtime.configurable?.authority).toBe(authority);
    if (route === 'runtime') runtime.writer?.(raw);
    if (route === 'nested') runtime.config?.writer?.(raw);
    if (route === 'fallback') (runtime.configurable?.writer as typeof writer)(selected());
    if (route === 'ambient') writeCustom(selected());
    return raw;
  } });
  const config: WriterConfig = { writer, configurable: { writer, authority } };
  const result = await new ToolNode({ tools: [tool], ...(configured ? { toolResultProtection: policy(true) } : {}) }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name, id: 'outer', args: {} }] })] }, config).catch((error: Error) => error);
  expect(executions).toBe(1);
  if (!configured) { expect(observed).toHaveLength(route === 'ambient' ? 0 : 1); return; }
  expect(observed).toEqual([]); if (selectedTool) expect(result).toBeInstanceOf(ToolResultProtectionError);
});

it.each((['cause', 'field', 'getter', 'control', 'late', 'default-off'] as const).map((mode) => [mode] as const))('admits unselected host rejections before error observers (%s)', async (mode) => {
  let reads = 0; const error = Object.assign(new Error('Allowed diagnostic'), { detail: { count: 42 } }), observed: string[] = []; let retained: Error | undefined;
  if (mode === 'cause') Object.defineProperty(error, 'cause', { value: selected() });
  if (mode === 'field' || mode === 'default-off') Object.assign(error, { result: selected() });
  if (mode === 'getter') Object.defineProperty(error, 'result', { get: () => { reads++; throw new Error(raw); } });
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => 'Must not run' });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, request): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) (request as ToolExecuteBatchRequest).reject(error); }, handleChainError: (error): void => { retained = error; observed.push(JSON.stringify(error)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [source], trace: true, eventDrivenMode: true, ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error) as Error & { detail: typeof error.detail };
  if (mode === 'default-off') { expect(result).toBe(error); expect(observed.join('')).toContain(raw); return; }
  expect(reads).toBe(0); safe(result); safe(observed);
  if (mode === 'control' || mode === 'late') { expect(result.message).toBe(error.message); expect(result).not.toBe(error); expect(result).not.toBe(retained); error.detail.count = 99; Object.defineProperty(retained!, 'result', { value: selected(), enumerable: true }); expect(result.detail.count).toBe(42); safe(result); }
  else expect(result).toBeInstanceOf(ToolResultProtectionError);
});

it.each([false, true])('gates enriched host rejections in real Run (eager=%s)', async (eager) => {
  const observed: string[] = [];
  const run = await Run.create({ runId: 'host-reject-' + eager, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, toolDefinitions: [{ name: 'other', description: 'Allowed', parameters: { type: 'object', properties: {} } }] }, eagerEventToolExecution: { enabled: eager }, toolResultProtection: policy(), customHandlers: { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(), [GraphEvents.ON_TOOL_EXECUTE]: { handle: (_event, request): void => { (request as ToolExecuteBatchRequest).reject(Object.assign(new Error('Allowed diagnostic'), { cause: selected() })); } } } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not run'], toolCalls: [{ name: 'other', id: 'outer', args: {} }] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainError: (error): void => { observed.push(JSON.stringify(error)); } }); callback.awaitHandlers = true;
  const result = await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2', callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result); safe(observed);
});

it.each((['Stop', 'StopFinalize'] as const).flatMap((stage) => (['content', 'nested', 'late', 'control', 'default-off', 'continuation'] as const).map((mode) => [stage, mode] as const)))('isolates terminal %s hook carriers (%s)', async (stage, mode) => {
  const hooks = new HookRegistry(), final: string[] = []; let held: BaseMessage[] | undefined, stops = 0;
  hooks.register('Stop', { hooks: [async (input) => {
    stops++; if (stage === 'Stop') { held = input.messages; const message = input.messages.find((m) => m.type === 'tool')! as ToolMessage; if (mode === 'content' || mode === 'default-off') { message.content = raw; message.lc_kwargs.content = raw; } if (mode === 'nested' || mode === 'continuation') message.additional_kwargs.selected = selected(); }
    return mode === 'continuation' && stops === 1 ? { decision: 'block', injectedMessages: [{ role: 'user', content: 'Allowed continuation' }] } : {};
  }] });
  hooks.register('StopFinalize', { hooks: [async (input) => {
    final.push(JSON.stringify(input.messages)); if (stage === 'StopFinalize') { held = input.messages; const message = input.messages.find((m) => m.type === 'tool')! as ToolMessage; if (mode === 'content' || mode === 'default-off') { message.content = raw; message.lc_kwargs.content = raw; } if (mode === 'nested' || mode === 'continuation') message.additional_kwargs.selected = selected(); } return {};
  }] });
  const tool = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed result' });
  const run = await Run.create({ runId: 'terminal-' + stage + mode, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [tool] }, hooks, ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed final', 'Allowed continued'], toolCalls: [{ name: 'lookup', id: 'outer', args: {} }] });
  const result = await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2' }).catch((error: Error) => error);
  if (mode === 'default-off') { expect(JSON.stringify(run.getRunMessages())).toContain(raw); return; }
  safe(final); safe(run.getRunMessages()); safe(result);
  if (mode === 'content' || mode === 'nested' || mode === 'continuation') expect(result).toBeInstanceOf(ToolResultProtectionError);
  else { expect(result).not.toBeInstanceOf(Error); const before = JSON.stringify(run.getRunMessages()); const message = held!.find((m) => m.type === 'tool')! as ToolMessage; message.content = raw; message.additional_kwargs.selected = selected(); expect(JSON.stringify(run.getRunMessages())).toBe(before); }
});

it.each(['throw', 'chain-mutation', 'proxy', 'wide'] as const)('gates alternate policy-bound rejection paths (%s)', async (mode) => {
  const observed: string[] = []; const error = Object.assign(new Error('Allowed diagnostic'), { detail: { count: 42 } }); let reads = 0;
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => 'Must not run' });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, value): void => {
    if (event !== GraphEvents.ON_TOOL_EXECUTE) return;
    if (mode === 'throw') throw Object.assign(error, { cause: selected() });
    if (mode === 'proxy') { (value as ToolExecuteBatchRequest).reject(new Proxy(error, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } })); return; }
    if (mode === 'wide') Object.assign(error, { result: Array.from({ length: 9000 }, () => ({ count: 42 })) });
    (value as ToolExecuteBatchRequest).reject(error);
  }, handleChainError: (error): void => { observed.push(JSON.stringify(error)); if (mode === 'chain-mutation') Object.assign(error, { result: selected() }); } }); callback.awaitHandlers = true; callback.raiseError = true;
  const result = await new ToolNode({ tools: [source], trace: true, eventDrivenMode: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); safe(result); safe(observed);
});

it.each(['Stop', 'StopFinalize'] as const)('gates %s continuation descendants before the next model turn', async (stage) => {
  const hooks = new HookRegistry(); let models = 0;
  hooks.register(stage, { hooks: [async () => ({ decision: 'block', injectedMessages: [{ role: 'user', content: [{ type: 'text', text: 'Allowed', nested: selected() }] }] })] });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { models++; yield* super._streamResponseChunks(...args); } }
  const run = await Run.create({ runId: 'terminal-injected-' + stage, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } }, hooks, toolResultProtection: policy() }); run.Graph!.overrideModel = new Model({ responses: ['Allowed', 'Must not run'] });
  const result = await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2' }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(models).toBe(1); safe(result); safe(run.getRunMessages());
});

it.each(['mutation', 'late', 'control'] as const)('isolates StopFailure message descendants (%s)', async (mode) => {
  const hooks = new HookRegistry(); let held: BaseMessage | undefined;
  hooks.register('StopFailure', { hooks: [async (input) => { held = input.lastAssistantMessage; expect(held).toBeDefined(); if (mode === 'mutation') held!.additional_kwargs.selected = selected(); return {}; }] });
  hooks.register('StopFinalize', { hooks: [async () => { throw new Error('Allowed terminal error'); }] });
  const run = await Run.create({ runId: 'stop-failure-' + mode, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } }, hooks, toolResultProtection: policy() }); run.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed reply'] });
  const result = await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2' }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(Error); expect(held).toBeDefined(); if (mode === 'mutation') expect(result).toBeInstanceOf(ToolResultProtectionError); if (held != null) held.additional_kwargs.selected = selected(); safe(result); safe(run.getRunMessages());
});

it.each(['mutation', 'late', 'control'] as const)('isolates foreground SubagentStop messages (%s)', async (mode) => {
  const hooks = new HookRegistry(); let held: BaseMessage[] | undefined;
  hooks.register('SubagentStop', { hooks: [async (input) => { held = input.messages; const message = input.messages.find((m) => m.type === 'tool')! as ToolMessage; if (mode === 'mutation') message.additional_kwargs.selected = selected(); return {}; }] });
  const source = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed result' });
  const run = await Run.create({ runId: 'subagent-stop-' + mode, graphConfig: { type: 'standard', agents: [{ agentId: 'parent', provider: Providers.OPENAI, subagentConfigs: [{ type: 'worker', name: 'Worker', description: 'Allowed', agentInputs: { agentId: 'child', provider: Providers.OPENAI, tools: [source] } }] }] }, hooks, toolResultProtection: policy() });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed parent answer'], toolCalls: [{ name: Constants.SUBAGENT, id: 'child', args: { description: 'Allowed task', subagent_type: 'worker' } }] });
  run.Graph!.setSubagentModelOverride(new FakeChatModel({ responses: ['', 'Allowed child answer'], toolCalls: [{ name: 'lookup', id: 'lookup', args: {} }] }));
  const result = await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2' }).catch((error: Error) => error);
  expect(held).toBeDefined(); if (mode === 'mutation') expect(result).toBeInstanceOf(ToolResultProtectionError); else expect(result).not.toBeInstanceOf(Error); if (held != null) held[0].additional_kwargs.selected = selected(); safe(result); safe(run.getRunMessages());
});

it('detaches public Run message views without poisoning release certificates', async () => {
  const source = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed result' });
  const run = await Run.create({ runId: 'public-message-alias', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [source] }, toolResultProtection: policy() }); run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed answer'], toolCalls: [{ name: 'lookup', id: 'outer', args: {} }] });
  await run.processStream({ messages: [new HumanMessage('Allowed')] }, { version: 'v2' });
  const before = JSON.stringify(run.getRunMessages()), message = run.getRunMessages()!.find((m) => m.type === 'tool')! as ToolMessage; message.content = raw; message.lc_kwargs.content = raw; message.additional_kwargs.selected = selected();
  expect(JSON.stringify(run.getRunMessages())).toBe(before);
});

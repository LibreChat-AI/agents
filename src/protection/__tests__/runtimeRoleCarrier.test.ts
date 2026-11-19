import { z } from 'zod';
import { RunnableLambda } from '@langchain/core/runnables';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { Command, Send, MemorySaver, emptyCheckpoint } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage, FunctionMessage, FunctionMessageChunk, ChatMessage, ChatMessageChunk } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '../toolResult';
import { withToolResultBoundary, ToolResultProtectionError, validateToolMessageHistory } from '../toolResult';
import { GraphEvents, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'runtime-role@example.invalid';
const policy = (block = false): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => block ? { version: 1, ok: false, error: { code: 'blocked' } } : { version: 1, ok: true, value: { content, replacements: 0, categories: [] } } });
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };
type RuntimeConfig = RunnableConfig & { config?: RuntimeConfig };

it.each([false, true].flatMap((selected) => (['value', 'message', 'error', 'control', 'default-off'] as const).map((mode) => [selected, mode] as const)))('isolates runtime.config callbacks (selected=%s/%s)', async (selected, mode) => {
  const observed: string[] = [], completions: string[] = [], authority = { principal: 'Allowed principal', resource: 'Allowed resource' }; let children = 0, executions = 0, body: RuntimeConfig | undefined;
  const child = RunnableLambda.from(async (_input, config) => { children++; expect(config.configurable?.authority).toBe(authority); expect(config.signal).toBe(body?.config?.signal); if (mode === 'error') throw Object.assign(new Error('Allowed diagnostic'), { result: new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }) }); if (mode === 'message') return new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }); return mode === 'control' ? 'Allowed value' : raw; });
  const name = selected ? 'lookup' : 'other';
  const source = new DynamicStructuredTool({ name, description: 'Allowed', schema: z.object({}), func: async (_input, _manager, runtime) => {
    executions++; body = runtime as RuntimeConfig; expect(body.config).toBeDefined(); expect(body.config!.configurable?.authority).toBe(authority);
    return child.invoke({}, body.config);
  } });
  const callback = BaseCallbackHandler.fromMethods({ handleChainEnd: (value): void => { observed.push(JSON.stringify(value)); }, handleChainError: (error): void => { observed.push(JSON.stringify(error)); }, handleToolEnd: (value): void => { observed.push(JSON.stringify(value)); }, handleToolError: (error): void => { observed.push(JSON.stringify(error)); }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [source], trace: true, toolCallStepIds: new Map([['outer', 'step']]), ...(mode === 'default-off' ? {} : { toolResultProtection: policy(selected && mode !== 'control') }) }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name, id: 'outer', args: {} }] })] }, { callbacks: [callback], configurable: { authority }, signal: new AbortController().signal }).catch((error: Error) => error);
  expect(executions).toBe(1); expect(children).toBe(1);
  if (mode === 'default-off') { expect(observed.join('')).toContain(raw); return; }
  if (!selected && mode === 'value') { expect(result).not.toBeInstanceOf(Error); return; }
  safe(observed); safe(completions); safe(result);
  if (mode !== 'control') { expect(result).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); } else expect(completions).toHaveLength(1);
});

const roles = [
  { name: 'function', make: (name = 'lookup') => new FunctionMessage({ name, content: raw }) },
  { name: 'function-chunk', make: (name = 'lookup') => new FunctionMessageChunk({ name, content: raw }) },
  ...['function', 'tool'].flatMap((role) => [{ name: 'ChatMessage-' + role, make: (name = 'lookup'): BaseMessage => Object.assign(new ChatMessage({ name, role, content: raw }), { tool_call_id: 'inner' }) }, { name: 'ChatMessageChunk-' + role, make: (name = 'lookup'): BaseMessage => Object.assign(new ChatMessageChunk({ name, role, content: raw }), { tool_call_id: 'inner' }) }]),
];

it.each(roles.flatMap(({ name, make }) => (['history', 'fresh', 'checkpoint', 'pending', 'send', 'native', 'host', 'command'] as const).map((route) => [name, route, make] as const)))('gates uncertified %s results before release (%s)', async (_kind, route, make) => {
  const message = make(), graph = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() }); let observed = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; }, handleToolEnd: (): void => { observed++; } }); callback.awaitHandlers = true;
  const invoke = async (): Promise<unknown> => {
    if (route === 'history') return validateToolMessageHistory(policy(), [message], []);
    if (route === 'send') return graph.validateToolResultSend(new Send('default', { messages: [message] }));
    if (route === 'checkpoint' || route === 'pending') { const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: route === 'checkpoint' ? [message] : [] }; return graph.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'role-carrier', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: route === 'pending' ? [['task', 'messages', [message]]] : [] }); }
    if (route === 'fresh') { const run = await Run.create({ runId: 'fresh-role-' + _kind, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } }, toolResultProtection: policy() }); run.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] }); return run.processStream({ messages: [new HumanMessage('Allowed'), message] }, { version: 'v2', callbacks: [callback] }); }
    const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => route === 'command' ? new Command({ update: { custom: message } }) : [{ type: 'text', text: 'Allowed', nested: message }] });
    const host = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, request): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) (request as { resolve: (results: object[]) => void }).resolve([{ toolCallId: 'outer', status: 'success', content: [{ nested: message }] }]); } }); host.awaitHandlers = true;
    return new ToolNode({ tools: [source], trace: true, eventDrivenMode: route === 'host', toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: {} }] })] }, { callbacks: [callback, host] });
  };
  const result = await invoke().catch((error: Error) => error); expect(result).toBeInstanceOf(ToolResultProtectionError); if (['fresh', 'history', 'checkpoint', 'pending', 'send'].includes(route)) expect(observed).toBe(0); safe(result);
});

it.each(roles.flatMap(({ name, make }) => (['unselected', 'default-off'] as const).map((mode) => [name, mode, make] as const)))('retains compatible %s history (%s)', async (_kind, mode, make) => {
  const message = make(mode === 'unselected' ? 'other' : 'lookup'); const result = await validateToolMessageHistory(mode === 'default-off' ? undefined : policy(), [message], []);
  expect(result[0].content).toBe(raw); if (mode === 'default-off') expect(result[0]).toBe(message); else expect(result[0]).not.toBe(message);
});

it.each(roles)('gates legacy %s resume updates before callbacks', async ({ name, make }) => {
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const runId = 'legacy-resume-' + name, run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: new MemorySaver() }, tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed output' })] }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Must not run'], toolCalls: [{ name: 'lookup', id: 'approval', args: {} }] }); const config = { version: 'v2' as const, configurable: { thread_id: runId } }; await run.processStream({ messages: [new HumanMessage('Allowed')] }, config); expect(run.getInterrupt()).toBeDefined(); let observed = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await run.resume([{ type: 'approve' }], { ...config, callbacks: [callback] }, undefined, { update: { messages: [make()] } }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); safe(result);
});

it.each(['tool', 'function'])('uses enclosing call authority for generic %s results', async (role) => {
  const message = Object.assign(new ChatMessage({ role, name: 'other', content: raw }), { tool_call_id: 'inner' });
  await expect(validateToolMessageHistory(policy(), [new AIMessage({ content: '', tool_calls: [{ id: 'inner', name: 'lookup', args: {} }] }), message], [])).rejects.toBeInstanceOf(ToolResultProtectionError);
});

it.each(['tool', 'function'])('rejects unnamed uncertified generic %s results', async (role) => {
  await expect(validateToolMessageHistory(policy(), [new ChatMessage({ role, content: raw })], [])).rejects.toBeInstanceOf(ToolResultProtectionError);
});

it.each(['assistant', 'user', 'system', 'developer', 'custom'])('does not reinterpret generic %s prose as selected results', async (role) => {
  const message = new ChatMessage({ role, name: 'lookup', content: 'Allowed prose' });
  const result = await validateToolMessageHistory(policy(), [message], []); expect(result[0].content).toBe('Allowed prose'); expect(result[0]).not.toBe(message);
});

it.each(['role', 'serialization-role'])('rejects callback-mutated generic %s before completion', async (field) => {
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => new ChatMessage({ role: 'user', name: 'lookup', content: 'Allowed' }) });
  const completions: string[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (value): void => { const message = value as ChatMessage; if (field === 'role') message.role = 'function'; else message.lc_kwargs.role = 'function'; }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [source], trace: true, toolCallStepIds: new Map([['outer', 'step']]), toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); safe(result);
});

it.each([false, true].flatMap((selected) => (['deep', 'cycle', 'proxy', 'getter', 'overflow'] as const).map((shape) => [selected, shape] as const)))('bounds nested runtime configs (selected=%s/%s)', async (selected, shape) => {
  let reads = 0, executions = 0; const authority = { principal: 'Allowed principal', resource: 'Allowed resource' };
  const nested: RuntimeConfig = { callbacks: [], configurable: { authority }, signal: new AbortController().signal };
  if (shape === 'cycle') nested.config = nested;
  if (shape === 'deep') nested.config = { callbacks: [], configurable: { authority } };
  if (shape === 'getter') Object.defineProperty(nested, 'config', { get: () => { reads++; throw new Error(raw); } });
  if (shape === 'overflow') { let current = nested; for (let i = 0; i < 70; i++) { current.config = {}; current = current.config; } }
  const config: RuntimeConfig = { config: shape === 'proxy' ? new Proxy(nested, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } }) : nested };
  const name = selected ? 'lookup' : 'other', tool = new DynamicStructuredTool({ name, description: 'Allowed', schema: z.object({}), func: async (_input, _manager, options) => {
    executions++; const runtime = options as RuntimeConfig; expect(runtime.config).not.toBe(nested); expect(runtime.config?.configurable?.authority).toBe(authority); expect(runtime.config?.signal).toBe(nested.signal); if (shape === 'cycle') expect(runtime.config?.config).toBe(runtime.config); return 'Allowed';
  } });
  const result = await withToolResultBoundary(tool, policy(), '', {}).invoke({}, config).catch((error: Error) => error);
  if (shape === 'cycle' || shape === 'deep') { expect(result).not.toBeInstanceOf(Error); expect(executions).toBe(1); } else { expect(result).toBeInstanceOf(ToolResultProtectionError); expect(executions).toBe(0); }
  expect(reads).toBe(0); safe(result);
});

it.each(roles)('rejects %s results injected into a real saver before observers', async ({ name, make }) => {
  let tamper = false, observed = 0;
  class Saver extends MemorySaver { override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> { const tuple = await super.getTuple(config); if (tamper && tuple != null) tuple.pendingWrites = [['task', 'messages', [make()]]]; return tuple; } }
  const saver = new Saver(), runId = 'saved-role-' + name;
  const create = () => Run.create({ runId, graphConfig: { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } }, first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] }); await first.processStream({ messages: [new HumanMessage('Allowed')] }, config);
  tamper = true; const next = await create(); next.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await next.processStream({ messages: [new HumanMessage('Allowed next')] }, { ...config, callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); safe(result);
});

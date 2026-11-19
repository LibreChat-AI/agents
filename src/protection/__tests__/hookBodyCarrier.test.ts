import { z } from 'zod';
import { RunnableLambda } from '@langchain/core/runnables';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '../toolResult';
import type { MessageContentComplex } from '@/types';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ToolResultProtectionError } from '../toolResult';
import { GraphEvents, Providers } from '@/common';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'hook-body@example.invalid';
const selected = (native = false): object => native ? new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }) : { type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw };
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const input = () => ({ messages: [new AIMessage({ content: '', tool_calls: [{ name: 'other', id: 'outer', args: {} }] })] });
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };

it.each([false, true].flatMap((native) => (['mutation', 'replacement', 'late', 'control', 'default-off'] as const).map((mode) => [native, mode] as const)))('admits direct post-hook carriers before completion (native=%s/%s)', async (native, mode) => {
  let executions = 0, calls = 0; const completions: string[] = [], hooks = new HookRegistry(), refs = new ToolOutputReferenceRegistry();
  let retained: Array<{ type: string; text: string; nested: object }> | undefined;
  hooks.register('PostToolUse', { hooks: [async (event) => {
    calls++; retained = event.toolOutput as typeof retained;
    if (mode === 'replacement') return { updatedOutput: [{ type: 'text', text: 'Allowed', nested: selected(native) }] };
    if (mode === 'mutation' || mode === 'default-off') retained![0].nested = selected(native);
    return {};
  }] });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { executions++; return [{ type: 'text', text: 'Allowed' }]; } });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [tool], trace: true, hookRegistry: hooks, toolOutputRegistry: refs, toolCallStepIds: new Map([['outer', 'step']]), ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) }).invoke(input(), { callbacks: [callback] }).catch((error: Error) => error) as Error | { messages: ToolMessage[] };
  expect(executions).toBe(1); expect(calls).toBe(1);
  if (mode === 'default-off') { expect(result).not.toBeInstanceOf(Error); expect(JSON.stringify(result)).toContain(raw); expect(completions).toHaveLength(1); return; }
  safe(completions); safe(result);
  if (mode === 'mutation' || mode === 'replacement') { expect(result).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); expect(refs.size).toBe(0); return; }
  expect(result).not.toBeInstanceOf(Error); expect(completions).toHaveLength(1);
  const before = JSON.stringify(result); retained![0].nested = selected(native); retained![0].text = raw; expect(JSON.stringify(result)).toBe(before); safe(result);
});

it.each(['mutation', 'late', 'control'] as const)('admits direct hook results before real Run model reuse (%s)', async (mode) => {
  let modelCalls = 0; const completions: string[] = [], wire: string[] = [], hooks = new HookRegistry(); let retained: MessageContentComplex[] | undefined;
  hooks.register('PostToolUse', { hooks: [async (event) => { retained = event.toolOutput as MessageContentComplex[]; if (mode === 'mutation') Object.assign(retained[0], { nested: selected() }); return {}; }] });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { modelCalls++; if (mode === 'late' && modelCalls === 2) Object.assign(retained![0], { nested: selected() }); wire.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => [{ type: 'text', text: 'Allowed' }] });
  const run = await Run.create({ runId: 'direct-hook-' + mode, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [tool] }, hooks, toolResultProtection: policy(), customHandlers: { [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideModel = new Model({ responses: ['', 'Allowed answer'], toolCalls: [{ name: 'other', id: 'outer', args: {} }] });
  const result = await run.processStream({ messages: [new HumanMessage('Allowed input')] }, { version: 'v2' }).catch((error: Error) => error);
  safe(completions); safe(wire); safe(result);
  expect(modelCalls).toBe(mode === 'mutation' ? 1 : 2); expect(completions).toHaveLength(mode === 'mutation' ? 0 : 1);
  if (mode === 'mutation') expect(result).toBeInstanceOf(ToolResultProtectionError);
});

it.each((['manager', 'config', 'ambient'] as const).flatMap((route) => [false, true].flatMap((native) => (['carrier', 'control', 'default-off'] as const).map((mode) => [route, native, mode] as const))))('isolates native body child callbacks (%s/native=%s/%s)', async (route, native, mode) => {
  const observed: string[] = [], completions: string[] = []; let executions = 0, children = 0, tools = 0;
  const child = RunnableLambda.from(async () => { children++; return mode === 'control' ? { count: 42 } : selected(native); });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async (_input, manager, config) => {
    executions++;
    let options = config;
    if (route === 'manager') options = { callbacks: manager?.getChild('body-child') };
    if (route === 'ambient') options = undefined;
    const value = await child.invoke({}, options);
    await manager?.handleText(raw); await dispatchCustomEvent('body-observation', { nested: selected(native) }, config);
    return [{ type: 'text', text: 'Allowed', nested: value }];
  } });
  const callback = BaseCallbackHandler.fromMethods({ handleChainEnd: (value): void => { observed.push(JSON.stringify(value)); }, handleText: (text): void => { observed.push(text); }, handleToolEnd: (value): void => { tools++; observed.push(JSON.stringify(value)); }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); if (event === 'body-observation') observed.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [tool], trace: true, toolCallStepIds: new Map([['outer', 'step']]), ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) }).invoke(input(), { callbacks: [callback] }).catch((error: Error) => error);
  expect(executions).toBe(1); expect(children).toBe(1);
  if (mode === 'default-off') { expect(observed.join('')).toContain(raw); return; }
  safe(observed); safe(completions); safe(result);
  if (mode === 'carrier') { expect(result).toBeInstanceOf(ToolResultProtectionError); expect(tools).toBe(0); expect(completions).toEqual([]); }
  else { expect(result).not.toBeInstanceOf(Error); expect(tools).toBe(1); expect(completions).toHaveLength(1); }
});

it.each(['getter', 'proxy', 'wide', 'throw-after-mutation'] as const)('rejects post-hook %s carriers before completion without invoking hostile behavior', async (mode) => {
  let reads = 0; const completions: string[] = [], hooks = new HookRegistry();
  hooks.register('PostToolUse', { hooks: [async (event) => {
    const block = (event.toolOutput as object[])[0];
    if (mode === 'getter') Object.defineProperty(block, 'nested', { get: () => { reads++; throw new Error(raw); } });
    if (mode === 'proxy') Object.assign(block, { nested: new Proxy({}, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } }) });
    if (mode === 'wide') Object.assign(block, { nested: Array.from({ length: 9000 }, () => ({ count: 42 })) });
    if (mode === 'throw-after-mutation') { Object.assign(block, { nested: selected() }); throw new Error('Allowed hook failure'); }
    return {};
  }] });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => [{ type: 'text', text: 'Allowed' }] });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [tool], hookRegistry: hooks, toolResultProtection: policy(), toolCallStepIds: new Map([['outer', 'step']]) }).invoke(input(), { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); expect(completions).toEqual([]); safe(result);
});

it.each(['child-tool', 'child-error', 'sequence', 'child-stream'] as const)('isolates additional child body observer paths (%s)', async (route) => {
  const observed: string[] = []; let children = 0;
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (value): void => { observed.push(JSON.stringify(value)); }, handleChainEnd: (value): void => { observed.push(JSON.stringify(value)); }, handleChainError: (error): void => { observed.push(JSON.stringify(error)); }, handleToolError: (error): void => { observed.push(JSON.stringify(error)); } }); callback.awaitHandlers = true;
  const child = RunnableLambda.from(async () => { children++; if (route === 'child-error') throw Object.assign(new Error('Allowed diagnostic'), { result: selected(true) }); return selected(true); });
  const nested = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => { children++; return raw; } });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async (_input, manager, config) => {
    const options = { ...config, callbacks: manager?.getChild() };
    if (route === 'child-tool') return nested.invoke({ name: 'lookup', id: 'inner', type: 'tool_call', args: {} }, options);
    if (route === 'sequence') return child.pipe(RunnableLambda.from(async (value) => value)).invoke({}, options);
    if (route === 'child-stream') { const values: object[] = []; for await (const value of await child.stream({}, options)) values.push(value); return values; }
    return child.invoke({}, options);
  } });
  const result = await new ToolNode({ tools: [tool], trace: true, handleToolErrors: false, toolResultProtection: policy() }).invoke(input(), { callbacks: [callback] }).catch((error: Error) => error);
  expect(children).toBe(1); expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result); safe(observed);
});

it('retains body signal, configured authority, child APIs and outer callback lifecycle', async () => {
  const controller = new AbortController(), authority = { principal: 'Allowed principal', resource: 'Allowed resource' }; let bodySignal: AbortSignal | undefined; let starts = 0, ends = 0, childRuns = 0;
  const child = RunnableLambda.from(async (_input, config) => { childRuns++; expect(config.signal).toBe(bodySignal); expect(config.configurable?.authority).toBe(authority); return { count: 42 }; });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async (_input, manager, config) => {
    expect(manager?.getChild()).toBeDefined(); bodySignal = config?.signal; expect(bodySignal).toBeInstanceOf(AbortSignal); expect(bodySignal?.aborted).toBe(false); expect(config?.configurable?.authority).toBe(authority);
    const value = await child.invoke({}, config); return [{ type: 'text', text: 'Allowed', nested: value }];
  } });
  const callback = BaseCallbackHandler.fromMethods({ handleToolStart: (): void => { starts++; }, handleToolEnd: (): void => { ends++; } }); callback.awaitHandlers = true;
  const original = Reflect.get(tool, '_call');
  const result = await new ToolNode({ tools: [tool], toolResultProtection: policy() }).invoke(input(), { signal: controller.signal, configurable: { authority }, callbacks: [callback] });
  expect(result).not.toBeInstanceOf(Error); expect({ starts, ends, childRuns }).toEqual({ starts: 1, ends: 1, childRuns: 1 }); expect(Reflect.get(tool, '_call')).toBe(original);
});

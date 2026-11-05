import { z } from 'zod';
import { Command, Send } from '@langchain/langgraph';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolExecuteResult, ToolExecuteBatchRequest } from '@/types';
import type { ToolResultProtection } from '../toolResult';
import { protectToolExecuteResult, ToolResultProtectionError } from '../toolResult';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ChatModelStreamHandler } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'host-descendant@example.invalid';
const policy = (inspect = jest.fn()): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => { inspect(content); return { version: 1, ok: true, value: { content, replacements: 0, categories: [] } }; } });
const serialized = () => ({ type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw });
const message = () => new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw });
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };
const shells = [
  { name: 'serialized', make: serialized },
  { name: 'native', make: message },
  { name: 'command', make: () => new Command({ update: { custom: { selected: serialized() } } }) },
  { name: 'send', make: () => new Send('default', { messages: [message()] }) },
  { name: 'metadata', make: () => new HumanMessage({ content: 'Allowed', additional_kwargs: { selected: message() } }) },
];

it.each(shells.flatMap((entry) => (['content', 'artifact', 'injected'] as const).map((field) => [entry, field] as const)))('gates %s descendants in unselected host %s', async ({ make }, field) => {
  const inspect = jest.fn(); const result: ToolExecuteResult = { toolCallId: 'outer', status: 'success', content: 'Allowed output' };
  if (field === 'content') result.content = [make()];
  if (field === 'artifact') result.artifact = { nested: make() };
  if (field === 'injected') result.injectedMessages = [{ role: 'user', content: [{ type: 'text', text: 'Allowed', nested: make() }] }] as ToolExecuteResult['injectedMessages'];
  const outcome = await protectToolExecuteResult(policy(inspect), { name: 'other', id: 'outer' }, result).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(inspect).not.toHaveBeenCalled(); safe(outcome);
});

it.each(['getter', 'proxy', 'wide'] as const)('rejects unselected host %s descendants with bounded, descriptor-only admission', async (mode) => {
  let reads = 0; const value = { selected: 'Allowed' };
  if (mode === 'getter') Object.defineProperty(value, 'selected', { get: () => { reads++; throw new Error(raw); } });
  const content = mode === 'wide' ? Array.from({ length: 9000 }, () => ({ value: 'Allowed' })) : [mode === 'proxy' ? new Proxy(value, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } }) : value];
  const outcome = await protectToolExecuteResult(policy(), { name: 'other', id: 'outer' }, { toolCallId: 'outer', status: 'success', content }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); safe(outcome);
  if (mode === 'wide') expect(outcome).toMatchObject({ code: 'overflow' });
});

it.each(['success', 'error'] as const)('preserves and detaches valid unselected %s host data', async (status) => {
  const content = [{ rows: [{ count: 42, rawText: raw }], status: 'allowed' }];
  const artifact = Buffer.from([1, 2, 3]);
  const inspect = jest.fn(); const source: ToolExecuteResult = { toolCallId: 'outer', status, content, artifact, outcome: 'Allowed outcome', ...(status === 'error' ? { errorMessage: 'Allowed diagnostic' } : {}) };
  const outcome = await protectToolExecuteResult(policy(inspect), { name: 'other', id: 'outer' }, source);
  expect(outcome).toEqual(source); expect(outcome).not.toBe(source); expect(outcome.content).not.toBe(content); expect(outcome.artifact).toBe(artifact); expect(inspect).not.toHaveBeenCalled();
  content[0].rows[0].count = 99; expect((outcome.content as typeof content)[0].rows[0].count).toBe(42);
});

it.each([false, true])('gates host descendants before completion, references and hooks (eagerRecord=%s)', async (eagerRecord) => {
  const refs = new ToolOutputReferenceRegistry(), observed: string[] = []; let executions = 0, postHooks = 0;
  const hooks = new HookRegistry(); if (!eagerRecord) hooks.register('PostToolUse', { hooks: [async () => { postHooks++; return {}; }] });
  const request = { id: 'outer', name: 'other', args: {}, stepId: 'step', turn: 0 };
  const result: ToolExecuteResult = { toolCallId: 'outer', status: 'success', content: [serialized()] };
  const node = new ToolNode({ tools: [new DynamicStructuredTool({ name: 'other', description: 'Host control', schema: z.object({}), func: async () => 'Must not execute' })], trace: true, eventDrivenMode: true, toolResultProtection: policy(), toolOutputRegistry: refs, toolCallStepIds: new Map([['outer', 'step']]), hookRegistry: hooks, ...(eagerRecord ? { eagerEventToolExecution: { enabled: true }, eagerEventToolExecutions: new Map([['outer', { toolCallId: 'outer', toolName: 'other', args: {}, request, promise: Promise.resolve({ results: [result] }) }]]) } : {}) });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) { executions++; (data as ToolExecuteBatchRequest).resolve([result]); } if (event === GraphEvents.ON_RUN_STEP_COMPLETED) observed.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const outcome = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback], configurable: { run_id: 'host-carrier' } }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observed).toEqual([]); expect(refs.size).toBe(0); expect(postHooks).toBe(0); expect(executions).toBe(eagerRecord ? 0 : 1); safe(outcome);
});

it.each([false, true])('rejects descendants before real host/eager model reuse (eager=%s)', async (eager) => {
  let executions = 0, modelCalls = 0; const completions: string[] = [];
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { modelCalls++; yield* super._streamResponseChunks(...args); } }
  const run = await Run.create({ runId: 'host-carrier-run-' + eager, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, toolDefinitions: [{ name: 'other', description: 'Allowed host tool', parameters: { type: 'object', properties: {} } }] }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, toolOutputReferences: { enabled: !eager }, customHandlers: { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(), [GraphEvents.ON_TOOL_EXECUTE]: { handle: async (_event, data): Promise<void> => { executions++; const batch = data as ToolExecuteBatchRequest; batch.onResult?.({ toolCallId: batch.toolCalls[0].id, status: 'success', content: [serialized()] }); batch.resolve([{ toolCallId: batch.toolCalls[0].id, status: 'success', content: [serialized()] }]); } }, [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideModel = new Model({ responses: ['', 'Must not run'], toolCalls: [{ id: 'outer', name: 'other', args: {} }] });
  const outcome = await run.processStream({ messages: [new HumanMessage('Allowed input')] }, { version: 'v2', configurable: { thread_id: 'host-carrier-run-' + eager } }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(executions).toBe(1); expect(modelCalls).toBe(1); expect(completions).toEqual([]); safe(outcome);
});

it.each(['replacement', 'mutation'] as const)('gates optional host post-hook %s carriers before publication', async (mode) => {
  const refs = new ToolOutputReferenceRegistry(), completions: string[] = [], hooks = new HookRegistry();
  hooks.register('PostToolUse', { hooks: [async (input) => { if (mode === 'replacement') return { updatedOutput: [serialized()] }; (input.toolOutput as Array<{ nested: unknown }>)[0].nested = serialized(); return {}; }] });
  const node = new ToolNode({ tools: [new DynamicStructuredTool({ name: 'other', description: 'Host control', schema: z.object({}), func: async () => 'Must not execute' })], trace: true, eventDrivenMode: true, hookRegistry: hooks, toolOutputRegistry: refs, toolCallStepIds: new Map([['outer', 'step']]), toolResultProtection: policy() });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) (data as ToolExecuteBatchRequest).resolve([{ toolCallId: 'outer', status: 'success', content: [{ nested: 'Allowed' }] }]); if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const outcome = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback], configurable: { run_id: 'host-hook-carrier' } }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(refs.size).toBe(0); expect(completions).toEqual([]); safe(outcome);
});

it('keeps default-off host results untouched', async () => {
  const result: ToolExecuteResult = { toolCallId: 'outer', status: 'success', content: [serialized()] };
  expect(await protectToolExecuteResult(undefined, { name: 'other', id: 'outer' }, result)).toBe(result);
});

it.each([false, true])('detaches host-held structured aliases before next-model reuse (eager=%s)', async (eager) => {
  const source = [{ nested: { count: 42 } }]; let executions = 0, modelCalls = 0;
  const wireInputs: string[] = [], completions: string[] = [];
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { modelCalls++; wireInputs.push(JSON.stringify(args[0])); yield* super._streamResponseChunks(...args); } }
  const runId = 'detached-host-' + eager;
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, toolDefinitions: [{ name: 'other', description: 'Allowed host tool', parameters: { type: 'object', properties: {} } }] }, toolResultProtection: policy(), eagerEventToolExecution: { enabled: eager }, customHandlers: { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(), [GraphEvents.ON_TOOL_EXECUTE]: { handle: async (_event, data): Promise<void> => {
    executions++; const batch = data as ToolExecuteBatchRequest;
    batch.resolve([{ toolCallId: batch.toolCalls[0].id, status: 'success', content: source }]);
    Object.defineProperty(source[0].nested, 'selected', { value: serialized() }); source[0].nested.count = 99;
  } }, [GraphEvents.ON_RUN_STEP_COMPLETED]: { handle: (_event, data): void => { completions.push(JSON.stringify(data)); } } } });
  run.Graph!.overrideModel = new Model({ responses: ['', 'Allowed final answer'], toolCalls: [{ id: 'outer', name: 'other', args: {} }] });
  await run.processStream({ messages: [new HumanMessage('Allowed input')] }, { version: 'v2', configurable: { thread_id: runId } });
  expect(executions).toBe(1); expect(modelCalls).toBe(2); expect(completions).toHaveLength(1); expect(wireInputs[1]).toContain('42'); safe(wireInputs); safe(completions);
});

it.each(['success', 'error'] as const)('gates hidden carriers in the unselected half of a mixed %s host batch', async (status) => {
  let executions = 0; const completions: string[] = [];
  const tools = ['lookup', 'other'].map((name) => new DynamicStructuredTool({ name, description: 'Host control', schema: z.object({}), func: async () => 'Must not execute' }));
  const node = new ToolNode({ tools, trace: true, eventDrivenMode: true, toolResultProtection: policy(), toolCallStepIds: new Map([['outer', 'step-outer'], ['selected', 'step-selected']]) });
  const callback = BaseCallbackHandler.fromMethods({ handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_TOOL_EXECUTE) { executions++; const batch = data as ToolExecuteBatchRequest; batch.resolve(batch.toolCalls.map((call) => ({ toolCallId: call.id, status, content: call.name === 'other' ? [serialized()] : 'Allowed selected output', ...(status === 'error' ? { errorMessage: 'Allowed diagnostic' } : {}) }))); } if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(JSON.stringify(data)); } }); callback.awaitHandlers = true;
  const outcome = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'selected', name: 'lookup', args: {} }, { id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(executions).toBe(1); expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); safe(outcome);
});

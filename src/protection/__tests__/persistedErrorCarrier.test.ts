import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { MemorySaver, emptyCheckpoint } from '@langchain/langgraph';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolResultProtection } from '../toolResult';
import type { RunStepResumeState } from '@/types';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { ToolResultProtectionError } from '../toolResult';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const raw = 'persisted-error@example.invalid';
const selected = (native = false): object => native ? new ToolMessage({ name: 'lookup', tool_call_id: 'inner', content: raw }) : { type: 'tool', name: 'lookup', tool_call_id: 'inner', content: raw };
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const graph = () => new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
const channels = ['__interrupt__', '__resume__', '__error__', '__error_source_node__', '__scheduled__', '__return__', '__no_writes__'];
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };

it.each(channels.flatMap((channel) => (['values', 'pending'] as const).map((route) => [channel, route] as const)))('admits persisted native %s payloads (%s)', async (channel, route) => {
  const checkpoint = emptyCheckpoint(), value = { nested: selected() }; checkpoint.channel_values = { messages: [], ...(route === 'values' ? { [channel]: value } : {}) };
  const result = await graph().validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'persisted-carrier', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: route === 'pending' ? [['task', channel, value]] : [] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result);
});

it.each(['root', 'child'] as const)('admits snapshot task interrupts (%s)', async (route) => {
  const state = { values: { messages: [] }, tasks: [{ id: 'task', name: 'tools', interrupts: [{ id: 'interrupt', value: { nested: selected() } }] }] };
  const result = await graph().validateToolResultCheckpoint(route === 'root' ? state : { values: { messages: [] }, tasks: [{ id: 'child', name: 'child', state }] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result);
});

it.each(['fresh', 'resume'] as const)('rejects real saver interrupt tampering before native observers (%s)', async (route) => {
  let tamper = false, observed = 0;
  class Saver extends MemorySaver { override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
    const tuple = await super.getTuple(config); if (!tamper || tuple == null) return tuple;
    tuple.pendingWrites = [...tuple.pendingWrites ?? [], ['injected', '__interrupt__', [{ id: 'injected-interrupt', value: { nested: selected() } }]]]; return tuple;
  } }
  const saver = new Saver(), hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const runId = 'persisted-interrupt-' + route;
  const create = () => Run.create({ runId, graphConfig: { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({}), func: async () => 'Allowed output' })], compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: [''], toolCalls: [{ id: 'approval', name: 'lookup', args: {} }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } }; await first.processStream({ messages: [new HumanMessage('Allowed input')] }, config); expect(first.getInterrupt()).toBeDefined(); tamper = true;
  const next = await create(); next.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await (route === 'resume' ? next.resume([{ type: 'approve' }], { ...config, callbacks: [callback] }) : next.processStream({ messages: [new HumanMessage('Allowed input')] }, { ...config, callbacks: [callback] })).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); expect(next.getInterrupt()).toBeUndefined(); safe(result);
});

it.each((['cause', 'result', 'nested-error', 'aggregate', 'hidden'] as const).flatMap((field) => [false, true].flatMap((native) => [false, true].map((handled) => [field, native, handled] as const))))('gates ordinary error %s descendants (native=%s/handled=%s)', async (field, native, handled) => {
  let error: Error = new Error('Allowed diagnostic'); const value = selected(native);
  if (field === 'cause') error = new Error('Allowed diagnostic', { cause: value });
  if (field === 'result') Object.assign(error, { result: value });
  if (field === 'hidden') Object.defineProperty(error, 'result', { value });
  if (field === 'nested-error') error = new TypeError('Allowed diagnostic', { cause: Object.assign(new Error('Allowed nested diagnostic'), { result: value }) });
  if (field === 'aggregate') error = new AggregateError([value], 'Allowed diagnostic');
  const observed: object[] = [], completions: object[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { observed.push(error); }, handleChainError: (error): void => { observed.push(error); }, handleCustomEvent: (event, data): void => { if (event === GraphEvents.ON_RUN_STEP_COMPLETED) completions.push(data as object); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [tool], trace: true, handleToolErrors: handled, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(completions).toEqual([]); for (const error of observed) { expect(error).toBeInstanceOf(ToolResultProtectionError); safe(error); } safe(result);
});

it.each((['native', 'chain'] as const).flatMap((stage) => (['result', 'cause', 'getter', 'control', 'late'] as const).map((mode) => [stage, mode] as const)))('readmits ordinary errors after %s callbacks (%s)', async (stage, mode) => {
  let retained: Error | undefined, reads = 0;
  const source = Object.assign(new Error('Allowed diagnostic', { cause: new Error('Allowed cause') }), { detail: { count: 42 } });
  const mutate = (error: Error): void => { retained = error; if (mode === 'result' || mode === 'cause') Object.defineProperty(error, mode, { value: selected(), configurable: true }); if (mode === 'getter') Object.defineProperty(error, 'result', { get: () => { reads++; throw new Error(raw); } }); };
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { if (stage === 'native') mutate(error); }, handleChainError: (error): void => { if (stage === 'chain' && !(error instanceof ToolResultProtectionError)) mutate(error); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw source; } });
  const result = await new ToolNode({ tools: [tool], trace: true, handleToolErrors: false, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error) as Error & { detail: typeof source.detail };
  expect(reads).toBe(0);
  if (mode !== 'control' && mode !== 'late') expect(result).toBeInstanceOf(ToolResultProtectionError);
  else { expect(result.message).toBe(source.message); expect(result).not.toBe(source); expect(result).not.toBe(retained); expect(result.cause).not.toBe(source.cause); source.detail.count = 99; Object.defineProperty(retained!, 'result', { value: selected(), enumerable: true }); expect(result.detail.count).toBe(42); }
  safe(result);
});

it.each(['snapshot', 'values', 'pending'] as const)('retains genuine approval argument data without result admission (%s)', async (route) => {
  const args = { payload: selected() }, message = new AIMessage({ id: 'proposal', content: '', tool_calls: [{ id: 'approval', name: 'lookup', args }] });
  const approval = { type: 'tool_approval', action_requests: [{ name: 'lookup', tool_call_id: 'approval', arguments: args }], review_configs: [{ action_name: 'lookup', tool_call_id: 'approval', allowed_decisions: ['approve'] }] };
  if (route === 'snapshot') {
    const result = await graph().validateToolResultCheckpoint({ values: { messages: [message] }, tasks: [{ id: 'task', name: 'tools', interrupts: [{ id: 'interrupt', value: approval }] }] });
    expect(result.tasks[0].interrupts[0].value).toEqual(approval); return;
  }
  const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [message], ...(route === 'values' ? { __interrupt__: [{ id: 'interrupt', value: approval }] } : {}) };
  const result = await graph().validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'approval-arguments', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: route === 'pending' ? [['task', '__interrupt__', [{ id: 'interrupt', value: approval }]]] : [] });
  const output = route === 'values' ? result.checkpoint.channel_values.__interrupt__ : result.pendingWrites![0][2];
  expect(output).toEqual([{ id: 'interrupt', value: approval }]); expect(message.tool_calls![0].args).toBe(args);
});

it('preserves real SDK approval and resume when arguments resemble selected results', async () => {
  const args = { payload: selected() }; let executions = 0;
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const source = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed', schema: z.object({ payload: z.unknown() }), func: async (input) => { executions++; expect(input).toEqual(args); return 'Allowed result'; } });
  const run = await Run.create({ runId: 'approval-carrier-args', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [source], compileOptions: { checkpointer: new MemorySaver() } }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed answer'], toolCalls: [{ id: 'approval', name: 'lookup', args }] });
  const config = { version: 'v2' as const, configurable: { thread_id: 'approval-carrier-args' } };
  await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config); expect(run.getInterrupt()).toBeDefined(); expect(executions).toBe(0);
  await run.resume([{ type: 'approve' }], config); expect(executions).toBe(1);
});

it.each(['getter', 'proxy', 'function', 'symbol', 'cycle', 'wide', 'subclass'] as const)('rejects unsupported ordinary error fields without invoking hostile behavior (%s)', async (mode) => {
  let reads = 0; let error: object = new Error('Allowed diagnostic');
  if (mode === 'getter') Object.defineProperty(error, 'result', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'proxy') error = new Proxy(error, { getPrototypeOf: (): never => { reads++; throw new Error(raw); } });
  if (mode === 'function') Object.assign(error, { result: selected });
  if (mode === 'symbol') Object.defineProperty(error, Symbol('selected'), { value: selected() });
  if (mode === 'cycle') Object.assign(error, { cause: error });
  if (mode === 'wide') Object.assign(error, { result: Array.from({ length: 9000 }, () => ({ count: 42 })) });
  if (mode === 'subclass') { class CustomError extends Error { result = selected(); } error = new CustomError('Allowed diagnostic'); }
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (visible): void => { expect(visible).toBeInstanceOf(ToolResultProtectionError); } }); callback.awaitHandlers = true;
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [source], toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); safe(result);
});

it.each(['extra', 'request-extra', 'review-extra', 'arguments-lookalike', 'wrong-name'] as const)('does not exempt non-argument approval fields (%s)', async (mode) => {
  const args = { count: 42 }, proposal = new AIMessage({ id: 'proposal', content: '', tool_calls: [{ id: 'approval', name: 'lookup', args }] });
  const approval = { type: 'tool_approval', action_requests: [{ name: 'lookup', tool_call_id: 'approval', arguments: args }], review_configs: [{ action_name: 'lookup', tool_call_id: 'approval', allowed_decisions: ['approve'] }] };
  if (mode === 'extra') Object.assign(approval, { selected: selected() });
  if (mode === 'request-extra') Object.assign(approval.action_requests[0], { result: selected() });
  if (mode === 'review-extra') Object.assign(approval.review_configs[0], { result: selected() });
  if (mode === 'arguments-lookalike') Object.assign(approval, { arguments: selected() });
  if (mode === 'wrong-name') { approval.action_requests[0].name = 'other'; Object.assign(args, { payload: selected() }); }
  const result = await graph().validateToolResultCheckpoint({ values: { messages: [proposal] }, tasks: [{ id: 'task', name: 'tools', interrupts: [{ id: 'interrupt', value: approval }] }] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result);
});

it.each(channels)('detaches ordinary native %s data with unchanged values', async (channel) => {
  const checkpoint = emptyCheckpoint(), value = { authority: { principal: 'Allowed principal', resource: 'Allowed resource' } }; checkpoint.channel_values = { messages: [], [channel]: value };
  const result = await graph().validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'control-data', checkpoint_id: checkpoint.id } }, checkpoint });
  expect(result.checkpoint.channel_values[channel]).toEqual(value); expect(result.checkpoint.channel_values[channel]).not.toBe(value); value.authority.principal = raw; safe(result);
});

it.each([Error, TypeError, RangeError, ReferenceError, SyntaxError, URIError, EvalError])('retains native error type and ordinary diagnostic text (%s)', async (Type) => {
  const error = Object.assign(new Type('Allowed diagnostic'), { detail: { count: 42 } });
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [source], handleToolErrors: false, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }).catch((error: Error) => error) as typeof error;
  expect(result).toBeInstanceOf(Type); expect(result.message).toBe(error.message); expect(result.name).toBe(error.name); expect(result.detail).toEqual(error.detail); expect(result.detail).not.toBe(error.detail);
});

it('does not gate or detach ordinary errors when protection is off', async () => {
  const error = Object.assign(new Error('Allowed diagnostic'), { result: selected() }); let retained: Error | undefined;
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { retained = error; } }); callback.awaitHandlers = true;
  const source = new DynamicStructuredTool({ name: 'other', description: 'Allowed', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [source], handleToolErrors: false }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBe(error); expect(retained).toBe(error); expect(JSON.stringify(result)).toContain(raw);
});

it.each(['native', 'function'] as const)('preserves explicit run-step %s arguments while gating adjacent fields', async (encoding) => {
  const args = { payload: selected() }, request = encoding === 'native' ? { id: 'approval', name: 'lookup', args } : { id: 'approval', type: 'function' as const, function: { name: 'lookup', arguments: args } };
  const runStepState: RunStepResumeState = { version: 1, revision: 0, nextIndex: 1, toolCallSteps: [{ toolCallId: 'approval', stepId: 'step' }], steps: [{ step: { id: 'step', index: 0, type: StepTypes.TOOL_CALLS, stepDetails: { type: StepTypes.TOOL_CALLS, tool_calls: [request] } }, pendingToolCallIds: ['approval'], openMessageStep: false }] };
  const input = { values: { messages: [], runStepState }, tasks: [] };
  const result = await graph().validateToolResultCheckpoint(input); expect(result.values.runStepState).toEqual(runStepState); expect(result.values.runStepState).not.toBe(runStepState);
  Object.assign(request, { result: selected() }); await expect(graph().validateToolResultCheckpoint(input)).rejects.toBeInstanceOf(ToolResultProtectionError);
});

import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { GraphDrained, GraphInterrupt, NodeInterrupt, ParentCommand, Command, Send, MemorySaver } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolResultProtection } from '../toolResult';
import { snapshotToolSummarizationRequest, ToolResultProtectionError } from '../toolResult';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Providers } from '@/common';
import { Run } from '@/run';

const raw = 'control-payload@example.invalid';
const selected = () => new ToolMessage({ name: 'lookup', tool_call_id: 'selected-call', content: raw });
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const summary = () => ({ agentId: 'default', remainingContextTokens: 100, reason: 'manual' as const, allowSummarization: true });
const safe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(String(JSON.stringify(value))).not.toContain(raw); };

it.each(['message', 'string', 'getter', 'symbol', 'control'] as const)('closes summarization envelopes (%s)', (mode) => {
  let reads = 0; const request = summary();
  if (mode === 'message' || mode === 'string') Object.defineProperty(request, 'extra', { value: mode === 'message' ? selected() : raw });
  if (mode === 'getter') Object.defineProperty(request, 'extra', { get: () => { reads++; throw new Error(raw); } });
  if (mode === 'symbol') Object.defineProperty(request, Symbol('extra'), { value: selected() });
  if (mode === 'control') expect(snapshotToolSummarizationRequest(request)).toEqual(request);
  else expect(() => snapshotToolSummarizationRequest(request)).toThrow(ToolResultProtectionError);
  expect(reads).toBe(0);
});

it.each(['fresh', 'values', 'pending', 'resume', 'send', 'window'] as const)('rejects summary extra fields before Run observers (%s)', async (carrier) => {
  let tamper = false, observed = 0;
  const request = { ...summary(), extra: selected() };
  class Saver extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(config); if (!tamper || tuple == null) return tuple;
      if (carrier === 'values') tuple.checkpoint.channel_values.summarizationRequest = request;
      if (carrier === 'pending') tuple.pendingWrites = [['task', 'summarizationRequest', request]];
      return tuple;
    }
  }
  const hooks = new HookRegistry(); hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const runId = 'summary-extra-' + carrier, run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: new Saver() }, tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed control', schema: z.object({}), func: async () => 'Allowed result' })] }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed answer'], toolCalls: [{ id: 'approval-call', name: 'lookup', args: {} }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  if (carrier !== 'fresh' && carrier !== 'window') { await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config); expect(run.getInterrupt()).toBeDefined(); }
  tamper = true;
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const options = { ...config, callbacks: [callback] };
  const invoke = async (): Promise<unknown> => {
    if (carrier === 'window') return run.Graph!.validateToolResultWindow(request);
    if (carrier === 'resume') return run.resume([{ type: 'approve' }], options, undefined, { update: { summarizationRequest: request } });
    if (carrier === 'send') return run.resume([{ type: 'approve' }], options, undefined, { goto: [new Send('default', { messages: [new HumanMessage('Allowed input')], summarizationRequest: request })] });
    if (carrier === 'fresh') return run.processStream({ messages: [new HumanMessage('Allowed input')], summarizationRequest: request } as Parameters<typeof run.processStream>[0], options);
    return run.resume([{ type: 'approve' }], options);
  };
  const result = await invoke().catch((error: Error) => error); expect(result).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); safe(result);
});

type Kind = 'interrupt' | 'parent';
const control = (kind: Kind, payload: object): Error => kind === 'interrupt' ? new GraphInterrupt([{ id: 'ordinary-control', value: payload }]) : new ParentCommand(new Command({ graph: Command.PARENT, update: { custom: payload } }));
it.each((['interrupt', 'parent'] as const).flatMap((kind) => (['producer', 'tool-callback', 'chain-callback', 'late', 'control', 'ordinary-error', 'default-off'] as const).map((mode) => [kind, mode] as const)))('admits thrown native %s payloads (%s)', async (kind, mode) => {
  const payload = { nested: mode === 'producer' || mode === 'default-off' ? selected() : 'Allowed payload' };
  const error = mode === 'ordinary-error' ? new Error('Allowed diagnostic', { cause: new Error('Allowed cause') }) : control(kind, payload);
  const observed: string[] = []; let retained: Error | undefined;
  const change = (visible: Error): void => {
    retained = visible;
    if (mode === 'ordinary-error') return;
    const update = kind === 'parent' ? (visible as ParentCommand).command.update : undefined;
    let data: unknown;
    if (kind === 'interrupt') data = (visible as GraphInterrupt).interrupts[0].value;
    else if (Array.isArray(update)) data = update.find(([key]) => key === 'custom')![1];
    else data = update!.custom;
    (data as typeof payload).nested = selected();
  };
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (visible): void => { observed.push(String(visible), JSON.stringify(visible)); retained = visible; if (mode === 'tool-callback') change(visible); }, handleChainError: (visible): void => { observed.push(String(visible), JSON.stringify(visible)); if (mode === 'chain-callback') change(visible); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw error; } });
  const node = new ToolNode({ tools: [tool], trace: true, handleToolErrors: false, ...(mode === 'default-off' ? {} : { toolResultProtection: policy() }) });
  const result = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((caught: Error) => caught) as Error;
  if (mode === 'default-off') { expect(observed.join('')).toContain(raw); return; }
  safe(observed); safe(result);
  if (mode === 'producer' || mode === 'tool-callback' || mode === 'chain-callback') expect(result).toBeInstanceOf(ToolResultProtectionError);
  else if (mode === 'ordinary-error') { expect(result).not.toBe(error); expect(result.message).toBe(error.message); expect(result.cause).toEqual(error.cause); expect(result.cause).not.toBe(error.cause); }
  else { expect(result).toBeInstanceOf(kind === 'interrupt' ? GraphInterrupt : ParentCommand); expect(result).not.toBe(retained); if (mode === 'late') payload.nested = selected(); safe(result); }
});

it.each(['interrupt', 'parent'] as const)('rejects %s payload accessors before native error callbacks', async (kind) => {
  let reads = 0, observed = 0;
  const error = control(kind, { nested: 'Allowed payload' });
  Object.defineProperty(error, kind === 'interrupt' ? 'interrupts' : 'command', { get: () => { reads++; throw new Error(raw); } });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw error; } });
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (): void => { observed++; } }); callback.awaitHandlers = true;
  const result = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); expect(observed).toBe(1); safe(result);
});

it.each(['producer', 'native-callback', 'control'] as const)('keeps unselected control payloads safe in real Run persistence (%s)', async (mode) => {
  let modelCalls = 0;
  const saver = new MemorySaver(), runId = 'unselected-control-' + mode;
  const payload = { nested: mode === 'producer' ? selected() : 'Allowed payload' };
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw control('interrupt', payload); } });
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [tool], compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  class Model extends FakeChatModel { override async *_streamResponseChunks(...args: Parameters<FakeChatModel['_streamResponseChunks']>): ReturnType<FakeChatModel['_streamResponseChunks']> { modelCalls++; yield* super._streamResponseChunks(...args); } }
  run.Graph!.overrideModel = new Model({ responses: [''], toolCalls: [{ id: 'outer-call', name: 'other', args: {} }] });
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { if (mode === 'native-callback' && error instanceof GraphInterrupt) (error.interrupts[0].value as typeof payload).nested = selected(); } }); callback.awaitHandlers = true;
  const config = { version: 'v2' as const, configurable: { thread_id: runId }, callbacks: [callback] };
  const result = await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config).catch((error: Error) => error);
  expect(modelCalls).toBe(1); safe(result); safe(run.getInterrupt() ?? {});
  for await (const tuple of saver.list(config)) { safe(tuple.checkpoint.channel_values); safe(tuple.pendingWrites ?? []); }
  if (mode === 'control') { expect(result).toBeUndefined(); expect(run.getInterrupt()).toBeDefined(); } else expect(result).toBeInstanceOf(ToolResultProtectionError);
});

it.each(['selected', 'allowed'] as const)('admits deprecated native NodeInterrupt payloads (%s)', async (mode) => {
  const error = new NodeInterrupt(mode === 'selected' ? { selected: selected() } : { allowed: 'Allowed data' });
  const observations: string[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { observations.push(JSON.stringify(error)); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  safe(observations); safe(result); expect(result).toBeInstanceOf(mode === 'selected' ? ToolResultProtectionError : GraphInterrupt);
});

it('gates selected descendants appended to a payload-free unselected interrupt', async () => {
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { if (error instanceof GraphInterrupt) error.interrupts.push({ value: { selected: selected() } }); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw new GraphInterrupt([]); } });
  const result = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result);
});

it.each(['producer', 'callback'] as const)('rejects inherited control-result fields (%s)', async (mode) => {
  const error = control('interrupt', { nested: 'Allowed payload' });
  const expand = (): void => { Object.defineProperty(GraphInterrupt.prototype, '_c1ControlExtra', { value: selected(), configurable: true }); };
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { if (mode === 'callback' && error instanceof GraphInterrupt) expand(); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { if (mode === 'producer') expand(); throw error; } });
  let result: unknown;
  try { result = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error); }
  finally { Reflect.deleteProperty(GraphInterrupt.prototype, '_c1ControlExtra'); }
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(result);
});

it('gates uncertified native control-error kinds before their result-bearing fields reach callbacks', async () => {
  const error = new GraphDrained('Allowed reason'); Object.defineProperty(error, 'reason', { value: { selected: selected() } });
  const observed: string[] = [];
  const callback = BaseCallbackHandler.fromMethods({ handleToolError: (error): void => { observed.push(JSON.stringify(error)); } }); callback.awaitHandlers = true;
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed control', schema: z.object({}), func: async () => { throw error; } });
  const result = await new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() }).invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(result).toBeInstanceOf(ToolResultProtectionError); safe(observed); safe(result);
});

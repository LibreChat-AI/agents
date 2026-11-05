import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { Command, Send, MemorySaver } from '@langchain/langgraph';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolResultProtection } from '../toolResult';
import { getToolMessageRelease, snapshotToolSend, validateToolMessageHistory, ToolResultProtectionError } from '../toolResult';
import { linkStreamLimitCanonical } from '@/llm/streamLimits';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Providers } from '@/common';
import { Run } from '@/run';

const raw = 'history-payload@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 10000, maxAttemptBytes: 65536, maxBufferedBytes: 262144, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = (): ToolMessage => new ToolMessage({ name: 'lookup', tool_call_id: 'nested-call', content: raw });
const assertSafe = (value: unknown): void => { expect(String(value)).not.toContain(raw); expect(JSON.stringify(value)).not.toContain(raw); };

it.each(['extra', 'symbol', 'getter', 'callback', 'control'] as const)('closes Send-envelope fields before native release (%s)', async (mode) => {
  let reads = 0, observed = 0;
  const packet = new Send('default', { messages: [new HumanMessage('Allowed input')] });
  if (mode === 'extra') Object.defineProperty(packet, 'extra', { value: selected() });
  if (mode === 'symbol') Object.defineProperty(packet, Symbol('extra'), { value: selected() });
  if (mode === 'getter') Object.defineProperty(packet, 'extra', { get: () => { reads++; throw new Error(raw); } });
  const tool = new DynamicStructuredTool({ name: 'other', description: 'Allowed routing control.', schema: z.object({}), func: async () => new Command({ graph: Command.PARENT, goto: [packet] }) });
  const callback = BaseCallbackHandler.fromMethods({ handleToolEnd: (output): void => { observed++; if (mode === 'callback') Object.defineProperty(((output as Command).goto as Send[])[0], 'extra', { value: selected() }); } }); callback.awaitHandlers = true;
  const node = new ToolNode({ tools: [tool], trace: true, toolResultProtection: policy() });
  const outcome = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'outer-call', name: 'other', args: {} }] })] }, { callbacks: [callback] }).catch((error: Error) => error);
  expect(reads).toBe(0); assertSafe(outcome);
  if (mode === 'control') { expect(outcome).not.toBeInstanceOf(Error); expect(snapshotToolSend(packet)).toBeInstanceOf(Send); }
  else { expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(mode === 'callback' ? 1 : 0); }
});

const fields = ['content', 'additional_kwargs', 'response_metadata', 'metadata', 'serialization'] as const;
function nestedHistory(field: typeof fields[number], kind: 'human' | 'ai' | 'system' | 'tool' = 'human'): BaseMessage {
  const Message = { human: HumanMessage, ai: AIMessage, system: SystemMessage, tool: ToolMessage }[kind];
  const message = new Message({ content: 'Allowed history', name: 'other', tool_call_id: 'other-call' });
  const value = { type: 'text', text: 'Allowed prose', nested: selected() };
  if (field === 'content') message.content = [value];
  else if (field === 'serialization') message.lc_kwargs.custom = value;
  else Object.defineProperty(message, field, { value: { value }, enumerable: true, configurable: true, writable: true });
  return message;
}
it.each(fields.flatMap((field) => (['human', 'ai', 'system', 'tool'] as const).map((kind) => [field, kind] as const)))('rejects nested %s in %s history', async (field, kind) => {
  const message = nestedHistory(field, kind);
  const outcome = await validateToolMessageHistory(policy(), [message], []).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); assertSafe(outcome);
});

it.each(['fresh', 'saver', 'pending', 'resume', 'send', 'window'] as const)('gates nested history payloads before real Run observers (%s)', async (carrier) => {
  let tamper = false, observed = 0;
  class Saver extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(config); if (!tamper || tuple == null) return tuple;
      if (carrier === 'saver') tuple.checkpoint.channel_values.messages = [...tuple.checkpoint.channel_values.messages as BaseMessage[], nestedHistory('additional_kwargs')];
      if (carrier === 'pending') tuple.pendingWrites = [['task', 'messages', [nestedHistory('additional_kwargs')]]];
      return tuple;
    }
  }
  const saver = new Saver(), runId = 'nested-history-' + carrier, hooks = new HookRegistry();
  if (['resume', 'send', 'pending'].includes(carrier)) hooks.register('PreToolUse', { hooks: [async () => ({ decision: 'ask', reason: 'Allowed approval' })] });
  const tool = new DynamicStructuredTool({ name: 'lookup', description: 'Allowed control.', schema: z.object({}), func: async () => 'Allowed tool output' });
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, tools: [tool], compileOptions: { checkpointer: saver } }, toolResultProtection: policy(), hooks, humanInTheLoop: { enabled: true } });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['', 'Allowed answer'], toolCalls: [{ id: 'approval-call', name: 'lookup', args: {} }] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  if (carrier !== 'fresh' && carrier !== 'window') await run.processStream({ messages: [new HumanMessage('Allowed input')] }, config);
  tamper = true;
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observed++; } }); callback.awaitHandlers = true;
  const options = { ...config, callbacks: [callback] };
  const invoke = async (): Promise<unknown> => {
    if (carrier === 'resume') return run.resume([{ type: 'approve' }], options, undefined, { update: { messages: [nestedHistory('additional_kwargs')] } });
    if (carrier === 'send') return run.resume([{ type: 'approve' }], options, undefined, { goto: [new Send('default', { messages: [nestedHistory('additional_kwargs')] })] });
    if (carrier === 'pending') return run.resume([{ type: 'approve' }], options);
    if (carrier === 'window') return run.Graph!.validateToolResultWindow({ agentId: 'default', remainingContextTokens: 100, windowMessages: [nestedHistory('additional_kwargs')] });
    return run.processStream({ messages: [carrier === 'fresh' ? nestedHistory('additional_kwargs') : new HumanMessage('Allowed next input')] }, options);
  };
  const outcome = await invoke().catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observed).toBe(0); assertSafe(outcome);
});

it('detaches valid nested metadata but preserves actual assistant argument identity', async () => {
  const data = { nested: { value: 'Allowed metadata' } }, args = { opaque: selected() };
  const message = new AIMessage({ content: 'Allowed history', additional_kwargs: { data }, tool_calls: [{ id: 'other-call', name: 'other', args }] });
  const result = await validateToolMessageHistory(policy(), [message], []);
  expect(result[0]).not.toBe(message); expect((result[0] as AIMessage).tool_calls![0].args).toBe(args);
  const before = JSON.stringify(result); data.nested.value = raw; expect(JSON.stringify(result)).toBe(before);
});

it('does not inspect nested payloads when protection is off', async () => {
  const message = nestedHistory('additional_kwargs'); const result = await validateToolMessageHistory(undefined, [message], []); expect(result[0]).toBe(message);
});

it('closes Send envelopes at checkpoint and graph-admission entry points', async () => {
  const packet = Object.assign(new Send('default', { messages: [] }), { extra: selected() });
  const graph = new StandardGraph({ agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
  await expect(graph.validateToolResultSend(packet)).rejects.toBeInstanceOf(ToolResultProtectionError);
});

it.each(['during-inspection', 'late-alias', 'control'] as const)('rechecks and detaches history payloads around policy awaits (%s)', async (mode) => {
  let reinspect = false;
  const started = Promise.withResolvers<void>(); const resume = Promise.withResolvers<void>();
  const gate: ToolResultProtection = { ...policy(), inspect: async ({ content }) => { if (reinspect) { started.resolve(); await resume.promise; } return { version: 1, ok: true, value: { content, replacements: 0, categories: [] } }; } };
  const producer = new ToolNode({ tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed result.', schema: z.object({}), func: async () => 'Allowed selected result' })], toolResultProtection: gate });
  const proposal = new AIMessage({ content: '', tool_calls: [{ id: 'reinspect-call', name: 'lookup', args: {} }] });
  const produced = await producer.invoke({ messages: [proposal] }) as { messages: ToolMessage[] };
  const released = produced.messages[0], receipt = getToolMessageRelease(gate, 'lookup', 'reinspect-call', released)!;
  const historical = new ToolMessage({ name: released.name, tool_call_id: released.tool_call_id, content: released.content, status: released.status });
  const retained = { value: 'Allowed metadata' };
  const supplied = new HumanMessage({ content: 'Allowed history', additional_kwargs: { retained } });
  reinspect = true;
  const task = validateToolMessageHistory(gate, [proposal, historical, supplied], [receipt]).catch((error: Error) => error);
  await started.promise;
  if (mode === 'during-inspection') Object.defineProperty(retained, 'selected', { value: selected() });
  resume.resolve();
  const result = await task;
  if (mode === 'late-alias') retained.value = raw;
  assertSafe(result);
  if (mode === 'during-inspection') expect(result).toBeInstanceOf(ToolResultProtectionError);
  else { expect(result).not.toBeInstanceOf(Error); expect((result as BaseMessage[])[2].additional_kwargs.retained).toEqual({ value: 'Allowed metadata' }); }
});

it('retains live certified nested results without executing their producer again', async () => {
  let executions = 0;
  const gate = policy();
  const producer = new ToolNode({ tools: [new DynamicStructuredTool({ name: 'lookup', description: 'Allowed result.', schema: z.object({}), func: async () => { executions++; return 'Allowed selected result'; } })], toolResultProtection: gate });
  const output = await producer.invoke({ messages: [new AIMessage({ content: '', tool_calls: [{ id: 'nested-call', name: 'lookup', args: {} }] })] }) as { messages: ToolMessage[] };
  const supplied = new HumanMessage({ content: 'Allowed history', additional_kwargs: { nested: output.messages[0] } });
  const admitted = await validateToolMessageHistory(gate, [supplied], []);
  expect(executions).toBe(1); expect(admitted[0].additional_kwargs.nested).not.toBe(output.messages[0]); expect((admitted[0].additional_kwargs.nested as ToolMessage).content).toBe('Allowed selected result');
});

it.each(['nameless', 'renamed', 'getter', 'symbol'] as const)('binds nested history results to enclosing call sources (%s)', async (kind) => {
  let reads = 0;
  const nested = new ToolMessage({ ...(kind === 'nameless' ? {} : { name: kind === 'renamed' ? 'other' : 'lookup' }), tool_call_id: 'nested-call', content: raw });
  const human = new HumanMessage({ content: 'Allowed history', additional_kwargs: { nested } });
  if (kind === 'getter') Object.defineProperty(human.additional_kwargs, 'nested', { get: () => { reads++; throw new Error(raw); } });
  if (kind === 'symbol') Object.defineProperty(human.additional_kwargs, Symbol('nested'), { value: nested });
  const proposal = new AIMessage({ id: 'nested-batch', content: '', tool_calls: [{ id: 'nested-call', name: 'lookup', args: {} }] });
  const outcome = await validateToolMessageHistory(policy(), [proposal, human], []).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(reads).toBe(0); assertSafe(outcome);
});

it('omits emission-only canonical links without exposing the linked message', async () => {
  const message = new AIMessage({ content: 'Allowed history' });
  const emission = new AIMessage({ content: raw }); linkStreamLimitCanonical(message, emission);
  const result = await validateToolMessageHistory(policy(), [message], []);
  expect(Reflect.ownKeys(result[0]).filter((key) => typeof key === 'symbol')).toEqual(Reflect.ownKeys(new AIMessage('Allowed')).filter((key) => typeof key === 'symbol'));
  assertSafe(result);
});

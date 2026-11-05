import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { MemorySaver, emptyCheckpoint, Send } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolResultProtection } from '@/protection/toolResult';
import { ToolResultProtectionError } from '@/protection/toolResult';
import { StandardGraph } from '@/graphs/Graph';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { Run } from '@/run';

const raw = 'unchecked-channel@example.invalid';
const policy = (): ToolResultProtection => ({ version: 1, toolNames: ['lookup'], timeoutMs: 1000, maxAttemptBytes: 65536, maxBufferedBytes: 65536, classify: () => 'prose', inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }) });
const selected = () => new ToolMessage({ name: 'lookup', tool_call_id: 'selected-call', content: raw });
const graph = () => new StandardGraph({ runId: 'channel-admission', agents: [{ agentId: 'default', provider: Providers.OPENAI }], toolResultProtection: policy() });
const badChannels = ['manualSummary', 'custom', 'branch:to:default'];
it.each(badChannels.flatMap((channel) => (['values', 'pending', 'snapshot', 'send'] as const).map((carrier) => [channel, carrier] as const)))('admits every %s channel (%s)', async (channel, carrier) => {
  const value = selected(); const instance = graph();
  const invoke = async (): Promise<unknown> => {
    if (carrier === 'snapshot') return instance.validateToolResultCheckpoint({ values: { messages: [], [channel]: value }, tasks: [] });
    if (carrier === 'send') return instance.validateToolResultSend(new Send('default', { messages: [], [channel]: value }));
    const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [], ...(carrier === 'values' ? { [channel]: value } : {}) };
    return instance.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'channel-admission', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: carrier === 'pending' ? [['task', channel, value]] : [] });
  };
  const outcome = await invoke().catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(String(outcome)).not.toContain(raw);
});

it.each(['values', 'pending'] as const)('rejects unknown string channel values before native loading (%s)', async (carrier) => {
  const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [], ...(carrier === 'values' ? { custom: raw } : {}) };
  await expect(graph().validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'channel-admission', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: carrier === 'pending' ? [['task', 'custom', raw]] : [] })).rejects.toBeInstanceOf(ToolResultProtectionError);
});

it.each(['values', 'pending', 'snapshot', 'send'] as const)('keeps manual summary primitives and native branch controls usable (%s)', async (carrier) => {
  const instance = graph(); const state = { messages: [new HumanMessage('Allowed input')], manualSummary: 'Allowed summary' };
  if (carrier === 'snapshot') { const result = await instance.validateToolResultCheckpoint({ values: state, tasks: [] }); expect(result.values.manualSummary).toBe(state.manualSummary); return; }
  if (carrier === 'send') { const result = await instance.validateToolResultSend(new Send('default', state)); expect(result.args.manualSummary).toBe(state.manualSummary); return; }
  const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { ...state, 'branch:to:default': null };
  const result = await instance.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'channel-admission', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: carrier === 'pending' ? [['task', 'manualSummary', state.manualSummary], ['task', 'branch:to:default', null]] : [] });
  expect(result.checkpoint.channel_values.manualSummary).toBe(state.manualSummary);
});

it.each(['manualSummary', 'custom'] as const)('rejects native saver %s channels before callbacks', async (channel) => {
  let tamper = false, observations = 0;
  class Saver extends MemorySaver {
    override async getTuple(config: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(config); if (tamper && tuple != null) tuple.checkpoint.channel_values[channel] = selected(); return tuple;
    }
  }
  const saver = new Saver(); const runId = `native-channel-${channel}`;
  const create = () => Run.create({ runId, graphConfig: { type: 'standard' as const, llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: saver } }, toolResultProtection: policy() });
  const first = await create(); first.Graph!.overrideModel = new FakeChatModel({ responses: ['Allowed answer'] });
  const config = { version: 'v2' as const, configurable: { thread_id: runId } };
  await first.processStream({ messages: [new HumanMessage('Allowed input')] }, config);
  tamper = true; const next = await create(); next.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observations++; } }); callback.awaitHandlers = true;
  const outcome = await next.processStream({ messages: [new HumanMessage('Allowed next input')] }, { ...config, callbacks: [callback] }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observations).toBe(0);
});

it.each(['manualSummary', 'custom', '__start__', '__resume__'] as const)('rejects invalid %s in fresh inputs before startup callbacks', async (channel) => {
  let observations = 0;
  const run = await Run.create({ runId: `fresh-channel-${channel}`, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI } }, toolResultProtection: policy() });
  run.Graph!.overrideModel = new FakeChatModel({ responses: ['Must not run'] });
  const callback = BaseCallbackHandler.fromMethods({ handleChainStart: (): void => { observations++; } }); callback.awaitHandlers = true;
  const outcome = await run.processStream({ messages: [new HumanMessage('Allowed input')], [channel]: selected() }, { version: 'v2', callbacks: [callback] }).catch((error: Error) => error);
  expect(outcome).toBeInstanceOf(ToolResultProtectionError); expect(observations).toBe(0);
});

it.each(['values', 'pending', 'snapshot', 'send'] as const)('detaches inert custom channel data before release (%s)', async (carrier) => {
  const authority = { principal: 'allowed-user', resources: [{ id: 'allowed-resource' }] };
  const instance = graph(); const state = { messages: [], authority }; let returned: typeof authority;
  if (carrier === 'snapshot') returned = (await instance.validateToolResultCheckpoint({ values: state, tasks: [] })).values.authority;
  else if (carrier === 'send') returned = (await instance.validateToolResultSend(new Send('default', state))).args.authority as typeof authority;
  else {
    const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [], ...(carrier === 'values' ? { authority } : {}) };
    const outcome = await instance.validateToolResultCheckpointTuple({ config: { configurable: { thread_id: 'channel-admission', checkpoint_id: checkpoint.id } }, checkpoint, pendingWrites: carrier === 'pending' ? [['task', 'authority', authority]] : [] });
    returned = (carrier === 'pending' ? outcome.pendingWrites![0][2] : outcome.checkpoint.channel_values.authority) as typeof authority;
  }
  expect(returned).toEqual(authority); expect(returned).not.toBe(authority); authority.resources[0].id = raw; expect(returned.resources[0].id).toBe('allowed-resource');
});

it.each(['branch:to:default', 'join:default:default'] as const)('admits only compiled routing source identities (%s)', async (channel) => {
  const instance = graph(); instance.createWorkflow();
  const checkpoint = emptyCheckpoint(); checkpoint.channel_values = { messages: [], [channel]: channel.startsWith('join:') ? ['default'] : 'default' };
  const config = { configurable: { thread_id: 'channel-admission', checkpoint_id: checkpoint.id } };
  await expect(instance.validateToolResultCheckpointTuple({ config, checkpoint })).resolves.toBeDefined();
  checkpoint.channel_values[channel] = channel.startsWith('join:') ? [raw] : raw;
  await expect(instance.validateToolResultCheckpointTuple({ config, checkpoint })).rejects.toBeInstanceOf(ToolResultProtectionError);
});

import { MemorySaver, emptyCheckpoint } from '@langchain/langgraph';
import type { BaseCheckpointSaver } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import { toolCheckpointSafety, bindToolResultCheckpoint, ToolResultProtectionError } from './toolResult';
import { ToolCheckpointSaver } from './toolCheckpoint';

const thread = 'admitted-checkpoints';
const config = (namespace = '', id?: string): RunnableConfig => ({
  configurable: { thread_id: thread, checkpoint_ns: namespace, ...(id == null ? {} : { checkpoint_id: id }) },
});

async function put(saver: BaseCheckpointSaver, namespace: string, id: string, parent?: string): Promise<RunnableConfig> {
  const checkpoint = emptyCheckpoint();
  checkpoint.id = id;
  checkpoint.channel_values = { control: 'Allowed checkpoint control' };
  return saver.put(config(namespace, parent), checkpoint, { source: 'input', step: 0, parents: {} }, {});
}

it('routes every admitted descendant and preserves explicit backend reads', async () => {
  const saver = new MemorySaver();
  for (const namespace of ['', 'agent:task', 'agent:task|nested:task']) {
    await put(saver, namespace, '001');
    await put(saver, namespace, '002', '001');
  }
  const snapshot = {
    config: config('', '001'), values: {}, tasks: [{ state: {
      config: config('agent:task', '001'), values: {}, tasks: [{ state: {
        config: config('agent:task|nested:task', '001'), values: {}, tasks: [],
      } }],
    } }],
  };
  const caller = config();
  const bound = bindToolResultCheckpoint(caller, snapshot, saver);
  const scoped = bound.configurable?.__pregel_checkpointer as ToolCheckpointSaver;
  expect(scoped).toBeInstanceOf(ToolCheckpointSaver);
  expect(bound.configurable?.checkpoint_map).toEqual({ '': '001' });
  expect(caller.configurable).not.toHaveProperty('checkpoint_map');
  for (const namespace of ['', 'agent:task', 'agent:task|nested:task']) {
    expect((await scoped.getTuple(config(namespace)))?.checkpoint.id).toBe('001');
    expect((await saver.getTuple(config(namespace)))?.checkpoint.id).toBe('002');
    expect((await scoped.getTuple(config(namespace, '002')))?.checkpoint.id).toBe('002');
  }
});

it('does not load later-created root or descendant state admitted as empty', async () => {
  const saver = new MemorySaver();
  const emptyRoot = bindToolResultCheckpoint(config(), { config: config(), values: {}, tasks: [] }, saver).configurable?.__pregel_checkpointer as ToolCheckpointSaver;
  await put(saver, '', '001');
  const bound = bindToolResultCheckpoint(config(), { config: config('', '001'), values: {}, tasks: [{ state: { config: config('agent:task'), values: {}, tasks: [] } }] }, saver);
  const scoped = bound.configurable?.__pregel_checkpointer as ToolCheckpointSaver;
  await put(saver, '', 'new-root');
  await put(saver, 'agent:task', 'new-descendant');
  expect(await emptyRoot.getTuple(config())).toBeUndefined();
  expect((await scoped.getTuple(config()))?.checkpoint.id).toBe('001');
  expect(await scoped.getTuple(config('agent:task'))).toBeUndefined();
});

it.each(['mismatched', 'unknown', 'foreign-thread', 'foreign-namespace'])('rejects an unverified descendant selection (%s)', (kind) => {
  const saver = new MemorySaver();
  const caller = config();
  const child = { config: config('agent:task', 'admitted'), values: {}, tasks: [] };
  if (kind === 'mismatched') caller.configurable!.checkpoint_map = { 'agent:task': 'not-admitted' };
  if (kind === 'unknown') caller.configurable!.checkpoint_map = { 'missing:task': 'not-admitted' };
  if (kind === 'foreign-thread') child.config = { configurable: { thread_id: 'foreign', checkpoint_ns: 'agent:task', checkpoint_id: 'admitted' } };
  const nested = { config: config(kind === 'foreign-namespace' ? 'unrelated:task' : 'agent:task|nested:task', 'admitted'), values: {}, tasks: [] };
  const snapshot = { config: config('', 'root'), values: {}, tasks: [{ state: { ...child, tasks: [{ state: nested }] } }] };
  expect(() => bindToolResultCheckpoint(caller, snapshot, saver)).toThrow(ToolResultProtectionError);
});

it('allows verified descendant cursors without native descendant time-travel selection', () => {
  const saver = new MemorySaver();
  const caller = { ...config(), configurable: { ...config().configurable, checkpoint_map: { 'agent:task': 'admitted' } } };
  const bound = bindToolResultCheckpoint(caller, { config: config('', 'root'), values: {}, tasks: [{ state: { config: config('agent:task', 'admitted'), values: {}, tasks: [] } }] }, saver);
  expect(bound.configurable.checkpoint_map).toEqual({ '': 'root' });
  expect(caller.configurable.checkpoint_map).toEqual({ 'agent:task': 'admitted' });
});

it('follows this execution’s own writes, not a competing writer’s later head', async () => {
  const saver = new MemorySaver();
  await put(saver, 'agent:task', '001');
  const scoped = new ToolCheckpointSaver(saver, thread, new Map([['agent:task', '001']]), toolCheckpointSafety);
  await put(saver, 'agent:task', '999', '001');
  expect((await scoped.getTuple(config('agent:task')))?.checkpoint.id).toBe('001');
  await put(scoped, 'agent:task', '002', '001');
  expect((await scoped.getTuple(config('agent:task')))?.checkpoint.id).toBe('002');
  expect((await saver.getTuple(config('agent:task')))?.checkpoint.id).toBe('999');
  expect((await scoped.getTuple({ configurable: { thread_id: 'another-thread', checkpoint_ns: 'agent:task' } }))).toBeUndefined();
});

it('delegates serializer, writes, history, delta reconstruction and deletion to the original backend', async () => {
  const saver = new MemorySaver();
  await put(saver, '', '001');
  const scoped = new ToolCheckpointSaver(saver, thread, new Map([['', '001']]), toolCheckpointSafety);
  expect(scoped.serde).toBe(saver.serde);
  const version = scoped.getNextVersion.bind(saver);
  expect(version(4)).toBe(saver.getNextVersion(4));
  await scoped.putWrites(config('', '001'), [['control', 'Allowed write']], 'task-control');
  expect((await saver.getTuple(config('', '001')))?.pendingWrites).toContainEqual(['task-control', 'control', 'Allowed write']);
  const stored = [];
  for await (const tuple of scoped.list(config())) stored.push(tuple.checkpoint.id);
  expect(stored).toEqual(['001']);
  const args = { config: config('', '001'), channels: ['control'] };
  expect(await scoped.getDeltaChannelHistory(args)).toEqual(await saver.getDeltaChannelHistory(args));
  await scoped.deleteThread(thread);
  expect(await saver.getTuple(config())).toBeUndefined();
});

it.each(['messages', 'agentMessages', 'runStepState', 'summarizationRequest', '__pregel_tasks', '__pregel_push'])('gates uncertified delta reconstruction of protected %s before calling the backend', async (channel) => {
  const saver = new MemorySaver(); await put(saver, '', '001');
  const source = jest.spyOn(saver, 'getDeltaChannelHistory');
  const scoped = new ToolCheckpointSaver(saver, thread, new Map([['', '001']]), toolCheckpointSafety);
  await expect(async () => scoped.getDeltaChannelHistory({ config: config(), channels: [channel] })).rejects.toMatchObject({ code: 'unsupported' }); expect(source).not.toHaveBeenCalled();
});

it('routes ordinary delta reads through the admitted cursor without rewriting backend data', async () => {
  const saver = new MemorySaver(); await put(saver, '', '001'); await put(saver, '', '999', '001');
  const source = jest.spyOn(saver, 'getDeltaChannelHistory');
  const scoped = new ToolCheckpointSaver(saver, thread, new Map([['', '001']]), toolCheckpointSafety);
  const read = await scoped.getDeltaChannelHistory({ config: config(), channels: ['control'] });
  expect(source).toHaveBeenCalledWith({ config: config('', '001'), channels: ['control'] });
  expect(read).toEqual(await saver.getDeltaChannelHistory({ config: config('', '001'), channels: ['control'] }));
});

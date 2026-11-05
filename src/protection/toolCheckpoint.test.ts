import { END, START, Annotation, StateGraph, MemorySaver, emptyCheckpoint } from '@langchain/langgraph';
import type { BaseCheckpointSaver, CheckpointTuple } from '@langchain/langgraph';
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

it.each((['explicit', 'latest', 'list'] as const).flatMap((route) => (['thread', 'namespace', 'config-id', 'body-id', 'both-ids', 'missing-id', 'id-getter', 'parent-thread', 'parent-namespace', 'mutate-lookup', 'control'] as const).map((change) => [route, change] as const)))('binds returned tuples before content admission (%s/%s)', async (route, change) => {
  let reads = 0;
  const alter = (tuple: NonNullable<Awaited<ReturnType<MemorySaver['getTuple']>>>): typeof tuple => {
    const scope = { ...tuple.config.configurable }; tuple.config = { ...tuple.config, configurable: scope };
    if (change === 'thread') scope.thread_id = 'foreign-thread';
    if (change === 'namespace') scope.checkpoint_ns = 'foreign:namespace';
    if (change === 'config-id' || change === 'both-ids') scope.checkpoint_id = 'foreign-id';
    if (change === 'body-id' || change === 'both-ids') tuple.checkpoint.id = 'foreign-id';
    if (change === 'missing-id') Reflect.deleteProperty(scope, 'checkpoint_id');
    if (change === 'id-getter') Object.defineProperty(tuple.checkpoint, 'id', { get: () => { reads++; throw new Error('Unreleased checkpoint diagnostic'); } });
    if (change === 'parent-thread') tuple.parentConfig = { configurable: { thread_id: 'foreign-thread', checkpoint_ns: 'agent:task', checkpoint_id: 'parent-id' } };
    if (change === 'parent-namespace') tuple.parentConfig = { configurable: { thread_id: thread, checkpoint_ns: 'foreign:namespace', checkpoint_id: 'parent-id' } };
    return tuple;
  };
  class Saver extends MemorySaver {
    override async getTuple(lookup: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(lookup);
      if (tuple == null) return tuple;
      if (change === 'mutate-lookup') { lookup.configurable!.thread_id = 'mutated-thread'; tuple.config = lookup; }
      return alter(tuple);
    }
    override async *list(...args: Parameters<MemorySaver['list']>): ReturnType<MemorySaver['list']> {
      for await (const tuple of super.list(...args)) {
        if (change === 'mutate-lookup') { args[0].configurable!.thread_id = 'mutated-thread'; tuple.config = args[0]; }
        yield alter(tuple);
      }
    }
  }
  const saver = new Saver(); await put(saver, 'agent:task', '001');
  let admissions = 0; const scoped = new ToolCheckpointSaver(saver, thread, new Map(), toolCheckpointSafety, async (tuple) => { admissions++; return tuple; });
  const lookup = config('agent:task', route === 'latest' ? undefined : '001');
  const operation = async (): Promise<void> => { if (route !== 'list') { expect(await scoped.getTuple(lookup)).toBeDefined(); return; } for await (const tuple of scoped.list(lookup)) expect(tuple).toBeDefined(); };
  const result = await operation().catch((error: Error) => error);
  const valid = change === 'control' || (route === 'latest' && change === 'both-ids');
  if (valid) { expect(result).toBeUndefined(); expect(admissions).toBe(1); }
  else { expect(result).toBeInstanceOf(ToolResultProtectionError); expect(admissions).toBe(0); }
  expect(reads).toBe(0); expect(lookup.configurable?.thread_id).toBe(thread);
});

it('preserves unconstrained history within a thread, explicit selection, latest and absent results', async () => {
  const saver = new MemorySaver();
  for (const namespace of ['', 'agent:task']) { await put(saver, namespace, '001'); await put(saver, namespace, '002', '001'); }
  const scoped = new ToolCheckpointSaver(saver, thread, new Map(), toolCheckpointSafety);
  expect((await scoped.getTuple(config()))?.checkpoint.id).toBe('002');
  expect((await scoped.getTuple(config('', '001')))?.checkpoint.id).toBe('001');
  expect(await scoped.getTuple(config('', 'absent'))).toBeUndefined();
  const history = [];
  for await (const tuple of scoped.list({ configurable: { thread_id: thread } })) history.push([tuple.config.configurable?.checkpoint_ns, tuple.checkpoint.id]);
  expect(history).toEqual(expect.arrayContaining([['', '001'], ['', '002'], ['agent:task', '001'], ['agent:task', '002']])); expect(history).toHaveLength(4);
});

it.each(['thread_id', 'checkpoint_ns', 'checkpoint_id', 'thread_ts', 'checkpoint', 'config', 'parentConfig'] as const)('rejects returned %s identity accessors before content admission', async (field) => {
  let reads = 0; let admissions = 0;
  class Saver extends MemorySaver {
    override async getTuple(lookup: RunnableConfig): ReturnType<MemorySaver['getTuple']> {
      const tuple = await super.getTuple(lookup); if (tuple == null) return tuple;
      const object = ['checkpoint', 'config', 'parentConfig'].includes(field) ? tuple : tuple.config.configurable!;
      Object.defineProperty(object, field, { get: () => { reads++; throw new Error('Unreleased identity diagnostic'); } }); return tuple;
    }
  }
  const saver = new Saver(); await put(saver, '', '001');
  const scoped = new ToolCheckpointSaver(saver, thread, new Map(), toolCheckpointSafety, async (tuple) => { admissions++; return tuple; });
  await expect(scoped.getTuple(config())).rejects.toMatchObject({ code: 'unsupported' }); expect(reads).toBe(0); expect(admissions).toBe(0);
});

it('matches a legacy checkpoint selector and rejects conflicting returned aliases', async () => {
  let tamper = false;
  class Saver extends MemorySaver { override async getTuple(lookup: RunnableConfig): ReturnType<MemorySaver['getTuple']> { const tuple = await super.getTuple(lookup); if (tuple != null && tamper) tuple.config = { configurable: { ...tuple.config.configurable, checkpoint_id: 'different' } }; return tuple; } }
  const saver = new Saver(); await put(saver, '', '001'); const scoped = new ToolCheckpointSaver(saver, thread, new Map(), toolCheckpointSafety);
  const lookup = { configurable: { thread_id: thread, checkpoint_ns: '', thread_ts: '001' } };
  expect((await scoped.getTuple(lookup))?.checkpoint.id).toBe('001'); tamper = true;
  await expect(scoped.getTuple(lookup)).rejects.toMatchObject({ code: 'incompatible' });
});

it.each(['read', 'history', 'delta'] as const)('keeps saver-side lookup mutation detached from caller identity (%s)', async (route) => {
  const caller = { configurable: { thread_id: thread, checkpoint_ns: '', checkpoint_id: '001', user_id: 'allowed-principal', checkpoint_map: { 'agent:task': 'admitted-child' } } };
  const mutate = (lookup: RunnableConfig): void => { lookup.configurable!.user_id = 'untrusted-principal'; (lookup.configurable!.checkpoint_map as Record<string, string>)['agent:task'] = 'untrusted-child'; };
  class Saver extends MemorySaver {
    override async getTuple(lookup: RunnableConfig): ReturnType<MemorySaver['getTuple']> { const tuple = await super.getTuple(lookup); mutate(lookup); return tuple; }
    override async *list(...args: Parameters<MemorySaver['list']>): ReturnType<MemorySaver['list']> { for await (const tuple of super.list(...args)) { mutate(args[0]); yield tuple; } }
    override async getDeltaChannelHistory(options: Parameters<MemorySaver['getDeltaChannelHistory']>[0]): ReturnType<MemorySaver['getDeltaChannelHistory']> { mutate(options.config); return super.getDeltaChannelHistory(options); }
  }
  const saver = new Saver(); await put(saver, '', '001'); const scoped = new ToolCheckpointSaver(saver, thread, new Map(), toolCheckpointSafety);
  if (route === 'read') expect(await scoped.getTuple(caller)).toBeDefined();
  if (route === 'history') for await (const tuple of scoped.list(caller)) expect(tuple).toBeDefined();
  if (route === 'delta') await scoped.getDeltaChannelHistory({ config: caller, channels: ['control'] });
  expect(caller.configurable).toEqual({ thread_id: thread, checkpoint_ns: '', checkpoint_id: '001', user_id: 'allowed-principal', checkpoint_map: { 'agent:task': 'admitted-child' } });
});

describe('selected checkpoint history', () => {
  class ThreadHistorySaver extends MemorySaver {
    override async *list(lookup: RunnableConfig, options?: Parameters<MemorySaver['list']>[1]): ReturnType<MemorySaver['list']> {
      const scope = { ...lookup.configurable };
      delete scope.checkpoint_id;
      delete scope.thread_ts;
      yield* super.list({ ...lookup, configurable: scope }, options);
    }
  }

  const selected = (selector: 'checkpoint_id' | 'thread_ts', id = '003', namespace = ''): RunnableConfig => ({
    configurable: { thread_id: thread, checkpoint_ns: namespace, [selector]: id },
  });
  const collect = async (tuples: AsyncIterable<CheckpointTuple>): Promise<CheckpointTuple[]> => {
    const result: CheckpointTuple[] = [];
    for await (const tuple of tuples) result.push(tuple);
    return result;
  };
  const seed = async (saver: BaseCheckpointSaver, namespace = ''): Promise<void> => {
    await put(saver, namespace, '001');
    await put(saver, namespace, '002', '001');
    await put(saver, namespace, '003', '002');
  };

  it.each(['checkpoint_id', 'thread_ts'] as const)('preserves installed MemorySaver and native getStateHistory semantics for %s', async (selector) => {
    const source = new MemorySaver();
    await seed(source);
    const ids = (tuples: CheckpointTuple[]): string[] => tuples.map((tuple) => tuple.checkpoint.id);
    expect(ids(await collect(source.list(selected(selector, '002'))))).toEqual(selector === 'checkpoint_id' ? ['002'] : ['003', '002', '001']);
    const admitted: string[] = [];
    const saver = new ToolCheckpointSaver(source, thread, new Map([['', '001']]), toolCheckpointSafety, async (tuple) => {
      admitted.push(tuple.checkpoint.id);
      return tuple;
    });
    const state = Annotation.Root({ control: Annotation<string>() });
    const workflow = new StateGraph(state).addNode('noop', (value) => value).addEdge(START, 'noop').addEdge('noop', END).compile({ checkpointer: source });
    const native: string[] = [];
    for await (const snapshot of workflow.getStateHistory(selected(selector))) native.push(snapshot.config.configurable!.checkpoint_id as string);
    expect(native).toEqual(selector === 'checkpoint_id' ? ['003'] : ['003', '002', '001']);
    const protectedHistory: string[] = [];
    for await (const snapshot of workflow.getStateHistory({ configurable: { ...selected(selector).configurable, __pregel_checkpointer: saver } })) protectedHistory.push(snapshot.config.configurable!.checkpoint_id as string);
    expect(protectedHistory).toEqual(native);
    expect(admitted).toEqual(native);
    expect((await saver.getTuple(config()))?.checkpoint.id).toBe('001');
    expect((await saver.getTuple(selected(selector, '002')))?.checkpoint.id).toBe('002');
    await expect(collect(saver.list(selected('thread_ts', '002')))).rejects.toMatchObject({ code: 'incompatible' });
  });

  it.each(['checkpoint_id', 'thread_ts'] as const)('admits older scoped history after a validated %s anchor without advancing the execution cursor', async (selector) => {
    const source = new ThreadHistorySaver();
    await seed(source);
    await seed(source, 'agent:task');
    const admitted: string[] = [];
    const saver = new ToolCheckpointSaver(source, thread, new Map([['agent:task', '002']]), toolCheckpointSafety, async (tuple) => {
      admitted.push(tuple.checkpoint.id);
      return tuple;
    });
    const lookup = selected(selector, '003', 'agent:task');
    const history = await collect(saver.list(lookup));
    expect(history.map((tuple) => tuple.checkpoint.id)).toEqual(['003', '002', '001']);
    expect(admitted).toEqual(['003', '002', '001']);
    for (const tuple of history) {
      expect(tuple.config.configurable).toMatchObject({ thread_id: thread, checkpoint_ns: 'agent:task' });
      expect(tuple.checkpoint.channel_values).toEqual({ control: 'Allowed checkpoint control' });
    }
    expect(lookup).toEqual(selected(selector, '003', 'agent:task'));
    expect((await saver.getTuple(config('agent:task')))?.checkpoint.id).toBe('002');
    expect((await saver.getTuple(selected(selector, '001', 'agent:task')))?.checkpoint.id).toBe('001');
    await put(saver, 'agent:task', '004', '002');
    expect((await saver.getTuple(config('agent:task')))?.checkpoint.id).toBe('004');
  });

  it.each((['first', 'later'] as const).flatMap((position) => (['wrong-id', 'thread', 'namespace', 'body-id', 'parent-thread', 'parent-namespace', 'accessor'] as const).map((change) => [position, change] as const)))('rejects adversarial %s history tuples before admission (%s)', async (position, change) => {
    let reads = 0;
    class Saver extends ThreadHistorySaver {
      override async *list(...args: Parameters<MemorySaver['list']>): ReturnType<MemorySaver['list']> {
        let index = 0;
        for await (const tuple of super.list(...args)) {
          if (index++ === (position === 'first' ? 0 : 1)) {
            const scope = tuple.config.configurable!;
            if (change === 'wrong-id') { scope.checkpoint_id = position === 'first' ? '002' : '999'; tuple.checkpoint.id = scope.checkpoint_id as string; }
            if (change === 'thread') scope.thread_id = 'foreign-thread';
            if (change === 'namespace') scope.checkpoint_ns = 'foreign:namespace';
            if (change === 'body-id') tuple.checkpoint.id = 'foreign-id';
            if (change === 'parent-thread') tuple.parentConfig!.configurable!.thread_id = 'foreign-thread';
            if (change === 'parent-namespace') tuple.parentConfig!.configurable!.checkpoint_ns = 'foreign:namespace';
            if (change === 'accessor') Object.defineProperty(scope, 'checkpoint_id', { get: () => { reads++; throw new Error('Unreleased history identity'); } });
          }
          yield tuple;
        }
      }
    }
    const source = new Saver();
    await seed(source);
    const admitted: string[] = [];
    const saver = new ToolCheckpointSaver(source, thread, new Map(), toolCheckpointSafety, async (tuple) => { admitted.push(tuple.checkpoint.id); return tuple; });
    await expect(collect(saver.list(selected('checkpoint_id')))).rejects.toMatchObject({ code: change === 'accessor' ? 'unsupported' : 'incompatible' });
    expect(admitted).toEqual(position === 'first' ? [] : ['003']);
    expect(reads).toBe(0);
  });

  it('binds an otherwise unscoped selected history to the validated anchor namespace', async () => {
    const source = new ThreadHistorySaver();
    await seed(source);
    await seed(source, 'agent:task');
    const admitted: string[] = [];
    const saver = new ToolCheckpointSaver(source, thread, new Map(), toolCheckpointSafety, async (tuple) => { admitted.push(tuple.checkpoint.id); return tuple; });
    const lookup: RunnableConfig = { configurable: { thread_id: thread, checkpoint_id: '003' } };
    await expect(collect(saver.list(lookup))).rejects.toMatchObject({ code: 'incompatible' });
    expect(admitted).toEqual(['003', '002', '001']);
  });

  it('preserves native unselected before/filter/limit controls instead of applying the admitted cursor', async () => {
    const source = new MemorySaver();
    await seed(source);
    const list = jest.spyOn(source, 'list');
    const saver = new ToolCheckpointSaver(source, thread, new Map([['', '001']]), toolCheckpointSafety);
    const options = { before: config('', '003'), filter: { source: 'input' }, limit: 1 };
    const history = await collect(saver.list(config(), options));
    expect(history.map((tuple) => tuple.checkpoint.id)).toEqual(['002']);
    expect(list).toHaveBeenCalledWith(config(), options);
    expect(await collect(saver.list(selected('checkpoint_id', 'absent')))).toEqual([]);
    expect(await collect(saver.list(selected('checkpoint_id'), { filter: { source: 'missing' } }))).toEqual([]);
    expect(await collect(saver.list(selected('checkpoint_id'), { limit: 0 }))).toEqual([]);
  });
});

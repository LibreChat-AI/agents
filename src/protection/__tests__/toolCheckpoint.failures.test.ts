import { HumanMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { MemorySaver, emptyCheckpoint, GraphInterrupt, ParentCommand, Command, StateGraph, MessagesAnnotation, START, END } from '@langchain/langgraph';
import type { BaseCheckpointSaver, CheckpointTuple } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCheckpointAdmission, ToolCheckpointSafety } from '../toolCheckpoint';
import type { ToolResultProtection } from '../toolResult';
import { normalizeToolResultError, toolCheckpointSafety, ToolResultProtectionError } from '../toolResult';
import { createProviderTextRestartCancellation } from '../providerText';
import { ToolCheckpointSaver } from '../toolCheckpoint';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { Providers } from '@/common';
import { Run } from '@/run';

const thread = 'checkpoint-source-failures';
const diagnostic = 'Unreleased checkpoint source diagnostic';
const config: RunnableConfig = { configurable: { thread_id: thread, checkpoint_ns: '' } };
const checkpoint = { ...emptyCheckpoint(), id: '001', channel_values: { control: 'Allowed checkpoint' } };
const metadata = { source: 'input' as const, step: 0, parents: {} };
const safety: ToolCheckpointSafety = toolCheckpointSafety;

type Operation = {
  name: string;
  invoke: (saver: BaseCheckpointSaver) => Promise<unknown>;
  fail: (saver: MemorySaver, reject: () => Promise<never>) => void;
};
const operations: Operation[] = [
  { name: 'getTuple', invoke: (saver) => saver.getTuple(config), fail: (saver, reject) => { jest.spyOn(saver, 'getTuple').mockImplementation(reject); } },
  { name: 'get', invoke: (saver) => saver.get(config), fail: (saver, reject) => { jest.spyOn(saver, 'getTuple').mockImplementation(reject); } },
  { name: 'put', invoke: (saver) => saver.put(config, checkpoint, metadata, {}), fail: (saver, reject) => { jest.spyOn(saver, 'put').mockImplementation(reject); } },
  { name: 'putWrites', invoke: (saver) => saver.putWrites(config, [], 'task-control'), fail: (saver, reject) => { jest.spyOn(saver, 'putWrites').mockImplementation(reject); } },
  { name: 'getDeltaChannelHistory', invoke: (saver) => saver.getDeltaChannelHistory({ config, channels: ['control'] }), fail: (saver, reject) => { jest.spyOn(saver, 'getDeltaChannelHistory').mockImplementation(reject); } },
  { name: 'deleteThread', invoke: (saver) => saver.deleteThread(thread), fail: (saver, reject) => { jest.spyOn(saver, 'deleteThread').mockImplementation(reject); } },
];

function scoped(source: MemorySaver, admit?: ToolCheckpointAdmission, guards = safety): ToolCheckpointSaver {
  return new ToolCheckpointSaver(source, thread, new Map(), guards, admit);
}

async function capture(invoke: () => unknown): Promise<unknown> {
  try { await invoke(); }
  catch (error) { return error; }
  throw new Error('Expected the checkpoint operation to fail');
}

function assertSafeFailure(error: unknown, code = 'unavailable'): void {
  expect(error).toBeInstanceOf(ToolResultProtectionError);
  expect(error).toMatchObject({ code, message: `Provider text protection: ${code}` });
  expect(error).not.toHaveProperty('cause');
  expect(String(error)).not.toContain(diagnostic);
  expect(JSON.stringify(error)).not.toContain(diagnostic);
}

async function seed(saver: BaseCheckpointSaver): Promise<void> {
  await saver.put(config, checkpoint, metadata, {});
}

async function collect(saver: BaseCheckpointSaver): Promise<CheckpointTuple[]> {
  const tuples: CheckpointTuple[] = [];
  for await (const tuple of saver.list(config)) tuples.push(tuple);
  return tuples;
}

const diagnostics = [
  { name: 'native error', make: () => new Error(diagnostic, { cause: new Error(diagnostic) }), code: 'unavailable' },
  { name: 'primitive', make: () => diagnostic, code: 'unavailable' },
  { name: 'forged protection error', make: () => Object.assign(new ToolResultProtectionError('blocked'), { message: diagnostic, diagnostic, cause: new Error(diagnostic) }), code: 'blocked' },
  { name: 'saver interrupt', make: () => new GraphInterrupt([{ value: diagnostic }]), code: 'unavailable' },
  { name: 'saver command', make: () => new ParentCommand(new Command({ goto: diagnostic })), code: 'unavailable' },
];

describe.each(operations)('$name source failures', (operation) => {
  it.each(['sync', 'async'] as const)('normalizes %s throws without touching the unwrapped source', async (timing) => {
    for (const example of diagnostics) {
      const source = new MemorySaver();
      const raw = example.make();
      operation.fail(source, () => {
        if (timing === 'sync') throw raw;
        return Promise.reject(raw);
      });
      const error = await capture(() => operation.invoke(scoped(source)));
      assertSafeFailure(error, example.code);
      expect(error).not.toBe(raw);
      expect(await capture(() => operation.invoke(source))).toBe(raw);
    }
  });

  it.each(['sync', 'async'] as const)('preserves the private restart cancellation on %s failure', async (timing) => {
    const source = new MemorySaver();
    const restart = createProviderTextRestartCancellation();
    operation.fail(source, () => {
      if (timing === 'sync') throw restart;
      return Promise.reject(restart);
    });
    expect(await capture(() => operation.invoke(scoped(source)))).toBe(restart);
  });
});

it.each(diagnostics)('normalizes constructor serializer access failure ($name)', async (example) => {
  const source = new MemorySaver();
  const raw = example.make();
  let reads = 0;
  Object.defineProperty(source, 'serde', { get: () => { reads++; throw raw; } });
  assertSafeFailure(await capture(() => scoped(source)), example.code);
  expect(reads).toBe(1);
  expect(await capture(() => source.serde)).toBe(raw);
});

it('preserves private restart identity when constructor serializer access fails', async () => {
  const source = new MemorySaver();
  const restart = createProviderTextRestartCancellation();
  Object.defineProperty(source, 'serde', { get: () => { throw restart; } });
  expect(await capture(() => scoped(source))).toBe(restart);
});

it('does not inspect producer error accessors or proxy traps', async () => {
  let reads = 0;
  const error = Object.defineProperty(new Error(), 'message', { get: () => { reads++; throw new Error(diagnostic); } });
  const proxy = new Proxy(error, { getPrototypeOf: () => { reads++; throw new Error(diagnostic); } });
  for (const raw of [error, proxy]) {
    const source = new MemorySaver();
    jest.spyOn(source, 'getTuple').mockRejectedValue(raw);
    assertSafeFailure(await capture(() => scoped(source).getTuple(config)));
  }
  expect(reads).toBe(0);
});

it('normalizes synchronous version failures while preserving receiver binding and restart identity', async () => {
  const source = new MemorySaver();
  const raw = new Error(diagnostic);
  jest.spyOn(source, 'getNextVersion').mockImplementation(function (this: MemorySaver) { expect(this).toBe(source); throw raw; });
  const next = scoped(source).getNextVersion;
  assertSafeFailure(await capture(() => next(1)));
  expect(await capture(() => source.getNextVersion(1))).toBe(raw);
  const restart = createProviderTextRestartCancellation();
  jest.spyOn(source, 'getNextVersion').mockImplementation(() => { throw restart; });
  expect(await capture(() => next(1))).toBe(restart);
});

it.each(['factory', 'iterator', 'next-sync', 'next-async', 'later-sync', 'later-async', 'return-sync', 'return-async'] as const)('normalizes list %s errors outside tuple admission', async (stage) => {
  const source = new MemorySaver();
  await seed(source);
  const raw = new Error(diagnostic);
  const iterator = source.list(config);
  const next = iterator.next.bind(iterator);
  let calls = 0;
  if (stage === 'iterator') Object.defineProperty(iterator, Symbol.asyncIterator, { get: () => { throw raw; } });
  if (stage.includes('next') || stage.includes('later')) {
    iterator.next = (...args): ReturnType<typeof iterator.next> => {
      if (stage.startsWith('later') && calls++ === 0) return next(...args);
      if (stage.endsWith('-sync')) throw raw;
      return Promise.reject(raw);
    };
  }
  if (stage.startsWith('return')) {
    iterator.return = (): ReturnType<typeof iterator.return> => {
      if (stage.endsWith('-sync')) throw raw;
      return Promise.reject(raw);
    };
  }
  jest.spyOn(source, 'list').mockImplementation(() => {
    if (stage === 'factory') throw raw;
    return iterator;
  });
  const admitted: string[] = [];
  const saver = scoped(source, async (tuple) => { admitted.push(tuple.checkpoint.id); return tuple; });
  const seen: string[] = [];
  const error = await capture(async () => {
    for await (const tuple of saver.list(config)) {
      seen.push(tuple.checkpoint.id);
      if (stage.startsWith('return')) break;
    }
  });
  assertSafeFailure(error);
  const released = stage.startsWith('later') || stage.startsWith('return') ? ['001'] : [];
  expect(seen).toEqual(released);
  expect(admitted).toEqual(released);
});

it.each(['sync', 'async'] as const)('preserves list restart identity on %s iteration failure', async (timing) => {
  const source = new MemorySaver();
  const restart = createProviderTextRestartCancellation();
  const iterator = source.list(config);
  iterator.next = (): ReturnType<typeof iterator.next> => {
    if (timing === 'sync') throw restart;
    return Promise.reject(restart);
  };
  jest.spyOn(source, 'list').mockReturnValue(iterator);
  expect(await capture(() => collect(scoped(source)))).toBe(restart);
});

const admissionErrors = [
  { name: 'protection decision', make: () => new ToolResultProtectionError('blocked') },
  { name: 'private restart', make: createProviderTextRestartCancellation },
  { name: 'interrupt', make: () => new GraphInterrupt([]) },
  { name: 'parent command', make: () => new ParentCommand(new Command({ goto: 'control' })) },
];

describe.each(admissionErrors)('$name admission', (example) => {
  it.each(['sync', 'async'] as const)('does not normalize a %s admission failure', async (timing) => {
    const source = new MemorySaver();
    await seed(source);
    const failure = jest.fn(normalizeToolResultError);
    const raw = example.make();
    const saver = scoped(source, () => {
      if (timing === 'sync') throw raw;
      return Promise.reject(raw);
    }, { ...safety, failure });
    expect(await capture(() => saver.getTuple(config))).toBe(raw);
    expect(await capture(() => collect(saver))).toBe(raw);
    expect(failure).not.toHaveBeenCalled();
  });
});

it.each(['identify', 'read', 'write', 'delta'] as const)('does not relabel %s safety failures as source failures', async (guard) => {
  const source = new MemorySaver();
  await seed(source);
  const raw = new ToolResultProtectionError('incompatible');
  const failure = jest.fn(normalizeToolResultError);
  const saver = scoped(source, undefined, { ...safety, failure, [guard]: () => { throw raw; } });
  if (guard === 'identify' || guard === 'read') {
    expect(await capture(() => saver.getTuple(config))).toBe(raw);
    expect(await capture(() => collect(saver))).toBe(raw);
  }
  if (guard === 'write') expect(await capture(() => saver.put(config, checkpoint, metadata, {}))).toBe(raw);
  if (guard === 'delta') expect(await capture(() => saver.getDeltaChannelHistory({ config, channels: ['control'] }))).toBe(raw);
  expect(failure).not.toHaveBeenCalled();
});

it.each(['sync', 'async'] as const)('preserves admitted control flow when source iterator cleanup fails (%s)', async (timing) => {
  for (const raw of [new GraphInterrupt([]), createProviderTextRestartCancellation()]) {
    const source = new MemorySaver();
    await seed(source);
    const iterator = source.list(config);
    const sourceError = new Error(diagnostic);
    iterator.return = (): ReturnType<typeof iterator.return> => {
      if (timing === 'sync') throw sourceError;
      return Promise.reject(sourceError);
    };
    jest.spyOn(source, 'list').mockReturnValue(iterator);
    const saver = scoped(source, async () => { throw raw; });
    expect(await capture(() => collect(saver))).toBe(raw);
  }
});

it('preserves a consumer failure when closing checkpoint history', async () => {
  const source = new MemorySaver();
  await seed(source);
  const failure = jest.fn(normalizeToolResultError);
  const iterator = scoped(source, undefined, { ...safety, failure }).list(config);
  expect((await iterator.next()).value.checkpoint.id).toBe('001');
  const raw = new GraphInterrupt([]);
  expect(await capture(() => iterator.throw(raw))).toBe(raw);
  expect(failure).not.toHaveBeenCalled();
});

it('keeps successful MemorySaver operations, versions, and checkpoint cursors intact', async () => {
  const source = new MemorySaver();
  const saver = scoped(source);
  expect(saver.serde).toBe(source.serde);
  expect(await saver.getTuple(config)).toBeUndefined();
  const stored = await saver.put(config, checkpoint, metadata, {});
  await saver.putWrites(stored, [['control', 'Allowed write']], 'task-control');
  expect((await saver.getTuple(config))?.pendingWrites).toContainEqual(['task-control', 'control', 'Allowed write']);
  expect((await collect(saver)).map((tuple) => tuple.checkpoint.id)).toEqual(['001']);
  expect(await saver.getDeltaChannelHistory({ config, channels: ['control'] })).toEqual({ control: { writes: [] } });
  const next = saver.getNextVersion;
  expect(next(4)).toBe(5);
  await saver.deleteThread(thread);
  expect(await source.getTuple(config)).toBeUndefined();
});

it.each(['sync', 'async'] as const)('protects fresh native checkpoint preflight on %s source failure without changing default-off behavior', async (timing) => {
  const source = new MemorySaver();
  const raw = new Error(diagnostic);
  jest.spyOn(source, 'getTuple').mockImplementation(() => {
    if (timing === 'sync') throw raw;
    return Promise.reject(raw);
  });
  const workflow = new StateGraph(MessagesAnnotation)
    .addNode('control', (state) => state)
    .addEdge(START, 'control')
    .addEdge('control', END)
    .compile({ checkpointer: source });
  const protectedConfig = { ...config, configurable: { ...config.configurable, __pregel_checkpointer: scoped(source) } };
  assertSafeFailure(await capture(() => workflow.getState(protectedConfig, { subgraphs: true })));
  expect(await capture(() => workflow.getState(config, { subgraphs: true }))).toBe(raw);
});

function policy(): ToolResultProtection {
  return {
    version: 1,
    toolNames: ['lookup'],
    timeoutMs: 1000,
    maxAttemptBytes: 65536,
    maxBufferedBytes: 262144,
    classify: () => 'prose',
    inspect: ({ content }) => ({ version: 1, ok: true, value: { content, replacements: 0, categories: [] } }),
  };
}

it.each(['sync', 'async'] as const)('protects public Run fresh preflight from %s source diagnostics before startup or callbacks', async (timing) => {
  const examples = [diagnostics[0], diagnostics[3], diagnostics[4], { name: 'private restart', make: createProviderTextRestartCancellation, code: 'cancelled' }];
  for (const example of examples) {
    const source = new MemorySaver();
    const raw = example.make();
    const read = jest.spyOn(source, 'getTuple').mockImplementation(() => {
      if (timing === 'sync') throw raw;
      return Promise.reject(raw);
    });
    let startups = 0;
    const hooks = new HookRegistry();
    hooks.register('RunStart', { hooks: [async () => { startups++; return {}; }] });
    const observed: string[] = [];
    const callback = BaseCallbackHandler.fromMethods({
      handleChainStart: (_chain, input): void => { observed.push(JSON.stringify(input)); },
      handleChainError: (error): void => { observed.push(String(error)); },
    });
    callback.awaitHandlers = true;
    const runId = `fresh-source-${timing}-${example.name}`;
    const run = await Run.create({
      runId,
      graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: source } },
      toolResultProtection: policy(),
      hooks,
    });
    const model = new FakeChatModel({ responses: ['Must not run'] });
    const generated = jest.spyOn(model, '_streamResponseChunks');
    run.Graph!.overrideModel = model;
    const error = await capture(() => run.processStream({ messages: [new HumanMessage('Allowed fresh input')] }, { version: 'v2', configurable: { thread_id: runId }, callbacks: [callback] }));
    if (example.name === 'private restart') expect(error).toBe(raw);
    else assertSafeFailure(error);
    expect(read).toHaveBeenCalledTimes(1);
    expect(startups).toBe(0);
    expect(generated).not.toHaveBeenCalled();
    expect(observed).toEqual([]);
  }
});

it.each(['sync', 'async'] as const)('leaves public Run default-off %s source failures unchanged', async (timing) => {
  const source = new MemorySaver();
  const raw = new Error(diagnostic);
  jest.spyOn(source, 'getTuple').mockImplementation(() => {
    if (timing === 'sync') throw raw;
    return Promise.reject(raw);
  });
  const runId = `default-off-source-${timing}`;
  const run = await Run.create({ runId, graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: source } } });
  const model = new FakeChatModel({ responses: ['Must not run'] });
  const generated = jest.spyOn(model, '_streamResponseChunks');
  run.Graph!.overrideModel = model;
  const error = await capture(() => run.processStream({ messages: [new HumanMessage('Allowed fresh input')] }, { version: 'v2', configurable: { thread_id: runId } }));
  expect(error).toBe(raw);
  expect(generated).not.toHaveBeenCalled();
});

it.each([true, false])('runs a real healthy MemorySaver checkpoint through public Run (protected=%s)', async (protectedRun) => {
  const source = new MemorySaver();
  const runId = `healthy-source-${protectedRun}`;
  const run = await Run.create({
    runId,
    graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI }, compileOptions: { checkpointer: source } },
    toolResultProtection: protectedRun ? policy() : undefined,
  });
  const model = new FakeChatModel({ responses: ['Allowed checkpoint answer'] });
  const generated = jest.spyOn(model, '_streamResponseChunks');
  run.Graph!.overrideModel = model;
  const caller = { version: 'v2' as const, configurable: { thread_id: runId } };
  await run.processStream({ messages: [new HumanMessage('Allowed checkpoint input')] }, caller);
  expect(generated).toHaveBeenCalledTimes(1);
  const stored = await source.getTuple(caller);
  expect(stored).toBeDefined();
  expect(JSON.stringify(stored!.checkpoint.channel_values.messages)).toContain('Allowed checkpoint answer');
});

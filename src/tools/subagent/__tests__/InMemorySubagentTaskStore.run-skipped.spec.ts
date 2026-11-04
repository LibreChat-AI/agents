import { InMemorySubagentTaskStore } from '../InMemorySubagentTaskStore';

type SkipReason = 'task_not_running' | 'signal_aborted';

type SkipEvent = { scopeId: string; taskId: string; reason: SkipReason };

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

function request(
  scopeId: string,
  idempotencyKey: string,
  run: (runtime: unknown) => Promise<{ content: string }>
) {
  return {
    scopeId,
    idempotencyKey,
    parentRunId: 'parent-run',
    parentToolCallId: 'parent-tool',
    input: 'child input',
    subagentKind: 'agent' as const,
    subagentType: 'researcher-agent',
    run
  };
}

async function waitUntil(condition: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  throw new Error('Timed out waiting for task settlement.');
}

class RecordingStore extends InMemorySubagentTaskStore {
  readonly skipped: SkipEvent[] = [];

  protected override onRunSkipped(
    scopeId: string,
    taskId: string,
    reason: SkipReason
  ): void {
    this.skipped.push({ scopeId, taskId, reason });
  }
}

class ThrowingStore extends InMemorySubagentTaskStore {
  skippedCalls = 0;

  protected override onRunSkipped(): void {
    this.skippedCalls += 1;
    throw new Error('observer projection failed');
  }
}

describe('InMemorySubagentTaskStore run-skipped lifecycle', () => {
  it('reports exact identity once when synchronous cancel skips the queued factory', async () => {
    const store = new RecordingStore();
    const factory = jest.fn(async () => ({ content: 'must not run' }));
    const started = store.start(request('scope-a', 'attempt-a', factory));
    if (!started.accepted || !started.isNew)
      throw new Error('task not accepted');

    expect(
      store.control('scope-a', started.task.taskId, { action: 'cancel' })
    ).toMatchObject({
      status: 'cancelled'
    });
    expect(
      store.control('scope-a', started.task.taskId, { action: 'cancel' })
    ).toMatchObject({
      status: 'not_running'
    });
    await waitUntil(() => store.skipped.length === 1);

    expect(factory).not.toHaveBeenCalled();
    expect(store.skipped).toEqual([
      {
        scopeId: 'scope-a',
        taskId: started.task.taskId,
        reason: 'task_not_running'
      }
    ]);
    expect(store.get('scope-a', started.task.taskId)?.status).toBe('cancelled');
    expect((store as unknown as { runningTasks: number }).runningTasks).toBe(0);
  });

  it('does not report skipped after factory entry, including held provider work', async () => {
    const provider = deferred<{ content: string }>();
    const entered = deferred<void>();
    const store = new RecordingStore();
    const started = store.start(
      request('scope-b', 'attempt-b', async () => {
        entered.resolve();
        return provider.promise;
      })
    );
    if (!started.accepted || !started.isNew)
      throw new Error('task not accepted');

    await entered.promise;
    expect(store.skipped).toEqual([]);
    expect(
      store.control('scope-b', started.task.taskId, { action: 'cancel' })
    ).toMatchObject({
      status: 'cancelled'
    });
    provider.resolve({ content: 'finished after abort' });
    await waitUntil(
      () => store.get('scope-b', started.task.taskId)?.status === 'cancelled'
    );
    expect(store.skipped).toEqual([]);
  });

  it('isolates a skipped notification to its exact scope and task', async () => {
    const store = new RecordingStore();
    const firstFactory = jest.fn(async () => ({ content: 'must not run' }));
    const secondFactory = jest.fn(async () => ({ content: 'ran' }));
    const first = store.start(request('scope-d1', 'attempt-d1', firstFactory));
    const second = store.start(
      request('scope-d2', 'attempt-d2', secondFactory)
    );
    if (!first.accepted || !first.isNew || !second.accepted || !second.isNew) {
      throw new Error('tasks not accepted');
    }

    expect(
      store.control('scope-d2', first.task.taskId, { action: 'cancel' })
    ).toMatchObject({
      status: 'not_found'
    });
    expect(
      store.control('scope-d1', first.task.taskId, { action: 'cancel' })
    ).toMatchObject({
      status: 'cancelled'
    });
    await waitUntil(() => store.skipped.length === 1);
    await waitUntil(
      () => store.get('scope-d2', second.task.taskId)?.status === 'completed'
    );

    expect(store.skipped).toEqual([
      {
        scopeId: 'scope-d1',
        taskId: first.task.taskId,
        reason: 'task_not_running'
      }
    ]);
    expect(firstFactory).not.toHaveBeenCalled();
    expect(secondFactory).toHaveBeenCalledTimes(1);
  });

  it('preserves task status and counters when the optional observer throws', async () => {
    const store = new ThrowingStore();
    const factory = jest.fn(async () => ({ content: 'must not run' }));
    const started = store.start(request('scope-c', 'attempt-c', factory));
    if (!started.accepted || !started.isNew)
      throw new Error('task not accepted');

    expect(
      store.control('scope-c', started.task.taskId, { action: 'cancel' }).status
    ).toBe('cancelled');
    await waitUntil(() => store.skippedCalls === 1);

    expect(factory).not.toHaveBeenCalled();
    expect(store.get('scope-c', started.task.taskId)?.status).toBe('cancelled');
    expect((store as unknown as { runningTasks: number }).runningTasks).toBe(0);
  });

  it('keeps the default subclass hook inert', async () => {
    const store = new InMemorySubagentTaskStore();
    const factory = jest.fn(async () => ({ content: 'must not run' }));
    const started = store.start(
      request('scope-vanilla', 'attempt-vanilla', factory)
    );
    if (!started.accepted || !started.isNew)
      throw new Error('task not accepted');

    expect(
      store.control('scope-vanilla', started.task.taskId, { action: 'cancel' })
        .status
    ).toBe('cancelled');
    await waitUntil(
      () =>
        store.get('scope-vanilla', started.task.taskId)?.status === 'cancelled'
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(factory).not.toHaveBeenCalled();
  });
});

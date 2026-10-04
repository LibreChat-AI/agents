import { describe, expect, it, jest } from '@jest/globals';
import {
  SubagentExecutionRegistry,
  SubagentExecutionInvalidatedError,
} from '@/tools/subagent/SubagentExecutionRegistry';
import {
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY,
} from '@/tools/subagent/SubagentReplay';

type Result = { content: string };
type ResolvedConfig = { agentId: string };
type ActiveRun = { graphId: string };
type SettledOutput = { content: string };

const createRegistry = (): SubagentExecutionRegistry<
  Result,
  ResolvedConfig,
  ActiveRun,
  SettledOutput
> =>
  new SubagentExecutionRegistry({
    parentRunId: 'parent-run',
    parentAgentId: 'parent-agent',
    durable: true,
  });

const createInput = (
  overrides: Record<string, unknown> = {}
): {
  threadId: string;
  parentToolCallId: string;
  parentConfigurable: Record<string, unknown>;
} => ({
  threadId: 'durable-thread',
  parentToolCallId: 'call_shared',
  parentConfigurable: {
    checkpoint_id: 'fork-a',
    [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'batch-a',
    [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-a',
    ...overrides,
  },
});

describe('SubagentExecutionRegistry', () => {
  it('derives one canonical address and isolates forks, batches, and attempts', () => {
    const registry = createRegistry();
    const first = registry.open(createInput());
    const duplicate = registry.open(createInput());
    const fork = registry.open(createInput({ checkpoint_id: 'fork-b' }));
    const batch = registry.open(
      createInput({ [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'batch-b' })
    );
    const attempt = registry.open(
      createInput({ [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-b' })
    );

    expect(duplicate).toBe(first);
    expect(
      new Set([
        first.address.key,
        fork.address.key,
        batch.address.key,
        attempt.address.key,
      ]).size
    ).toBe(4);
    expect(attempt.address.baseChildThreadId).toBe(
      first.address.baseChildThreadId
    );
    expect(attempt.address.branchChildThreadId).not.toBe(
      first.address.branchChildThreadId
    );
  });

  it('keeps identity immutable and protects an effective definition binding', () => {
    const record = createRegistry().open(createInput());
    const identity = {
      childRunId: 'child-run',
      childThreadId: 'child-thread',
      approvalExecutionScope: 'approval-scope',
    };

    record.bindIdentity(identity);
    record.bindDefinition(
      { subagentType: 'researcher', configId: 'researcher@v1' },
      'provisional'
    );
    record.bindDefinition(
      { subagentType: 'coder', configId: 'coder@v2' },
      'effective'
    );
    record.bindDefinition(
      { subagentType: 'researcher', configId: 'researcher@v1' },
      'provisional'
    );
    expect(() =>
      record.bindDefinition(
        { subagentType: 'coder', configId: 'coder@v2' },
        'effective'
      )
    ).not.toThrow();

    expect(record.snapshot).toMatchObject({
      identity,
      binding: { subagentType: 'coder', configId: 'coder@v2' },
    });
    expect(() =>
      record.bindIdentity({
        ...identity,
        childThreadId: 'another-child-thread',
      })
    ).toThrow('Subagent execution identity cannot change after binding.');
    expect(() =>
      record.bindDefinition(
        { subagentType: 'writer', configId: 'writer@v3' },
        'effective'
      )
    ).toThrow('Subagent effective definition binding cannot change.');
  });

  it('coalesces resolution and isolates a stale completion after invalidation', async () => {
    const registry = createRegistry();
    const staleRecord = registry.open(createInput());
    let finishStale = (_config: ResolvedConfig): void => undefined;
    let finishCurrent = (_config: ResolvedConfig): void => undefined;
    let markStaleStarted = (): void => undefined;
    const staleStarted = new Promise<void>((resolve) => {
      markStaleStarted = resolve;
    });
    const staleValue = new Promise<ResolvedConfig>((resolve) => {
      finishStale = resolve;
    });
    const currentValue = new Promise<ResolvedConfig>((resolve) => {
      finishCurrent = resolve;
    });
    const signal = new AbortController().signal;
    const resolveStale = jest.fn(() => {
      markStaleStarted();
      return staleValue;
    });

    const stale = staleRecord.resolveConfig(resolveStale, signal);
    const duplicate = staleRecord.resolveConfig(resolveStale, signal);
    expect(duplicate).toBe(stale);
    await staleStarted;

    registry.clear();
    const currentRecord = registry.open(createInput());
    const current = currentRecord.resolveConfig(() => currentValue, signal);

    finishStale({ agentId: 'stale-agent' });
    await expect(stale).rejects.toBeInstanceOf(
      SubagentExecutionInvalidatedError
    );
    expect(currentRecord.snapshot).toMatchObject({ resolving: true });

    finishCurrent({ agentId: 'current-agent' });
    await expect(current).resolves.toEqual({ agentId: 'current-agent' });
    expect(currentRecord.snapshot).toMatchObject({
      resolving: false,
      resolvedConfig: { agentId: 'current-agent' },
    });
    expect(resolveStale).toHaveBeenCalledTimes(1);
  });

  it('retains failed identity attempts without committing or persisting them', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    const identity = {
      childRunId: 'attempted-run',
      childThreadId: 'attempted-thread',
      approvalExecutionScope: 'attempted-scope',
    };
    const failure = new Error('Preparation failed');
    await expect(
      record.resolveIdentity(async () => {
        const lease = registry.beginIdentityPreparation(record, identity);
        expect(record.identity).toBeUndefined();
        expect(record.attemptedIdentity).toEqual(identity);
        return lease.rollback(failure);
      })
    ).rejects.toBe(failure);

    expect(record.identity).toBeUndefined();
    expect(record.attemptedIdentity).toEqual(identity);
    expect(Object.isFrozen(record.attemptedIdentity)).toBe(true);
    const attemptedIdentity = record.attemptedIdentity;
    record.recordIdentityAttempt({ ...identity });
    expect(record.attemptedIdentity).toBe(attemptedIdentity);
    expect(record.snapshot).not.toHaveProperty('identity');
    expect(record.snapshot).not.toHaveProperty('attemptedIdentity');
    expect(registry.selectForResume()).toEqual([]);

    await expect(
      record.resolveIdentity(async () => {
        expect(record.attemptedIdentity).toBeUndefined();
        throw failure;
      })
    ).rejects.toBe(failure);
    expect(record.attemptedIdentity).toBeUndefined();

    const nextIdentity = { ...identity, childRunId: 'retry-run' };
    await expect(
      record.resolveIdentity(async () => ({
        identity: nextIdentity,
        lease: registry.beginIdentityPreparation(record, nextIdentity),
      }))
    ).resolves.toEqual(nextIdentity);
    expect(record.identity).toEqual(nextIdentity);
    expect(record.attemptedIdentity).toEqual(nextIdentity);
  });

  it('retains an invalidated attempt for its pending failure without leaking to a replacement record', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    const identity = {
      childRunId: 'stale-run',
      childThreadId: 'stale-thread',
      approvalExecutionScope: 'stale-scope',
    };
    let started = (): void => undefined;
    let finish = (): void => undefined;
    const preparationStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const pending = record.resolveIdentity(async () => {
      const lease = registry.beginIdentityPreparation(record, identity);
      started();
      await gate;
      return { identity, lease };
    });
    await preparationStarted;
    registry.clear();
    const replacement = registry.open(createInput());
    expect(record.identity).toBeUndefined();
    expect(record.attemptedIdentity).toEqual(identity);
    expect(replacement.attemptedIdentity).toBeUndefined();
    finish();
    await expect(pending).rejects.toBeInstanceOf(
      SubagentExecutionInvalidatedError
    );
    expect(record.identity).toBeUndefined();
    expect(record.attemptedIdentity).toEqual(identity);
    expect(replacement.identity).toBeUndefined();
  });

  it('coalesces identity resolution before binding the resolved identity', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    let finish = (_identity: {
      childRunId: string;
      childThreadId: string;
      approvalExecutionScope: string;
    }): void => undefined;
    const identity = new Promise<{
      childRunId: string;
      childThreadId: string;
      approvalExecutionScope: string;
    }>((resolve) => {
      finish = resolve;
    });
    const resolveIdentity = jest.fn(async () => {
      const resolvedIdentity = await identity;
      return {
        identity: resolvedIdentity,
        lease: registry.beginIdentityPreparation(record, resolvedIdentity),
      };
    });

    const first = record.resolveIdentity(resolveIdentity);
    const duplicate = record.resolveIdentity(resolveIdentity);

    expect(duplicate).toBe(first);
    expect(record.snapshot.resolvingIdentity).toBe(true);
    finish({
      childRunId: 'child-run',
      childThreadId: 'child-thread',
      approvalExecutionScope: 'approval-scope',
    });

    await expect(first).resolves.toMatchObject({ childRunId: 'child-run' });
    expect(resolveIdentity).toHaveBeenCalledTimes(1);
    expect(record.snapshot).toMatchObject({
      resolvingIdentity: false,
      identity: { childThreadId: 'child-thread' },
    });
  });

  it('rolls back prepared identity side effects after invalidation', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    let finishPreparation = (): void => undefined;
    let markPreparationStarted = (): void => undefined;
    const preparationStarted = new Promise<void>((resolve) => {
      markPreparationStarted = resolve;
    });
    const preparationGate = new Promise<void>((resolve) => {
      finishPreparation = resolve;
    });
    const commit = jest.fn();
    const rollback = jest.fn(async (): Promise<void> => undefined);
    const staleIdentity = {
      childRunId: 'stale-run',
      childThreadId: 'stale-thread',
      approvalExecutionScope: 'stale-scope',
    };
    const lease = registry.beginIdentityPreparation(record, staleIdentity);
    const pending = record.resolveIdentity(async () => {
      markPreparationStarted();
      await preparationGate;
      return {
        identity: staleIdentity,
        lease,
        commit: () => commit(),
        rollback: async () => rollback(),
      };
    });

    await preparationStarted;
    registry.clear();
    finishPreparation();

    await expect(pending).rejects.toBeInstanceOf(
      SubagentExecutionInvalidatedError
    );
    expect(commit).not.toHaveBeenCalled();
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(record.snapshot.identity).toBeUndefined();
    expect(registry.selectForResume()).toEqual([]);
  });

  it('keeps created-resource ownership exclusive through rollback', async () => {
    const registry = createRegistry();
    const staleRecord = registry.open(createInput());
    const sharedIdentity = {
      childRunId: 'shared-run',
      childThreadId: 'shared-thread',
      approvalExecutionScope: 'shared-scope',
    };
    const staleLease = registry.beginIdentityPreparation(
      staleRecord,
      sharedIdentity
    );
    const checkpointCleanup = jest.fn(async (): Promise<void> => undefined);
    const approvalCleanup = jest.fn(async (): Promise<void> => undefined);
    staleLease.markCheckpointCreated(checkpointCleanup);
    staleLease.markApprovalScopeCreated(approvalCleanup);
    registry.clear();
    const currentRecord = registry.open(createInput());
    const staleCause = new Error('stale preparation');

    expect(() =>
      registry.beginIdentityPreparation(currentRecord, sharedIdentity)
    ).toThrow(SubagentExecutionInvalidatedError);

    await expect(staleLease.rollback(staleCause)).rejects.toBe(staleCause);

    expect(staleLease.ownsCheckpoint).toBe(false);
    expect(staleLease.ownsApprovalScope).toBe(false);
    expect(checkpointCleanup).toHaveBeenCalledTimes(1);
    expect(approvalCleanup).toHaveBeenCalledTimes(1);
    const currentLease = registry.beginIdentityPreparation(
      currentRecord,
      sharedIdentity
    );
    expect(currentLease.ownsCreatedCheckpoint).toBe(false);
    expect(currentLease.ownsCreatedApprovalScope).toBe(false);
    currentLease.release();
  });

  it('blocks a cleaning checkpoint without claiming an independent approval scope', async () => {
    const registry = createRegistry();
    const staleRecord = registry.open(createInput());
    const staleIdentity = {
      childRunId: 'stale-run',
      childThreadId: 'shared-cleaning-thread',
      approvalExecutionScope: 'stale-scope',
    };
    const staleLease = registry.beginIdentityPreparation(
      staleRecord,
      staleIdentity
    );
    let finishCleanup = (): void => undefined;
    let markCleanupStarted = (): void => undefined;
    const cleanupGate = new Promise<void>((resolve) => {
      finishCleanup = resolve;
    });
    const cleanupStarted = new Promise<void>((resolve) => {
      markCleanupStarted = resolve;
    });
    const cleanup = jest.fn(async (): Promise<void> => {
      markCleanupStarted();
      await cleanupGate;
    });
    staleLease.markCheckpointCreated(cleanup);
    registry.clear();
    const currentRecord = registry.open(createInput());
    const replacementIdentity = {
      childRunId: 'replacement-run',
      childThreadId: staleIdentity.childThreadId,
      approvalExecutionScope: 'replacement-scope',
    };
    const staleCause = new Error('stale preparation');
    const rollback = staleLease.rollback(staleCause);
    await cleanupStarted;

    expect(() =>
      registry.beginIdentityPreparation(currentRecord, replacementIdentity)
    ).toThrow(SubagentExecutionInvalidatedError);
    const independentRecord = registry.open(
      createInput({ checkpoint_id: 'independent-fork' })
    );
    const independentLease = registry.beginIdentityPreparation(
      independentRecord,
      {
        childRunId: 'independent-run',
        childThreadId: 'independent-thread',
        approvalExecutionScope: replacementIdentity.approvalExecutionScope,
      }
    );
    expect(independentLease.ownsCheckpoint).toBe(true);
    expect(independentLease.ownsApprovalScope).toBe(true);
    independentLease.release();

    finishCleanup();
    await expect(rollback).rejects.toBe(staleCause);
    expect(cleanup).toHaveBeenCalledTimes(1);
    const retryLease = registry.beginIdentityPreparation(
      currentRecord,
      replacementIdentity
    );
    expect(retryLease.ownsCheckpoint).toBe(true);
    expect(retryLease.ownsApprovalScope).toBe(true);
    retryLease.release();
  });

  it('preserves the primary failure when preparation rollback also fails', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    const lease = registry.beginIdentityPreparation(record, {
      childRunId: 'failed-run',
      childThreadId: 'failed-thread',
      approvalExecutionScope: 'failed-scope',
    });
    const cause = new Error('checkpoint fork failed');
    const rollbackError = new Error('checkpoint cleanup failed');

    await expect(
      lease.rollback(cause, async () => {
        throw rollbackError;
      })
    ).rejects.toBe(cause);

    expect((cause as Error & { rollbackError?: unknown }).rollbackError).toBe(
      rollbackError
    );
  });

  it('preserves a frozen primary failure and releases a failed lease', async () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    const identity = {
      childRunId: 'frozen-run',
      childThreadId: 'frozen-thread',
      approvalExecutionScope: 'frozen-scope',
    };
    const lease = registry.beginIdentityPreparation(record, identity);
    const cause = Object.freeze(new Error('frozen preparation failure'));
    lease.markCheckpointCreated(async () => {
      throw new Error('cleanup failure');
    });

    await expect(lease.rollback(cause)).rejects.toBe(cause);

    expect(lease.ownsCheckpoint).toBe(false);
    expect(lease.ownsApprovalScope).toBe(false);
    const retryLease = registry.beginIdentityPreparation(record, identity);
    expect(retryLease.ownsCheckpoint).toBe(true);
    expect(retryLease.ownsApprovalScope).toBe(true);
    retryLease.release();
  });

  it('coalesces invocation and owns active, interrupted, and completed state', async () => {
    const record = createRegistry().open(createInput());
    const invocation = {
      description: 'Run the child.',
      subagentType: 'researcher',
      configId: 'researcher@v1',
    };
    let finish = (_result: Result): void => undefined;
    const result = new Promise<Result>((resolve) => {
      finish = resolve;
    });
    const invoke = jest.fn(async (): Promise<Result> => {
      record.activate({ graphId: 'graph-a' });
      record.markStarted();
      record.markInterrupted();
      const completed = await result;
      record.activate({ graphId: 'graph-a' });
      record.markCompleted(completed);
      return completed;
    });

    const first = record.execute(invocation, invoke);
    const duplicate = record.execute(invocation, invoke);
    expect(duplicate).toBe(first);
    expect(() =>
      record.execute({ ...invocation, description: 'Changed request.' }, invoke)
    ).toThrow('Subagent invocation binding cannot change.');
    expect(() =>
      record.execute({ ...invocation, configId: 'researcher@v2' }, invoke)
    ).toThrow('Subagent effective definition binding cannot change.');
    finish({ content: 'done' });

    await expect(first).resolves.toEqual({ content: 'done' });
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(record.snapshot).toMatchObject({
      phase: 'completed',
      activeRun: { graphId: 'graph-a' },
      completedResult: { content: 'done' },
      started: true,
      completed: true,
      executing: false,
    });
    expect(record.releaseActiveRun()).toEqual({ graphId: 'graph-a' });
    expect(record.snapshot.activeRun).toBeUndefined();
    expect(() => record.activate({ graphId: 'graph-b' })).toThrow(
      'Cannot transition subagent execution from completed to active.'
    );
    expect(() => record.markFailed()).toThrow(
      'Cannot transition subagent execution from completed to failed.'
    );

    const retryable = createRegistry().open(createInput());
    retryable.markFailed();
    expect(retryable.snapshot.phase).toBe('failed');
    expect(retryable.activate({ graphId: 'graph-retry' })).toEqual({
      graphId: 'graph-retry',
    });
  });

  it('coalesces settlement and retains only its minimal replay output', async () => {
    const record = createRegistry().open(createInput());
    const invocation = {
      description: 'Run the child.',
      subagentType: 'researcher',
      configId: 'researcher@v1',
    };
    const settlement = {
      definitionAuthority: 'effective' as const,
      fingerprint: 'settlement-a',
      invocation,
      subagentType: invocation.subagentType,
      configId: invocation.configId,
    };
    const settledOutput = { content: 'settled result' };
    const persist = jest.fn(async (): Promise<void> => undefined);
    await record.resolveConfig(
      async () => ({ agentId: 'heavy-agent-config' }),
      new AbortController().signal
    );
    record.activate({ graphId: 'heavy-graph' });
    record.markStarted();
    record.markCompleted({ content: 'heavy execution result' });
    expect(record.snapshot.resolvedConfig).toEqual({
      agentId: 'heavy-agent-config',
    });

    const first = record.settle(settlement, settledOutput, persist);
    const duplicate = record.settle(settlement, settledOutput, persist);

    expect(duplicate).toBe(first);
    await first;
    expect(persist).toHaveBeenCalledTimes(1);
    expect(record.snapshot).toMatchObject({
      phase: 'completed',
      settled: true,
      settling: false,
    });
    expect(record.settledOutput).toEqual(settledOutput);
    expect(record.snapshot.activeRun).toBeUndefined();
    expect(record.snapshot.completedResult).toBeUndefined();
    expect(record.snapshot.resolvedConfig).toBeUndefined();
    await record.settle(settlement, settledOutput, persist);
    expect(persist).toHaveBeenCalledTimes(1);
    expect(() =>
      record.settle(
        { ...settlement, fingerprint: 'settlement-b' },
        settledOutput,
        persist
      )
    ).toThrow('Subagent settlement binding cannot change.');
    expect(() =>
      record.settle(
        {
          ...settlement,
          invocation: { ...invocation, description: 'Changed request.' },
        },
        settledOutput,
        persist
      )
    ).toThrow('Subagent invocation binding cannot change.');
  });

  it('fails closed for ambiguous tool calls and selects an exact resume address', () => {
    const registry = createRegistry();
    const forkA = registry.open(createInput());
    const forkB = registry.open(createInput({ checkpoint_id: 'fork-b' }));
    const unrelated = registry.open({
      ...createInput(),
      parentToolCallId: 'call_unrelated',
    });
    forkA.bindIdentity({
      childRunId: 'run-a',
      childThreadId: 'thread-a',
      approvalExecutionScope: 'scope-a',
    });
    forkB.bindIdentity({
      childRunId: 'run-b',
      childThreadId: 'thread-b',
      approvalExecutionScope: 'scope-b',
    });
    unrelated.bindIdentity({
      childRunId: 'run-c',
      childThreadId: 'thread-c',
      approvalExecutionScope: 'scope-c',
    });

    expect(
      registry.selectForResume({
        parentToolCallIds: new Set(['call_shared']),
      })
    ).toEqual([]);
    expect(registry.selectForResume()).toEqual([unrelated]);
    expect(
      registry.selectForResume({
        parentToolCallIds: new Set(['call_shared']),
        config: {
          configurable: {
            ...createInput().parentConfigurable,
            thread_id: 'durable-thread',
          },
        },
      })
    ).toEqual([forkA]);
  });

  it('retires heavy state from superseded resume sources', () => {
    const registry = createRegistry();
    const attemptA = registry.open(createInput());
    const attemptB = registry.open(
      createInput({
        checkpoint_id: 'fork-b',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-b',
      })
    );
    const attemptC = registry.open(
      createInput({
        checkpoint_id: 'fork-c',
        [SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY]: 'attempt-c',
      })
    );
    attemptA.bindIdentity({
      childRunId: 'child-run',
      childThreadId: 'thread-a',
      approvalExecutionScope: 'scope-a',
    });
    attemptB.bindIdentity({
      childRunId: 'child-run',
      childThreadId: 'thread-b',
      approvalExecutionScope: 'scope-b',
    });
    attemptC.bindIdentity({
      childRunId: 'child-run',
      childThreadId: 'thread-c',
      approvalExecutionScope: 'scope-c',
    });
    attemptA.activate({ graphId: 'graph-a' });
    attemptB.activate({ graphId: 'graph-b' });
    const retired: string[] = [];

    registry.retireResumeSources(attemptB, new Set(['thread-a']), (source) =>
      retired.push(source.activeRun?.graphId ?? 'missing')
    );
    attemptC.activate({ graphId: 'graph-c' });
    registry.retireResumeSources(attemptC, new Set(['thread-b']), (source) =>
      retired.push(source.activeRun?.graphId ?? 'missing')
    );

    expect(retired).toEqual(['graph-a', 'graph-b']);
    expect(attemptA.snapshot.phase).toBe('invalidated');
    expect(attemptB.snapshot.phase).toBe('invalidated');
    expect(attemptA.snapshot.identity).toBeUndefined();
    expect(attemptB.snapshot.identity).toBeUndefined();
    expect(attemptC.snapshot.activeRun).toEqual({ graphId: 'graph-c' });
    expect(registry.selectForResume()).toEqual([attemptC]);
  });

  it('invalidates records atomically while retaining checkpoint cleanup data', () => {
    const registry = createRegistry();
    const record = registry.open(createInput());
    record.bindIdentity({
      childRunId: 'child-run',
      childThreadId: 'child-thread',
      approvalExecutionScope: 'approval-scope',
    });
    record.activate({ graphId: 'graph-a' });
    registry.rememberCheckpointThread('child-thread');
    const disposed: string[] = [];

    registry.clear((current) => {
      disposed.push(current.activeRun?.graphId ?? 'missing');
    });

    expect(disposed).toEqual(['graph-a']);
    expect(record.snapshot).toMatchObject({ phase: 'invalidated' });
    expect(record.snapshot.identity).toBeUndefined();
    expect(record.snapshot.activeRun).toBeUndefined();
    expect(registry.getCheckpointThreadIds()).toEqual(['child-thread']);
  });
});

import { AIMessage } from '@langchain/core/messages';
import type { StopHookInput, StopFinalizeHookInput } from '@/hooks/types';
import type {
  RunLedger,
  VerifierDefinition,
  VerifierContext,
  VerifierVerdict,
  VerificationSnapshot,
} from '..';
import { HookRegistry } from '@/hooks/HookRegistry';
import { executeHooks } from '@/hooks/executeHooks';
import { createCompletionGate, MemoryRunLedger } from '..';

const signal = (): AbortSignal => new AbortController().signal;
const input = (runId = 'run'): StopHookInput => ({
  hook_event_name: 'Stop',
  runId,
  messages: [new AIMessage('done')],
  stopHookActive: false,
  continuationCount: 0,
  continuationBudgetRemaining: 3,
});
const finalize = (
  overrides: Partial<StopFinalizeHookInput> = {}
): StopFinalizeHookInput => ({
  ...input(),
  hook_event_name: 'StopFinalize',
  continuationPlanned: false,
  continuationPrevented: false,
  ...overrides,
});
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
const requirements = [{ mark: 'tests.pass', verifier: 'tests' }];
const snapshot: VerificationSnapshot = {
  tenantId: 'tenant',
  workspaceId: 'workspace',
  revision: '1',
  environmentId: 'env',
};
const versioned = [{ ...requirements[0], version: '1' }];
const reusable = (run: VerifierDefinition['run']): VerifierDefinition => ({
  id: 'tests',
  reuse: { version: 'test-command-v1' },
  run,
});

describe('completion evidence boundaries', () => {
  it('does not run a verifier with an already aborted signal', async () => {
    const run = jest.fn(async () => ({ passed: true }));
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', run }],
    });
    const controller = new AbortController();
    controller.abort();
    await gate.hook(input(), controller.signal);
    expect(run).not.toHaveBeenCalled();
    expect(gate.getCompletion()).toMatchObject({ verified: false, marks: [] });
  });

  it('rejects a late pass from a verifier that ignores cancellation', async () => {
    const result = deferred<VerifierVerdict>();
    const entered = deferred<void>();
    const ledger = new MemoryRunLedger();
    const gate = createCompletionGate({
      requirements,
      ledger,
      verifiers: [
        {
          id: 'tests',
          run: () => {
            entered.resolve();
            return result.promise;
          },
        },
      ],
    });
    const controller = new AbortController();
    const evaluation = gate.hook(input(), controller.signal);
    await entered.promise;
    controller.abort();
    result.resolve({ passed: true });
    await evaluation;
    expect(gate.getCompletion()).toMatchObject({ verified: false, marks: [] });
    expect(ledger.rows().some((row) => row.kind === 'run_completed')).toBe(
      false
    );
  });

  it('fails closed when the final write throws even if a custom ledger still reports healthy', async () => {
    const backing = new MemoryRunLedger();
    const ledger: RunLedger = {
      healthy: true,
      rows: () => backing.rows(),
      blob: (hash) => backing.blob(hash),
      append: (entry, payload) => {
        if (entry.kind === 'run_completed') throw new Error('disk full');
        return backing.append(entry, payload);
      },
    };
    const gate = createCompletionGate({
      requirements,
      ledger,
      verifiers: [
        {
          id: 'tests',
          run: async () => ({ passed: true }),
        },
      ],
    });
    await gate.hook(input(), signal());
    expect(gate.getCompletion()).toMatchObject({ verified: false, marks: [] });
    expect(gate.getCompletion().unverifiedReason).toContain(
      'ledger unavailable'
    );
  });

  it('does not run checks while another Stop hook has planned continuation', async () => {
    const run = jest.fn(async () => ({ passed: true }));
    const registry = new HookRegistry();
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', run }],
    });
    gate.register(registry);
    await executeHooks({
      registry,
      input: finalize({ continuationPlanned: true }),
    });
    expect(run).not.toHaveBeenCalled();
    expect(gate.getCompletion().verified).toBe(false);
    await executeHooks({ registry, input: finalize() });
    expect(run).toHaveBeenCalledTimes(1);
    expect(gate.getCompletion().verified).toBe(true);
    await executeHooks({
      registry,
      input: finalize({ continuationPrevented: true }),
    });
    expect(gate.getCompletion().verified).toBe(false);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('does not publish success if cancellation happens inside the terminal write', async () => {
    const controller = new AbortController();
    const backing = new MemoryRunLedger();
    const ledger: RunLedger = {
      healthy: true, rows: () => backing.rows(), blob: (hash) => backing.blob(hash),
      append: (entry, payload) => {
        const row = backing.append(entry, payload);
        if (entry.kind === 'run_completed') controller.abort();
        return row;
      },
    };
    const gate = createCompletionGate({ requirements, ledger, verifiers: [{ id: 'tests', run: async () => ({ passed: true }) }] });
    await gate.hook(input(), controller.signal);
    expect(gate.getCompletion()).toMatchObject({ verified: false, marks: [] });
    expect(backing.rows().at(-1)?.kind).toBe('completion_unverified');
  });

  it('keeps concurrent runs and their attempt budgets isolated', async () => {
    const result = deferred<VerifierVerdict>();
    const gate = createCompletionGate({
      requirements,
      maxAttempts: 1,
      verifiers: [
        {
          id: 'tests',
          run: async ({ runId }) =>
            runId === 'slow' ? result.promise : { passed: true },
        },
      ],
    });
    const slow = gate.hook(input('slow'), signal());
    await gate.hook(input('fast'), signal());
    expect(gate.getCompletion('fast')).toMatchObject({
      verified: true,
      attempts: 1,
    });
    result.resolve({ passed: false });
    await slow;
    expect(gate.getCompletion('slow')).toMatchObject({
      verified: false,
      attempts: 1,
    });
    expect(gate.getCompletion('fast').verified).toBe(true);
  });

  it('rejects overlapping finalization of the same run', async () => {
    const result = deferred<VerifierVerdict>();
    const entered = deferred<void>();
    const gate = createCompletionGate({
      requirements,
      verifiers: [
        {
          id: 'tests',
          run: () => {
            entered.resolve();
            return result.promise;
          },
        },
      ],
    });
    const first = gate.hook(input(), signal());
    await entered.promise;
    await gate.hook(input(), signal());
    result.resolve({ passed: true });
    await first;
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      marks: [],
      attempts: 1,
    });
  });

  it('retains a timed out operation until the verifier actually settles', async () => {
    const result = deferred<VerifierVerdict>();
    const run = jest.fn(() => result.promise);
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', timeoutMs: 10, run }],
    });
    await gate.hook(input(), signal());
    expect(gate.getCompletion().pendingOperations).toEqual([
      'check:tests.pass',
    ]);
    expect(() => gate.releaseRun('run')).toThrow('active work');
    await gate.hook(input(), signal());
    expect(run).toHaveBeenCalledTimes(1);
    result.resolve({ passed: true });
    await result.promise;
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      pendingOperations: [],
    });
    await gate.hook(input(), signal());
    expect(run).toHaveBeenCalledTimes(2);
    expect(gate.getCompletion().verified).toBe(true);
    gate.releaseRun('run');
    expect(gate.getCompletion('run').attempts).toBe(0);
  });

  it.each([0, -1, NaN, Infinity, 0.5])(
    'rejects invalid attempt and timeout budgets: %s',
    (limit) => {
      expect(() =>
        createCompletionGate({
          requirements: [],
          verifiers: [],
          maxAttempts: limit,
        })
      ).toThrow();
      expect(() =>
        createCompletionGate({
          requirements,
          verifiers: [
            {
              id: 'tests',
              timeoutMs: limit,
              run: async () => ({ passed: true }),
            },
          ],
        })
      ).toThrow();
    }
  );

  it('bounds feedback while retaining complete failure evidence', async () => {
    const ledger = new MemoryRunLedger();
    const feedback = 'diagnostic '.repeat(1000);
    const gate = createCompletionGate({
      requirements,
      ledger,
      maxFeedbackChars: 300,
      verifiers: [
        {
          id: 'tests',
          run: async () => ({ passed: false, feedback }),
        },
      ],
    });
    const output = await gate.hook(input(), signal());
    expect(output.additionalContext?.length).toBeLessThanOrEqual(300);
    expect(output.additionalContext).toContain('Feedback truncated');
    const failure = ledger
      .rows()
      .find((row) => row.kind === 'verification_failed');
    expect(output.additionalContext).toContain(failure!.artifact!.sha256);
    expect(JSON.parse(ledger.blob(failure!.artifact!.sha256)!).feedback).toBe(
      feedback
    );
  });
});

describe('exact-state receipts', () => {
  it('retains an eligible pass across a repair turn while rerunning the failing check', async () => {
    const passing = jest.fn(async () => ({ passed: true }));
    let calls = 0;
    const gate = createCompletionGate({
      requirements: [...versioned, { mark: 'answer.ready', verifier: 'answer' }],
      verifiers: [
        { ...reusable(passing), reuse: { version: '1', includeFinalText: false } },
        { id: 'answer', run: async () => ({ passed: ++calls > 1 }) },
      ],
      captureSnapshot: async () => snapshot,
    });
    expect((await gate.hook(input(), signal())).decision).toBe('block');
    await gate.hook({ ...input(), messages: [new AIMessage('fixed answer')] }, signal());
    expect(gate.getCompletion()).toMatchObject({ verified: true, snapshot });
    expect(passing).toHaveBeenCalledTimes(1);
    expect(calls).toBe(2);
  });

  it('reuses only explicitly eligible passes and records reuse as reuse', async () => {
    const run = jest.fn(async () => ({ passed: true }));
    const ledger = new MemoryRunLedger();
    const gate = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(run)],
      ledger,
      captureSnapshot: async () => snapshot,
    });
    await gate.hook(input(), signal());
    await gate.hook(input(), signal());
    expect(run).toHaveBeenCalledTimes(1);
    expect(gate.getCompletion().marks[0].reused).toBe(true);
    expect(
      ledger.rows().filter((row) => row.kind === 'verification_receipt')
    ).toHaveLength(1);
    expect(
      ledger
        .rows()
        .filter((row) => row.kind === 'mark_admitted')
        .map((row) => row.metadata?.reused)
    ).toEqual([false, true]);
  });

  it.each(['tenantId', 'workspaceId', 'revision', 'environmentId'] as const)(
    'invalidates on a changed %s',
    async (field) => {
      let current = { ...snapshot };
      const run = jest.fn(async () => ({ passed: true }));
      const gate = createCompletionGate({
        requirements: versioned,
        verifiers: [reusable(run)],
        captureSnapshot: async () => current,
      });
      await gate.hook(input(), signal());
      current = { ...current, [field]: 'changed' };
      await gate.hook(input(), signal());
      expect(run).toHaveBeenCalledTimes(2);
      expect(gate.getCompletion().marks[0].reused).toBe(false);
    }
  );

  it('binds evidence to the final answer by default and isolates returned state', async () => {
    const run = jest.fn(async () => ({
      passed: true,
      attestation: {
        issuer: 'issuer',
        keyId: 'key',
        token: 'original',
        trust: 'paired' as const,
      },
    }));
    const gate = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(run)],
      captureSnapshot: async () => snapshot,
    });
    await gate.hook(input(), signal());
    gate.getCompletion().marks[0].attestation!.token = 'mutated';
    await gate.hook(input(), signal());
    expect(gate.getCompletion().marks[0].attestation!.token).toBe('original');
    await gate.hook(
      { ...input(), messages: [new AIMessage('different answer')] },
      signal()
    );
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('does not reuse failures or any result without a snapshot provider', async () => {
    const run = jest.fn(async () => ({ passed: false }));
    const gate = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(run)],
      captureSnapshot: async () => snapshot,
    });
    await gate.hook(input(), signal());
    await gate.hook(input(), signal());
    expect(run).toHaveBeenCalledTimes(2);
    const passing = jest.fn(async () => ({ passed: true }));
    const legacy = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(passing)],
    });
    await legacy.hook(input(), signal());
    await legacy.hook(input(), signal());
    expect(passing).toHaveBeenCalledTimes(2);
    expect(legacy.getCompletion().marks[0].receiptId).toBeUndefined();
  });

  it('rejects a result if the workspace changes while the check is running', async () => {
    let revision = '1';
    const run = jest.fn(async () => {
      revision = '2';
      return { passed: true };
    });
    const ledger = new MemoryRunLedger();
    const gate = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(run)],
      ledger,
      captureSnapshot: async () => ({ ...snapshot, revision }),
    });
    await gate.hook(input(), signal());
    expect(gate.getCompletion()).toMatchObject({ verified: false, marks: [] });
    expect(
      ledger.rows().some((row) => row.kind === 'verification_receipt')
    ).toBe(false);
    await gate.hook(input(), signal());
    expect(run).toHaveBeenCalledTimes(2);
    expect(gate.getCompletion().verified).toBe(true);
  });

  it('fails closed if snapshot capture fails or returns malformed identity', async () => {
    const run = jest.fn(async () => ({ passed: true }));
    const gate = createCompletionGate({
      requirements: versioned,
      verifiers: [reusable(run)],
      captureSnapshot: async () => ({ ...snapshot, revision: '' }),
    });
    await gate.hook(input(), signal());
    expect(run).not.toHaveBeenCalled();
    expect(gate.getCompletion().verified).toBe(false);
    expect(gate.getCompletion().unverifiedReason).toContain(
      'snapshot identity'
    );
  });

  it('freezes requirements, verifier configuration and params at construction', async () => {
    const declared = [{ ...versioned[0], params: { expected: 'original' } }];
    const run = jest.fn(async ({ params }: VerifierContext) => ({
      passed: params.expected === 'original',
    }));
    const definition = reusable(run);
    const gate = createCompletionGate({
      requirements: declared,
      verifiers: [definition],
      captureSnapshot: async () => snapshot,
    });
    declared[0].params.expected = 'mutated';
    definition.run = async () => ({ passed: false });
    await gate.hook(input(), signal());
    expect(gate.getCompletion().verified).toBe(true);
    expect(run).toHaveBeenCalledTimes(1);
  });
});

describe('owned operations', () => {
  it('guards pending work and seals admission even with no declared checks', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const settle = gate.beginOperation('run', 'read', true);
    await gate.hook(input(), signal());
    expect(gate.getCompletion().verified).toBe(false);
    settle();
    await gate.hook(input(), signal());
    expect(gate.getCompletion().verified).toBe(true);
    expect(() => gate.beginOperation('run', 'later')).toThrow('sealed');
  });

  it('blocks unfinished work, makes settlement idempotent, and seals successful runs', async () => {
    const run = jest.fn(async () => ({ passed: true }));
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', run }],
    });
    const settle = gate.beginOperation('run', 'background-build');
    expect(gate.beginOperation('run', 'background-build')).toBe(settle);
    await gate.hook(input(), signal());
    expect(run).not.toHaveBeenCalled();
    expect(gate.getCompletion().pendingOperations).toEqual([
      'background-build',
    ]);
    settle();
    settle();
    await gate.hook(input(), signal());
    expect(gate.getCompletion().verified).toBe(true);
    expect(() => gate.beginOperation('run', 'late-write')).toThrow('sealed');
    gate.invalidate('run');
    expect(gate.getCompletion().verified).toBe(false);
  });

  it('rejects evidence when an operation starts and finishes during verification', async () => {
    const gate = createCompletionGate({
      requirements,
      verifiers: [
        {
          id: 'tests',
          run: async () => {
            const settle = gate.beginOperation('run', 'write');
            settle();
            return { passed: true };
          },
        },
      ],
    });
    await gate.hook(input(), signal());
    expect(gate.getCompletion().verified).toBe(false);
    expect(gate.getCompletion().unverifiedReason).toContain(
      'changed during verification'
    );
  });

  it('preserves eager tool execution and unregisters cleanly', () => {
    const registry = new HookRegistry();
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', run: async () => ({ passed: true }) }],
    });
    const unregister = gate.register(registry);
    expect(registry.hasResultAlteringHooks()).toBe(false);
    expect(registry.hasHookFor('StopFinalize')).toBe(true);
    unregister();
    expect(registry.hasHookFor('StopFinalize')).toBe(false);
  });

  it('isolates identical background operation IDs across runs', async () => {
    const gate = createCompletionGate({
      requirements,
      verifiers: [{ id: 'tests', run: async () => ({ passed: true }) }],
    });
    const first = gate.beginOperation('run', 'build');
    const second = gate.beginOperation('other', 'build');
    first();
    expect(gate.getCompletion('run').pendingOperations).toEqual([]);
    expect(gate.getCompletion('other').pendingOperations).toEqual(['build']);
    await gate.hook(input('run'), signal());
    expect(gate.getCompletion('run').verified).toBe(true);
    second();
  });
});

// src/verification/__tests__/verification.test.ts
import { tmpdir } from 'os';
import { join } from 'path';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { appendFileSync, mkdirSync, mkdtempSync, writeFileSync } from 'fs';
import {
  JsonlRunLedger,
  MemoryRunLedger,
  artifactExists,
  commandPasses,
  createCompletionGate,
  jsonFileValid,
  localCommandRunner,
  registerLedgerHooks,
} from '..';
import type { RunLedger, VerifierDefinition, VerifierVerdict } from '..';
import type {
  HookCallback,
  HookEvent,
  HookMatcher,
  StopHookInput,
} from '@/hooks/types';
import type * as t from '@/types';
import { HookRegistry } from '@/hooks/HookRegistry';
import { Providers } from '@/common';
import { Run } from '@/run';

const tempDir = (): string => mkdtempSync(join(tmpdir(), 'verification-'));

const stopInput = (overrides: Partial<StopHookInput> = {}): StopHookInput => ({
  hook_event_name: 'Stop',
  runId: 'run-1',
  messages: [new HumanMessage('do it'), new AIMessage('done')],
  stopHookActive: false,
  continuationCount: 0,
  continuationBudgetRemaining: 8,
  ...overrides,
});

const signal = (): AbortSignal => new AbortController().signal;

const streamConfig = (threadId: string) => ({
  configurable: { thread_id: threadId },
  streamMode: 'values' as const,
  version: 'v2' as const,
});

/** A verifier that returns the scripted verdicts in order, then repeats the last. */
function scripted(
  id: string,
  verdicts: VerifierVerdict[]
): VerifierDefinition & { calls: number } {
  const definition = {
    id,
    calls: 0,
    run: async (): Promise<VerifierVerdict> => {
      const verdict = verdicts[Math.min(definition.calls, verdicts.length - 1)];
      definition.calls += 1;
      return verdict;
    },
  };
  return definition;
}

/** Captures the callbacks registered on a registry so a test can call one directly. */
function capture(
  registry: HookRegistry
): <E extends HookEvent>(event: E) => HookCallback<E> {
  const matchers = new Map<HookEvent, HookMatcher>();
  const original = registry.register.bind(registry);
  registry.register = (<E extends HookEvent>(
    event: E,
    matcher: HookMatcher<E>
  ) => {
    matchers.set(event, matcher as unknown as HookMatcher);
    return original(event, matcher);
  }) as HookRegistry['register'];
  return <E extends HookEvent>(event: E) =>
    (matchers.get(event) as unknown as HookMatcher<E>).hooks[0];
}

describe('run ledger', () => {
  it('appends contiguous rows with content-addressed blobs', () => {
    const ledger = new JsonlRunLedger(tempDir());
    const a = ledger.append(
      { runId: 'r', kind: 'tool_call', actor: 'model' },
      { label: 'tool.input', content: '{"x":1}' }
    );
    const b = ledger.append({ runId: 'r', kind: 'tool_result', actor: 'tool' });
    expect([a.seq, b.seq]).toEqual([0, 1]);
    expect(a.artifact?.label).toBe('tool.input');
    expect(ledger.blob(a.artifact!.sha256)).toBe('{"x":1}');
    expect(ledger.rows().map((r) => r.kind)).toEqual([
      'tool_call',
      'tool_result',
    ]);
  });

  it('continues the sequence when reopened', () => {
    const dir = tempDir();
    new JsonlRunLedger(dir).append({
      runId: 'r',
      kind: 'tool_call',
      actor: 'model',
    });
    const reopened = new JsonlRunLedger(dir);
    expect(
      reopened.append({ runId: 'r', kind: 'tool_result', actor: 'tool' }).seq
    ).toBe(1);
  });

  it('refuses a log whose rows are not contiguous', () => {
    const dir = tempDir();
    new JsonlRunLedger(dir).append({
      runId: 'r',
      kind: 'tool_call',
      actor: 'model',
    });
    appendFileSync(
      join(dir, 'ledger.jsonl'),
      `${JSON.stringify({ seq: 5, kind: 'mark_admitted' })}\n`
    );
    expect(() => new JsonlRunLedger(dir)).toThrow('not contiguous');
  });

  it('does not return a blob whose content no longer matches its hash', () => {
    const dir = tempDir();
    const ledger = new JsonlRunLedger(dir);
    const row = ledger.append(
      { runId: 'r', kind: 'tool_result', actor: 'tool' },
      { label: 'tool.output', content: 'ok' }
    );
    writeFileSync(join(dir, 'blobs', row.artifact!.sha256), 'tampered');
    expect(ledger.blob(row.artifact!.sha256)).toBeUndefined();
  });

  it('stays unhealthy after a failed write', () => {
    const dir = tempDir();
    const ledger = new JsonlRunLedger(dir);
    // Replace the log path with a directory so the next append fails.
    mkdirSync(join(dir, 'ledger.jsonl'));
    expect(() =>
      ledger.append({ runId: 'r', kind: 'tool_call', actor: 'model' })
    ).toThrow();
    expect(ledger.healthy).toBe(false);
    expect(() =>
      ledger.append({ runId: 'r', kind: 'tool_call', actor: 'model' })
    ).toThrow('ledger unavailable');
  });
});

describe('ledger hooks', () => {
  it('isolates timing when tool IDs collide across runs', async () => {
    const registry = new HookRegistry();
    const hookFor = capture(registry);
    const ledger = new MemoryRunLedger();
    const unregister = registerLedgerHooks(registry, ledger);
    const tool = { toolName: 'bash', toolInput: {}, toolUseId: 'same' };
    for (const runId of ['first', 'second']) {
      await hookFor('PreToolUse')({ ...tool, runId, hook_event_name: 'PreToolUse' }, signal());
    }
    for (const runId of ['first', 'second']) {
      await hookFor('PostToolUse')({ ...tool, runId, hook_event_name: 'PostToolUse', toolOutput: 'ok' }, signal());
    }
    expect(ledger.rows().filter((row) => row.kind === 'tool_result').map((row) => row.metadata?.durationMs)).toEqual([expect.any(Number), expect.any(Number)]);
    unregister();
  });

  it('records calls and results with durations, and denies tools when the ledger is down', async () => {
    const registry = new HookRegistry();
    const hookFor = capture(registry);
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const base = {
      runId: 'r',
      toolName: 'bash',
      toolInput: { command: 'ls' },
      toolUseId: 'u1',
    };
    expect(
      await hookFor('PreToolUse')(
        { hook_event_name: 'PreToolUse', ...base },
        signal()
      )
    ).toEqual({});
    await hookFor('PostToolUse')(
      { hook_event_name: 'PostToolUse', ...base, toolOutput: 'a\nb' },
      signal()
    );
    const [call, result] = ledger.rows();
    expect(call.kind).toBe('tool_call');
    expect(ledger.blob(call.artifact!.sha256)).toBe('{"command":"ls"}');
    expect(result.kind).toBe('tool_result');
    expect(typeof result.metadata?.durationMs).toBe('number');

    const broken: RunLedger = {
      healthy: false,
      failureReason: 'disk full',
      append: () => {
        throw new Error('disk full');
      },
      rows: () => [],
      blob: () => undefined,
    };
    const registry2 = new HookRegistry();
    const hookFor2 = capture(registry2);
    registerLedgerHooks(registry2, broken);
    const denied = await hookFor2('PreToolUse')(
      { hook_event_name: 'PreToolUse', ...base },
      signal()
    );
    expect(denied).toMatchObject({ decision: 'deny' });
  });
});

describe('completion gate', () => {
  it('rejects duplicate marks and unknown verifiers at creation', () => {
    const v = scripted('v', [{ passed: true }]);
    expect(() =>
      createCompletionGate({
        requirements: [
          { mark: 'a', verifier: 'v' },
          { mark: 'a', verifier: 'v' },
        ],
        verifiers: [v],
      })
    ).toThrow('duplicate');
    expect(() =>
      createCompletionGate({
        requirements: [{ mark: 'a', verifier: 'nope' }],
        verifiers: [v],
      })
    ).toThrow('unknown verifier');
  });

  it('admits marks only when every requirement passes in the same evaluation', async () => {
    const ledger = new MemoryRunLedger();
    const tests = scripted('tests', [{ passed: true }, { passed: true }]);
    const file = scripted('file', [
      { passed: false, feedback: 'missing report' },
      { passed: true },
    ]);
    const gate = createCompletionGate({
      requirements: [
        { mark: 'tests.pass', verifier: 'tests' },
        { mark: 'report.exists', verifier: 'file' },
      ],
      verifiers: [tests, file],
      ledger,
    });
    const first = await gate.hook(stopInput(), signal());
    expect(first.decision).toBe('block');
    expect(first.additionalContext).toContain('report.exists: missing report');
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      missing: ['report.exists'],
      attempts: 1,
    });

    const second = await gate.hook(
      stopInput({ continuationCount: 1 }),
      signal()
    );
    expect(second).toEqual({});
    expect(gate.getCompletion()).toMatchObject({
      verified: true,
      missing: [],
      attempts: 2,
    });
    // The earlier pass of `tests` was re-checked, not carried over.
    expect(tests.calls).toBe(2);
    expect(ledger.rows().map((r) => r.kind)).toEqual([
      'mark_admitted',
      'verification_failed',
      'mark_admitted',
      'mark_admitted',
      'run_completed',
    ]);
    expect(
      ledger
        .rows()
        .every((r) => r.kind !== 'mark_admitted' || r.actor === 'runtime')
    ).toBe(true);
  });

  it('ends unverified after maxAttempts or when the continuation budget is spent', async () => {
    const failing = scripted('t', [{ passed: false }]);
    const gate = createCompletionGate({
      requirements: [{ mark: 'm', verifier: 't' }],
      verifiers: [failing],
      maxAttempts: 2,
    });
    expect((await gate.hook(stopInput(), signal())).decision).toBe('block');
    expect(
      await gate.hook(stopInput({ continuationCount: 1 }), signal())
    ).toEqual({});
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      missing: ['m'],
    });
    expect(gate.getCompletion().unverifiedReason).toContain(
      'after 2 evaluations'
    );

    const budget = createCompletionGate({
      requirements: [{ mark: 'm', verifier: 't' }],
      verifiers: [failing],
    });
    expect(
      await budget.hook(stopInput({ continuationBudgetRemaining: 0 }), signal())
    ).toEqual({});
    expect(budget.getCompletion().unverifiedReason).toBe(
      'continuation budget exhausted'
    );
  });

  it('fails closed on a throw, a malformed verdict, and an unhealthy ledger', async () => {
    const throwing: VerifierDefinition = {
      id: 'throw',
      run: async () => {
        throw new Error('boom');
      },
    };
    const malformed: VerifierDefinition = {
      id: 'bad',
      run: async () => ({}) as VerifierVerdict,
    };
    const gate = createCompletionGate({
      requirements: [
        { mark: 'a', verifier: 'throw' },
        { mark: 'b', verifier: 'bad' },
      ],
      verifiers: [throwing, malformed],
    });
    const out = await gate.hook(stopInput(), signal());
    expect(out.decision).toBe('block');
    expect(gate.getCompletion().missing).toEqual(['a', 'b']);
    expect(out.additionalContext).toContain('could not run: boom');
    expect(out.additionalContext).toContain('no boolean `passed`');

    const ledger = new MemoryRunLedger();
    const unhealthy = Object.create(ledger, {
      healthy: { value: false },
      failureReason: { value: 'disk full' },
    }) as RunLedger;
    const gated = createCompletionGate({
      requirements: [{ mark: 'a', verifier: 'ok' }],
      verifiers: [scripted('ok', [{ passed: true }])],
      ledger: unhealthy,
    });
    expect(await gated.hook(stopInput(), signal())).toEqual({});
    expect(gated.getCompletion()).toMatchObject({ verified: false });
    expect(gated.getCompletion().unverifiedReason).toContain(
      'ledger unavailable'
    );
  });

  it('treats an early stop as unverified and reports not-evaluated before any stop', async () => {
    const gate = createCompletionGate({
      requirements: [{ mark: 'a', verifier: 'ok' }],
      verifiers: [scripted('ok', [{ passed: true }])],
    });
    expect(gate.getCompletion().unverifiedReason).toBe(
      'requirements were not evaluated'
    );
    expect(
      await gate.hook(stopInput({ stopReason: 'output_truncated' }), signal())
    ).toEqual({});
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      attempts: 0,
    });
  });

  it('keeps a real Run going until the check passes, inside one processStream', async () => {
    jest.setTimeout(15000);
    const registry = new HookRegistry();
    const ledger = new MemoryRunLedger();
    registerLedgerHooks(registry, ledger);
    const check = scripted('tests', [
      { passed: false, feedback: '1 test failing' },
      { passed: true },
    ]);
    const gate = createCompletionGate({
      requirements: [{ mark: 'tests.pass', verifier: 'tests' }],
      verifiers: [check],
      ledger,
    });
    gate.register(registry);
    const llmConfig: t.LLMConfig = {
      provider: Providers.OPENAI,
      streaming: true,
      streamUsage: false,
    };
    const run = await Run.create<t.IState>({
      runId: 'gated-run',
      graphConfig: { type: 'standard', llmConfig },
      returnContent: true,
      skipCleanup: true,
      hooks: registry,
    });
    run.Graph!.overrideTestModel(['first answer', 'fixed answer']);
    await run.processStream(
      { messages: [new HumanMessage('fix the bug')] },
      streamConfig('gated-thread')
    );
    expect(check.calls).toBe(2);
    expect(gate.getCompletion()).toMatchObject({ verified: true, attempts: 2 });
    const contents = run.Graph!.messages.map((m) => m.content);
    expect(contents[1]).toBe('first answer');
    expect(String(contents[2])).toContain('1 test failing');
    expect(contents[contents.length - 1]).toBe('fixed answer');
    expect(ledger.rows().map((r) => r.kind)).toEqual(
      expect.arrayContaining([
        'stop_requested',
        'verification_failed',
        'mark_admitted',
        'run_completed',
      ])
    );
  });

  it('ends a real Run unverified when the check never passes', async () => {
    const registry = new HookRegistry();
    const gate = createCompletionGate({
      requirements: [{ mark: 'tests.pass', verifier: 'tests' }],
      verifiers: [scripted('tests', [{ passed: false }])],
      maxAttempts: 2,
    });
    gate.register(registry);
    const llmConfig: t.LLMConfig = {
      provider: Providers.OPENAI,
      streaming: true,
      streamUsage: false,
    };
    const run = await Run.create<t.IState>({
      runId: 'unverified-run',
      graphConfig: { type: 'standard', llmConfig },
      returnContent: true,
      skipCleanup: true,
      hooks: registry,
    });
    run.Graph!.overrideTestModel(['a', 'b', 'c']);
    await run.processStream(
      { messages: [new HumanMessage('fix the bug')] },
      streamConfig('unverified-thread')
    );
    expect(gate.getCompletion()).toMatchObject({
      verified: false,
      attempts: 2,
      missing: ['tests.pass'],
    });
  });

  it('checks the final segment only when another Stop hook continues a real Run', async () => {
    const registry = new HookRegistry();
    registry.register('Stop', { hooks: [(input) => input.continuationCount === 0
      ? { decision: 'block', additionalContext: 'Consider the remaining requirement.' }
      : {}] });
    const checked: Array<string | null> = [];
    const gate = createCompletionGate({
      requirements: [{ mark: 'tests.pass', verifier: 'tests' }],
      verifiers: [{ id: 'tests', run: async ({ finalText }) => { checked.push(finalText); return { passed: true }; } }],
    });
    gate.register(registry);
    const run = await Run.create<t.IState>({
      runId: 'final-segment', graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI, streaming: true, streamUsage: false } },
      returnContent: true, skipCleanup: true, hooks: registry,
    });
    run.Graph!.overrideTestModel(['first answer', 'final answer']);
    await run.processStream({ messages: [new HumanMessage('do it')] }, streamConfig('final-segment-thread'));
    expect(checked).toEqual(['final answer']);
    expect(gate.getCompletion()).toMatchObject({ verified: true, attempts: 1 });
  });
});

describe('verifiers', () => {
  it('artifact_exists and json_file_valid stay inside the workspace', async () => {
    const root = tempDir();
    writeFileSync(join(root, 'ANSWER.json'), '{"q1":1}');
    writeFileSync(join(root, 'bad.json'), '{');
    const ctx = (params: Record<string, unknown>) => ({
      runId: 'r',
      requirement: { mark: 'm', verifier: 'v' },
      params,
      finalText: null,
      signal: signal(),
    });
    expect(
      (await artifactExists(root).run(ctx({ path: 'ANSWER.json' }))).passed
    ).toBe(true);
    expect(
      (await artifactExists(root).run(ctx({ path: 'missing.json' }))).passed
    ).toBe(false);
    expect(
      (await artifactExists(root).run(ctx({ path: '../etc/passwd' }))).passed
    ).toBe(false);
    expect(
      (await artifactExists(root).run(ctx({ path: '/etc/passwd' }))).passed
    ).toBe(false);
    expect(
      (await jsonFileValid(root).run(ctx({ path: 'ANSWER.json' }))).passed
    ).toBe(true);
    expect(
      (await jsonFileValid(root).run(ctx({ path: 'bad.json' }))).passed
    ).toBe(false);
  });

  it('command_passes runs the host command through the local engine', async () => {
    const root = tempDir();
    const runner = localCommandRunner({ cwd: root });
    const ctx = {
      runId: 'r',
      requirement: { mark: 'm', verifier: 'v' },
      params: {},
      finalText: null,
      signal: signal(),
    };
    expect(
      (await commandPasses({ id: 'ok', command: 'echo fine', runner }).run(ctx))
        .passed
    ).toBe(true);
    const failed = await commandPasses({
      id: 'no',
      command: 'echo broken >&2; exit 3',
      runner,
    }).run(ctx);
    expect(failed.passed).toBe(false);
    expect(failed.details).toMatchObject({ exitCode: 3 });
    expect(failed.feedback).toContain('broken');
  });
});

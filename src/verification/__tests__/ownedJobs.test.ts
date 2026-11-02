import { HumanMessage } from '@langchain/core/messages';
import { readFile, rm } from 'fs/promises';
import type { StopHookInput, StopFinalizeHookInput } from '@/hooks/types';
import type * as t from '@/types';
import { createOwnedJobManager, createCompletionGate } from '..';
import { createLocalProgrammaticToolCallingTool } from '@/tools/local/LocalProgrammaticToolCalling';
import { createLocalBackgroundTools } from '@/tools/local/LocalBackgroundTools';
import { executeHooks } from '@/hooks/executeHooks';
import { HookRegistry } from '@/hooks';
import { Providers } from '@/common';
import { Run } from '@/run';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  return { promise: new Promise<T>((done) => { resolve = done; }), resolve: (value) => resolve(value) };
}
const stop = (remaining = 3): StopHookInput => ({ hook_event_name: 'Stop', runId: 'owned', messages: [], stopHookActive: false, continuationCount: 0, continuationBudgetRemaining: remaining });
const finalize = (planned = false): StopFinalizeHookInput => ({ ...stop(), hook_event_name: 'StopFinalize', continuationPlanned: planned, continuationPrevented: false });

describe('owned deterministic jobs', () => {
  it('admits once, preserves capacity through cancellation, and rejects changed replay inputs', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const jobs = createOwnedJobManager({ runId: 'owned', gate, maxConcurrent: 1 });
    const work = deferred<string>();
    const entered = deferred<AbortSignal>();
    const execute = jest.fn((signal: AbortSignal) => { entered.resolve(signal); return work.promise; });
    try {
      expect(jobs.start({ id: 'check', fingerprint: 'input', execute }).status).toBe('running');
      const signal = await entered.promise;
      jobs.start({ id: 'check', fingerprint: 'input', execute });
      expect(execute).toHaveBeenCalledTimes(1);
      expect(() => jobs.start({ id: 'check', fingerprint: 'changed', execute })).toThrow('different inputs');
      expect(jobs.cancel('check').status).toBe('cancelling');
      expect(signal.aborted).toBe(true);
      expect(() => jobs.start({ id: 'second', fingerprint: 'x', execute })).toThrow('capacity');
      await gate.hook(finalize(), new AbortController().signal);
      expect(gate.getCompletion().verified).toBe(false);
      let closed = false;
      const closing = jobs.closeAndDrain().then(() => { closed = true; });
      await Promise.resolve();
      expect(closed).toBe(false);
      work.resolve('completed effects remain');
      await closing;
      expect(jobs.get('check')).toMatchObject({ status: 'cancelled', output: 'completed effects remain' });
      expect(() => jobs.start({ id: 'later', fingerprint: 'x', execute })).toThrow('closed');
    } finally { work.resolve('cleanup'); await jobs.closeAndDrain(); }
  });

  it('delivers each result once and prevents verification of undelivered work', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const hooks = new HookRegistry();
    const jobs = createOwnedJobManager({ runId: 'owned', gate, maxResultChars: 64 });
    const unregister = jobs.register(hooks);
    try {
      jobs.start({ id: 'failure', fingerprint: 'x', execute: async () => { throw new Error('failed operation'); } });
      await jobs.wait();
      await gate.hook(finalize(), new AbortController().signal);
      expect(gate.getCompletion().verified).toBe(false);
      const first = await executeHooks({ registry: hooks, sessionId: 'owned', input: stop() });
      expect(first.stopDecision).toBe('block');
      expect(first.additionalContexts.join('')).toContain('failed operation');
      await executeHooks({ registry: hooks, sessionId: 'owned', input: finalize(true) });
      const second = await executeHooks({ registry: hooks, sessionId: 'owned', input: { ...stop(), continuationCount: 1 } });
      expect(second.additionalContexts).toEqual([]);
      await executeHooks({ registry: hooks, sessionId: 'owned', input: finalize() });
      expect(gate.getCompletion().verified).toBe(true);
      expect(() => jobs.start({ id: 'late', fingerprint: 'x', execute: async () => 'late' })).toThrow('closed');
    } finally { unregister(); await jobs.closeAndDrain(); }
  });

  it('does not acknowledge results discarded by a competing Stop halt', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const hooks = new HookRegistry();
    const jobs = createOwnedJobManager({ runId: 'owned', gate });
    const unregister = jobs.register(hooks);
    hooks.registerSession('owned', 'Stop', { hooks: [() => ({ preventContinuation: true })] });
    const run = await Run.create<t.IState>({ runId: 'owned', hooks, returnContent: true, skipCleanup: true,
      graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI, streaming: true } },
    });
    run.Graph!.overrideTestModel(['done']);
    try {
      jobs.start({ id: 'ready', fingerprint: 'x', execute: async () => 'undelivered-result' });
      await run.processStream({ messages: [new HumanMessage('Do work.')] }, { configurable: { thread_id: 'owned' }, version: 'v2' });
      expect(jobs.get('ready').delivered).toBe(false);
      expect(gate.getCompletion().verified).toBe(false);
      expect(run.Graph!.messages.some((message) => String(message.content).includes('undelivered-result'))).toBe(false);
    } finally { unregister(); await jobs.closeAndDrain(); }
  });

  it('reopens admission for a continuation requested by the completion verifier', async () => {
    const gate = createCompletionGate({ requirements: [{ mark: 'repair', verifier: 'check' }], verifiers: [{ id: 'check', run: async () => ({ passed: false, feedback: 'Repair needed.' }) }] });
    const hooks = new HookRegistry();
    const jobs = createOwnedJobManager({ runId: 'owned', gate });
    const unregister = jobs.register(hooks);
    try {
      await executeHooks({ registry: hooks, sessionId: 'owned', input: stop() });
      const result = await executeHooks({ registry: hooks, sessionId: 'owned', input: finalize() });
      expect(result.stopDecision).toBe('block');
      expect(jobs.start({ id: 'repair', fingerprint: 'x', execute: async () => 'repaired' }).status).toBe('running');
    } finally { unregister(); await jobs.closeAndDrain(); }
  });

  it.each([[false, 64], [true, 64], [false, 10_000], [true, 10_000]] as const)('preserves output recovery and execution status across background truncation (failed=%s, engine cap=%s)', async (failed, maxOutputChars) => {
    const jobs = createOwnedJobManager({ runId: 'owned', gate: createCompletionGate({ requirements: [], verifiers: [] }), maxResultChars: 32 });
    const bundle = createLocalBackgroundTools(jobs, { maxOutputChars });
    let path: string | undefined;
    try {
      await bundle.tools[0].invoke({ id: 'verbose', command: `printf '%0100d' 0${failed ? '; exit 7' : ''}` });
      await jobs.wait();
      const result = jobs.get('verbose');
      path = result.fullOutputPath;
      expect(result.status).toBe(failed ? 'failed' : 'succeeded');
      expect(result.exitCode).toBe(failed ? 7 : 0);
      expect(result.outputTruncated).toBe(true);
      expect(path).toBeDefined();
      expect(await readFile(path!, 'utf8')).toBe('0'.repeat(100));
    } finally { await jobs.closeAndDrain(); if (path != null) await rm(path, { force: true }); }
  });

  it('retains undelivered results when continuation is exhausted', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const hooks = new HookRegistry();
    const jobs = createOwnedJobManager({ runId: 'owned', gate });
    const unregister = jobs.register(hooks);
    try {
      jobs.start({ id: 'ready', fingerprint: 'x', execute: async () => 'ready-result' });
      await jobs.wait();
      const output = await executeHooks({ registry: hooks, sessionId: 'owned', input: stop(0) });
      expect(output.additionalContexts).toEqual([]);
      await gate.hook({ ...finalize(), continuationBudgetRemaining: 0 }, new AbortController().signal);
      expect(gate.getCompletion().verified).toBe(false);
      expect(jobs.get('ready').output).toBe('ready-result');
    } finally { unregister(); await jobs.closeAndDrain(); }
  });

  it('abortable event waits leave actual ownership pending until settlement', async () => {
    const parent = new AbortController();
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const jobs = createOwnedJobManager({ runId: 'owned', gate, signal: parent.signal });
    const work = deferred<string>();
    const entered = deferred<void>();
    jobs.start({ id: 'work', fingerprint: 'x', execute: async () => { entered.resolve(); return work.promise; } });
    try {
      await entered.promise;
      const waiting = jobs.wait(parent.signal);
      parent.abort(new Error('cancel parent'));
      await expect(waiting).rejects.toThrow('cancel parent');
      expect(jobs.get('work').status).toBe('cancelling');
      await gate.hook(finalize(), new AbortController().signal);
      expect(gate.getCompletion().unverifiedReason).toContain('not settled');
    } finally { work.resolve('cleanup'); await jobs.closeAndDrain(); }
  });

  it('a real Run overlaps foreground work and waits once before a single result continuation', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const hooks = new HookRegistry();
    const jobs = createOwnedJobManager({ runId: 'owned', gate });
    const unregister = jobs.register(hooks);
    const work = deferred<string>();
    const foregroundFinished = deferred<void>();
    hooks.registerSession('owned', 'Stop', { hooks: [() => { foregroundFinished.resolve(); return {}; }] });
    const run = await Run.create<t.IState>({ runId: 'owned', hooks, returnContent: true, skipCleanup: true,
      graphConfig: { type: 'standard', llmConfig: { provider: Providers.OPENAI, streaming: true, streamUsage: false } },
    });
    run.Graph!.overrideTestModel(['foreground done', 'background result incorporated']);
    jobs.start({ id: 'check', fingerprint: 'x', execute: () => work.promise });
    const processing = run.processStream({ messages: [new HumanMessage('Do independent work.')] }, { configurable: { thread_id: 'owned' }, version: 'v2' });
    try {
      await foregroundFinished.promise;
      expect(jobs.get('check').status).toBe('running');
      expect(run.Graph!.messages.filter((message) => message.getType() === 'ai')).toHaveLength(1);
      work.resolve('background-evidence');
      await processing;
      expect(run.Graph!.messages.filter((message) => message.getType() === 'ai')).toHaveLength(2);
      expect(run.Graph!.messages.filter((message) => String(message.content).includes('background-evidence'))).toHaveLength(1);
      expect(gate.getCompletion().verified).toBe(true);
    } finally { work.resolve('cleanup'); await jobs.closeAndDrain(); await processing; unregister(); }
  });

  it('closing a programmatic bridge does not cancel the command owned by its Run', async () => {
    const gate = createCompletionGate({ requirements: [], verifiers: [] });
    const jobs = createOwnedJobManager({ runId: 'owned', gate });
    const bundle = createLocalBackgroundTools(jobs);
    const runner = createLocalProgrammaticToolCallingTool();
    try {
      await runner.invoke({ lang: 'py', tool_manifest: ['start_owned_command'], code: 'print(await start_owned_command(id="long",command="sleep 30"))' },
        { toolCall: { name: runner.name, args: {}, toolMap: new Map(bundle.tools.map((bound) => [bound.name, bound])), toolDefs: [...bundle.toolRegistry.values()] } });
      expect(jobs.get('long').status).toBe('running');
      jobs.cancel('long');
      await jobs.wait();
      expect(jobs.get('long').status).toBe('cancelled');
    } finally { await jobs.closeAndDrain(); }
  });
});

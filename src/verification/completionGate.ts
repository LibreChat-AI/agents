import type {
  CompletionRequirement,
  CompletionState,
  RunLedger,
  VerificationReceipt,
  VerificationSnapshot,
  VerifierDefinition,
  VerifierVerdict,
} from './types';
import type {
  StopFinalizeHookInput,
  StopHookInput,
  StopHookOutput,
} from '@/hooks/types';
import type { HookRegistry } from '@/hooks/HookRegistry';
import { finalAssistantText } from './ledgerHooks';
import { isAborted, withinDeadline } from './deadline';
import { sha256 } from './ledger';

export type CompletionGateOptions = {
  requirements: CompletionRequirement[];
  verifiers: VerifierDefinition[];
  ledger?: RunLedger;
  maxAttempts?: number;
  /** Must observe a monotonic workspace epoch or an immutable snapshot, including external inputs. */
  captureSnapshot?: (
    runId: string,
    signal: AbortSignal
  ) => Promise<VerificationSnapshot>;
  snapshotTimeoutMs?: number;
  /** Maximum characters of model-visible failure feedback, default 4000. */
  maxFeedbackChars?: number;
};

export type CompletionGate = {
  /** Registers only StopFinalize, preserving the existing eager tool execution path. */
  register(registry: HookRegistry): () => void;
  hook(
    input: StopHookInput | StopFinalizeHookInput,
    signal: AbortSignal
  ): Promise<StopHookOutput>;
  readonly timeoutMs: number;
  /** With no ID, returns the run most recently touched by this gate. */
  getCompletion(runId?: string): CompletionState;
  /** Call before launching detached work; settle only when it actually finishes. */
  beginOperation(
    runId: string,
    operationId: string,
    readOnly?: boolean
  ): () => void;
  /** Host notification of writes, requirement changes or uncertain external state. */
  invalidate(runId: string): void;
  /** Releases terminal state. Refuses to forget active work. */
  releaseRun(runId: string): void;
};

type Evaluation = {
  requirement: CompletionRequirement;
  verdict: VerifierVerdict;
  receipt?: VerificationReceipt;
  reused: boolean;
  artifactHash?: string;
};
type RunEvidence = {
  state: CompletionState;
  epoch: number;
  evaluating: boolean;
  sealed: boolean;
  pending: Map<string, () => void>;
  unsettled: Map<symbol, string>;
  receipts: Map<string, VerificationReceipt>;
};
const DEFAULT_TIMEOUT = 60_000;

function positiveInteger(value: number, name: string, minimum = 1): number {
  if (
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > 2_147_483_647
  ) {
    throw new Error(`${name} must be an integer from ${minimum} to 2147483647`);
  }
  return value;
}

function snapshotKey(snapshot: VerificationSnapshot): string {
  const parts = [
    snapshot.tenantId,
    snapshot.workspaceId,
    snapshot.revision,
    snapshot.environmentId,
  ];
  if (parts.some((part) => typeof part !== 'string' || part.length === 0)) {
    throw new Error('snapshot identity fields must be nonempty strings');
  }
  return JSON.stringify(parts);
}

/** Host-owned checks with bounded repair, exact-state reuse and explicit unfinished work. */
export function createCompletionGate(
  options: CompletionGateOptions
): CompletionGate {
  const requirements = structuredClone(options.requirements);
  const verifiers = new Map<string, VerifierDefinition>();
  for (const definition of options.verifiers) {
    if (verifiers.has(definition.id))
      throw new Error(`duplicate verifier: ${definition.id}`);
    positiveInteger(
      definition.timeoutMs ?? DEFAULT_TIMEOUT,
      'verifier timeout'
    );
    if (
      definition.reuse !== undefined &&
      (typeof definition.reuse.version !== 'string' ||
        definition.reuse.version.length === 0)
    )
      throw new Error('verifier reuse version must be nonempty');
    verifiers.set(definition.id, {
      ...definition,
      defaultParams: structuredClone(definition.defaultParams),
      reuse:
        definition.reuse === undefined ? undefined : { ...definition.reuse },
    });
  }
  const marks = new Set<string>();
  for (const requirement of requirements) {
    if (marks.has(requirement.mark))
      throw new Error(`duplicate completion mark: ${requirement.mark}`);
    marks.add(requirement.mark);
    const definition = verifiers.get(requirement.verifier);
    if (definition === undefined)
      throw new Error(
        `completion mark ${requirement.mark} names unknown verifier ${requirement.verifier}`
      );
    if (
      definition.reuse !== undefined &&
      (typeof requirement.version !== 'string' ||
        requirement.version.length === 0)
    ) {
      throw new Error(
        `completion mark ${requirement.mark} needs a version for reuse`
      );
    }
    const limit =
      requirement.params?.timeout_ms ?? definition.defaultParams?.timeout_ms;
    if (limit !== undefined) {
      if (typeof limit !== 'number')
        throw new Error('requirement timeout must be numeric');
      positiveInteger(limit, 'requirement timeout');
    }
  }
  const maxAttempts = positiveInteger(options.maxAttempts ?? 3, 'maxAttempts');
  const snapshotTimeout = positiveInteger(
    options.snapshotTimeoutMs ?? 5000,
    'snapshotTimeoutMs'
  );
  const feedbackLimit = positiveInteger(
    options.maxFeedbackChars ?? 4000,
    'maxFeedbackChars',
    256
  );
  const { ledger, captureSnapshot } = options;
  const timeoutMs = positiveInteger(
    requirements.reduce(
      (sum, r) =>
        sum + (verifiers.get(r.verifier)?.timeoutMs ?? DEFAULT_TIMEOUT),
      5000
    ) + (captureSnapshot === undefined ? 0 : 2 * snapshotTimeout),
    'gate timeout'
  );
  const runs = new Map<string, RunEvidence>();
  let latestRun: string | undefined;
  let ledgerFailure: string | undefined;
  const initial = (): CompletionState => ({
    verified: requirements.length === 0,
    attempts: 0,
    marks: [],
    missing: requirements.map((r) => r.mark),
    pendingOperations: [],
    ...(requirements.length === 0
      ? {}
      : { unverifiedReason: 'requirements were not evaluated' }),
  });
  const runFor = (runId: string): RunEvidence => {
    latestRun = runId;
    let run = runs.get(runId);
    if (run === undefined) {
      run = {
        state: initial(),
        epoch: 0,
        evaluating: false,
        sealed: false,
        pending: new Map(),
        unsettled: new Map(),
        receipts: new Map(),
      };
      runs.set(runId, run);
    }
    return run;
  };
  const clearCompletion = (run: RunEvidence, reason: string): void => {
    run.state.verified = false;
    run.state.marks = [];
    run.state.missing = requirements.map((r) => r.mark);
    run.state.unverifiedReason = reason;
    delete run.state.snapshot;
    run.sealed = false;
  };
  const invalidate = (runId: string): void => {
    const run = runFor(runId);
    run.epoch += 1;
    run.receipts.clear();
    clearCompletion(run, 'evidence invalidated');
  };
  const beginOperation = (
    runId: string,
    operationId: string,
    readOnly = false
  ): (() => void) => {
    const run = runFor(runId);
    const existing = run.pending.get(operationId);
    if (existing !== undefined) return existing;
    if (run.sealed)
      throw new Error(
        'verified run is sealed; invalidate it before admitting more work'
      );
    if (readOnly !== true) invalidate(runId);
    else clearCompletion(run, 'operation pending');
    let finished = false;
    const settle = (): void => {
      if (finished) return;
      finished = true;
      run.pending.delete(operationId);
      if (readOnly !== true) invalidate(runId);
    };
    run.pending.set(operationId, settle);
    return settle;
  };
  const append = (...args: Parameters<RunLedger['append']>): number | null => {
    if (ledger === undefined || ledgerFailure !== undefined) return null;
    try {
      return ledger.append(...args).seq;
    } catch (error) {
      ledgerFailure = String(error);
      return null;
    }
  };
  const unhealthy = (): boolean =>
    ledgerFailure !== undefined || ledger?.healthy === false;
  const hasPending = (run: RunEvidence): boolean =>
    run.pending.size + run.unsettled.size > 0;
  const endUnverified = (
    runId: string,
    run: RunEvidence,
    reason: string,
    clear = true
  ): StopHookOutput => {
    if (clear) clearCompletion(run, reason);
    run.state.verified = false;
    run.state.unverifiedReason = reason;
    append({
      runId,
      kind: 'completion_unverified',
      actor: 'runtime',
      metadata: {
        reason,
        missing: run.state.missing,
        attempts: run.state.attempts,
      },
    });
    if (unhealthy()) {
      clearCompletion(
        run,
        `evidence ledger unavailable: ${ledgerFailure ?? ledger?.failureReason ?? 'write failed'}`
      );
      run.receipts.clear();
    }
    return {};
  };
  const bounded = async <T>(
    run: RunEvidence,
    id: string,
    operation: (signal: AbortSignal) => Promise<T>,
    signal: AbortSignal,
    limit: number
  ): Promise<T> => {
    const token = Symbol(id);
    run.unsettled.set(token, id);
    const result = await withinDeadline(operation, signal, limit, () => {
      run.unsettled.delete(token);
    });
    if (result.status !== 'completed') throw new Error(result.message);
    return result.value;
  };
  const evaluate = async (
    input: StopHookInput | StopFinalizeHookInput,
    signal: AbortSignal,
    run: RunEvidence
  ): Promise<StopHookOutput> => {
    const { runId } = input;
    clearCompletion(run, 'verification in progress');
    if (isAborted(signal))
      return endUnverified(runId, run, 'verification cancelled');
    if (input.stopReason != null)
      return endUnverified(
        runId,
        run,
        `run stopped early: ${input.stopReason}`
      );
    if (input.hook_event_name === 'StopFinalize') {
      if (input.continuationPrevented)
        return endUnverified(runId, run, 'continuation prevented');
      if (input.continuationPlanned) return {};
    }
    if (unhealthy())
      return endUnverified(runId, run, 'evidence ledger unavailable');
    if (hasPending(run))
      return endUnverified(runId, run, 'owned operations have not settled');
    if (run.state.attempts >= maxAttempts)
      return endUnverified(
        runId,
        run,
        `required checks still failing after ${run.state.attempts} evaluations`
      );
    if (requirements.length === 0) {
      run.sealed = true;
      run.state.verified = true;
      delete run.state.unverifiedReason;
      return {};
    }
    run.state.attempts += 1;
    const attempt = run.state.attempts;
    const epoch = run.epoch;
    const finalText = finalAssistantText(input.messages);
    const capture = async (): Promise<VerificationSnapshot | undefined> => {
      if (captureSnapshot === undefined) return undefined;
      const snapshot = structuredClone(
        await bounded(
          run,
          'snapshot',
          (child) => captureSnapshot(runId, child),
          signal,
          snapshotTimeout
        )
      );
      snapshotKey(snapshot);
      return snapshot;
    };
    const snapshot = await capture();
    const results: Evaluation[] = [];
    for (const requirement of requirements) {
      if (isAborted(signal) || run.epoch !== epoch || hasPending(run)) break;
      const definition = verifiers.get(
        requirement.verifier
      ) as VerifierDefinition;
      const params = { ...definition.defaultParams, ...requirement.params };
      const reuse = definition.reuse;
      const id =
        snapshot === undefined || reuse === undefined
          ? undefined
          : sha256(
            JSON.stringify([
              runId,
              snapshotKey(snapshot),
              requirement.mark,
              requirement.version,
              definition.id,
              reuse.version,
              definition.timeoutMs,
              params,
              reuse.includeFinalText === false ? null : finalText,
            ])
          );
      const cached = run.receipts.get(requirement.mark);
      if (id !== undefined && cached?.inputHash === id) {
        results.push({
          requirement,
          verdict: cached.verdict,
          receipt: cached,
          reused: true,
        });
        continue;
      }
      let verdict: VerifierVerdict;
      try {
        const ceiling = definition.timeoutMs ?? DEFAULT_TIMEOUT;
        const limit =
          typeof params.timeout_ms === 'number'
            ? Math.min(params.timeout_ms, ceiling)
            : ceiling;
        const evaluated: Partial<VerifierVerdict> | null | undefined =
          structuredClone(
            await bounded<Partial<VerifierVerdict> | null | undefined>(
              run,
              `check:${requirement.mark}`,
              (child) =>
                definition.run({
                  runId,
                  requirement: structuredClone(requirement),
                  params: structuredClone(params),
                  finalText,
                  signal: child,
                }),
              signal,
              limit
            )
          );
        verdict =
          typeof evaluated?.passed === 'boolean'
            ? (evaluated as VerifierVerdict)
            : {
              passed: false,
              feedback: 'verifier returned no boolean `passed`',
            };
      } catch (error) {
        verdict = {
          passed: false,
          feedback: `Check ${requirement.mark} could not run: ${error instanceof Error ? error.message : String(error)}`,
        };
      }
      const receipt: VerificationReceipt | undefined =
        verdict.passed &&
        id !== undefined &&
        snapshot !== undefined &&
        reuse !== undefined
          ? {
            schemaVersion: 1,
            id: sha256(`${id}:${attempt}`),
            inputHash: id,
            runId,
            mark: requirement.mark,
            requirementVersion: requirement.version as string,
            verifier: definition.id,
            verifierVersion: reuse.version,
            snapshot,
            attempt,
            verdict,
          }
          : undefined;
      results.push({ requirement, verdict, receipt, reused: false });
    }
    if (isAborted(signal))
      return endUnverified(runId, run, 'verification cancelled');
    if (hasPending(run)) {
      for (const { requirement, verdict } of results) {
        if (verdict.passed) continue;
        append(
          {
            runId,
            kind: 'verification_failed',
            actor: 'runtime',
            metadata: { mark: requirement.mark, verifier: requirement.verifier, attempt },
          },
          {
            label: 'verification.verdict',
            content: JSON.stringify(verdict),
          }
        );
      }
      return endUnverified(runId, run, 'owned operations have not settled');
    }
    const after = await capture();
    if (
      run.epoch !== epoch ||
      hasPending(run) ||
      (snapshot !== undefined &&
        after !== undefined &&
        snapshotKey(snapshot) !== snapshotKey(after))
    ) {
      invalidate(runId);
      return endUnverified(
        runId,
        run,
        'workspace or environment changed during verification'
      );
    }
    if (isAborted(signal) || results.length !== requirements.length)
      return endUnverified(runId, run, 'verification interrupted');
    const failures: Evaluation[] = [];
    for (const result of results) {
      const { requirement, verdict, receipt, reused } = result;
      const payload = {
        label: 'verification.verdict',
        content: JSON.stringify(verdict),
      };
      if (receipt !== undefined && !reused) {
        const { verdict: _verdict, ...identity } = receipt;
        append(
          {
            runId,
            kind: 'verification_receipt',
            actor: 'runtime',
            metadata: {
              ...identity,
            },
          },
          payload
        );
        run.receipts.set(requirement.mark, receipt);
      }
      const seq = append(
        {
          runId,
          kind: verdict.passed ? 'mark_admitted' : 'verification_failed',
          actor: 'runtime',
          metadata: {
            mark: requirement.mark,
            verifier: requirement.verifier,
            attempt,
            ...(receipt === undefined ? {} : { receiptId: receipt.id, reused }),
          },
        },
        payload
      );
      if (ledger !== undefined) result.artifactHash = sha256(payload.content);
      if (!verdict.passed) {
        failures.push(result);
        continue;
      }
      run.state.marks.push({
        mark: requirement.mark,
        verifier: requirement.verifier,
        seq,
        attempt,
        ...(verdict.attestation === undefined
          ? {}
          : { attestation: verdict.attestation }),
        ...(receipt === undefined ? {} : { receiptId: receipt.id, reused }),
      });
    }
    run.state.missing = failures.map((failure) => failure.requirement.mark);
    if (snapshot !== undefined) run.state.snapshot = snapshot;
    if (unhealthy())
      return endUnverified(runId, run, 'evidence ledger unavailable');
    if (failures.length === 0) {
      if (isAborted(signal) || run.epoch !== epoch || hasPending(run)) {
        invalidate(runId);
        return endUnverified(runId, run, 'verification interrupted during admission');
      }
      run.sealed = true;
      append({
        runId,
        kind: 'run_completed',
        actor: 'runtime',
        metadata: {
          verified: true,
          attempts: attempt,
          marks: run.state.marks.map((mark) => mark.mark),
          snapshot,
        },
      });
      if (unhealthy())
        return endUnverified(runId, run, 'evidence ledger unavailable');
      if (isAborted(signal) || run.epoch !== epoch || hasPending(run)) {
        invalidate(runId);
        return endUnverified(runId, run, 'verification interrupted during admission');
      }
      run.state.verified = true;
      delete run.state.unverifiedReason;
      return {};
    }
    if (attempt >= maxAttempts)
      return endUnverified(
        runId,
        run,
        `required checks still failing after ${attempt} evaluations`,
        false
      );
    if (input.continuationBudgetRemaining <= 0)
      return endUnverified(runId, run, 'continuation budget exhausted', false);
    run.state.unverifiedReason = 'completion requirements failed';
    const feedback = [
      'The task is not complete yet: these required checks failed.',
      ...failures.map(
        ({ requirement, verdict, artifactHash }) =>
          `${artifactHash === undefined ? '' : `Evidence ${requirement.mark}: ${artifactHash}\n`}- ${requirement.mark}: ${verdict.feedback ?? JSON.stringify(verdict.details ?? {})}`
      ),
      `Fix the cause, then finish again. ${maxAttempts - attempt} more checks may run.`,
    ].join('\n');
    const suffix =
      ledger === undefined
        ? '\n[Feedback truncated; full diagnostics require a configured run ledger.]'
        : '\n[Feedback truncated; inspect verification artifacts in the run ledger.]';
    return {
      decision: 'block',
      reason: 'completion requirements failed',
      additionalContext:
        feedback.length <= feedbackLimit
          ? feedback
          : feedback.slice(0, feedbackLimit - suffix.length) + suffix,
    };
  };
  const hook: CompletionGate['hook'] = async (input, signal) => {
    const run = runFor(input.runId);
    if (run.evaluating) {
      invalidate(input.runId);
      return endUnverified(
        input.runId,
        run,
        'completion evaluation already in progress'
      );
    }
    run.evaluating = true;
    try {
      return await evaluate(input, signal, run);
    } catch (error) {
      return endUnverified(
        input.runId,
        run,
        `verification unavailable: ${String(error)}`
      );
    } finally {
      run.evaluating = false;
    }
  };
  return {
    timeoutMs,
    hook,
    invalidate,
    beginOperation,
    register(registry): () => void {
      return registry.register('StopFinalize', {
        timeout: timeoutMs,
        hooks: [hook],
      });
    },
    getCompletion(runId = latestRun): CompletionState {
      const run = runId === undefined ? undefined : runs.get(runId);
      if (run === undefined) return initial();
      if (unhealthy()) clearCompletion(run, 'evidence ledger unavailable');
      return structuredClone({
        ...run.state,
        pendingOperations: [...run.pending.keys(), ...run.unsettled.values()],
      });
    },
    releaseRun(runId): void {
      const run = runs.get(runId);
      if (run !== undefined && (run.evaluating || hasPending(run)))
        throw new Error('cannot release a run with active work');
      runs.delete(runId);
      if (latestRun === runId) latestRun = undefined;
    },
  };
}

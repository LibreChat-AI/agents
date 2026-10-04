import type { RunnableConfig } from '@langchain/core/runnables';
import type { SubagentResumeExecution } from './SubagentReplay';
import {
  getSubagentResumeManifest,
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY,
} from './SubagentReplay';

const SUBAGENT_THREAD_ID_PREFIX = 'subagent:';
const SUBAGENT_EXECUTION_INVALIDATED_MESSAGE =
  'Subagent execution was invalidated.';

export type SubagentExecutionAddressInput = {
  threadId?: string;
  parentToolCallId: string;
  parentConfigurable?: Record<string, unknown>;
};

export type SubagentExecutionAddress = Readonly<{
  key: string;
  baseChildThreadId: string;
  branchChildThreadId: string;
  currentChildRunId: string;
  explicitResumeAttempt: boolean;
  parentToolCallId: string;
  resumeAttemptId: string;
}>;

export type SubagentExecutionIdentity = Readonly<{
  childRunId: string;
  childThreadId: string;
  approvalExecutionScope: string;
}>;

export interface SubagentExecutionPreparationLease {
  readonly ownsCheckpoint: boolean;
  readonly ownsCreatedCheckpoint: boolean;
  readonly ownsApprovalScope: boolean;
  readonly ownsCreatedApprovalScope: boolean;
  assertOwned(): void;
  markCheckpointCreated(cleanup: () => Promise<void>): void;
  markApprovalScopeCreated(cleanup: () => Promise<void>): void;
  release(): void;
  rollback(
    cause: unknown,
    cleanup?: (lease: SubagentExecutionPreparationLease) => Promise<void>
  ): Promise<never>;
}

export type PreparedSubagentExecutionIdentity = Readonly<{
  identity: SubagentExecutionIdentity;
  lease: SubagentExecutionPreparationLease;
  commit?: (lease: SubagentExecutionPreparationLease) => void;
  rollback?: (lease: SubagentExecutionPreparationLease) => Promise<void>;
}>;

export type SubagentDefinitionBinding = Readonly<{
  subagentType?: string;
  configId?: string;
}>;

export type SubagentInvocationBinding = Readonly<{
  description: string;
  subagentType: string;
  configId?: string;
}>;

export type SubagentSettlementBinding = Readonly<{
  definitionAuthority: 'provisional' | 'effective';
  fingerprint: string;
  invocation: SubagentInvocationBinding;
  subagentType?: string;
  configId?: string;
}>;

export type SubagentExecutionPhase =
  | 'registered'
  | 'active'
  | 'interrupted'
  | 'failed'
  | 'completed'
  | 'invalidated';

export interface ExecutionSnapshot<TResult, TResolvedConfig, TActiveRun> {
  readonly address: SubagentExecutionAddress;
  readonly phase: SubagentExecutionPhase;
  readonly identity?: SubagentExecutionIdentity;
  readonly binding?: SubagentDefinitionBinding;
  readonly invocation?: SubagentInvocationBinding;
  readonly settlement?: SubagentSettlementBinding;
  readonly resumeExecution?: SubagentResumeExecution;
  readonly activeRun?: TActiveRun;
  readonly completedResult?: TResult;
  readonly resolvedConfig?: TResolvedConfig;
  readonly started: boolean;
  readonly completed: boolean;
  readonly resolvingIdentity: boolean;
  readonly resolving: boolean;
  readonly executing: boolean;
  readonly settling: boolean;
  readonly settled: boolean;
}

export type SubagentExecutionRegistryOptions = {
  parentRunId: string;
  parentAgentId?: string;
  durable: boolean;
};

type ResumeSelection = {
  parentToolCallIds?: ReadonlySet<string>;
  config?: RunnableConfig;
};

type OwnedPreparationResource = {
  status: 'owned';
  token: number;
  created: boolean;
  cleanup?: () => Promise<void>;
};

type CleaningPreparationResource = {
  status: 'cleaning';
  token: number;
};

type PreparationResourceState =
  | OwnedPreparationResource
  | CleaningPreparationResource;

function getPreparationResourceKey(
  kind: 'checkpoint' | 'approval',
  identity: string
): string {
  return JSON.stringify([kind, identity]);
}

function attachRollbackError(cause: unknown, rollbackError: unknown): void {
  if (!(cause instanceof Error)) {
    return;
  }
  try {
    Object.defineProperty(cause, 'rollbackError', {
      configurable: true,
      value: rollbackError,
    });
  } catch {
    return;
  }
}

class PreparationLease implements SubagentExecutionPreparationLease {
  constructor(
    private readonly owners: Map<string, PreparationResourceState>,
    private readonly token: number,
    private readonly checkpointKey: string,
    private readonly approvalKey: string
  ) {}

  get ownsCheckpoint(): boolean {
    return this.owns(this.checkpointKey);
  }

  get ownsCreatedCheckpoint(): boolean {
    return this.isCreated(this.checkpointKey);
  }

  get ownsApprovalScope(): boolean {
    return this.owns(this.approvalKey);
  }

  get ownsCreatedApprovalScope(): boolean {
    return this.isCreated(this.approvalKey);
  }

  assertOwned(): void {
    if (!this.owns(this.checkpointKey) || !this.owns(this.approvalKey)) {
      throw new SubagentExecutionInvalidatedError();
    }
  }

  markCheckpointCreated(cleanup: () => Promise<void>): void {
    this.markCreated(this.checkpointKey, cleanup);
  }

  markApprovalScopeCreated(cleanup: () => Promise<void>): void {
    this.markCreated(this.approvalKey, cleanup);
  }

  release(): void {
    this.releaseResource(this.checkpointKey);
    this.releaseResource(this.approvalKey);
  }

  async rollback(
    cause: unknown,
    cleanup?: (lease: SubagentExecutionPreparationLease) => Promise<void>
  ): Promise<never> {
    let rollbackError: unknown;
    let rollbackFailed = false;
    try {
      await cleanup?.(this);
    } catch (error) {
      rollbackError = error;
      rollbackFailed = true;
    }
    const resourceRollback = await this.rollbackOwnedResources();
    if (!rollbackFailed && resourceRollback.failed) {
      rollbackError = resourceRollback.error;
      rollbackFailed = true;
    }
    try {
      if (rollbackFailed) {
        attachRollbackError(cause, rollbackError);
      }
    } finally {
      this.release();
    }
    throw cause;
  }

  private owns(key: string): boolean {
    const owner = this.owners.get(key);
    return owner?.status === 'owned' && owner.token === this.token;
  }

  private isCreated(key: string): boolean {
    const owner = this.owners.get(key);
    return (
      owner?.status === 'owned' && owner.token === this.token && owner.created
    );
  }

  private markCreated(key: string, cleanup: () => Promise<void>): void {
    const owner = this.owners.get(key);
    if (owner?.status !== 'owned' || owner.token !== this.token) {
      throw new SubagentExecutionInvalidatedError();
    }
    owner.created = true;
    owner.cleanup = cleanup;
  }

  private async rollbackOwnedResources(): Promise<{
    failed: boolean;
    error?: unknown;
  }> {
    const cleanups: Array<{
      key: string;
      state: CleaningPreparationResource;
      cleanup: () => Promise<void>;
    }> = [];
    for (const key of [this.checkpointKey, this.approvalKey]) {
      const owner = this.owners.get(key);
      if (owner?.status !== 'owned' || owner.token !== this.token) {
        continue;
      }
      if (!owner.created || owner.cleanup == null) {
        this.owners.delete(key);
        continue;
      }
      const state: CleaningPreparationResource = {
        status: 'cleaning',
        token: this.token,
      };
      this.owners.set(key, state);
      cleanups.push({ key, state, cleanup: owner.cleanup });
    }
    const results = await Promise.all(
      cleanups.map(({ key, state, cleanup }) =>
        this.cleanResource(key, state, cleanup)
      )
    );
    return results.find((result) => result.failed) ?? { failed: false };
  }

  private async cleanResource(
    key: string,
    state: CleaningPreparationResource,
    cleanup: () => Promise<void>
  ): Promise<{ failed: boolean; error?: unknown }> {
    try {
      await cleanup();
      return { failed: false };
    } catch (error) {
      return { failed: true, error };
    } finally {
      if (this.owners.get(key) === state) {
        this.owners.delete(key);
      }
    }
  }

  private releaseResource(key: string): void {
    const owner = this.owners.get(key);
    if (owner?.status === 'owned' && owner.token === this.token) {
      this.owners.delete(key);
    }
  }
}

function getConfigurable(
  config?: RunnableConfig
): Record<string, unknown> | undefined {
  return config?.configurable as Record<string, unknown> | undefined;
}

function getConfigurableString(
  configurable: Record<string, unknown> | undefined,
  key: string
): string | undefined {
  const value = configurable?.[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function encodeThreadIdentity(identity: string[]): string {
  return `${SUBAGENT_THREAD_ID_PREFIX}${Buffer.from(
    JSON.stringify(identity)
  ).toString('base64url')}`;
}

function assertSameIdentity(
  current: SubagentExecutionIdentity,
  next: SubagentExecutionIdentity
): void {
  if (
    current.childRunId === next.childRunId &&
    current.childThreadId === next.childThreadId &&
    current.approvalExecutionScope === next.approvalExecutionScope
  ) {
    return;
  }
  throw new Error('Subagent execution identity cannot change after binding.');
}

function assertCompatibleResumeExecution(
  current: SubagentResumeExecution,
  next: SubagentResumeExecution
): void {
  if (
    current.parentToolCallId === next.parentToolCallId &&
    current.childRunId === next.childRunId &&
    current.subagentType === next.subagentType &&
    current.configId === next.configId
  ) {
    return;
  }
  throw new Error('Subagent resume source cannot change after binding.');
}

function assertSameInvocation(
  current: SubagentInvocationBinding,
  next: SubagentInvocationBinding
): void {
  if (
    current.description === next.description &&
    current.subagentType === next.subagentType &&
    current.configId === next.configId
  ) {
    return;
  }
  throw new SubagentInvocationBindingError();
}

function assertSameSettlement(
  current: SubagentSettlementBinding,
  next: SubagentSettlementBinding
): void {
  if (
    current.definitionAuthority === next.definitionAuthority &&
    current.fingerprint === next.fingerprint &&
    current.subagentType === next.subagentType &&
    current.configId === next.configId
  ) {
    return;
  }
  throw new SubagentSettlementBindingError();
}

function getAbortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new Error('Subagent execution aborted.');
}

export function getSubagentApprovalExecutionScope(
  childRunId: string,
  resumeAttemptId: string
): string {
  return `subagent-approval:${Buffer.from(
    JSON.stringify([childRunId, resumeAttemptId])
  ).toString('base64url')}`;
}

export class SubagentExecutionInvalidatedError extends Error {
  constructor() {
    super(SUBAGENT_EXECUTION_INVALIDATED_MESSAGE);
    this.name = 'SubagentExecutionInvalidatedError';
  }
}

export class SubagentDefinitionBindingError extends Error {
  constructor() {
    super('Subagent effective definition binding cannot change.');
    this.name = 'SubagentDefinitionBindingError';
  }
}

export class SubagentInvocationBindingError extends Error {
  constructor() {
    super('Subagent invocation binding cannot change.');
    this.name = 'SubagentInvocationBindingError';
  }
}

export class SubagentSettlementBindingError extends Error {
  constructor() {
    super('Subagent settlement binding cannot change.');
    this.name = 'SubagentSettlementBindingError';
  }
}

/** In-process source of truth for one Durable Subagent Execution. */
export class SubagentExecutionRecord<
  TResult,
  TResolvedConfig,
  TActiveRun,
  TSettledOutput = never,
> {
  private identityValue?: SubagentExecutionIdentity;
  private attemptedIdentityValue?: SubagentExecutionIdentity;
  private bindingValue?: SubagentDefinitionBinding;
  private bindingAuthority?: 'provisional' | 'effective';
  private invocationValue?: SubagentInvocationBinding;
  private settlementValue?: SubagentSettlementBinding;
  private settledOutputValue?: TSettledOutput;
  private resumeExecutionValue?: SubagentResumeExecution;
  private activeRunValue?: TActiveRun;
  private completedResultValue?: TResult;
  private resolvedConfigValue?: TResolvedConfig;
  private pendingIdentityResolution?: Promise<SubagentExecutionIdentity>;
  private pendingConfigResolution?: Promise<TResolvedConfig>;
  private pendingExecution?: Promise<TResult>;
  private pendingSettlement?: Promise<void>;
  private phaseValue: SubagentExecutionPhase = 'registered';
  private startedValue = false;
  private completedValue = false;
  private settledValue = false;

  constructor(
    readonly address: SubagentExecutionAddress,
    resumeExecution?: SubagentResumeExecution
  ) {
    this.resumeExecutionValue = resumeExecution;
  }

  get snapshot(): ExecutionSnapshot<TResult, TResolvedConfig, TActiveRun> {
    return {
      address: this.address,
      phase: this.phaseValue,
      ...(this.identityValue == null ? {} : { identity: this.identityValue }),
      ...(this.bindingValue == null ? {} : { binding: this.bindingValue }),
      ...(this.invocationValue == null
        ? {}
        : { invocation: this.invocationValue }),
      ...(this.settlementValue == null
        ? {}
        : { settlement: this.settlementValue }),
      ...(this.resumeExecutionValue == null
        ? {}
        : { resumeExecution: this.resumeExecutionValue }),
      ...(this.activeRunValue == null
        ? {}
        : { activeRun: this.activeRunValue }),
      ...(this.completedResultValue == null
        ? {}
        : { completedResult: this.completedResultValue }),
      ...(this.resolvedConfigValue == null
        ? {}
        : { resolvedConfig: this.resolvedConfigValue }),
      started: this.startedValue,
      completed: this.completedValue,
      resolvingIdentity: this.pendingIdentityResolution != null,
      resolving: this.pendingConfigResolution != null,
      executing: this.pendingExecution != null,
      settling: this.pendingSettlement != null,
      settled: this.settledValue,
    };
  }

  get identity(): SubagentExecutionIdentity | undefined {
    return this.identityValue;
  }

  /** Correlation only; never committed identity or resume authority. */
  get attemptedIdentity(): SubagentExecutionIdentity | undefined {
    return this.attemptedIdentityValue;
  }

  recordIdentityAttempt(identity: SubagentExecutionIdentity): void {
    const current = this.attemptedIdentityValue;
    if (
      current != null &&
      current.childRunId === identity.childRunId &&
      current.childThreadId === identity.childThreadId &&
      current.approvalExecutionScope === identity.approvalExecutionScope
    ) {
      return;
    }
    this.attemptedIdentityValue = Object.freeze({ ...identity });
  }

  get binding(): SubagentDefinitionBinding | undefined {
    return this.bindingValue;
  }

  get resumeExecution(): SubagentResumeExecution | undefined {
    return this.resumeExecutionValue;
  }

  get invocation(): SubagentInvocationBinding | undefined {
    return this.invocationValue;
  }

  get activeRun(): TActiveRun | undefined {
    return this.activeRunValue;
  }

  get settlement(): SubagentSettlementBinding | undefined {
    return this.settlementValue;
  }

  get settledOutput(): TSettledOutput | undefined {
    return this.settledOutputValue;
  }

  get completedResult(): TResult | undefined {
    return this.completedResultValue;
  }

  get resolvedConfig(): TResolvedConfig | undefined {
    return this.resolvedConfigValue;
  }

  get started(): boolean {
    return this.startedValue;
  }

  get completed(): boolean {
    return this.completedValue;
  }

  assertUsable(signal?: AbortSignal): void {
    if (signal?.aborted === true) {
      throw getAbortReason(signal);
    }
    if (this.phaseValue === 'invalidated') {
      throw new SubagentExecutionInvalidatedError();
    }
  }

  attachResumeExecution(resumeExecution?: SubagentResumeExecution): void {
    if (resumeExecution == null) {
      return;
    }
    if (this.resumeExecutionValue != null) {
      assertCompatibleResumeExecution(
        this.resumeExecutionValue,
        resumeExecution
      );
      return;
    }
    this.resumeExecutionValue = resumeExecution;
  }

  bindIdentity(identity: SubagentExecutionIdentity): void {
    this.assertUsable();
    if (this.identityValue != null) {
      assertSameIdentity(this.identityValue, identity);
      return;
    }
    this.identityValue = Object.freeze({ ...identity });
  }

  resolveIdentity(
    resolve: () => Promise<PreparedSubagentExecutionIdentity>
  ): Promise<SubagentExecutionIdentity> {
    this.assertUsable();
    if (this.identityValue != null) {
      return Promise.resolve(this.identityValue);
    }
    if (this.pendingIdentityResolution != null) {
      return this.pendingIdentityResolution;
    }
    this.attemptedIdentityValue = undefined;
    const pending = Promise.resolve()
      .then(() => {
        this.assertCurrentIdentityResolution(pending);
        return resolve();
      })
      .then(async (resolution): Promise<SubagentExecutionIdentity> => {
        try {
          this.assertCurrentIdentityResolution(pending);
          resolution.lease.assertOwned();
          resolution.commit?.(resolution.lease);
          this.assertCurrentIdentityResolution(pending);
          resolution.lease.assertOwned();
          this.bindIdentity(resolution.identity);
          const boundIdentity = this.identityValue;
          if (boundIdentity == null) {
            throw new SubagentExecutionInvalidatedError();
          }
          resolution.lease.release();
          return boundIdentity;
        } catch (error) {
          return resolution.lease.rollback(
            error,
            async (lease): Promise<void> => {
              await resolution.rollback?.(lease);
            }
          );
        }
      })
      .finally(() => {
        if (this.pendingIdentityResolution === pending) {
          this.pendingIdentityResolution = undefined;
        }
      });
    this.pendingIdentityResolution = pending;
    return pending;
  }

  bindDefinition(
    binding: SubagentDefinitionBinding,
    authority: 'provisional' | 'effective'
  ): void {
    this.assertUsable();
    if (this.bindingAuthority === 'effective') {
      if (authority === 'provisional') {
        return;
      }
      if (
        this.bindingValue?.subagentType !== binding.subagentType ||
        this.bindingValue?.configId !== binding.configId
      ) {
        throw new SubagentDefinitionBindingError();
      }
      return;
    }
    if (authority === 'provisional' && this.bindingValue != null) {
      return;
    }
    this.bindingValue = Object.freeze({ ...binding });
    this.bindingAuthority = authority;
  }

  resolveConfig(
    resolve: () => Promise<TResolvedConfig>,
    signal: AbortSignal
  ): Promise<TResolvedConfig> {
    this.assertUsable(signal);
    if (this.resolvedConfigValue != null) {
      return Promise.resolve(this.resolvedConfigValue);
    }
    if (this.pendingConfigResolution != null) {
      return this.pendingConfigResolution;
    }
    const pending = Promise.resolve()
      .then(resolve)
      .then((resolvedConfig): TResolvedConfig => {
        this.assertUsable(signal);
        if (this.pendingConfigResolution !== pending) {
          throw new SubagentExecutionInvalidatedError();
        }
        this.resolvedConfigValue = resolvedConfig;
        return resolvedConfig;
      })
      .finally(() => {
        if (this.pendingConfigResolution === pending) {
          this.pendingConfigResolution = undefined;
        }
      });
    this.pendingConfigResolution = pending;
    return pending;
  }

  execute(
    invocation: SubagentInvocationBinding,
    execute: () => Promise<TResult>
  ): Promise<TResult> {
    this.bindDefinition(
      {
        subagentType: invocation.subagentType,
        ...(invocation.configId == null
          ? {}
          : { configId: invocation.configId }),
      },
      'effective'
    );
    this.bindInvocation(invocation);
    if (this.pendingExecution != null) {
      return this.pendingExecution;
    }
    const pending = Promise.resolve()
      .then(execute)
      .finally(() => {
        if (this.pendingExecution === pending) {
          this.pendingExecution = undefined;
        }
      });
    this.pendingExecution = pending;
    return pending;
  }

  settle(
    settlement: SubagentSettlementBinding,
    settledOutput: TSettledOutput,
    persist: () => Promise<void>
  ): Promise<void> {
    this.bindDefinition(
      {
        ...(settlement.subagentType == null
          ? {}
          : { subagentType: settlement.subagentType }),
        ...(settlement.configId == null
          ? {}
          : { configId: settlement.configId }),
      },
      settlement.definitionAuthority
    );
    this.bindInvocation(settlement.invocation);
    if (this.settlementValue != null) {
      assertSameSettlement(this.settlementValue, settlement);
    } else {
      this.settlementValue = Object.freeze({
        ...settlement,
        invocation: Object.freeze({ ...settlement.invocation }),
      });
    }
    if (this.settledValue) {
      return Promise.resolve();
    }
    if (this.pendingSettlement != null) {
      return this.pendingSettlement;
    }
    const pending = Promise.resolve()
      .then(persist)
      .then((): void => {
        this.assertUsable();
        if (this.pendingSettlement !== pending) {
          throw new SubagentExecutionInvalidatedError();
        }
        this.settledValue = true;
        this.settledOutputValue = settledOutput;
        this.activeRunValue = undefined;
        this.completedResultValue = undefined;
        this.resolvedConfigValue = undefined;
      })
      .finally(() => {
        if (this.pendingSettlement === pending) {
          this.pendingSettlement = undefined;
        }
      });
    this.pendingSettlement = pending;
    return pending;
  }

  activate(activeRun: TActiveRun): TActiveRun {
    this.transitionTo('active', [
      'registered',
      'active',
      'interrupted',
      'failed',
    ]);
    if (this.activeRunValue != null) {
      return this.activeRunValue;
    }
    this.activeRunValue = activeRun;
    return activeRun;
  }

  markStarted(): void {
    this.assertUsable();
    if (this.phaseValue === 'completed') {
      throw new Error('Completed subagent execution cannot start again.');
    }
    this.startedValue = true;
  }

  markInterrupted(): void {
    this.transitionTo('interrupted', ['active', 'interrupted']);
  }

  markFailed(): void {
    this.transitionTo('failed', [
      'registered',
      'active',
      'interrupted',
      'failed',
    ]);
    this.activeRunValue = undefined;
    this.resolvedConfigValue = undefined;
  }

  markCompleted(result: TResult): void {
    this.transitionTo('completed', ['active']);
    this.completedValue = true;
    this.completedResultValue = result;
  }

  releaseActiveRun(): TActiveRun | undefined {
    const activeRun = this.activeRunValue;
    this.activeRunValue = undefined;
    return activeRun;
  }

  releaseResolvedConfig(): void {
    this.resolvedConfigValue = undefined;
  }

  invalidate(): void {
    this.phaseValue = 'invalidated';
    this.identityValue = undefined;
    this.bindingValue = undefined;
    this.bindingAuthority = undefined;
    this.invocationValue = undefined;
    this.settlementValue = undefined;
    this.settledOutputValue = undefined;
    this.resumeExecutionValue = undefined;
    this.activeRunValue = undefined;
    this.completedResultValue = undefined;
    this.resolvedConfigValue = undefined;
    this.pendingIdentityResolution = undefined;
    this.pendingConfigResolution = undefined;
    this.pendingExecution = undefined;
    this.pendingSettlement = undefined;
    this.startedValue = false;
    this.completedValue = false;
    this.settledValue = false;
  }

  private bindInvocation(invocation: SubagentInvocationBinding): void {
    if (this.invocationValue != null) {
      assertSameInvocation(this.invocationValue, invocation);
      return;
    }
    this.invocationValue = Object.freeze({ ...invocation });
  }

  private assertCurrentIdentityResolution(
    pending: Promise<SubagentExecutionIdentity>
  ): void {
    this.assertUsable();
    if (this.pendingIdentityResolution !== pending) {
      throw new SubagentExecutionInvalidatedError();
    }
  }

  private transitionTo(
    next: SubagentExecutionPhase,
    allowed: ReadonlyArray<SubagentExecutionPhase>
  ): void {
    this.assertUsable();
    if (!allowed.includes(this.phaseValue)) {
      throw new Error(
        `Cannot transition subagent execution from ${this.phaseValue} to ${next}.`
      );
    }
    this.phaseValue = next;
  }
}

/** Owns canonical addressing, record selection, and invalidation. */
export class SubagentExecutionRegistry<
  TResult,
  TResolvedConfig,
  TActiveRun,
  TSettledOutput = never,
> {
  private readonly recordsByAddress = new Map<
    string,
    SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >
  >();
  private readonly checkpointThreadIds = new Set<string>();
  private readonly preparationResourceOwners = new Map<
    string,
    PreparationResourceState
  >();
  private nextPreparationToken = 0;

  constructor(private readonly options: SubagentExecutionRegistryOptions) {}

  open(
    input: SubagentExecutionAddressInput
  ): SubagentExecutionRecord<
    TResult,
    TResolvedConfig,
    TActiveRun,
    TSettledOutput
  > {
    const address = this.createAddress(input);
    const resumeExecution = getSubagentResumeManifest(
      input.parentConfigurable
    )?.executions.find(
      (execution) => execution.parentToolCallId === input.parentToolCallId
    );
    const current = this.recordsByAddress.get(address.key);
    if (current != null) {
      current.attachResumeExecution(resumeExecution);
      return current;
    }
    const record = new SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >(address, resumeExecution);
    this.recordsByAddress.set(address.key, record);
    return record;
  }

  remove(
    record: SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >
  ): void {
    if (this.recordsByAddress.get(record.address.key) !== record) {
      return;
    }
    this.recordsByAddress.delete(record.address.key);
    record.invalidate();
  }

  beginIdentityPreparation(
    record: SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >,
    identity: SubagentExecutionIdentity
  ): SubagentExecutionPreparationLease {
    record.assertUsable();
    if (this.recordsByAddress.get(record.address.key) !== record) {
      throw new SubagentExecutionInvalidatedError();
    }
    record.recordIdentityAttempt(identity);
    const checkpointKey = getPreparationResourceKey(
      'checkpoint',
      identity.childThreadId
    );
    const approvalKey = getPreparationResourceKey(
      'approval',
      identity.approvalExecutionScope
    );
    const resourceKeys = [checkpointKey, approvalKey];
    const currentResources = resourceKeys.map((key) =>
      this.preparationResourceOwners.get(key)
    );
    if (currentResources.some((resource) => resource != null)) {
      throw new SubagentExecutionInvalidatedError();
    }
    this.nextPreparationToken += 1;
    const token = this.nextPreparationToken;
    for (let i = 0; i < resourceKeys.length; i++) {
      const key = resourceKeys[i];
      this.preparationResourceOwners.set(key, {
        status: 'owned',
        token,
        created: false,
      });
    }
    return new PreparationLease(
      this.preparationResourceOwners,
      token,
      checkpointKey,
      approvalKey
    );
  }

  selectForResume(
    selection: ResumeSelection = {}
  ): SubagentExecutionRecord<
    TResult,
    TResolvedConfig,
    TActiveRun,
    TSettledOutput
  >[] {
    if (selection.config != null) {
      return this.selectScopedRecords(selection);
    }
    const counts = new Map<string, number>();
    const candidates: SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >[] = [];
    for (const record of this.recordsByAddress.values()) {
      if (record.identity == null) {
        continue;
      }
      const parentToolCallId = record.address.parentToolCallId;
      if (
        selection.parentToolCallIds != null &&
        !selection.parentToolCallIds.has(parentToolCallId)
      ) {
        continue;
      }
      candidates.push(record);
      counts.set(parentToolCallId, (counts.get(parentToolCallId) ?? 0) + 1);
    }
    return candidates.filter(
      (record) => counts.get(record.address.parentToolCallId) === 1
    );
  }

  rememberCheckpointThread(...threadIds: string[]): void {
    for (const threadId of threadIds) {
      this.checkpointThreadIds.add(threadId);
    }
  }

  getCheckpointThreadIds(
    getNestedThreadIds?: (activeRun: TActiveRun) => ReadonlyArray<string>
  ): string[] {
    const threadIds = new Set(this.checkpointThreadIds);
    if (getNestedThreadIds == null) {
      return [...threadIds];
    }
    for (const record of this.recordsByAddress.values()) {
      if (record.activeRun == null) {
        continue;
      }
      for (const threadId of getNestedThreadIds(record.activeRun)) {
        threadIds.add(threadId);
      }
    }
    return [...threadIds];
  }

  resetCheckpointThreadIds(): void {
    this.checkpointThreadIds.clear();
  }

  retireResumeSources(
    current: SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >,
    sourceThreadIds: ReadonlySet<string>,
    retire: (
      source: SubagentExecutionRecord<
        TResult,
        TResolvedConfig,
        TActiveRun,
        TSettledOutput
      >
    ) => void
  ): void {
    for (const record of this.recordsByAddress.values()) {
      if (
        record === current ||
        record.identity == null ||
        !sourceThreadIds.has(record.identity.childThreadId)
      ) {
        continue;
      }
      try {
        retire(record);
      } finally {
        this.remove(record);
      }
    }
  }

  clear(
    dispose?: (
      record: SubagentExecutionRecord<
        TResult,
        TResolvedConfig,
        TActiveRun,
        TSettledOutput
      >
    ) => void
  ): void {
    for (const record of this.recordsByAddress.values()) {
      dispose?.(record);
      record.invalidate();
    }
    this.recordsByAddress.clear();
  }

  private selectScopedRecords(
    selection: ResumeSelection
  ): SubagentExecutionRecord<
    TResult,
    TResolvedConfig,
    TActiveRun,
    TSettledOutput
  >[] {
    const parentConfigurable = getConfigurable(selection.config);
    const threadId = getConfigurableString(parentConfigurable, 'thread_id');
    const parentToolCallIds =
      selection.parentToolCallIds ??
      new Set(
        [...this.recordsByAddress.values()].map(
          (record) => record.address.parentToolCallId
        )
      );
    const records: SubagentExecutionRecord<
      TResult,
      TResolvedConfig,
      TActiveRun,
      TSettledOutput
    >[] = [];
    for (const parentToolCallId of parentToolCallIds) {
      const address = this.createAddress({
        threadId,
        parentToolCallId,
        parentConfigurable,
      });
      const record = this.recordsByAddress.get(address.key);
      if (record?.identity != null) {
        records.push(record);
      }
    }
    return records;
  }

  private createAddress(
    input: SubagentExecutionAddressInput
  ): SubagentExecutionAddress {
    const durableParentId = input.threadId ?? this.options.parentRunId;
    const parentFork =
      getConfigurableString(input.parentConfigurable, 'checkpoint_id') ??
      'root';
    const parentBatch =
      getConfigurableString(
        input.parentConfigurable,
        SUBAGENT_PARENT_BATCH_CONFIG_KEY
      ) ?? 'batch';
    const configuredResumeAttemptId = getConfigurableString(
      input.parentConfigurable,
      SUBAGENT_RESUME_ATTEMPT_CONFIG_KEY
    );
    const resumeAttemptId =
      configuredResumeAttemptId ?? this.options.parentRunId;
    const identity = [
      durableParentId,
      parentFork,
      this.options.parentAgentId ?? 'agent',
      input.parentToolCallId,
      parentBatch,
    ];
    const baseChildThreadId = encodeThreadIdentity(identity);
    const branchChildThreadId = this.options.durable
      ? encodeThreadIdentity([...identity, resumeAttemptId])
      : baseChildThreadId;
    return Object.freeze({
      key: branchChildThreadId,
      baseChildThreadId,
      branchChildThreadId,
      currentChildRunId: `${this.options.parentRunId}_sub_${baseChildThreadId.slice(
        SUBAGENT_THREAD_ID_PREFIX.length
      )}`,
      explicitResumeAttempt: configuredResumeAttemptId != null,
      parentToolCallId: input.parentToolCallId,
      resumeAttemptId,
    });
  }
}

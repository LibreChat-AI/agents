import { nanoid } from 'nanoid';
import type {
  InjectedMessage,
  SubagentTaskBoundary,
  SubagentTaskClaim,
  SubagentTaskControlCommand,
  SubagentTaskControlReceipt,
  SubagentTaskControlResult,
  SubagentTaskProgress,
  SubagentTaskRuntime,
  SubagentTaskSnapshot,
  SubagentTaskStartRequest,
  SubagentTaskStartResult,
  SubagentTaskStatus,
  SubagentTaskStore,
  SubagentUpdateEvent,
} from '@/types';

const DEFAULT_COMPLETED_TTL_MS = 60 * 60 * 1000;
const DEFAULT_MAX_ERROR_CHARS = 4 * 1024;
const DEFAULT_MAX_MESSAGE_CHARS = 64 * 1024;
const DEFAULT_TASK_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_MAX_CONTROLS = 32;
const DEFAULT_MAX_CONTROL_RECEIPTS = 64;
const DEFAULT_MAX_RESULT_CHARS = 100_000;
const DEFAULT_MAX_RUNNING_PER_SCOPE = 10;
const DEFAULT_MAX_RUNNING_TOTAL = 100;
const DEFAULT_MAX_TASKS_PER_SCOPE = 200;
const DEFAULT_MAX_TASKS_TOTAL = 2_000;

export interface InMemorySubagentTaskStoreOptions {
  completedTtlMs?: number;
  maxControlMessageChars?: number;
  maxControlsPerTask?: number;
  maxControlReceiptsPerTask?: number;
  maxErrorChars?: number;
  maxResultChars?: number;
  maxRunningPerScope?: number;
  maxRunningTotal?: number;
  maxTasksPerScope?: number;
  maxTasksTotal?: number;
  taskTimeoutMs?: number;
}

type PendingControl = {
  id: string;
  action: 'steer' | 'queue' | 'interrupt';
  message: string;
};

type StoredTask = {
  id: string;
  idempotencyKey: string;
  requestFingerprint?: string;
  scopeId: string;
  threadId: string;
  subagentType: string;
  status: SubagentTaskStatus;
  createdAt: number;
  updatedAt: number;
  controller: AbortController;
  controls: PendingControl[];
  controlReceipts: Map<string, SubagentTaskControlReceipt>;
  progressEvents: number;
  resultClaimed: boolean;
  acceptingControls: boolean;
  result?: string;
  error?: string;
  progress?: SubagentTaskProgress;
  expiry?: ReturnType<typeof setTimeout>;
  timeout?: ReturnType<typeof setTimeout>;
};

type TaskBucket = {
  tasks: Map<string, StoredTask>;
  taskIdByIdempotencyKey: Map<string, string>;
};

type ResolvedOptions = Required<InMemorySubagentTaskStoreOptions>;

function resolvePositiveInteger(
  value: number | undefined,
  fallback: number
): number {
  return Number.isSafeInteger(value) && value != null && value > 0
    ? value
    : fallback;
}

function resolveOptions(
  options: InMemorySubagentTaskStoreOptions
): ResolvedOptions {
  const maxTasksPerScope = resolvePositiveInteger(
    options.maxTasksPerScope,
    DEFAULT_MAX_TASKS_PER_SCOPE
  );
  const maxTasksTotal = resolvePositiveInteger(
    options.maxTasksTotal,
    DEFAULT_MAX_TASKS_TOTAL
  );
  return {
    completedTtlMs: resolvePositiveInteger(
      options.completedTtlMs,
      DEFAULT_COMPLETED_TTL_MS
    ),
    maxControlMessageChars: resolvePositiveInteger(
      options.maxControlMessageChars,
      DEFAULT_MAX_MESSAGE_CHARS
    ),
    maxControlsPerTask: resolvePositiveInteger(
      options.maxControlsPerTask,
      DEFAULT_MAX_CONTROLS
    ),
    maxControlReceiptsPerTask: Math.max(
      resolvePositiveInteger(
        options.maxControlReceiptsPerTask,
        DEFAULT_MAX_CONTROL_RECEIPTS
      ),
      resolvePositiveInteger(options.maxControlsPerTask, DEFAULT_MAX_CONTROLS)
    ),
    maxErrorChars: resolvePositiveInteger(
      options.maxErrorChars,
      DEFAULT_MAX_ERROR_CHARS
    ),
    maxResultChars: resolvePositiveInteger(
      options.maxResultChars,
      DEFAULT_MAX_RESULT_CHARS
    ),
    maxRunningPerScope: Math.min(
      resolvePositiveInteger(
        options.maxRunningPerScope,
        DEFAULT_MAX_RUNNING_PER_SCOPE
      ),
      maxTasksPerScope
    ),
    maxRunningTotal: Math.min(
      resolvePositiveInteger(
        options.maxRunningTotal,
        DEFAULT_MAX_RUNNING_TOTAL
      ),
      maxTasksTotal
    ),
    maxTasksPerScope,
    maxTasksTotal,
    taskTimeoutMs: resolvePositiveInteger(
      options.taskTimeoutMs,
      DEFAULT_TASK_TIMEOUT_MS
    ),
  };
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim() !== '') {
    return error.message;
  }
  return 'Detached subagent task failed.';
}

function truncateMiddle(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  const marker = '\n…[truncated]…\n';
  const available = Math.max(0, maxChars - marker.length);
  const head = Math.ceil(available / 2);
  return `${value.slice(0, head)}${marker}${value.slice(
    value.length - (available - head)
  )}`;
}

function toInjectedMessage(control: PendingControl): InjectedMessage {
  return {
    role: 'user',
    content: control.message,
    source: 'steer',
  };
}

function snapshot(task: StoredTask): SubagentTaskSnapshot {
  return {
    taskId: task.id,
    threadId: task.threadId,
    subagentType: task.subagentType,
    status: task.status,
    createdAt: task.createdAt,
    updatedAt: task.updatedAt,
    resultAvailable:
      task.status === 'completed' && task.result != null && !task.resultClaimed,
    resultClaimed: task.resultClaimed,
    pendingControls: task.controls.length,
    controlReceipts: [...task.controlReceipts.values()].map((receipt) => ({
      ...receipt,
    })),
    ...(task.progress == null ? {} : { progress: { ...task.progress } }),
    ...(task.error == null ? {} : { error: task.error }),
  };
}

function cloneControlReceipt(
  receipt: SubagentTaskControlReceipt
): SubagentTaskControlReceipt {
  return { ...receipt };
}

function abortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error('Detached subagent task cancelled.');
}

/**
 * Bounded process-local task ownership for detached subagents. Terminal tasks
 * keep only a bounded claimable result: the child graph, checkpoint, and full
 * transcript are released. Hosts that need later child-chat continuation may
 * replace this store and persist the canonical messages returned by `run`.
 * This default deliberately makes no restart or cross-replica durability
 * claim.
 */
export class InMemorySubagentTaskStore implements SubagentTaskStore {
  private readonly buckets = new Map<string, TaskBucket>();
  private readonly options: ResolvedOptions;
  private runningTasks = 0;
  private totalTasks = 0;

  constructor(options: InMemorySubagentTaskStoreOptions = {}) {
    this.options = resolveOptions(options);
  }

  /**
   * Payload-free transition seam for hosts that durably project authoritative
   * receipts. Implementations must return synchronously, must not reproduce
   * task-store transition rules, and cannot veto task-store state transitions:
   * hook failures are deliberately isolated by the caller.
   */
  protected onControlReceipt(
    _scopeId: string,
    _taskId: string,
    _receipt: SubagentTaskControlReceipt
  ): void {}

  /**
   * Payload-free lifecycle seam for hosts that own work scheduled by `run`.
   * This reports only that the queued dispatcher did not enter the provider
   * factory; it cannot change task-store state or veto a transition.
   */
  protected onRunSkipped(
    _scopeId: string,
    _taskId: string,
    _reason: 'task_not_running' | 'signal_aborted'
  ): void {}

  private emitControlReceipt(
    task: StoredTask,
    receipt: SubagentTaskControlReceipt
  ): void {
    try {
      this.onControlReceipt(
        task.scopeId,
        task.id,
        cloneControlReceipt(receipt)
      );
    } catch {
      // A host projection is observability only. Task admission, draining, and
      // settlement must remain correct when that projection is unavailable.
    }
  }

  private emitRunSkipped(
    task: StoredTask,
    reason: 'task_not_running' | 'signal_aborted'
  ): void {
    try {
      this.onRunSkipped(task.scopeId, task.id, reason);
    } catch {
      // A host lifecycle projection cannot change task settlement or results.
    }
  }

  start(request: SubagentTaskStartRequest): SubagentTaskStartResult {
    const scopeId = request.scopeId.trim();
    const idempotencyKey = request.idempotencyKey.trim();
    const requestFingerprint = request.requestFingerprint?.trim();
    if (scopeId === '' || idempotencyKey === '') {
      throw new Error('Subagent task scope and idempotency key are required.');
    }
    const now = Date.now();
    const bucket = this.getBucket(scopeId);
    this.sweepBucket(bucket, now);
    const existingId = bucket.taskIdByIdempotencyKey.get(idempotencyKey);
    if (existingId != null) {
      const existing = bucket.tasks.get(existingId);
      if (existing != null) {
        if (
          requestFingerprint != null &&
          requestFingerprint !== '' &&
          existing.requestFingerprint != null &&
          existing.requestFingerprint !== requestFingerprint
        ) {
          return {
            accepted: false,
            reason: 'conflict',
            task: snapshot(existing),
          };
        }
        return { accepted: true, isNew: false, task: snapshot(existing) };
      }
      bucket.taskIdByIdempotencyKey.delete(idempotencyKey);
    }
    let running = 0;
    for (const task of bucket.tasks.values()) {
      if (task.status === 'running') {
        running += 1;
      }
    }
    if (
      running >= this.options.maxRunningPerScope ||
      this.runningTasks >= this.options.maxRunningTotal
    ) {
      this.dropEmptyBucket(scopeId, bucket);
      return { accepted: false, reason: 'capacity' };
    }
    if (!this.makeRoom(bucket) || !this.makeGlobalRoom()) {
      this.dropEmptyBucket(scopeId, bucket);
      return { accepted: false, reason: 'capacity' };
    }
    const taskId = nanoid();
    const requestedThreadId = request.threadId?.trim();
    const task: StoredTask = {
      id: taskId,
      idempotencyKey,
      ...(requestFingerprint == null || requestFingerprint === ''
        ? {}
        : { requestFingerprint }),
      scopeId,
      threadId:
        requestedThreadId != null && requestedThreadId !== ''
          ? requestedThreadId
          : taskId,
      subagentType: request.subagentType,
      status: 'running',
      createdAt: now,
      updatedAt: now,
      controller: new AbortController(),
      controls: [],
      controlReceipts: new Map(),
      progressEvents: 0,
      resultClaimed: false,
      acceptingControls: true,
    };
    this.buckets.set(scopeId, bucket);
    bucket.tasks.set(task.id, task);
    bucket.taskIdByIdempotencyKey.set(idempotencyKey, task.id);
    this.runningTasks += 1;
    this.totalTasks += 1;
    task.timeout = setTimeout(() => {
      this.finishWithError(
        task,
        'error',
        new Error('Detached subagent task timed out.')
      );
    }, this.options.taskTimeoutMs);
    task.timeout.unref();
    const runtime = this.createRuntime(task);
    void Promise.resolve()
      .then(() => {
        if (task.status !== 'running' || task.controller.signal.aborted) {
          this.emitRunSkipped(
            task,
            task.status !== 'running' ? 'task_not_running' : 'signal_aborted'
          );
          return undefined;
        }
        return request.run(runtime);
      })
      .then(
        (result) => {
          if (task.status !== 'running' || result == null) {
            return;
          }
          if (task.controller.signal.aborted) {
            this.finishWithError(
              task,
              'cancelled',
              abortReason(task.controller.signal)
            );
            return;
          }
          task.status = 'completed';
          task.acceptingControls = false;
          this.failPendingControls(task, 'task_completed');
          task.controls.length = 0;
          task.result = truncateMiddle(
            result.content,
            this.options.maxResultChars
          );
          task.updatedAt = Date.now();
          this.runningTasks -= 1;
          this.scheduleExpiry(task);
        },
        (error: unknown) => {
          const status = task.controller.signal.aborted ? 'cancelled' : 'error';
          this.finishWithError(task, status, error);
        }
      );
    return { accepted: true, isNew: true, task: snapshot(task) };
  }

  get(scopeId: string, taskId: string): SubagentTaskSnapshot | undefined {
    const task = this.find(scopeId, taskId);
    return task == null ? undefined : snapshot(task);
  }

  list(scopeId: string): SubagentTaskSnapshot[] {
    const bucket = this.buckets.get(scopeId.trim());
    if (bucket == null) {
      return [];
    }
    this.sweepBucket(bucket, Date.now());
    return [...bucket.tasks.values()]
      .sort((left, right) => left.createdAt - right.createdAt)
      .map(snapshot);
  }

  claim(scopeId: string, taskId: string): SubagentTaskClaim {
    const task = this.find(scopeId, taskId);
    if (task == null) {
      return { status: 'not_found' };
    }
    if (task.status === 'running') {
      return { status: 'running', task: snapshot(task) };
    }
    if (task.resultClaimed) {
      return { status: 'claimed', task: snapshot(task) };
    }
    task.resultClaimed = true;
    task.updatedAt = Date.now();
    const taskSnapshot = snapshot(task);
    this.scheduleExpiry(task);
    if (task.status === 'completed') {
      const result = task.result ?? '';
      task.result = undefined;
      return { status: 'completed', task: taskSnapshot, result };
    }
    const error = task.error ?? 'Detached subagent task did not complete.';
    return { status: task.status, task: taskSnapshot, error };
  }

  control(
    scopeId: string,
    taskId: string,
    command: SubagentTaskControlCommand
  ): SubagentTaskControlResult {
    const task = this.find(scopeId, taskId);
    if (task == null) {
      return { status: 'not_found' };
    }
    if (command.action === 'cancel') {
      if (task.status !== 'running') {
        return { status: 'not_running', task: snapshot(task) };
      }
      this.finishWithError(
        task,
        'cancelled',
        new Error('Detached subagent task cancelled by its parent.')
      );
      return { status: 'cancelled', task: snapshot(task) };
    }
    if (task.status !== 'running' || !task.acceptingControls) {
      return { status: 'not_running', task: snapshot(task) };
    }
    if (command.action === 'cancel_message') {
      const index = task.controls.findIndex(
        (control) => control.id === command.controlId
      );
      if (index < 0) {
        return { status: 'control_not_found', task: snapshot(task) };
      }
      const [control] = task.controls.splice(index, 1);
      task.updatedAt = Date.now();
      this.transitionControl(task, control.id, 'rejected', {
        reason: 'withdrawn',
      });
      return {
        status: 'accepted',
        task: snapshot(task),
        controlId: control.id,
      };
    }
    const message = command.message.trim();
    if (message === '') {
      return { status: 'invalid', message: 'A non-empty message is required.' };
    }
    if (message.length > this.options.maxControlMessageChars) {
      return {
        status: 'invalid',
        message: `Message exceeds ${this.options.maxControlMessageChars} characters.`,
      };
    }
    if (task.controls.length >= this.options.maxControlsPerTask) {
      return {
        status: 'invalid',
        message: `Task already has ${this.options.maxControlsPerTask} pending messages.`,
      };
    }
    if (!this.makeRoomForControlReceipt(task)) {
      return {
        status: 'invalid',
        message: 'Task control receipt capacity is unavailable.',
      };
    }
    const control: PendingControl = {
      id: nanoid(),
      action: command.action,
      message,
    };
    task.controls.push(control);
    task.updatedAt = Date.now();
    const receipt: SubagentTaskControlReceipt = {
      controlId: control.id,
      action: control.action,
      status: 'accepted',
      createdAt: task.updatedAt,
      updatedAt: task.updatedAt,
    };
    task.controlReceipts.set(control.id, receipt);
    this.emitControlReceipt(task, receipt);
    return {
      status: 'accepted',
      task: snapshot(task),
      controlId: control.id,
    };
  }

  private getBucket(scopeId: string): TaskBucket {
    let bucket = this.buckets.get(scopeId);
    if (bucket == null) {
      bucket = {
        tasks: new Map(),
        taskIdByIdempotencyKey: new Map(),
      };
      this.buckets.set(scopeId, bucket);
    }
    return bucket;
  }

  private find(scopeId: string, taskId: string): StoredTask | undefined {
    const bucket = this.buckets.get(scopeId.trim());
    if (bucket == null) {
      return undefined;
    }
    this.sweepBucket(bucket, Date.now());
    return bucket.tasks.get(taskId);
  }

  private makeRoom(bucket: TaskBucket): boolean {
    if (bucket.tasks.size < this.options.maxTasksPerScope) {
      return true;
    }
    const terminal = [...bucket.tasks.values()]
      .filter((task) => task.status !== 'running')
      .sort((left, right) => left.updatedAt - right.updatedAt);
    let removeCount = bucket.tasks.size - this.options.maxTasksPerScope + 1;
    for (const task of terminal) {
      if (removeCount <= 0) {
        break;
      }
      this.removeTask(task);
      removeCount -= 1;
    }
    return bucket.tasks.size < this.options.maxTasksPerScope;
  }

  private makeGlobalRoom(): boolean {
    if (this.totalTasks < this.options.maxTasksTotal) {
      return true;
    }
    const terminal = [...this.buckets.values()]
      .flatMap((bucket) => [...bucket.tasks.values()])
      .filter((task) => task.status !== 'running')
      .sort((left, right) => left.updatedAt - right.updatedAt);
    let removeCount = this.totalTasks - this.options.maxTasksTotal + 1;
    for (const task of terminal) {
      if (removeCount <= 0) {
        break;
      }
      this.removeTask(task);
      removeCount -= 1;
    }
    return this.totalTasks < this.options.maxTasksTotal;
  }

  private sweepBucket(bucket: TaskBucket, now: number): void {
    for (const task of bucket.tasks.values()) {
      if (
        task.status !== 'running' &&
        now - task.updatedAt > this.options.completedTtlMs
      ) {
        this.removeTask(task);
      }
    }
  }

  private removeTask(task: StoredTask): void {
    const bucket = this.buckets.get(task.scopeId);
    if (bucket?.tasks.get(task.id) !== task) {
      return;
    }
    bucket.tasks.delete(task.id);
    this.totalTasks -= 1;
    if (bucket.taskIdByIdempotencyKey.get(task.idempotencyKey) === task.id) {
      bucket.taskIdByIdempotencyKey.delete(task.idempotencyKey);
    }
    if (bucket.tasks.size === 0) {
      this.buckets.delete(task.scopeId);
    }
    this.clearTaskExpiry(task);
    this.clearTaskTimeout(task);
  }

  private dropEmptyBucket(scopeId: string, bucket: TaskBucket): void {
    if (bucket.tasks.size === 0 && this.buckets.get(scopeId) === bucket) {
      this.buckets.delete(scopeId);
    }
  }

  private makeRoomForControlReceipt(task: StoredTask): boolean {
    while (
      task.controlReceipts.size >= this.options.maxControlReceiptsPerTask
    ) {
      const terminal = [...task.controlReceipts].find(
        ([, receipt]) => receipt.status !== 'accepted'
      );
      if (terminal == null) {
        return false;
      }
      task.controlReceipts.delete(terminal[0]);
    }
    return true;
  }

  private transitionControl(
    task: StoredTask,
    controlId: string,
    status: Exclude<SubagentTaskControlReceipt['status'], 'accepted'>,
    detail: Pick<SubagentTaskControlReceipt, 'boundary' | 'reason'> = {}
  ): void {
    const receipt = task.controlReceipts.get(controlId);
    if (receipt == null || receipt.status !== 'accepted') {
      return;
    }
    const transitioned: SubagentTaskControlReceipt = {
      ...receipt,
      status,
      updatedAt: Date.now(),
      ...(detail.boundary == null ? {} : { boundary: detail.boundary }),
      ...(detail.reason == null ? {} : { reason: detail.reason }),
    };
    task.controlReceipts.set(controlId, transitioned);
    this.emitControlReceipt(task, transitioned);
  }

  private failPendingControls(
    task: StoredTask,
    reason: Extract<
      NonNullable<SubagentTaskControlReceipt['reason']>,
      'task_completed' | 'task_cancelled' | 'task_failed'
    >
  ): void {
    for (const control of task.controls) {
      this.transitionControl(
        task,
        control.id,
        reason === 'task_failed' ? 'failed' : 'rejected',
        { reason }
      );
    }
  }

  private scheduleExpiry(task: StoredTask): void {
    this.clearTaskTimeout(task);
    this.clearTaskExpiry(task);
    task.expiry = setTimeout(() => {
      this.removeTask(task);
    }, this.options.completedTtlMs);
    task.expiry.unref();
  }

  private clearTaskExpiry(task: StoredTask): void {
    if (task.expiry == null) {
      return;
    }
    clearTimeout(task.expiry);
    task.expiry = undefined;
  }

  private clearTaskTimeout(task: StoredTask): void {
    if (task.timeout == null) {
      return;
    }
    clearTimeout(task.timeout);
    task.timeout = undefined;
  }

  private finishWithError(
    task: StoredTask,
    status: 'error' | 'cancelled',
    error: unknown
  ): void {
    if (task.status !== 'running') {
      return;
    }
    const message = truncateMiddle(
      toErrorMessage(error),
      this.options.maxErrorChars
    );
    const resolved = new Error(message);
    task.status = status;
    this.runningTasks -= 1;
    task.acceptingControls = false;
    this.failPendingControls(
      task,
      status === 'cancelled' ? 'task_cancelled' : 'task_failed'
    );
    task.controls.length = 0;
    task.error = message;
    task.updatedAt = Date.now();
    this.scheduleExpiry(task);
    task.controller.abort(resolved);
  }

  private createRuntime(task: StoredTask): SubagentTaskRuntime {
    const take = (
      boundary: SubagentTaskBoundary,
      accept: (control: PendingControl) => boolean
    ): InjectedMessage[] => {
      if (task.status !== 'running') {
        return [];
      }
      const selected: PendingControl[] = [];
      const retained: PendingControl[] = [];
      for (const control of task.controls) {
        (accept(control) ? selected : retained).push(control);
      }
      task.controls = retained;
      if (selected.length > 0) {
        task.updatedAt = Date.now();
        for (const control of selected) {
          this.transitionControl(task, control.id, 'applied', { boundary });
        }
      }
      return selected.map(toInjectedMessage);
    };
    return {
      taskId: task.id,
      signal: task.controller.signal,
      shouldPreempt: (): boolean =>
        task.status === 'running' &&
        task.controls.some((control) => control.action === 'interrupt'),
      drain: (boundary: SubagentTaskBoundary): InjectedMessage[] => {
        if (boundary === 'preempt') {
          return take(boundary, (control) => control.action === 'interrupt');
        }
        if (boundary === 'tool') {
          return take(boundary, (control) => control.action !== 'queue');
        }
        return take(boundary, () => true);
      },
      closeTurn: (): { closed: boolean; messages: InjectedMessage[] } => {
        const messages = take('turn', () => true);
        if (messages.length > 0) {
          return { closed: false, messages };
        }
        task.acceptingControls = false;
        return { closed: true, messages: [] };
      },
      reportProgress: (event: SubagentUpdateEvent): void => {
        if (task.status !== 'running') {
          return;
        }
        task.progressEvents += 1;
        task.updatedAt = Date.now();
        task.progress = {
          phase: event.phase,
          at: task.updatedAt,
          eventCount: task.progressEvents,
          ...(event.label == null ? {} : { label: event.label }),
        };
      },
    };
  }
}

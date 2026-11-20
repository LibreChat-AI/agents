import { performance } from 'node:perf_hooks';
import type { ProviderTextAdmission, ProviderTextAdmissionLease } from './providerTextAdmission';
import { PreparedSubagentError } from '@/tools/preparedSubagents';
import { StreamLimitExceededError } from '@/llm/streamLimits';

export const PROVIDER_TEXT_PROTECTION_VERSION = 1;

export type ProviderTextProtectionErrorCode =
  | 'blocked'
  | 'unavailable'
  | 'timeout'
  | 'overflow'
  | 'cancelled'
  | 'unsupported'
  | 'incompatible';

export type ProviderTextProtectionResult =
  | {
      readonly version: 1;
      readonly ok: true;
      readonly value: {
        readonly content: string;
        readonly replacements: number;
        readonly categories: readonly {
          readonly category: string;
          readonly count: number;
        }[];
      };
    }
  | {
      readonly version: 1;
      readonly ok: false;
      readonly error: { readonly code: ProviderTextProtectionErrorCode };
    };

/** Trusted host adapter. Classification must reject code, JSON and ambiguous text. */
export interface ProviderTextProtection {
  readonly version: 1;
  readonly timeoutMs: number;
  readonly maxAttemptBytes: number;
  /** Shared across all attempts using this policy, including child graphs. */
  readonly maxBufferedBytes: number;
  readonly classify: (content: string) => 'prose' | 'unsupported';
  readonly inspect: (input: {
    readonly version: 1;
    readonly content: string;
    readonly target: {
      readonly source: 'message';
      readonly field: 'text';
      readonly provenance: 'model';
    };
    readonly signal: AbortSignal;
  }) => ProviderTextProtectionResult | Promise<ProviderTextProtectionResult>;
}

export class ProviderTextProtectionError extends Error {
  constructor(readonly code: ProviderTextProtectionErrorCode) {
    const safeCode = errorCodes.has(code) ? code : 'incompatible';
    super(`Provider text protection: ${safeCode}`);
    this.code = safeCode;
    this.name = 'ProviderTextProtectionError';
  }
}

const restartCancellations = new WeakSet<ProviderTextProtectionError>();

/** Internal control flow, never a host policy decision or public capability. */
export function createProviderTextRestartCancellation(): ProviderTextProtectionError {
  const error = new ProviderTextProtectionError('cancelled');
  restartCancellations.add(error);
  return error;
}

export function isProviderTextRestartCancellation(error: unknown): boolean {
  return error instanceof ProviderTextProtectionError && restartCancellations.has(error);
}

function protectionAbortError(parent?: AbortSignal): ProviderTextProtectionError | StreamLimitExceededError | PreparedSubagentError {
  const reason: unknown = parent?.reason;
  if (reason instanceof StreamLimitExceededError || reason instanceof PreparedSubagentError) return reason;
  if (reason instanceof ProviderTextProtectionError) {
    return isProviderTextRestartCancellation(reason) ? reason : new ProviderTextProtectionError(reason.code);
  }
  return new ProviderTextProtectionError('cancelled');
}

const protectedSignals = new WeakSet<AbortSignal>();

/** Provider smoothing must stay lazy while the release boundary owns raw text. */
export function isProtectedProviderTextSignal(signal?: AbortSignal): boolean {
  return signal != null && protectedSignals.has(signal);
}

const budgets = new WeakMap<ProviderTextProtection, { bytes: number }>();
const errorCodes = new Set<ProviderTextProtectionErrorCode>([
  'blocked',
  'unavailable',
  'timeout',
  'overflow',
  'cancelled',
  'unsupported',
  'incompatible',
]);

export type ProviderTextProtectionFailure = Extract<ProviderTextProtectionResult, { ok: false }>;

function isProtectionErrorCode(code: unknown): code is ProviderTextProtectionErrorCode {
  return typeof code === 'string' && (errorCodes as ReadonlySet<string>).has(code);
}

/** Safe wire failure, including errors raised by a separate CJS/ESM copy of the SDK. */
export function serializeProviderTextProtectionError(error: unknown): ProviderTextProtectionFailure {
  let code: ProviderTextProtectionErrorCode = 'unavailable';
  try {
    if (error != null && typeof error === 'object' &&
        'name' in error && error.name === 'ProviderTextProtectionError' &&
        'code' in error && isProtectionErrorCode(error.code)) code = error.code;
  } catch {
    // Never serialize arbitrary exception messages, stacks, causes, or getters.
  }
  return { version: 1, ok: false, error: { code } };
}

/** Decode only an authenticated host's failure envelope; unknown versions/codes fail closed. */
export function deserializeProviderTextProtectionError(failure: unknown): ProviderTextProtectionError {
  try {
    if (failure != null && typeof failure === 'object' &&
        'version' in failure && failure.version === 1 &&
        'ok' in failure && failure.ok === false &&
        'error' in failure && failure.error != null && typeof failure.error === 'object' &&
        'code' in failure.error && isProtectionErrorCode(failure.error.code)) {
      return new ProviderTextProtectionError(failure.error.code);
    }
  } catch {
    // Malformed failures cannot become host policy decisions.
  }
  return new ProviderTextProtectionError('incompatible');
}

export function validateProviderTextTimeout(timeoutMs: number): void {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_147_483_647)
    throw new ProviderTextProtectionError('incompatible');
}

export function validateProviderTextProtection(
  policy: ProviderTextProtection
): void {
  const version: number = policy.version;
  if (version !== 1)
    throw new ProviderTextProtectionError('incompatible');
  if (
    typeof policy.inspect !== 'function' ||
    typeof policy.classify !== 'function'
  ) {
    throw new ProviderTextProtectionError('unavailable');
  }
  for (const value of [
    policy.timeoutMs,
    policy.maxAttemptBytes,
    policy.maxBufferedBytes,
  ]) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new ProviderTextProtectionError('incompatible');
    }
  }
  if (
    policy.timeoutMs > 2_147_483_647 ||
    policy.maxAttemptBytes > policy.maxBufferedBytes
  ) {
    throw new ProviderTextProtectionError('incompatible');
  }
}

export interface ProviderTextAttemptOptions {
  readonly timeoutMs?: number;
  readonly admission?: ProviderTextAdmission;
}

/** A lease stays charged while a cancellation-ignoring producer/handler still owns raw text. */
export class ProviderTextAttempt {
  private readonly controller = new AbortController();
  private readonly budget: { bytes: number };
  private readonly deadline: number;
  private bytes = 0;
  private textLength = 0;
  private readonly fragments: string[] = [];
  private readonly pending = new Set<Promise<unknown>>();
  private readonly timer: ReturnType<typeof setTimeout>;
  private readonly abort: () => void;
  private finished = false;
  private readonly admission?: ProviderTextAdmissionLease;

  constructor(
    private readonly policy: ProviderTextProtection,
    private readonly parent?: AbortSignal,
    options: ProviderTextAttemptOptions = {}
  ) {
    validateProviderTextProtection(policy);
    const timeoutMs = options.timeoutMs ?? policy.timeoutMs;
    validateProviderTextTimeout(timeoutMs);
    this.deadline = performance.now() + Math.min(timeoutMs, policy.timeoutMs);
    this.budget = budgets.get(policy) ?? { bytes: 0 };
    budgets.set(policy, this.budget);
    this.abort = (): void =>
      this.controller.abort(protectionAbortError(this.parent));
    if (parent?.aborted === true) throw protectionAbortError(parent);
    this.admission = options.admission?.acquire();
    try {
      this.charge(512);
      parent?.addEventListener('abort', this.abort, { once: true });
      this.check();
      this.timer = setTimeout(() => {
        this.controller.abort(new ProviderTextProtectionError('timeout'));
      }, Math.min(timeoutMs, policy.timeoutMs));
      this.timer.unref();
      protectedSignals.add(this.signal);
    } catch (error) {
      this.budget.bytes -= this.bytes;
      this.bytes = 0;
      this.admission?.release();
      try {
        parent?.removeEventListener('abort', this.abort);
      } catch {
        // Malformed transport signals must not prevent admission rollback.
      }
      if (error instanceof ProviderTextProtectionError || error instanceof StreamLimitExceededError || error instanceof PreparedSubagentError) throw error;
      throw new ProviderTextProtectionError('incompatible');
    }
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  check(): void {
    if (this.finished) throw new ProviderTextProtectionError('incompatible');
    if (this.signal.aborted) throw this.signal.reason;
    if (this.parent?.aborted === true)
      throw protectionAbortError(this.parent);
    if (performance.now() >= this.deadline) {
      this.controller.abort(new ProviderTextProtectionError('timeout'));
      throw this.signal.reason;
    }
  }

  private charge(bytes: number): void {
    if (
      this.bytes + bytes > this.policy.maxAttemptBytes ||
      this.budget.bytes + bytes > this.policy.maxBufferedBytes
    ) {
      this.controller.abort(new ProviderTextProtectionError('overflow'));
      throw this.signal.reason;
    }
    try {
      this.admission?.charge(bytes);
    } catch (error) {
      this.controller.abort(error);
      throw error;
    }
    this.bytes += bytes;
    this.budget.bytes += bytes;
  }

  append(text: string, cumulativeReplay = false): void {
    this.check();
    if (cumulativeReplay && this.hasCandidatePrefix(text)) text = text.slice(this.textLength);
    if (!text) return;
    const bytes = text.length * 2;
    this.charge(bytes * 2 + 64);
    this.fragments.push(text);
    this.textLength += text.length;
  }

  /** Compare buffered fragments without allocating an uncharged joined copy. */
  private hasCandidatePrefix(text: string): boolean {
    if (this.textLength === 0 || text.length < this.textLength) return false;
    let offset = 0;
    for (const fragment of this.fragments) {
      if (!text.startsWith(fragment, offset)) return false;
      offset += fragment.length;
    }
    return true;
  }

  matchesText(text: string): boolean {
    return text.length === this.textLength && (text.length === 0 || this.hasCandidatePrefix(text));
  }

  retain(bytes: number): void {
    this.check();
    this.charge(bytes);
  }

  observeChunk(): void {
    this.check();
    this.charge(128);
  }

  async wait<T>(work: Promise<T>): Promise<T> {
    this.pending.add(work);
    return new Promise<T>((resolve, reject) => {
      const abort = (): void => {
        this.signal.removeEventListener('abort', abort);
        reject(this.signal.reason);
      };
      this.signal.addEventListener('abort', abort, { once: true });
      if (this.signal.aborted) abort();
      const settle = (): void => {
        this.signal.removeEventListener('abort', abort);
        this.pending.delete(work);
      };
      void work.then(
        (value) => { settle(); resolve(value); },
        (error: Error) => { settle(); reject(error); }
      );
    });
  }

  async release(): Promise<string> {
    this.check();
    const content = this.fragments.join('');
    if (!content) return '';
    let result: ProviderTextProtectionResult;
    try {
      const treatment = this.policy.classify(content);
      this.check();
      if (treatment !== 'prose') {
        throw new ProviderTextProtectionError('unsupported');
      }
      result = await this.wait(
        Promise.resolve(
          this.policy.inspect({
            version: 1,
            content,
            target: { source: 'message', field: 'text', provenance: 'model' },
            signal: this.signal,
          })
        )
      );
    } catch (error) {
      this.check();
      if (isProviderTextRestartCancellation(error)) throw error;
      throw deserializeProviderTextProtectionError(serializeProviderTextProtectionError(error));
    }
    this.check();
    const canonical = validateResult(result);
    this.check();
    this.charge(canonical.length * 2);
    return canonical;
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    clearTimeout(this.timer);
    try {
      this.parent?.removeEventListener('abort', this.abort);
    } catch {
      // Cleanup must still drain leases if a malformed transport signal throws.
    }
    const dispose = (): void => {
      this.fragments.length = 0;
      this.budget.bytes -= this.bytes;
      this.bytes = 0;
      this.admission?.release();
    };
    if (this.pending.size === 0) dispose();
    else {
      this.admission?.retain();
      void Promise.allSettled([...this.pending]).then(dispose);
    }
  }
}

interface UncheckedResult {
  version?: number;
  ok?: boolean;
  value?: { content?: string; replacements?: number; categories?: readonly { category?: string; count?: number }[] };
  error?: { code?: ProviderTextProtectionErrorCode };
}

function validateResult(result: UncheckedResult | undefined): string {
  try {
    if (result?.version !== 1 || typeof result.ok !== 'boolean') {
      throw new ProviderTextProtectionError('incompatible');
    }
    if (!result.ok) {
      const code = result.error?.code;
      throw new ProviderTextProtectionError(code != null && errorCodes.has(code) ? code : 'incompatible');
    }
    const value = result.value;
    if (typeof value?.content !== 'string' || typeof value.replacements !== 'number' ||
        !Number.isSafeInteger(value.replacements) || value.replacements < 0 ||
        !Array.isArray(value.categories) || value.categories.length > 32 ||
        value.categories.some((entry) => typeof entry.category !== 'string' ||
          typeof entry.count !== 'number' || !Number.isSafeInteger(entry.count) || entry.count <= 0)) {
      throw new ProviderTextProtectionError('incompatible');
    }
    return value.content;
  } catch (error) {
    if (error instanceof ProviderTextProtectionError) throw error;
    throw new ProviderTextProtectionError('incompatible');
  }
}

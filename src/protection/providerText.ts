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
    super(`Provider text protection: ${code}`);
    this.name = 'ProviderTextProtectionError';
  }
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

/** A lease stays charged while a cancellation-ignoring producer/handler still owns raw text. */
export class ProviderTextAttempt {
  private readonly controller = new AbortController();
  private readonly budget: { bytes: number };
  private bytes = 0;
  private readonly fragments: string[] = [];
  private readonly pending = new Set<Promise<unknown>>();
  private readonly timer: ReturnType<typeof setTimeout>;
  private readonly abort: () => void;
  private readonly stopped: Promise<never>;
  private finished = false;

  constructor(
    private readonly policy: ProviderTextProtection,
    private readonly parent?: AbortSignal
  ) {
    validateProviderTextProtection(policy);
    this.budget = budgets.get(policy) ?? { bytes: 0 };
    budgets.set(policy, this.budget);
    if (parent?.aborted === true) throw new ProviderTextProtectionError('cancelled');
    this.charge(512);
    this.abort = (): void =>
      this.controller.abort(new ProviderTextProtectionError('cancelled'));
    parent?.addEventListener('abort', this.abort, { once: true });
    this.stopped = new Promise<never>((_resolve, reject) => {
      const rejectAbort = (): void => reject(this.controller.signal.reason);
      this.controller.signal.addEventListener('abort', rejectAbort, {
        once: true,
      });
      if (this.controller.signal.aborted) rejectAbort();
    });
    void this.stopped.catch(() => {});
    this.timer = setTimeout(() => {
      this.controller.abort(new ProviderTextProtectionError('timeout'));
    }, policy.timeoutMs);
    this.timer.unref();
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  check(): void {
    if (this.signal.aborted) throw this.signal.reason;
    if (this.parent?.aborted === true)
      throw new ProviderTextProtectionError('cancelled');
  }

  private charge(bytes: number): void {
    if (
      this.bytes + bytes > this.policy.maxAttemptBytes ||
      this.budget.bytes + bytes > this.policy.maxBufferedBytes
    ) {
      this.controller.abort(new ProviderTextProtectionError('overflow'));
      throw this.signal.reason;
    }
    this.bytes += bytes;
    this.budget.bytes += bytes;
  }

  append(text: string): void {
    this.check();
    if (!text) return;
    const bytes = text.length * 2;
    this.charge(bytes * 2 + 128);
    this.fragments.push(text);
  }

  async wait<T>(work: Promise<T>): Promise<T> {
    this.pending.add(work);
    void work.then(
      () => this.pending.delete(work),
      () => this.pending.delete(work)
    );
    return Promise.race([work, this.stopped]);
  }

  async release(): Promise<string> {
    this.check();
    const content = this.fragments.join('');
    if (!content) return '';
    let result: ProviderTextProtectionResult;
    try {
      if (this.policy.classify(content) !== 'prose') {
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
      if (error instanceof ProviderTextProtectionError) throw error;
      throw new ProviderTextProtectionError('unavailable');
    }
    this.check();
    const canonical = validateResult(result);
    this.charge(canonical.length * 2);
    return canonical;
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    clearTimeout(this.timer);
    this.parent?.removeEventListener('abort', this.abort);
    const dispose = (): void => {
      this.fragments.length = 0;
      this.budget.bytes -= this.bytes;
      this.bytes = 0;
    };
    if (this.pending.size === 0) dispose();
    else void Promise.allSettled([...this.pending]).then(dispose);
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

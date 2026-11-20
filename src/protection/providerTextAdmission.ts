import { ProviderTextProtectionError } from './providerText';

export interface ProviderTextAdmissionLimits {
  readonly maxConcurrentInspections: number;
  readonly maxBufferedBytes: number;
}

export interface ProviderTextAdmissionSnapshot {
  readonly inspections: number;
  readonly bufferedBytes: number;
  /** Finished callers whose inspectors have not settled. Included in the totals above. */
  readonly retainedInspections: number;
  readonly retainedBytes: number;
}

/** Internal ownership lease; only the attempt that acquires it may release it. */
export interface ProviderTextAdmissionLease {
  charge(bytes: number): void;
  retain(): void;
  release(): void;
}

/** Host-owned admission scope. Reuse across bindings/reconnects belonging to the same quota. */
export class ProviderTextAdmission {
  private readonly limits: ProviderTextAdmissionLimits;
  private inspections = 0;
  private bufferedBytes = 0;
  private retainedInspections = 0;
  private retainedBytes = 0;

  constructor(limits: ProviderTextAdmissionLimits) {
    for (const value of [
      limits.maxConcurrentInspections,
      limits.maxBufferedBytes,
    ]) {
      if (!Number.isSafeInteger(value) || value <= 0)
        throw new ProviderTextProtectionError('incompatible');
    }
    this.limits = Object.freeze({ ...limits });
  }

  snapshot(): Readonly<ProviderTextAdmissionSnapshot> {
    return Object.freeze({
      inspections: this.inspections,
      bufferedBytes: this.bufferedBytes,
      retainedInspections: this.retainedInspections,
      retainedBytes: this.retainedBytes,
    });
  }

  /** @internal Acquired by ProviderTextAttempt, never by a transport caller. */
  acquire(): ProviderTextAdmissionLease {
    if (this.inspections >= this.limits.maxConcurrentInspections)
      throw new ProviderTextProtectionError('overflow');
    this.inspections++;
    let bytes = 0;
    let retained = false;
    let released = false;
    return {
      charge: (amount): void => {
        if (released || !Number.isSafeInteger(amount) || amount < 0)
          throw new ProviderTextProtectionError('incompatible');
        if (amount > this.limits.maxBufferedBytes - this.bufferedBytes)
          throw new ProviderTextProtectionError('overflow');
        this.bufferedBytes += amount;
        bytes += amount;
        if (retained) this.retainedBytes += amount;
      },
      retain: (): void => {
        if (retained || released) return;
        retained = true;
        this.retainedInspections++;
        this.retainedBytes += bytes;
      },
      release: (): void => {
        if (released) return;
        released = true;
        this.inspections--;
        this.bufferedBytes -= bytes;
        if (!retained) return;
        this.retainedInspections--;
        this.retainedBytes -= bytes;
      },
    };
  }
}

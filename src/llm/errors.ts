import type { UsageMetadata } from '@langchain/core/messages';
import type { NativeMediaProviderOutcome } from '@/types/nativeMedia';

/** Provider consumption survives a failed persistence or cancellation outcome. */
export class UsageBearingError extends Error {
  constructor(
    cause: Error,
    readonly usage?: UsageMetadata
  ) {
    super(cause.message, { cause });
    this.name = cause.name;
  }
}

export class NativeMediaError extends UsageBearingError {
  constructor(
    cause: Error,
    usage?: UsageMetadata,
    readonly providerOutcome?: NativeMediaProviderOutcome
  ) {
    super(cause, usage);
  }
}

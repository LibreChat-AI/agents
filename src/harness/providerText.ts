import { z } from 'zod';
import type { ProviderTextAdmission } from '@/protection/providerTextAdmission';
import type { ProviderTextProtection } from '@/protection/providerText';
import type { ExecutionIdentity } from './identity';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  validateProviderTextProtection,
  validateProviderTextTimeout,
} from '@/protection/providerText';
import {
  executionIdentitySchema,
  snapshotExecutionIdentity,
  matchesExecutionIdentity,
} from './identity';

/** Host-owned limits, negotiated before executing a graph. Contains no credentials or callbacks. */
export type HostProviderTextDescriptor = Pick<
  ProviderTextProtection,
  'version' | 'timeoutMs' | 'maxAttemptBytes' | 'maxBufferedBytes'
>;

export interface HostProviderTextRequest {
  readonly version: 1;
  readonly identity: Readonly<ExecutionIdentity>;
  readonly content: string;
}

/** Transport implementations authenticate callers and validate incoming data before dispatch. */
export interface HostProviderTextService {
  readonly version: 1;
  inspect(
    request: HostProviderTextRequest,
    signal?: AbortSignal
  ): Promise<string>;
}

const requestSchema = z.object({
  version: z.literal(1),
  content: z.string(),
  identity: executionIdentitySchema,
});

function boundIdentity(
  source: Readonly<ExecutionIdentity>
): Readonly<ExecutionIdentity> {
  try {
    return snapshotExecutionIdentity(source);
  } catch {
    throw new ProviderTextProtectionError('incompatible');
  }
}

/** Classification, inspection, deadlines and shared raw-text budgets remain authoritative on the host. */
export function createHostProviderTextService(options: {
  identity: Readonly<ExecutionIdentity>;
  policy: ProviderTextProtection;
  /** Reuse a host-owned scope for all bindings sharing a tenant or runtime quota. */
  admission?: ProviderTextAdmission;
  /** Host inspection deadline; cannot extend the policy's model-attempt timeout. */
  inspectionTimeoutMs?: number;
}): {
  descriptor: HostProviderTextDescriptor;
  service: HostProviderTextService;
} {
  const identity = boundIdentity(options.identity);
  const policy = options.policy;
  validateProviderTextProtection(policy);
  const { admission, inspectionTimeoutMs } = options;
  if (inspectionTimeoutMs !== undefined)
    validateProviderTextTimeout(inspectionTimeoutMs);
  return {
    descriptor: Object.freeze({
      version: 1,
      timeoutMs: policy.timeoutMs,
      maxAttemptBytes: policy.maxAttemptBytes,
      maxBufferedBytes: policy.maxBufferedBytes,
    }),
    service: {
      version: 1,
      inspect: async (request, signal): Promise<string> => {
        const parsed = requestSchema.safeParse(request);
        if (
          !parsed.success ||
          !matchesExecutionIdentity(parsed.data.identity, identity)
        )
          throw new ProviderTextProtectionError('incompatible');
        const attempt = new ProviderTextAttempt(policy, signal, {
          admission,
          timeoutMs: inspectionTimeoutMs,
        });
        try {
          attempt.append(parsed.data.content);
          return await attempt.release();
        } finally {
          attempt.finish();
        }
      },
    },
  };
}

/**
 * Installs the SDK's existing release boundary around a transport-neutral host service.
 * The SDK release boundary buffers each attempt before one host request.
 * Classification and inspection statistics belong to the host. The adapter returns only canonical text.
 */
export function createHostProviderTextProtection(options: {
  identity: Readonly<ExecutionIdentity>;
  descriptor: HostProviderTextDescriptor;
  service: HostProviderTextService;
}): ProviderTextProtection {
  const identity = boundIdentity(options.identity);
  const service = options.service;
  const serviceVersion: number = service.version;
  const descriptorVersion: number = options.descriptor.version;
  if (serviceVersion !== 1 || descriptorVersion !== 1)
    throw new ProviderTextProtectionError('incompatible');
  const policy: ProviderTextProtection = {
    version: 1,
    timeoutMs: options.descriptor.timeoutMs,
    maxAttemptBytes: options.descriptor.maxAttemptBytes,
    maxBufferedBytes: options.descriptor.maxBufferedBytes,
    // The synchronous SDK classification gate defers to authoritative host classification.
    classify: () => 'prose',
    inspect: async ({ content, signal }) => {
      const canonical = await service.inspect(
        { version: 1, identity, content },
        signal
      );
      return {
        version: 1,
        ok: true,
        value: { content: canonical, replacements: 0, categories: [] },
      };
    },
  };
  validateProviderTextProtection(policy);
  return Object.freeze(policy);
}

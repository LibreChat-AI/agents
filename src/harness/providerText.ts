import type { ProviderTextProtection } from '@/protection/providerText';
import type { ExecutionIdentity } from './index';
import {
  ProviderTextAttempt,
  ProviderTextProtectionError,
  validateProviderTextProtection,
} from '@/protection/providerText';

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

const identityKeys = [
  'runtimeId',
  'agentId',
  'runId',
  'userId',
  'tenantId',
  'conversationId',
] as const;

function boundIdentity(
  source: Readonly<ExecutionIdentity>
): Readonly<ExecutionIdentity> {
  for (const key of ['runtimeId', 'agentId', 'runId', 'userId'] as const) {
    if (typeof source[key] !== 'string' || !source[key].trim())
      throw new ProviderTextProtectionError('incompatible');
  }
  return Object.freeze({
    runtimeId: source.runtimeId,
    agentId: source.agentId,
    runId: source.runId,
    userId: source.userId,
    tenantId: source.tenantId,
    conversationId: source.conversationId,
  });
}

/** Classification, inspection, deadlines and shared raw-text budgets remain authoritative on the host. */
export function createHostProviderTextService(options: {
  identity: Readonly<ExecutionIdentity>;
  policy: ProviderTextProtection;
}): {
  descriptor: HostProviderTextDescriptor;
  service: HostProviderTextService;
} {
  const identity = boundIdentity(options.identity);
  const policy = options.policy;
  validateProviderTextProtection(policy);
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
        const version: number = request.version;
        if (
          version !== 1 ||
          typeof request.content !== 'string' ||
          identityKeys.some((key) => request.identity[key] !== identity[key])
        )
          throw new ProviderTextProtectionError('incompatible');
        const attempt = new ProviderTextAttempt(policy, signal);
        try {
          attempt.append(request.content);
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
      let canonical: string;
      try {
        canonical = await service.inspect(
          { version: 1, identity, content },
          signal
        );
      } catch (error) {
        if (error instanceof ProviderTextProtectionError) throw error;
        throw new ProviderTextProtectionError('unavailable');
      }
      signal.throwIfAborted();
      if (typeof canonical !== 'string')
        throw new ProviderTextProtectionError('incompatible');
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

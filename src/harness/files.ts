import { z } from 'zod';
import { executionIdentitySchema } from './identity';

const reference = z
  .string()
  .min(1)
  .max(512)
  .refine((value) => value.trim().length > 0);
const count = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const digest = z.string().regex(/^[a-f0-9]{64}$/);

/** Opaque host references and immutable revisions, never filesystem paths or signed URLs. */
export const hostFileDescriptorSchema = z.object({
  reference,
  revision: reference,
  filename: z.string().min(1).max(255),
  mediaType: z.string().min(1).max(255),
  bytes: count,
  sha256: digest.optional(),
});
export type HostFileDescriptor = Readonly<
  z.infer<typeof hostFileDescriptorSchema>
>;

export const hostFileServiceDescriptorSchema = z.object({
  version: z.literal(1),
  operations: z.array(z.enum(['read', 'stage', 'publish'])).max(3),
  maxFileBytes: count.positive(),
  maxChunkBytes: count.positive(),
});
export type HostFileServiceDescriptor = Readonly<
  Omit<z.infer<typeof hostFileServiceDescriptorSchema>, 'operations'>
> & { readonly operations: readonly ('read' | 'stage' | 'publish')[] };

const request = z.object({
  version: z.literal(1),
  identity: executionIdentitySchema,
});
export const hostFileReadRequestSchema = request.extend({
  reference,
  revision: reference,
  offset: count,
  length: count.positive(),
});
export const hostFileStageRequestSchema = request.extend({
  operationId: reference,
  filename: z.string().min(1).max(255),
  mediaType: z.string().min(1).max(255),
  bytes: count,
  sha256: digest.optional(),
});
export const hostFilePublishRequestSchema = request.extend({
  operationId: reference,
  reference,
  revision: reference,
});
export type HostFileReadRequest = Readonly<
  z.infer<typeof hostFileReadRequestSchema>
>;
export type HostFileStageRequest = Readonly<
  z.infer<typeof hostFileStageRequestSchema>
>;
export type HostFilePublishRequest = Readonly<
  z.infer<typeof hostFilePublishRequestSchema>
>;
export interface HostFileChunk {
  readonly reference: string;
  readonly revision: string;
  readonly offset: number;
  readonly data: Uint8Array;
}
export interface HostFilePublication {
  readonly operationId: string;
  readonly file: HostFileDescriptor;
}

/**
 * Authenticate the transport separately. Each operation must authorize the bound identity,
 * reference and revision against canonical host storage. Staging is private; only publish
 * enters the host's native attachment/event pipeline. Mutations are never automatically retried.
 */
export interface HostFileService {
  readonly version: 1;
  read(
    request: HostFileReadRequest,
    signal?: AbortSignal
  ): Promise<HostFileChunk>;
  stage(
    request: HostFileStageRequest,
    content: AsyncIterable<Uint8Array>,
    signal?: AbortSignal
  ): Promise<HostFileDescriptor>;
  publish(
    request: HostFilePublishRequest,
    signal?: AbortSignal
  ): Promise<HostFilePublication>;
}

const codes = [
  'incompatible',
  'forbidden',
  'conflict',
  'overflow',
  'cancelled',
  'timeout',
  'unavailable',
] as const;
export type HostFileErrorCode = (typeof codes)[number];
export class HostFileServiceError extends Error {
  readonly code: HostFileErrorCode;
  constructor(code: HostFileErrorCode) {
    const safe = codes.includes(code) ? code : 'incompatible';
    super(`Host file service: ${safe}`);
    this.name = 'HostFileServiceError';
    this.code = safe;
  }
}

/** @internal Never forward adapter error text or abort reasons over the service boundary. */
export function normalizeHostFileError(
  error: unknown,
  signal?: AbortSignal
): HostFileServiceError {
  try {
    if (signal?.aborted === true) {
      return new HostFileServiceError(
        signal.reason instanceof HostFileServiceError &&
        signal.reason.code === 'timeout'
          ? 'timeout'
          : 'cancelled'
      );
    }
    if (error instanceof HostFileServiceError)
      return new HostFileServiceError(error.code);
  } catch {
    /** Adapter errors may contain throwing accessors; never expose them. */
  }
  return new HostFileServiceError('unavailable');
}

/** @internal Snapshot only public metadata, dropping host-specific fields. */
export function checkedHostFile(value: HostFileDescriptor): HostFileDescriptor {
  const parsed = hostFileDescriptorSchema.safeParse(value);
  if (!parsed.success) throw new HostFileServiceError('incompatible');
  return Object.freeze(parsed.data);
}

/** @internal Reject mismatched/stale ranges before bytes enter a graph. */
export function checkedHostFileChunk(
  value: HostFileChunk,
  input: HostFileReadRequest
): HostFileChunk {
  if (
    value.reference !== input.reference ||
    value.revision !== input.revision ||
    value.offset !== input.offset ||
    !(value.data instanceof Uint8Array) ||
    value.data.byteLength !== input.length
  )
    throw new HostFileServiceError('incompatible');
  return {
    reference: input.reference,
    revision: input.revision,
    offset: input.offset,
    data: Uint8Array.from(value.data),
  };
}

/** @internal Negotiation validates limits before constructing any graph adapter. */
export function checkedHostFileService(
  value: HostFileServiceDescriptor
): HostFileServiceDescriptor {
  const parsed = hostFileServiceDescriptorSchema.safeParse(value);
  if (
    !parsed.success ||
    parsed.data.maxChunkBytes > parsed.data.maxFileBytes ||
    parsed.data.maxChunkBytes > Number.MAX_SAFE_INTEGER / 2 ||
    new Set(parsed.data.operations).size !== parsed.data.operations.length
  )
    throw new HostFileServiceError('incompatible');
  return Object.freeze({
    ...parsed.data,
    operations: Object.freeze([...parsed.data.operations]),
  });
}

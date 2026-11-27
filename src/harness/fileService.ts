import type {
  HostFileService,
  HostFileChunk,
  HostFileDescriptor,
  HostFilePublication,
  HostFileServiceDescriptor,
  HostFileReadRequest,
  HostFileStageRequest,
  HostFilePublishRequest,
} from './files';
import type { ExecutionIdentity } from './identity';
import {
  HostFileServiceError,
  normalizeHostFileError,
  checkedHostFile,
  checkedHostFileChunk,
  checkedHostFileService,
  hostFileReadRequestSchema,
  hostFileStageRequestSchema,
  hostFilePublishRequestSchema,
} from './files';
import {
  snapshotExecutionIdentity,
  matchesExecutionIdentity,
} from './identity';
import { createFileUpload } from './fileStream';

function positive(value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new HostFileServiceError('incompatible');
}

/** Share across bindings in the host's quota scope. Hung storage retains its slot and byte reservation. */
export class HostFileAdmission {
  private operations = 0;
  private bytes = 0;
  private readonly maxConcurrentOperations: number;
  private readonly maxBufferedBytes: number;
  constructor(limits: {
    maxConcurrentOperations: number;
    maxBufferedBytes: number;
  }) {
    positive(limits.maxConcurrentOperations);
    positive(limits.maxBufferedBytes);
    this.maxConcurrentOperations = limits.maxConcurrentOperations;
    this.maxBufferedBytes = limits.maxBufferedBytes;
  }
  snapshot(): Readonly<{ operations: number; reservedBytes: number }> {
    return Object.freeze({
      operations: this.operations,
      reservedBytes: this.bytes,
    });
  }
  /** @internal Reservations are released only when storage and input cleanup settle. */
  acquire(bytes: number): () => void {
    if (!Number.isSafeInteger(bytes) || bytes < 0)
      throw new HostFileServiceError('incompatible');
    if (
      this.operations >= this.maxConcurrentOperations ||
      bytes > this.maxBufferedBytes - this.bytes
    )
      throw new HostFileServiceError('overflow');
    this.operations++;
    this.bytes += bytes;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.operations--;
      this.bytes -= bytes;
    };
  }
}

/**
 * The host adapter owns canonical ACL/revision checks on every operation, durable operation-ID
 * reconciliation, private staging/cleanup, and publication through the native file pipeline.
 * A request identity is asserted against the authenticated binding before any adapter call.
 */
export type HostFileStore = Omit<HostFileService, 'version'>;

export function createHostFileService(options: {
  identity: Readonly<ExecutionIdentity>;
  descriptor: HostFileServiceDescriptor;
  store: HostFileStore;
  admission: HostFileAdmission;
  timeoutMs: number;
  /** Lifetime of the authenticated execution, independent of individual RPC cancellation. */
  signal?: AbortSignal;
}): { descriptor: HostFileServiceDescriptor; service: HostFileService } {
  const identity = snapshotExecutionIdentity(options.identity);
  const descriptor = checkedHostFileService(options.descriptor);
  const { store, admission, timeoutMs, signal: runSignal } = options;
  positive(timeoutMs);
  if (timeoutMs > 2147483647) throw new HostFileServiceError('incompatible');

  function authorize(
    operation: HostFileServiceDescriptor['operations'][number],
    value: ExecutionIdentity
  ): void {
    if (
      !descriptor.operations.includes(operation) ||
      !matchesExecutionIdentity(identity, value)
    )
      throw new HostFileServiceError('forbidden');
  }

  async function execute<T>(
    bytes: number,
    signal: AbortSignal | undefined,
    work: (signal: AbortSignal) => Promise<T>
  ): Promise<T> {
    const parent =
      runSignal && signal
        ? AbortSignal.any([runSignal, signal])
        : (runSignal ?? signal);
    if (parent?.aborted === true) throw new HostFileServiceError('cancelled');
    const release = admission.acquire(bytes);
    const controller = new AbortController();
    const cancel = (): void =>
      controller.abort(new HostFileServiceError('cancelled'));
    parent?.addEventListener('abort', cancel, { once: true });
    const timer = setTimeout(
      () => controller.abort(new HostFileServiceError('timeout')),
      timeoutMs
    );
    timer.unref();
    let onAbort = (): void => undefined;
    const aborted = new Promise<never>((_resolve, reject) => {
      onAbort = (): void =>
        reject(normalizeHostFileError(undefined, controller.signal));
      controller.signal.addEventListener('abort', onAbort, { once: true });
    });
    const pending = Promise.resolve()
      .then(async () => {
        controller.signal.throwIfAborted();
        return work(controller.signal);
      })
      .finally(release);
    try {
      const result = await Promise.race([pending, aborted]);
      controller.signal.throwIfAborted();
      return result;
    } catch (error) {
      throw normalizeHostFileError(error, controller.signal);
    } finally {
      clearTimeout(timer);
      parent?.removeEventListener('abort', cancel);
      controller.signal.removeEventListener('abort', onAbort);
      controller.abort();
    }
  }

  return {
    descriptor,
    service: {
      version: 1,
      read: async (input, signal): Promise<HostFileChunk> => {
        const parsed = hostFileReadRequestSchema.safeParse(input);
        if (!parsed.success) throw new HostFileServiceError('incompatible');
        const request: HostFileReadRequest = { ...parsed.data, identity };
        authorize('read', parsed.data.identity);
        if (
          request.length > descriptor.maxChunkBytes ||
          request.offset > descriptor.maxFileBytes - request.length
        )
          throw new HostFileServiceError('overflow');
        return execute(request.length * 2, signal, async (active) =>
          checkedHostFileChunk(await store.read(request, active), request)
        );
      },
      stage: async (input, content, signal): Promise<HostFileDescriptor> => {
        const parsed = hostFileStageRequestSchema.safeParse(input);
        if (!parsed.success) throw new HostFileServiceError('incompatible');
        authorize('stage', parsed.data.identity);
        const request: HostFileStageRequest = { ...parsed.data, identity };
        if (request.bytes > descriptor.maxFileBytes)
          throw new HostFileServiceError('overflow');
        return execute(descriptor.maxChunkBytes * 2, signal, async (active) => {
          const upload = createFileUpload(
            content,
            request,
            descriptor.maxChunkBytes,
            active,
            true
          );
          try {
            const file = checkedHostFile(
              await store.stage(request, upload.content, active)
            );
            const sha256 = upload.finish();
            if (file.bytes !== request.bytes || file.sha256 !== sha256)
              throw new HostFileServiceError('incompatible');
            return file;
          } finally {
            await upload.close();
          }
        });
      },
      publish: async (input, signal): Promise<HostFilePublication> => {
        const parsed = hostFilePublishRequestSchema.safeParse(input);
        if (!parsed.success) throw new HostFileServiceError('incompatible');
        authorize('publish', parsed.data.identity);
        const request: HostFilePublishRequest = { ...parsed.data, identity };
        return execute(0, signal, async (active) => {
          const result = await store.publish(request, active);
          if (result.operationId !== request.operationId)
            throw new HostFileServiceError('incompatible');
          const file = checkedHostFile(result.file);
          if (file.bytes > descriptor.maxFileBytes)
            throw new HostFileServiceError('overflow');
          return { operationId: request.operationId, file };
        });
      },
    },
  };
}

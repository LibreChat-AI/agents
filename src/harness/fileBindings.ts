import type {
  HostFileDescriptor,
  HostFileStageRequest,
  HostFileReadRequest,
  HostFilePublication,
  HostFileServiceDescriptor,
  HostFileService,
} from './files';
import type { ExecutionIdentity } from './identity';
import {
  HostFileServiceError,
  normalizeHostFileError,
  checkedHostFile,
  checkedHostFileChunk,
  checkedHostFileService,
  hostFileStageRequestSchema,
  hostFilePublishRequestSchema,
} from './files';
import { snapshotExecutionIdentity } from './identity';
import { createFileUpload } from './fileStream';

export interface HostFileBindings {
  read(
    file: HostFileDescriptor,
    signal?: AbortSignal
  ): AsyncIterable<Uint8Array>;
  stage(
    artifact: Omit<HostFileStageRequest, 'version' | 'identity'>,
    content: AsyncIterable<Uint8Array>,
    signal?: AbortSignal
  ): Promise<HostFileDescriptor>;
  publish(
    file: HostFileDescriptor,
    operationId: string,
    signal?: AbortSignal
  ): Promise<HostFilePublication>;
}

/** Pull-based graph adapter; at most one range or upload chunk is requested at a time. */
export function createHostFileBindings(options: {
  identity: Readonly<ExecutionIdentity>;
  descriptor: HostFileServiceDescriptor;
  service: HostFileService;
}): HostFileBindings {
  const identity = snapshotExecutionIdentity(options.identity);
  const descriptor = checkedHostFileService(options.descriptor);
  const service = options.service;
  if (Number(service.version) !== 1)
    throw new HostFileServiceError('incompatible');
  function allow(
    operation: HostFileServiceDescriptor['operations'][number],
    signal?: AbortSignal
  ): void {
    if (!descriptor.operations.includes(operation))
      throw new HostFileServiceError('forbidden');
    if (signal?.aborted === true) throw new HostFileServiceError('cancelled');
  }
  return {
    read: async function* (source, signal): AsyncGenerator<Uint8Array> {
      try {
        allow('read', signal);
        const file = checkedHostFile(source);
        if (file.bytes > descriptor.maxFileBytes)
          throw new HostFileServiceError('overflow');
        for (let offset = 0; offset < file.bytes; ) {
          allow('read', signal);
          const input: HostFileReadRequest = {
            version: 1,
            identity,
            reference: file.reference,
            revision: file.revision,
            offset,
            length: Math.min(descriptor.maxChunkBytes, file.bytes - offset),
          };
          const chunk = checkedHostFileChunk(
            await service.read(input, signal),
            input
          );
          allow('read', signal);
          offset += chunk.data.byteLength;
          yield chunk.data;
        }
      } catch (error) {
        throw normalizeHostFileError(error, signal);
      }
    },
    stage: async (artifact, content, signal): Promise<HostFileDescriptor> => {
      try {
        allow('stage', signal);
        const parsed = hostFileStageRequestSchema.safeParse({
          ...artifact,
          version: 1,
          identity,
        });
        if (!parsed.success) throw new HostFileServiceError('incompatible');
        if (parsed.data.bytes > descriptor.maxFileBytes)
          throw new HostFileServiceError('overflow');
        const upload = createFileUpload(
          content,
          parsed.data,
          descriptor.maxChunkBytes,
          signal
        );
        try {
          const file = checkedHostFile(
            await service.stage(parsed.data, upload.content, signal)
          );
          upload.finish();
          allow('stage', signal);
          if (
            file.sha256 === undefined ||
            file.bytes !== parsed.data.bytes ||
            (parsed.data.sha256 !== undefined &&
              file.sha256 !== parsed.data.sha256)
          )
            throw new HostFileServiceError('incompatible');
          return file;
        } finally {
          /** The host retains admission until source cleanup settles. Do not delay
           * a cancellation/timeout response on an uncooperative producer here. */
          void upload.close().catch(() => undefined);
        }
      } catch (error) {
        throw normalizeHostFileError(error, signal);
      }
    },
    publish: async (
      source,
      operationId,
      signal
    ): Promise<HostFilePublication> => {
      try {
        allow('publish', signal);
        const file = checkedHostFile(source);
        const parsed = hostFilePublishRequestSchema.safeParse({
          version: 1,
          identity,
          operationId,
          reference: file.reference,
          revision: file.revision,
        });
        if (!parsed.success) throw new HostFileServiceError('incompatible');
        const result = await service.publish(parsed.data, signal);
        allow('publish', signal);
        if (result.operationId !== operationId)
          throw new HostFileServiceError('incompatible');
        const published = checkedHostFile(result.file);
        if (published.bytes > descriptor.maxFileBytes)
          throw new HostFileServiceError('overflow');
        return { operationId, file: published };
      } catch (error) {
        throw normalizeHostFileError(error, signal);
      }
    },
  };
}

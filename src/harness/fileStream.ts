import { createHash } from 'node:crypto';
import type { HostFileStageRequest } from './files';
import { HostFileServiceError } from './files';

/** Single-consumer, bounded upload. Cleanup is owned by the operation, not the transport. */
export function createFileUpload(
  source: AsyncIterable<Uint8Array>,
  request: Pick<HostFileStageRequest, 'bytes' | 'sha256'>,
  maxChunkBytes: number,
  signal?: AbortSignal,
  hash = false
): {
  content: AsyncIterable<Uint8Array>;
  finish: () => string | undefined;
  close: () => Promise<void>;
} {
  const iterator = source[Symbol.asyncIterator]();
  let closing: Promise<void> | undefined;
  let started = false;
  let complete = false;
  let bytes = 0;
  const checksum = hash ? createHash('sha256') : undefined;
  let sha256: string | undefined;
  const close = (): Promise<void> => {
    closing ??= Promise.resolve().then(async () => {
      await iterator.return?.();
    });
    return closing;
  };
  async function* stream(): AsyncGenerator<Uint8Array> {
    try {
      for (;;) {
        signal?.throwIfAborted();
        const next = await iterator.next();
        signal?.throwIfAborted();
        if (next.done === true) break;
        if (!(next.value instanceof Uint8Array) || next.value.byteLength === 0)
          throw new HostFileServiceError('incompatible');
        if (
          next.value.byteLength > maxChunkBytes ||
          next.value.byteLength > request.bytes - bytes
        )
          throw new HostFileServiceError('overflow');
        const chunk = Uint8Array.from(next.value);
        bytes += chunk.byteLength;
        checksum?.update(chunk);
        yield chunk;
      }
      if (bytes !== request.bytes)
        throw new HostFileServiceError('incompatible');
      sha256 = checksum?.digest('hex');
      if (
        sha256 !== undefined &&
        request.sha256 !== undefined &&
        sha256 !== request.sha256
      )
        throw new HostFileServiceError('conflict');
      complete = true;
    } finally {
      await close();
    }
  }
  return {
    content: {
      [Symbol.asyncIterator](): AsyncIterator<Uint8Array> {
        if (started || closing) throw new HostFileServiceError('incompatible');
        started = true;
        const generator = stream();
        return {
          next: (): Promise<IteratorResult<Uint8Array>> =>
            closing
              ? closing.then(() => ({ done: true, value: undefined }))
              : generator.next(),
          return: async (): Promise<IteratorResult<Uint8Array>> => {
            try {
              return await generator.return(undefined);
            } finally {
              await close();
            }
          },
          throw: async (
            error: unknown
          ): Promise<IteratorResult<Uint8Array>> => {
            try {
              return await generator.throw(error);
            } finally {
              await close();
            }
          },
        };
      },
    },
    finish: (): string | undefined => {
      if (!complete) throw new HostFileServiceError('incompatible');
      return sha256;
    },
    close,
  };
}

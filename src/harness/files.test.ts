import { createHash } from 'node:crypto';
import type {
  HostFileDescriptor,
  HostFileServiceDescriptor,
  HostFileStageRequest,
} from './files';
import type { HostFileStore } from './fileService';
import { createHostFileService, HostFileAdmission } from './fileService';
import { createHostFileBindings } from './fileBindings';
import { HostFileServiceError } from './files';

const identity = {
  runtimeId: 'runtime',
  agentId: 'agent',
  runId: 'run',
  userId: 'user',
  tenantId: 'tenant',
  conversationId: 'chat',
};
const descriptor: HostFileServiceDescriptor = {
  version: 1,
  operations: ['read', 'stage', 'publish'],
  maxFileBytes: 1024,
  maxChunkBytes: 4,
};
const digest = (value: string) =>
  createHash('sha256').update(value).digest('hex');
const file: HostFileDescriptor = {
  reference: 'private-ref',
  revision: 'v1',
  filename: 'report.txt',
  mediaType: 'text/plain',
  bytes: 6,
  sha256: digest('abcdef'),
};
const stage: HostFileStageRequest = {
  version: 1,
  identity,
  operationId: 'upload-1',
  filename: file.filename,
  mediaType: file.mediaType,
  bytes: file.bytes,
};
async function* chunks(...values: string[]): AsyncGenerator<Uint8Array> {
  for (const value of values) yield Buffer.from(value);
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function fixture(overrides: Partial<HostFileStore> = {}, timeoutMs = 1000) {
  const received: string[] = [];
  const store: HostFileStore = {
    read: async (request) => ({
      ...request,
      data: Buffer.from('abcdef').subarray(
        request.offset,
        request.offset + request.length
      ),
    }),
    stage: async (_request, content) => {
      for await (const chunk of content)
        received.push(Buffer.from(chunk).toString());
      return file;
    },
    publish: async (request) => ({
      operationId: request.operationId,
      file: { ...file, reference: 'published-ref' },
    }),
    ...overrides,
  };
  const admission = new HostFileAdmission({
    maxConcurrentOperations: 1,
    maxBufferedBytes: 8,
  });
  const host = createHostFileService({
    identity,
    descriptor,
    admission,
    store,
    timeoutMs,
  });
  return {
    ...host,
    store,
    received,
    admission,
    bindings: createHostFileBindings({ identity, ...host }),
  };
}

it('reads lazily in bounded ranges and snapshots identity/metadata', async () => {
  const { service, store } = fixture();
  const read = jest.spyOn(store, 'read');
  const mutable = { ...identity };
  const bindings = createHostFileBindings({
    identity: mutable,
    descriptor,
    service,
  });
  mutable.userId = 'attacker';
  const source = { ...file };
  const iterator = bindings.read(source)[Symbol.asyncIterator]();
  expect(read).not.toHaveBeenCalled();
  expect(Buffer.from((await iterator.next()).value!).toString()).toBe('abcd');
  source.revision = 'mutated';
  expect(read).toHaveBeenCalledTimes(1);
  expect(Buffer.from((await iterator.next()).value!).toString()).toBe('ef');
  expect((await iterator.next()).done).toBe(true);
  expect(
    read.mock.calls.map(([request]) => [
      request.offset,
      request.length,
      request.revision,
      request.identity.userId,
    ])
  ).toEqual([
    [0, 4, 'v1', 'user'],
    [4, 2, 'v1', 'user'],
  ]);
});

it('stages privately before a separate publication request', async () => {
  const { bindings, store, received, admission } = fixture();
  const publish = jest.spyOn(store, 'publish');
  const staged = await bindings.stage(stage, chunks('abcd', 'ef'));
  expect(received).toEqual(['abcd', 'ef']);
  expect(publish).not.toHaveBeenCalled();
  expect(await bindings.publish(staged, 'publish-1')).toEqual({
    operationId: 'publish-1',
    file: { ...file, reference: 'published-ref' },
  });
  expect(admission.snapshot()).toEqual({ operations: 0, reservedBytes: 0 });
});

it.each(Object.keys(identity) as Array<keyof typeof identity>)(
  'rejects foreign %s before storage access',
  async (key) => {
    const { service, store, admission } = fixture();
    const read = jest.spyOn(store, 'read');
    const upload = jest.spyOn(store, 'stage');
    const publish = jest.spyOn(store, 'publish');
    const foreign = { ...identity, [key]: 'foreign' };
    await expect(
      service.read({
        version: 1,
        identity: foreign,
        reference: file.reference,
        revision: file.revision,
        offset: 0,
        length: 4,
      })
    ).rejects.toMatchObject({ code: 'forbidden' });
    await expect(
      service.stage({ ...stage, identity: foreign }, chunks('abcd', 'ef'))
    ).rejects.toMatchObject({ code: 'forbidden' });
    await expect(
      service.publish({
        version: 1,
        identity: foreign,
        operationId: 'p',
        reference: file.reference,
        revision: file.revision,
      })
    ).rejects.toMatchObject({ code: 'forbidden' });
    expect(read).not.toHaveBeenCalled();
    expect(upload).not.toHaveBeenCalled();
    expect(publish).not.toHaveBeenCalled();
    expect(admission.snapshot().operations).toBe(0);
  }
);

it('preserves opaque reference spelling and strips host-only metadata', async () => {
  const source = {
    ...file,
    reference: ' private ',
    path: '/private/secret',
    signedUrl: 'secret',
  };
  const { service } = fixture({
    stage: async (_request, content) => {
      for await (const _chunk of content) {
        /* Consume completely. */
      }
      return source;
    },
  });
  const result = await service.stage(stage, chunks('abcd', 'ef'));
  expect(result).toEqual({ ...file, reference: ' private ' });
  expect(Object.isFrozen(result)).toBe(true);
});

it.each([
  ['oversized chunk', ['abcdef'], 6, undefined, 'overflow'],
  ['short upload', ['abcd'], 6, undefined, 'incompatible'],
  ['long upload', ['abcd', 'efg'], 6, undefined, 'overflow'],
  ['empty chunk', [''], 0, undefined, 'incompatible'],
  ['wrong checksum', ['abcd', 'ef'], 6, digest('wrong'), 'conflict'],
] as const)(
  'rejects %s and closes the source',
  async (_name, values, bytes, sha256, code) => {
    const { service, admission } = fixture();
    let closed = false;
    async function* source() {
      try {
        yield* chunks(...values);
      } finally {
        closed = true;
      }
    }
    await expect(
      service.stage({ ...stage, bytes, sha256 }, source())
    ).rejects.toMatchObject({ code });
    expect(closed).toBe(true);
    expect(admission.snapshot().operations).toBe(0);
  }
);

it('accepts an empty artifact with the canonical empty digest', async () => {
  const { service } = fixture({
    stage: async (_request, content) => {
      for await (const _chunk of content) throw new Error('unexpected chunk');
      return { ...file, bytes: 0, sha256: digest('') };
    },
  });
  expect((await service.stage({ ...stage, bytes: 0 }, chunks())).bytes).toBe(0);
});

it('rejects storage that acknowledges before consuming and verifying the upload', async () => {
  const { service } = fixture({ stage: async () => file });
  await expect(
    service.stage(stage, chunks('abcd', 'ef'))
  ).rejects.toMatchObject({ code: 'incompatible' });
});

it.each(['reference', 'revision', 'offset', 'data'] as const)(
  'rejects a mismatched read %s',
  async (key) => {
    const { bindings } = fixture({
      read: async (request) => {
        const value = { ...request, data: Buffer.from('abcd') };
        if (key === 'offset') return { ...value, offset: 1 };
        if (key === 'data') return { ...value, data: Buffer.from('a') };
        return { ...value, [key]: 'foreign' };
      },
    });
    await expect(
      bindings.read(file)[Symbol.asyncIterator]().next()
    ).rejects.toMatchObject({ code: 'incompatible' });
  }
);

it('enforces per-operation capabilities and numeric bounds before storage access', async () => {
  const { store, admission } = fixture();
  const read = jest.spyOn(store, 'read');
  const host = createHostFileService({
    identity,
    descriptor: { ...descriptor, operations: ['read'] },
    store,
    admission,
    timeoutMs: 1000,
  });
  await expect(
    host.service.stage(stage, chunks('abcd', 'ef'))
  ).rejects.toMatchObject({ code: 'forbidden' });
  for (const [offset, length] of [
    [-1, 4],
    [0, 0],
    [0, 5],
    [1023, 4],
    [Number.MAX_SAFE_INTEGER, 1],
  ]) {
    await expect(
      host.service.read({
        version: 1,
        identity,
        reference: file.reference,
        revision: file.revision,
        offset,
        length,
      })
    ).rejects.toBeInstanceOf(HostFileServiceError);
  }
  expect(read).not.toHaveBeenCalled();
});

it('keeps a shared quota charged after cancellation until storage drains', async () => {
  const started = deferred<void>();
  const finish = deferred<void>();
  const { service, admission } = fixture({
    read: async (request) => {
      started.resolve();
      await finish.promise;
      return { ...request, data: Buffer.from('abcd') };
    },
  });
  const controller = new AbortController();
  const request = {
    version: 1 as const,
    identity,
    reference: file.reference,
    revision: file.revision,
    offset: 0,
    length: 4,
  };
  const pending = service.read(request, controller.signal);
  await started.promise;
  controller.abort(new Error('private abort reason'));
  await expect(pending).rejects.toMatchObject({
    code: 'cancelled',
    message: 'Host file service: cancelled',
  });
  expect(admission.snapshot()).toEqual({ operations: 1, reservedBytes: 8 });
  await expect(service.read(request)).rejects.toMatchObject({
    code: 'overflow',
  });
  finish.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(admission.snapshot()).toEqual({ operations: 0, reservedBytes: 0 });
});

it('retains admission through asynchronous upload-source cleanup', async () => {
  const closing = deferred<void>();
  const finish = deferred<void>();
  const { service, admission } = fixture({}, 20);
  async function* source() {
    try {
      yield* chunks('abcd', 'ef');
    } finally {
      closing.resolve();
      await finish.promise;
    }
  }
  const pending = service.stage(stage, source());
  await closing.promise;
  await expect(pending).rejects.toMatchObject({ code: 'timeout' });
  expect(admission.snapshot().operations).toBe(1);
  finish.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(admission.snapshot().operations).toBe(0);
});

it('does not retry publication after an ambiguous failure and sanitizes storage errors', async () => {
  let commits = 0;
  const { bindings } = fixture({
    publish: async () => {
      commits++;
      throw new Error('/private/path credential-secret');
    },
  });
  await expect(bindings.publish(file, 'publish-1')).rejects.toMatchObject({
    code: 'unavailable',
    message: 'Host file service: unavailable',
  });
  expect(commits).toBe(1);
});

it('honors the host execution lifetime even when the caller supplies no signal', async () => {
  const { store, admission } = fixture();
  const controller = new AbortController();
  controller.abort();
  const host = createHostFileService({
    identity,
    descriptor,
    store,
    admission,
    timeoutMs: 1000,
    signal: controller.signal,
  });
  const read = jest.spyOn(store, 'read');
  await expect(
    host.service.read({
      version: 1,
      identity,
      reference: file.reference,
      revision: file.revision,
      offset: 0,
      length: 4,
    })
  ).rejects.toMatchObject({ code: 'cancelled' });
  expect(read).not.toHaveBeenCalled();
});

it('negotiates versions and snapshots the operation allowlist', () => {
  const { service } = fixture();
  const operations = ['read'] as Array<'read' | 'publish'>;
  const publicDescriptor = { ...descriptor, operations };
  const binding = createHostFileBindings({
    identity,
    descriptor: publicDescriptor,
    service,
  });
  operations.push('publish');
  return expect(binding.publish(file, 'p')).rejects.toMatchObject({
    code: 'forbidden',
  });
});

it.each([0, -1, Infinity, NaN, 1.5])('rejects invalid limit %s', (value) => {
  expect(
    () =>
      new HostFileAdmission({
        maxConcurrentOperations: value,
        maxBufferedBytes: 8,
      })
  ).toThrow(HostFileServiceError);
  const { store, admission } = fixture();
  expect(() =>
    createHostFileService({
      identity,
      descriptor: { ...descriptor, maxChunkBytes: value },
      store,
      admission,
      timeoutMs: 1000,
    })
  ).toThrow(HostFileServiceError);
});

it('returns binding cancellation promptly while the host retains upload cleanup', async () => {
  const entered = deferred<void>();
  const finish = deferred<void>();
  const { bindings, admission } = fixture();
  async function* source() {
    entered.resolve();
    await finish.promise;
    yield* chunks('abcd', 'ef');
  }
  const controller = new AbortController();
  const pending = bindings.stage(stage, source(), controller.signal);
  await entered.promise;
  controller.abort();
  await expect(pending).rejects.toMatchObject({ code: 'cancelled' });
  expect(admission.snapshot().operations).toBe(1);
  finish.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(admission.snapshot().operations).toBe(0);
});

it('sanitizes malformed adapter errors without invoking their messages', async () => {
  const error = new HostFileServiceError('unavailable');
  Object.defineProperty(error, 'code', {
    get() {
      throw new Error('private credential');
    },
  });
  const { bindings } = fixture({
    publish: async () => {
      throw error;
    },
  });
  await expect(bindings.publish(file, 'p')).rejects.toMatchObject({
    code: 'unavailable',
    message: 'Host file service: unavailable',
  });
});

it('rejects unsupported protocol versions before consuming upload bytes', async () => {
  const { service } = fixture();
  const incompatible = JSON.parse(
    JSON.stringify({ ...stage, version: 2 })
  ) as HostFileStageRequest;
  let consumed = false;
  async function* content() {
    consumed = true;
    yield Buffer.from('abcd');
  }
  await expect(service.stage(incompatible, content())).rejects.toMatchObject({
    code: 'incompatible',
  });
  expect(consumed).toBe(false);
  expect(() =>
    createHostFileBindings({
      identity,
      descriptor: JSON.parse(
        JSON.stringify({ ...descriptor, version: 2 })
      ) as HostFileServiceDescriptor,
      service,
    })
  ).toThrow(HostFileServiceError);
});

it('charges shared byte capacity across independent authenticated bindings', async () => {
  const entered = deferred<void>();
  const finish = deferred<void>();
  const { store } = fixture({
    read: async (request) => {
      entered.resolve();
      await finish.promise;
      return { ...request, data: Buffer.from('abcd') };
    },
  });
  const admission = new HostFileAdmission({
    maxConcurrentOperations: 4,
    maxBufferedBytes: 8,
  });
  const options = { descriptor, store, admission, timeoutMs: 1000 };
  const first = createHostFileService({ ...options, identity });
  const other = { ...identity, runId: 'other' };
  const second = createHostFileService({ ...options, identity: other });
  const input = {
    version: 1 as const,
    identity,
    reference: file.reference,
    revision: file.revision,
    offset: 0,
    length: 4,
  };
  const pending = first.service.read(input);
  await entered.promise;
  await expect(
    second.service.read({ ...input, identity: other })
  ).rejects.toMatchObject({ code: 'overflow' });
  finish.resolve();
  await pending;
  expect(admission.snapshot().operations).toBe(0);
});

it('rejects a remote stage receipt without its verified digest', async () => {
  const { service } = fixture();
  const remote = {
    ...service,
    stage: async (
      _request: HostFileStageRequest,
      content: AsyncIterable<Uint8Array>
    ) => {
      for await (const _chunk of content) {
        /* Consume the upload. */
      }
      return { ...file, sha256: undefined };
    },
  };
  const bindings = createHostFileBindings({
    identity,
    descriptor,
    service: remote,
  });
  await expect(
    bindings.stage(stage, chunks('abcd', 'ef'))
  ).rejects.toMatchObject({ code: 'incompatible' });
});

it('closes an opened upload source even if storage never pulls it', async () => {
  const close = jest.fn(async () => ({
    done: true as const,
    value: undefined,
  }));
  const next = jest.fn(async () => ({
    done: false as const,
    value: Buffer.from('abcd'),
  }));
  const source = { [Symbol.asyncIterator]: () => ({ next, return: close }) };
  const { service } = fixture({ stage: async () => file });
  await expect(service.stage(stage, source)).rejects.toMatchObject({
    code: 'incompatible',
  });
  expect(next).not.toHaveBeenCalled();
  expect(close).toHaveBeenCalledTimes(1);
});

it('retains host admission when storage rejects an unstarted binding upload', async () => {
  const closing = deferred<void>();
  const finish = deferred<void>();
  const close = jest.fn(async () => {
    closing.resolve();
    await finish.promise;
    return { done: true as const, value: undefined };
  });
  const next = jest.fn(async () => ({
    done: false as const,
    value: Buffer.from('abcd'),
  }));
  const content = { [Symbol.asyncIterator]: () => ({ next, return: close }) };
  const { bindings, admission } = fixture(
    {
      stage: async () => {
        throw new HostFileServiceError('overflow');
      },
    },
    20
  );
  const pending = bindings.stage(stage, content);
  await closing.promise;
  await expect(pending).rejects.toMatchObject({ code: 'timeout' });
  expect(next).not.toHaveBeenCalled();
  expect(admission.snapshot()).toEqual({ operations: 1, reservedBytes: 8 });
  finish.resolve();
  await new Promise<void>((resolve) => setImmediate(resolve));
  expect(close).toHaveBeenCalledTimes(1);
  expect(admission.snapshot()).toEqual({ operations: 0, reservedBytes: 0 });
});

it('validates publication receipts again on the transport consumer', async () => {
  const { service } = fixture();
  const bindings = createHostFileBindings({
    identity,
    descriptor,
    service: {
      ...service,
      publish: async (request) => ({
        operationId: request.operationId,
        file: { ...file, bytes: descriptor.maxFileBytes + 1 },
      }),
    },
  });
  await expect(bindings.publish(file, 'p')).rejects.toMatchObject({
    code: 'overflow',
  });
});

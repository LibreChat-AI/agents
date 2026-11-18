import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { createHash, randomUUID } from 'node:crypto';
import { createServer, request as httpRequest } from 'node:http';
import { mkdtemp, open, readFile, rm, writeFile } from 'node:fs/promises';
import { Annotation, StateGraph, START, END } from '@langchain/langgraph';
import type {
  HostFileDescriptor,
  HostFileService,
  HostFileReadRequest,
  HostFileStageRequest,
  HostFilePublishRequest,
  HostFileChunk,
  HostFilePublication,
} from './files';
import type { HostFileStore } from './fileService';
import { createHostFileService, HostFileAdmission } from './fileService';
import { createHostFileBindings } from './fileBindings';
import { HostFileServiceError } from './files';

/** Loopback conformance transport only; production adapters own framing, auth and recovery. */
it('runs a conditional custom graph against authorized file storage over HTTP', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'sdk-host-files-'));
  const identity = {
    runtimeId: 'worker',
    agentId: 'graph',
    runId: 'run',
    userId: 'user',
    tenantId: 'tenant',
  };
  const secret = randomUUID();
  const descriptor = {
    version: 1 as const,
    operations: ['read', 'stage', 'publish'] as const,
    maxFileBytes: 1024,
    maxChunkBytes: 4,
  };
  const entries = new Map<
    string,
    { file: HostFileDescriptor; path: string; published: boolean }
  >();
  const publications = new Map<string, HostFilePublication>();
  const events: HostFileDescriptor[] = [];
  const input: HostFileDescriptor = {
    reference: 'input',
    revision: 'immutable-input',
    filename: 'input.txt',
    mediaType: 'text/plain',
    bytes: 6,
  };
  await writeFile(join(directory, 'input'), 'abcdef');
  entries.set(input.reference, {
    file: input,
    path: join(directory, 'input'),
    published: true,
  });
  function resolve(reference: string, revision: string) {
    const entry = entries.get(reference);
    if (!entry) throw new HostFileServiceError('forbidden');
    if (entry.file.revision !== revision)
      throw new HostFileServiceError('conflict');
    return entry;
  }
  const store: HostFileStore = {
    read: async (request, signal) => {
      const entry = resolve(request.reference, request.revision);
      signal?.throwIfAborted();
      const handle = await open(entry.path, 'r');
      try {
        const data = Buffer.alloc(request.length);
        const { bytesRead } = await handle.read(
          data,
          0,
          data.length,
          request.offset
        );
        return {
          reference: request.reference,
          revision: request.revision,
          offset: request.offset,
          data: data.subarray(0, bytesRead),
        };
      } finally {
        await handle.close();
      }
    },
    stage: async (request, content, signal) => {
      const reference = randomUUID();
      const path = join(directory, reference);
      const handle = await open(path, 'wx');
      const checksum = createHash('sha256');
      try {
        for await (const chunk of content) {
          signal?.throwIfAborted();
          await handle.write(chunk);
          checksum.update(chunk);
        }
        const file: HostFileDescriptor = {
          reference,
          revision: randomUUID(),
          filename: request.filename,
          mediaType: request.mediaType,
          bytes: request.bytes,
          sha256: checksum.digest('hex'),
        };
        entries.set(reference, { file, path, published: false });
        return file;
      } catch (error) {
        await rm(path, { force: true });
        throw error;
      } finally {
        await handle.close();
      }
    },
    publish: async (request) => {
      const entry = resolve(request.reference, request.revision);
      const previous = publications.get(request.operationId);
      if (previous) {
        if (
          previous.file.reference !== request.reference ||
          previous.file.revision !== request.revision
        )
          throw new HostFileServiceError('conflict');
        return previous;
      }
      entry.published = true;
      const result = { operationId: request.operationId, file: entry.file };
      publications.set(request.operationId, result);
      events.push(entry.file);
      return result;
    },
  };
  const admission = new HostFileAdmission({
    maxConcurrentOperations: 4,
    maxBufferedBytes: 32,
  });
  const host = createHostFileService({
    identity,
    descriptor,
    admission,
    store,
    timeoutMs: 1000,
  });
  const server = createServer(async (req, res) => {
    const controller = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) controller.abort();
    });
    if (req.headers.authorization !== `Bearer ${secret}`) {
      res.writeHead(403).end();
      return;
    }
    try {
      const metadata = JSON.parse(
        Buffer.from(String(req.headers['x-file-request']), 'base64').toString()
      ) as HostFileReadRequest & HostFileStageRequest & HostFilePublishRequest;
      let result:
        | HostFileDescriptor
        | HostFilePublication
        | (Omit<HostFileChunk, 'data'> & { data: string });
      if (req.url === '/read') {
        const chunk = await host.service.read(metadata, controller.signal);
        result = { ...chunk, data: Buffer.from(chunk.data).toString('base64') };
      } else if (req.url === '/stage') {
        async function* content() {
          for await (const chunk of req) {
            const data = Buffer.from(chunk as Uint8Array);
            for (
              let offset = 0;
              offset < data.length;
              offset += descriptor.maxChunkBytes
            )
              yield data.subarray(offset, offset + descriptor.maxChunkBytes);
          }
        }
        result = await host.service.stage(
          metadata,
          content(),
          controller.signal
        );
      } else if (req.url === '/publish')
        result = await host.service.publish(metadata, controller.signal);
      else {
        res.writeHead(404).end();
        return;
      }
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(result));
    } catch (error) {
      res.writeHead(400).end(
        JSON.stringify({
          code:
            error instanceof HostFileServiceError ? error.code : 'unavailable',
        })
      );
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (address === null || typeof address === 'string')
    throw new Error('Invalid test server');
  const port = address.port;
  async function rpc<T>(
    method: string,
    metadata:
      | HostFileReadRequest
      | HostFileStageRequest
      | HostFilePublishRequest,
    signal?: AbortSignal,
    content?: AsyncIterable<Uint8Array>
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const request = httpRequest(
        {
          hostname: '127.0.0.1',
          port,
          method: 'POST',
          path: `/${method}`,
          signal,
          headers: {
            authorization: `Bearer ${secret}`,
            'x-file-request': Buffer.from(JSON.stringify(metadata)).toString(
              'base64'
            ),
          },
        },
        (response) => {
          void (async () => {
            const chunks: Buffer[] = [];
            for await (const chunk of response)
              chunks.push(Buffer.from(chunk as Uint8Array));
            const value = JSON.parse(Buffer.concat(chunks).toString()) as T & {
              code?: HostFileServiceError['code'];
            };
            if (response.statusCode !== 200)
              throw new HostFileServiceError(value.code ?? 'unavailable');
            resolve(value);
          })().catch(reject);
        }
      );
      request.on('error', reject);
      void (async () => {
        if (content)
          for await (const chunk of content) {
            signal?.throwIfAborted();
            if (!request.write(chunk)) await once(request, 'drain');
          }
        request.end();
      })().catch((error: Error) => {
        request.destroy(error);
        reject(error);
      });
    });
  }
  const remote: HostFileService = {
    version: 1,
    read: async (request, signal) => {
      const value = await rpc<Omit<HostFileChunk, 'data'> & { data: string }>(
        'read',
        request,
        signal
      );
      return { ...value, data: Buffer.from(value.data, 'base64') };
    },
    stage: (request, content, signal) => rpc('stage', request, signal, content),
    publish: (request, signal) => rpc('publish', request, signal),
  };
  const files = createHostFileBindings({
    identity,
    descriptor: host.descriptor,
    service: remote,
  });
  try {
    const State = Annotation.Root({
      text: Annotation<string>,
      artifact: Annotation<HostFileDescriptor>,
    });
    const graph = new StateGraph(State)
      .addNode('read', async () => {
        let text = '';
        for await (const chunk of files.read(input))
          text += Buffer.from(chunk).toString();
        return { text };
      })
      .addNode('transform', ({ text }) => ({ text: text.toUpperCase() }))
      .addNode('stage', async ({ text }) => {
        async function* content() {
          for (let offset = 0; offset < text.length; offset += 4)
            yield Buffer.from(text.slice(offset, offset + 4));
        }
        const artifact = await files.stage(
          {
            operationId: 'graph-upload',
            filename: 'report.txt',
            mediaType: 'text/plain',
            bytes: Buffer.byteLength(text),
          },
          content()
        );
        expect(events).toHaveLength(0);
        return { artifact };
      })
      .addNode('publish', async ({ artifact }) => {
        await files.publish(artifact, 'graph-publication');
        return {};
      })
      .addEdge(START, 'read')
      .addEdge('read', 'transform')
      .addConditionalEdges('transform', ({ text }) => (text ? 'stage' : END))
      .addEdge('stage', 'publish')
      .addEdge('publish', END)
      .compile();
    const result = await graph.invoke({});
    expect(events).toEqual([result.artifact]);
    expect(
      await readFile(
        resolve(result.artifact.reference, result.artifact.revision).path,
        'utf8'
      )
    ).toBe('ABCDEF');
    await files.publish(result.artifact, 'graph-publication');
    expect(events).toHaveLength(1);
    const stale = files.read({ ...input, revision: 'stale' });
    await expect(stale[Symbol.asyncIterator]().next()).rejects.toMatchObject({
      code: 'conflict',
    });
    const forbidden = files.read({ ...input, reference: '../secret' });
    await expect(
      forbidden[Symbol.asyncIterator]().next()
    ).rejects.toMatchObject({ code: 'forbidden' });
    expect(admission.snapshot()).toEqual({ operations: 0, reservedBytes: 0 });
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await rm(directory, { recursive: true, force: true });
  }
});

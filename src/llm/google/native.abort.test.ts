import { createServer } from 'node:http';
import { afterEach, describe, expect, it } from '@jest/globals';
import type { Server, ServerResponse } from 'node:http';
import { fixturePort, imagePart } from './__tests__/nativeMediaFixtures';
import { CustomChatGoogleGenerativeAI } from './index';

describe('native Google stream transport cleanup', () => {
  let server: Server | undefined;

  afterEach(async () => {
    server?.closeAllConnections();
    await new Promise<void>((resolve, reject) => {
      if (!server) return resolve();
      server.close((error) => (error ? reject(error) : resolve()));
    });
  });

  it.each(['abort', 'disconnect'] as const)(
    'settles the live provider stream and its aggregate response on %s',
    async (mode) => {
      let providerResponse: ServerResponse | undefined;
      let closed!: () => void;
      const transportClosed = new Promise<void>((resolve) => {
        closed = resolve;
      });
      server = createServer((request, response) => {
        request.resume();
        providerResponse = response;
        response.once('close', closed);
        response.writeHead(200, { 'Content-Type': 'text/event-stream' });
        response.write(
          `data: ${JSON.stringify({
            candidates: [
              { index: 0, content: { role: 'model', parts: [imagePart] } },
            ],
            usageMetadata: {
              promptTokenCount: 5,
              candidatesTokenCount: 5,
              totalTokenCount: 10,
            },
          })}\n\n`
        );
      });
      await new Promise<void>((resolve) =>
        server!.listen(0, '127.0.0.1', resolve)
      );
      const address = server.address();
      if (address == null || typeof address === 'string')
        throw new Error('Missing fixture address');
      const port = fixturePort();
      const controller = new AbortController();
      const model = new CustomChatGoogleGenerativeAI({
        model: 'gemini-3-pro-image-preview',
        apiKey: 'fixture',
        baseUrl: `http://127.0.0.1:${address.port}`,
        nativeMedia: port,
        maxRetries: 0,
        _lc_stream_delay: 0,
      });
      const stream = await model.stream('Draw', { signal: controller.signal });
      const first = await stream.next();
      expect(first.value?.content).toEqual([
        expect.objectContaining({ type: 'image_file' }),
      ]);
      const next = stream.next();
      const rejected = expect(next).rejects.toThrow();
      if (mode === 'abort') controller.abort();
      else providerResponse!.destroy();
      await rejected;
      await transportClosed;
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(port.complete).not.toHaveBeenCalled();
      expect(port.fail).toHaveBeenCalledTimes(1);
      expect(port.fail).toHaveBeenCalledWith(
        expect.objectContaining({
          reason: mode === 'abort' ? 'aborted' : 'provider',
          usage: { input_tokens: 5, output_tokens: 5, total_tokens: 10 },
        })
      );
    },
    10000
  );
});

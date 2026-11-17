import { once } from 'node:events';
import { createServer } from 'node:http';
import { runInThisContext } from 'node:vm';
import { setImmediate } from 'node:timers/promises';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { afterAll, beforeAll, describe, expect, test } from '@jest/globals';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { GenerateContentRequest } from '@google/generative-ai';
import type { RequestListener, ServerResponse } from 'node:http';
import type { GoogleClientOptions } from '@/types';
import { CustomChatGoogleGenerativeAI } from './index';
import { GEMINI_PARTS } from './utils/signatures';

const dispatcherSymbol = Symbol.for('undici.globalDispatcher.1');

beforeAll(() => {
  // The SDK looks up Node's dispatcher on Jest's isolated global.
  new Headers();
  const descriptor = Object.getOwnPropertyDescriptor(
    runInThisContext('globalThis'),
    dispatcherSymbol
  );
  if (!descriptor) throw new Error('Missing native fetch dispatcher');
  Object.defineProperty(globalThis, dispatcherSymbol, {
    ...descriptor,
    configurable: true,
  });
});

afterAll(() => {
  Reflect.deleteProperty(globalThis, dispatcherSymbol);
});

const question = [new HumanMessage('Question.')];

function frame(text: string): string {
  return `data: ${JSON.stringify({
    candidates: [{ index: 0, content: { role: 'model', parts: [{ text }] } }],
  })}\n\n`;
}

function startStream(response: ServerResponse): void {
  response.writeHead(200, { 'content-type': 'text/event-stream' });
  response.flushHeaders();
}

async function withServer(
  handler: RequestListener,
  run: (baseUrl: string) => Promise<void>
): Promise<void> {
  const server = createServer(handler);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (address == null || typeof address === 'string')
    throw new Error('Missing port');
  try {
    await run(`http://127.0.0.1:${address.port}/gateway`);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }
}

function model(
  baseUrl: string,
  fields: Partial<GoogleClientOptions> = {}
): CustomChatGoogleGenerativeAI {
  return new CustomChatGoogleGenerativeAI({
    baseUrl,
    apiKey: 'test-key',
    model: 'gemini-4-carbon',
    _lc_stream_delay: 0,
    maxRetries: 0,
    ...fields,
  });
}

async function text(client: CustomChatGoogleGenerativeAI): Promise<string> {
  let output = '';
  for await (const chunk of client._streamResponseChunks(question, {})) {
    output += chunk.text;
  }
  return output;
}

async function within<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error('Connection did not close')),
          3000
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

describe('Google HTTP transport', () => {
  test('reads heartbeats and fragmented UTF-8/CRLF @scenario:google-sse-framing', async () => {
    const parts = Buffer.from(
      '\n: keep-alive\r\n\r\n' +
        frame('Before ').replace(/\n/g, '\r\n') +
        ': another heartbeat\n\n' +
        frame('🙂 After')
    );
    await withServer(
      (request, response) => {
        request.resume();
        startStream(response);
        void (async (): Promise<void> => {
          for (let index = 0; index < parts.length; index++) {
            if (response.destroyed) return;
            response.write(parts.subarray(index, index + 1));
            await setImmediate();
          }
          response.end();
        })();
      },
      async (baseUrl) => {
        expect(await text(model(baseUrl))).toBe('Before 🙂 After');
      }
    );
  });

  test.each([
    ['abort', 0],
    ['return', 0],
    ['abort', 25],
    ['return', 25],
  ] as const)(
    'closes an idle upstream connection on %s with smoothing=%s @scenario:google-stream-cleanup',
    async (mode, delay) => {
      const closed = Promise.withResolvers<void>();
      await withServer(
        (request, response) => {
          request.resume();
          response.once('close', () => closed.resolve());
          startStream(response);
          response.write(frame('First'));
        },
        async (baseUrl) => {
          const controller = new AbortController();
          const stream = model(baseUrl, {
            _lc_stream_delay: delay,
          })._streamResponseChunks(question, {
            signal: controller.signal,
          });
          expect((await stream.next()).value?.text).toBe('First');
          if (mode === 'return') {
            await stream.return(undefined);
          } else {
            const pending = stream.next();
            controller.abort();
            await expect(pending).rejects.toThrow(/abort/i);
          }
          await within(closed.promise);
        }
      );
    }
  );

  test.each([false, true])(
    'cancels before response headers with streaming=%s @scenario:google-request-abort',
    async (streaming) => {
      const received = Promise.withResolvers<void>();
      const closed = Promise.withResolvers<void>();
      let calls = 0;
      await withServer(
        (request, response) => {
          calls++;
          request.resume();
          response.once('close', () => closed.resolve());
          received.resolve();
        },
        async (baseUrl) => {
          const controller = new AbortController();
          const client = model(baseUrl);
          const pending = streaming
            ? client
              ._streamResponseChunks(question, { signal: controller.signal })
              .next()
            : client._generate(question, { signal: controller.signal });
          const rejected = expect(pending).rejects.toThrow(/abort/i);
          await within(received.promise);
          controller.abort();
          await rejected;
          await within(closed.promise);
          expect(calls).toBe(1);
        }
      );
    }
  );

  test.each([
    [
      'provider',
      'data: {"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}\n\n',
      /Quota exceeded/,
    ],
    ['malformed JSON', 'data: {broken}\n\n', /JSON|property/i],
    ['truncated JSON', 'data: {"candidates":', /JSON/i],
  ] as const)(
    'surfaces %s rather than silently ending @scenario:google-stream-errors',
    async (_name, payload, error) => {
      await withServer(
        (request, response) => {
          request.resume();
          startStream(response);
          response.end(payload);
        },
        async (baseUrl) => {
          await expect(text(model(baseUrl))).rejects.toThrow(error);
        }
      );
    }
  );

  test('preserves HTTP provider errors @scenario:google-http-errors', async () => {
    await withServer(
      (request, response) => {
        request.resume();
        response.writeHead(403, { 'content-type': 'application/json' });
        response.end('{"error":{"code":403,"message":"Permission denied"}}');
      },
      async (baseUrl) => {
        await expect(text(model(baseUrl))).rejects.toMatchObject({
          status: 403,
        });
      }
    );
  });

  test.each([false, true])(
    'preserves the wire contract with streaming=%s @scenario:google-transport-compatibility',
    async (streaming) => {
      const bodies: GenerateContentRequest[] = [];
      const urls: string[] = [];
      const parts = [
        {
          thought: true,
          text: 'Reasoning',
          thoughtSignature: 'reasoning-signature',
        },
        { text: 'Answer.', thoughtSignature: 'answer-signature' },
        {
          functionCall: {
            id: 'call-1',
            name: 'lookup',
            args: { query: 'value' },
          },
          thoughtSignature: 'call-signature',
        },
      ];
      const payload = {
        candidates: [
          { index: 0, content: { role: 'model', parts }, finishReason: 'STOP' },
        ],
        usageMetadata: {
          promptTokenCount: 3,
          candidatesTokenCount: 2,
          thoughtsTokenCount: 4,
          cachedContentTokenCount: 1,
          totalTokenCount: 9,
        },
      };
      await withServer(
        (request, response) => {
          urls.push(request.url ?? '');
          expect(request.headers['x-gateway']).toBe('preserved');
          expect(request.headers['x-goog-api-key']).toBe('test-key');
          let body = '';
          request.setEncoding('utf8');
          request.on('data', (chunk: string) => {
            body += chunk;
          });
          request.on('end', () => {
            bodies.push(JSON.parse(body) as GenerateContentRequest);
            if (request.url?.includes(':streamGenerateContent') === true) {
              startStream(response);
              response.end(`data: ${JSON.stringify(payload)}\n\n`);
            } else {
              response.writeHead(200, { 'content-type': 'application/json' });
              response.end(JSON.stringify(payload));
            }
          });
        },
        async (baseUrl) => {
          const client = model(baseUrl + '/', {
            model: 'models/gemini-4-hydrogen',
            apiVersion: 'v1alpha',
            customHeaders: { 'x-gateway': 'preserved' },
            thinkingConfig: { thinkingLevel: 'XHIGH' },
            temperature: 0.7,
            maxOutputTokens: 2048,
            includeServerSideToolInvocations: true,
          });
          const messages = [new SystemMessage('System.'), ...question];
          const options = {
            tools: [
              {
                functionDeclarations: [
                  {
                    name: 'lookup',
                    parameters: {
                      type: 'object',
                      properties: { query: { type: 'string' } },
                    },
                  },
                ],
              },
            ],
          };
          let output;
          if (streaming) {
            for await (const chunk of client._streamResponseChunks(
              messages,
              options
            )) {
              output = output ? output.concat(chunk.message) : chunk.message;
            }
          } else {
            output = (await client._generate(messages, options)).generations[0]
              .message;
          }
          expect(output).toMatchObject({
            additional_kwargs: { reasoning: 'Reasoning' },
            tool_calls: [
              { id: 'call-1', name: 'lookup', args: { query: 'value' } },
            ],
            usage_metadata: {
              input_tokens: 3,
              output_tokens: 6,
              total_tokens: 9,
              input_token_details: { cache_read: 1 },
            },
          });
          expect(output?.additional_kwargs[GEMINI_PARTS]).toBeDefined();
          if (!output) throw new Error('Missing model response');
          await client._generate(
            [...question, output, new HumanMessage('Follow up.')],
            {}
          );
          expect(urls[0]).toBe(
            `/gateway/v1alpha/models/gemini-4-hydrogen:${streaming ? 'streamGenerateContent?alt=sse' : 'generateContent'}`
          );
          expect(bodies[0]).toMatchObject({
            systemInstruction: { role: 'system', parts: [{ text: 'System.' }] },
            contents: [{ role: 'user', parts: [{ text: 'Question.' }] }],
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 2048,
              thinkingConfig: { thinkingLevel: 'XHIGH' },
            },
            tools: options.tools,
            toolConfig: { includeServerSideToolInvocations: true },
          });
          expect(bodies[1].systemInstruction).toBeUndefined();
          expect(bodies[1].contents[1].parts).toEqual(parts);
          expect(bodies).toHaveLength(2);
        }
      );
    }
  );
});

test.each([
  'Cached system.',
  { text: 'Cached system.' },
  { role: 'system', parts: [{ text: 'Cached system.' }] },
])(
  'keeps cached instructions %j on the maintained transport @scenario:google-cached-transport',
  async (systemInstruction) => {
    const bodies: GenerateContentRequest[] = [];
    const tools = [{ functionDeclarations: [{ name: 'lookup' }] }];
    await withServer(
      (request, response) => {
        let body = '';
        request.setEncoding('utf8');
        request.on('data', (chunk: string) => {
          body += chunk;
        });
        request.on('end', () => {
          bodies.push(JSON.parse(body) as GenerateContentRequest);
          if (request.url?.includes(':streamGenerateContent') === true) {
            startStream(response);
            response.end(': cached heartbeat\n\n' + frame('Cached answer.'));
          } else {
            response.writeHead(200, { 'content-type': 'application/json' });
            response.end(frame('Cached answer.').slice(6).trim());
          }
        });
      },
      async (baseUrl) => {
        const client = model(baseUrl);
        client.useCachedContent(
          {
            name: 'cachedContents/test',
            model: 'models/gemini-4-carbon',
            contents: [],
            tools,
            systemInstruction,
          },
          { model: 'gemini-4-carbon' },
          { baseUrl, customHeaders: new Headers({ 'x-cache': 'yes' }) }
        );
        expect(await text(client)).toBe('Cached answer.');
        await client._generate(question, {});
        expect(bodies).toHaveLength(2);
        for (const body of bodies) {
          expect(body).toMatchObject({
            cachedContent: 'cachedContents/test',
            systemInstruction: {
              role: 'system',
              parts: [{ text: 'Cached system.' }],
            },
            tools,
          });
        }
      }
    );
  }
);

test('retains cached-content input validation', () => {
  const client = model('http://127.0.0.1');
  expect(() =>
    client.useCachedContent({ model: 'gemini-4-carbon', contents: [] })
  ).toThrow(/name/);
  expect(() =>
    client.useCachedContent({ name: 'cachedContents/test', contents: [] })
  ).toThrow(/model/);
  const cache = {
    name: 'cachedContents/test',
    model: 'models/gemini-4-carbon',
    contents: [],
    systemInstruction: 'Cached.',
  };
  expect(() =>
    client.useCachedContent(cache, { model: 'gemini-2.5-pro' })
  ).toThrow(/Different model/);
  expect(() =>
    client.useCachedContent(cache, {
      model: 'gemini-4-carbon',
      systemInstruction: 'Other.',
    })
  ).toThrow(/Different systemInstruction/);
});

test('uses the maintained transport for inherited stream events @scenario:google-events-transport', async () => {
  class EventModel extends CustomChatGoogleGenerativeAI {
    events(): AsyncGenerator<ChatModelStreamEvent> {
      return this._streamChatModelEvents(question, {});
    }
  }
  await withServer(
    (request, response) => {
      request.resume();
      startStream(response);
      response.end(': heartbeat\n\n' + frame('Event answer.'));
    },
    async (baseUrl) => {
      const client = new EventModel({
        baseUrl,
        model: 'gemini-4-carbon',
        apiKey: 'test-key',
        maxRetries: 0,
      });
      const events = [];
      for await (const event of client.events()) events.push(event);
      expect(events).toContainEqual(
        expect.objectContaining({
          event: 'content-block-delta',
          delta: { type: 'text-delta', text: 'Event answer.' },
        })
      );
    }
  );
});

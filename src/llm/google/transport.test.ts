import { once } from 'node:events';
import { createServer } from 'node:http';
import { setImmediate } from 'node:timers/promises';
import { describe, expect, test } from '@jest/globals';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { GenerateContentRequest, Part } from '@google/generative-ai';
import type { RequestListener, ServerResponse } from 'node:http';
import type { GoogleClientOptions } from '@/types';
import { CustomChatGoogleGenerativeAI } from './index';
import { GEMINI_PARTS } from './utils/signatures';
import { attemptInvoke } from '@/llm/invoke';
import { Providers } from '@/common';

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
          {
            index: 0,
            content: { role: 'model', parts },
            finishReason: 'STOP',
            finishMessage: 'Complete.',
            citationMetadata: {
              citationSources: [{ uri: 'https://example.com/source' }],
            },
          },
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
          const metadata = [];
          if (streaming) {
            for await (const chunk of client._streamResponseChunks(
              messages,
              options
            )) {
              if (chunk.generationInfo != null)
                metadata.push(chunk.generationInfo);
              output = output ? output.concat(chunk.message) : chunk.message;
            }
          } else {
            const result = await client._generate(messages, options);
            output = result.generations[0].message;
            metadata.push(result.generations[0].generationInfo);
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
          expect(metadata).toContainEqual(
            expect.objectContaining({
              finishMessage: 'Complete.',
              citationMetadata: {
                citationSources: [{ uri: 'https://example.com/source' }],
              },
            })
          );
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

test('preserves protected lazy streaming and the attempt byte budget @scenario:google-protected-stream', async () => {
  const content = 'word '.repeat(80);
  await withServer(
    (request, response) => {
      request.resume();
      startStream(response);
      response.end(frame(content));
    },
    async (baseUrl) => {
      const result = await attemptInvoke({
        model: model(baseUrl, { _lc_stream_delay: 25 }),
        messages: question,
        provider: Providers.GOOGLE,
        onChunk: () => {},
        providerTextProtection: {
          version: 1,
          timeoutMs: 10000,
          maxAttemptBytes: 3500,
          maxBufferedBytes: 7000,
          classify: () => 'prose',
          inspect: ({ content }) => ({
            version: 1,
            ok: true,
            value: { content, replacements: 0, categories: [] },
          }),
        },
      });
      expect(result.messages?.[0].content).toBe(content);
    }
  );
});

test('closes the upstream request when the protection budget rejects a frame @scenario:google-protected-abort', async () => {
  const closed = Promise.withResolvers<void>();
  let inspections = 0;
  let publications = 0;
  await withServer(
    (request, response) => {
      request.resume();
      response.once('close', () => closed.resolve());
      startStream(response);
      response.write(frame('word '.repeat(80)));
    },
    async (baseUrl) => {
      await expect(
        attemptInvoke({
          model: model(baseUrl, { _lc_stream_delay: 25 }),
          messages: question,
          provider: Providers.GOOGLE,
          onChunk: () => {
            publications++;
          },
          providerTextProtection: {
            version: 1,
            timeoutMs: 10000,
            maxAttemptBytes: 1500,
            maxBufferedBytes: 7000,
            classify: () => 'prose',
            inspect: ({ content }) => {
              inspections++;
              return {
                version: 1,
                ok: true,
                value: { content, replacements: 0, categories: [] },
              };
            },
          },
        })
      ).rejects.toMatchObject({ code: 'overflow' });
      await within(closed.promise);
      expect(inspections).toBe(0);
      expect(publications).toBe(0);
    }
  );
});

describe('inherited completionWithRetry contract', () => {
  test.each([
    ['string', 'Question.', { role: 'user', parts: [{ text: 'Question.' }] }],
    [
      'parts',
      ['Describe.', { inlineData: { mimeType: 'image/png', data: 'AA==' } }],
      {
        role: 'user',
        parts: [
          { text: 'Describe.' },
          { inlineData: { mimeType: 'image/png', data: 'AA==' } },
        ],
      },
    ],
    [
      'function response',
      [{ functionResponse: { name: 'lookup', response: { value: 42 } } }],
      {
        role: 'function',
        parts: [
          { functionResponse: { name: 'lookup', response: { value: 42 } } },
        ],
      },
    ],
    [
      'request',
      { contents: [{ role: 'user', parts: [{ text: 'Question.' }] }] },
      { role: 'user', parts: [{ text: 'Question.' }] },
    ],
  ] as const)(
    'accepts %s shorthand and returns response helpers @scenario:google-public-completion',
    async (_name, request, expected) => {
      let body: GenerateContentRequest | undefined;
      await withServer(
        (incoming, response) => {
          let raw = '';
          incoming.setEncoding('utf8');
          incoming.on('data', (chunk: string) => {
            raw += chunk;
          });
          incoming.on('end', () => {
            body = JSON.parse(raw) as GenerateContentRequest;
            response.writeHead(200, { 'content-type': 'application/json' });
            response.end(
              JSON.stringify({
                candidates: [
                  {
                    index: 0,
                    content: {
                      role: 'model',
                      parts: [
                        { text: 'Answer.' },
                        {
                          executableCode: {
                            language: 'python',
                            code: 'print(42)',
                          },
                        },
                        {
                          codeExecutionResult: {
                            outcome: 'OUTCOME_OK',
                            output: '42',
                          },
                        },
                        {
                          functionCall: { name: 'lookup', args: { value: 42 } },
                          thoughtSignature: 'signature',
                        },
                      ],
                    },
                  },
                ],
              })
            );
          });
        },
        async (baseUrl) => {
          const input = request as
            | string
            | GenerateContentRequest
            | (string | Part)[];
          const { response } = await model(baseUrl, {
            temperature: 0.7,
          }).completionWithRetry(input);
          expect(body?.contents).toEqual([expected]);
          expect(body?.generationConfig).toMatchObject({ temperature: 0.7 });
          const legacy = new GoogleGenerativeAI('test-key').getGenerativeModel(
            { model: 'gemini-4-carbon' },
            { baseUrl }
          );
          const old = await legacy.generateContent(input);
          expect(response.text()).toBe(old.response.text());
          expect(response.functionCalls()).toEqual(
            old.response.functionCalls()
          );
          expect(response.text()).toBe(
            'Answer.\n```python\nprint(42)\n```\n\n```\n42\n```\n'
          );
          expect(response.functionCalls()).toEqual([
            { name: 'lookup', args: { value: 42 } },
          ]);
          expect(response.functionCall()).toEqual({
            name: 'lookup',
            args: { value: 42 },
          });
          expect(response.candidates?.[0].content.parts[3]).toHaveProperty(
            'thoughtSignature',
            'signature'
          );
        }
      );
    }
  );

  test.each([
    { candidates: [] },
    { promptFeedback: { blockReason: 'SAFETY' } },
    ...['SAFETY', 'RECITATION', 'LANGUAGE'].map((finishReason) => ({
      candidates: [
        {
          index: 0,
          finishReason,
          content: { role: 'model', parts: [{ text: 'Blocked.' }] },
        },
      ],
    })),
  ])(
    'keeps response helper refusal semantics for %j @scenario:google-public-response-helpers',
    async (payload) => {
      await withServer(
        (request, response) => {
          request.resume();
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify(payload));
        },
        async (baseUrl) => {
          const { response } = await model(baseUrl).completionWithRetry({
            contents: [],
          });
          if ('candidates' in payload && payload.candidates.length === 0) {
            expect(response.text()).toBe('');
            expect(response.functionCalls()).toBeUndefined();
            expect(response.functionCall()).toBeUndefined();
            return;
          }
          for (const helper of [
            response.text,
            response.functionCalls,
            response.functionCall,
          ]) {
            expect(helper).toThrow(/blocked/);
            try {
              helper();
            } catch (error) {
              expect(error).toMatchObject({ response });
            }
          }
        }
      );
    }
  );

  test('rejects empty or mixed shorthand before HTTP @scenario:google-public-request-validation', async () => {
    const client = model('http://127.0.0.1:1');
    await expect(client.completionWithRetry([])).rejects.toThrow(/No content/);
    await expect(
      client.completionWithRetry([
        'Question.',
        { functionResponse: { name: 'lookup', response: {} } },
      ])
    ).rejects.toThrow(/cannot be mixed/);
  });
});

test('completionWithRetry aborts its HTTP request @scenario:google-public-completion-abort', async () => {
  const received = Promise.withResolvers<void>();
  const closed = Promise.withResolvers<void>();
  await withServer(
    (request, response) => {
      request.resume();
      response.once('close', () => closed.resolve());
      received.resolve();
    },
    async (baseUrl) => {
      const controller = new AbortController();
      const pending = model(baseUrl).completionWithRetry('Question.', {
        signal: controller.signal,
      });
      const rejected = expect(pending).rejects.toThrow(/abort/i);
      await within(received.promise);
      controller.abort();
      await rejected;
      await within(closed.promise);
    }
  );
});

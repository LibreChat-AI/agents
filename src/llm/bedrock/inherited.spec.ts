// Inherited from @langchain/aws@1.4.2
//   tests/chat_models_stream_events.test.ts   (native streamEvents)
//   tests/chat_models.invocation.test.ts      (invocationParams / system / bearer / headers)
//   tests/convertToConverseTools.test.ts       (tool conversion util)
//   utils/tests/message_outputs.test.ts        (usage-metadata conversion util)
//
// Consolidates the deterministic, unit-testable cases from the four upstream
// suites and runs them against OUR fork, `CustomChatBedrockConverse` (imported
// here as `ChatBedrockConverse` from `@/llm/bedrock`).
//
// Applicability to our fork:
//   `CustomChatBedrockConverse extends ChatBedrockConverse` (the installed
//   @langchain/aws@1.4.2). It overrides `invocationParams`, `_streamResponseChunks`,
//   `_generateNonStreaming`, and `getModelId`, but does NOT override the native
//   `_streamChatModelEvents` (the streamEvents path) — that is inherited verbatim,
//   so the streamEvents cases are a parity check that our overridden
//   `_streamResponseChunks` still feeds the inherited event machinery correctly.
//   Bearer-token auth and the defaultHeaders middleware are constructor-level
//   features inherited unchanged from the base class.
//
// Adaptation (vitest -> jest):
//   - `@jest/globals`; `vi.*` -> `jest.*`; `zod/v3` -> `zod` (not needed; see below).
//   - Import the class from `@/llm/bedrock`, not `../chat_models.js`.
//   - Upstream drove the SDK via `vi.mock('@aws-sdk/client-bedrock-runtime', ...)`.
//     We instead use the harness from `llm.spec.ts`: inject a mock `client.send`
//     (capturing the real `ConverseCommand`/`ConverseStreamCommand`), or spy on a
//     real injected client's `middlewareStack.add`. This keeps the SDK command
//     classes real and exercises the fork's real generation paths — no live API.
//   - Upstream's streamEvents suite used vitest custom matchers
//     (`toHaveStreamText` / `toHaveStreamReasoning` / `toHaveStreamToolCalls` /
//     `toHaveStreamUsage`). Those don't exist in jest; they wrap the public
//     `ChatModelStream` sub-streams (`.text` / `.reasoning` / `.toolCalls` /
//     `.usage`), which we assert directly.
//
// Dropped (inherited): tests/convertToConverseTools.test.ts — the whole file.
//   `convertToConverseTools` and `supportedToolChoiceValuesForModel` live in
//   @langchain/aws's internal `utils/tools.js`, are NOT on the package's public
//   export surface, and our fork neither uses nor re-exports them (verified via
//   grep over src/ and the package `exports` map). Nothing in our fork to assert.
//
// Dropped (inherited): live — none of the four source files contained live
//   `.invoke()`/`.stream()`-to-API cases; every retained case runs against a
//   mocked transport.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { test, expect, describe, jest, afterEach } from '@jest/globals';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import {
  HumanMessage,
  SystemMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  Message as BedrockMessage,
  ConverseResponse,
  ConverseCommandInput,
  ConverseStreamCommandInput,
} from '@aws-sdk/client-bedrock-runtime';
import type { BaseChatModelCallOptions } from '@langchain/core/language_models/chat_models';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import {
  convertConverseMessageToLangChainMessage,
  handleConverseStreamMetadata,
} from './utils';
import { CustomChatBedrockConverse as ChatBedrockConverse } from './index';

jest.setTimeout(120000);

const baseConstructorArgs = {
  region: 'us-east-1',
  credentials: {
    secretAccessKey: 'test-secret',
    accessKeyId: 'test-key',
  },
  model: 'anthropic.claude-3-sonnet-20240229-v1:0',
};

afterEach(() => {
  jest.restoreAllMocks();
});

// ─── Mock transport helpers ─────────────────────────────────────
//
// Mirrors `llm.spec.ts`: inject a mock `client.send` that captures the real
// command objects and returns canned non-stream / stream responses.

type CapturingClient = {
  client: BedrockRuntimeClient;
  sent: unknown[];
};

function nonStreamingResponse(): unknown {
  return {
    output: {
      message: { role: 'assistant', content: [{ text: 'Response' }] },
    },
    stopReason: 'end_turn',
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
  };
}

function streamingResponse(events: Array<Record<string, unknown>>): unknown {
  return {
    stream: (async function* () {
      for (const event of events) {
        yield event;
      }
    })(),
  };
}

function capturingClient(makeResponse: () => unknown): CapturingClient {
  const sent: unknown[] = [];
  const send = jest
    .fn<(command: unknown) => Promise<unknown>>()
    .mockImplementation(async (command) => {
      sent.push(command);
      return makeResponse();
    });
  return { client: { send } as unknown as BedrockRuntimeClient, sent };
}

function streamEventsGen(
  model: ChatBedrockConverse,
  messages: HumanMessage[],
  options: BaseChatModelCallOptions = {} as BaseChatModelCallOptions
): AsyncGenerator<ChatModelStreamEvent> {
  return (
    model as unknown as {
      _streamChatModelEvents: (
        messages: unknown[],
        options: BaseChatModelCallOptions
      ) => AsyncGenerator<ChatModelStreamEvent>;
    }
  )._streamChatModelEvents(messages, options);
}

// ─── streamEvents fixtures (Bedrock Converse stream events) ─────

function bedrockTextStream(): Array<Record<string, unknown>> {
  return [
    { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'Hello' } } },
    { contentBlockDelta: { contentBlockIndex: 0, delta: { text: ' world' } } },
  ];
}

function bedrockReasoningStream(): Array<Record<string, unknown>> {
  return [
    {
      contentBlockDelta: {
        contentBlockIndex: 0,
        delta: { reasoningContent: { text: 'Let me reason...' } },
      },
    },
  ];
}

function bedrockToolStream(): Array<Record<string, unknown>> {
  return [
    {
      contentBlockDelta: {
        contentBlockIndex: 0,
        delta: { text: 'Let me search.' },
      },
    },
    {
      contentBlockStart: {
        contentBlockIndex: 1,
        start: { toolUse: { toolUseId: 'toolu_1', name: 'web_search' } },
      },
    },
    {
      contentBlockDelta: {
        contentBlockIndex: 1,
        delta: { toolUse: { input: '{"query":"weather"}' } },
      },
    },
  ];
}

function bedrockUsageStream(): Array<Record<string, unknown>> {
  return [
    {
      metadata: {
        usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 },
      },
    },
  ];
}

function mockBedrock(
  events: Array<Record<string, unknown>>
): ChatBedrockConverse {
  const { client } = capturingClient(() => streamingResponse(events));
  return new ChatBedrockConverse({
    ...baseConstructorArgs,
    model: 'anthropic.claude-3-haiku-20240307-v1:0',
    client,
  });
}

// ─── streamEvents (native, inherited `_streamChatModelEvents`) ──
//
// From tests/chat_models_stream_events.test.ts. Upstream's custom matchers wrap
// the `ChatModelStream` sub-streams asserted here directly.
describe('CustomChatBedrockConverse.streamEvents (inherited native path)', () => {
  test('streams text', async () => {
    const stream = new ChatModelStream(
      streamEventsGen(mockBedrock(bedrockTextStream()), [
        new HumanMessage('Hello'),
      ])
    );
    expect(await stream.text).toBe('Hello world');
  });

  test('streams reasoning', async () => {
    const stream = new ChatModelStream(
      streamEventsGen(mockBedrock(bedrockReasoningStream()), [
        new HumanMessage('Hello'),
      ])
    );
    expect(await stream.reasoning).toBe('Let me reason...');
  });

  test('streams tool calls', async () => {
    const stream = new ChatModelStream(
      streamEventsGen(mockBedrock(bedrockToolStream()), [
        new HumanMessage('Hello'),
      ])
    );
    const calls = await stream.toolCalls;
    expect(calls).toEqual([
      expect.objectContaining({
        name: 'web_search',
        args: { query: 'weather' },
      }),
    ]);
  });

  test('streams usage', async () => {
    const stream = new ChatModelStream(
      streamEventsGen(
        mockBedrock(bedrockUsageStream()),
        [new HumanMessage('Hello')],
        { streamUsage: true } as BaseChatModelCallOptions
      )
    );
    // Fork note: our metadata handler always emits an `input_token_details`
    // object (here empty `{}`) where upstream's matcher only checked the three
    // top-level token counts. Assert the counts upstream asserted via toMatchObject.
    expect(await stream.usage).toMatchObject({
      input_tokens: 5,
      output_tokens: 2,
      total_tokens: 7,
    });
  });
});

// ─── invocationParams / system / inferenceConfig ───────────────
//
// From tests/chat_models.invocation.test.ts. Upstream read the command input via
// `Reflect.get(ConverseCommand, 'lastInput')` from a vi.mock'd SDK; we read the
// real captured command's `.input` instead.
describe('CustomChatBedrockConverse invocationParams', () => {
  describe('inferenceConfig conditional logic', () => {
    test('covers all inferenceConfig scenarios compactly', () => {
      const cases: Array<{
        name: string;
        ctor?: Partial<{
          maxTokens: number;
          temperature: number;
          topP: number;
        }>;
        opts?: { stop?: string[] };
        expectDefined: boolean;
        expectValues?: Partial<{
          maxTokens: number;
          temperature: number;
          topP: number;
          stopSequences: string[];
        }>;
        expectUndefinedKeys?: Array<
          'maxTokens' | 'temperature' | 'topP' | 'stopSequences'
        >;
      }> = [
        {
          name: 'undefined when no inference values are set',
          expectDefined: false,
        },
        {
          name: 'includes only maxTokens when set',
          ctor: { maxTokens: 100 },
          expectDefined: true,
          expectValues: { maxTokens: 100 },
          expectUndefinedKeys: ['temperature', 'topP', 'stopSequences'],
        },
        {
          name: 'includes only temperature when set',
          ctor: { temperature: 0.7 },
          expectDefined: true,
          expectValues: { temperature: 0.7 },
          expectUndefinedKeys: ['maxTokens', 'topP', 'stopSequences'],
        },
        {
          name: 'includes only topP when set',
          ctor: { topP: 0.9 },
          expectDefined: true,
          expectValues: { topP: 0.9 },
          expectUndefinedKeys: ['maxTokens', 'temperature', 'stopSequences'],
        },
        {
          name: 'includes stopSequences when provided',
          opts: { stop: ['END', 'STOP'] },
          expectDefined: true,
          expectValues: { stopSequences: ['END', 'STOP'] },
          expectUndefinedKeys: ['maxTokens', 'temperature', 'topP'],
        },
        {
          name: 'includes all values when all are set',
          ctor: { maxTokens: 200, temperature: 0.5, topP: 0.95 },
          opts: { stop: ['END'] },
          expectDefined: true,
          expectValues: {
            maxTokens: 200,
            temperature: 0.5,
            topP: 0.95,
            stopSequences: ['END'],
          },
        },
        {
          name: 'undefined when stop sequences is empty array',
          opts: { stop: [] },
          expectDefined: false,
        },
      ];

      for (const c of cases) {
        const model = new ChatBedrockConverse({
          ...baseConstructorArgs,
          ...(c.ctor ?? {}),
        });
        const params = model.invocationParams(c.opts);
        if (!c.expectDefined) {
          expect(params.inferenceConfig).toBeUndefined();
          continue;
        }
        expect(params.inferenceConfig).toBeDefined();
        const ic = params.inferenceConfig as Record<string, unknown>;
        if (c.expectValues?.maxTokens !== undefined) {
          expect(ic.maxTokens).toBe(c.expectValues.maxTokens);
        }
        if (c.expectValues?.temperature !== undefined) {
          expect(ic.temperature).toBe(c.expectValues.temperature);
        }
        if (c.expectValues?.topP !== undefined) {
          expect(ic.topP).toBe(c.expectValues.topP);
        }
        if (c.expectValues?.stopSequences !== undefined) {
          expect(ic.stopSequences).toEqual(c.expectValues.stopSequences);
        }
        for (const k of c.expectUndefinedKeys ?? []) {
          expect(ic[k]).toBeUndefined();
        }
      }
    });
  });

  describe('system parameter conditional logic (invoke)', () => {
    test.each([
      {
        name: 'no system messages',
        messages: [new HumanMessage('Hello')],
        expectedSystem: { present: false, length: 0, texts: [] as string[] },
      },
      {
        name: 'one system message',
        messages: [
          new SystemMessage('You are a helpful assistant.'),
          new HumanMessage('Hello'),
        ],
        expectedSystem: {
          present: true,
          length: 1,
          texts: ['You are a helpful assistant.'],
        },
      },
      {
        name: 'multiple system messages',
        messages: [
          new SystemMessage('You are a helpful assistant.'),
          new SystemMessage('Be concise in your responses.'),
          new HumanMessage('Hello'),
        ],
        expectedSystem: {
          present: true,
          length: 2,
          texts: [
            'You are a helpful assistant.',
            'Be concise in your responses.',
          ],
        },
      },
    ])(
      'invoke should handle system parameter: $name',
      async ({ messages, expectedSystem }) => {
        const { client, sent } = capturingClient(nonStreamingResponse);
        const model = new ChatBedrockConverse({
          ...baseConstructorArgs,
          client,
        });
        await model.invoke(messages);
        const command = sent[0] as ConverseCommand;
        expect(command).toBeInstanceOf(ConverseCommand);
        const input = command.input as ConverseCommandInput;
        if (!expectedSystem.present) {
          expect(input).not.toHaveProperty('system');
          return;
        }
        expect(input).toHaveProperty('system');
        const system = input.system as NonNullable<typeof input.system>;
        expect(system).toHaveLength(expectedSystem.length);
        expectedSystem.texts.forEach((t, i) => {
          expect(system[i]).toHaveProperty('text', t);
        });
      }
    );
  });

  describe('stream method system parameter logic', () => {
    test.each([
      {
        name: 'no system messages',
        messages: [new HumanMessage('Hello')],
        expectedPresent: false,
        expectedLength: 0,
        expectedTexts: [] as string[],
      },
      {
        name: 'one system message',
        messages: [
          new SystemMessage('You are a helpful assistant.'),
          new HumanMessage('Hello'),
        ],
        expectedPresent: true,
        expectedLength: 1,
        expectedTexts: ['You are a helpful assistant.'],
      },
    ])(
      'stream should handle system parameter: $name',
      async ({ messages, expectedPresent, expectedLength, expectedTexts }) => {
        const { client, sent } = capturingClient(() =>
          streamingResponse([
            {
              contentBlockDelta: {
                contentBlockIndex: 0,
                delta: { text: 'Response' },
              },
            },
            {
              metadata: {
                usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
              },
            },
          ])
        );
        const model = new ChatBedrockConverse({
          ...baseConstructorArgs,
          client,
        });
        const stream = await model.stream(messages);
        let chunks = 0;
        for await (const _chunk of stream) {
          chunks += 1;
        }
        expect(chunks).toBeGreaterThan(0);
        const command = sent[0] as ConverseStreamCommand;
        expect(command).toBeInstanceOf(ConverseStreamCommand);
        const input = command.input as ConverseStreamCommandInput;
        if (!expectedPresent) {
          // Fork divergence: our `_streamResponseChunks` always spreads
          // `system: converseSystem` into the command, so with no system message
          // it sends an empty `system: []` rather than omitting it like upstream
          // (and like our non-stream invoke path). Bedrock accepts an empty
          // system array. Assert OURS.
          expect(input.system).toEqual([]);
          return;
        }
        expect(input).toHaveProperty('system');
        const system = input.system as NonNullable<typeof input.system>;
        expect(system).toHaveLength(expectedLength);
        expectedTexts.forEach((t, i) => {
          expect(system[i]).toHaveProperty('text', t);
        });
      }
    );
  });

  // Dropped (inherited): "prompt caching request mapping" (invoke/stream map
  // cache_control to system/messages/tools) — cache_control cases are owned by a
  // separate agent (inherited-cache.spec.ts).

  describe('bearer token auth (inherited from base constructor)', () => {
    // Upstream asserted on a vi.mock'd `BedrockRuntimeClient.lastConfig`. We
    // assert on the real constructed client's resolved config instead, which
    // exercises the actual SDK auth wiring our fork inherits unchanged.
    test('configures bearer auth from constructor token', async () => {
      const model = new ChatBedrockConverse({
        ...baseConstructorArgs,
        bedrockBearerToken: 'test-bearer-token',
      } as never);
      expect(
        (model as unknown as { bedrockBearerToken?: string }).bedrockBearerToken
      ).toBe('test-bearer-token');

      const config = (
        model as unknown as {
          client: {
            config: {
              authSchemePreference?: (() => Promise<string[]>) | string[];
              credentials?: unknown;
              token?: () => Promise<{ token: string }>;
            };
          };
        }
      ).client.config;

      const authPref = config.authSchemePreference;
      const resolvedAuth =
        typeof authPref === 'function' ? await authPref() : authPref;
      expect(resolvedAuth).toEqual(['httpBearerAuth']);
      await expect(config.token?.()).resolves.toEqual({
        token: 'test-bearer-token',
      });
    });

    test('configures bearer auth from AWS_BEARER_TOKEN_BEDROCK', async () => {
      process.env.AWS_BEARER_TOKEN_BEDROCK = 'env-bearer-token';
      try {
        const model = new ChatBedrockConverse({
          ...baseConstructorArgs,
        } as never);
        expect(
          (model as unknown as { bedrockBearerToken?: string })
            .bedrockBearerToken
        ).toBe('env-bearer-token');

        const config = (
          model as unknown as {
            client: {
              config: {
                authSchemePreference?: (() => Promise<string[]>) | string[];
                token?: () => Promise<{ token: string }>;
              };
            };
          }
        ).client.config;
        const authPref = config.authSchemePreference;
        const resolvedAuth =
          typeof authPref === 'function' ? await authPref() : authPref;
        expect(resolvedAuth).toEqual(['httpBearerAuth']);
        await expect(config.token?.()).resolves.toEqual({
          token: 'env-bearer-token',
        });
      } finally {
        delete process.env.AWS_BEARER_TOKEN_BEDROCK;
      }
    });
  });

  describe('defaultHeaders middleware (inherited from base constructor)', () => {
    // Spy on a real injected client's `middlewareStack.add` (instead of a
    // vi.mock'd stack) so the actual header-injection middleware is captured and
    // invoked.
    test('registers middleware on client when defaultHeaders are provided', () => {
      const injected = new BedrockRuntimeClient({
        region: 'us-east-1',
        credentials: { secretAccessKey: 's', accessKeyId: 'a' },
      });
      const addSpy = jest.spyOn(injected.middlewareStack, 'add');

      new ChatBedrockConverse({
        ...baseConstructorArgs,
        defaultHeaders: {
          'X-Foo': 'Bar',
          'anthropic-beta': 'prompt-caching-2024-07-31',
        },
        client: injected,
      } as never);

      expect(addSpy).toHaveBeenCalledTimes(1);
      const [middlewareFn, options] = addSpy.mock.calls[0] as [
        (
          next: (a: unknown) => Promise<unknown>
        ) => (a: {
          request: { headers: Record<string, string> };
        }) => Promise<unknown>,
        unknown,
      ];
      expect(options).toEqual({
        step: 'build',
        name: 'langchain_aws_default_headers',
      });

      const fakeRequest = { headers: {} as Record<string, string> };
      const fakeNext = jest.fn<() => Promise<unknown>>().mockResolvedValue({});
      void middlewareFn(fakeNext)({ request: fakeRequest });
      expect(fakeRequest.headers['X-Foo']).toBe('Bar');
      expect(fakeRequest.headers['anthropic-beta']).toBe(
        'prompt-caching-2024-07-31'
      );
    });

    test('does not register middleware when defaultHeaders is absent', () => {
      const injected = new BedrockRuntimeClient({
        region: 'us-east-1',
        credentials: { secretAccessKey: 's', accessKeyId: 'a' },
      });
      const addSpy = jest.spyOn(injected.middlewareStack, 'add');

      new ChatBedrockConverse({
        ...baseConstructorArgs,
        client: injected,
      } as never);

      expect(addSpy).not.toHaveBeenCalled();
    });
  });
});

// ─── usage-metadata conversion utils ───────────────────────────
//
// From utils/tests/message_outputs.test.ts. Our fork re-exports these from
// `./utils` (utils/message_outputs.ts), so they are tested directly.
describe('message output usage metadata conversion', () => {
  test('maps Bedrock prompt cache tokens for non-stream responses', () => {
    const message: BedrockMessage = {
      role: 'assistant',
      content: [{ text: 'Hello' }],
    };
    const responseMetadata = {
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 25,
        cacheReadInputTokens: 7,
        cacheWriteInputTokens: 3,
      },
    } as Omit<ConverseResponse, 'output'>;

    const result = convertConverseMessageToLangChainMessage(
      message,
      responseMetadata
    );

    // Match the non-streaming decoder's cache-inclusive input accounting.
    expect(result.usage_metadata).toEqual({
      input_tokens: 20,
      output_tokens: 5,
      total_tokens: 25,
      input_token_details: {
        cache_read: 7,
        cache_creation: 3,
      },
    });
  });

  test('does not add input_token_details when Bedrock cache fields are absent', () => {
    const message: BedrockMessage = {
      role: 'assistant',
      content: [{ text: 'Hello' }],
    };
    const responseMetadata = {
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        totalTokens: 15,
      },
    } as Omit<ConverseResponse, 'output'>;

    const result = convertConverseMessageToLangChainMessage(
      message,
      responseMetadata
    );

    expect(result.usage_metadata?.input_token_details).toBeUndefined();
  });

  test('maps Bedrock prompt cache tokens for stream metadata', () => {
    const chunk = handleConverseStreamMetadata(
      {
        usage: {
          inputTokens: 20,
          outputTokens: 4,
          totalTokens: 39,
          cacheReadInputTokens: 9,
          cacheWriteInputTokens: 6,
        },
        metrics: { latencyMs: 100 },
      },
      { streamUsage: true }
    );
    const message = chunk.message as AIMessageChunk;

    // Streaming metadata keeps cache counts separate from input_tokens.
    expect(message.usage_metadata).toEqual({
      input_tokens: 20,
      output_tokens: 4,
      total_tokens: 39,
      input_token_details: {
        cache_read: 9,
        cache_creation: 6,
      },
    });
  });
});

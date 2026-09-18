// Inherited stream-events specs for the LibreChat Google fork
// (`CustomChatGoogleGenerativeAI` from `@/llm/google`), merged from two
// upstream-derived suites adapted to the fork's actual surface:
//
//   1. converter — Inherited from @langchain/google-genai@2.2.0
//      src/utils/stream_events.test.ts (tests `convertGoogleGenAIStream`,
//      the raw Gemini stream -> `ChatModelStreamEvent` converter).
//      `convertGoogleGenAIStream` is NOT exported from the package root and the
//      deep subpath (`dist/utils/stream_events.js`) is blocked by the package
//      `exports` map (`ERR_PACKAGE_PATH_NOT_EXPORTED`), so it cannot be imported
//      and unit-tested directly. Instead each converter case is routed through
//      the fork's inherited `_streamChatModelEvents` (which calls the converter
//      internally) and asserted on the collected `ChatModelStreamEvent[]`.
//
//   2. stream events — Inherited from @langchain/google-genai@2.2.0
//      src/tests/chat_models_stream_events.test.ts (tests
//      `ChatGoogleGenerativeAI.streamEvents()` typed sub-streams).
//      Without a native media port, the fork delegates `_streamChatModelEvents`
//      to the inherited Google implementation. The Google transport is mocked the way
//      `src/llm/google/llm.spec.ts` accesses it: by spying on the private
//      `client.generateContentStream`. The upstream suite used vitest custom
//      matchers (`toHaveStreamText` / `toHaveStreamReasoning` /
//      `toHaveStreamToolCalls` / `toHaveStreamUsage`) unavailable in jest; they
//      are re-expressed against the public `ChatModelStream` sub-streams
//      (`.text` / `.reasoning` / `.toolCalls` / `.usage`) that those matchers
//      wrap.

import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { describe, test, expect, jest, afterEach } from '@jest/globals';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { GenerateContentRequest } from '@google/generative-ai';
import type { BaseMessage } from '@langchain/core/messages';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';

type GeminiStreamChunk = Record<string, unknown>;

type TestGoogleGenAIClient = {
  systemInstruction?: unknown;
  generateContentStream: (
    request: GenerateContentRequest,
    options?: unknown
  ) => Promise<{ stream: AsyncIterable<GeminiStreamChunk> }>;
};

type StreamEventsModel = {
  _streamChatModelEvents: (
    messages: BaseMessage[],
    options: Record<string, unknown>
  ) => AsyncGenerator<ChatModelStreamEvent>;
};

function getTestClient(
  model: CustomChatGoogleGenerativeAI
): TestGoogleGenAIClient {
  return (model as unknown as { client: TestGoogleGenAIClient }).client;
}

function toStream(
  chunks: GeminiStreamChunk[]
): AsyncIterable<GeminiStreamChunk> {
  return (async function* () {
    for (const chunk of chunks) {
      yield chunk;
    }
  })();
}

function newModel(): CustomChatGoogleGenerativeAI {
  return new CustomChatGoogleGenerativeAI({
    apiKey: 'fake-key',
    model: 'gemini-2.0-flash',
  });
}

function mockGoogleGenAI(
  stream: AsyncIterable<GeminiStreamChunk>
): CustomChatGoogleGenerativeAI {
  const model = newModel();
  jest
    .spyOn(getTestClient(model), 'generateContentStream')
    .mockResolvedValue({ stream });
  return model;
}

function streamEvents(
  model: CustomChatGoogleGenerativeAI,
  messages: BaseMessage[] = [new HumanMessage('Hello')],
  options: Record<string, unknown> = {}
): AsyncGenerator<ChatModelStreamEvent> {
  return (model as unknown as StreamEventsModel)._streamChatModelEvents(
    messages,
    options
  );
}

async function collectEvents(
  chunks: GeminiStreamChunk[],
  options: Record<string, unknown> = {}
): Promise<ChatModelStreamEvent[]> {
  const events: ChatModelStreamEvent[] = [];
  for await (const event of streamEvents(
    mockGoogleGenAI(toStream(chunks)),
    [new HumanMessage('Hello')],
    options
  )) {
    events.push(event);
  }
  return events;
}

afterEach(() => {
  jest.restoreAllMocks();
});

// Routed from upstream `convertGoogleGenAIStream` unit tests: the converter is
// not importable from the fork, so each case drives the inherited
// `_streamChatModelEvents` (which invokes the converter) and asserts the
// emitted `ChatModelStreamEvent[]`.
describe('convertGoogleGenAIStream (via inherited _streamChatModelEvents)', () => {
  test('text-only streaming', async () => {
    const events = await collectEvents([
      { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
      { candidates: [{ content: { parts: [{ text: ' world' }] } }] },
    ]);

    expect(
      events.find((e) => e.event === 'content-block-finish')
    ).toMatchObject({
      content: { text: 'Hello world' },
    });
  });

  test('maps Gemini finish reasons', async () => {
    const lengthEvents = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Hello' }] },
            finishReason: 'MAX_TOKENS',
          },
        ],
      },
    ]);
    expect(
      lengthEvents.find((e) => e.event === 'message-finish')
    ).toMatchObject({ reason: 'length' });

    const filterEvents = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Hello' }] },
            finishReason: 'SAFETY',
          },
        ],
      },
    ]);
    expect(
      filterEvents.find((e) => e.event === 'message-finish')
    ).toMatchObject({ reason: 'content_filter' });
  });

  test('thought parts map to reasoning', async () => {
    const events = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Let me think', thought: true }] },
          },
        ],
      },
    ]);

    expect(
      events.find(
        (e) =>
          e.event === 'content-block-finish' &&
          (
            e as Extract<
              ChatModelStreamEvent,
              { event: 'content-block-finish' }
            >
          ).content.type === 'reasoning'
      )
    ).toMatchObject({
      content: { reasoning: 'Let me think' },
    });
  });
});

function geminiTextStream(): AsyncIterable<GeminiStreamChunk> {
  return toStream([
    { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
    { candidates: [{ content: { parts: [{ text: ' world' }] } }] },
  ]);
}

function geminiReasoningStream(): AsyncIterable<GeminiStreamChunk> {
  return toStream([
    {
      candidates: [
        { content: { parts: [{ text: 'Let me reason...', thought: true }] } },
      ],
    },
  ]);
}

function geminiToolStream(): AsyncIterable<GeminiStreamChunk> {
  return toStream([
    { candidates: [{ content: { parts: [{ text: 'Let me search.' }] } }] },
    {
      candidates: [
        {
          content: {
            parts: [
              {
                functionCall: {
                  name: 'web_search',
                  args: { query: 'weather' },
                },
              },
            ],
          },
        },
      ],
    },
  ]);
}

function geminiUsageStream(): AsyncIterable<GeminiStreamChunk> {
  return toStream([
    {
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 4,
        totalTokenCount: 14,
      },
      candidates: [{ content: { parts: [{ text: 'Hi' }] } }],
    },
  ]);
}

// Re-expressed from upstream `ChatGoogleGenerativeAI.streamEvents` describe.
// The vitest matchers (`toHaveStreamText` etc.) are unavailable in jest, so the
// cases assert the `ChatModelStream` sub-streams those matchers wrap.
describe('CustomChatGoogleGenerativeAI.streamEvents (sub-stream assertions)', () => {
  test('streams text', async () => {
    const stream = new ChatModelStream(
      streamEvents(mockGoogleGenAI(geminiTextStream())) as never
    );
    expect(await stream.text).toBe('Hello world');
  });

  test('streams reasoning', async () => {
    const stream = new ChatModelStream(
      streamEvents(mockGoogleGenAI(geminiReasoningStream())) as never
    );
    expect(await stream.reasoning).toBe('Let me reason...');
  });

  test('streams tool calls', async () => {
    const stream = new ChatModelStream(
      streamEvents(mockGoogleGenAI(geminiToolStream())) as never
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
      streamEvents(
        mockGoogleGenAI(geminiUsageStream()),
        [new HumanMessage('Hello')],
        { streamUsage: true }
      ) as never
    );
    expect(await stream.usage).toMatchObject({
      input_tokens: 10,
      output_tokens: 4,
      total_tokens: 14,
    });
  });

  test('passes system instructions per streamEvents request', async () => {
    const model = newModel();
    const generateContentStream = jest
      .spyOn(getTestClient(model), 'generateContentStream')
      .mockResolvedValue({ stream: geminiTextStream() });

    const stream = new ChatModelStream(
      streamEvents(model, [
        new SystemMessage('StreamV2 system instruction'),
        new HumanMessage('Hello'),
      ]) as never
    );
    expect(await stream.text).toBe('Hello world');

    const [[request]] = generateContentStream.mock.calls;
    expect(request.systemInstruction).toEqual({
      role: 'system',
      parts: [{ text: 'StreamV2 system instruction' }],
    });
    expect(request.contents).toEqual([
      { role: 'user', parts: [{ text: 'Hello' }] },
    ]);
    expect(getTestClient(model).systemInstruction).toBeUndefined();
  });

  test('passes system instructions per stream request', async () => {
    const model = newModel();
    const generateContentStream = jest
      .spyOn(getTestClient(model), 'generateContentStream')
      .mockResolvedValue({ stream: geminiTextStream() });

    const stream = await model.stream([
      new SystemMessage('Stream system instruction'),
      new HumanMessage('Hello'),
    ]);
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }

    const [[request]] = generateContentStream.mock.calls;
    expect(chunks.join('')).toBe('Hello world');
    expect(request.systemInstruction).toEqual({
      role: 'system',
      parts: [{ text: 'Stream system instruction' }],
    });
    expect(request.contents).toEqual([
      { role: 'user', parts: [{ text: 'Hello' }] },
    ]);
    expect(getTestClient(model).systemInstruction).toBeUndefined();
  });
});

/**
 * Mistral reasoning models (Mistral Large 4, GLM served by Mistral) answer an
 * OpenAI-compatible Chat Completions request with `content` as a list of typed
 * chunks instead of a string: `thinking` chunks carry the reasoning and `text`
 * chunks the answer. The chunk that closes the reasoning also carries the first
 * words of the answer, or the first tool call delta (the only one with its id
 * and name). LibreChat sends every custom endpoint through `ChatOpenAI`, so
 * these specs drive the real model over a stubbed `fetch` with that wire shape
 * (LibreChat-AI/LibreChat#17346).
 */
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { AIMessageChunk } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { AzureChatOpenAI } from '@/llm/openai';
import { ToolEndHandler } from '@/events';
import { Run } from '@/run';

type TextChunk = { type: 'text'; text: string };
type ThinkChunk = { type: 'thinking'; thinking: TextChunk[]; closed?: boolean };
type TypedContent = string | Array<TextChunk | ThinkChunk> | null;
type ToolCallDelta = {
  index: number;
  id?: string;
  type?: 'function';
  function: { name?: string; arguments: string };
};
type Delta = {
  role?: 'assistant';
  content?: TypedContent;
  tool_calls?: ToolCallDelta[];
};
type FetchArgs = Parameters<typeof globalThis.fetch>;

const MODEL = 'mistral-large-4';
const BASE_URL = 'https://api.mistral.ai/v1';
const REASONING = 'The user asks what a freelancer is. Answer briefly.';
const streamConfig = {
  configurable: { thread_id: 'mistral-typed-content' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};
const ANSWER = 'A freelancer is a self-employed worker.';

function think(text: string, closed?: boolean): ThinkChunk {
  return {
    type: 'thinking',
    thinking: [{ type: 'text', text }],
    ...(closed === true ? { closed } : {}),
  };
}

function text(value: string): TextChunk {
  return { type: 'text', text: value };
}

function streamChunk(delta: Delta, finishReason: string | null = null): object {
  return {
    id: 'mistral-stream',
    object: 'chat.completion.chunk',
    created: 0,
    model: MODEL,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

/** Reasoning in typed chunks; the closing chunk also opens the answer. */
const answerStream: object[] = [
  streamChunk({ role: 'assistant', content: [think('The user asks')] }),
  streamChunk({ content: [think(' what a freelancer is.')] }),
  streamChunk({ content: [think(' Answer briefly.', true), text('A freel')] }),
  streamChunk({ content: 'ancer is a self-employed worker.' }),
  streamChunk({ content: '' }, 'stop'),
];

function toSse(chunks: object[]): Response {
  const body = chunks
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .concat('data: [DONE]\n\n')
    .join('');
  return new Response(body, {
    headers: { 'content-type': 'text/event-stream' },
  });
}

function createFetch(
  responses: Array<(stream: boolean) => Response>
): jest.Mock<ReturnType<typeof globalThis.fetch>, FetchArgs> {
  let call = 0;
  return jest.fn<ReturnType<typeof globalThis.fetch>, FetchArgs>(
    async (_input, init) => {
      const request = JSON.parse(String(init?.body)) as { stream?: boolean };
      const respond = responses[Math.min(call, responses.length - 1)];
      call += 1;
      return respond(request.stream === true);
    }
  );
}

function llmConfig(
  fetch: typeof globalThis.fetch,
  overrides: Partial<t.LLMConfig> = {}
): t.LLMConfig {
  return {
    provider: Providers.OPENAI,
    model: MODEL,
    apiKey: 'test-key',
    streaming: true,
    streamUsage: false,
    maxRetries: 0,
    configuration: { baseURL: BASE_URL, fetch },
    ...overrides,
  } as t.LLMConfig;
}

async function runTurn(
  config: t.LLMConfig,
  tools: t.GraphTools = []
): Promise<{
  thoughts: string;
  answer: string;
  contentTypes: Array<string | undefined>;
  messages: ReturnType<Run<t.IState>['getRunMessages']>;
}> {
  const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
  const messageDeltas: t.MessageDeltaEvent[] = [];
  const { contentParts, aggregateContent } = createContentAggregator();
  const run = await Run.create<t.IState>({
    runId: 'mistral-typed-content',
    graphConfig: { type: 'standard', llmConfig: config, tools },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.ON_RUN_STEP]: {
        handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData) =>
          aggregateContent({ event, data: data as t.RunStep }),
      },
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (
          event: GraphEvents.ON_RUN_STEP_COMPLETED,
          data: t.StreamEventData
        ) =>
          aggregateContent({
            event,
            data: data as unknown as { result: t.ToolEndEvent },
          }),
      },
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (
          event: GraphEvents.ON_MESSAGE_DELTA,
          data: t.StreamEventData
        ) => {
          messageDeltas.push(data as t.MessageDeltaEvent);
          aggregateContent({ event, data: data as t.MessageDeltaEvent });
        },
      },
      [GraphEvents.ON_REASONING_DELTA]: {
        handle: (
          event: GraphEvents.ON_REASONING_DELTA,
          data: t.StreamEventData
        ) => {
          reasoningDeltas.push(data as t.ReasoningDeltaEvent);
          aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
        },
      },
    },
  });
  await run.processStream(
    { messages: [new HumanMessage('What is a freelancer?')] },
    streamConfig
  );
  return {
    thoughts: reasoningDeltas
      .flatMap((delta) => delta.delta.content ?? [])
      .map((part) => (part as { think?: string }).think ?? '')
      .join(''),
    answer: messageDeltas
      .flatMap((delta) => delta.delta.content ?? [])
      .map((part) => (part as { text?: string }).text ?? '')
      .join(''),
    contentTypes: contentParts.map((part) => part?.type),
    messages: run.getRunMessages(),
  };
}

describe('Mistral typed content on Chat Completions', () => {
  it.each([false, true])(
    'keeps the first words of a streamed answer and the reasoning they close (includeReasoningContent: %p) @scenario:mistral-stream-keeps-answer-start',
    async (includeReasoningContent) => {
      const fetch = createFetch([() => toSse(answerStream)]);

      const { thoughts, answer, contentTypes } = await runTurn(
        llmConfig(fetch, { includeReasoningContent } as Partial<t.LLMConfig>)
      );

      expect(answer).toBe(ANSWER);
      expect(thoughts).toBe(REASONING);
      expect(contentTypes).toEqual([ContentTypes.THINK, ContentTypes.TEXT]);
    }
  );

  it('keeps the first words when the closing chunk carries only text @scenario:mistral-stream-keeps-answer-start', async () => {
    const fetch = createFetch([
      () =>
        toSse([
          streamChunk({ role: 'assistant', content: [think(REASONING)] }),
          streamChunk({ content: [text('A freel')] }),
          streamChunk({ content: 'ancer is a self-employed worker.' }, 'stop'),
        ]),
    ]);

    const { thoughts, answer, contentTypes } = await runTurn(llmConfig(fetch));

    expect(answer).toBe(ANSWER);
    expect(thoughts).toBe(REASONING);
    expect(contentTypes).toEqual([ContentTypes.THINK, ContentTypes.TEXT]);
  });

  it('runs a tool call whose first delta arrives with the reasoning @scenario:mistral-stream-keeps-tool-call', async () => {
    let pageArgs: unknown;
    const readPage = new DynamicStructuredTool({
      name: 'read_page',
      description: 'Read a page by id',
      schema: {
        type: 'object',
        properties: { page_id: { type: 'string' } },
        required: ['page_id'],
      },
      func: async (input: unknown): Promise<string> => {
        pageArgs = input;
        return 'The page says hello.';
      },
    });
    const toolStream = [
      streamChunk({ role: 'assistant', content: [think('Read the page.')] }),
      streamChunk({
        content: [think(' Calling the tool.', true)],
        tool_calls: [
          {
            index: 0,
            id: 'call_mistral_1',
            type: 'function',
            function: { name: 'read_page', arguments: '{"page_id": "' },
          },
        ],
      }),
      streamChunk(
        { tool_calls: [{ index: 0, function: { arguments: '42"}' } }] },
        'tool_calls'
      ),
    ];
    const finalStream = [
      streamChunk({
        role: 'assistant',
        content: [think('Got it.', true), text('The page')],
      }),
      streamChunk({ content: ' says hello.' }, 'stop'),
    ];
    const fetch = createFetch([
      () => toSse(toolStream),
      () => toSse(finalStream),
    ]);

    const { thoughts, answer, messages } = await runTurn(llmConfig(fetch), [
      readPage,
    ]);

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(pageArgs).toEqual({ page_id: '42' });
    expect(messages?.some((message) => message.getType() === 'tool')).toBe(
      true
    );
    expect(thoughts).toBe('Read the page. Calling the tool.Got it.');
    expect(answer).toBe('The page says hello.');
  });

  it('keeps a non-streamed answer and its reasoning @scenario:mistral-non-streaming-keeps-answer', async () => {
    const fetch = createFetch([
      () =>
        Response.json({
          id: 'mistral-completion',
          object: 'chat.completion',
          created: 0,
          model: MODEL,
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: [think(REASONING, true), text(ANSWER)],
              },
              finish_reason: 'stop',
            },
          ],
          usage: { prompt_tokens: 9, completion_tokens: 21, total_tokens: 30 },
        }),
    ]);

    const { thoughts, answer, messages } = await runTurn(
      llmConfig(fetch, { disableStreaming: true } as Partial<t.LLMConfig>)
    );

    expect(answer).toBe(ANSWER);
    expect(thoughts).toBe(REASONING);
    const reply = messages?.find(
      (message): message is AIMessage => message.getType() === 'ai'
    );
    expect(reply?.content).toBe(ANSWER);
  });

  it('keeps the first words on Azure Chat Completions @scenario:mistral-azure-stream-keeps-answer-start', async () => {
    const fetch = createFetch([() => toSse(answerStream)]);
    const fields = {
      streaming: true,
      maxRetries: 0,
      azureOpenAIApiKey: 'test-key',
      azureOpenAIApiInstanceName: 'mistral-instance',
      azureOpenAIApiDeploymentName: MODEL,
      azureOpenAIApiVersion: '2025-04-01-preview',
      configuration: { fetch },
    };
    const model = new AzureChatOpenAI(fields);

    let message: AIMessageChunk | undefined;
    for await (const chunk of await model.stream('What is a freelancer?')) {
      message = message == null ? chunk : message.concat(chunk);
    }

    expect(message?.content).toBe(ANSWER);
    expect(message?.additional_kwargs.reasoning_content).toBe(REASONING);
  });
});

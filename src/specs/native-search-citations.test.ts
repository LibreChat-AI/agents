import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { ChatVertexAI } from '@/llm/vertexai';
import { ChatOpenAI } from '@/llm/openai';
import { Run } from '@/run';

const A = 'https://www.wetter.example/osnabrueck';
const B = 'https://fog.example/warning';

type Collected = {
  handlers: Record<string, t.EventHandler>;
  contentParts: Array<t.MessageContentComplex | undefined>;
  deltas: t.MessageContentComplex[];
  artifacts: Record<string, unknown>[];
};

function collect(): Collected {
  const { contentParts, aggregateContent } = createContentAggregator();
  const deltas: t.MessageContentComplex[] = [];
  const artifacts: Record<string, unknown>[] = [];
  const aggregate = {
    handle: (event: string, data: t.StreamEventData): void => {
      aggregateContent({ event, data } as Parameters<
        typeof aggregateContent
      >[0]);
    },
  };
  return {
    contentParts,
    deltas,
    artifacts,
    handlers: {
      [GraphEvents.ON_RUN_STEP]: aggregate,
      [GraphEvents.ON_RUN_STEP_DELTA]: aggregate,
      [GraphEvents.ON_RUN_STEP_COMPLETED]: aggregate,
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (event: string, data: t.StreamEventData): void => {
          deltas.push(...((data as t.MessageDeltaEvent).delta.content ?? []));
          aggregate.handle(event, data);
        },
      },
      [GraphEvents.TOOL_END]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          const output = (data as t.ToolEndData).output;
          if (output instanceof ToolMessage) {
            artifacts.push(output.artifact as Record<string, unknown>);
          }
        },
      },
    },
  };
}

function getText(collected: Collected): string {
  return collected.contentParts
    .flatMap((part) =>
      part?.type === 'text' && typeof part.text === 'string' ? [part.text] : []
    )
    .join('');
}

function getStreamedText(collected: Collected): string {
  return collected.deltas
    .map((part) => {
      const { text } = part as { text?: unknown };
      return typeof text === 'string' ? text : '';
    })
    .join('');
}

function eventStream(frames: object[], named = true): Response {
  return new Response(
    frames
      .map((frame) => {
        const data = `data: ${JSON.stringify(frame)}\n\n`;
        return named
          ? `event: ${(frame as { type: string }).type}\n${data}`
          : data;
      })
      .join(''),
    { headers: { 'content-type': 'text/event-stream' } }
  );
}

async function stream(run: Run<t.IState>, thread_id: string): Promise<void> {
  await run.processStream(
    { messages: [new HumanMessage('What is the weather today?')] },
    { version: 'v2', configurable: { thread_id } }
  );
}

describe('Anthropic web search citations', () => {
  type Block =
    | { type: 'server_tool_use'; id: string; name: string; input: object }
    | { type: 'web_search_tool_result'; tool_use_id: string; content: object[] }
    | { type: 'text'; text: string; citations?: string[] };

  /** Real stream order: a cited block's citations arrive before its text. */
  function frames(blocks: Block[]): object[] {
    return [
      {
        type: 'message_start',
        message: {
          id: 'msg_cited',
          type: 'message',
          role: 'assistant',
          model: 'claude-sonnet-4-5',
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 5, output_tokens: 0 },
        },
      },
      ...blocks.flatMap((block, index): object[] => {
        if (block.type === 'server_tool_use') {
          return [
            {
              type: 'content_block_start',
              index,
              content_block: { ...block, input: {} },
            },
            {
              type: 'content_block_delta',
              index,
              delta: {
                type: 'input_json_delta',
                partial_json: JSON.stringify(block.input),
              },
            },
            { type: 'content_block_stop', index },
          ];
        }
        if (block.type === 'web_search_tool_result') {
          return [
            { type: 'content_block_start', index, content_block: block },
            { type: 'content_block_stop', index },
          ];
        }
        return [
          {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'text',
              text: '',
              ...(block.citations ? { citations: [] } : {}),
            },
          },
          ...(block.citations ?? []).map((url) => ({
            type: 'content_block_delta',
            index,
            delta: {
              type: 'citations_delta',
              citation: {
                type: 'web_search_result_location',
                url,
                title: url,
                cited_text: 'quoted source text',
                encrypted_index: 'opaque',
              },
            },
          })),
          {
            type: 'content_block_delta',
            index,
            delta: { type: 'text_delta', text: block.text },
          },
          { type: 'content_block_stop', index },
        ];
      }),
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: 20 },
      },
      { type: 'message_stop' },
    ];
  }

  it('streams an anchor after each cited block and keeps it in the saved text @scenario:anthropic-cited-blocks-stream-anchors', async () => {
    const response = eventStream(
      frames([
        {
          type: 'server_tool_use',
          id: 'srvtoolu_weather',
          name: 'web_search',
          input: { query: 'Wetter Osnabrück heute' },
        },
        {
          type: 'web_search_tool_result',
          tool_use_id: 'srvtoolu_weather',
          content: [A, B].map((url) => ({
            type: 'web_search_result',
            url,
            title: url,
            encrypted_content: 'opaque',
            page_age: null,
          })),
        },
        { type: 'text', text: 'Today it is dry in Osnabrück', citations: [A] },
        { type: 'text', text: '. ' },
        { type: 'text', text: 'There is a fog warning', citations: [B, A] },
        { type: 'text', text: '.' },
      ])
    );
    const collected = collect();
    const run = await Run.create<t.IState>({
      runId: 'anthropic-native-citations',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          model: 'claude-sonnet-4-5',
          streaming: true,
          streamUsage: false,
          anthropicApiKey: 'test-key',
          clientOptions: { fetch: async () => response },
          maxRetries: 0,
          _lc_stream_delay: 0,
        },
        tools: [{ type: 'web_search_20250305', name: 'web_search' }],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: collected.handlers,
    });
    await stream(run, 'anthropic-native-citations');

    const expected =
      'Today it is dry in Osnabrück\ue202turn0search0. There is a fog warning\ue200\ue202turn0search1\ue202turn0search0\ue201.';
    expect(getText(collected)).toBe(expected);
    expect(getStreamedText(collected)).toBe(expected);
    expect(collected.artifacts).toEqual([
      {
        web_search: expect.objectContaining({
          turn: 0,
          organic: [
            expect.objectContaining({ link: A, position: 1 }),
            expect.objectContaining({ link: B, position: 2 }),
          ],
        }),
      },
    ]);
  });
});

describe('Gemini grounding citations', () => {
  const first = 'Die Mensa öffnet um 11 Uhr.';
  const second = 'Heute gibt es Pasta.';
  const bytes = (value: string): number => Buffer.byteLength(value);

  afterEach(() => jest.restoreAllMocks());

  const frames = [
    {
      candidates: [
        {
          content: { role: 'model', parts: [{ text: `${first} ` }] },
          index: 0,
        },
      ],
    },
    {
      candidates: [
        {
          content: { role: 'model', parts: [{ text: second }] },
          finishReason: 'STOP',
          index: 0,
          groundingMetadata: {
            webSearchQueries: ['mensa schlossgarten heute'],
            groundingChunks: [
              { web: { uri: A, title: 'studentenwerk.example' } },
              { web: { uri: B, title: 'mensa.example' } },
            ],
            groundingSupports: [
              {
                segment: {
                  startIndex: 0,
                  endIndex: bytes(first),
                  text: first,
                },
                groundingChunkIndices: [0],
              },
              {
                segment: {
                  startIndex: bytes(`${first} `),
                  endIndex: bytes(`${first} ${second}`),
                  text: second,
                },
                groundingChunkIndices: [0, 1],
              },
            ],
          },
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 12,
        totalTokenCount: 22,
      },
    },
  ];

  function respond(vertex: boolean): void {
    jest.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      vertex
        ? new Response(JSON.stringify(frames), {
          headers: { 'content-type': 'application/json' },
        })
        : eventStream(frames, false)
    );
  }

  async function run(
    runId: string,
    { tools, vertex = false }: { tools?: t.GraphTools; vertex?: boolean } = {}
  ): Promise<Collected> {
    respond(vertex);
    const collected = collect();
    const graphRun = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: vertex ? Providers.VERTEXAI : Providers.GOOGLE },
        tools,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: collected.handlers,
    });
    graphRun.Graph!.overrideModel = vertex
      ? new ChatVertexAI({
        model: 'gemini-2.5-flash',
        apiKey: 'synthetic-test-key',
        location: 'global',
        maxRetries: 0,
      })
      : new CustomChatGoogleGenerativeAI({
        model: 'gemini-2.5-flash',
        apiKey: 'synthetic-test-key',
        streaming: true,
      });
    await stream(graphRun, runId);
    return collected;
  }

  function expectGrounded(collected: Collected): void {
    expect(getText(collected)).toBe(
      `${first}\ue202turn0search0 ${second}\ue200\ue202turn0search0\ue202turn0search1\ue201`
    );
    expect(collected.artifacts).toEqual([
      {
        web_search: {
          turn: 0,
          organic: [
            {
              link: A,
              position: 1,
              title: 'studentenwerk.example',
              attribution: 'studentenwerk.example',
            },
            {
              link: B,
              position: 2,
              title: 'mensa.example',
              attribution: 'mensa.example',
            },
          ],
        },
      },
    ]);
  }

  it('adds the grounding sources and anchors each grounded segment @scenario:gemini-grounding-adds-sources-and-anchors', async () => {
    expectGrounded(await run('gemini-grounding-citations'));
  });

  it('reads Vertex AI grounding from the response metadata @scenario:vertex-grounding-adds-sources-and-anchors', async () => {
    expectGrounded(await run('vertex-grounding-citations', { vertex: true }));
  });

  it('leaves the answer alone when the SDK search tool numbers the turns @scenario:gemini-grounding-skips-own-search-agents', async () => {
    const webSearch = new DynamicStructuredTool({
      name: 'web_search',
      description: 'Search the web.',
      schema: z.object({ query: z.string() }),
      func: async (): Promise<string> => 'unused',
    });
    const collected = await run('gemini-grounding-own-search', {
      tools: [webSearch],
    });

    expect(getText(collected)).toBe(`${first} ${second}`);
    expect(collected.artifacts).toEqual([]);
  });
});

describe('OpenAI url_citation', () => {
  const url = 'https://weather.example/rome?utm_source=openai';
  const link = `([weather.example](${url}))`;
  const lead = 'It is sunny in Rome today ';
  const text = `${lead}${link}.`;
  const response = {
    id: 'resp_cited',
    created_at: 1,
    object: 'response',
    model: 'synthetic-model',
    status: 'completed',
    error: null,
    output: [
      {
        id: 'ws_weather',
        type: 'web_search_call',
        status: 'completed',
        action: { type: 'search', query: 'weather rome today' },
      },
      {
        id: 'msg_cited',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text,
            annotations: [
              {
                type: 'url_citation',
                start_index: lead.length,
                end_index: lead.length + link.length,
                url,
                title: 'Rome weather',
              },
            ],
          },
        ],
      },
    ],
    incomplete_details: null,
    instructions: null,
    metadata: {},
    tools: [],
    tool_choice: 'auto',
    text: { format: { type: 'text' } },
    parallel_tool_calls: true,
    temperature: 1,
    top_p: 1,
    truncation: 'disabled',
    store: false,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens_details: { reasoning_tokens: 0 },
    },
  };
  const message = { output_index: 1, item_id: 'msg_cited', content_index: 0 };
  const frames = [
    {
      type: 'response.created',
      response: { ...response, status: 'in_progress', output: [] },
    },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: { ...response.output[0], status: 'in_progress' },
    },
    {
      type: 'response.output_item.done',
      output_index: 0,
      item: response.output[0],
    },
    {
      type: 'response.output_item.added',
      output_index: 1,
      item: { ...response.output[1], content: [], status: 'in_progress' },
    },
    { type: 'response.output_text.delta', ...message, delta: lead },
    { type: 'response.output_text.delta', ...message, delta: link },
    {
      type: 'response.output_text.annotation.added',
      ...message,
      annotation_index: 0,
      annotation: response.output[1].content![0].annotations[0],
    },
    { type: 'response.output_text.delta', ...message, delta: '.' },
    { type: 'response.completed', response },
  ].map((frame, sequence_number) => ({ ...frame, sequence_number }));

  function model(): ChatOpenAI {
    return new ChatOpenAI({
      model: 'synthetic-model',
      apiKey: 'synthetic-test-key',
      useResponsesApi: true,
      streaming: true,
      configuration: {
        apiKey: 'synthetic-test-key',
        fetch: async () => eventStream(frames),
      },
    });
  }

  it('replaces the cited link with an anchor to the new search source @scenario:openai-url-citation-anchored', async () => {
    const collected = collect();
    const run = await Run.create<t.IState>({
      runId: 'openai-url-citations',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: collected.handlers,
    });
    run.Graph!.overrideModel = model();
    await stream(run, 'openai-url-citations');

    expect(getText(collected)).toBe(`${lead}\ue202turn0search0.`);
    expect(collected.artifacts).toEqual([
      {
        web_search: {
          turn: 0,
          organic: [
            {
              link: url,
              position: 1,
              title: 'Rome weather',
              attribution: 'weather.example',
            },
          ],
        },
      },
    ]);
  });

  it('leaves the link alone when the SDK search tool numbers the turns @scenario:openai-url-citation-skips-own-search-agents', async () => {
    const collected = collect();
    const webSearch = new DynamicStructuredTool({
      name: 'web_search',
      description: 'Search the web.',
      schema: z.object({ query: z.string() }),
      func: async (): Promise<string> => 'unused',
    });
    const run = await Run.create<t.IState>({
      runId: 'openai-url-citations-own-search',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        tools: [webSearch],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: collected.handlers,
    });
    run.Graph!.overrideModel = model();
    await stream(run, 'openai-url-citations-own-search');

    expect(getText(collected)).toBe(text);
    expect(collected.artifacts).toEqual([]);
  });
});

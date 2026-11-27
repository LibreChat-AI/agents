import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { CustomAnthropicCallOptions } from './index';
import { convertInjectedMessages } from '@/messages/injected';
import { CustomAnthropic } from './index';
import { Providers } from '@/common';
import { Run } from '@/run';

const modelName = 'claude-opus-4-7';
const container = {
  id: 'container_dynamic_search',
  expires_at: '2026-12-01T00:00:00Z',
};
const caller = {
  type: 'code_execution_20260120',
  tool_id: 'srvtoolu_code',
} as const;
const serverBlocks = [
  {
    type: 'server_tool_use',
    id: caller.tool_id,
    name: 'code_execution',
    input: { code: 'search()' },
    caller: { type: 'direct' },
  },
  {
    type: 'server_tool_use',
    id: 'srvtoolu_search',
    name: 'web_search',
    input: { query: 'test' },
    caller,
  },
  {
    type: 'web_search_tool_result',
    tool_use_id: 'srvtoolu_search',
    content: [
      {
        type: 'web_search_result',
        title: 'Example',
        url: 'https://example.com',
        encrypted_content: 'opaque',
        page_age: null,
      },
    ],
    caller,
  },
  {
    type: 'code_execution_tool_result',
    tool_use_id: caller.tool_id,
    content: {
      type: 'encrypted_code_execution_result',
      content: [],
      encrypted_stdout: 'opaque stdout',
      stderr: '',
      return_code: 0,
    },
  },
];
const nativeTools = [
  { type: 'web_search_20260209', name: 'web_search' },
  { type: 'code_execution_20260120', name: 'code_execution' },
] as const;

function clientCall(n: number) {
  return {
    type: 'tool_use',
    id: `toolu_lookup_${n}`,
    name: 'lookup',
    input: { n },
    caller: n === 1 ? caller : { type: 'direct' },
  };
}

function sseResponse({
  id,
  blocks,
  startContainer = null,
  deltaContainer = null,
  stopReason = 'end_turn',
}: {
  id: number;
  blocks: Array<{ type: string; input?: object; [key: string]: unknown }>;
  startContainer?: typeof container | null;
  deltaContainer?: typeof container | null;
  stopReason?: 'tool_use' | 'end_turn';
}): Response {
  const frames = [
    {
      type: 'message_start',
      message: {
        id: `msg_${id}`,
        type: 'message',
        role: 'assistant',
        model: modelName,
        content: [],
        container: startContainer,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 5, output_tokens: 0 },
      },
    },
    ...blocks.flatMap((block, index) => [
      {
        type: 'content_block_start',
        index,
        content_block: block.input == null ? block : { ...block, input: {} },
      },
      ...(block.input == null
        ? []
        : [
          {
            type: 'content_block_delta',
            index,
            delta: {
              type: 'input_json_delta',
              partial_json: JSON.stringify(block.input),
            },
          },
        ]),
      { type: 'content_block_stop', index },
    ]),
    {
      type: 'message_delta',
      delta: {
        container: deltaContainer,
        stop_reason: stopReason,
        stop_sequence: null,
      },
      usage: { output_tokens: 4 },
    },
    { type: 'message_stop' },
  ];
  return new Response(
    frames
      .map(
        (frame) => `event: ${frame.type}\ndata: ${JSON.stringify(frame)}\n\n`
      )
      .join(''),
    { headers: { 'content-type': 'text/event-stream' } }
  );
}

function mockTransport(responses: Response[]) {
  const requests: Array<{
    container?: string;
    messages: Array<{ role: string; content: object[] | string }>;
    tools: Array<{ name: string; type?: string }>;
  }> = [];
  const fetch: typeof globalThis.fetch = async (_url, init) => {
    if (typeof init?.body !== 'string') {
      throw new Error('Expected an Anthropic JSON request');
    }
    const response = responses.at(requests.length);
    if (response == null) {
      throw new Error('Unexpected Anthropic request');
    }
    requests.push(JSON.parse(init.body));
    return response;
  };
  return { fetch, requests };
}

it('replays native search and code execution across two real client-tool rounds, then starts a fresh user turn', async () => {
  const { fetch, requests } = mockTransport([
    sseResponse({
      id: 1,
      blocks: [...serverBlocks, clientCall(1)],
      startContainer: container,
      deltaContainer: container,
      stopReason: 'tool_use',
    }),
    sseResponse({ id: 2, blocks: [clientCall(2)], stopReason: 'tool_use' }),
    sseResponse({ id: 3, blocks: [{ type: 'text', text: 'First answer' }] }),
    sseResponse({ id: 4, blocks: [{ type: 'text', text: 'Next answer' }] }),
  ]);
  const executed: number[] = [];
  const lookup = new DynamicStructuredTool({
    name: 'lookup',
    description: 'Look up a number',
    schema: {
      type: 'object',
      properties: { n: { type: 'integer' } },
      required: ['n'],
    },
    func: async ({ n }: { n: number }): Promise<string> => {
      executed.push(n);
      return `number ${n}`;
    },
  });
  const createRun = () =>
    Run.create({
      runId: 'anthropic-server-tool-continuation',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          model: modelName,
          streaming: true,
          streamUsage: false,
          anthropicApiKey: 'test-key',
          clientOptions: { fetch },
          maxRetries: 0,
          _lc_stream_delay: 0,
        },
        tools: [...nativeTools, lookup],
      },
    });
  const question = new HumanMessage('Search and look up two numbers');
  const run = await createRun();
  await run.processStream({ messages: [question] }, { version: 'v2' });
  expect(executed).toEqual([1, 2]);
  const history = run.getRunMessages();
  expect(history?.at(-1)?.content).toEqual(
    expect.arrayContaining([{ type: 'text', index: 0, text: 'First answer' }])
  );

  const nextRun = await createRun();
  await nextRun.processStream(
    {
      messages: [
        question,
        ...(history ?? []),
        new HumanMessage('New question'),
      ],
    },
    { version: 'v2' }
  );

  expect(requests.map((request) => request.container)).toEqual([
    undefined,
    container.id,
    container.id,
    undefined,
  ]);
  expect(requests[0].tools).toEqual(
    expect.arrayContaining(nativeTools.slice())
  );
  for (const request of requests.slice(1)) {
    expect(request.messages[1].content).toEqual([
      ...serverBlocks,
      clientCall(1),
    ]);
    expect(request.messages[2].content).toEqual(
      expect.arrayContaining([
        {
          type: 'tool_result',
          tool_use_id: 'toolu_lookup_1',
          content: 'number 1',
        },
      ])
    );
  }
  expect(requests[2].messages[4].content).toEqual(
    expect.arrayContaining([
      {
        type: 'tool_result',
        tool_use_id: 'toolu_lookup_2',
        content: 'number 2',
      },
    ])
  );
});

it('replays native event output and uses the latest container across injected messages unless explicitly overridden', async () => {
  const newer = { ...container, id: 'container_latest' };
  const text = { type: 'text', text: 'Finished' };
  const { fetch, requests } = mockTransport([
    sseResponse({
      id: 1,
      blocks: [...serverBlocks, clientCall(1)],
      startContainer: container,
      deltaContainer: newer,
      stopReason: 'tool_use',
    }),
    ...[2, 3, 4].map((id) => sseResponse({ id, blocks: [text] })),
  ]);
  const model = new CustomAnthropic({
    model: modelName,
    anthropicApiKey: 'test-key',
    clientOptions: { fetch },
    maxRetries: 0,
  });
  const question = new HumanMessage('Search');
  const output = await model.streamEvents([question], {
    tools: [...nativeTools],
  }).output;
  const history = [
    question,
    output,
    new ToolMessage({ tool_call_id: 'toolu_lookup_1', content: 'number 1' }),
    ...convertInjectedMessages([
      { role: 'system', content: 'Continue searching' },
    ]),
    new HumanMessage({
      content: 'Routing cue',
      additional_kwargs: { source: 'routing' },
    }),
  ];
  await model.streamEvents(history).output;
  const explicit: CustomAnthropicCallOptions = {
    container: 'container_explicit',
  };
  await model.streamEvents(history, { ...explicit, tools: [...nativeTools] })
    .output;
  await model.streamEvents([
    ...history,
    new HumanMessage({
      content: 'New turn',
      additional_kwargs: { isMeta: true, source: 'custom' },
    }),
  ]).output;

  expect(requests.map((request) => request.container)).toEqual([
    undefined,
    newer.id,
    explicit.container,
    undefined,
  ]);
  expect(requests[1].messages[1].content).toEqual([
    ...serverBlocks,
    clientCall(1),
  ]);
});

it('uses the legacy stream delta container even without tools or streamed usage', async () => {
  const newer = { ...container, id: 'container_delta' };
  const text = { type: 'text', text: 'Finished' };
  const { fetch, requests } = mockTransport([
    sseResponse({
      id: 1,
      blocks: [text],
      startContainer: container,
      deltaContainer: newer,
    }),
    sseResponse({ id: 2, blocks: [text] }),
    sseResponse({ id: 3, blocks: [text] }),
  ]);
  const model = new CustomAnthropic({
    model: modelName,
    streaming: true,
    streamUsage: false,
    _lc_stream_delay: 0,
    anthropicApiKey: 'test-key',
    clientOptions: { fetch },
    maxRetries: 0,
  });
  const history = [
    new HumanMessage('Search'),
    await model.invoke('Search'),
    new AIMessage({
      content: 'Other provider',
      response_metadata: {
        model_provider: 'openai',
        container: { id: 'foreign' },
      },
    }),
    ...convertInjectedMessages([{ role: 'system', content: 'Continue' }]),
  ];
  await model.invoke(history);
  const explicit: CustomAnthropicCallOptions = {
    container: 'container_explicit',
  };
  await model.invoke(history, explicit);
  expect(requests.map((request) => request.container)).toEqual([
    undefined,
    newer.id,
    explicit.container,
  ]);
});

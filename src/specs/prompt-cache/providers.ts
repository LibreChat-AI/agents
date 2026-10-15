import type * as t from '@/types';
import type { CapturedRequest, Json } from './harness';
import { CAPTURED } from './harness';
import { Providers } from '@/common';

type JsonObject = { [key: string]: Json };
type Restore = () => void;

/** A tool call the faked model reply makes. */
export type FakeToolCall = {
  id: string;
  name: string;
  args: { [key: string]: Json };
};

/**
 * Receives every request body. Returning a tool call answers the request
 * with a streamed reply making that call, in the provider's own wire format;
 * returning nothing aborts the request after capture.
 */
export type RecordRequest = (body: Json) => FakeToolCall | void;

/**
 * One provider configuration the harness drives. `intercept` installs a
 * capture point that records the serialized request and aborts it before it
 * leaves the process; `reduce` maps that request to cache order.
 */
export interface ProviderCase {
  name: string;
  provider: Providers;
  clientOptions: () => t.ClientOptions;
  intercept: (agent: t.AgentInputs, record: RecordRequest) => Restore;
  reduce: (body: Json) => CapturedRequest;
  /** Explicit breakpoint family and the provider's limit per request. */
  breakpoints?: { pattern: RegExp; max: number };
  /** Prompt caching moves `additional_instructions` ahead of the newest human message. */
  movesDynamicPart?: boolean;
  /** Provider rejects consecutive messages of the same role. */
  strictAlternation?: boolean;
}

const API_KEY = 'test-key';
const ANTHROPIC_BREAKPOINTS = { pattern: /"cache_control":/g, max: 4 };
const BEDROCK_BREAKPOINTS = { pattern: /"cachePoint":/g, max: 4 };

function asObject(value: Json | undefined): JsonObject {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value
    : {};
}

function asArray(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}

function sse(events: Array<{ event?: string; data: Json | string }>): Response {
  const text = events
    .map(({ event, data }) =>
      [
        ...(event != null ? [`event: ${event}`] : []),
        `data: ${typeof data === 'string' ? data : JSON.stringify(data)}`,
        '',
        '',
      ].join('\n')
    )
    .join('');
  return new Response(text, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

/** Argument JSON split in two fragments, as providers stream it. */
function argumentFragments(call: FakeToolCall): [string, string] {
  const json = JSON.stringify(call.args);
  const middle = Math.floor(json.length / 2);
  return [json.slice(0, middle), json.slice(middle)];
}

function chatCompletionsToolCall(call: FakeToolCall): Response {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${call.id}`,
    object: 'chat.completion.chunk',
    created: 0,
    model: 'harness',
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const [head, tail] = argumentFragments(call);
  return sse([
    {
      data: chunk(
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              index: 0,
              id: call.id,
              type: 'function',
              function: { name: call.name, arguments: head },
            },
          ],
        },
        null
      ),
    },
    {
      data: chunk(
        { tool_calls: [{ index: 0, function: { arguments: tail } }] },
        null
      ),
    },
    { data: chunk({}, 'tool_calls') },
    { data: '[DONE]' },
  ]);
}

function responsesToolCall(call: FakeToolCall): Response {
  const [head, tail] = argumentFragments(call);
  const item = {
    type: 'function_call',
    id: `fc_${call.id}`,
    call_id: call.id,
    name: call.name,
    arguments: head + tail,
    status: 'completed',
  };
  const response = {
    id: `resp_${call.id}`,
    object: 'response',
    model: 'gpt-5',
    output: [] as Json[],
    status: 'in_progress',
  };
  const events: Array<{ type: string; [key: string]: Json }> = [
    { type: 'response.created', response },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: { ...item, arguments: '', status: 'in_progress' },
    },
    ...[head, tail].map((delta) => ({
      type: 'response.function_call_arguments.delta',
      item_id: item.id,
      output_index: 0,
      delta,
    })),
    {
      type: 'response.function_call_arguments.done',
      item_id: item.id,
      output_index: 0,
      arguments: item.arguments,
    },
    { type: 'response.output_item.done', output_index: 0, item },
    {
      type: 'response.completed',
      response: {
        ...response,
        status: 'completed',
        output: [item],
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      },
    },
  ];
  return sse(
    events.map((data, sequence_number) => ({
      event: data.type,
      data: { ...data, sequence_number },
    }))
  );
}

function anthropicToolCall(call: FakeToolCall): Response {
  const [head, tail] = argumentFragments(call);
  const events: Array<{ type: string; [key: string]: Json }> = [
    {
      type: 'message_start',
      message: {
        id: `msg_${call.id}`,
        type: 'message',
        role: 'assistant',
        model: 'claude-sonnet-4-5',
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 1 },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
    },
    ...[head, tail].map((partial_json) => ({
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json },
    })),
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'tool_use', stop_sequence: null },
      usage: { output_tokens: 5 },
    },
    { type: 'message_stop' },
  ];
  return sse(events.map((data) => ({ event: data.type, data })));
}

function googleToolCall(call: FakeToolCall): Response {
  return sse([
    {
      data: {
        candidates: [
          {
            index: 0,
            content: {
              role: 'model',
              parts: [{ functionCall: { name: call.name, args: call.args } }],
            },
            finishReason: 'STOP',
          },
        ],
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
        },
      },
    },
  ]);
}

async function* bedrockToolCall(
  call: FakeToolCall
): AsyncGenerator<{ [key: string]: object }> {
  yield { messageStart: { role: 'assistant' } };
  yield {
    contentBlockStart: {
      contentBlockIndex: 0,
      start: { toolUse: { toolUseId: call.id, name: call.name } },
    },
  };
  for (const input of argumentFragments(call)) {
    yield { contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input } } } };
  }
  yield { contentBlockStop: { contentBlockIndex: 0 } };
  yield { messageStop: { stopReason: 'tool_use' } };
  yield {
    metadata: {
      usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
      metrics: { latencyMs: 1 },
    },
  };
}

function interceptFetch(
  record: RecordRequest,
  replyWithToolCall: (call: FakeToolCall) => Response
): Restore {
  const spy = jest
    .spyOn(global, 'fetch')
    .mockImplementation(async (input, init) => {
      const raw =
        input instanceof Request
          ? await input.clone().text()
          : String(init?.body ?? '{}');
      const call = record(JSON.parse(raw) as Json);
      if (call == null) {
        throw new Error(CAPTURED);
      }
      return replyWithToolCall(call);
    });
  return () => spy.mockRestore();
}

function reduceChatCompletions(body: Json): CapturedRequest {
  const request = asObject(body);
  const messages = asArray(request.messages);
  const isSystem = (message: Json): boolean => {
    const role = asObject(message).role;
    return role === 'system' || role === 'developer';
  };
  return {
    tools: asArray(request.tools),
    system: messages.filter(isSystem),
    messages: messages.filter((message) => !isSystem(message)),
    raw: body,
  };
}

function reduceResponses(body: Json): CapturedRequest {
  const request = asObject(body);
  const input = asArray(request.input);
  const isSystem = (item: Json): boolean => {
    const role = asObject(item).role;
    return role === 'system' || role === 'developer';
  };
  return {
    tools: asArray(request.tools),
    system: [request.instructions ?? null, ...input.filter(isSystem)],
    messages: input.filter((item) => !isSystem(item)),
    raw: body,
  };
}

function reduceAnthropic(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(request.tools),
    system: request.system ?? null,
    messages: asArray(request.messages),
    raw: body,
  };
}

function reduceGoogle(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(request.tools).flatMap((entry) => {
      const declarations = asObject(entry).functionDeclarations;
      return Array.isArray(declarations) ? declarations : [entry];
    }),
    system: request.systemInstruction ?? null,
    messages: asArray(request.contents),
    raw: body,
  };
}

function reduceBedrock(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(asObject(request.toolConfig).tools),
    system: request.system ?? null,
    messages: asArray(request.messages),
    raw: body,
  };
}

function fetchCase(
  fields: Omit<ProviderCase, 'intercept'>,
  replyWithToolCall: (call: FakeToolCall) => Response = chatCompletionsToolCall
): ProviderCase {
  return {
    ...fields,
    intercept: (_agent, record) => interceptFetch(record, replyWithToolCall),
  };
}

function bedrockCase(name: string, promptCache: boolean): ProviderCase {
  return {
    name,
    provider: Providers.BEDROCK,
    clientOptions: () =>
      ({
        model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
        region: 'us-east-1',
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
        maxRetries: 0,
        promptCache,
      }) as t.ClientOptions,
    intercept: (agent, record) => {
      const options = agent.clientOptions as { client?: object };
      options.client = {
        send: async (command: {
          input: object;
        }): Promise<{ stream: AsyncIterable<object>; $metadata: object }> => {
          const call = record(JSON.parse(JSON.stringify(command.input)) as Json);
          if (call == null) {
            throw new Error(CAPTURED);
          }
          return { stream: bedrockToolCall(call), $metadata: {} };
        },
        config: {},
        destroy: (): void => undefined,
        middlewareStack: {
          add: (): void => undefined,
          use: (): void => undefined,
        },
      };
      return () => undefined;
    },
    reduce: reduceBedrock,
    breakpoints: promptCache ? BEDROCK_BREAKPOINTS : undefined,
    strictAlternation: true,
  };
}

export const PROVIDER_CASES: ProviderCase[] = [
  fetchCase({
    name: 'openAI',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({ model: 'gpt-4.1', apiKey: API_KEY, maxRetries: 0 }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'openAI responses',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({
        model: 'gpt-5',
        apiKey: API_KEY,
        maxRetries: 0,
        useResponsesApi: true,
      }) as t.ClientOptions,
    reduce: reduceResponses,
  }, responsesToolCall),
  fetchCase({
    name: 'openAI-compatible endpoint',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({
        model: 'local-model',
        apiKey: API_KEY,
        maxRetries: 0,
        configuration: { baseURL: 'https://openai-compatible.invalid/v1' },
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'azureOpenAI',
    provider: Providers.AZURE,
    clientOptions: () =>
      ({
        model: 'gpt-4.1',
        azureOpenAIApiKey: API_KEY,
        azureOpenAIApiInstanceName: 'harness',
        azureOpenAIApiDeploymentName: 'gpt-4-1',
        azureOpenAIApiVersion: '2024-10-21',
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'deepseek',
    provider: Providers.DEEPSEEK,
    clientOptions: () =>
      ({
        model: 'deepseek-chat',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'xai',
    provider: Providers.XAI,
    clientOptions: () =>
      ({ model: 'grok-4', apiKey: API_KEY, maxRetries: 0 }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'moonshot',
    provider: Providers.MOONSHOT,
    clientOptions: () =>
      ({
        model: 'kimi-k2.5',
        apiKey: API_KEY,
        maxRetries: 0,
        configuration: {
          apiKey: API_KEY,
          baseURL: 'https://api.moonshot.ai/v1',
        },
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'anthropic',
    provider: Providers.ANTHROPIC,
    clientOptions: () =>
      ({
        model: 'claude-sonnet-4-5',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceAnthropic,
  }, anthropicToolCall),
  fetchCase({
    name: 'anthropic promptCache',
    provider: Providers.ANTHROPIC,
    clientOptions: () =>
      ({
        model: 'claude-sonnet-4-5',
        apiKey: API_KEY,
        maxRetries: 0,
        promptCache: true,
      }) as t.ClientOptions,
    reduce: reduceAnthropic,
    breakpoints: ANTHROPIC_BREAKPOINTS,
    movesDynamicPart: true,
  }, anthropicToolCall),
  fetchCase({
    name: 'openrouter',
    provider: Providers.OPENROUTER,
    clientOptions: () =>
      ({
        model: 'anthropic/claude-sonnet-4.5',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'openrouter promptCache',
    provider: Providers.OPENROUTER,
    clientOptions: () =>
      ({
        model: 'anthropic/claude-sonnet-4.5',
        apiKey: API_KEY,
        maxRetries: 0,
        promptCache: true,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
    breakpoints: ANTHROPIC_BREAKPOINTS,
    movesDynamicPart: true,
  }),
  bedrockCase('bedrock', false),
  bedrockCase('bedrock promptCache', true),
  fetchCase({
    name: 'google',
    provider: Providers.GOOGLE,
    clientOptions: () =>
      ({
        model: 'gemini-2.5-flash',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceGoogle,
  }, googleToolCall),
];

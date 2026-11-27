import { convertMessagesToResponsesInput } from '@langchain/openai';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { OpenAIChatInput, OpenAIClient } from '@langchain/openai';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type {
  ChatAnthropicToolType,
  AnthropicMCPServerURLDefinition,
} from '@/llm/anthropic/types';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import type { CustomAnthropicInput } from '@/llm/anthropic';
import {
  ChatXAI,
  ChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  AzureChatOpenAI,
  CustomOpenAIClient,
  CustomAzureOpenAIClient,
} from '@/llm/openai';
import {
  OPENAI_RESPONSES_REPLAY_POSITIONS_KEY,
  projectOpenAIResponsesToolMessageContent,
} from '@/messages/core';
import {
  OPENAI_RESPONSES_ORIGIN_KEY,
  getResponsesReplayOrigin,
} from '@/messages/responsesToolRounds';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { ChatOpenRouter } from '@/llm/openrouter';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatVertexAI } from '@/llm/vertexai';

type OpenAIRequestOptions = Parameters<ChatOpenAI['_getClientOptions']>[0];
type OpenAIRequestOptionsWithBaseURL = ReturnType<
  ChatXAI['_getClientOptions']
> & {
  baseURL?: string;
};
type OpenAIResponsesDelegate = {
  client?: unknown;
  _getClientOptions: (options?: OpenAIRequestOptions) => OpenAIRequestOptions;
};
type StreamingOpenAIResponsesDelegate = OpenAIResponsesDelegate & {
  completionWithRetry: (
    request: OpenAIClient.Responses.ResponseCreateParamsStreaming
  ) => Promise<AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>>;
};
type MockableResponsesDelegate = OpenAIResponsesDelegate & {
  completionWithRetry: (request: {
    input?: unknown;
    stream?: boolean;
  }) => Promise<unknown>;
  _generate: (
    messages: AIMessage[],
    options: Record<string, unknown>
  ) => Promise<unknown>;
  _streamChatModelEvents: (
    messages: AIMessage[],
    options: Record<string, unknown>
  ) => AsyncGenerator<unknown>;
  _streamResponseChunks: (
    messages: AIMessage[],
    options: Record<string, unknown>
  ) => AsyncGenerator<ChatGenerationChunk>;
};
type AnthropicCallOptions = Parameters<
  CustomAnthropic['invocationParams']
>[0] & {
  outputConfig?: CustomAnthropicInput['outputConfig'] & {
    task_budget?: { type: 'token_budget'; value: number };
  };
  inferenceGeo?: CustomAnthropicInput['inferenceGeo'];
  betas?: CustomAnthropicInput['betas'];
  container?: string;
  mcp_servers?: AnthropicMCPServerURLDefinition[];
  tools?: ChatAnthropicToolType[];
};
type AzureReasoningModel = AzureChatOpenAI & {
  reasoning?: { effort: 'low' | 'high' };
};
type OpenRouterFields = Partial<
  ChatOpenRouterCallOptions & Pick<OpenAIChatInput, 'model' | 'apiKey'>
>;
type CompletionDelegate = {
  completionWithRetry: (request: { messages?: unknown }) => Promise<unknown>;
};
type CompletionBackedModel = {
  completions: CompletionDelegate;
};
type StreamingCompletionRequest = {
  messages?: unknown;
  stream?: boolean;
};
type StreamingCompletionDelegate = {
  completionWithRetry: (
    request: StreamingCompletionRequest
  ) => Promise<
    AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
  >;
};
type StreamingCompletionBackedModel = {
  completions: StreamingCompletionDelegate;
};
type OpenAIStreamEvent = {
  event: string;
  data?: unknown;
};
type OpenAIStreamItem =
  | OpenAIClient.Chat.Completions.ChatCompletionChunk
  | OpenAIStreamEvent;
type FetchOutcome = 'resolved' | 'rejected' | 'pending';
type AbortableFetchCapture = {
  fetch: typeof fetch;
  getSignal: () => AbortSignal | undefined;
};
type FetchTimeoutClient = {
  fetchWithTimeout: (
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController
  ) => Promise<Response>;
};
type MockableCompletionCreate = (
  request: unknown,
  options?: unknown
) => Promise<
  AsyncIterable<OpenAIStreamItem> | OpenAIClient.Chat.Completions.ChatCompletion
>;
type MockableCompletionClient = {
  chat: {
    completions: {
      create: MockableCompletionCreate;
    };
  };
};
type MockableCompletionDelegate = OpenAIResponsesDelegate & {
  client?: MockableCompletionClient;
};
type OpenRouterReasoningStreamDelta =
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice.Delta & {
    reasoning_details?: Array<
      | {
          type: 'reasoning.text';
          text?: string;
          format?: string;
          index?: number;
        }
      | {
          type: 'reasoning.encrypted';
          id?: string;
          data?: string;
          format?: string;
          index?: number;
        }
    >;
  };
type OpenRouterReasoningStreamChoice = Omit<
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice,
  'delta'
> & {
  delta: OpenRouterReasoningStreamDelta;
};
type OpenAICompatibleReasoningStreamDelta =
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice.Delta & {
    reasoning?: string;
    reasoning_details?: Array<{
      type: 'reasoning.text';
      text?: string;
      format?: string;
      index?: number;
    }>;
  };
type OpenAICompatibleReasoningStreamChoice = Omit<
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice,
  'delta'
> & {
  delta: OpenAICompatibleReasoningStreamDelta;
};
type PromptTokensDetailsWithCacheWrite = NonNullable<
  OpenAIClient.Completions.CompletionUsage['prompt_tokens_details']
> & {
  cache_write_tokens?: number;
};
type CompletionUsageWithCacheWrite = Omit<
  OpenAIClient.Completions.CompletionUsage,
  'prompt_tokens_details'
> & {
  prompt_tokens_details?: PromptTokensDetailsWithCacheWrite;
};
type OpenAIStreamModel = ChatOpenAI | AzureChatOpenAI;

class CallbackTestChatOpenRouter extends ChatOpenRouter {
  streamChunksWithCallbacks(
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    return this._streamResponseChunks(
      [new HumanMessage('hi')],
      {} as this['ParsedCallOptions'],
      runManager
    );
  }
}

const baseAzureFields = {
  azureOpenAIApiKey: 'test-azure-key',
  azureOpenAIApiVersion: '2024-10-21',
  azureOpenAIApiInstanceName: 'test-instance',
  azureOpenAIApiDeploymentName: 'test-deployment',
};

function createPersistedPreemptedAzureMessage(): AIMessage {
  return new AIMessage({
    id: 'msg_persisted_azure',
    content: [{ type: 'text', text: 'Partial Azure answer.' }],
    additional_kwargs: {
      reasoning: {
        id: 'rs_persisted_azure',
        type: 'reasoning',
        status: 'in_progress',
        summary: [],
      },
      tool_outputs: [
        {
          id: 'local_persisted_azure',
          type: 'local_shell_call_output',
          status: 'completed',
          output: 'AZURE_SERVER_RESULT',
        },
      ],
      [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
        {
          itemId: 'local_persisted_azure',
          kind: 'output',
          outputIndex: 0,
        },
        {
          itemId: 'rs_persisted_azure',
          kind: 'reasoning',
          outputIndex: 1,
        },
        {
          itemId: 'msg_persisted_azure',
          kind: 'text',
          outputIndex: 2,
          contentIndex: 0,
        },
      ],
    },
    response_metadata: {
      id: 'resp_persisted_azure',
      model_provider: 'openai',
      preempted: true,
    },
  });
}

function expectNeutralizedAzureResponsesRequest(request: unknown): void {
  const serialized = JSON.stringify(request);
  expect(serialized).toContain('AZURE_SERVER_RESULT');
  expect(serialized.match(/AZURE_SERVER_RESULT/g)).toHaveLength(1);
  expect(serialized).toContain('serverToolResult');
  expect(serialized).not.toContain('msg_persisted_azure');
  expect(serialized).not.toContain('rs_persisted_azure');
  expect(serialized).not.toContain('local_persisted_azure');
  expect(serialized).not.toContain('resp_persisted_azure');
  expect(serialized).not.toContain(OPENAI_RESPONSES_REPLAY_POSITIONS_KEY);
  expect(serialized).not.toContain('local_shell_call_output');
}

function createCompletedAzureResponse(): OpenAIClient.Responses.Response {
  return {
    id: 'resp_fresh_azure',
    created_at: 0,
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: {},
    model: 'gpt-5',
    object: 'response',
    output: [
      {
        id: 'msg_fresh_azure',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'Fresh Azure answer.',
            annotations: [],
          },
        ],
      },
    ],
    output_text: 'Fresh Azure answer.',
    parallel_tool_calls: true,
    status: 'completed',
    temperature: null,
    tool_choice: 'auto',
    tools: [],
    top_p: null,
    usage: {
      input_tokens: 1,
      input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
      output_tokens: 1,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 2,
    },
  } as OpenAIClient.Responses.Response;
}

const waitForFetchOutcome = (
  promise: Promise<Response>,
  timeoutMs = 100
): Promise<FetchOutcome> =>
  Promise.race([
    promise.then(
      () => 'resolved' as const,
      () => 'rejected' as const
    ),
    new Promise<'pending'>((resolve) => {
      setTimeout(() => resolve('pending'), timeoutMs);
    }),
  ]);

const createAbortableFetch = (): AbortableFetchCapture => {
  let requestSignal: AbortSignal | undefined;
  return {
    fetch: async (_url, init): Promise<Response> =>
      new Promise<Response>((_resolve, reject) => {
        requestSignal = init?.signal ?? undefined;
        requestSignal?.addEventListener(
          'abort',
          () => reject(new Error('Aborted')),
          { once: true }
        );
      }),
    getSignal: () => requestSignal,
  };
};

const expectFetchTimeoutAbort = async (
  client: FetchTimeoutClient,
  capturedFetch: AbortableFetchCapture,
  url: string
): Promise<void> => {
  const controller = new AbortController();

  const response = client.fetchWithTimeout(
    url,
    { method: 'post' },
    10,
    controller
  );

  await expect(waitForFetchOutcome(response)).resolves.toBe('rejected');
  expect(controller.signal.aborted).toBe(true);
  expect(capturedFetch.getSignal()?.aborted).toBe(true);
};

const baseBedrockFields = {
  region: 'us-east-1',
  credentials: {
    accessKeyId: 'test-access-key',
    secretAccessKey: 'test-secret-key',
  },
};

const createOpenAIStreamChunk = (
  content: string,
  finishReason: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['finish_reason'] = null
): OpenAIClient.Chat.Completions.ChatCompletionChunk => ({
  id: 'chatcmpl-hermes-test',
  object: 'chat.completion.chunk',
  created: 0,
  model: 'hermes-agent',
  choices: [
    {
      index: 0,
      delta: { content },
      finish_reason: finishReason,
    },
  ],
});

async function* createOpenAIStreamWithCustomEvents(): AsyncGenerator<OpenAIStreamItem> {
  yield createOpenAIStreamChunk('Hello ');
  yield {
    event: 'hermes.tool.progress',
    data: {
      tool: 'execute_code',
      toolCallId: 'call_1',
      status: 'running',
    },
  };
  yield {
    event: 'hermes.tool.progress',
    data: null,
  };
  yield {
    event: 'message',
    data: createOpenAIStreamChunk('world', 'stop'),
  };
}

function mockCompletionStream(
  model: OpenAIStreamModel
): MockableCompletionCreate {
  const completions = (
    model as unknown as { completions: MockableCompletionDelegate }
  ).completions;
  completions._getClientOptions(undefined);
  const client = completions.client;
  if (client == null) {
    throw new Error('Expected OpenAI completions client');
  }

  const createMock = jest.fn(async () =>
    createOpenAIStreamWithCustomEvents()
  ) as MockableCompletionCreate;
  client.chat.completions.create = createMock;
  return createMock;
}

function mockCompletion(
  model: ChatOpenAI,
  response: OpenAIClient.Chat.Completions.ChatCompletion
): MockableCompletionCreate {
  const completions = (
    model as unknown as { completions: MockableCompletionDelegate }
  ).completions;
  completions._getClientOptions(undefined);
  const client = completions.client;
  if (client == null) {
    throw new Error('Expected OpenAI completions client');
  }

  const createMock = jest.fn(async () => response) as MockableCompletionCreate;
  client.chat.completions.create = createMock;
  return createMock;
}

async function expectCustomSSEEventsSkipped(
  model: OpenAIStreamModel
): Promise<void> {
  const createMock = mockCompletionStream(model);
  const chunks: AIMessageChunk[] = [];
  const stream = await model.stream([new HumanMessage('use a tool')]);
  for await (const chunk of stream) {
    chunks.push(chunk);
  }

  const text = chunks
    .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
    .join('');
  expect(chunks).toHaveLength(2);
  expect(text).toBe('Hello world');
  expect(createMock).toHaveBeenCalledWith(
    expect.objectContaining({ stream: true }),
    expect.any(Object)
  );
}

describe('custom chat model class smoke tests', () => {
  it('keeps the custom OpenAI client, stream delay, and reasoning precedence', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      reasoning: { effort: 'low' },
      _lc_stream_delay: 3,
    });

    const requestOptions = model._getClientOptions({
      headers: { 'x-smoke': 'openai' },
    } as OpenAIRequestOptions);

    expect(ChatOpenAI.lc_name()).toBe('LibreChatOpenAI');
    expect(model._lc_stream_delay).toBe(3);
    expect(model.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(requestOptions.headers).toEqual(
      expect.objectContaining({ 'x-smoke': 'openai' })
    );
    expect(model.getReasoningParams({ reasoning: { effort: 'high' } })).toEqual(
      { effort: 'high' }
    );
    const params = model.invocationParams({ reasoningEffort: 'medium' }) as {
      reasoning?: unknown;
      reasoning_effort?: unknown;
    };
    expect(params.reasoning ?? { effort: params.reasoning_effort }).toEqual({
      effort: 'low',
    });
  });

  it('keeps the custom OpenAI client on the Responses API path', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
      maxTokens: 32,
      reasoning: { effort: 'low' },
    });

    const params = model.invocationParams({
      reasoningEffort: 'high',
    }) as {
      max_output_tokens?: number;
      max_completion_tokens?: number;
      reasoning?: unknown;
    };
    const responses = (
      model as unknown as { responses: OpenAIResponsesDelegate }
    ).responses;
    const requestOptions = responses._getClientOptions({
      headers: { 'x-smoke': 'responses' },
    } as OpenAIRequestOptions);

    expect(params.max_output_tokens).toBe(32);
    expect(params.max_completion_tokens).toBeUndefined();
    expect(params.reasoning).toEqual({ effort: 'low' });
    expect(responses.client).toBeInstanceOf(CustomOpenAIClient);
    const responsesClient = responses.client as CustomOpenAIClient;
    responsesClient.abortHandler = (): void => undefined;
    expect(model.exposedClient).toBe(responsesClient);
    expect(requestOptions?.headers).toEqual(
      expect.objectContaining({ 'x-smoke': 'responses' })
    );
  });

  it('retains replayable Responses output items before a terminal event', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingOpenAIResponsesDelegate }
    ).responses;
    const items = [
      {
        id: 'local_output_item',
        type: 'local_shell_call_output',
        status: 'completed',
        output: 'local output',
      },
      {
        id: 'shell_output_item',
        call_id: 'shell_call_id',
        type: 'shell_call_output',
        status: 'completed',
        max_output_length: 1_000,
        output: [
          {
            stdout: 'shell output',
            stderr: '',
            outcome: { type: 'exit', exit_code: 0 },
          },
        ],
      },
      {
        id: 'patch_output_item',
        call_id: 'patch_call_id',
        type: 'apply_patch_call_output',
        status: 'completed',
        output: 'patch output',
      },
      {
        id: 'program_output_item',
        call_id: 'program_call_id',
        type: 'program_output',
        status: 'completed',
        result: 'program output',
      },
    ] as const;
    responses.completionWithRetry = async () =>
      (async function* () {
        for (let i = 0; i < items.length; i++) {
          yield {
            type: 'response.output_item.done',
            sequence_number: i,
            output_index: i,
            item: items[i],
          } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
        }
        yield {
          type: 'response.output_item.added',
          sequence_number: items.length,
          output_index: items.length,
          item: {
            id: 'msg_partial',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_text.delta',
          sequence_number: items.length + 1,
          output_index: items.length,
          content_index: 0,
          item_id: 'msg_partial',
          delta: 'Partial ',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_text.delta',
          sequence_number: items.length + 2,
          output_index: items.length,
          content_index: 0,
          item_id: 'msg_partial',
          delta: 'answer.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
      })();

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream([new HumanMessage('run tools')]);
    for await (const chunk of stream) {
      chunks.push(chunk as AIMessageChunk);
    }
    const combined = chunks.reduce<AIMessageChunk | undefined>(
      (current, chunk) =>
        current == null ? chunk : (current.concat(chunk) as AIMessageChunk),
      undefined
    );

    expect(chunks).toHaveLength(items.length + 3);
    expect(combined?.text).toBe('Partial answer.');
    expect(combined?.additional_kwargs.tool_outputs).toEqual(items);
    expect(
      combined?.additional_kwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]
    ).toEqual([
      ...items.map((item, outputIndex) => ({
        itemId: item.id,
        kind: 'output',
        outputIndex,
      })),
      {
        itemId: 'msg_partial',
        kind: 'message',
        outputIndex: items.length,
      },
      {
        contentIndex: 0,
        itemId: 'msg_partial',
        kind: 'text',
        outputIndex: items.length,
      },
    ]);
  });

  it('rehydrates one-chunk Responses replay ordering from constructor kwargs', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingOpenAIResponsesDelegate }
    ).responses;
    responses.completionWithRetry = async () =>
      (async function* () {
        yield {
          type: 'response.output_text.delta',
          sequence_number: 0,
          output_index: 3,
          content_index: 2,
          item_id: 'msg_one_chunk',
          delta: 'Partial answer.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
      })();

    const stream = await model.stream([new HumanMessage('answer')]);
    const result = await stream.next();
    expect(result.done).toBe(false);
    const chunk = result.value as AIMessageChunk;
    const serializedFields = JSON.parse(
      JSON.stringify(chunk.lc_kwargs)
    ) as ConstructorParameters<typeof AIMessageChunk>[0];
    const rehydrated = new AIMessageChunk(serializedFields);

    expect(chunk.content).toEqual([
      { type: 'text', text: 'Partial answer.', index: 0 },
    ]);
    expect(rehydrated.content).toEqual([
      { type: 'text', text: 'Partial answer.', index: 0 },
    ]);
    expect(
      rehydrated.additional_kwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]
    ).toEqual([
      {
        contentIndex: 2,
        itemId: 'msg_one_chunk',
        kind: 'text',
        outputIndex: 3,
      },
    ]);
  });

  it.each(['response.completed', 'response.incomplete'] as const)(
    'removes provisional replay outputs after %s supplies the authoritative output',
    async (terminalType) => {
      const model = new ChatOpenAI({
        model: 'gpt-5',
        apiKey: 'test-key',
        useResponsesApi: true,
      });
      const responses = (
        model as unknown as { responses: StreamingOpenAIResponsesDelegate }
      ).responses;
      const streamedToolOutput = {
        id: 'web_search_item',
        type: 'web_search_call',
        status: 'completed',
      } as const;
      const provisionalReplayOutput = {
        id: 'local_output_item',
        type: 'local_shell_call_output',
        status: 'completed',
        output: 'stale provisional local output',
      } as const;
      const authoritativeReplayOutput = {
        ...provisionalReplayOutput,
        output: 'unique authoritative local output',
      } as const;
      responses.completionWithRetry = async () =>
        (async function* () {
          yield {
            type: 'response.output_item.done',
            sequence_number: 0,
            output_index: 0,
            item: streamedToolOutput,
          } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
          yield {
            type: 'response.output_item.done',
            sequence_number: 1,
            output_index: 1,
            item: provisionalReplayOutput,
          } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
          yield {
            type: terminalType,
            sequence_number: 2,
            response: {
              id: 'resp_terminal',
              model: 'gpt-5',
              output: [streamedToolOutput, authoritativeReplayOutput],
              status:
                terminalType === 'response.completed'
                  ? 'completed'
                  : 'incomplete',
              usage: null,
            },
          } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
        })();

      let tracedResult: unknown;
      const chunks: AIMessageChunk[] = [];
      const stream = await model.stream([new HumanMessage('run tools')], {
        callbacks: [
          {
            handleLLMEnd(result: unknown): void {
              tracedResult = result;
            },
          },
        ],
      });
      for await (const chunk of stream) {
        chunks.push(chunk as AIMessageChunk);
      }
      const combined = chunks.reduce<AIMessageChunk | undefined>(
        (current, chunk) =>
          current == null ? chunk : (current.concat(chunk) as AIMessageChunk),
        undefined
      );

      expect(combined?.additional_kwargs.tool_outputs).toEqual([
        streamedToolOutput,
      ]);
      expect(combined?.response_metadata.output).toEqual([
        streamedToolOutput,
        authoritativeReplayOutput,
      ]);
      expect(combined?.toJSON()).toEqual(
        expect.objectContaining({
          id: ['langchain_core', 'messages', 'AIMessageChunk'],
        })
      );
      expect(
        JSON.stringify(combined).match(/unique authoritative local output/g)
      ).toHaveLength(1);
      expect(JSON.stringify(combined)).not.toContain(
        'stale provisional local output'
      );
      expect(JSON.stringify(combined?.lc_kwargs)).not.toContain(
        'stale provisional local output'
      );
      expect(combined?.lc_kwargs.additional_kwargs).toEqual(
        expect.objectContaining({ tool_outputs: [streamedToolOutput] })
      );
      expect(combined?.additional_kwargs).not.toHaveProperty(
        OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
      );
      expect(combined?.lc_kwargs.additional_kwargs).not.toHaveProperty(
        OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
      );
      expect(
        JSON.stringify(tracedResult).match(/unique authoritative local output/g)
      ).toHaveLength(1);
      expect(JSON.stringify(tracedResult)).not.toContain(
        'stale provisional local output'
      );
      expect(JSON.stringify(tracedResult)).not.toContain(
        OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
      );
    }
  );

  it('stamps a finished Responses output with the model and endpoint that produced it', async () => {
    const stamp = async (baseURL?: string): Promise<unknown> => {
      const model = new ChatOpenAI({
        model: 'gpt-5',
        apiKey: 'test-key',
        useResponsesApi: true,
        ...(baseURL != null && { configuration: { baseURL } }),
      });
      const responses = (
        model as unknown as { responses: StreamingOpenAIResponsesDelegate }
      ).responses;
      responses.completionWithRetry = async () =>
        (async function* () {
          yield {
            type: 'response.completed',
            sequence_number: 0,
            response: {
              id: 'resp_stamp',
              model: 'gpt-5',
              output: [
                {
                  id: 'fc_1',
                  type: 'function_call',
                  status: 'completed',
                  call_id: 'call_1',
                  name: 'lookup',
                  arguments: '{}',
                },
              ],
              status: 'completed',
              usage: null,
            },
          } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
        })();
      let combined: AIMessageChunk | undefined;
      for await (const chunk of await model.stream([new HumanMessage('go')])) {
        combined =
          combined == null ? chunk : (combined.concat(chunk) as AIMessageChunk);
      }
      return combined?.response_metadata[OPENAI_RESPONSES_ORIGIN_KEY];
    };

    expect(await stamp()).toBe(getResponsesReplayOrigin('gpt-5', ''));
    expect(await stamp('https://proxy.example/v1')).toBe(
      getResponsesReplayOrigin('gpt-5', 'https://proxy.example/v1')
    );
  });

  it('removes replay positions on a text-only terminal response', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingOpenAIResponsesDelegate }
    ).responses;
    const messageItem = {
      id: 'msg_terminal',
      type: 'message',
      role: 'assistant',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'Complete answer.',
          annotations: [],
          logprobs: [],
        },
      ],
    } as const;
    responses.completionWithRetry = async () =>
      (async function* () {
        yield {
          type: 'response.output_text.delta',
          sequence_number: 0,
          output_index: 0,
          content_index: 0,
          item_id: messageItem.id,
          delta: messageItem.content[0].text,
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.completed',
          sequence_number: 1,
          response: {
            id: 'resp_text_terminal',
            model: 'gpt-5',
            output: [messageItem],
            status: 'completed',
            usage: null,
          },
        } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
      })();

    let combined: AIMessageChunk | undefined;
    const stream = await model.stream([new HumanMessage('answer')]);
    for await (const chunk of stream) {
      combined =
        combined == null
          ? (chunk as AIMessageChunk)
          : (combined.concat(chunk as AIMessageChunk) as AIMessageChunk);
    }

    expect(combined?.text).toBe('Complete answer.');
    expect(combined?.response_metadata.output).toEqual([messageItem]);
    expect(combined?.additional_kwargs).not.toHaveProperty(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );
    expect(combined?.lc_kwargs.additional_kwargs).not.toHaveProperty(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );
  });

  it('keeps a raw server result between distinct Responses output messages', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingOpenAIResponsesDelegate }
    ).responses;
    responses.completionWithRetry = async () =>
      (async function* () {
        yield {
          type: 'response.output_text.delta',
          sequence_number: 0,
          output_index: 0,
          content_index: 0,
          item_id: 'msg_first',
          delta: 'First narration.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_item.done',
          sequence_number: 1,
          output_index: 1,
          item: {
            id: 'local_output_item',
            type: 'local_shell_call_output',
            status: 'completed',
            output: 'middle server result',
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_text.delta',
          sequence_number: 2,
          output_index: 2,
          content_index: 0,
          item_id: 'msg_second',
          delta: 'Second narration.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
      })();

    let combined: AIMessageChunk | undefined;
    const stream = await model.stream([new HumanMessage('run a server tool')]);
    for await (const chunk of stream) {
      combined =
        combined == null
          ? (chunk as AIMessageChunk)
          : (combined.concat(chunk as AIMessageChunk) as AIMessageChunk);
    }
    expect(combined).toBeDefined();
    combined!.response_metadata.preempted = true;

    const [projected] = projectOpenAIResponsesToolMessageContent([combined!]);
    const providerInput = convertMessagesToResponsesInput({
      messages: [projected],
      model: 'gpt-5',
      zdrEnabled: false,
    });
    const serialized = JSON.stringify(providerInput);
    const firstIndex = serialized.indexOf('First narration.');
    const resultIndex = serialized.indexOf('middle server result');
    const secondIndex = serialized.indexOf('Second narration.');

    expect(firstIndex).toBeGreaterThanOrEqual(0);
    expect(firstIndex).toBeLessThan(resultIndex);
    expect(resultIndex).toBeLessThan(secondIndex);
  });

  it('keeps streamed generated images and server results in cross-item order', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingOpenAIResponsesDelegate }
    ).responses;
    responses.completionWithRetry = async () =>
      (async function* () {
        yield {
          type: 'response.output_text.delta',
          sequence_number: 0,
          output_index: 0,
          content_index: 0,
          item_id: 'msg_first',
          delta: 'First narration.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_item.done',
          sequence_number: 1,
          output_index: 1,
          item: {
            id: 'ig_middle',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_item.done',
          sequence_number: 2,
          output_index: 2,
          item: {
            id: 'local_middle',
            type: 'local_shell_call_output',
            status: 'completed',
            output: 'middle server result',
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_item.added',
          sequence_number: 3,
          output_index: 3,
          item: {
            id: 'rs_middle',
            type: 'reasoning',
            status: 'in_progress',
            summary: [],
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_item.done',
          sequence_number: 4,
          output_index: 3,
          item: {
            id: 'rs_middle',
            type: 'reasoning',
            status: 'incomplete',
            summary: [],
            encrypted_content: 'opaque-middle-reasoning',
          },
        } as OpenAIClient.Responses.ResponseStreamEvent;
        yield {
          type: 'response.output_text.delta',
          sequence_number: 5,
          output_index: 4,
          content_index: 0,
          item_id: 'msg_second',
          delta: 'Second narration.',
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
      })();

    let combined: AIMessageChunk | undefined;
    const stream = await model.stream([new HumanMessage('generate and run')]);
    for await (const chunk of stream) {
      combined =
        combined == null
          ? (chunk as AIMessageChunk)
          : (combined.concat(chunk as AIMessageChunk) as AIMessageChunk);
    }
    expect(combined).toBeDefined();
    expect(combined!.additional_kwargs.reasoning).toEqual({
      id: 'rs_middle',
      type: 'reasoning',
      status: 'incomplete',
      summary: [],
      encrypted_content: 'opaque-middle-reasoning',
    });
    expect(
      combined!.additional_kwargs[OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]
    ).toEqual([
      {
        contentIndex: 0,
        itemId: 'msg_first',
        kind: 'text',
        outputIndex: 0,
      },
      { itemId: 'ig_middle', kind: 'output', outputIndex: 1 },
      { itemId: 'local_middle', kind: 'output', outputIndex: 2 },
      { itemId: 'rs_middle', kind: 'reasoning', outputIndex: 3 },
      {
        contentIndex: 0,
        itemId: 'msg_second',
        kind: 'text',
        outputIndex: 4,
      },
    ]);
    combined!.response_metadata.preempted = true;

    const [projected] = projectOpenAIResponsesToolMessageContent([combined!]);
    const providerInput = convertMessagesToResponsesInput({
      messages: [projected],
      model: 'gpt-5',
      zdrEnabled: false,
    });
    const serialized = JSON.stringify(providerInput);
    const firstIndex = serialized.indexOf('First narration.');
    const imageIndex = serialized.indexOf('data:image/png;base64,AA==');
    const resultIndex = serialized.indexOf('middle server result');
    const reasoningIndex = serialized.indexOf('opaque-middle-reasoning');
    const secondIndex = serialized.indexOf('Second narration.');

    expect(firstIndex).toBeGreaterThanOrEqual(0);
    expect(firstIndex).toBeLessThan(imageIndex);
    expect(imageIndex).toBeLessThan(resultIndex);
    expect(resultIndex).toBeLessThan(reasoningIndex);
    expect(reasoningIndex).toBeLessThan(secondIndex);
    expect(serialized).not.toContain('ig_middle');
    expect(serialized).not.toContain('local_middle');
    expect(projected.additional_kwargs).not.toHaveProperty(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );
  });

  it('keeps Azure client customization and gates reasoning to reasoning models', () => {
    const model = new AzureChatOpenAI({
      ...baseAzureFields,
      _lc_stream_delay: 4,
    }) as AzureReasoningModel;
    model.model = 'gpt-5';
    model.reasoning = { effort: 'low' };

    const requestOptions = model._getClientOptions({
      headers: { 'x-smoke': 'azure' },
    });

    expect(AzureChatOpenAI.lc_name()).toBe('LibreChatAzureOpenAI');
    expect(model._lc_stream_delay).toBe(4);
    expect(model.exposedClient).toBeInstanceOf(CustomAzureOpenAIClient);
    const azureResponses = (
      model as unknown as { responses: OpenAIResponsesDelegate }
    ).responses;
    azureResponses._getClientOptions(undefined);
    expect(azureResponses.client).toBeInstanceOf(CustomAzureOpenAIClient);
    const azureResponsesClient =
      azureResponses.client as CustomAzureOpenAIClient;
    azureResponsesClient.abortHandler = (): void => undefined;
    expect(model.exposedClient).toBe(azureResponsesClient);
    expect(requestOptions.headers).toEqual(
      expect.objectContaining({
        'api-key': 'test-azure-key',
        'x-smoke': 'azure',
      })
    );
    expect(requestOptions.query).toEqual(
      expect.objectContaining({ 'api-version': '2024-10-21' })
    );
    expect(model.getReasoningParams()).toEqual({ effort: 'low' });

    const nonReasoningModel = new AzureChatOpenAI({
      ...baseAzureFields,
    }) as AzureReasoningModel;
    nonReasoningModel.model = 'gpt-4o';
    nonReasoningModel.reasoning = { effort: 'low' };
    expect(nonReasoningModel.getReasoningParams()).toBeUndefined();
  });

  it('sanitizes persisted preempted history on direct Azure Responses streams', async () => {
    const model = new AzureChatOpenAI({
      ...baseAzureFields,
    });
    model.model = 'gpt-5';
    const responses = (
      model as unknown as { responses: MockableResponsesDelegate }
    ).responses;
    const requests: unknown[] = [];
    responses.completionWithRetry = async (request) => {
      requests.push(request);
      return (async function* () {})();
    };
    const message = createPersistedPreemptedAzureMessage();
    const originalSerialized = JSON.stringify(message.toJSON());

    for await (const _chunk of responses._streamResponseChunks([message], {})) {
      // The empty mock stream intentionally yields no chunks.
    }
    for await (const _event of responses._streamChatModelEvents(
      [message],
      {}
    )) {
      // The empty mock stream intentionally yields no events.
    }

    expect(requests).toHaveLength(2);
    for (const request of requests) {
      expectNeutralizedAzureResponsesRequest(request);
    }
    expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);
  });

  it('sanitizes persisted preempted history on direct Azure Responses generation', async () => {
    const model = new AzureChatOpenAI({
      ...baseAzureFields,
    });
    model.model = 'gpt-5';
    const responses = (
      model as unknown as { responses: MockableResponsesDelegate }
    ).responses;
    let request: unknown;
    responses.completionWithRetry = async (nextRequest) => {
      request = nextRequest;
      return createCompletedAzureResponse();
    };
    const message = createPersistedPreemptedAzureMessage();
    const originalSerialized = JSON.stringify(message.toJSON());

    await responses._generate([message], {});

    expectNeutralizedAzureResponsesRequest(request);
    expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);
  });

  it('keeps DeepSeek, Moonshot, and xAI on LibreChat wrapper semantics', () => {
    const deepSeek = new ChatDeepSeek({
      model: 'deepseek-chat',
      apiKey: 'test-key',
      _lc_stream_delay: 5,
    });
    deepSeek._getClientOptions();

    const moonshot = new ChatMoonshot({
      model: 'moonshot-v1-8k',
      apiKey: 'test-key',
      _lc_stream_delay: 6,
    });

    const xai = new ChatXAI({
      model: 'grok-3-fast',
      apiKey: 'test-key',
      configuration: { baseURL: 'https://xai.test/v1' },
      _lc_stream_delay: 7,
    });
    const xaiRequestOptions =
      xai._getClientOptions() as OpenAIRequestOptionsWithBaseURL;

    expect(ChatDeepSeek.lc_name()).toBe('LibreChatDeepSeek');
    expect(deepSeek._lc_stream_delay).toBe(5);
    expect(deepSeek.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(ChatMoonshot.lc_name()).toBe('LibreChatMoonshot');
    expect(moonshot._lc_stream_delay).toBe(6);
    expect(ChatXAI.lc_name()).toBe('LibreChatXAI');
    expect(xai._lc_stream_delay).toBe(7);
    expect(xai.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(xaiRequestOptions.baseURL).toBe('https://xai.test/v1');
  });

  it('skips custom OpenAI-compatible SSE events during OpenAI streaming', async () => {
    await expectCustomSSEEventsSkipped(
      new ChatOpenAI({
        model: 'hermes-agent',
        apiKey: 'test-key',
        streaming: true,
      })
    );
  });

  it('preserves OpenAI-compatible reasoning deltas during OpenAI streaming', async () => {
    const model = new ChatOpenAI({
      model: 'openai/gpt-oss-120b',
      apiKey: 'test-key',
      streaming: true,
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    const createChunk = (
      choice: OpenAICompatibleReasoningStreamChoice
    ): OpenAIClient.Chat.Completions.ChatCompletionChunk => ({
      id: 'chatcmpl-openai-compatible-reasoning',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'openai/gpt-oss-120b',
      choices: [choice],
    });

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createChunk({
        index: 0,
        delta: {
          role: 'assistant',
          content: '',
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: {
          reasoning: 'Think ',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Think ',
              format: 'text',
              index: 0,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: {
          reasoning: 'hard',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'hard',
              format: 'text',
              index: 0,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: { content: 'answer' },
        finish_reason: 'stop',
      });
    }

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => streamChunks();

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream([new HumanMessage('think')]);
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(
      chunks
        .map((chunk) => chunk.additional_kwargs.reasoning_content)
        .filter((reasoningContent) => reasoningContent != null)
    ).toEqual(['Think ', 'hard']);
    expect(chunks[1].additional_kwargs.reasoning_details).toEqual([
      {
        type: 'reasoning.text',
        text: 'Think ',
        format: 'text',
        index: 0,
      },
    ]);
    expect(chunks[2].additional_kwargs.reasoning_details).toEqual([
      {
        type: 'reasoning.text',
        text: 'hard',
        format: 'text',
        index: 0,
      },
    ]);
    expect(chunks.at(-1)?.content).toBe('answer');
  });

  it('skips custom OpenAI-compatible SSE events during Azure streaming', async () => {
    await expectCustomSSEEventsSkipped(
      new AzureChatOpenAI({
        ...baseAzureFields,
      })
    );
  });

  it('passes non-streaming OpenAI completions through unchanged', async () => {
    const model = new ChatOpenAI({
      model: 'hermes-agent',
      apiKey: 'test-key',
    });
    const createMock = mockCompletion(model, {
      id: 'chatcmpl-nonstream-test',
      object: 'chat.completion',
      created: 0,
      model: 'hermes-agent',
      choices: [
        {
          index: 0,
          finish_reason: 'stop',
          logprobs: null,
          message: {
            role: 'assistant',
            content: 'plain response',
            refusal: null,
          },
        },
      ],
    });

    const response = await model.invoke([new HumanMessage('no stream')]);

    expect(response.content).toBe('plain response');
    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ stream: false }),
      expect.any(Object)
    );
  });

  it('keeps Moonshot reasoning content in completion requests', async () => {
    const moonshot = new ChatMoonshot({
      model: 'moonshot-v1-8k',
      apiKey: 'test-key',
      streaming: false,
    });
    const completions = (moonshot as unknown as CompletionBackedModel)
      .completions;
    let requestMessages: unknown;

    completions.completionWithRetry = async (request): Promise<unknown> => {
      requestMessages = request.messages;
      return {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        created: 0,
        model: 'moonshot-v1-8k',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            message: {
              role: 'assistant',
              content: 'ok',
            },
          },
        ],
      };
    };

    await moonshot.invoke([
      new AIMessage({
        content: '',
        additional_kwargs: { reasoning_content: 'kept-thinking' },
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: { q: 'test' },
            type: 'tool_call',
          },
        ],
      }),
    ]);

    expect(requestMessages).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'kept-thinking',
        tool_calls: expect.any(Array),
      }),
    ]);
  });

  it('keeps OpenRouter reasoning isolated from OpenAI reasoning_effort', () => {
    const fields: OpenRouterFields = {
      model: 'openrouter/test-model',
      apiKey: 'test-key',
      reasoning: { effort: 'xhigh', max_tokens: 2048 },
    };
    const model = new ChatOpenRouter(fields);

    const params = model.invocationParams();

    expect(ChatOpenRouter.lc_name()).toBe('LibreChatOpenRouter');
    expect(params.reasoning).toEqual({ effort: 'xhigh', max_tokens: 2048 });
    expect(params.reasoning_effort).toBeUndefined();

    const callParams = model.invocationParams({
      reasoning: { effort: 'low', exclude: true },
    } as ChatOpenRouterCallOptions);
    expect(callParams.reasoning).toEqual({
      effort: 'low',
      max_tokens: 2048,
      exclude: true,
    });

    const legacyModel = new ChatOpenRouter({
      model: 'openrouter/test-model',
      apiKey: 'test-key',
      include_reasoning: true,
    });
    expect(legacyModel.invocationParams().reasoning).toEqual({
      enabled: true,
    });
  });

  it('keeps OpenRouter streaming reasoning details stable', async () => {
    const model = new ChatOpenRouter({
      model: 'anthropic/claude-sonnet-test',
      apiKey: 'test-key',
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    let requestMessages: unknown;
    const createChunk = (
      choice: OpenRouterReasoningStreamChoice
    ): OpenAIClient.Chat.Completions.ChatCompletionChunk => ({
      id: 'chatcmpl-openrouter-test',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'anthropic/claude-sonnet-test',
      choices: [choice],
    });

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createChunk({
        index: 0,
        delta: {
          role: 'assistant',
          content: '',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Think ',
              format: 'text',
              index: 0,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: {
          content: 'answer',
          reasoning_details: [
            { type: 'reasoning.text', text: 'hard', index: 0 },
            {
              type: 'reasoning.encrypted',
              id: 'sig_1',
              data: 'encrypted',
              format: 'anthropic',
              index: 1,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: { content: '' },
        finish_reason: 'stop',
      });
    }

    completions.completionWithRetry = async (
      request
    ): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => {
      requestMessages = request.messages;
      return streamChunks();
    };

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream([
      new AIMessage({
        content: '',
        additional_kwargs: {
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'previous thought',
              index: 0,
            },
            {
              type: 'reasoning.encrypted',
              id: 'prev_sig',
              data: 'previous encrypted',
              index: 1,
            },
          ],
        },
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: { q: 'test' },
            type: 'tool_call',
          },
        ],
      }),
    ]);
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(requestMessages).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.any(Array),
        content: [
          expect.objectContaining({
            type: 'thinking',
            thinking: 'previous thought',
          }),
          expect.objectContaining({
            type: 'redacted_thinking',
            data: 'previous encrypted',
            id: 'prev_sig',
          }),
        ],
      }),
    ]);
    expect(chunks).toHaveLength(3);
    expect(chunks[0].additional_kwargs.reasoning).toBe('Think ');
    expect(chunks[0].additional_kwargs.reasoning_details).toBeUndefined();
    expect(chunks[1].additional_kwargs.reasoning).toBe('hard');
    expect(chunks[1].additional_kwargs.reasoning_details).toBeUndefined();
    expect(chunks[2].additional_kwargs.reasoning_details).toEqual([
      {
        type: 'reasoning.text',
        text: 'Think hard',
        format: 'text',
        index: 0,
      },
      {
        type: 'reasoning.encrypted',
        id: 'sig_1',
        data: 'encrypted',
        format: 'anthropic',
        index: 1,
      },
    ]);
  });

  it('maps OpenRouter cache write usage to cache_creation in streaming responses', async () => {
    const model = new ChatOpenRouter({
      model: 'anthropic/claude-sonnet-test',
      apiKey: 'test-key',
      streamUsage: true,
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    const usage: CompletionUsageWithCacheWrite = {
      prompt_tokens: 11,
      completion_tokens: 7,
      total_tokens: 18,
      prompt_tokens_details: {
        audio_tokens: 2,
        cached_tokens: 3,
        cache_write_tokens: 5,
      },
      completion_tokens_details: {
        audio_tokens: 4,
        reasoning_tokens: 6,
      },
    };

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createOpenAIStreamChunk('answer', 'stop');
      yield {
        id: 'chatcmpl-openrouter-usage',
        object: 'chat.completion.chunk',
        created: 0,
        model: 'anthropic/claude-sonnet-test',
        choices: [],
        usage,
      } as OpenAIClient.Chat.Completions.ChatCompletionChunk;
    }

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => streamChunks();

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream([new HumanMessage('hi')]);
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const usageChunk = chunks.find(
      (chunk) => chunk.usage_metadata?.input_token_details?.cache_creation === 5
    );
    expect(usageChunk?.usage_metadata).toEqual({
      input_tokens: 11,
      output_tokens: 7,
      total_tokens: 18,
      input_token_details: {
        audio: 2,
        cache_read: 3,
        cache_creation: 5,
      },
      output_token_details: {
        audio: 4,
        reasoning: 6,
      },
    });
  });

  it('emits OpenRouter callbacks before an early stream break', async () => {
    const model = new CallbackTestChatOpenRouter({
      model: 'openai/gpt-4o-mini',
      apiKey: 'test-key',
      _lc_stream_delay: 1,
    } as OpenRouterFields & { _lc_stream_delay: number });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    const textChunks: string[] = [];
    const callbackTokens: string[] = [];

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createOpenAIStreamChunk('alpha beta gamma');
    }

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => streamChunks();

    const runManager = {
      handleLLMNewToken(token: string): void {
        if (token !== '') {
          callbackTokens.push(token);
        }
      },
    } as unknown as CallbackManagerForLLMRun;

    for await (const chunk of model.streamChunksWithCallbacks(runManager)) {
      if (chunk.text !== '') {
        textChunks.push(chunk.text);
      }
      break;
    }

    expect(textChunks).toEqual(['alpha beta gamma']);
    expect(callbackTokens).toEqual(textChunks);
  });

  it('keeps Anthropic output, residency, compaction, and stream-delay options', () => {
    const contextManagement = {
      edits: [
        {
          type: 'compact_20260112' as const,
          trigger: { type: 'input_tokens' as const, value: 50000 },
        },
      ],
    };
    const model = new CustomAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      apiKey: 'test-key',
      maxTokens: 4096,
      outputConfig: { effort: 'medium' },
      inferenceGeo: 'us',
      contextManagement,
      _lc_stream_delay: 8,
    });
    const defaultModel = new CustomAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      apiKey: 'test-key',
    });

    const params = model.invocationParams({
      outputConfig: { effort: 'low' },
      inferenceGeo: 'eu',
    } as AnthropicCallOptions);

    expect(CustomAnthropic.lc_name()).toBe('LibreChatAnthropic');
    expect(model._lc_stream_delay).toBe(8);
    expect(defaultModel._lc_stream_delay).toBe(25);
    expect(params.output_config).toEqual({ effort: 'low' });
    expect(params.inference_geo).toBe('eu');
    expect(params.context_management).toEqual(contextManagement);
  });

  it('keeps Anthropic beta, MCP, and container request wiring current', () => {
    const contextManagement = {
      edits: [
        {
          type: 'compact_20260112' as const,
          trigger: { type: 'input_tokens' as const, value: 50000 },
        },
      ],
    };
    const mcpServers: AnthropicMCPServerURLDefinition[] = [
      {
        type: 'url',
        url: 'https://example.com/mcp',
        name: 'docs',
      },
    ];
    const model = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      contextManagement,
      betas: ['model-beta'],
    });

    const params = model.invocationParams({
      outputConfig: {
        effort: 'low',
        task_budget: { type: 'token_budget', value: 1024 },
      },
      betas: ['request-beta', 'model-beta'],
      container: 'container_123',
      mcp_servers: mcpServers,
      tools: [
        {
          type: 'tool_search_tool_bm25_20251119',
          name: 'search',
        } as ChatAnthropicToolType,
      ],
    } as AnthropicCallOptions);

    expect(params.betas).toEqual([
      'model-beta',
      'request-beta',
      'advanced-tool-use-2025-11-20',
      'compact-2026-01-12',
      'task-budgets-2026-03-13',
    ]);
    expect(params.container).toBe('container_123');
    expect(params.mcp_servers).toBe(mcpServers);
    expect(params.temperature).toBeUndefined();
    expect(params.top_k).toBeUndefined();
    expect(params.top_p).toBeUndefined();
  });

  it('matches Anthropic Opus 4.7 sampling compatibility checks', () => {
    const thinkingModel = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      thinking: { type: 'enabled', budget_tokens: 1024 },
    });
    const topKModel = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      topK: 5,
    });

    expect(() => thinkingModel.invocationParams()).toThrow(
      'thinking.type="enabled" is not supported for claude-opus-4-7'
    );
    expect(() => topKModel.invocationParams()).toThrow(
      'topK is not supported for claude-opus-4-7'
    );
  });

  it('keeps Bedrock Converse application profiles and service tier passthroughs', () => {
    const applicationInferenceProfile =
      'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
    const model = new CustomChatBedrockConverse({
      ...baseBedrockFields,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      applicationInferenceProfile,
      serviceTier: 'priority',
      _lc_stream_delay: 12,
    });
    const defaultStreamDelayModel = new CustomChatBedrockConverse({
      ...baseBedrockFields,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
    });

    expect(CustomChatBedrockConverse.lc_name()).toBe(
      'LibreChatBedrockConverse'
    );
    expect(model.applicationInferenceProfile).toBe(applicationInferenceProfile);
    expect(model._lc_stream_delay).toBe(12);
    expect(defaultStreamDelayModel._lc_stream_delay).toBe(25);
    expect(model.invocationParams({}).serviceTier).toEqual({
      type: 'priority',
    });
    expect(model.invocationParams({ serviceTier: 'flex' }).serviceTier).toEqual(
      { type: 'flex' }
    );
  });

  it('keeps Google and Vertex thinking configuration wiring offline', () => {
    const thinkingConfig = {
      thinkingLevel: 'HIGH' as const,
      includeThoughts: true,
    };
    const google = new CustomChatGoogleGenerativeAI({
      model: 'models/gemini-3-pro-preview',
      apiKey: 'test-key',
      thinkingConfig,
    });
    const vertex = new ChatVertexAI({
      model: 'gemini-3-pro-preview',
      location: 'global',
      thinkingBudget: -1,
      thinkingConfig,
    });

    expect(CustomChatGoogleGenerativeAI.lc_name()).toBe(
      'LibreChatGoogleGenerativeAI'
    );
    expect(google.model).toBe('gemini-3-pro-preview');
    expect(google._isMultimodalModel).toBe(true);
    expect(google.thinkingConfig).toEqual(thinkingConfig);
    expect(ChatVertexAI.lc_name()).toBe('LibreChatVertexAI');
    expect(vertex.dynamicThinkingBudget).toBe(true);
    expect(vertex.thinkingConfig).toEqual(thinkingConfig);
    expect(vertex.invocationParams({}).maxReasoningTokens).toBe(-1);
  });

  it('uppercases custom OpenAI fetch methods before dispatch', async () => {
    let method: string | undefined;
    const client = new CustomOpenAIClient({
      apiKey: 'test-key',
      fetch: async (_url, init): Promise<Response> => {
        method = init?.method;
        return new Response('{}', { status: 200 });
      },
    });

    const response = await client.fetchWithTimeout(
      'https://example.test/v1/chat/completions',
      { method: 'patch' },
      1000,
      new AbortController()
    );

    expect(response.status).toBe(200);
    expect(method).toBe('PATCH');
    expect(client.abortHandler).toBeDefined();
  });

  it('aborts custom OpenAI fetches when the request timeout elapses', async () => {
    const capturedFetch = createAbortableFetch();
    const client = new CustomOpenAIClient({
      apiKey: 'test-key',
      fetch: capturedFetch.fetch,
    });

    await expectFetchTimeoutAbort(
      client,
      capturedFetch,
      'https://example.test/v1/chat/completions'
    );
  });

  it('aborts custom Azure OpenAI fetches when the request timeout elapses', async () => {
    const capturedFetch = createAbortableFetch();
    const client = new CustomAzureOpenAIClient({
      apiKey: 'test-azure-key',
      apiVersion: '2024-10-21',
      baseURL: 'https://example.test/openai/deployments/test-deployment',
      fetch: capturedFetch.fetch,
    });

    await expectFetchTimeoutAbort(
      client,
      capturedFetch,
      'https://example.test/openai/deployments/test-deployment/chat/completions'
    );
  });

  it('propagates caller abort signals to custom OpenAI fetches', async () => {
    const capturedFetch = createAbortableFetch();
    const client = new CustomOpenAIClient({
      apiKey: 'test-key',
      fetch: capturedFetch.fetch,
    });
    const callerController = new AbortController();
    const requestController = new AbortController();

    const response = client.fetchWithTimeout(
      'https://example.test/v1/chat/completions',
      { method: 'post', signal: callerController.signal },
      1000,
      requestController
    );

    callerController.abort();

    await expect(waitForFetchOutcome(response)).resolves.toBe('rejected');
    expect(requestController.signal.aborted).toBe(true);
    expect(capturedFetch.getSignal()?.aborted).toBe(true);
  });
});

describe('StreamSmoothingOptions type surface', () => {
  it('accepts _lc_stream_delay: 0 across the typed provider options', () => {
    const azure: import('@/types').AzureClientOptions = { _lc_stream_delay: 0 };
    const openrouter: ChatOpenRouterCallOptions &
      import('@/types').StreamSmoothingOptions = { _lc_stream_delay: 0 };
    const mistral: import('@/types').MistralAIClientOptions = {
      model: 'mistral-large-latest',
      _lc_stream_delay: 0,
    };
    expect(azure._lc_stream_delay).toBe(0);
    expect(openrouter._lc_stream_delay).toBe(0);
    expect(mistral._lc_stream_delay).toBe(0);
  });
});

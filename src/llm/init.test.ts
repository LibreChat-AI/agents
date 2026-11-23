import { RunnableBinding } from '@langchain/core/runnables';
import type { ChatOpenAICallOptions } from '@langchain/openai';
import type OpenAI from 'openai';
import type * as t from '@/types';
import { getProviderFamily } from '@/llm/providerRegistry';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { ChatOpenAI, ChatMoonshot } from '@/llm/openai';
import { CustomAnthropic } from '@/llm/anthropic';
import { initializeModel } from '@/llm/init';
import { Providers } from '@/common';

const tools: t.GraphTools = [
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Calculate a result',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string' } },
      },
    },
  },
];

it('preserves the normal Anthropic tool prefix and thinking options on summaries', () => {
  const model = new CustomAnthropic({
    apiKey: 'offline-test',
    model: 'claude-haiku-4-5',
    thinking: { type: 'enabled', budget_tokens: 1024 },
    maxTokens: 2048,
  });
  const normal = initializeModel({
    provider: Providers.ANTHROPIC,
    override: model,
    tools,
  });
  const summary = initializeModel({
    provider: Providers.ANTHROPIC,
    override: model,
    tools,
    disableToolCalling: true,
  });
  if (
    !(normal instanceof RunnableBinding) ||
    !(summary instanceof RunnableBinding)
  )
    throw new Error('Expected tool bindings');
  const normalParams = model.invocationParams(normal.config);
  const summaryParams = model.invocationParams(summary.config);
  expect(normalParams.tool_choice).toBeUndefined();
  expect(summaryParams.tool_choice).toBeUndefined();
  expect(summaryParams.tool_choice).toEqual(normalParams.tool_choice);
  expect(summaryParams.tools).toEqual(normalParams.tools);
  expect(summaryParams.thinking).toEqual(normalParams.thinking);
  expect(summaryParams.max_tokens).toBe(2048);
});

it.each(['auto', 'none', 'calculate'])(
  'preserves the normal Anthropic binding with an unbound client tool_choice of %s',
  (tool_choice) => {
    const clientOptions = {
      apiKey: 'offline-test',
      model: 'claude-haiku-4-5',
      tool_choice,
    };
    const model = new CustomAnthropic(clientOptions);
    const bindTools = jest.spyOn(model, 'bindTools');
    const normal = initializeModel({
      provider: Providers.ANTHROPIC,
      clientOptions,
      override: model,
      tools,
    });
    const summary = initializeModel({
      provider: Providers.ANTHROPIC,
      clientOptions,
      override: model,
      tools,
      disableToolCalling: true,
    });
    if (
      !(normal instanceof RunnableBinding) ||
      !(summary instanceof RunnableBinding)
    )
      throw new Error('Expected tool bindings');

    expect(bindTools).toHaveBeenNthCalledWith(1, tools);
    expect(bindTools).toHaveBeenNthCalledWith(2, tools);
    expect(summary.config).toEqual(normal.config);
    expect(normal.config).not.toHaveProperty('tool_choice');
    expect(model.invocationParams(normal.config).tool_choice).toBeUndefined();
    expect(model.invocationParams(summary.config).tool_choice).toBeUndefined();
    bindTools.mockRestore();
  }
);

it.each(['auto', 'none'] as const)(
  'preserves an explicit Anthropic invocation default of %s on summaries',
  (type) => {
    const model = new CustomAnthropic({
      apiKey: 'offline-test',
      model: 'claude-haiku-4-5',
      invocationKwargs: { tool_choice: { type } },
    });
    const normal = initializeModel({
      provider: Providers.ANTHROPIC,
      override: model,
      tools,
    });
    const summary = initializeModel({
      provider: Providers.ANTHROPIC,
      override: model,
      tools,
      disableToolCalling: true,
    });
    if (
      !(normal instanceof RunnableBinding) ||
      !(summary instanceof RunnableBinding)
    )
      throw new Error('Expected tool bindings');
    const normalParams = model.invocationParams(normal.config);
    const summaryParams = model.invocationParams(summary.config);

    expect(normalParams.tool_choice).toEqual({ type });
    expect(summaryParams.tool_choice).toEqual(normalParams.tool_choice);
    expect(summaryParams.tools).toEqual(normalParams.tools);
  }
);

it('preserves OpenAI tool definitions when disabling summary tool calls', () => {
  const model = new ChatOpenAI({
    apiKey: 'offline-test',
    model: 'gpt-4.1-mini',
  });
  const normal = initializeModel({
    provider: Providers.OPENAI,
    override: model,
    tools,
  });
  const summary = initializeModel({
    provider: Providers.OPENAI,
    override: model,
    tools,
    disableToolCalling: true,
  });
  const summaryOptions = Reflect.get(
    summary,
    'defaultOptions'
  ) as ChatOpenAICallOptions;
  const normalOptions = Reflect.get(
    normal,
    'defaultOptions'
  ) as ChatOpenAICallOptions;
  expect(summaryOptions).toMatchObject({ tool_choice: 'none' });
  expect(summaryOptions.tools).toEqual(normalOptions.tools);
  expect(normalOptions).not.toHaveProperty('tool_choice');
});

it.each([undefined, false])(
  'disables Moonshot summary tool calls without changing normal calls (normal flag: %s)',
  async (disableToolCalling) => {
    const requests: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming[] = [];
    const capturingFetch: typeof fetch = async (_input, init) => {
      if (typeof init?.body !== 'string')
        throw new Error('Expected a JSON request body');
      requests.push(
        JSON.parse(
          init.body
        ) as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
      );
      return new Response(
        JSON.stringify({
          id: 'moonshot-test',
          object: 'chat.completion',
          created: 0,
          model: 'kimi-k2.5',
          choices: [
            {
              index: 0,
              message: { role: 'assistant', content: 'Summary' },
              finish_reason: 'stop',
            },
          ],
        }),
        { headers: { 'Content-Type': 'application/json' } }
      );
    };
    const model = new ChatMoonshot({
      apiKey: 'offline-test',
      model: 'kimi-k2.5',
      maxRetries: 0,
      configuration: {
        baseURL: 'https://api.moonshot.ai/v1',
        fetch: capturingFetch,
      },
    });
    const normal = initializeModel({
      provider: Providers.MOONSHOT,
      override: model,
      tools,
      disableToolCalling,
    });
    const summary = initializeModel({
      provider: Providers.MOONSHOT,
      override: model,
      tools,
      disableToolCalling: true,
    });
    const summaryOptions = Reflect.get(
      summary,
      'defaultOptions'
    ) as ChatOpenAICallOptions;
    const normalOptions = Reflect.get(
      normal,
      'defaultOptions'
    ) as ChatOpenAICallOptions;

    expect(getProviderFamily(Providers.MOONSHOT)).toBe('generic');
    expect(summaryOptions.tool_choice).toBe('none');
    expect(summaryOptions.tools).toEqual(tools);
    expect(normalOptions.tools).toEqual(tools);
    expect(normalOptions).not.toHaveProperty('tool_choice');

    await summary.invoke('Summarize the conversation');
    await normal.invoke('Calculate 2 + 2');

    expect(requests).toHaveLength(2);
    expect(requests[0].tool_choice).toBe('none');
    expect(requests[1]).not.toHaveProperty('tool_choice');
    expect(requests[0].tools).toEqual(tools);
    expect(requests[1].tools).toEqual(tools);
  }
);

it('does not send an unsupported none choice to Bedrock Converse', () => {
  const model = new CustomChatBedrockConverse({
    region: 'us-east-1',
    model: 'anthropic.claude-haiku-4-5-20251001-v1:0',
    credentials: {
      accessKeyId: 'offline-test',
      secretAccessKey: 'offline-test',
    },
  });
  const summary = initializeModel({
    provider: Providers.BEDROCK,
    override: model,
    tools,
    disableToolCalling: true,
  });
  if (!(summary instanceof RunnableBinding))
    throw new Error('Expected a tool binding');
  expect(summary.config).not.toHaveProperty('tool_choice');
  expect(model.invocationParams(summary.config).toolConfig?.tools).toHaveLength(
    1
  );
});

it('leaves a tool-less model untouched', () => {
  const model = new CustomAnthropic({
    apiKey: 'offline-test',
    model: 'claude-haiku-4-5',
  });
  expect(
    initializeModel({
      provider: Providers.ANTHROPIC,
      override: model,
      disableToolCalling: true,
    })
  ).toBe(model);
});

describe('client tool call options on OpenAI-compatible providers', () => {
  const completion = {
    id: 'openai-test',
    object: 'chat.completion',
    created: 0,
    model: 'custom-model',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: 'ok' },
        finish_reason: 'stop',
      },
    ],
  };

  type Capture = {
    requests: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming[];
    fetch: typeof fetch;
  };

  function createCapture(): Capture {
    const requests: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming[] = [];
    const capturingFetch: typeof fetch = async (_input, init) => {
      if (typeof init?.body !== 'string')
        throw new Error('Expected a JSON request body');
      requests.push(
        JSON.parse(
          init.body
        ) as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming
      );
      return new Response(JSON.stringify(completion), {
        headers: { 'Content-Type': 'application/json' },
      });
    };
    return { requests, fetch: capturingFetch };
  }

  function createOptions(
    capturingFetch: typeof fetch,
    extra: Record<string, unknown>
  ): t.OpenAIClientOptions {
    return {
      apiKey: 'offline-test',
      model: 'custom-model',
      streaming: false,
      maxRetries: 0,
      configuration: {
        baseURL: 'https://custom.example/v1',
        fetch: capturingFetch,
      },
      ...extra,
    } as t.OpenAIClientOptions;
  }

  it('sends tool_choice and parallel_tool_calls when tools are bound', async () => {
    const capture = createCapture();
    const model = initializeModel({
      provider: Providers.OPENAI,
      clientOptions: createOptions(capture.fetch, {
        tool_choice: 'required',
        parallel_tool_calls: false,
      }),
      tools,
    });

    await model.invoke('Calculate 2 + 2');

    expect(capture.requests).toHaveLength(1);
    expect(capture.requests[0].tool_choice).toBe('required');
    expect(capture.requests[0].parallel_tool_calls).toBe(false);
    expect(capture.requests[0].tools).toEqual(tools);
  });

  it('forwards a named function tool_choice', async () => {
    const capture = createCapture();
    const model = initializeModel({
      provider: Providers.OPENAI,
      clientOptions: createOptions(capture.fetch, {
        tool_choice: { type: 'function', function: { name: 'calculate' } },
      }),
      tools,
    });

    await model.invoke('Calculate 2 + 2');

    expect(capture.requests[0].tool_choice).toEqual({
      type: 'function',
      function: { name: 'calculate' },
    });
    expect(capture.requests[0]).not.toHaveProperty('parallel_tool_calls');
  });

  it('keeps tool_choice none when tool calling is disabled', async () => {
    const capture = createCapture();
    const model = initializeModel({
      provider: Providers.OPENAI,
      clientOptions: createOptions(capture.fetch, {
        tool_choice: 'required',
        parallel_tool_calls: false,
      }),
      tools,
      disableToolCalling: true,
    });

    await model.invoke('Summarize the conversation');

    expect(capture.requests[0].tool_choice).toBe('none');
    expect(capture.requests[0]).not.toHaveProperty('parallel_tool_calls');
    expect(capture.requests[0].tools).toEqual(tools);
  });

  it('sends neither option on a tool-less call', async () => {
    const capture = createCapture();
    const model = initializeModel({
      provider: Providers.OPENAI,
      clientOptions: createOptions(capture.fetch, {
        tool_choice: 'required',
        parallel_tool_calls: false,
      }),
    });

    await model.invoke('Write a title');

    expect(capture.requests[0]).not.toHaveProperty('tool_choice');
    expect(capture.requests[0]).not.toHaveProperty('parallel_tool_calls');
    expect(capture.requests[0]).not.toHaveProperty('tools');
  });

  it('ignores malformed values', () => {
    const model = new ChatOpenAI({
      apiKey: 'offline-test',
      model: 'gpt-4.1-mini',
    });
    const bindTools = jest.spyOn(model, 'bindTools');
    initializeModel({
      provider: Providers.OPENAI,
      clientOptions: {
        tool_choice: 42,
        parallel_tool_calls: 'false',
      } as unknown as t.OpenAIClientOptions,
      override: model,
      tools,
    });

    expect(bindTools).toHaveBeenCalledWith(tools);
    bindTools.mockRestore();
  });
});

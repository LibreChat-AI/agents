import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createSummarizeNode } from '@/summarization/node';
import { AgentContext } from '@/agents/AgentContext';
import { GraphEvents, Providers } from '@/common';
import { CustomAnthropic } from '@/llm/anthropic';
import { FakeChatModel } from '@/llm/fake';
import { ChatOpenAI } from '@/llm/openai';
import * as events from '@/utils/events';
import * as init from '@/llm/init';

const count: t.TokenCounter = (message) =>
  Math.ceil(JSON.stringify(message.content).length / 4) + 4;

class ReasoningBudgetModel extends FakeChatModel {
  calls: BaseMessage[][] = [];
  outputLimit = 0;
  requiredOutput = 16_000;

  constructor() {
    super({ responses: ['REASONING_CHECKPOINT'] });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    this.calls.push(messages);
    const message =
      this.outputLimit < this.requiredOutput
        ? new AIMessageChunk({
          content: [{ type: 'thinking', thinking: 'PRIVATE_REASONING' }],
          response_metadata: { finish_reason: 'length' },
        })
        : new AIMessageChunk('REASONING_CHECKPOINT');
    yield new ChatGenerationChunk({
      text: typeof message.content === 'string' ? message.content : '',
      message,
    });
  }
}

function fixture({
  provider = Providers.OPENAI,
  options,
  config = {},
  maxContextTokens = 129_200,
}: {
  provider?: Providers;
  options: t.ClientOptions;
  config?: t.SummarizationConfig;
  maxContextTokens?: number;
}) {
  const model = new ReasoningBudgetModel();
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'reasoner',
      provider,
      clientOptions: options,
      instructions: 'Preserve the working context.',
      maxContextTokens,
      summarizationEnabled: true,
      summarizationConfig: {
        prompt: 'Write a checkpoint.',
        retainRecent: { turns: 0 },
        ...config,
      },
    },
    count
  );
  const emitted: Array<{ event: string; data: t.StreamEventData }> = [];
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      emitted.push({ event, data: data as t.StreamEventData });
    });
  jest
    .spyOn(init, 'initializeModel')
    .mockImplementation(({ clientOptions }) => {
      const output = clientOptions as {
        maxTokens?: number;
        maxOutputTokens?: number;
      };
      model.outputLimit = output.maxOutputTokens ?? output.maxTokens ?? 0;
      return model;
    });
  const graph = {
    contentData: [] as t.RunStep[],
    contentIndexMap: new Map<string, number>(),
    isMultiAgent: false,
    dispatchRunStep: jest.fn(async (): Promise<void> => {}),
    dispatchRunStepCompleted: jest.fn(async (): Promise<void> => {}),
    closeRunStep: jest.fn(async (): Promise<void> => {}),
  };
  const node = createSummarizeNode({
    agentContext,
    graph,
    generateStepId: () => ['summary', 0],
  });
  return { model, agentContext, emitted, node };
}

afterEach(() => jest.restoreAllMocks());

const reasoningCases = [
  {
    provider: Providers.OPENAI,
    options: { maxTokens: 64_000, reasoningEffort: 'xhigh' as const },
  },
  {
    provider: Providers.OPENAI,
    options: { maxTokens: 64_000, reasoning: { effort: 'high' as const } },
  },
  {
    provider: Providers.OPENAI,
    options: { maxTokens: 64_000, modelKwargs: { reasoning_effort: 'high' } },
  },
  {
    provider: Providers.OPENAI,
    options: {
      maxTokens: 64_000,
      modelKwargs: { reasoning: { effort: 'high' } },
    },
  },
  {
    provider: Providers.ANTHROPIC,
    options: { maxTokens: 64_000, thinking: { type: 'adaptive' as const } },
  },
  {
    provider: Providers.BEDROCK,
    options: {
      maxTokens: 64_000,
      additionalModelRequestFields: { thinking: { type: 'adaptive' } },
    },
  },
  {
    provider: Providers.OPENROUTER,
    options: { maxTokens: 64_000, reasoning: { enabled: true } },
  },
  {
    provider: Providers.GOOGLE,
    options: {
      maxOutputTokens: 64_000,
      thinkingConfig: { thinkingBudget: -1 },
    },
  },
  {
    provider: Providers.ANTHROPIC,
    options: {
      maxTokens: 64_000,
      invocationKwargs: { thinking: { type: 'adaptive' } },
    },
  },
  {
    provider: Providers.OPENROUTER,
    options: { maxTokens: 64_000, include_reasoning: true },
  },
] as const;

it.each(reasoningCases)(
  'retains reasoning output room and sends the protected input once: %j',
  async ({ provider, options }) => {
    const before = JSON.stringify(options);
    const { model, node, agentContext } = fixture({ provider, options });
    const head = new HumanMessage({
      id: 'protected-head',
      content: 'x'.repeat(400_000),
    });
    const result = await node(
      {
        messages: [head],
        summarizationRequest: {
          agentId: 'reasoner',
          remainingContextTokens: 0,
          reason: 'manual',
          windowMessages: [head],
        },
      },
      {}
    );
    expect(model.outputLimit).toBe(64_000);
    expect(model.calls).toHaveLength(1);
    expect(
      model.calls[0].some(
        (message) => message.id === head.id || message.text.includes(head.text)
      )
    ).toBe(true);
    expect(
      model.calls[0].reduce((total, message) => total + count(message), 0)
    ).toBeLessThanOrEqual(129_200);
    expect(agentContext.getSummaryText()).toBe('REASONING_CHECKPOINT');
    expect(result.messages?.[0].getType()).toBe('remove');
    expect(JSON.stringify(options)).toBe(before);
    expect(init.initializeModel).toHaveBeenCalledWith(
      expect.objectContaining({
        clientOptions: expect.objectContaining(options),
      })
    );
  }
);

it.each([
  { maxTokens: 64_000 },
  { maxTokens: 64_000, reasoningEffort: 'none' as const },
  { maxTokens: 64_000, reasoning: { effort: 'none' as const } },
  { maxTokens: 64_000, thinking: { type: 'disabled' as const } },
])(
  'keeps the independent default cap when reasoning is absent or disabled: %j',
  async (options) => {
    const { node, model } = fixture({ options });
    model.requiredOutput = 0;
    const messages = [new HumanMessage('Visible history')];
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'reasoner',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.outputLimit).toBe(4096);
    expect(model.calls).toHaveLength(1);
  }
);

it.each([
  { maxSummaryTokens: 1024 },
  { parameters: { maxSummaryTokens: 1024 } },
  { parameters: { maxTokens: 1024 } },
])(
  'honors an explicit summary cap even when inherited reasoning needs more room: %j',
  async (config) => {
    const { node, model, agentContext, emitted } = fixture({
      options: { maxTokens: 64_000, reasoningEffort: 'high' },
      config,
    });
    const messages = [new HumanMessage('Visible history')];
    const before = JSON.stringify(messages);
    const result = await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'reasoner',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.outputLimit).toBe(1024);
    expect(model.calls).toHaveLength(1);
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(
      emitted.filter(({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA)
    ).toHaveLength(0);
  }
);

it.each([
  { maxContextTokens: 200_000, fits: true },
  { maxContextTokens: 129_200, fits: false },
])(
  'reserves the actual reasoning allowance only for a full-window ceiling: %j',
  async ({ maxContextTokens, fits }) => {
    const { node, model, agentContext } = fixture({
      options: { maxTokens: 64_000, reasoningEffort: 'high' },
      config: { contextBudgetMode: 'window' },
      maxContextTokens,
    });
    const messages = [
      new HumanMessage({ id: 'protected-head', content: 'x'.repeat(400_000) }),
    ];
    const before = JSON.stringify(messages);
    const result = await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'reasoner',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.outputLimit).toBe(64_000);
    expect(model.calls).toHaveLength(fits ? 1 : 0);
    if (fits) {
      expect(
        model.calls[0].reduce(
          (total, message) => total + count(message),
          64_000
        )
      ).toBeLessThanOrEqual(maxContextTokens);
      expect(result.messages?.[0].getType()).toBe('remove');
    } else {
      expect(result.messages).toBeUndefined();
      expect(agentContext.getSummaryText()).toBeUndefined();
    }
    expect(JSON.stringify(messages)).toBe(before);
  }
);

it('sends the preserved total allowance to an OpenAI reasoning transport', async () => {
  const requests: Array<{
    max_completion_tokens?: number;
    max_tokens?: number;
    reasoning_effort?: string;
  }> = [];
  const fetch = jest.fn<
    ReturnType<typeof globalThis.fetch>,
    Parameters<typeof globalThis.fetch>
  >(async (_input, init) => {
    if (typeof init?.body !== 'string')
      throw new Error('Expected a JSON request body');
    requests.push(JSON.parse(init.body));
    return Response.json({
      id: 'offline',
      created: 0,
      object: 'chat.completion',
      model: 'gpt-5.4',
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'REASONING_CHECKPOINT' },
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    });
  });
  const model = new ChatOpenAI({
    apiKey: 'offline-test',
    model: 'gpt-5.4',
    useResponsesApi: false,
    maxTokens: 64_000,
    reasoning: { effort: 'high' },
    maxRetries: 0,
    configuration: { fetch, maxRetries: 0 },
  });
  await model.invoke([new HumanMessage('Write a checkpoint.')]);
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    max_completion_tokens: 64_000,
    reasoning_effort: 'high',
  });
});

it('keeps adaptive Anthropic thinking and its preserved total output allowance on the wire', () => {
  const model = new CustomAnthropic({
    apiKey: 'offline-test',
    model: 'claude-sonnet-4-5',
    maxTokens: 64_000,
    thinking: { type: 'adaptive' },
  });
  expect(model.invocationParams()).toMatchObject({
    max_tokens: 64_000,
    thinking: { type: 'adaptive' },
  });
});

it.each([
  {
    provider: Providers.ANTHROPIC,
    options: {
      maxTokens: 64_000,
      thinking: { type: 'enabled' as const, budget_tokens: 8192 },
      invocationKwargs: { thinking: { type: 'disabled' } },
    },
  },
  {
    provider: Providers.OPENROUTER,
    options: {
      maxTokens: 64_000,
      include_reasoning: true,
      reasoning: { enabled: false },
    },
  },
  {provider: Providers.OPENROUTER, options: {maxTokens: 64_000, include_reasoning: true, reasoning: {effort: 'none' as const}}},
])(
  'honors effective reasoning disables rather than stale lower-priority settings: %j',
  async ({ provider, options }) => {
    const { node, model } = fixture({ provider, options });
    model.requiredOutput = 0;
    const messages = [new HumanMessage('Visible history')];
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'reasoner',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.outputLimit).toBe(4096);
    expect(model.calls).toHaveLength(1);
  }
);

it('keeps a fixed thinking allowance above an inherited smaller agent cap', async () => {
  const {node, model} = fixture({provider: Providers.ANTHROPIC, options: {maxTokens: 1024,
    thinking: {type: 'enabled', budget_tokens: 8192},
  }});
  model.requiredOutput = 0;
  const messages = [new HumanMessage('Visible history')];
  await node({messages, summarizationRequest: {agentId: 'reasoner', remainingContextTokens: 0, windowMessages: messages}}, {});
  expect(model.outputLimit).toBe(12_288);
  expect(model.calls).toHaveLength(1);
});

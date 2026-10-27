import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {ConverseStreamCommandOutput, ConverseCommandOutput, ConverseStreamOutput} from '@aws-sdk/client-bedrock-runtime';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { createSummarizeNode } from '@/summarization/node';
import { resolveClientOptionsModel } from '@/llm/request';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { resolveStreamLimits } from '@/llm/streamLimits';
import { AgentContext } from '@/agents/AgentContext';
import { createTokenCounter } from '@/utils/tokens';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { FakeChatModel } from '@/llm/fake';
import * as events from '@/utils/events';
import * as init from '@/llm/init';
import { Run } from '@/run';

const count: t.TokenCounter = (message) =>
  Math.ceil(JSON.stringify(message.content).length / 4) + 4;

class SummaryModel extends FakeChatModel {
  readonly calls: BaseMessage[][] = [];
  failure?: (messages: BaseMessage[], attempt: number) => Error | undefined;
  summaryUsage?: UsageMetadata;

  constructor() {
    super({ responses: ['CLEAN_SUMMARY'] });
  }

  async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    this.calls.push(messages);
    const error = this.failure?.(messages, this.calls.length);
    if (error != null) {
      yield new ChatGenerationChunk({
        text: 'FAILED_ONLY',
        message: new AIMessageChunk({ content: 'FAILED_ONLY' }),
      });
      throw error;
    }
    yield new ChatGenerationChunk({
      text: 'CLEAN_',
      message: new AIMessageChunk({ content: 'CLEAN_' }),
    });
    yield new ChatGenerationChunk({
      text: 'SUMMARY',
      message: new AIMessageChunk({
        content: 'SUMMARY',
        usage_metadata: this.summaryUsage,
      }),
    });
  }
}

class ReasoningOnlyModel extends SummaryModel {
  constructor(
    private readonly reasoningAttempts: number,
    private readonly stopReason?: string
  ) {
    super();
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.calls.length >= this.reasoningAttempts) {
      yield* super._streamResponseChunks(messages);
      return;
    }
    this.calls.push(messages);
    yield new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: [{ type: 'thinking', thinking: 'PRIVATE_REASONING_ONLY' }],
        additional_kwargs:
          this.stopReason == null ? {} : { stop_reason: this.stopReason },
        usage_metadata: {
          input_tokens: 100,
          output_tokens: 74,
          total_tokens: 174,
        },
      }),
    });
  }
}

function history(turns: number): BaseMessage[] {
  const result: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    result.push(
      new HumanMessage({
        id: `u${i}`,
        content: `question ${i} ${'q'.repeat(100)}`,
      })
    );
    result.push(
      new AIMessage({ id: `a${i}`, content: `answer ${i} ${'a'.repeat(100)}` })
    );
  }
  return result;
}

function harness({
  maxContextTokens = 32_000,
  model = new SummaryModel(),
  provider = Providers.OPENAI,
  clientOptions,
  tokenCounter = count,
  summarizationConfig = { maxSummaryTokens: 128 },
}: {
  maxContextTokens?: number;
  model?: SummaryModel;
  provider?: Providers;
  clientOptions?: t.ClientOptions & Pick<t.BuiltInLLMConfig, 'fallbacks'>;
  tokenCounter?: t.TokenCounter;
  summarizationConfig?: t.SummarizationConfig;
} = {}) {
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'agent',
      provider,
      clientOptions,
      instructions: 'The turn system prompt',
      maxContextTokens,
      summarizationEnabled: true,
      summarizationConfig,
    },
    tokenCounter
  );
  const emitted: Array<{ event: string; data: t.StreamEventData }> = [];
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      emitted.push({ event, data: data as t.StreamEventData });
    });
  jest.spyOn(init, 'initializeModel').mockReturnValue(model);
  const graph = {
    contentData: [] as t.RunStep[],
    contentIndexMap: new Map<string, number>(),
    isMultiAgent: false,
    streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
    dispatchRunStep: jest.fn(async (_step: t.RunStep): Promise<void> => {}),
    dispatchRunStepCompleted: jest.fn(async (): Promise<void> => {}),
    closeRunStep: jest.fn(async (): Promise<void> => {}),
  };
  return {
    agentContext,
    model,
    emitted,
    graph,
    node: createSummarizeNode({
      agentContext,
      graph,
      generateStepId: () => ['summary-step', 0],
    }),
  };
}

afterEach(() => {
  jest.restoreAllMocks();
  jest.useRealTimers();
});

describe('compaction request windows', () => {
  it.each(['manual', 'trigger'] as const)(
    'replays only the masked turn window for %s over 700 stored messages',
    async (reason) => {
      const { node, model, agentContext } = harness();
      const messages = history(360);
      const windowMessages = messages.slice(-40);
      const toolCall = new AIMessage({
        content: '',
        tool_calls: [{ id: 'tool', name: 'search', args: {} }],
      });
      const toolResult = new ToolMessage({
        content: 'MASKED_OUTPUT',
        tool_call_id: 'tool',
      });
      windowMessages.splice(3, 0, toolCall, toolResult);
      agentContext.pendingOriginalToolContent = new Map([
        [3, 'FULL_STORED_OUTPUT'.repeat(10_000)],
      ]);
      const expectedPrefix =
        await agentContext.systemRunnable!.invoke(windowMessages);

      const result = await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 1000,
            reason,
            windowMessages,
          },
        },
        {}
      );

      expect(model.calls).toHaveLength(1);
      expect(model.calls[0].slice(0, -1)).toEqual(expectedPrefix);
      expect(model.calls[0].at(-1)).toBeInstanceOf(HumanMessage);
      expect(JSON.stringify(model.calls[0])).not.toContain(
        'FULL_STORED_OUTPUT'
      );
      expect(model.calls[0]).toContain(toolResult);
      expect(model.calls[0]).toContain(windowMessages.at(-1));
      expect(result.messages?.slice(1)).toEqual(windowMessages.slice(-4));
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
      expect(init.initializeModel).toHaveBeenCalledWith(
        expect.objectContaining({ disableToolCalling: true })
      );
    }
  );

  it.each(['manual', 'trigger', 'overflow'] as const)(
    'does not skip %s when pruning leaves only the retained recent turn',
    async (reason) => {
      const { node, model, agentContext } = harness();
      const messages = history(360);
      const windowMessages = messages.slice(reason === 'overflow' ? -4 : -2);
      const result = await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            reason,
            allowSummarization: true,
            windowMessages,
          },
        },
        {}
      );
      expect(model.calls).toHaveLength(1);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
      expect(result.messages?.slice(1)).toEqual(windowMessages);
    }
  );

  it.each(['manual', 'trigger', 'overflow'] as const)(
    'keeps paired recent tool results when the %s window starts mid-turn',
    async (reason) => {
      const { node, model } = harness({
        summarizationConfig: {
          maxSummaryTokens: 128,
          retainRecent: { tokens: 30 },
        },
      });
      const windowMessages: BaseMessage[] = [];
      for (let i = 0; i < 4; i++) {
        windowMessages.push(
          new AIMessage({
            content: '',
            tool_calls: [{ id: `call-${i}`, name: 'search', args: {} }],
          })
        );
        windowMessages.push(
          new ToolMessage({ content: `result ${i}`, tool_call_id: `call-${i}` })
        );
      }
      const messages = [...history(100), ...windowMessages];
      const result = await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            reason,
            allowSummarization: true,
            windowMessages,
          },
        },
        {}
      );
      expect(model.calls).toHaveLength(1);
      expect(result.messages?.slice(1)).toEqual(windowMessages.slice(-4));
      expect(result.messages?.at(-1)).toBe(windowMessages.at(-1));
    }
  );

  it('honors an explicit turns-only policy without adding a default token cap', async () => {
    const { node } = harness({
      maxContextTokens: 100_000,
      summarizationConfig: { retainRecent: { turns: 3 } },
    });
    const messages = history(5);
    for (const message of messages) message.content = 'context '.repeat(1750);
    const result = await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(result.messages?.slice(1)).toEqual(messages.slice(-6));
  });

  it('reserves instruction and output space without splitting parallel tool calls', async () => {
    const { node, model, agentContext } = harness({ maxContextTokens: 2_000 });
    const messages = history(30);
    messages.push(new HumanMessage('newest exchange'));
    messages.push(
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'a', name: 'search', args: {} },
          { id: 'b', name: 'search', args: {} },
        ],
      })
    );
    messages.push(
      new ToolMessage({ content: 'x'.repeat(20_000), tool_call_id: 'a' })
    );
    messages.push(
      new ToolMessage({ content: 'y'.repeat(20_000), tool_call_id: 'b' })
    );
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.calls).toHaveLength(1);
    const sent = model.calls[0];
    expect(
      sent.reduce((sum, message) => sum + count(message), 128) +
        agentContext.toolSchemaTokens
    ).toBeLessThanOrEqual(2_000);
    const toolIds = sent
      .filter(
        (message): message is ToolMessage => message instanceof ToolMessage
      )
      .map((message) => message.tool_call_id);
    expect(toolIds).toEqual(['a', 'b']);
    expect(
      sent.some(
        (message) =>
          message instanceof AIMessage && message.tool_calls?.length === 2
      )
    ).toBe(true);
  });

  it('shrinks an overflow window before sending it and retries context rejections smaller', async () => {
    const { node, model } = harness();
    const messages = history(30);
    model.failure = (sent) =>
      sent.length > 24
        ? new Error('maximum context length exceeded')
        : undefined;
    const result = await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          reason: 'overflow',
          allowSummarization: true,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.calls.length).toBeGreaterThan(1);
    expect(model.calls[0].length).toBeLessThan(messages.length);
    for (let i = 1; i < model.calls.length; i++)
      expect(model.calls[i].length).toBeLessThan(model.calls[i - 1].length);
    for (const sent of model.calls) expect(sent).toContain(messages.at(-1));
    expect(result.messages?.[0].getType()).toBe('remove');
  });

  it('discards partial output from an overloaded attempt before retrying', async () => {
    jest.useFakeTimers();
    const { node, model, emitted, graph } = harness();
    model.failure = (_sent, attempt) =>
      attempt === 1
        ? new Error(
          'Our servers are currently overloaded. Please try again later.'
        )
        : undefined;
    const messages = history(6);
    const pending = node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    await jest.runAllTimersAsync();
    await pending;
    expect(model.calls).toHaveLength(2);
    const deltas = emitted.filter(
      (event) => event.event === GraphEvents.ON_SUMMARIZE_DELTA
    );
    expect(deltas).toHaveLength(2);
    expect(JSON.stringify(deltas)).not.toContain('FAILED_ONLY');
    expect(graph.dispatchRunStep).toHaveBeenCalledTimes(1);
    const completed = emitted.filter(
      (event) => event.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(completed).toHaveLength(1);
    expect(JSON.stringify(completed)).toContain('CLEAN_SUMMARY');
  });

  it.each([
    [
      Providers.ANTHROPIC,
      { thinking: { type: 'enabled', budget_tokens: 1024 } },
    ],
    [
      Providers.BEDROCK,
      {
        additionalModelRequestFields: {
          thinking: { type: 'enabled', budget_tokens: 1024 },
        },
      },
    ],
    [
      Providers.OPENAI,
      { modelKwargs: { thinking: { type: 'enabled', budget_tokens: 1024 } } },
    ],
    [
      Providers.BEDROCK,
      {
        additionalModelRequestFields: { thinking: true, thinkingBudget: 1024 },
      },
    ],
  ] as const)(
    'reserves inherited thinking plus visible summary output for %s',
    async (provider, clientOptions) => {
      const { node, agentContext } = harness({
        maxContextTokens: 3_000,
        provider,
        clientOptions,
        summarizationConfig: {},
      });
      const messages = history(6);
      await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      expect(init.initializeModel).toHaveBeenCalledWith(
        expect.objectContaining({
          clientOptions: expect.objectContaining({ maxTokens: 1324 }),
        })
      );
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    }
  );

  it('keeps summary callback echoes out of ordinary message steps during real streamed retries', async () => {
    const model = new SummaryModel();
    model.failure = (_messages, attempt) =>
      attempt === 1
        ? new Error('Our servers are currently overloaded')
        : undefined;
    jest.spyOn(init, 'initializeModel').mockReturnValue(model);
    let ordinaryDeltaCount = 0;
    const summaries: t.SummarizeCompleteEvent[] = [];
    const run = await Run.create<t.IState>({
      runId: 'summary-retry-callbacks',
      streamLimits: { maxDeltaEventsPerTurn: 2 },
      tokenCounter: count,
      returnContent: true,
      skipCleanup: true,
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI },
        instructions: 'Test system prompt',
        maxContextTokens: 32_000,
        summarizationEnabled: true,
        summarizeOnly: true,
        summarizationConfig: {
          maxSummaryTokens: 128,
          retainRecent: { turns: 0 },
        },
      },
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            ordinaryDeltaCount++;
          },
        },
        [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
          handle: (_event, data): void => {
            summaries.push(data as t.SummarizeCompleteEvent);
          },
        },
      },
    });
    await run.processStream(
      { messages: history(6) },
      {
        configurable: { thread_id: 'summary-retry' },
        version: 'v2',
      }
    );
    expect(model.calls).toHaveLength(2);
    expect(ordinaryDeltaCount).toBe(0);
    expect(run.Graph?.contentData).toHaveLength(1);
    expect(summaries).toHaveLength(1);
    expect(JSON.stringify(summaries)).toContain('CLEAN_SUMMARY');
    expect(JSON.stringify(summaries)).not.toContain('FAILED_ONLY');
  });

  it('does not commit a summary when cancellation arrives during final token counting', async () => {
    const controller = new AbortController();
    const reason = new DOMException('Cancelled by user', 'AbortError');
    const { node, agentContext, model } = harness({
      tokenCounter: (message) => {
        if (String(message.content).includes('CLEAN_SUMMARY'))
          controller.abort(reason);
        return count(message);
      },
    });
    const messages = history(6);
    await expect(
      node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        { signal: controller.signal }
      )
    ).rejects.toBe(reason);
    expect(model.calls).toHaveLength(1);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(agentContext.summaryVersion).toBe(0);
  });

  it.each([
    { maxContextTokens: 3_000, maxTokens: 1_000 },
    { maxContextTokens: 2_000, maxTokens: 64 },
  ])('uses the serving fallback allowance: %j', async (fallbackLimits) => {
    const primary = new SummaryModel();
    const fallback = new SummaryModel();
    const { node, agentContext } = harness({
      maxContextTokens: 2_000,
      model: primary,
      clientOptions: {
        model: 'primary',
        fallbacks: [
          {
            provider: Providers.OPENAI,
            maxContextTokens: fallbackLimits.maxContextTokens,
            clientOptions: {
              model: 'fallback',
              maxTokens: fallbackLimits.maxTokens,
            },
          },
        ],
      },
      summarizationConfig: {
        maxSummaryTokens: 1_000,
        prompt: 'Write a checkpoint.',
        retainRecent: { turns: 0 },
      },
    });
    jest
      .spyOn(init, 'initializeModel')
      .mockImplementation(({ clientOptions }) =>
        resolveClientOptionsModel(clientOptions) === 'fallback'
          ? fallback
          : primary
      );
    const newest = new HumanMessage('q'.repeat(7_000));
    const messages = [
      new HumanMessage('old'),
      new AIMessage('old answer'),
      newest,
    ];
    const result = await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          reason: 'manual',
          windowMessages: messages,
        },
      },
      {}
    );
    expect(primary.calls).toHaveLength(0);
    expect(fallback.calls).toHaveLength(1);
    expect(fallback.calls[0]).toContain(newest);
    expect(fallback.calls[0][0].getType()).toBe('system');
    expect(
      fallback.calls[0].reduce(
        (total, message) => total + count(message),
        fallbackLimits.maxTokens
      )
    ).toBeLessThanOrEqual(fallbackLimits.maxContextTokens);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    expect(result.messages?.[0].getType()).toBe('remove');
  });

  it('skips a too-small fallback without preventing a viable later fallback', async () => {
    const primary = new SummaryModel();
    const small = new SummaryModel();
    const large = new SummaryModel();
    const { node, agentContext } = harness({
      maxContextTokens: 2_000,
      model: primary,
      clientOptions: {
        model: 'primary',
        fallbacks: [
          {
            provider: Providers.OPENAI,
            maxContextTokens: 1_000,
            clientOptions: { model: 'small', maxTokens: 64 },
          },
          {
            provider: Providers.ANTHROPIC,
            maxContextTokens: 6_000,
            clientOptions: { model: 'large', maxTokens: 128 },
          },
        ],
      },
      summarizationConfig: {
        maxSummaryTokens: 1_000,
        prompt: 'Write a checkpoint.',
        retainRecent: { turns: 0 },
      },
    });
    jest
      .spyOn(init, 'initializeModel')
      .mockImplementation(({ clientOptions }) => {
        if (resolveClientOptionsModel(clientOptions) === 'small') return small;
        return resolveClientOptionsModel(clientOptions) === 'large'
          ? large
          : primary;
      });
    const messages = [new HumanMessage('q'.repeat(7_000))];
    const before = JSON.stringify(messages);
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          reason: 'manual',
          windowMessages: messages,
        },
      },
      {}
    );
    expect(primary.calls).toHaveLength(0);
    expect(small.calls).toHaveLength(0);
    expect(large.calls).toHaveLength(1);
    expect(large.calls[0]).toContain(messages[0]);
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
  });

  it('publishes only the successful fallback’s deltas before completion', async () => {
    const primary = new SummaryModel();
    const failed = new SummaryModel();
    const successful = new SummaryModel();
    primary.failure = () => new Error('Primary failed');
    failed.failure = () => new Error('Fallback failed');
    const { node, emitted } = harness({
      model: primary,
      clientOptions: {
        fallbacks: [
          { provider: Providers.OPENAI, clientOptions: { model: 'failed' } },
          {
            provider: Providers.ANTHROPIC,
            clientOptions: { model: 'successful' },
          },
        ],
      },
    });
    jest
      .spyOn(init, 'initializeModel')
      .mockImplementation(({ clientOptions }) => {
        const name = resolveClientOptionsModel(clientOptions);
        if (name === 'failed') return failed;
        return name === 'successful' ? successful : primary;
      });
    const messages = history(6);
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(primary.calls).toHaveLength(1);
    expect(failed.calls).toHaveLength(1);
    expect(successful.calls).toHaveLength(1);
    const deltas = emitted.filter(
      ({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA
    );
    expect(deltas).toHaveLength(2);
    expect(JSON.stringify(deltas)).not.toContain('FAILED_ONLY');
    expect(deltas.map(({ data }) => data)).toEqual([
      expect.objectContaining({
        delta: {
          summary: expect.objectContaining({
            provider: Providers.ANTHROPIC,
            model: 'successful',
            content: [{ type: 'text', text: 'CLEAN_' }],
          }),
        },
      }),
      expect.objectContaining({
        delta: {
          summary: expect.objectContaining({
            provider: Providers.ANTHROPIC,
            model: 'successful',
            content: [{ type: 'text', text: 'SUMMARY' }],
          }),
        },
      }),
    ]);
    expect(
      emitted.findIndex(
        ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
      )
    ).toBeGreaterThan(
      emitted
        .map(({ event }) => event)
        .lastIndexOf(GraphEvents.ON_SUMMARIZE_DELTA)
    );
  });

  it('does not flush failed fallback chunks when a later non-streaming fallback succeeds', async () => {
    const primary = new SummaryModel();
    const failed = new SummaryModel();
    primary.failure = () => new Error('Primary failed');
    failed.failure = () => new Error('Fallback failed');
    const { node, emitted, agentContext } = harness({
      model: primary,
      clientOptions: {
        fallbacks: [
          { provider: Providers.OPENAI, clientOptions: { model: 'failed' } },
          {
            provider: Providers.OPENAI,
            clientOptions: { model: 'invoke-only' },
          },
        ],
      },
    });
    const invokeOnly = new SummaryModel();
    Object.defineProperty(invokeOnly, 'stream', { value: undefined });
    jest
      .spyOn(invokeOnly, 'invoke')
      .mockResolvedValue(new AIMessageChunk('CLEAN_SUMMARY'));
    jest
      .spyOn(init, 'initializeModel')
      .mockImplementation(({ clientOptions }) => {
        const name = resolveClientOptionsModel(clientOptions);
        if (name === 'failed') return failed;
        return name === 'invoke-only' ? invokeOnly : primary;
      });
    const messages = history(6);
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    expect(
      emitted.filter(({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA)
    ).toHaveLength(0);
    expect(
      JSON.stringify(
        emitted.filter(
          ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
        )
      )
    ).not.toContain('FAILED_ONLY');
  });

  it.each([
    { provider: Providers.ANTHROPIC, options: { model: 'production' } },
    {
      provider: Providers.OPENROUTER,
      options: { modelName: 'anthropic/claude-sonnet-4' },
    },
  ])(
    'bounds CJK input using the serving fallback tokenizer: %j',
    async ({ provider, options }) => {
      const openaiCount = await createTokenCounter('o200k_base');
      const claudeCount = await createTokenCounter('claude');
      const primary = new SummaryModel();
      const fallback = new SummaryModel();
      primary.failure = () => new Error('Primary failed');
      fallback.failure = (sent) =>
        sent.reduce((total, message) => total + claudeCount(message), 64) > 512
          ? new Error('maximum context length exceeded')
          : undefined;
      const { node, agentContext } = harness({
        model: primary,
        maxContextTokens: 512,
        tokenCounter: openaiCount,
        clientOptions: {
          model: 'gpt-4o',
          fallbacks: [
            {
              provider,
              maxContextTokens: 512,
              clientOptions: { ...options, maxTokens: 64 },
            },
          ],
        },
        summarizationConfig: {
          prompt: 'Write a checkpoint.',
          maxSummaryTokens: 64,
          retainRecent: { turns: 0 },
        },
      });
      jest
        .spyOn(init, 'initializeModel')
        .mockImplementation(({ provider: serving }) =>
          serving === provider ? fallback : primary
        );
      const text =
        '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다.';
      const messages: BaseMessage[] = [];
      for (let i = 0; i < 12; i++)
        messages.push(new HumanMessage(text), new AIMessage(text));
      const newest = new HumanMessage('Continue.');
      messages.push(newest);
      expect(claudeCount(messages[0])).toBeGreaterThan(
        openaiCount(messages[0])
      );
      await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      expect(fallback.calls).toHaveLength(1);
      expect(fallback.calls[0]).toContain(newest);
      expect(
        fallback.calls[0].reduce(
          (total, message) => total + claudeCount(message),
          64
        )
      ).toBeLessThanOrEqual(512);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    }
  );

  it('reserves no unsent system or prior-summary overhead for a dedicated provider', async () => {
    const { node, model, agentContext } = harness({
      maxContextTokens: 2_000,
      summarizationConfig: {
        provider: Providers.ANTHROPIC,
        model: 'production',
        prompt: 'Checkpoint.',
        updatePrompt: 'Update checkpoint.',
        maxSummaryTokens: 128,
        retainRecent: { turns: 0 },
      },
    });
    agentContext.systemMessageTokens = 3_000;
    agentContext.setInitialSummary('Prior checkpoint.', 1_000);
    const messages = history(4);
    await node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          reason: 'manual',
          windowMessages: messages,
        },
      },
      {}
    );
    expect(model.calls).toHaveLength(1);
    expect(
      model.calls[0].some((message) => message.getType() === 'system')
    ).toBe(false);
    expect(model.calls[0].at(-1)?.content).toContain('Prior checkpoint.');
    const claudeCount = await createTokenCounter('claude');
    expect(
      model.calls[0].reduce(
        (total, message) => total + claudeCount(message),
        128
      )
    ).toBeLessThanOrEqual(2_000);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
  });

  it.each(['manual', 'trigger'] as const)(
    'retries reasoning-only %s summaries without changing the cached request or leaking partial output',
    async (reason) => {
      jest.useFakeTimers();
      const model = new ReasoningOnlyModel(1);
      const { node, emitted, agentContext, graph } = harness({
        model,
        provider: Providers.ANTHROPIC,
      });
      const messages = history(6);
      const pending = node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            reason,
            windowMessages: messages,
          },
        },
        {}
      );
      await jest.runAllTimersAsync();
      const result = await pending;
      expect(model.calls).toHaveLength(2);
      expect(model.calls[1]).toEqual(model.calls[0]);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
      expect(agentContext.summarizationFailures).toBe(0);
      expect(graph.dispatchRunStep).toHaveBeenCalledTimes(1);
      expect(result.messages?.[0].getType()).toBe('remove');
      const deltas = emitted.filter(
        ({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA
      );
      expect(deltas).toHaveLength(2);
      expect(JSON.stringify(deltas)).not.toContain('PRIVATE_REASONING_ONLY');
      expect(
        emitted.filter(
          ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
        )
      ).toHaveLength(1);
    }
  );

  it('keeps history and publishes no reasoning after reasoning-only retry exhaustion', async () => {
    jest.useFakeTimers();
    const model = new ReasoningOnlyModel(3);
    const { node, emitted, agentContext } = harness({
      model,
      provider: Providers.ANTHROPIC,
    });
    const messages = history(6);
    const before = JSON.stringify(messages);
    const pending = node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    await jest.runAllTimersAsync();
    const result = await pending;
    expect(model.calls).toHaveLength(3);
    expect(result.messages).toBeUndefined();
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(agentContext.summarizationFailures).toBe(1);
    expect(JSON.stringify(messages)).toBe(before);
    expect(
      emitted.filter(({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA)
    ).toHaveLength(0);
    const completed = emitted.filter(
      ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(completed).toHaveLength(1);
    expect(completed[0].data).toMatchObject({ errorCode: 'unknown' });
    expect(JSON.stringify(completed)).not.toContain('PRIVATE_REASONING_ONLY');
  });

  it('tries the configured fallback after reasoning-only primary retries are exhausted', async () => {
    jest.useFakeTimers();
    const primary = new ReasoningOnlyModel(3);
    const fallback = new SummaryModel();
    const { node, emitted, agentContext } = harness({
      model: primary,
      clientOptions: {
        fallbacks: [
          {
            provider: Providers.ANTHROPIC,
            clientOptions: { model: 'fallback' },
          },
        ],
      },
    });
    jest
      .spyOn(init, 'initializeModel')
      .mockImplementation(({ provider }) =>
        provider === Providers.ANTHROPIC ? fallback : primary
      );
    const messages = history(6);
    const pending = node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          windowMessages: messages,
        },
      },
      {}
    );
    await jest.runAllTimersAsync();
    await pending;
    expect(primary.calls).toHaveLength(3);
    expect(fallback.calls).toHaveLength(1);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    const deltas = emitted.filter(
      ({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA
    );
    expect(deltas).toHaveLength(2);
    expect(JSON.stringify(deltas)).not.toContain('PRIVATE_REASONING_ONLY');
  });

  it.each([
    { response_metadata: { stop_reason: 'max_tokens' } },
    { response_metadata: { stop_reason: 'refusal' } },
    { response_metadata: { stop_reason: 'stop_sequence' } },
    { additional_kwargs: { refusal: 'Cannot comply' } },
    { tool_calls: [{ id: 'tool', name: 'search', args: {} }] },
  ])(
    'does not retry known truncation, refusal, or tool-use outcomes as reasoning-only: %j',
    async (extra) => {
      const model = new SummaryModel();
      Object.defineProperty(model, 'stream', { value: undefined });
      const invoke = jest.spyOn(model, 'invoke').mockResolvedValue(
        new AIMessageChunk({
          content: [{ type: 'thinking', thinking: 'PRIVATE_REASONING_ONLY' }],
          ...extra,
        })
      );
      const { node, agentContext } = harness({ model });
      const messages = history(6);
      const result = await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      expect(invoke).toHaveBeenCalledTimes(1);
      expect(result.messages).toBeUndefined();
      expect(agentContext.getSummaryText()).toBeUndefined();
    }
  );

  it.each(['stream', 'invoke'] as const)(
    'keeps successful fallback identity and usage together in %s completions',
    async (mode) => {
      const primary = new SummaryModel();
      const failed = new SummaryModel();
      const successful = new SummaryModel();
      primary.failure = () => new Error('Primary failed');
      failed.failure = () => new Error('First fallback failed');
      const usage = { input_tokens: 321, output_tokens: 45, total_tokens: 366 };
      successful.summaryUsage = usage;
      if (mode === 'invoke') {
        Object.defineProperty(successful, 'stream', { value: undefined });
        jest.spyOn(successful, 'invoke').mockResolvedValue(
          new AIMessageChunk({
            content: 'CLEAN_SUMMARY',
            usage_metadata: usage,
          })
        );
      }
      const { node, emitted, graph } = harness({
        model: primary,
        clientOptions: {
          model: 'primary',
          fallbacks: [
            { provider: Providers.OPENAI, clientOptions: { model: 'failed' } },
            {
              provider: Providers.ANTHROPIC,
              clientOptions: { modelName: 'successful' },
            },
          ],
        },
        summarizationConfig: {
          model: 'primary-summary',
          maxSummaryTokens: 128,
        },
      });
      jest
        .spyOn(init, 'initializeModel')
        .mockImplementation(({ clientOptions }) => {
          const name = resolveClientOptionsModel(clientOptions);
          if (name === 'failed') return failed;
          return name === 'successful' ? successful : primary;
        });
      const messages = history(6);
      await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      const completed = emitted.filter(
        ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
      );
      expect(completed).toHaveLength(1);
      expect(completed[0].data).toMatchObject({
        summary: { provider: Providers.ANTHROPIC, model: 'successful' },
      });
      const step = graph.dispatchRunStep.mock.calls[0][0];
      expect(step.summary).toMatchObject({
        provider: Providers.ANTHROPIC,
        model: 'successful',
      });
      expect(step.usage).toEqual({
        prompt_tokens: 321,
        completion_tokens: 45,
        total_tokens: 366,
      });
      expect(graph.dispatchRunStepCompleted).toHaveBeenCalledWith(
        'summary-step',
        expect.objectContaining({
          summary: expect.objectContaining({
            provider: Providers.ANTHROPIC,
            model: 'successful',
          }),
        }),
        expect.anything()
      );
    }
  );

  it.each(['refusal', 'stop_sequence', 'max_tokens'] as const)(
    'does not retry a reasoning-only stream terminated by %s',
    async (stopReason) => {
      jest.useFakeTimers();
      const model = new ReasoningOnlyModel(3, stopReason);
      const { node, emitted, agentContext } = harness({
        model,
        provider: Providers.ANTHROPIC,
      });
      const messages = history(6);
      const pending = node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      await jest.runAllTimersAsync();
      const result = await pending;
      expect(model.calls).toHaveLength(1);
      expect(result.messages).toBeUndefined();
      expect(agentContext.getSummaryText()).toBeUndefined();
      expect(
        emitted.filter(({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA)
      ).toHaveLength(0);
    }
  );

  it.each(['stream', 'invoke'] as const)('preserves history without retrying a native Bedrock %s stop sequence', async mode => {
    const model = new CustomChatBedrockConverse({model: 'anthropic.claude-sonnet-4-5',
      region: 'us-east-1', credentials: {accessKeyId: 'offline-test', secretAccessKey: 'offline-test'},
      _lc_stream_delay: 0,
    });
    let response: ConverseCommandOutput | ConverseStreamCommandOutput;
    if (mode === 'stream') {
      response = {$metadata: {}, stream: (async function* (): AsyncGenerator<ConverseStreamOutput> {
        yield {messageStart: {role: 'assistant' as const}};
        yield {contentBlockDelta: {contentBlockIndex: 0, delta: {reasoningContent: {text: 'PRIVATE_REASONING_ONLY'}}}};
        yield {contentBlockStop: {contentBlockIndex: 0}};
        yield {messageStop: {stopReason: 'stop_sequence' as const}};
        yield {metadata: {usage: {inputTokens: 100, outputTokens: 10, totalTokens: 110}, metrics: {latencyMs: 0}}};
      })()} satisfies ConverseStreamCommandOutput;
    } else {
      Object.defineProperty(model, 'stream', {value: undefined});
      response = {$metadata: {}, output: {message: {role: 'assistant',
        content: [{reasoningContent: {reasoningText: {text: 'PRIVATE_REASONING_ONLY', signature: 'offline-signature'}}}],
      }}, stopReason: 'stop_sequence', usage: {inputTokens: 100, outputTokens: 10, totalTokens: 110},
      metrics: {latencyMs: 0}} satisfies ConverseCommandOutput;
    }
    const send = jest.fn(async () => response);
    Object.defineProperty(model.client, 'send', {value: send});
    const {node, emitted, agentContext, graph} = harness({provider: Providers.BEDROCK});
    graph.streamLimits = resolveStreamLimits({maxDeltaEventsPerTurn: 10});
    jest.spyOn(init, 'initializeModel').mockReturnValue(model);
    const messages = history(6);
    const before = JSON.stringify(messages);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(send).toHaveBeenCalledTimes(1);
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(agentContext.summarizationFailures).toBe(1);
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA)).toHaveLength(0);
    const completed = emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE);
    expect(completed).toHaveLength(1);
    expect(completed[0].data).toMatchObject({errorCode: 'unknown'});
    expect(JSON.stringify(completed)).not.toContain('PRIVATE_REASONING_ONLY');
  });

  const cancellations = [
    Object.assign(new Error('SDK failure'), { name: 'CancelledError' }),
    Object.assign(new Error('SDK failure'), { name: 'CanceledError' }),
    Object.assign(new Error('SDK failure'), { name: 'APIUserAbortError' }),
    Object.assign(new Error('SDK failure'), { code: 'ERR_CANCELED' }),
    Object.assign(new Error('SDK failure'), { code: 'ABORT_ERR' }),
    new Error('Request was cancelled'),
    new Error('Gateway failure', { cause: { code: 'ERR_CANCELED' } }),
  ];
  it.each(
    cancellations.flatMap((error) =>
      ['primary', 'fallback'].map((phase) => ({ phase, error }))
    )
  )(
    'does not recover from an SDK cancellation without an aborted signal: %j',
    async ({ phase, error }) => {
      const primary = new SummaryModel();
      const fallback = new SummaryModel();
      const unused = new SummaryModel();
      primary.failure = () =>
        phase === 'primary' ? error : new Error('Primary failed');
      fallback.failure = () => error;
      const { node, emitted, agentContext } = harness({
        model: primary,
        clientOptions: {
          fallbacks: [
            {
              provider: Providers.OPENAI,
              clientOptions: { model: 'fallback' },
            },
            { provider: Providers.OPENAI, clientOptions: { model: 'unused' } },
          ],
        },
      });
      jest
        .spyOn(init, 'initializeModel')
        .mockImplementation(({ clientOptions }) => {
          const name = resolveClientOptionsModel(clientOptions);
          if (name === 'fallback') return fallback;
          return name === 'unused' ? unused : primary;
        });
      const messages = history(6);
      const before = JSON.stringify(messages);
      const signal = new AbortController().signal;
      await expect(
        node(
          {
            messages,
            summarizationRequest: {
              agentId: 'agent',
              remainingContextTokens: 0,
              windowMessages: messages,
            },
          },
          { signal }
        )
      ).rejects.toBe(error);
      expect(signal.aborted).toBe(false);
      expect(primary.calls).toHaveLength(1);
      expect(fallback.calls).toHaveLength(phase === 'fallback' ? 1 : 0);
      expect(unused.calls).toHaveLength(0);
      expect(JSON.stringify(messages)).toBe(before);
      expect(agentContext.getSummaryText()).toBeUndefined();
      expect(
        emitted.filter(
          ({ event }) =>
            event === GraphEvents.ON_SUMMARIZE_COMPLETE ||
            event === GraphEvents.ON_SUMMARIZE_DELTA
        )
      ).toHaveLength(0);
    }
  );

  it.each(['failure', 'empty'] as const)(
    'discards all fallback deltas after %s',
    async (outcome) => {
      const primary = new SummaryModel();
      const failed = new SummaryModel();
      primary.failure = () => new Error('Primary failed');
      failed.failure = () => new Error('Fallback failed');
      const empty = new SummaryModel();
      Object.defineProperty(empty, 'stream', { value: undefined });
      jest.spyOn(empty, 'invoke').mockResolvedValue(new AIMessageChunk(''));
      const { node, emitted, agentContext } = harness({
        model: primary,
        clientOptions: {
          fallbacks: [
            { provider: Providers.OPENAI, clientOptions: { model: 'failed' } },
            ...(outcome === 'empty'
              ? [
                {
                  provider: Providers.OPENAI,
                  clientOptions: { model: 'empty' },
                },
              ]
              : []),
          ],
        },
      });
      jest
        .spyOn(init, 'initializeModel')
        .mockImplementation(({ clientOptions }) => {
          const name = resolveClientOptionsModel(clientOptions);
          if (name === 'failed') return failed;
          return name === 'empty' ? empty : primary;
        });
      const messages = history(6);
      const before = JSON.stringify(messages);
      const result = await node(
        {
          messages,
          summarizationRequest: {
            agentId: 'agent',
            remainingContextTokens: 0,
            windowMessages: messages,
          },
        },
        {}
      );
      expect(result.messages).toBeUndefined();
      expect(JSON.stringify(messages)).toBe(before);
      expect(agentContext.getSummaryText()).toBeUndefined();
      expect(
        emitted.filter(({ event }) => event === GraphEvents.ON_SUMMARIZE_DELTA)
      ).toHaveLength(0);
      expect(
        JSON.stringify(
          emitted.filter(
            ({ event }) => event === GraphEvents.ON_SUMMARIZE_COMPLETE
          )
        )
      ).not.toContain('FAILED_ONLY');
    }
  );

  it('preserves automatic history and reports a safe code after retry exhaustion', async () => {
    jest.useFakeTimers();
    const { node, model, emitted, agentContext, graph } = harness();
    model.failure = () =>
      new Error(
        'Our servers are currently overloaded. private-provider-detail'
      );
    const messages = history(6);
    const before = JSON.stringify(messages);
    const pending = node(
      {
        messages,
        summarizationRequest: {
          agentId: 'agent',
          remainingContextTokens: 0,
          reason: 'trigger',
          windowMessages: messages,
        },
      },
      {}
    );
    await jest.runAllTimersAsync();
    const result = await pending;
    expect(model.calls).toHaveLength(3);
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(agentContext.summaryVersion).toBe(0);
    expect(graph.closeRunStep).toHaveBeenCalledWith(
      'summary-step',
      'failed',
      expect.anything()
    );
    const completed = emitted.filter(
      (event) => event.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(completed).toHaveLength(1);
    expect(completed[0].data).toMatchObject({
      errorCode: 'provider_overloaded',
    });
    expect(JSON.stringify(completed)).not.toContain('private-provider-detail');
    expect(
      emitted.filter((event) => event.event === GraphEvents.ON_SUMMARIZE_DELTA)
    ).toHaveLength(0);
  });
});

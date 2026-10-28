import { FinishReason } from '@google/generative-ai';
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
import { convertResponseContentToChatGenerationChunk } from '@/llm/google/utils/common';
import { countSummarizationToolTokens } from '@/summarization/toolBudget';
import { prepareProviderRequest } from '@/llm/prepareProviderRequest';
import { addBedrockTailCacheControl } from '@/messages/cache';
import { createSummarizeNode } from '@/summarization/node';
import { resolveClientOptionsModel } from '@/llm/request';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { resolveStreamLimits } from '@/llm/streamLimits';
import { AgentContext } from '@/agents/AgentContext';
import { createTokenCounter } from '@/utils/tokens';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { FakeChatModel } from '@/llm/fake';
import { ChatOpenAI } from '@/llm/openai';
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
    private readonly stopReason?: string,
    private readonly reasoningResponse?: AIMessageChunk
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
      message: this.reasoningResponse ?? new AIMessageChunk({
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
  tools,
  tokenCounter = count,
  summarizationConfig = { maxSummaryTokens: 128 },
}: {
  maxContextTokens?: number;
  model?: SummaryModel;
  provider?: Providers;
  clientOptions?: t.ClientOptions & Pick<t.BuiltInLLMConfig, 'fallbacks'>;
  tokenCounter?: t.TokenCounter;
  tools?: t.GraphTools;
  summarizationConfig?: t.SummarizationConfig;
} = {}) {
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'agent',
      provider,
      clientOptions,
      tools,
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

  it.each(['trigger', 'overflow'] as const)('skips %s when the entire visible window would be retained', async reason => {
    const {node, model, agentContext} = harness();
    const messages = history(360);
    const windowMessages = messages.slice(-2);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason, allowSummarization: true, windowMessages}}, {});
    expect(model.calls).toHaveLength(0);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(result.messages).toBeUndefined();
  });

  it('reports nothing to compact rather than duplicating an entirely retained visible window', async () => {
    const {node, model} = harness();
    const messages = history(360);
    await expect(node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'manual', windowMessages: messages.slice(-2)}}, {}))
      .rejects.toMatchObject({reason: 'nothing_to_summarize'});
    expect(model.calls).toHaveLength(0);
  });

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
    const { node, model, agentContext } = harness({ maxContextTokens: 2_000, summarizationConfig: {maxSummaryTokens: 128, contextBudgetMode: 'window'} });
    const messages = history(15);
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

  it('preserves the visible head when context rejections cannot fit it without dropping messages', async () => {
    const {node, model, agentContext} = harness();
    const messages = history(30);
    const before = JSON.stringify(messages);
    model.failure = sent => sent.length > 24 ? new Error('maximum context length exceeded') : undefined;
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'overflow', allowSummarization: true, windowMessages: messages}}, {});
    expect(model.calls.length).toBeLessThanOrEqual(5);
    for (const sent of model.calls) expect(sent.slice(1, 1 + messages.length - 4)).toEqual(messages.slice(0, -4));
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
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
        maxSummaryTokens: 1_000, contextBudgetMode: 'window',
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
        maxSummaryTokens: 1_000, contextBudgetMode: 'window',
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
      expect(fallback.calls).toHaveLength(0);
      expect(agentContext.getSummaryText()).toBeUndefined();
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

  const reasoningKwargs = [
    {reasoning_content: 'PRIVATE_REASONING_ONLY'},
    {reasoning: 'PRIVATE_REASONING_ONLY'},
    {reasoning: {summary: [{text: 'PRIVATE_REASONING_ONLY'}]}},
    {reasoning_details: [{type: 'reasoning.text', text: 'PRIVATE_REASONING_ONLY'}]},
    {reasoning_details: [{type: 'reasoning.encrypted', data: 'PRIVATE_REASONING_ONLY'}]},
    {reasoning_details: [{type: 'reasoning.summary', summary: 'PRIVATE_REASONING_ONLY'}]},
  ];
  it.each(reasoningKwargs.flatMap(kwargs => ['stream', 'invoke'].map(mode => ({kwargs, mode}))))(
    'recovers kwargs-borne reasoning without exposing it: %j', async ({kwargs, mode}) => {
      jest.useFakeTimers();
      const response = new AIMessageChunk({content: '', additional_kwargs: kwargs, response_metadata: {finish_reason: 'stop'}});
      const model = new ReasoningOnlyModel(1, undefined, response);
      if (mode === 'invoke') {
        Object.defineProperty(model, 'stream', {value: undefined});
        jest.spyOn(model, 'invoke').mockImplementation(async messages => {
          model.calls.push(messages as BaseMessage[]);
          return model.calls.length === 1 ? response : new AIMessageChunk('CLEAN_SUMMARY');
        });
      }
      const {node, emitted, agentContext} = harness({model});
      const messages = history(6);
      const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'manual', windowMessages: messages}}, {});
      await jest.runAllTimersAsync();
      await pending;
      expect(model.calls).toHaveLength(2);
      expect(model.calls[1]).toEqual(model.calls[0]);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
      const output = emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA || event === GraphEvents.ON_SUMMARIZE_COMPLETE);
      expect(JSON.stringify(output)).not.toContain('PRIVATE_REASONING_ONLY');
      expect(output.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toHaveLength(1);
    }
  );

  it.each([
    {content: '', additional_kwargs: {reasoning_content: 'PRIVATE_REASONING_ONLY'}, response_metadata: {finish_reason: 'length'}},
    {content: '', additional_kwargs: {reasoning_content: 'PRIVATE_REASONING_ONLY'}, response_metadata: {finish_reason: 'content_filter'}},
    {content: '', additional_kwargs: {reasoning_content: '', reasoning_details: []}},
    {content: 'CLEAN_SUMMARY', additional_kwargs: {reasoning_content: 'PRIVATE_REASONING_ONLY'}},
  ])('does not retry terminal, empty, or visible responses with kwargs: %j', async fields => {
    jest.useFakeTimers();
    const model = new ReasoningOnlyModel(3, undefined, new AIMessageChunk(fields));
    const {node, emitted, agentContext} = harness({model});
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync();
    const result = await pending;
    expect(model.calls).toHaveLength(1);
    if (fields.content === '') {
      expect(result.messages).toBeUndefined();
      expect(agentContext.getSummaryText()).toBeUndefined();
    } else expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    expect(JSON.stringify(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE))).not.toContain('PRIVATE_REASONING_ONLY');
  });

  it.each(['stream', 'invoke'] as const)('recovers reasoning-only replies through the actual OpenAI %s adapter', async mode => {
    const bodies: string[] = [];
    const fetch = jest.fn<ReturnType<typeof globalThis.fetch>, Parameters<typeof globalThis.fetch>>(async (_input, init) => {
      if (typeof init?.body !== 'string') throw new Error('Expected a JSON request body');
      bodies.push(init.body);
      const first = bodies.length === 1;
      const reply = first ? {content: '', reasoning_content: 'PRIVATE_REASONING_ONLY'} : {content: 'CLEAN_SUMMARY'};
      const base = {id: `offline-${bodies.length}`, created: 0, model: 'gpt-4.1-mini'};
      if (mode === 'invoke') return Response.json({...base, object: 'chat.completion',
        choices: [{index: 0, message: {role: 'assistant', ...reply}, finish_reason: 'stop', logprobs: null}],
        usage: {prompt_tokens: 100, completion_tokens: 10, total_tokens: 110},
      });
      const chunks = [
        {...base, object: 'chat.completion.chunk', choices: [{index: 0, delta: {role: 'assistant', ...reply}, finish_reason: null, logprobs: null}]},
        {...base, object: 'chat.completion.chunk', choices: [{index: 0, delta: {}, finish_reason: 'stop', logprobs: null}]},
      ];
      return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n', {
        headers: {'Content-Type': 'text/event-stream'},
      });
    });
    const model = new ChatOpenAI({apiKey: 'offline-test', model: 'gpt-4.1-mini',
      useResponsesApi: false, streaming: mode === 'stream', maxRetries: 0, _lc_stream_delay: 0,
      configuration: {fetch},
    });
    if (mode === 'invoke') Object.defineProperty(model, 'stream', {value: undefined});
    const {node, emitted, agentContext} = harness();
    jest.spyOn(init, 'initializeModel').mockReturnValue(model);
    const messages = history(6);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'manual', windowMessages: messages}}, {});
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(bodies[1]).toBe(bodies[0]);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    const output = emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA || event === GraphEvents.ON_SUMMARIZE_COMPLETE);
    expect(JSON.stringify(output)).not.toContain('PRIVATE_REASONING_ONLY');
    expect(output.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toHaveLength(1);
  });

  it.each([1, 3])('continues fallback recovery after %i reasoning-only responses', async reasoningAttempts => {
    jest.useFakeTimers();
    const primary = new SummaryModel();
    const first = new ReasoningOnlyModel(reasoningAttempts);
    const last = new SummaryModel();
    primary.failure = () => new Error('Primary failed');
    const {node, emitted, agentContext} = harness({model: primary,
      clientOptions: {fallbacks: [
        {provider: Providers.OPENAI, clientOptions: {model: 'first'}},
        {provider: Providers.ANTHROPIC, clientOptions: {model: 'last'}},
      ]},
    });
    jest.spyOn(init, 'initializeModel').mockImplementation(({clientOptions}) => {
      const model = resolveClientOptionsModel(clientOptions);
      if (model === 'first') return first;
      return model === 'last' ? last : primary;
    });
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync();
    await pending;
    expect(primary.calls).toHaveLength(1);
    expect(first.calls).toHaveLength(reasoningAttempts === 1 ? 2 : 3);
    expect(last.calls).toHaveLength(reasoningAttempts === 3 ? 1 : 0);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    const output = emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA || event === GraphEvents.ON_SUMMARIZE_COMPLETE);
    expect(JSON.stringify(output)).not.toContain('PRIVATE_REASONING_ONLY');
    expect(output.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA)).toHaveLength(2);
    expect(output.find(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)?.data).toMatchObject({summary: {
      provider: reasoningAttempts === 3 ? Providers.ANTHROPIC : Providers.OPENAI,
      model: reasoningAttempts === 3 ? 'last' : 'first',
    }});
  });

  it('does not advance the fallback chain after a terminal reasoning-only response', async () => {
    const primary = new SummaryModel();
    const terminal = new ReasoningOnlyModel(3, 'stop_sequence');
    const unused = new SummaryModel();
    primary.failure = () => new Error('Primary failed');
    const {node, agentContext} = harness({model: primary, clientOptions: {fallbacks: [
      {provider: Providers.OPENAI, clientOptions: {model: 'terminal'}},
      {provider: Providers.ANTHROPIC, clientOptions: {model: 'unused'}},
    ]}});
    jest.spyOn(init, 'initializeModel').mockImplementation(({clientOptions}) => {
      const name = resolveClientOptionsModel(clientOptions);
      if (name === 'terminal') return terminal;
      return name === 'unused' ? unused : primary;
    });
    const messages = history(6);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(terminal.calls).toHaveLength(1);
    expect(unused.calls).toHaveLength(0);
    expect(result.messages).toBeUndefined();
    expect(agentContext.getSummaryText()).toBeUndefined();
  });

  it('retries provider TimeoutError aborts before advancing to a fallback', async () => {
    jest.useFakeTimers();
    const primary = new SummaryModel();
    const fallback = new SummaryModel();
    primary.failure = () => new DOMException('The operation was aborted due to timeout', 'TimeoutError');
    const {node, agentContext} = harness({model: primary, clientOptions: {fallbacks: [
      {provider: Providers.ANTHROPIC, clientOptions: {model: 'fallback'}},
    ]}});
    jest.spyOn(init, 'initializeModel').mockImplementation(({provider}) => provider === Providers.ANTHROPIC ? fallback : primary);
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync();
    await pending;
    expect(primary.calls).toHaveLength(3);
    expect(fallback.calls).toHaveLength(1);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
  });

  it.each([1, 3])('reports all billed usage after %i reasoning-only attempts', async reasoningAttempts => {
    jest.useFakeTimers();
    const model = new ReasoningOnlyModel(reasoningAttempts);
    model.summaryUsage = {input_tokens: 200, output_tokens: 20, total_tokens: 220,
      input_token_details: {cache_read: 150}, output_token_details: {reasoning: 5}};
    const {node, graph, emitted, agentContext} = harness({model, clientOptions: {model: 'reasoner'}});
    const messages = history(6);
    const before = JSON.stringify(messages);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync();
    const result = await pending;
    const completed = emitted.find(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)?.data as t.SummarizeCompleteEvent;
    expect(completed.attemptUsage).toHaveLength(reasoningAttempts === 1 ? 2 : 3);
    expect(completed.attemptUsage?.every(entry => entry.provider === Providers.OPENAI && entry.model === 'reasoner')).toBe(true);
    const step = graph.dispatchRunStep.mock.calls[0][0];
    expect(step.usage).toMatchObject(reasoningAttempts === 1 ? {
      prompt_tokens: 300, completion_tokens: 94, total_tokens: 394,
    } : {prompt_tokens: 300, completion_tokens: 222, total_tokens: 522});
    if (reasoningAttempts === 1) {
      expect(completed.attemptUsage?.[1].usage).toEqual(model.summaryUsage);
      expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    } else {
      expect(result.messages).toBeUndefined();
      expect(JSON.stringify(messages)).toBe(before);
      expect(agentContext.getSummaryText()).toBeUndefined();
      expect(completed.errorCode).toBe('unknown');
      expect(graph.closeRunStep).toHaveBeenCalledWith('summary-step', 'failed', expect.anything());
    }
  });

  it('separates attempt usage across providers and models when multiple fallback retries are billed', async () => {
    jest.useFakeTimers();
    const primary = new ReasoningOnlyModel(3);
    const first = new ReasoningOnlyModel(3);
    const last = new SummaryModel();
    last.summaryUsage = {input_tokens: 321, output_tokens: 45, total_tokens: 366};
    const {node, emitted, graph} = harness({model: primary, clientOptions: {model: 'reasoner', fallbacks: [
      {provider: Providers.ANTHROPIC, clientOptions: {model: 'reasoner'}},
      {provider: Providers.OPENROUTER, clientOptions: {model: 'success'}},
    ]}});
    jest.spyOn(init, 'initializeModel').mockImplementation(({provider}) => {
      if (provider === Providers.ANTHROPIC) return first;
      return provider === Providers.OPENROUTER ? last : primary;
    });
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync();
    await pending;
    const completed = emitted.find(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)?.data as t.SummarizeCompleteEvent;
    expect(completed.attemptUsage).toHaveLength(7);
    expect(completed.attemptUsage?.map(entry => [entry.provider, entry.model])).toEqual([
      ...Array.from({length: 3}, () => [Providers.OPENAI, 'reasoner']),
      ...Array.from({length: 3}, () => [Providers.ANTHROPIC, 'reasoner']),
      [Providers.OPENROUTER, 'success'],
    ]);
    expect(completed.summary).toMatchObject({provider: Providers.OPENROUTER, model: 'success'});
    expect(graph.dispatchRunStep.mock.calls[0][0].usage).toMatchObject({
      prompt_tokens: 921, completion_tokens: 489, total_tokens: 1410,
      attemptUsage: completed.attemptUsage,
    });
  });

  it('does not duplicate model-end callbacks while recording completed-attempt usage', async () => {
    const model = new ReasoningOnlyModel(1);
    model.summaryUsage = {input_tokens: 200, output_tokens: 20, total_tokens: 220};
    jest.spyOn(init, 'initializeModel').mockReturnValue(model);
    const modelEnds: UsageMetadata[] = [];
    const completions: t.SummarizeCompleteEvent[] = [];
    const run = await Run.create<t.IState>({runId: 'summary-attempt-accounting', tokenCounter: count, returnContent: true, skipCleanup: true,
      graphConfig: {type: 'standard', llmConfig: {provider: Providers.OPENAI, model: 'reasoner'},
        maxContextTokens: 32_000, instructions: 'Checkpoint.', summarizeOnly: true, summarizationEnabled: true,
        summarizationConfig: {maxSummaryTokens: 128, retainRecent: {turns: 0}},
      }, customHandlers: {
        [GraphEvents.CHAT_MODEL_END]: {handle: (_event, data, metadata): void => {
          if (metadata?.summarization === true) {
            const usage = (data as t.ModelEndData)?.output?.usage_metadata;
            if (usage != null) modelEnds.push(usage);
          }
        }},
        [GraphEvents.ON_SUMMARIZE_COMPLETE]: {handle: (_event, data): void => {completions.push(data as t.SummarizeCompleteEvent);}},
      },
    });
    await run.processStream({messages: history(6)}, {configurable: {thread_id: 'summary-attempt-accounting'}, version: 'v2'});
    expect(model.calls).toHaveLength(2);
    expect(modelEnds).toHaveLength(2);
    expect(completions).toHaveLength(1);
    expect(completions[0].attemptUsage?.map(entry => entry.usage)).toEqual(modelEnds);
  });

  it.each(['', ' \n\t'])('preserves empty Google thought output without retrying: %j', async text => {
    const thoughtPart = {thought: true, text};
    const response: Parameters<typeof convertResponseContentToChatGenerationChunk>[0] = {
      text: () => '', functionCall: () => undefined, functionCalls: () => undefined,
      candidates: [{index: 0, content: {role: 'model', parts: [thoughtPart]},
        finishReason: FinishReason.STOP, safetyRatings: [],
      }],
    };
    const chunk = convertResponseContentToChatGenerationChunk(response, {index: 0});
    if (!(chunk?.message instanceof AIMessageChunk)) throw new Error('Expected a Google message chunk');
    expect(chunk.message.content).toBe('');
    expect(chunk.message.additional_kwargs.reasoning).toBe(text);
    const model = new ReasoningOnlyModel(3, undefined, chunk.message);
    const {node, agentContext, emitted} = harness({model, provider: Providers.GOOGLE});
    const messages = history(6);
    const before = JSON.stringify(messages);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(model.calls).toHaveLength(1);
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA)).toHaveLength(0);
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toHaveLength(1);
  });

  it.each([
    {provider: Providers.GOOGLE, options: {thinkingConfig: {thinkingBudget: 8192}}, expected: 12288},
    {provider: Providers.VERTEXAI, options: {thinkingBudget: 8192}, expected: 12288},
    {provider: Providers.VERTEXAI, options: {thinkingConfig: {thinkingBudget: 8192}}, expected: 12288},
    {provider: Providers.GOOGLE, options: {thinkingConfig: {thinkingBudget: -1}}, expected: 4096},
    {provider: Providers.VERTEXAI, options: {thinkingConfig: {thinkingBudget: 8192, thinkingLevel: 'HIGH' as const}}, expected: 4096},
    {provider: Providers.GOOGLE, options: {thinkingConfig: {thinkingBudget: 8192}, maxOutputTokens: 1024}, expected: 12288},
  ])('reserves the serving Google fixed thinking budget without overriding explicit caps: %j', async ({provider, options, expected}) => {
    const {node, model} = harness({provider, maxContextTokens: 100_000, clientOptions: options, summarizationConfig: {}});
    const messages = history(6);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(init.initializeModel).toHaveBeenCalledWith(expect.objectContaining({clientOptions: expect.objectContaining({maxOutputTokens: expected})}));
    expect(model.calls).toHaveLength(1);
  });

  const emptyReasoningBlocks = [
    {type: 'thinking', thinking: ''}, {type: 'thinking', thinking: ' \n\t'},
    {type: 'reasoning_content', reasoningText: {text: ''}},
    {type: 'reasoning_content', reasoningText: {text: ' \n\t'}},
    {type: 'redacted_thinking', data: ''},
    {type: 'reasoning_content', redactedContent: ''},
  ];
  it.each(emptyReasoningBlocks)('does not retry an empty native reasoning block: %j', async block => {
    const model = new ReasoningOnlyModel(3, undefined, new AIMessageChunk({content: [block], response_metadata: {stopReason: 'end_turn'}}));
    const {node, emitted, agentContext} = harness({model});
    const messages = history(6);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(model.calls).toHaveLength(1);
    expect(result.messages).toBeUndefined();
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA)).toHaveLength(0);
  });

  it.each([
    {type: 'redacted_thinking', data: 'OPAQUE_REASONING'},
    {type: 'reasoning_content', redactedContent: 'OPAQUE_REASONING'},
    {type: 'reasoning_content', reasoningText: {signature: 'OPAQUE_REASONING'}},
    {type: 'reasoning', encrypted_content: 'OPAQUE_REASONING', summary: []},
  ])('retains recovery for genuine opaque reasoning: %j', async block => {
    jest.useFakeTimers();
    const model = new ReasoningOnlyModel(1, undefined, new AIMessageChunk({content: [block]}));
    const {node, emitted, agentContext} = harness({model});
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    await jest.runAllTimersAsync(); await pending;
    expect(model.calls).toHaveLength(2);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    expect(JSON.stringify(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA || event === GraphEvents.ON_SUMMARIZE_COMPLETE))).not.toContain('OPAQUE_REASONING');
  });

  it.each(['stream', 'invoke'] as const)('accounts completed %s usage even when terminal tool validation rejects the response', async mode => {
    const model = new SummaryModel();
    const response = new AIMessageChunk({content: '', tool_calls: [{id: 'truncated', name: 'search', args: {}}],
      response_metadata: {stopReason: 'max_tokens'}, usage_metadata: {input_tokens: 100, output_tokens: 50, total_tokens: 150},
    });
    let calls = 0;
    if (mode === 'invoke') {
      Object.defineProperty(model, 'stream', {value: undefined});
      jest.spyOn(model, 'invoke').mockImplementation(async () => {calls++; return response;});
    } else {
      jest.spyOn(model, '_streamResponseChunks').mockImplementation(async function* () {
        calls++; yield new ChatGenerationChunk({text: '', message: response});
      });
    }
    const {node, emitted, graph, agentContext} = harness({model, provider: Providers.BEDROCK, clientOptions: {model: 'bedrock-summary'}});
    const messages = history(6);
    const before = JSON.stringify(messages);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(calls).toBe(1);
    expect(result.messages).toBeUndefined(); expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBeUndefined();
    expect(graph.dispatchRunStep.mock.calls[0][0].usage).toEqual({prompt_tokens: 100, completion_tokens: 50, total_tokens: 150});
    const completed = emitted.find(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)?.data as t.SummarizeCompleteEvent;
    expect(completed.attemptUsage).toEqual([{provider: Providers.BEDROCK, model: 'bedrock-summary', usage: response.usage_metadata}]);
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_DELTA)).toHaveLength(0);
  });

  it('reserves the configured Google thinking allowance in the request input budget', async () => {
    const {node, model} = harness({provider: Providers.GOOGLE, maxContextTokens: 20_000,
      clientOptions: {thinkingConfig: {thinkingBudget: 8192}}, summarizationConfig: {contextBudgetMode: 'window', retainRecent: {turns: 0}},
    });
    const messages = history(100);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(model.calls).toHaveLength(1);
    expect(model.calls[0]).toHaveLength(messages.length + 2);
    expect(model.calls[0].reduce((total, message) => total + count(message), 10192)).toBeLessThanOrEqual(20_000);
    expect(init.initializeModel).toHaveBeenCalledWith(expect.objectContaining({clientOptions: expect.objectContaining({maxOutputTokens: 10192})}));
  });

  it('preserves rejected fallback usage before a later provider succeeds', async () => {
    const primary = new SummaryModel(); primary.failure = () => new Error('Primary failed');
    const truncated = new SummaryModel();
    Object.defineProperty(truncated, 'stream', {value: undefined});
    jest.spyOn(truncated, 'invoke').mockResolvedValue(new AIMessageChunk({content: '',
      tool_calls: [{id: 'truncated', name: 'search', args: {}}], response_metadata: {stopReason: 'max_tokens'},
      usage_metadata: {input_tokens: 100, output_tokens: 50, total_tokens: 150},
    }));
    const successful = new SummaryModel(); successful.summaryUsage = {input_tokens: 321, output_tokens: 45, total_tokens: 366};
    const {node, emitted, graph} = harness({model: primary, clientOptions: {fallbacks: [
      {provider: Providers.BEDROCK, clientOptions: {model: 'truncated'}},
      {provider: Providers.ANTHROPIC, clientOptions: {model: 'successful'}},
    ]}});
    jest.spyOn(init, 'initializeModel').mockImplementation(({provider}) => {
      if (provider === Providers.BEDROCK) return truncated;
      return provider === Providers.ANTHROPIC ? successful : primary;
    });
    const messages = history(6);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    const completed = emitted.find(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)?.data as t.SummarizeCompleteEvent;
    expect(completed.summary).toMatchObject({provider: Providers.ANTHROPIC, model: 'successful'});
    expect(completed.attemptUsage?.map(entry => [entry.provider, entry.model, entry.usage.total_tokens])).toEqual([
      [Providers.BEDROCK, 'truncated', 150], [Providers.ANTHROPIC, 'successful', 366],
    ]);
    expect(graph.dispatchRunStep.mock.calls[0][0].usage).toMatchObject({prompt_tokens: 421, completion_tokens: 95, total_tokens: 516});
  });

  it.each([
    [Providers.OPENAI, 'initial'], [Providers.OPENAI, 'mid-run'],
    [Providers.ANTHROPIC, 'initial'], [Providers.ANTHROPIC, 'mid-run'],
    [Providers.OPENROUTER, 'initial'], [Providers.OPENROUTER, 'mid-run'],
    [Providers.BEDROCK, 'initial'], [Providers.BEDROCK, 'mid-run'],
  ] as const)('sends the prior checkpoint only once for %s %s self-compaction', async (provider, location) => {
    const {node, model, agentContext} = harness({provider,
      clientOptions: {promptCache: true, model: 'claude-sonnet-4-5'},
      summarizationConfig: {maxSummaryTokens: 128},
    });
    const prior = 'UNIQUE_PREVIOUS_CHECKPOINT ' + 'verified context '.repeat(150);
    if (location === 'initial') agentContext.setInitialSummary(prior, count(new HumanMessage(prior)));
    else agentContext.setSummary(prior, count(new HumanMessage(prior)), {precedesMessages: true});
    const messages = history(6);
    const cachedWindow = provider === Providers.BEDROCK ? addBedrockTailCacheControl(messages, '1h') : messages;
    const prefix = await agentContext.systemRunnable!.invoke(cachedWindow);
    const projectedPrior = JSON.stringify(prefix);
    expect(projectedPrior).toContain('UNIQUE_PREVIOUS_CHECKPOINT');
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(model.calls).toHaveLength(1);
    expect(model.calls[0].slice(0, -1)).toEqual(prepareProviderRequest({model, messages: prefix, provider}).messages);
    expect(JSON.stringify(model.calls[0]).split('UNIQUE_PREVIOUS_CHECKPOINT')).toHaveLength(2);
    expect(model.calls[0].at(-1)?.content).toContain('Hold on again, update your checkpoint');
    expect(model.calls[0].at(-1)?.content).toContain('already present in the request prefix');
  });

  it.each(['dedicated', 'fallback'] as const)('still supplies the prior checkpoint once when the %s omits the agent system prefix', async mode => {
    const primary = new SummaryModel();
    const secondary = new SummaryModel();
    if (mode === 'fallback') primary.failure = () => new Error('Primary failed');
    const {node, agentContext} = harness({model: primary, clientOptions: {fallbacks: mode === 'fallback' ? [
      {provider: Providers.ANTHROPIC, clientOptions: {model: 'secondary'}},
    ] : []}, summarizationConfig: {provider: mode === 'dedicated' ? Providers.ANTHROPIC : undefined, maxSummaryTokens: 128}});
    agentContext.setInitialSummary('UNIQUE_PREVIOUS_CHECKPOINT', 20);
    jest.spyOn(init, 'initializeModel').mockImplementation(({provider}) => provider === Providers.ANTHROPIC ? secondary : primary);
    const messages = history(6);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(secondary.calls).toHaveLength(1);
    expect(secondary.calls[0].some(message => message.getType() === 'system')).toBe(false);
    expect(JSON.stringify(secondary.calls[0]).split('UNIQUE_PREVIOUS_CHECKPOINT')).toHaveLength(2);
    expect(secondary.calls[0].at(-1)?.content).toContain('<previous-summary>\nUNIQUE_PREVIOUS_CHECKPOINT');
  });

  it.each(['one-user-turn', 'retained-recent-turns'] as const)('skips overflow compaction with no removable history: %s', async scenario => {
    const {node, model, agentContext, emitted, graph} = harness();
    const prior = 'EXISTING_CHECKPOINT';
    agentContext.setInitialSummary(prior, 20);
    const version = agentContext.summaryVersion;
    const messages = scenario === 'one-user-turn' ? [new HumanMessage('large payload '.repeat(100))] : history(2);
    const before = JSON.stringify(messages);
    const result = await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'overflow', allowSummarization: true, windowMessages: messages}}, {});
    expect(model.calls).toHaveLength(0);
    expect(graph.dispatchRunStep).not.toHaveBeenCalled();
    expect(result.messages).toBeUndefined();
    expect(JSON.stringify(messages)).toBe(before);
    expect(agentContext.getSummaryText()).toBe(prior);
    expect(agentContext.summaryVersion).toBe(version);
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toHaveLength(0);
  });

  it.each(['dedicated', 'fallback'] as const)('measures the actual bound inventory with the %s serving tokenizer', async mode => {
    const countOpenAI = await createTokenCounter('o200k_base');
    const countClaude = await createTokenCounter('claude');
    const tools: t.GraphTools = Array.from({length: 6}, (_, i) => ({type: 'function', function: {
      name: `inspect_${i}`, description: '사용자가 인증 미들웨어를 리팩터링하여 토큰 검증을 요청했습니다. '.repeat(12),
      parameters: {type: 'object', properties: {query: {type: 'string', description: '조회할 사용자 인증 정보와 필요한 검증 결과를 자세히 기록합니다.'.repeat(8)}}},
    }}));
    const toolTokens = countSummarizationToolTokens({tools, provider: Providers.ANTHROPIC, clientOptions: {model: 'production'}, count: countClaude});
    expect(toolTokens).toBeGreaterThan(countSummarizationToolTokens({tools, provider: Providers.OPENAI, clientOptions: {model: 'gpt-4o'}, count: countOpenAI}));
    const maxContextTokens = toolTokens + 1_000;
    const primary = new SummaryModel();
    const secondary = new SummaryModel();
    if (mode === 'fallback') primary.failure = () => new Error('Primary failed');
    secondary.failure = sent => sent.reduce((total, message) => total + countClaude(message), toolTokens + 128) > maxContextTokens
      ? new Error('maximum context length exceeded') : undefined;
    const {node, agentContext} = harness({model: primary, tools, tokenCounter: countOpenAI, maxContextTokens,
      clientOptions: {model: 'gpt-4o', fallbacks: mode === 'fallback' ? [{provider: Providers.ANTHROPIC,
        maxContextTokens, clientOptions: {model: 'production', maxTokens: 128},
      }] : []},
      summarizationConfig: {provider: mode === 'dedicated' ? Providers.ANTHROPIC : undefined,
        prompt: 'Checkpoint.', maxSummaryTokens: 128, contextBudgetMode: 'window', retainRecent: {turns: 0}},
    });
    await agentContext.tokenCalculationPromise;
    jest.spyOn(init, 'initializeModel').mockImplementation(({provider}) => provider === Providers.ANTHROPIC ? secondary : primary);
    const messages = history(3);
    await node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, windowMessages: messages}}, {});
    expect(secondary.calls).toHaveLength(1);
    expect(secondary.calls[0].length).toBe(messages.length + 1);
    expect(secondary.calls[0].reduce((total, message) => total + countClaude(message), toolTokens + 128)).toBeLessThanOrEqual(maxContextTokens);
    expect(init.initializeModel).toHaveBeenCalledWith(expect.objectContaining({provider: Providers.ANTHROPIC, tools}));
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
  });

  it('retries an Anthropic tool-only summary without executing the tool or changing request options', async () => {
    jest.useFakeTimers();
    const model = new SummaryModel();
    Object.defineProperty(model, 'stream', {value: undefined});
    const invoke = jest.spyOn(model, 'invoke').mockResolvedValueOnce(new AIMessageChunk({content: '',
      tool_calls: [{id: 'calculator', name: 'calculator', args: {expression: '2+2'}}],
      response_metadata: {stop_reason: 'tool_use'},
    })).mockResolvedValue(new AIMessageChunk('CLEAN_SUMMARY'));
    const {node, emitted, agentContext} = harness({provider: Providers.ANTHROPIC, model});
    const messages = history(6);
    const pending = node({messages, summarizationRequest: {agentId: 'agent', remainingContextTokens: 0, reason: 'manual', windowMessages: messages}}, {});
    await jest.runAllTimersAsync(); await pending;
    expect(invoke).toHaveBeenCalledTimes(2);
    expect(invoke.mock.calls[1][0]).toEqual(invoke.mock.calls[0][0]);
    expect(agentContext.getSummaryText()).toBe('CLEAN_SUMMARY');
    expect(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE)).toHaveLength(1);
    expect(JSON.stringify(emitted.filter(({event}) => event === GraphEvents.ON_SUMMARIZE_COMPLETE))).not.toContain('calculator');
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

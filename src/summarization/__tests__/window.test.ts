import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createSummarizeNode } from '@/summarization/node';
import { resolveStreamLimits } from '@/llm/streamLimits';
import { AgentContext } from '@/agents/AgentContext';
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
      message: new AIMessageChunk({ content: 'SUMMARY' }),
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
  summarizationConfig = { maxSummaryTokens: 128 },
}: {
  maxContextTokens?: number;
  model?: SummaryModel;
  provider?: Providers;
  clientOptions?: t.ClientOptions;
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
    count
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
    dispatchRunStep: jest.fn(async (): Promise<void> => {}),
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
          handle: (): void => { ordinaryDeltaCount++; },
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

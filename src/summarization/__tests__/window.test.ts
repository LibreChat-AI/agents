import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessage, AIMessageChunk, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createSummarizeNode } from '@/summarization/node';
import { resolveStreamLimits } from '@/llm/streamLimits';
import { AgentContext } from '@/agents/AgentContext';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import * as events from '@/utils/events';
import * as init from '@/llm/init';

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
} = {}) {
  const agentContext = AgentContext.fromConfig(
    {
      agentId: 'agent',
      provider: Providers.OPENAI,
      instructions: 'The turn system prompt',
      maxContextTokens,
      summarizationEnabled: true,
      summarizationConfig: { maxSummaryTokens: 128 },
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

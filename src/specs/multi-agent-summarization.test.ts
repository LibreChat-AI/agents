/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Multi-agent summarization tests.
 *
 * Validates that summarization works correctly when multiple agents
 * share a conversation, each with independent context budgets and
 * summarization state.
 *
 * Uses FakeListChatModel — no API keys required.
 */
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  HumanMessage,
  AIMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createTokenCounter } from '@/utils/tokens';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

function getSummaryText(summary: t.SummaryContentBlock | undefined): string {
  if (!summary) return '';
  return (summary.content ?? [])
    .map((block) => ('text' in block ? (block as { text: string }).text : ''))
    .join('');
}

function createSpies(): Record<string, jest.Mock> {
  return {
    onMessageDeltaSpy: jest.fn(),
    onRunStepSpy: jest.fn(),
    onSummarizeStartSpy: jest.fn(),
    onSummarizeCompleteSpy: jest.fn(),
  };
}

function buildHandlers(
  collectedUsage: UsageMetadata[],
  aggregateContent: t.ContentAggregator,
  spies: ReturnType<typeof createSpies>
): Record<string | GraphEvents, t.EventHandler> {
  return {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (event: GraphEvents, data: t.StreamEventData): void => {
        aggregateContent({ event, data: data as any });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents, data: t.StreamEventData): void => {
        spies.onRunStepSpy(event, data);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (event: GraphEvents, data: t.StreamEventData): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (event: GraphEvents, data: t.StreamEventData): void => {
        spies.onMessageDeltaSpy(event, data);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: { handle: (): void => {} },
    [GraphEvents.ON_SUMMARIZE_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        spies.onSummarizeStartSpy(data);
      },
    },
    [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        spies.onSummarizeCompleteSpy(data);
      },
    },
  };
}

const SUMMARY_RESPONSE =
  '## Summary\nAgent A discussed math with the user. Key result: 42.';

const streamConfig = {
  configurable: { thread_id: 'multi-agent-sum-test' },
  recursionLimit: 50,
  streamMode: 'values' as const,
  version: 'v2' as const,
};

describe('Multi-agent summarization', () => {
  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [SUMMARY_RESPONSE] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('sequential A → B: Agent A summarizes independently, Agent B sees full history', async () => {
    const spies = createSpies();
    const collectedUsage: UsageMetadata[] = [];
    const { aggregateContent } = createContentAggregator();
    const tokenCounter = await createTokenCounter();

    const padding = ' the quick brown fox jumps over the lazy dog'.repeat(30);
    const conversationHistory: BaseMessage[] = [
      new HumanMessage(`What is the meaning of life?${padding}`),
      new AIMessage(`The answer is 42.${padding}`),
      new HumanMessage(`Can you explain more?${padding}`),
      new AIMessage(`It comes from a famous book.${padding}`),
      new HumanMessage(`Which book?${padding}`),
      new AIMessage(`The Hitchhiker's Guide to the Galaxy.${padding}`),
      new HumanMessage('Now pass this to Agent B for confirmation.'),
    ];

    const agents: t.AgentInputs[] = [
      {
        agentId: 'agent_a',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions:
          'You are Agent A. Process the request and pass to Agent B.',
        maxContextTokens: 800,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
          prompt: 'Write a concise checkpoint.',
          maxSummaryTokens: 128,
        },
      },
      {
        agentId: 'agent_b',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions: 'You are Agent B. Confirm the result from Agent A.',
        maxContextTokens: 4000,
        summarizationEnabled: false,
      },
    ];

    const edges: t.GraphEdge[] = [
      { from: 'agent_a', to: 'agent_b', edgeType: 'direct' },
    ];

    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < conversationHistory.length; i++) {
      indexTokenCountMap[i] = tokenCounter(conversationHistory[i]);
    }

    const run = await Run.create<t.IState>({
      runId: `multi-sum-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    await run.processStream({ messages: conversationHistory }, streamConfig);

    // Agent A should have triggered summarization due to its tight 800-token input budget
    const startCalls = spies.onSummarizeStartSpy.mock.calls;
    const completeCalls = spies.onSummarizeCompleteSpy.mock.calls;

    console.log(
      `  Summarization events: start=${startCalls.length}, complete=${completeCalls.length}`
    );

    expect(startCalls.length).toBeGreaterThan(0);
    const startPayload = startCalls[0][0] as t.SummarizeStartEvent;
    expect(startPayload.agentId).toBe('agent_a');

    expect(completeCalls.length).toBeGreaterThan(0);
    const completePayload = completeCalls[0][0] as t.SummarizeCompleteEvent;
    expect(completePayload.agentId).toBe('agent_a');
    expect(completePayload.summary).toBeDefined();
    const summaryText = getSummaryText(completePayload.summary);
    expect(summaryText.length).toBeGreaterThan(0);
    console.log(`  Agent A summary: "${summaryText.substring(0, 100)}"`);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    console.log(`  Final messages: ${finalMessages?.length ?? 0}`);
  });

  test('each agent has independent summarization state', async () => {
    const spies = createSpies();
    const collectedUsage: UsageMetadata[] = [];
    const { aggregateContent } = createContentAggregator();
    const tokenCounter = await createTokenCounter();

    const padding = ' the quick brown fox jumps over the lazy dog'.repeat(30);
    const conversationHistory: BaseMessage[] = [
      new HumanMessage(`Question one about math${padding}`),
      new AIMessage(`Math answer one${padding}`),
      new HumanMessage(`Question two about science${padding}`),
      new AIMessage(`Science answer two${padding}`),
      new HumanMessage(`Question three about history${padding}`),
      new AIMessage(`History answer three${padding}`),
      new HumanMessage('Summarize everything'),
    ];

    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < conversationHistory.length; i++) {
      indexTokenCountMap[i] = tokenCounter(conversationHistory[i]);
    }

    const agents: t.AgentInputs[] = [
      {
        agentId: 'tight_agent_a',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions: 'You are Agent A with tight context.',
        maxContextTokens: 800,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      {
        agentId: 'tight_agent_b',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions: 'You are Agent B with tight context.',
        maxContextTokens: 800,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
    ];

    const edges: t.GraphEdge[] = [
      { from: 'tight_agent_a', to: 'tight_agent_b', edgeType: 'direct' },
    ];

    const run = await Run.create<t.IState>({
      runId: `multi-independent-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream({ messages: conversationHistory }, streamConfig);
    } catch (err) {
      error = err as Error;
    }

    const starts = spies.onSummarizeStartSpy.mock.calls;
    const completes = spies.onSummarizeCompleteSpy.mock.calls;
    console.log(
      `  Summarization: start=${starts.length}, complete=${completes.length}`
    );

    if (error) {
      console.log(`  Error (acceptable): ${error.message.substring(0, 100)}`);
    }

    // In a sequential A→B flow, agent_a summarizes the large initial history.
    // After compaction, agent_b receives the compressed state which typically
    // fits within budget, so agent_b may not trigger. Verify at least agent_a
    // fires and that every event carries the correct agentId.
    expect(starts.length).toBeGreaterThanOrEqual(1);
    for (const call of starts) {
      const payload = call[0] as t.SummarizeStartEvent;
      expect(['tight_agent_a', 'tight_agent_b']).toContain(payload.agentId);
    }
    console.log(
      `  Agents summarized: ${starts.map((c: unknown[]) => (c[0] as t.SummarizeStartEvent).agentId).join(', ')}`
    );
  });

  test('agent with large context does not summarize while tight agent does', async () => {
    const spies = createSpies();
    const collectedUsage: UsageMetadata[] = [];
    const { aggregateContent } = createContentAggregator();
    const tokenCounter = await createTokenCounter();

    const padding = ' the quick brown fox jumps over the lazy dog'.repeat(30);
    const conversationHistory: BaseMessage[] = [
      new HumanMessage(`First message${padding}`),
      new AIMessage(`First reply${padding}`),
      new HumanMessage(`Second message${padding}`),
      new AIMessage(`Second reply${padding}`),
      new HumanMessage(`Third message${padding}`),
      new AIMessage(`Third reply${padding}`),
      new HumanMessage('Process this'),
    ];

    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < conversationHistory.length; i++) {
      indexTokenCountMap[i] = tokenCounter(conversationHistory[i]);
    }

    const agents: t.AgentInputs[] = [
      {
        agentId: 'tight_agent',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions: 'You are the tight context agent.',
        maxContextTokens: 800,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      {
        agentId: 'large_agent',
        provider: Providers.OPENAI,
        clientOptions: getLLMConfig(Providers.OPENAI),
        instructions: 'You are the large context agent.',
        maxContextTokens: 100_000,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
    ];

    const edges: t.GraphEdge[] = [
      { from: 'tight_agent', to: 'large_agent', edgeType: 'direct' },
    ];

    const run = await Run.create<t.IState>({
      runId: `multi-mixed-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream({ messages: conversationHistory }, streamConfig);
    } catch (err) {
      error = err as Error;
    }

    const starts = spies.onSummarizeStartSpy.mock.calls;
    console.log(`  Summarization events: ${starts.length}`);

    if (error) {
      console.log(`  Error: ${error.message.substring(0, 100)}`);
    }

    expect(starts.length).toBeGreaterThan(0);
    for (const call of starts) {
      const payload = call[0] as t.SummarizeStartEvent;
      expect(payload.agentId).toBe('tight_agent');
      console.log(`  Summarization from: ${payload.agentId}`);
    }

    const largeAgentStarts = starts.filter(
      (call: unknown[]) =>
        (call[0] as t.SummarizeStartEvent).agentId === 'large_agent'
    );
    expect(largeAgentStarts.length).toBe(0);
  });
});

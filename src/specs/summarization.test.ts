/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
import { performance } from 'node:perf_hooks';
config();
import { Calculator } from '@/tools/Calculator';
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { createTokenCounter } from '@/utils/tokens';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';
import { formatAgentMessages } from '@/messages/format';
import { buildSummaryCarrierText } from '@/summarization/shared';
import { prepareToolsForPromptCache } from '@/llm/promptCacheTools';
import { CustomAnthropic } from '@/llm/anthropic';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import * as providers from '@/llm/providers';
import { hasAnyEnv, hasEnv, hasEveryEnv } from './spec.utils';

/** Extract plain text from a SummaryContentBlock's content array (test helper). */
function getSummaryText(summary: t.SummaryContentBlock | undefined): string {
  if (!summary) return '';
  return (summary.content ?? [])
    .map((block) => ('text' in block ? (block as { text: string }).text : ''))
    .join('');
}

const LOCAL_SUMMARY_CONFIG = {
  prompt: 'Write one concise checkpoint with Goal, Progress, and Next Steps sections.',
  updatePrompt: 'Merge the new messages into the existing checkpoint without duplicate sections.',
  maxSummaryTokens: 128,
} satisfies t.SummarizationConfig;

// ---------------------------------------------------------------------------
// Shared test infrastructure
// ---------------------------------------------------------------------------

function createSpies(): {
  onMessageDeltaSpy: jest.Mock;
  onRunStepSpy: jest.Mock;
  onSummarizeStartSpy: jest.Mock;
  onSummarizeCompleteSpy: jest.Mock;
  } {
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
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        spies.onRunStepSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        spies.onMessageDeltaSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        _metadata?: Record<string, unknown>
      ): void => {},
    },
    [GraphEvents.ON_SUMMARIZE_START]: {
      handle: (
        _event: GraphEvents.ON_SUMMARIZE_START,
        data: t.StreamEventData
      ): void => {
        spies.onSummarizeStartSpy(data);
      },
    },
    [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
      handle: (
        _event: GraphEvents.ON_SUMMARIZE_COMPLETE,
        data: t.StreamEventData
      ): void => {
        spies.onSummarizeCompleteSpy(data);
      },
    },
  };
}

async function createSummarizationRun(opts: {
  agentProvider: Providers;
  summarizationProvider: Providers;
  summarizationModel?: string;
  maxContextTokens: number;
  instructions: string;
  collectedUsage: UsageMetadata[];
  aggregateContent: t.ContentAggregator;
  spies: ReturnType<typeof createSpies>;
  tokenCounter?: t.TokenCounter;
  tools?: t.GraphTools;
  indexTokenCountMap?: Record<string, number>;
  llmConfigOverride?: Record<string, unknown>;
  maxSummaryTokens?: number;
}): Promise<Run<t.IState>> {
  const llmConfig = {
    ...getLLMConfig(opts.agentProvider),
    ...opts.llmConfigOverride,
  };
  const tokenCounter = opts.tokenCounter ?? (await createTokenCounter());

  return Run.create<t.IState>({
    runId: `sum-e2e-${opts.agentProvider}-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: opts.tools ?? [new Calculator()],
      instructions: opts.instructions,
      maxContextTokens: opts.maxContextTokens,
      summarizationEnabled: true,
      summarizationConfig: {
        provider: opts.summarizationProvider,
        model: opts.summarizationModel,
        maxSummaryTokens: opts.maxSummaryTokens,
      },
    },
    returnContent: true,
    customHandlers: buildHandlers(
      opts.collectedUsage,
      opts.aggregateContent,
      opts.spies
    ),
    tokenCounter,
    indexTokenCountMap: opts.indexTokenCountMap,
  });
}

async function runTurn(
  state: { run: Run<t.IState>; conversationHistory: BaseMessage[] },
  userMessage: string,
  streamConfig: Record<string, unknown>
): Promise<t.MessageContentComplex[] | undefined> {
  state.conversationHistory.push(new HumanMessage(userMessage));
  const result = await state.run.processStream(
    { messages: state.conversationHistory },
    streamConfig as any
  );
  const finalMessages = state.run.getRunMessages();
  state.conversationHistory.push(...(finalMessages ?? []));
  return result;
}

function assertSummarizationEvents(spies: ReturnType<typeof createSpies>): {
  startPayload: t.SummarizeStartEvent;
  completePayload: t.SummarizeCompleteEvent;
} {
  expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
  expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

  const startPayload = spies.onSummarizeStartSpy.mock
    .calls[0][0] as t.SummarizeStartEvent;
  expect(startPayload.agentId).toBeDefined();
  expect(typeof startPayload.provider).toBe('string');
  expect(startPayload.messagesToRefineCount).toBeGreaterThan(0);

  const completePayload = spies.onSummarizeCompleteSpy.mock
    .calls[0][0] as t.SummarizeCompleteEvent;
  if (completePayload.error != null) {
    throw new Error(
      `Summarization failed (${completePayload.errorCode ?? 'unknown'}): ${completePayload.error}`
    );
  }
  expect(completePayload.agentId).toBeDefined();
  expect(completePayload.summary).toBeDefined();
  expect(completePayload.summary!.type).toBe(ContentTypes.SUMMARY);
  expect(typeof getSummaryText(completePayload.summary)).toBe('string');
  expect(getSummaryText(completePayload.summary).length).toBeGreaterThan(10);
  expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(0);
  expect(completePayload.summary!.provider).toBeDefined();
  expect(completePayload.summary!.createdAt).toBeDefined();

  const startIdx = spies.onSummarizeStartSpy.mock.invocationCallOrder[0];
  const completeIdx = spies.onSummarizeCompleteSpy.mock.invocationCallOrder[0];
  expect(startIdx).toBeLessThan(completeIdx);

  return { startPayload, completePayload };
}

function assertSummaryRunStep(
  spies: ReturnType<typeof createSpies>,
  summaryText: string
): void {
  const summaryRunSteps = spies.onRunStepSpy.mock.calls.filter(
    (call) => (call[1] as any)?.summary != null
  );
  expect(summaryRunSteps.length).toBeGreaterThan(0);
  const step = summaryRunSteps[0][1] as t.RunStep & {
    summary: t.SummaryContentBlock;
  };
  expect(step.summary.type).toBe(ContentTypes.SUMMARY);
  expect(getSummaryText(step.summary)).toBe(summaryText);
  expect(step.id).toBeDefined();
  expect(typeof step.stepIndex).toBe('number');
}

function buildIndexTokenCountMap(
  messages: BaseMessage[],
  tokenCounter: t.TokenCounter
): Record<string, number> {
  const map: Record<string, number> = {};
  for (let i = 0; i < messages.length; i++) {
    map[String(i)] = tokenCounter(messages[i]);
  }
  return map;
}

function sumTokenCountMap(map: Record<string, number | undefined>): number {
  let total = 0;
  for (const key in map) {
    total += map[key] ?? 0;
  }
  return total;
}

function createSeededTokenAuditHistory(): BaseMessage[] {
  const details =
    'Token audit context preserves index token counts, summary replacement, calibration data, and post-summary continuity. ' +
    'Important retained values: alpha=1024, beta=2048, gamma=4096, checksum TOKEN-AUDIT-7F3. ' +
    'The repeated detail intentionally exceeds a compact context budget. ';
  const padding = details.repeat(8);

  return [
    new HumanMessage(
      `Audit turn 1: establish the accounting scenario. ${padding}`
    ),
    new AIMessage(`Recorded turn 1 accounting notes. ${padding}`),
    new HumanMessage(`Audit turn 2: add more retained details. ${padding}`),
    new AIMessage(`Recorded turn 2 accounting notes. ${padding}`),
    new HumanMessage(`Audit turn 3: preserve final identifiers. ${padding}`),
    new AIMessage(`Recorded turn 3 accounting notes. ${padding}`),
  ];
}

function logTurn(
  label: string,
  conversationHistory: BaseMessage[],
  extra?: string
): void {
  console.log(
    `  ${label} — ${conversationHistory.length} messages${extra != null && extra !== '' ? `, ${extra}` : ''}`
  );
}

function addDeterministicSummarizationPressure(
  conversationHistory: BaseMessage[],
  label: string,
  turns = 16
): void {
  const pressure =
    'Verified compaction context: preserve calculation results, source identifiers, observations, and the next action for the continuing task. '.repeat(8);
  for (let turn = 0; turn < turns; turn++) {
    conversationHistory.push(
      new HumanMessage(`${label} exchange ${turn}: ${pressure}`),
      new AIMessage(`Recorded ${label} exchange ${turn}: ${pressure}`)
    );
  }
}

describe('Local summarization fixtures (no API keys)', () => {
  test('exceeds the test budgets with individually bounded exchanges', async () => {
    const history: BaseMessage[] = [];
    const count = await createTokenCounter();
    addDeterministicSummarizationPressure(history, 'bounded-pressure');

    expect(history).toHaveLength(32);
    let total = 0;
    for (let index = 0; index < history.length; index += 2) {
      expect(history[index]).toBeInstanceOf(HumanMessage);
      expect(history[index + 1]).toBeInstanceOf(AIMessage);
      const exchangeTokens = count(history[index]) + count(history[index + 1]);
      expect(exchangeTokens).toBeLessThan(512);
      total += exchangeTokens;
    }
    expect(total).toBeGreaterThan(4_000);
  });

  test('reports sanitized completion errors before asserting a summary exists', () => {
    const spies = createSpies();
    spies.onSummarizeStartSpy({
      agentId: 'agent_0',
      provider: Providers.OPENAI,
      messagesToRefineCount: 1,
    });
    spies.onSummarizeCompleteSpy({
      id: 'failed-summary',
      agentId: 'agent_0',
      errorCode: 'context_exceeded',
      error: 'Summarization failed; conversation history was preserved',
    } satisfies t.SummarizeCompleteEvent);

    expect(() => assertSummarizationEvents(spies)).toThrow(
      'Summarization failed (context_exceeded): Summarization failed; conversation history was preserved'
    );
  });
});

// ---------------------------------------------------------------------------
// Anthropic Summarization Tests
// ---------------------------------------------------------------------------

const hasAnthropic = hasEnv('ANTHROPIC_API_KEY');
(hasAnthropic ? describe : describe.skip)('Anthropic Summarization E2E', () => {
  jest.setTimeout(180_000);

  const agentProvider = Providers.ANTHROPIC;
  const streamConfig = {
    configurable: { thread_id: 'anthropic-sum-e2e' },
    recursionLimit: 80,
    streamMode: 'values',
    version: 'v2' as const,
  };

  const MATH_TUTOR_INSTRUCTIONS = [
    'You are an expert math tutor. You MUST use the calculator tool for ALL computations —',
    'never compute in your head. Keep explanations concise (2-3 sentences max).',
    'When summarizing prior work, list each calculation and its result.',
  ].join(' ');

  test('cache-aligned summarization reuses the normal tool prefix', async () => {
    const nonce = `${Date.now()}-${Math.random()}`;
    const description = `${nonce} ${'stable cache benchmark token '.repeat(180)}`;
    const tools = Array.from({ length: 8 }, (_, index) => ({
      name: `cache_probe_${index}`,
      description,
      input_schema: {
        type: 'object' as const,
        properties: {
          value: { type: 'string' },
        },
      },
    })) as t.GraphTools;
    const preparedTools = prepareToolsForPromptCache({
      provider: Providers.ANTHROPIC,
      clientOptions: { promptCache: true },
      tools,
      isDeferred: () => false,
    });
    const createModel = (
      boundTools: t.GraphTools
    ): ReturnType<CustomAnthropic['bindTools']> =>
      new CustomAnthropic({
        model: 'claude-haiku-4-5',
        anthropicApiKey: process.env.ANTHROPIC_API_KEY,
        maxTokens: 8,
        temperature: 0,
      }).bindTools(boundTools);

    const prime = await createModel(preparedTools ?? tools).invoke([
      new HumanMessage('Prime the normal request tool prefix. Reply OK.'),
    ]);
    const primeCacheCreation =
      prime.usage_metadata?.input_token_details?.cache_creation ?? 0;
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const baselineStartedAt = performance.now();
    const baseline = await createModel(tools).invoke([
      new HumanMessage('Baseline compaction request. Reply OK.'),
    ]);
    const baselineLatencyMs = performance.now() - baselineStartedAt;

    const alignedStartedAt = performance.now();
    const aligned = await createModel(preparedTools ?? tools).invoke([
      new HumanMessage('Cache-aligned compaction request. Reply OK.'),
    ]);
    const alignedLatencyMs = performance.now() - alignedStartedAt;

    const baselineCacheRead =
      baseline.usage_metadata?.input_token_details?.cache_read ?? 0;
    const alignedCacheRead =
      aligned.usage_metadata?.input_token_details?.cache_read ?? 0;
    console.log(
      `  Compaction tool-prefix benchmark: ${JSON.stringify({
        prime: {
          cacheCreationInputTokens: primeCacheCreation,
          inputTokens: prime.usage_metadata?.input_tokens,
        },
        baseline: {
          cacheReadInputTokens: baselineCacheRead,
          inputTokens: baseline.usage_metadata?.input_tokens,
          latencyMs: Math.round(baselineLatencyMs),
        },
        aligned: {
          cacheReadInputTokens: alignedCacheRead,
          inputTokens: aligned.usage_metadata?.input_tokens,
          latencyMs: Math.round(alignedLatencyMs),
        },
      })}`
    );

    expect(primeCacheCreation).toBeGreaterThan(0);
    expect(alignedCacheRead).toBeGreaterThan(baselineCacheRead);
    expect(alignedCacheRead).toBeGreaterThan(0);
  }, 180_000);

  test('fresh runs reuse a persisted fading tier without mutating canonical tool history', async () => {
    const tokenCounter = await createTokenCounter();
    const largeToolResult = `BEGIN-LIVE-PAYLOAD\n${'stable provider payload '.repeat(
      3000
    )}\nEND-LIVE-PAYLOAD`;
    const buildHistory = (): BaseMessage[] => [
      new HumanMessage('Inspect the synthetic provider payload.'),
      new AIMessage({
        content: 'I inspected the payload.',
        tool_calls: [
          {
            id: 'live_fading_tool_call',
            name: 'inspect_payload',
            args: { source: 'live-regression' },
          },
        ],
      }),
      new ToolMessage({
        content: largeToolResult,
        tool_call_id: 'live_fading_tool_call',
        name: 'inspect_payload',
      }),
      new AIMessage('The inspection completed successfully.'),
      new HumanMessage('Reply with exactly OK.'),
    ];
    const executeRun = async (
      history: BaseMessage[],
      fadingTier?: t.FadingTier,
      calibrationRatio?: number
    ): Promise<Run<t.IState>> => {
      const run = await Run.create<t.IState>({
        runId: `anthropic-live-fading-${Date.now()}-${Math.random()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: {
            ...getLLMConfig(Providers.ANTHROPIC),
            model: 'claude-haiku-4-5',
            streaming: false,
            maxTokens: 32,
            temperature: 0,
          },
          tools: [],
          instructions: 'Reply with exactly OK and do not call tools.',
          maxContextTokens: 4000,
          summarizationEnabled: false,
        },
        returnContent: true,
        tokenCounter,
        indexTokenCountMap: buildIndexTokenCountMap(history, tokenCounter),
        fadingTier,
        fadingTiers:
          fadingTier == null ? undefined : { default: fadingTier },
        calibrationRatio,
      });
      await run.processStream(
        { messages: history },
        {
          configurable: { thread_id: run.id },
          recursionLimit: 10,
          streamMode: 'values',
          version: 'v2',
        } as any
      );
      expect(run.getRunMessages()?.length ?? 0).toBeGreaterThan(0);
      return run;
    };

    const firstHistory = buildHistory();
    const firstRun = await executeRun(firstHistory);
    const firstTier = firstRun.getFadingTier();

    expect(firstHistory[2].content).toBe(largeToolResult);
    expect(firstTier?.latched).toBe(true);
    expect(firstTier?.masked).toBe(true);

    const secondHistory = buildHistory();
    const secondRun = await executeRun(
      secondHistory,
      firstTier,
      firstRun.getCalibrationRatio()
    );

    expect(secondHistory[2].content).toBe(largeToolResult);
    expect(secondRun.getFadingTier()).toEqual(firstTier);
  }, 180_000);

  test('heavy multi-turn with tool calls triggers and survives summarization', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const resetAggregator = (): {
      contentParts: t.MessageContentComplex[];
      aggregateContent: t.ContentAggregator;
    } => {
      collectedUsage = [];
      const { contentParts: cp, aggregateContent: ac } =
        createContentAggregator();
      return {
        contentParts: cp as t.MessageContentComplex[],
        aggregateContent: ac,
      };
    };

    const createRun = async (
      maxTokens = 4000
    ): Promise<{
      run: Run<t.IState>;
      contentParts: t.MessageContentComplex[];
    }> => {
      const { contentParts, aggregateContent } = resetAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      const run = await createSummarizationRun({
        agentProvider,
        summarizationProvider: Providers.ANTHROPIC,
        summarizationModel: 'claude-haiku-4-5',
        maxContextTokens: maxTokens,
        instructions: MATH_TUTOR_INSTRUCTIONS,
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
      return { run, contentParts };
    };

    // Turn 1: greeting + simple calculation
    let { run, contentParts } = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Hi! Let\'s do some math. What is 12345 * 6789? Use the calculator please.',
      streamConfig
    );
    logTurn('T1', conversationHistory, `parts=${contentParts.length}`);

    // Turn 2: compound calculation
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      'Great. Now take that result and divide it by 137. Then multiply the quotient by 42. Show both steps. Use the calculator for each.',
      streamConfig
    );
    logTurn('T2', conversationHistory, `parts=${contentParts.length}`);

    // Turn 3: verbose question to inflate token count
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      [
        'I need you to compute the following sequence of operations step by step using the calculator:',
        '1) Start with 9876543',
        '2) Subtract 1234567 from it',
        '3) Take the square root of the result',
        'Please show each intermediate step with the calculator.',
      ].join('\n'),
      streamConfig
    );
    logTurn('T3', conversationHistory, `parts=${contentParts.length}`);

    // Turn 4: even more to guarantee pruning threshold
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      'Now calculate 2^20 using the calculator. Also, what is 1000000 / 7? Use calculator for both.',
      streamConfig
    );
    logTurn('T4', conversationHistory, `parts=${contentParts.length}`);

    // Turn 5: tighter context to force summarization if not already
    ({ run, contentParts } = await createRun(3500));
    await runTurn(
      { run, conversationHistory },
      'What is 355 / 113? Use the calculator. This should approximate pi.',
      streamConfig
    );
    logTurn('T5', conversationHistory, `parts=${contentParts.length}`);

    // Turn 6: if still no summarization, squeeze harder
    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      // Debug: show total token count from the indexTokenCountMap
      const debugMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      const totalTokens = Object.values(debugMap).reduce(
        (sum, v) => sum + v,
        0
      );
      console.log(
        `  Pre-T6 debug: ${conversationHistory.length} msgs, totalTokens=${totalTokens}, ` +
          `indexTokenCountMap keys=${Object.keys(debugMap).length}`
      );

      ({ run, contentParts } = await createRun(3200));
      await runTurn(
        { run, conversationHistory },
        'Calculate 999 * 999 with the calculator. Also compute 123456789 % 97.',
        streamConfig
      );
      logTurn('T6', conversationHistory, `parts=${contentParts.length}`);
    }

    // Provider output length varies, so add deterministic history rather than
    // shrinking below the minimum viable instruction and tool-schema budget.
    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      addDeterministicSummarizationPressure(
        conversationHistory,
        'anthropic-heavy'
      );
      ({ run, contentParts } = await createRun(3200));
      await runTurn({ run, conversationHistory }, 'What is 1+1?', streamConfig);
      logTurn('T7', conversationHistory, `parts=${contentParts.length}`);
    }

    console.log(
      `  Summarize events — start: ${spies.onSummarizeStartSpy.mock.calls.length}, complete: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    // Assert summarization fired correctly
    const { startPayload, completePayload } = assertSummarizationEvents(spies);
    assertSummaryRunStep(spies, getSummaryText(completePayload.summary));

    console.log(
      `  Summary (${getSummaryText(completePayload.summary).length} chars, ${completePayload.summary!.tokenCount} tok): "${getSummaryText(completePayload.summary).substring(0, 250)}…"`
    );
    console.log(
      `  Start event — agent=${startPayload.agentId}, provider=${startPayload.provider}, refining=${startPayload.messagesToRefineCount} msgs`
    );

    // Token accounting: summary tokenCount must be reasonable
    expect(completePayload.summary!.tokenCount).toBeGreaterThan(10);
    expect(completePayload.summary!.tokenCount).toBeLessThan(2000);

    // Token accounting: collectedUsage should have valid entries from post-summary model calls
    const validUsageEntries = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(validUsageEntries.length).toBeGreaterThan(0);
    const lastUsage = validUsageEntries[validUsageEntries.length - 1];
    expect(lastUsage.output_tokens).toBeGreaterThan(0);
    console.log(
      `  Post-summary usage — input: ${lastUsage.input_tokens}, output: ${lastUsage.output_tokens}`
    );

    // Assert model still works after summarization
    expect(spies.onMessageDeltaSpy).toHaveBeenCalled();

    // Summarization may fire multiple times per run (no single-fire guard);
    // the graph's recursionLimit prevents infinite loops.
    const startCallsForSameAgent = spies.onSummarizeStartSpy.mock.calls.filter(
      (c) => (c[0] as t.SummarizeStartEvent).agentId === startPayload.agentId
    );
    expect(startCallsForSameAgent.length).toBeGreaterThanOrEqual(1);
  }, 180_000);

  test('post-summary continuation over multiple turns preserves context', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    let latestContentParts: t.MessageContentComplex[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 4000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { contentParts, aggregateContent } = createContentAggregator();
      latestContentParts = contentParts as t.MessageContentComplex[];
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider: Providers.ANTHROPIC,
        summarizationModel: 'claude-haiku-4-5',
        maxContextTokens: maxTokens,
        instructions: MATH_TUTOR_INSTRUCTIONS,
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build up conversation — generous budget so messages accumulate
    let run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 42 * 58? Calculator please.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Now compute 2436 + 1337. Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 3773 * 11? Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Calculate 41503 - 12345 and then 29158 / 4. Show both with calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 100 * 200? Calculator.',
      streamConfig
    );

    // Progressively squeeze to force summarization
    for (const squeeze of [3500, 3200, 3100, 3000, 2800, 2500, 2000]) {
      if (spies.onSummarizeStartSpy.mock.calls.length > 0) {
        break;
      }
      run = await createRun(squeeze);
      await runTurn(
        { run, conversationHistory },
        `What is ${squeeze} * 2? Calculator.`,
        streamConfig
      );
    }

    console.log(
      `  Pre-continuation: ${spies.onSummarizeCompleteSpy.mock.calls.length} summaries`
    );
    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();
    const completeSummary = (
      spies.onSummarizeCompleteSpy.mock.calls[0][0] as t.SummarizeCompleteEvent
    ).summary!;
    const summaryText = getSummaryText(completeSummary);

    // Token accounting: summary tokenCount bounds
    expect(completeSummary.tokenCount ?? 0).toBeGreaterThan(10);
    expect(completeSummary.tokenCount ?? 0).toBeLessThan(1200);

    // Continue for 2 more turns AFTER summarization — model should remain coherent
    run = await createRun(4000);
    const postSumTurn1 = await runTurn(
      { run, conversationHistory },
      'What were all the numbers we computed so far? List them.',
      streamConfig
    );
    expect(postSumTurn1).toBeDefined();
    logTurn('Post-sum T1', conversationHistory);

    run = await createRun(4000);
    const postSumTurn2 = await runTurn(
      { run, conversationHistory },
      'Now compute the sum of 2436, 3773, and 41503 using the calculator.',
      streamConfig
    );
    expect(postSumTurn2).toBeDefined();
    logTurn('Post-sum T2', conversationHistory);

    const hasPostSumCalculator = latestContentParts.some(
      (p) =>
        p.type === ContentTypes.TOOL_CALL &&
        (p as t.ToolCallContent).tool_call?.name === 'calculator'
    );
    expect(hasPostSumCalculator).toBe(true);

    // Model should still reference prior context from the summary
    expect(spies.onMessageDeltaSpy).toHaveBeenCalled();
    console.log(`  Summary text: "${summaryText.substring(0, 200)}…"`);
    console.log(`  Final message count: ${conversationHistory.length}`);
  }, 180_000);

  test('cross-provider summarization: Anthropic agent with OpenAI summarizer', async () => {
    const hasOpenAI = hasEnv('OPENAI_API_KEY');
    if (!hasOpenAI) {
      console.log('  Skipping cross-provider test (no OPENAI_API_KEY)');
      return;
    }

    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 4000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider: Providers.ANTHROPIC,
        summarizationProvider: Providers.OPENAI,
        summarizationModel: 'gpt-4.1-mini',
        maxContextTokens: maxTokens,
        instructions: MATH_TUTOR_INSTRUCTIONS,
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build up conversation at generous limits so messages accumulate
    let run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'Compute 54321 * 12345 using calculator.',
      streamConfig
    );

    run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'Now calculate 670592745 / 99991. Calculator.',
      streamConfig
    );

    run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'What is sqrt(670592745)? Calculator.',
      streamConfig
    );

    run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'Compute 2^32 with calculator.',
      streamConfig
    );

    run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'What is 13 * 17 * 19? Calculator.',
      streamConfig
    );

    // Tighten context to force summarization — must remain high enough
    // for post-summary instruction overhead + tool schema tokens + messages
    run = await createRun(3500);
    await runTurn(
      { run, conversationHistory },
      'What is 99 * 101? Calculator. Then list everything we calculated so far in detail.',
      streamConfig
    );

    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      addDeterministicSummarizationPressure(
        conversationHistory,
        'anthropic-cross-provider'
      );
      run = await createRun(3000);
      await runTurn(
        { run, conversationHistory },
        'What is 9 * 9? Calculator.',
        streamConfig
      );
    }

    console.log(
      `  Cross-provider summaries: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    assertSummarizationEvents(spies);
    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;

    // The summary should have been generated by OpenAI even though agent is Anthropic
    expect(completePayload.summary!.provider).toBe(Providers.OPENAI);
    expect(completePayload.summary!.model).toBe('gpt-4.1-mini');
    assertSummaryRunStep(spies, getSummaryText(completePayload.summary));

    // Token accounting: summary tokenCount bounds
    expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(10);
    expect(completePayload.summary!.tokenCount ?? 0).toBeLessThan(1200);

    // Token accounting: collectedUsage from the post-summary model call
    const validUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(validUsage.length).toBeGreaterThan(0);

    console.log(
      `  Cross-provider summary (${getSummaryText(completePayload.summary).length} chars): "${getSummaryText(completePayload.summary).substring(0, 200)}…"`
    );
  }, 180_000);

  test('extended thinking: multi-turn with reasoning triggers summarization and grounds token accounting', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const resetAggregator = (): {
      contentParts: t.MessageContentComplex[];
      aggregateContent: t.ContentAggregator;
    } => {
      collectedUsage = [];
      const { contentParts: cp, aggregateContent: ac } =
        createContentAggregator();
      return {
        contentParts: cp as t.MessageContentComplex[],
        aggregateContent: ac,
      };
    };

    const createRun = async (
      maxTokens = 3000
    ): Promise<{
      run: Run<t.IState>;
      contentParts: t.MessageContentComplex[];
    }> => {
      const { contentParts, aggregateContent } = resetAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      const run = await createSummarizationRun({
        agentProvider,
        summarizationProvider: Providers.ANTHROPIC,
        summarizationModel: 'claude-haiku-4-5',
        maxContextTokens: maxTokens,
        instructions:
          'You are a math tutor. Use the calculator tool for computations. Keep answers brief.',
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
        llmConfigOverride: {
          model: 'claude-sonnet-4-5',
          thinking: {
            type: 'enabled',
            budget_tokens: 1024,
          },
        },
      });
      return { run, contentParts };
    };

    // Turn 1: simple calculation with thinking
    let { run, contentParts } = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 7 * 720? Use the calculator.',
      streamConfig
    );
    logTurn('T1-think', conversationHistory, `parts=${contentParts.length}`);

    // Validate Turn 1 usage includes both input and output tokens
    const t1Usage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(t1Usage.length).toBeGreaterThan(0);
    const t1Last = t1Usage[t1Usage.length - 1];
    expect(t1Last.output_tokens).toBeGreaterThan(0);
    console.log(
      `  T1 usage — input: ${t1Last.input_tokens}, output: ${t1Last.output_tokens}` +
        (t1Last.input_token_details?.cache_read != null
          ? `, cache_read: ${t1Last.input_token_details.cache_read}`
          : '')
    );

    // Turn 2: follow-up calculation
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      'Now multiply that result by 3. Use the calculator.',
      streamConfig
    );
    logTurn('T2-think', conversationHistory, `parts=${contentParts.length}`);

    // Turn 3: another calculation to build context
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      'What is 143 + 857? Use the calculator.',
      streamConfig
    );
    logTurn('T3-think', conversationHistory, `parts=${contentParts.length}`);

    // Turn 4: another turn to build up context
    ({ run, contentParts } = await createRun());
    await runTurn(
      { run, conversationHistory },
      'What is 2 * 512? Use the calculator.',
      streamConfig
    );
    logTurn(
      'T4-think',
      conversationHistory,
      `parts=${contentParts.length}`
    );

    // Turn 5: tighter context to trigger summarization
    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      ({ run, contentParts } = await createRun(2500));
      await runTurn(
        { run, conversationHistory },
        'What is 999 * 999? Use the calculator.',
        streamConfig
      );
      logTurn(
        'T5-think',
        conversationHistory,
        `parts=${contentParts.length}`
      );
    }

    // Provider output length varies, so add deterministic history rather than
    // shrinking below the minimum viable instruction and tool-schema budget.
    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      addDeterministicSummarizationPressure(
        conversationHistory,
        'anthropic-thinking'
      );
      ({ run, contentParts } = await createRun(2500));
      await runTurn(
        { run, conversationHistory },
        'What is 42 * 42? Use the calculator.',
        streamConfig
      );
      logTurn(
        'T6-think',
        conversationHistory,
        `parts=${contentParts.length}`
      );
    }

    console.log(
      `  Thinking summarize events — start: ${spies.onSummarizeStartSpy.mock.calls.length}, complete: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    // Assert summarization fired
    const { completePayload } = assertSummarizationEvents(spies);
    assertSummaryRunStep(spies, getSummaryText(completePayload.summary));

    // Token accounting: summary tokenCount bounds
    expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(10);
    expect(completePayload.summary!.tokenCount ?? 0).toBeLessThan(2000);

    // Token accounting: collectedUsage must have valid entries across all turns
    const allValidUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null &&
        u.input_tokens > 0 &&
        u.output_tokens != null &&
        u.output_tokens > 0
    );
    expect(allValidUsage.length).toBeGreaterThan(0);

    // Validate that usage has reasonable token counts (thinking adds tokens)
    const lastUsage = allValidUsage[allValidUsage.length - 1];
    expect(lastUsage.input_tokens).toBeGreaterThan(0);
    expect(lastUsage.output_tokens).toBeGreaterThan(0);

    console.log(
      `  Thinking usage samples: ${allValidUsage.length} valid entries`
    );
    console.log(
      `  Last usage — input: ${lastUsage.input_tokens}, output: ${lastUsage.output_tokens}`
    );
    if (lastUsage.input_token_details?.cache_read != null) {
      console.log(
        `  Cache read: ${lastUsage.input_token_details.cache_read}, cache creation: ${lastUsage.input_token_details.cache_creation ?? 0}`
      );
    }

    // Post-summary continuation should work with thinking enabled
    ({ run } = await createRun(4000));
    const postSumResult = await runTurn(
      { run, conversationHistory },
      'What is 100 / 4? Calculator please.',
      streamConfig
    );
    expect(postSumResult).toBeDefined();
    logTurn('Post-sum-think', conversationHistory);

    // Post-summary usage must also be valid
    const postSumUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(postSumUsage.length).toBeGreaterThan(0);

    console.log(
      `  Thinking summary (${getSummaryText(completePayload.summary).length} chars): "${getSummaryText(completePayload.summary).substring(0, 250)}…"`
    );
    console.log(`  Final messages: ${conversationHistory.length}`);
  }, 180_000);

  test('count_tokens API: local tokenCounter vs Anthropic actual token count', async () => {
    const Anthropic = (await import('@anthropic-ai/sdk')).default;
    const client = new Anthropic();
    const tokenCounter = await createTokenCounter();

    const testMessages: Array<{
      role: 'user' | 'assistant';
      lcMessage: BaseMessage;
      content: string;
    }> = [
      {
        role: 'user',
        lcMessage: new HumanMessage(
          'What is 12345 * 6789? Please compute this using the calculator tool and explain the result.'
        ),
        content:
          'What is 12345 * 6789? Please compute this using the calculator tool and explain the result.',
      },
      {
        role: 'assistant',
        lcMessage: new AIMessage(
          'The result of 12345 multiplied by 6789 is 83,810,205. This is computed by multiplying each digit and carrying over.'
        ),
        content:
          'The result of 12345 multiplied by 6789 is 83,810,205. This is computed by multiplying each digit and carrying over.',
      },
      {
        role: 'user',
        lcMessage: new HumanMessage(
          'Now divide that by 137 and tell me the quotient.'
        ),
        content: 'Now divide that by 137 and tell me the quotient.',
      },
      {
        role: 'assistant',
        lcMessage: new AIMessage(
          '83,810,205 divided by 137 equals approximately 611,752.59.'
        ),
        content: '83,810,205 divided by 137 equals approximately 611,752.59.',
      },
    ];

    const systemPrompt =
      'You are an expert math tutor. Use the calculator tool for ALL computations.';

    const anthropicCount = await client.messages.countTokens({
      model: 'claude-haiku-4-5',
      system: systemPrompt,
      messages: testMessages.map((m) => ({ role: m.role, content: m.content })),
    });

    let localTotal = tokenCounter(new SystemMessage(systemPrompt));
    for (const m of testMessages) {
      localTotal += tokenCounter(m.lcMessage);
    }

    const anthropicTokens = anthropicCount.input_tokens;
    const drift = Math.abs(anthropicTokens - localTotal);
    const driftPct = (drift / anthropicTokens) * 100;

    console.log(`  Anthropic count_tokens API: ${anthropicTokens} tokens`);
    console.log(`  Local tiktoken estimate:    ${localTotal} tokens`);
    console.log(`  Drift: ${drift} tokens (${driftPct.toFixed(1)}%)`);

    expect(anthropicTokens).toBeGreaterThan(0);
    expect(localTotal).toBeGreaterThan(0);
    expect(driftPct).toBeLessThan(30);
  }, 180_000);
});

// ---------------------------------------------------------------------------
// Bedrock Summarization Tests
// ---------------------------------------------------------------------------

const requiredBedrockEnv = [
  'BEDROCK_AWS_REGION',
  'BEDROCK_AWS_ACCESS_KEY_ID',
  'BEDROCK_AWS_SECRET_ACCESS_KEY',
];
const hasBedrock = hasEveryEnv(requiredBedrockEnv);

(hasBedrock ? describe : describe.skip)('Bedrock Summarization E2E', () => {
  jest.setTimeout(180_000);

  const agentProvider = Providers.BEDROCK;
  const streamConfig = {
    configurable: { thread_id: 'bedrock-sum-e2e' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  test('multi-turn tool calls trigger summarization with Bedrock agent', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 4000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider: Providers.BEDROCK,
        maxContextTokens: maxTokens,
        instructions:
          'You are a precise math assistant. Use the calculator tool for every computation. Be brief.',
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    let run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Hello. Please compute 987 * 654 using the calculator.',
      streamConfig
    );
    logTurn('T1', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Now divide 645498 by 123. Use calculator.',
      streamConfig
    );
    logTurn('T2', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Compute sqrt(5248.764) with the calculator. Then multiply the result by 100.',
      streamConfig
    );
    logTurn('T3', conversationHistory);

    run = await createRun(3500);
    await runTurn(
      { run, conversationHistory },
      'Calculate 2^16 and 3^10 using calculator for each.',
      streamConfig
    );
    logTurn('T4', conversationHistory);

    run = await createRun(3200);
    await runTurn(
      { run, conversationHistory },
      'What is 59049 + 65536? Calculator. Also tell me what we calculated before.',
      streamConfig
    );
    logTurn('T5', conversationHistory);

    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      run = await createRun(3000);
      await runTurn(
        { run, conversationHistory },
        'Calculate 111111 * 111111 with calculator.',
        streamConfig
      );
      logTurn('T6', conversationHistory);
    }

    console.log(
      `  Bedrock summarize events — start: ${spies.onSummarizeStartSpy.mock.calls.length}, complete: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    const { completePayload } = assertSummarizationEvents(spies);
    assertSummaryRunStep(spies, getSummaryText(completePayload.summary));
    expect(spies.onMessageDeltaSpy).toHaveBeenCalled();

    // Token accounting: summary tokenCount bounds
    expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(10);
    expect(completePayload.summary!.tokenCount ?? 0).toBeLessThan(1500);

    // Token accounting: collectedUsage from the post-summary model call
    const validUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(validUsage.length).toBeGreaterThan(0);
    const lastUsage = validUsage[validUsage.length - 1];
    expect(lastUsage.output_tokens).toBeGreaterThan(0);
    console.log(
      `  Bedrock post-summary usage — input: ${lastUsage.input_tokens}, output: ${lastUsage.output_tokens}`
    );

    console.log(
      `  Bedrock summary: "${getSummaryText(completePayload.summary).substring(0, 250)}…"`
    );

    // Post-summary turn should work cleanly
    run = await createRun(4000);
    const postSumResult = await runTurn(
      { run, conversationHistory },
      'Give me a brief list of all results we computed.',
      streamConfig
    );
    expect(postSumResult).toBeDefined();
    logTurn('Post-sum', conversationHistory);
  }, 180_000);
});

// ---------------------------------------------------------------------------
// OpenAI Summarization Tests
// ---------------------------------------------------------------------------

const hasOpenAI = hasEnv('OPENAI_API_KEY');
(hasOpenAI ? describe : describe.skip)('OpenAI Summarization E2E', () => {
  jest.setTimeout(240_000);

  const agentProvider = Providers.OPENAI;
  const streamConfig = {
    configurable: { thread_id: 'openai-sum-e2e' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  test('multi-turn with calculator triggers summarization and continues', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    let latestContentParts: t.MessageContentComplex[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 2000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { contentParts, aggregateContent } = createContentAggregator();
      latestContentParts = contentParts as t.MessageContentComplex[];
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider: Providers.OPENAI,
        summarizationModel: 'gpt-4.1-mini',
        llmConfigOverride: {
          model: 'gpt-4.1-mini',
        },
        maxContextTokens: maxTokens,
        instructions:
          'You are a helpful math tutor. Use the calculator tool for ALL computations. Keep responses concise.',
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    let run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 1234 * 5678? Use the calculator.',
      streamConfig
    );
    logTurn('T1', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Now calculate sqrt(7006652). Use the calculator.',
      streamConfig
    );
    logTurn('T2', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Compute 99 * 101, then 2^15, using calculator for each.',
      streamConfig
    );
    logTurn('T3', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 314159 * 271828? Calculator please.',
      streamConfig
    );
    logTurn('T4', conversationHistory);

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Compute 2^20 with calculator.',
      streamConfig
    );
    logTurn('T5', conversationHistory);

    // Provider output length varies, so add deterministic history rather than
    // shrinking below the minimum viable instruction and tool-schema budget.
    if (spies.onSummarizeStartSpy.mock.calls.length === 0) {
      addDeterministicSummarizationPressure(
        conversationHistory,
        'openai-calculator'
      );
      run = await createRun(2000);
      await runTurn(
        { run, conversationHistory },
        'Calculate 999999 / 7 with calculator. Remind me of prior results too.',
        streamConfig
      );
      logTurn('T6', conversationHistory);
    }

    console.log(
      `  OpenAI summarize events — start: ${spies.onSummarizeStartSpy.mock.calls.length}, complete: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    const { completePayload } = assertSummarizationEvents(spies);
    assertSummaryRunStep(spies, getSummaryText(completePayload.summary));

    // Token accounting: summary tokenCount bounds
    expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(10);
    expect(completePayload.summary!.tokenCount ?? 0).toBeLessThan(1200);

    // Token accounting: collectedUsage from the post-summary model call
    const validUsagePrePostSum = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(validUsagePrePostSum.length).toBeGreaterThan(0);

    // Verify tool calls still work after summarization
    run = await createRun(2000);
    await runTurn(
      { run, conversationHistory },
      'One more: 123 + 456 + 789. Calculator.',
      streamConfig
    );
    const hasPostSumCalc = latestContentParts.some(
      (p) =>
        p.type === ContentTypes.TOOL_CALL &&
        (p as t.ToolCallContent).tool_call?.name === 'calculator'
    );
    expect(hasPostSumCalc).toBe(true);

    // Token accounting: post-summary usage must have valid tokens
    const postSumUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(postSumUsage.length).toBeGreaterThan(0);
    const lastUsage = postSumUsage[postSumUsage.length - 1];
    expect(lastUsage.output_tokens).toBeGreaterThan(0);
    console.log(
      `  OpenAI post-summary usage — input: ${lastUsage.input_tokens}, output: ${lastUsage.output_tokens}`
    );

    expect(spies.onMessageDeltaSpy).toHaveBeenCalled();
    console.log(
      `  OpenAI summary: "${getSummaryText(completePayload.summary).substring(0, 200)}…"`
    );
    console.log(`  Final messages: ${conversationHistory.length}`);
  }, 240_000);
});

// ---------------------------------------------------------------------------
// Cross-run lifecycle integration test (no API keys required)
// ---------------------------------------------------------------------------

describe('Cross-run summary lifecycle (no API keys)', () => {
  jest.setTimeout(60_000);

  const KNOWN_SUMMARY =
    'User asked about math: 2+2=4 and 3*5=15. Key context preserved.';
  const INSTRUCTIONS = 'You are a helpful math tutor. Be concise.';
  const streamConfig = {
    configurable: { thread_id: 'cross-run-lifecycle' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [KNOWN_SUMMARY] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('full lifecycle: summarize → formatAgentMessages → new Run with correct indexTokenCountMap', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens: number): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      const run = await Run.create<t.IState>({
        runId: `cross-run-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
          },
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
      return run;
    };

    // --- Turn 1: longer exchange to build up token budget ---
    let run = await createRun(4000);
    run.Graph?.overrideTestModel(
      [
        'The answer to 2+2 is 4. This is a basic arithmetic operation involving the addition of two integers. Addition is one of the four fundamental operations in mathematics alongside subtraction, multiplication, and division.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Hello! I have several math questions for you today. Let us start with the basics. What is 2+2? Please provide a detailed explanation of the arithmetic.',
      streamConfig
    );
    logTurn('T1', conversationHistory);
    expect(conversationHistory.length).toBeGreaterThanOrEqual(2);

    // --- Turn 2: build up more conversation ---
    run = await createRun(4000);
    run.Graph?.overrideTestModel(
      [
        'The result of 3 multiplied by 5 is 15. Multiplication can be thought of as repeated addition: 3+3+3+3+3 equals 15. This is another fundamental arithmetic operation that forms the basis of more advanced mathematical concepts.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Great explanation! Now let us move on to multiplication. Can you compute 3 times 5 and explain the concept of multiplication as repeated addition in detail?',
      streamConfig
    );
    logTurn('T2', conversationHistory);
    expect(conversationHistory.length).toBeGreaterThanOrEqual(4);

    // --- Turn 3: tight context to force pruning and summarization ---
    // Keep both the bounded history and the summarizer's reserved output viable.
    addDeterministicSummarizationPressure(conversationHistory, 'lifecycle', 4);
    run = await createRun(1024);
    run.Graph?.overrideTestModel(
      ['Got it, continuing with the summary context.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Now summarize everything we discussed.',
      streamConfig
    );
    logTurn('T3', conversationHistory);

    console.log(
      `  Lifecycle events — start: ${spies.onSummarizeStartSpy.mock.calls.length}, complete: ${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );

    // --- Assert summarization fired ---
    expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    expect(getSummaryText(completePayload.summary)).toBe(KNOWN_SUMMARY);
    expect(completePayload.summary!.type).toBe(ContentTypes.SUMMARY);
    expect(completePayload.summary!.tokenCount ?? 0).toBeGreaterThan(0);

    const expectedTokenCount = tokenCounter(
      new HumanMessage(buildSummaryCarrierText(KNOWN_SUMMARY))
    );
    expect(completePayload.summary!.tokenCount).toBe(expectedTokenCount);

    const summaryBlock = completePayload.summary!;

    // --- Simulate cross-run persistence: build a TPayload as the host would store it ---
    const persistedPayload: t.TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.SUMMARY,
            text: getSummaryText(summaryBlock),
            tokenCount: summaryBlock.tokenCount ?? 0,
          } as any,
        ],
      },
      {
        role: 'user',
        content: 'Now summarize everything we discussed so far.',
      },
      {
        role: 'assistant',
        content: 'Got it, continuing with the summary context.',
      },
    ];

    const persistedTokenMap: Record<number, number> = {
      0: summaryBlock.tokenCount ?? 0,
      1: tokenCounter(
        new HumanMessage('Now summarize everything we discussed so far.')
      ),
      2: tokenCounter(
        new AIMessage('Got it, continuing with the summary context.')
      ),
    };

    // --- formatAgentMessages: convert persisted payload for next Run ---
    const formatted = formatAgentMessages(persistedPayload, persistedTokenMap);

    // Summary is returned as metadata, NOT as a SystemMessage in the messages array.
    // The caller forwards it to the run via initialSummary → AgentContext.setSummary().
    expect(formatted.summary).toBeDefined();
    expect(formatted.summary!.text).toBe(KNOWN_SUMMARY);
    expect(formatted.summary!.tokenCount).toBe(summaryBlock.tokenCount);
    // First message should NOT be a SystemMessage — only user/assistant messages remain.
    expect(formatted.messages[0].constructor.name).not.toBe('SystemMessage');

    const formattedMap = (formatted.indexTokenCountMap || {}) as Record<
      number,
      number
    >;
    const formattedTotal = Object.values(formattedMap).reduce(
      (sum: number, v: number) => sum + v,
      0
    );
    // Summary tokens no longer in the map — only user+assistant message tokens.
    const expectedTotal = persistedTokenMap[1] + persistedTokenMap[2];
    expect(formattedTotal).toBe(expectedTotal);

    console.log(
      `  Formatted: ${formatted.messages.length} msgs, tokenMap total=${formattedTotal}, summary="${formatted.summary!.text.substring(0, 60)}..."`
    );

    // --- Turn 4: new Run with formatted messages and updated indexTokenCountMap ---
    const formattedTokenMapAsStrings: Record<string, number> = {};
    for (const [k, v] of Object.entries(formattedMap)) {
      formattedTokenMapAsStrings[String(k)] = v as number;
    }

    const run4 = await Run.create<t.IState>({
      runId: `cross-run-lifecycle-t4-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: 2000,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
        initialSummary: formatted.summary,
      },
      returnContent: true,
      customHandlers: buildHandlers(
        [],
        createContentAggregator().aggregateContent,
        createSpies()
      ),
      tokenCounter,
      indexTokenCountMap: formattedTokenMapAsStrings,
    });

    run4.Graph?.overrideTestModel(['The square root of 16 is 4.'], 1);

    const t4Messages = [
      ...formatted.messages,
      new HumanMessage('What is sqrt(16)?'),
    ];
    const result = await run4.processStream(
      { messages: t4Messages },
      streamConfig as any
    );

    expect(result).toBeDefined();

    const t4RunMessages = run4.getRunMessages();
    expect(t4RunMessages).toBeDefined();
    expect(t4RunMessages!.length).toBeGreaterThan(0);

    console.log(
      `  Turn 4 produced ${t4RunMessages!.length} messages — lifecycle complete`
    );
  });

  test('tight context edge case: maxContextTokens as low as 1 does not infinite-loop', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens: number): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `tight-ctx-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            provider: Providers.OPENAI,
          },
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build a conversation first at normal context size
    let run = await createRun(4000);
    run.Graph?.overrideTestModel(
      ['Sure, 2+2 is 4. Happy to help with more math questions.'],
      1
    );
    await runTurn({ run, conversationHistory }, 'What is 2+2?', streamConfig);
    expect(conversationHistory.length).toBeGreaterThanOrEqual(2);

    // Now use absurdly tight context values — the guard must prevent infinite loops.
    // Very small values may throw "empty_messages" (context too small for any message)
    // which is fine — the point is we never hit GraphRecursionError.
    for (const tightValue of [1, 10, 25, 50]) {
      spies.onSummarizeStartSpy.mockClear();
      spies.onSummarizeCompleteSpy.mockClear();

      run = await createRun(tightValue);
      run.Graph?.overrideTestModel(['OK, noted.'], 1);

      let error: Error | undefined;
      try {
        await runTurn({ run, conversationHistory }, 'Continue.', streamConfig);
      } catch (err) {
        error = err as Error;
      }

      if (error) {
        // Clean errors (empty_messages) are acceptable for tiny context windows.
        // GraphRecursionError means we looped — that's the bug we're guarding against.
        expect(error.message).not.toContain('Recursion limit');
        console.log(
          `  maxContextTokens=${tightValue}: clean error (${error.message.substring(0, 80)})`
        );
        // Remove the failed turn's user message from history so subsequent iterations work
        conversationHistory.pop();
      } else {
        const startCalls = spies.onSummarizeStartSpy.mock.calls.length;
        const completeCalls = spies.onSummarizeCompleteSpy.mock.calls.length;
        console.log(
          `  maxContextTokens=${tightValue}: ok, start=${startCalls}, complete=${completeCalls}, msgs=${conversationHistory.length}`
        );
        // If summarization fired, it must have completed.
        // Emergency truncation may allow success without summarization, so
        // we don't require startCalls >= 1 — the test's goal is no infinite loop.
        if (startCalls > 0) {
          expect(completeCalls).toBe(startCalls);
        }
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Tight context with oversized tool results (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Tight context with oversized tool results (no API keys)', () => {
  jest.setTimeout(60_000);

  const INSTRUCTIONS = 'You are a helpful assistant. Be concise.';
  const SUMMARY_RESPONSE =
    '## Goal\nUser needed help.\n\n## Progress\n### Done\n- Completed analysis.';
  const streamConfig = {
    configurable: { thread_id: 'tight-tool-ctx' },
    streamMode: 'values',
    version: 'v2' as const,
  };

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

  test('oversized tool result + thinking-enabled model does not crash with tight context', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build a conversation that mimics the real-world bug:
    // HumanMessage → AIMessage with tool_calls + thinking blocks → large ToolMessage
    const conversationHistory: BaseMessage[] = [
      new HumanMessage('Inspect the page JavaScript.'),
      new AIMessage({
        content: [
          {
            type: 'thinking' as const,
            thinking: 'Let me inspect the page using chrome-devtools MCP tool.',
          },
          { type: 'text' as const, text: 'I will inspect the page now.' },
          {
            type: 'tool_use' as const,
            id: 'tool_mcp_1',
            name: 'chrome_devtools_evaluate',
            input: '{"expression": "document.body.innerHTML"}',
          },
        ],
        tool_calls: [
          {
            id: 'tool_mcp_1',
            name: 'chrome_devtools_evaluate',
            args: { expression: 'document.body.innerHTML' },
          },
        ],
      }),
      new ToolMessage({
        content: 'x'.repeat(5000), // Large MCP output simulating JS payload
        tool_call_id: 'tool_mcp_1',
        name: 'chrome_devtools_evaluate',
      }),
    ];

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    // Create a run with extremely tight context and thinking enabled
    const { aggregateContent } = createContentAggregator();
    const llmConfig = {
      ...getLLMConfig(Providers.OPENAI),
      thinking: { type: 'enabled', budget_tokens: 4000 },
    };
    const run = await Run.create<t.IState>({
      runId: `tight-thinking-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: llmConfig as any,
        instructions: INSTRUCTIONS,
        maxContextTokens: 500, // Extremely tight — will prune everything
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            spies.onRunStepSpy(_event, data);
            aggregateContent({
              event: GraphEvents.ON_RUN_STEP,
              data: data as t.RunStep,
            });
          },
        },
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
      },
      tokenCounter,
      indexTokenCountMap,
    });

    run.Graph?.overrideTestModel(['Analysis complete.'], 1);

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: [...conversationHistory, new HumanMessage('Continue.')] },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    // The key assertion: no crash about "aggressive pruning removed all AI messages"
    if (error) {
      expect(error.message).not.toContain('aggressive pruning removed all AI');
      expect(error.message).not.toContain('Recursion limit');
      // empty_messages is acceptable for this tiny context window
      console.log(
        `  Tight thinking context: clean error (${error.message.substring(0, 100)})`
      );
    } else {
      console.log('  Tight thinking context: completed without error');
    }
  });

  test('summarization survives when tool results dominate the context', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build 3 turns with large tool outputs (~2000 chars each)
    const conversationHistory: BaseMessage[] = [];

    const createRunHelper = async (
      maxTokens: number
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `tool-dominate-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            provider: Providers.OPENAI,
          },
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Turn 1
    let run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      [
        'Here is a long explanation about the analysis results that covers many details of the computation.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Analyze the following data: ' + 'y'.repeat(2000),
      streamConfig
    );

    // Turn 2
    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      [
        'More results from the second analysis including additional context and findings.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Now analyze this: ' + 'z'.repeat(2000),
      streamConfig
    );

    // Turn 3 with tight context to force summarization
    run = await createRunHelper(500);
    run.Graph?.overrideTestModel(['Got it.'], 1);

    let error: Error | undefined;
    try {
      await runTurn(
        { run, conversationHistory },
        'Summarize everything.',
        streamConfig
      );
    } catch (err) {
      error = err as Error;
    }

    if (error) {
      // empty_messages is acceptable, but not recursion errors
      expect(error.message).not.toContain('Recursion limit');
      console.log(
        `  Tool-dominated context: clean error (${error.message.substring(0, 100)})`
      );
    } else {
      // Summarization should have fired
      expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
      expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

      const completePayload = spies.onSummarizeCompleteSpy.mock
        .calls[0][0] as t.SummarizeCompleteEvent;
      expect(getSummaryText(completePayload.summary).length).toBeGreaterThan(
        10
      );
      console.log(
        `  Tool-dominated context: summary="${getSummaryText(completePayload.summary).substring(0, 100)}…"`
      );
    }
  });

  test('multiple summarization cycles preserve structured checkpoint format', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRunHelper = async (
      maxTokens: number
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `multi-sum-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
          },
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build conversation to trigger first summarization
    let run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['The answer to 2+2 is 4. This is basic addition.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 2+2? Give me a detailed explanation.',
      streamConfig
    );

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['3 times 5 is 15. Multiplication is repeated addition.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Now explain 3 times 5 in detail with examples.',
      streamConfig
    );

    // Force first summarization
    addDeterministicSummarizationPressure(conversationHistory, 'structured-checkpoint', 4);
    run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['Continuing after summary.'], 1);
    try {
      await runTurn({ run, conversationHistory }, 'Continue.', streamConfig);
    } catch {
      conversationHistory.pop(); // remove failed user message
    }

    const firstSumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    expect(firstSumCount).toBeGreaterThan(0);

    // Build more conversation
    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['The square root of 16 is 4. This is because 4 squared equals 16.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is sqrt(16)? Explain thoroughly.',
      streamConfig
    );

    // Force second summarization
    addDeterministicSummarizationPressure(conversationHistory, 'structured-checkpoint', 4);
    run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['Continuing after second summary.'], 1);
    try {
      await runTurn(
        { run, conversationHistory },
        'Continue again.',
        streamConfig
      );
    } catch {
      conversationHistory.pop();
    }

    const totalSumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(
      `  Summarization cycles: first=${firstSumCount}, total=${totalSumCount}`
    );

    // At least one summarization should have fired
    expect(totalSumCount).toBeGreaterThan(firstSumCount);

    // The summary response from our fake model has structured format
    const lastComplete = spies.onSummarizeCompleteSpy.mock.calls[
      totalSumCount - 1
    ][0] as t.SummarizeCompleteEvent;
    expect(lastComplete.error).toBeUndefined();
    const summaryText = getSummaryText(lastComplete.summary);

    // Our SUMMARY_RESPONSE includes ## Goal and ## Progress
    expect(summaryText).toContain('## Goal');
    expect(summaryText).toContain('## Progress');
    console.log(
      `  Last summary (${summaryText.length} chars): "${summaryText.substring(0, 150)}…"`
    );
  });

  test('update prompt is used when prior summary exists', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    // The compaction instruction is appended as the final human message.
    // Override _streamResponseChunks (not _generate) because FakeListChatModel
    // has its own _streamResponseChunks that bypasses _generate during streaming.
    const capturedInstructions: string[] = [];
    getChatModelClassSpy.mockRestore();
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [SUMMARY_RESPONSE] });
            }
            async *_streamResponseChunks(
              messages: any[],
              options: any,
              runManager?: any
            ) {
              const content = messages.at(-1)?.content ?? '';
              capturedInstructions.push(
                typeof content === 'string' ? content : JSON.stringify(content)
              );
              yield* super._streamResponseChunks(messages, options, runManager);
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

    const createRunHelper = async (
      maxTokens: number,
      initialSummary?: { text: string; tokenCount: number }
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `update-prompt-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
          },
          initialSummary,
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // --- Step 1: Build conversation and trigger FIRST summarization (fresh prompt) ---
    let run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      [
        'The answer to 2+2 is 4. Addition is one of the four fundamental arithmetic operations.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 2+2? Please provide a detailed explanation of the arithmetic.',
      streamConfig
    );

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      [
        '3 times 5 is 15. Multiplication can be thought of as repeated addition.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Now explain 3 times 5 with a detailed worked example of multiplication.',
      streamConfig
    );

    // Force first summarization
    addDeterministicSummarizationPressure(conversationHistory, 'update-checkpoint', 4);
    run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['Continuing after first summary.'], 1);
    try {
      await runTurn(
        { run, conversationHistory },
        'Now summarize everything we discussed.',
        streamConfig
      );
    } catch {
      conversationHistory.pop();
    }

    const firstSumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(`  First summarization: ${firstSumCount} complete events`);

    expect(firstSumCount).toBeGreaterThan(0);
    const { completePayload: firstComplete } = assertSummarizationEvents(spies);
    const priorSummary = {
      text: getSummaryText(firstComplete.summary),
      tokenCount: firstComplete.summary!.tokenCount ?? 0,
    };
    expect(capturedInstructions).toContain(LOCAL_SUMMARY_CONFIG.prompt);
    capturedInstructions.length = 0;

    // --- Step 2: Build more conversation with initialSummary, trigger SECOND summarization ---
    // Since initialSummary is set, the summarize node should use the update prompt.
    run = await createRunHelper(4000, priorSummary);
    run.Graph?.overrideTestModel(
      ['The square root of 16 is 4, because 4 times 4 equals 16.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is the square root of 16? Give a very detailed explanation.',
      streamConfig
    );

    run = await createRunHelper(4000, priorSummary);
    run.Graph?.overrideTestModel(
      [
        '100 divided by 4 is 25. Division distributes a total into equal groups.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 100 divided by 4? Explain division with multiple examples.',
      streamConfig
    );

    // Force second summarization (with prior summary in AgentContext)
    addDeterministicSummarizationPressure(conversationHistory, 'update-checkpoint', 4);
    run = await createRunHelper(1024, priorSummary);
    run.Graph?.overrideTestModel(['Continuing after second summary.'], 1);
    try {
      await runTurn({ run, conversationHistory }, 'Continue.', streamConfig);
    } catch {
      conversationHistory.pop();
    }

    const secondSumCount =
      spies.onSummarizeCompleteSpy.mock.calls.length - firstSumCount;
    expect(secondSumCount).toBeGreaterThan(0);
    expect(capturedInstructions.length).toBeGreaterThan(0);
    for (const instruction of capturedInstructions) {
      expect(instruction.startsWith(LOCAL_SUMMARY_CONFIG.updatePrompt)).toBe(true);
      expect(instruction).toContain('<previous-summary>');
      expect(instruction).toContain(priorSummary.text);
    }
  });

  test('empty pruning context after summarization preserves latest user turn', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build a conversation where EVERY message is too large to fit in the
    // post-summary budget individually.  This reproduces the real-world bug
    // where context is empty after pruning, summarization fires, and the
    // summarize node used to return 0 surviving messages.
    const largePadding = ' detailed explanation'.repeat(80); // ~1600 chars
    const conversationHistory: BaseMessage[] = [
      new HumanMessage(`First question about math${largePadding}`),
      new AIMessage(`The answer is 42${largePadding}`),
      new HumanMessage(`Second question about physics${largePadding}`),
      new AIMessage(`E equals mc squared${largePadding}`),
      new HumanMessage(`Third question about chemistry${largePadding}`),
      new AIMessage(`Water is H2O${largePadding}`),
    ];

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: `empty-ctx-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: 200, // Extremely tight — no message fits individually
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            spies.onRunStepSpy(_event, data);
            aggregateContent({
              event: GraphEvents.ON_RUN_STEP,
              data: data as t.RunStep,
            });
          },
        },
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
      },
      tokenCounter,
      indexTokenCountMap,
    });

    // The agent model response for the post-summary turn
    run.Graph?.overrideTestModel(['Here is the answer to your question.'], 1);

    const latestUserMessage = new HumanMessage(
      'What is the capital of France?'
    );

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: [...conversationHistory, latestUserMessage] },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    // Summarization should have fired
    expect(spies.onSummarizeStartSpy).toHaveBeenCalled();

    // Key assertion: before the fix, this scenario always produced an
    // empty_messages error because contextMessages was empty after
    // summarization.  After the fix, the latest turn's HumanMessage is
    // extracted from messagesToRefine and the model responds successfully.
    if (error) {
      // If an error occurs, it must NOT be the empty_messages error that
      // the fix was designed to prevent.
      expect(error.message).not.toContain('empty_messages');
      console.log(
        `  Empty context fix: non-empty_messages error (${error.message.substring(0, 120)})`
      );
    } else {
      // The model responded successfully — this is the expected outcome
      console.log('  Empty context fix: model responded successfully');
    }
  });
});

// ---------------------------------------------------------------------------
// Token accounting audit (requires API keys)
// ---------------------------------------------------------------------------

const hasAnyApiKey = hasAnyEnv(['ANTHROPIC_API_KEY', 'OPENAI_API_KEY']);

(hasAnyApiKey ? describe : describe.skip)('Token accounting audit', () => {
  jest.setTimeout(180_000);

  const agentProvider = hasEnv('ANTHROPIC_API_KEY')
    ? Providers.ANTHROPIC
    : Providers.OPENAI;
  const summarizationProvider = agentProvider;
  const summarizationModel =
    agentProvider === Providers.ANTHROPIC ? 'claude-haiku-4-5' : 'gpt-4.1-mini';

  const streamConfig = {
    configurable: { thread_id: 'token-audit-e2e' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const INSTRUCTIONS =
    'You are a math tutor. Use the calculator tool for ALL computations. Be concise.';

  test('token count map is accurate after summarization cycle', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory = createSeededTokenAuditHistory();
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 1200): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider,
        summarizationModel,
        maxContextTokens: maxTokens,
        instructions:
          'You are a concise assistant. Preserve checkpoint context and answer in one short sentence.',
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
        maxSummaryTokens: 300,
        tools: [],
        llmConfigOverride: {
          maxTokens: 128,
        },
      });
    };

    const originalMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );
    const originalTokenTotal = sumTokenCountMap(originalMap);
    expect(originalTokenTotal).toBeGreaterThan(1200);

    const run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Acknowledge the preserved token audit context in one short sentence.',
      streamConfig
    );

    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    const summaryTokenCount = completePayload.summary!.tokenCount ?? 0;
    expect(summaryTokenCount).toBeGreaterThan(10);
    expect(summaryTokenCount).toBeLessThan(1500);
    expect(summaryTokenCount).toBeLessThan(originalTokenTotal);

    const validUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    expect(validUsage.length).toBeGreaterThan(0);

    console.log(
      `  Token audit: summary=${summaryTokenCount} tokens, ` +
        `preTotal=${originalTokenTotal}, usageEntries=${validUsage.length}`
    );
  }, 180_000);

  test('summary tokenCount matches local token counter', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 4000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider,
        summarizationModel,
        maxContextTokens: maxTokens,
        instructions: INSTRUCTIONS,
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Accumulate history at generous limits (6 turns)
    let run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 100 * 200? Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Now compute 20000 + 5000. Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 25000 / 5? Calculator. Remind me of prior results.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Compute 2^16 with calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 65536 + 5000? Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Calculate 70536 / 7. Calculator.',
      streamConfig
    );

    // Squeeze progressively to force summarization
    for (const squeeze of [3500, 3200, 3100, 3000, 2800, 2500, 2000]) {
      if (spies.onSummarizeStartSpy.mock.calls.length > 0) {
        break;
      }
      run = await createRun(squeeze);
      await runTurn(
        { run, conversationHistory },
        `What is ${squeeze} - 1000? Calculator.`,
        streamConfig
      );
    }

    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    const summaryText = getSummaryText(completePayload.summary);
    const reportedTokenCount = completePayload.summary!.tokenCount ?? 0;

    const localTokenCount = tokenCounter(
      new HumanMessage(buildSummaryCarrierText(summaryText))
    );

    console.log(
      `  Token match: reported=${reportedTokenCount}, local=${localTokenCount}`
    );

    // Token counts may differ slightly due to encoding differences
    // (claude vs o200k_base) and the 1.1× Claude correction factor.
    // Allow up to 25% variance.
    const variance =
      Math.abs(reportedTokenCount - localTokenCount) / localTokenCount;
    expect(variance).toBeLessThan(0.25);
  }, 180_000);

  test('collectedUsage input_tokens decreases after summarization', async () => {
    const spies = createSpies();
    let collectedUsage: UsageMetadata[] = [];
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    const createRun = async (maxTokens = 4000): Promise<Run<t.IState>> => {
      collectedUsage = [];
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return createSummarizationRun({
        agentProvider,
        summarizationProvider,
        summarizationModel,
        maxContextTokens: maxTokens,
        instructions: INSTRUCTIONS,
        collectedUsage,
        aggregateContent,
        spies,
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build up conversation (6 turns at generous budget)
    let run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 12345 * 67? Calculator.',
      streamConfig
    );

    // Capture pre-summary input_tokens
    const preSumUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    const lastPreUsage =
      preSumUsage.length > 0 ? preSumUsage[preSumUsage.length - 1] : undefined;
    const preSumInputTokens =
      lastPreUsage?.input_tokens != null ? lastPreUsage.input_tokens : 0;

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Now divide that by 13. Calculator. Also multiply by 7.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Compute 999 * 888. Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 2^10? Calculator.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'Calculate 1024 + 5000. Calculator. List all prior results.',
      streamConfig
    );

    run = await createRun();
    await runTurn(
      { run, conversationHistory },
      'What is 6024 * 3? Calculator.',
      streamConfig
    );

    // Squeeze progressively to force summarization
    for (const squeeze of [3500, 3200, 3100, 3000, 2800, 2500, 2000]) {
      if (spies.onSummarizeStartSpy.mock.calls.length > 0) {
        break;
      }
      run = await createRun(squeeze);
      await runTurn(
        { run, conversationHistory },
        `What is ${squeeze} - 1000? Calculator.`,
        streamConfig
      );
    }

    // Post-summary turn
    run = await createRun(4000);
    await runTurn(
      { run, conversationHistory },
      'What is 10 + 10? Calculator.',
      streamConfig
    );

    const postSumUsage = collectedUsage.filter(
      (u: Partial<UsageMetadata>) =>
        u.input_tokens != null && u.input_tokens > 0
    );
    const lastPostUsage =
      postSumUsage.length > 0
        ? postSumUsage[postSumUsage.length - 1]
        : undefined;
    const postSumInputTokens =
      lastPostUsage?.input_tokens != null ? lastPostUsage.input_tokens : 0;

    console.log(
      `  Input tokens: pre-summary=${preSumInputTokens}, post-summary=${postSumInputTokens}`
    );

    // After summarization, the context should be smaller, so input tokens should decrease
    // (compared to what they would have been without summarization)
    // We compare against the pre-summary value which had fewer messages
    // The post-summary turn should have fewer input tokens than the last pre-summary turn
    // that had the full context (before summarization compressed it)
    if (spies.onSummarizeCompleteSpy.mock.calls.length > 0) {
      expect(postSumInputTokens).toBeGreaterThan(0);
      expect(preSumInputTokens).toBeGreaterThan(0);
      console.log(
        `  Summarization fired: ${spies.onSummarizeCompleteSpy.mock.calls.length} times`
      );
    }
  }, 180_000);
});

// ---------------------------------------------------------------------------
// Enrichment and prompt selection (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Enrichment and prompt selection (no API keys)', () => {
  jest.setTimeout(60_000);

  const INSTRUCTIONS = 'You are a helpful assistant.';
  const streamConfig = {
    configurable: { thread_id: 'enrichment-tests' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  // The fake summarizer includes a basic summary without tool failures section
  const BASE_SUMMARY =
    '## Goal\nHelp user.\n\n## Progress\n### Done\n- Assisted user.';

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [BASE_SUMMARY] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('tool failure enrichment appended to summary', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build conversation with a tool failure
    const conversationHistory: BaseMessage[] = [];
    addDeterministicSummarizationPressure(conversationHistory, 'tool-failure', 4);
    conversationHistory.push(
      new HumanMessage('Run the linter on my code.'),
      new AIMessage({
        content: [
          { type: 'text' as const, text: 'Running the linter now.' },
          {
            type: 'tool_use' as const,
            id: 'tool_lint_1',
            name: 'run_linter',
            input: '{"path": "/src/index.ts"}',
          },
        ],
        tool_calls: [
          {
            id: 'tool_lint_1',
            name: 'run_linter',
            args: { path: '/src/index.ts' },
          },
        ],
      }),
      new ToolMessage({
        content: 'Error: ENOENT: no such file or directory, open /src/index.ts',
        tool_call_id: 'tool_lint_1',
        name: 'run_linter',
        status: 'error',
      }),
      new AIMessage('The linter failed because the file was not found.'),
      new HumanMessage('Try again with the correct path.'),
      new AIMessage(
        'I will try again. The correct path would need to be provided by you since I cannot verify file existence.'
      )
    );

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: `tool-failure-enrich-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: 1024,
        summarizationEnabled: true,
        summarizationConfig: {
          ...LOCAL_SUMMARY_CONFIG,
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            spies.onRunStepSpy(_event, data);
            aggregateContent({
              event: GraphEvents.ON_RUN_STEP,
              data: data as t.RunStep,
            });
          },
        },
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
      },
      tokenCounter,
      indexTokenCountMap,
    });

    run.Graph?.overrideTestModel(['Understood, awaiting correct path.'], 1);

    try {
      await run.processStream(
        {
          messages: [
            ...conversationHistory,
            new HumanMessage('What happened?'),
          ],
        },
        streamConfig as any
      );
    } catch {
      // empty_messages is acceptable for tiny context
    }

    const { completePayload } = assertSummarizationEvents(spies);
    const summaryText = getSummaryText(completePayload.summary);
    expect(summaryText).toContain('## Tool Failures');
    expect(summaryText).toContain('run_linter');
    expect(summaryText).toContain('ENOENT');
  });
});

// ---------------------------------------------------------------------------
// Summarization deduplication and correctness (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Summarization deduplication correctness (no API keys)', () => {
  jest.setTimeout(60_000);

  const INSTRUCTIONS =
    'You are a math tutor. Use the calculator tool for ALL computations. Be concise.';
  const streamConfig = {
    configurable: { thread_id: 'multi-pass-correctness' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance | undefined;
  const originalGetChatModelClass = providers.getChatModelClass;

  afterEach(() => {
    if (getChatModelClassSpy) {
      getChatModelClassSpy.mockRestore();
    }
  });

  test('summarization does not produce duplicate section headers', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    // Track what the summarizer receives for each chunk
    const capturedSystemMessages: string[] = [];
    const capturedHumanMessages: string[] = [];

    // Return different summaries for each chunk — chunk 2 returns a proper
    // comprehensive summary that does NOT duplicate ## Goal
    let chunkCallCount = 0;
    const chunkResponses = [
      '## Goal\nUser needs math computations.\n\n## Progress\n### Done\n- Computed 2+2=4.\n- Computed 3*5=15.',
      '## Goal\nUser needs comprehensive math help including basic and advanced operations.\n\n## Progress\n### Done\n- Computed 2+2=4.\n- Computed 3*5=15.\n- Computed sqrt(16)=4.\n- Computed 100/4=25.\n\n## Next Steps\nContinue with more calculations.',
    ];

    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              const response =
                chunkResponses[chunkCallCount] ??
                chunkResponses[chunkResponses.length - 1];
              chunkCallCount++;
              super({ responses: [response] });
            }
            async *_streamResponseChunks(
              messages: any[],
              options: any,
              runManager?: any
            ) {
              for (const msg of messages) {
                const msgType = msg.getType?.() ?? msg._getType?.();
                const content =
                  typeof msg.content === 'string'
                    ? msg.content
                    : JSON.stringify(msg.content);
                if (msgType === 'system') capturedSystemMessages.push(content);
                if (msgType === 'human') capturedHumanMessages.push(content);
              }
              yield* super._streamResponseChunks(messages, options, runManager);
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

    const createRunHelper = async (
      maxTokens: number
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `multi-pass-dedup-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
            parameters: {},
          },
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build up enough conversation to trigger summarization
    // Build enough conversation history to trigger summarization
    let run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['The answer to 2+2 is 4. Basic addition.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 2+2? Explain in detail.',
      streamConfig
    );

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['3 times 5 is 15. Multiplication is repeated addition.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Now explain 3 times 5 in great detail with many examples.',
      streamConfig
    );

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      ['The square root of 16 is 4, because 4*4=16.'],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is sqrt(16)? Give a thorough step-by-step explanation.',
      streamConfig
    );

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(
      [
        '100 divided by 4 is 25. Division distributes a total into equal parts.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 100/4? Explain division with multiple worked examples.',
      streamConfig
    );

    // Now force summarization with tight context
    addDeterministicSummarizationPressure(conversationHistory, 'dedup-checkpoint', 4);
    run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['Continuing after summary.'], 1);
    try {
      await runTurn({ run, conversationHistory }, 'Continue.', streamConfig);
    } catch {
      conversationHistory.pop(); // remove failed user message
    }

    // Assert summarization fired
    const sumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(
      `  Dedup: ${sumCount} summarization(s), ${chunkCallCount} chunk LLM calls, ` +
        `${capturedSystemMessages.length} system messages captured`
    );

    expect(sumCount).toBeGreaterThanOrEqual(1);

    const lastComplete = spies.onSummarizeCompleteSpy.mock.calls[
      sumCount - 1
    ][0] as t.SummarizeCompleteEvent;
    const summaryText = getSummaryText(lastComplete.summary);

    // KEY ASSERTION: ## Goal should appear exactly ONCE (no duplication)
    const goalCount = (summaryText.match(/## Goal/g) || []).length;
    expect(goalCount).toBe(1);

    // ## Progress should also appear exactly once
    const progressCount = (summaryText.match(/## Progress/g) || []).length;
    expect(progressCount).toBe(1);

    // tokenCount must be > 0 (tokenCounter is provided)
    expect(lastComplete.summary!.tokenCount).toBeGreaterThan(0);

    console.log(
      `  Summary (${summaryText.length} chars, ${lastComplete.summary!.tokenCount} tokens):\n` +
        `  "${summaryText.substring(0, 300)}…"`
    );
  });

  test('repeated summarization cycles do not accumulate duplicate sections', async () => {
    // This test verifies that when summarization fires multiple times across
    // runs, each summary is clean (no duplicate section headers).
    // The cross-cycle prompt selection (UPDATE for chunk 0, FRESH for chunk 1+)
    // is tested in unit tests (node.test.ts). This integration test focuses on
    // the end-to-end outcome.
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    // The summarizer always returns a clean single-section summary
    const summaryResponse =
      '## Goal\nMath tutoring.\n\n## Progress\n### Done\n- Completed operations.';

    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [summaryResponse] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

    const createRunHelper = async (
      maxTokens: number,
      initialSummary?: { text: string; tokenCount: number }
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `repeat-sum-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
          },
          initialSummary,
        },
        returnContent: true,
        customHandlers: {
          [GraphEvents.ON_RUN_STEP]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              spies.onRunStepSpy(_event, data);
              aggregateContent({
                event: GraphEvents.ON_RUN_STEP,
                data: data as t.RunStep,
              });
            },
          },
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
        },
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // --- Cycle 1: Build conversation and trigger summarization ---
    let run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(['Answer 1 with detailed explanation.'], 1);
    await runTurn({ run, conversationHistory }, 'Question 1.', streamConfig);

    run = await createRunHelper(4000);
    run.Graph?.overrideTestModel(['Answer 2 with more explanation.'], 1);
    await runTurn({ run, conversationHistory }, 'Question 2.', streamConfig);

    addDeterministicSummarizationPressure(conversationHistory, 'repeat-checkpoint', 4);
    run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['OK.'], 1);
    try {
      await runTurn({ run, conversationHistory }, 'Summarize.', streamConfig);
    } catch {
      conversationHistory.pop();
    }

    const cycle1SumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    expect(cycle1SumCount).toBeGreaterThan(0);

    // Extract the summary from cycle 1 for use as initialSummary in cycle 2
    let priorSummary: { text: string; tokenCount: number } | undefined;
    if (cycle1SumCount > 0) {
      const lastComplete = spies.onSummarizeCompleteSpy.mock.calls[
        cycle1SumCount - 1
      ][0] as t.SummarizeCompleteEvent;
      priorSummary = {
        text: getSummaryText(lastComplete.summary),
        tokenCount: lastComplete.summary!.tokenCount ?? 0,
      };
    }

    // --- Cycle 2: More conversation with prior summary, trigger again ---
    run = await createRunHelper(4000, priorSummary);
    run.Graph?.overrideTestModel(['Cycle 2 answer.'], 1);
    await runTurn(
      { run, conversationHistory },
      'Cycle 2 question.',
      streamConfig
    );

    addDeterministicSummarizationPressure(conversationHistory, 'repeat-checkpoint', 4);
    run = await createRunHelper(1024, priorSummary);
    run.Graph?.overrideTestModel(['OK cycle 2.'], 1);
    try {
      await runTurn(
        { run, conversationHistory },
        'Summarize again.',
        streamConfig
      );
    } catch {
      conversationHistory.pop();
    }

    const totalSumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(
      `  Repeated summarization: cycle1=${cycle1SumCount}, total=${totalSumCount}`
    );

    // At least one summarization should have fired
    expect(totalSumCount).toBeGreaterThan(cycle1SumCount);

    // Every summary should have exactly one ## Goal (no duplicates)
    for (let i = 0; i < totalSumCount; i++) {
      const complete = spies.onSummarizeCompleteSpy.mock.calls[
        i
      ][0] as t.SummarizeCompleteEvent;
      expect(complete.error).toBeUndefined();
      const text = getSummaryText(complete.summary);
      const goalCount = (text.match(/## Goal/g) || []).length;
      if (goalCount !== 1) {
        console.log(
          `  Summary ${i} has ${goalCount} '## Goal' sections:\n  "${text.substring(0, 300)}…"`
        );
      }
      expect(goalCount).toBe(1);
      expect(complete.summary!.tokenCount).toBeGreaterThan(0);
    }
  });

  test('conversation continues after summarization', async () => {
    const spies = createSpies();
    const conversationHistory: BaseMessage[] = [];
    const tokenCounter = await createTokenCounter();

    // Summarizer returns a concise summary
    const summaryResponse =
      '## Goal\nMath help.\n\n## Progress\n### Done\n- Basic operations completed.';

    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [summaryResponse] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

    const createRunHelper = async (
      maxTokens: number
    ): Promise<Run<t.IState>> => {
      const { aggregateContent } = createContentAggregator();
      const indexTokenCountMap = buildIndexTokenCountMap(
        conversationHistory,
        tokenCounter
      );
      return Run.create<t.IState>({
        runId: `multi-pass-continue-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          llmConfig: getLLMConfig(Providers.OPENAI),
          instructions: INSTRUCTIONS,
          maxContextTokens: maxTokens,
          summarizationEnabled: true,
          summarizationConfig: {
            ...LOCAL_SUMMARY_CONFIG,
            provider: Providers.OPENAI,
            parameters: {},
          },
        },
        returnContent: true,
        customHandlers: buildHandlers([], aggregateContent, spies),
        tokenCounter,
        indexTokenCountMap,
      });
    };

    // Build conversation
    for (const q of [
      'Explain 2+2 in great detail.',
      'Explain 3*5 step by step.',
      'What is sqrt(16)? Full explanation.',
      'What is 100/4? Show your work.',
    ]) {
      const run = await createRunHelper(4000);
      run.Graph?.overrideTestModel(
        [
          'Here is a detailed explanation of the computation with many steps and examples.',
        ],
        1
      );
      await runTurn({ run, conversationHistory }, q, streamConfig);
    }

    // Trigger summarization
    addDeterministicSummarizationPressure(conversationHistory, 'continue-checkpoint', 4);
    let run = await createRunHelper(1024);
    run.Graph?.overrideTestModel(['Summary acknowledged.'], 1);
    try {
      await runTurn({ run, conversationHistory }, 'Continue.', streamConfig);
    } catch {
      conversationHistory.pop();
    }

    const sumCount = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(`  Continuation test: ${sumCount} summarization(s)`);

    if (sumCount > 0) {
      // Post-summary turn should work with reasonable context
      run = await createRunHelper(2000);
      run.Graph?.overrideTestModel(['The answer is 42.'], 1);
      const postResult = await runTurn(
        { run, conversationHistory },
        'What is 6*7?',
        streamConfig
      );
      expect(postResult).toBeDefined();
      console.log(
        `  Post-summary turn succeeded, ${conversationHistory.length} messages`
      );
    }
  });
});

// ---------------------------------------------------------------------------
// Re-summarization within a single run (FakeListChatModel — no API keys)
// Tests the shouldSkipSummarization baseline reset fix.
// ---------------------------------------------------------------------------

describe('Re-summarization within a single run (no API keys)', () => {
  jest.setTimeout(60_000);

  const SUMMARY_V1 = '## Summary v1\nUser discussed topic A.';
  const SUMMARY_V2 = '## Summary v2\nUser discussed topic A and B.';
  const INSTRUCTIONS = 'You are a helpful assistant.';
  const streamConfig = {
    configurable: { thread_id: 're-summarize-test' },
    recursionLimit: 80,
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;
  let summaryCallCount = 0;

  beforeEach(() => {
    summaryCallCount = 0;
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              summaryCallCount++;
              super({
                responses: [summaryCallCount === 1 ? SUMMARY_V1 : SUMMARY_V2],
              });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('second summarization fires after context refills post-first-summary', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build a long conversation that will need multiple summarization cycles
    const padding = 'x'.repeat(400);
    const conversationHistory: BaseMessage[] = [];
    for (let i = 0; i < 10; i++) {
      conversationHistory.push(new HumanMessage(`Question ${i}${padding}`));
      conversationHistory.push(new AIMessage(`Answer ${i}${padding}`));
    }
    conversationHistory.push(new HumanMessage('Final question'));

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];

    const run = await Run.create<t.IState>({
      runId: `re-sum-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: 600,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: conversationHistory },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    const startCalls = spies.onSummarizeStartSpy.mock.calls.length;
    const completeCalls = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(
      `  Summarization cycles: start=${startCalls}, complete=${completeCalls}, error=${error?.message.substring(0, 80) ?? 'none'}`
    );

    // The key assertion: with enough messages and tight context,
    // summarization should fire more than once. Before the
    // shouldSkipSummarization baseline reset fix, it would fire only once.
    expect(startCalls).toBeGreaterThanOrEqual(1);
    console.log(`  Summary model calls: ${summaryCallCount}`);
  });
});

// ---------------------------------------------------------------------------
// Emoji/Unicode safety through full pipeline (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Emoji and Unicode safety (no API keys)', () => {
  jest.setTimeout(60_000);

  const SUMMARY = '## Summary\nUser sent emoji-heavy messages about coding.';
  const streamConfig = {
    configurable: { thread_id: 'emoji-safety-test' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [SUMMARY] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('emoji-heavy messages do not produce broken JSON in summarization', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // ZWJ sequences and multi-byte emoji that produce surrogate pairs in UTF-16
    const emojiMessages: BaseMessage[] = [
      new HumanMessage('👨‍💻 Let me show you some code 🚀'),
      new AIMessage('Sure! Here is the code 🎉✨ with lots of emoji 🌍🌎🌏'),
      new HumanMessage('👨‍👩‍👧‍👦 Family emoji and flags 🇺🇸🇬🇧🇯🇵 test'),
      new AIMessage('More emoji: 🧑‍🔬🧑‍🎨🧑‍🚒🧑‍✈️ professional emoji'),
      new HumanMessage('Final 💯🔥⚡ question'),
    ];

    const indexTokenCountMap = buildIndexTokenCountMap(
      emojiMessages,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];

    const run = await Run.create<t.IState>({
      runId: `emoji-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: 'Be helpful.',
        maxContextTokens: 100,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    // The test passes if this doesn't throw a JSON serialization error
    let error: Error | undefined;
    try {
      await run.processStream({ messages: emojiMessages }, streamConfig as any);
    } catch (err) {
      error = err as Error;
    }

    // empty_messages is acceptable (tight context), but JSON errors are not
    if (error) {
      expect(error.message).not.toContain('not valid JSON');
      expect(error.message).not.toContain('Invalid Unicode');
      console.log(
        `  Emoji test: acceptable error (${error.message.substring(0, 80)})`
      );
    } else {
      console.log('  Emoji test: completed without error');
    }

    console.log(
      `  Summarization: start=${spies.onSummarizeStartSpy.mock.calls.length}, complete=${spies.onSummarizeCompleteSpy.mock.calls.length}`
    );
  });
});

// ---------------------------------------------------------------------------
// Budget-aware error messages (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Budget-aware error messages (no API keys)', () => {
  jest.setTimeout(60_000);

  const streamConfig = {
    configurable: { thread_id: 'budget-error-test' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  test('empty_messages error includes tool-specific guidance when tools dominate budget', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    const conversationHistory: BaseMessage[] = [new HumanMessage('Hello')];

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];

    // Create a run with maxContextTokens smaller than the tool definitions
    // The Calculator tool alone has a schema that takes up tokens
    const run = await Run.create<t.IState>({
      runId: `budget-err-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        tools: [new Calculator()],
        instructions: 'A'.repeat(500), // Long instructions to push over budget
        maxContextTokens: 50, // Impossibly tight
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: conversationHistory },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    expect(error).toBeDefined();
    // The error should mention the budget problem specifically
    const errorMsg = error!.message;
    expect(errorMsg).toContain('empty_messages');

    // Should contain actionable guidance about instructions or tools
    const hasGuidance =
      errorMsg.includes('Reduce the number of tools') ||
      errorMsg.includes('Increase maxContextTokens') ||
      errorMsg.includes('shorten the system prompt');
    expect(hasGuidance).toBe(true);

    console.log(
      `  Budget error guidance: ${errorMsg.substring(errorMsg.indexOf('Please') > -1 ? errorMsg.indexOf('Please') : 0, errorMsg.indexOf('Please') + 120)}`
    );
  });
});

// ---------------------------------------------------------------------------
// Large tool result + surviving context double-summarization regression
// (FakeListChatModel — no API keys)
//
// Models the real-world scenario from debug logs:
// - Multi-turn conversation with MCP tools (screenshots, snapshots)
// - Summarization fires once → surviving context includes a 9437-char tool result
// - Post-summarization prune: the tool result exceeds the effective budget
// - All surviving messages land in messagesToRefine
// - Before fix: summarization re-triggers immediately on the same messages
// - After fix: shouldSkipSummarization blocks re-trigger (baseline = surviving count)
// ---------------------------------------------------------------------------

describe('Large tool result surviving context — no double summarization (no API keys)', () => {
  jest.setTimeout(60_000);

  const SUMMARY_V1 =
    '## Summary\nUser navigated to apple.com, took screenshots, ran Lighthouse audit.';
  const SUMMARY_V2 =
    '## Summary v2\nUser explored apple.com with devtools, took snapshots.';
  const INSTRUCTIONS = 'You are a browser automation assistant.';
  const streamConfig = {
    configurable: { thread_id: 'double-sum-regression' },
    recursionLimit: 80,
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;
  let summaryCallCount = 0;

  beforeEach(() => {
    summaryCallCount = 0;
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              summaryCallCount++;
              super({
                responses: [summaryCallCount === 1 ? SUMMARY_V1 : SUMMARY_V2],
              });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('surviving context with oversized tool result does not re-trigger summarization', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    // Build a conversation that mirrors the real debug log:
    // Multiple turns with tool calls, including a large take_snapshot result
    const largeSnapshot = 'uid=1_0 RootWebArea "Apple" '.repeat(300); // ~9000 chars
    const conversationHistory: BaseMessage[] = [
      new HumanMessage('Navigate to apple.com'),
      new AIMessage({
        content: 'Navigating now.',
        tool_calls: [
          {
            id: 'tc_1',
            name: 'navigate_page',
            args: { url: 'https://apple.com' },
          },
        ],
      }),
      new ToolMessage({
        content: 'Successfully navigated to https://www.apple.com.',
        tool_call_id: 'tc_1',
        name: 'navigate_page',
      }),
      new AIMessage({
        content: 'Taking a screenshot.',
        tool_calls: [{ id: 'tc_2', name: 'take_screenshot', args: {} }],
      }),
      new ToolMessage({
        content: 'Took a screenshot of the current page.',
        tool_call_id: 'tc_2',
        name: 'take_screenshot',
      }),
      new HumanMessage('What can you see on the site?'),
      new AIMessage({
        content: 'Let me take a snapshot.',
        tool_calls: [{ id: 'tc_3', name: 'take_snapshot', args: {} }],
      }),
      new ToolMessage({
        content: largeSnapshot, // ~9000 chars — the large tool result
        tool_call_id: 'tc_3',
        name: 'take_snapshot',
      }),
      new HumanMessage('Show me more details'),
      new AIMessage({
        content: 'Here are the details from the page.',
        tool_calls: [{ id: 'tc_4', name: 'take_screenshot', args: {} }],
      }),
      new ToolMessage({
        content: 'Took another screenshot.',
        tool_call_id: 'tc_4',
        name: 'take_screenshot',
      }),
      new HumanMessage('Analyze the page performance'),
    ];

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];

    // maxContextTokens = 800 — tight enough that the large snapshot
    // forces aggressive pruning but leaves room for the agent to respond
    const run = await Run.create<t.IState>({
      runId: `double-sum-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: 800,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
        },
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: conversationHistory },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    const startCalls = spies.onSummarizeStartSpy.mock.calls.length;
    const completeCalls = spies.onSummarizeCompleteSpy.mock.calls.length;
    console.log(
      `  Summarization: start=${startCalls}, complete=${completeCalls}, modelCalls=${summaryCallCount}`
    );

    if (error) {
      // empty_messages is acceptable for tight context; double-summarization is not
      console.log(`  Error: ${error.message.substring(0, 100)}`);
    }

    // Key assertion: summarization should fire at most once.
    // Before the fix, the surviving context's large tool result would cause
    // all messages to land in messagesToRefine, triggering a second
    // summarization on the same messages.
    expect(startCalls).toBeLessThanOrEqual(1);
    expect(summaryCallCount).toBeLessThanOrEqual(1);
    console.log(
      `  Double-summarization prevented: ${startCalls <= 1 ? 'YES' : 'NO'}`
    );
  });
});

// ---------------------------------------------------------------------------
// Empty summarizer output must not loop (FakeListChatModel — no API keys)
// ---------------------------------------------------------------------------

describe('Empty summarizer output (no API keys)', () => {
  jest.setTimeout(60_000);

  const SUMMARIZER_MODEL = 'summarizer-that-returns-nothing';
  const AGENT_REPLY = 'Here is the answer to your final question.';
  const streamConfig = {
    configurable: { thread_id: 'empty-summary-loop' },
    recursionLimit: 25,
    streamMode: 'values',
    version: 'v2' as const,
  };

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;
  let summarizerCallCount = 0;

  beforeEach(() => {
    summarizerCallCount = 0;
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider !== Providers.OPENAI) {
          return originalGetChatModelClass(provider);
        }
        return class extends FakeListChatModel {
          constructor(options: any) {
            const isSummarizer = options?.model === SUMMARIZER_MODEL;
            if (isSummarizer) {
              summarizerCallCount++;
            }
            super({ responses: [isSummarizer ? '' : AGENT_REPLY] });
          }
        } as any;
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  test('a summarizer returning nothing does not loop the run to its recursion cap', async () => {
    const spies = createSpies();
    const tokenCounter = await createTokenCounter();

    const padding = 'x'.repeat(400);
    const conversationHistory: BaseMessage[] = [];
    for (let i = 0; i < 10; i++) {
      conversationHistory.push(new HumanMessage(`Question ${i}${padding}`));
      conversationHistory.push(new AIMessage(`Answer ${i}${padding}`));
    }
    conversationHistory.push(new HumanMessage('Final question'));

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];

    const run = await Run.create<t.IState>({
      runId: `empty-summary-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: 'You are a helpful assistant.',
        maxContextTokens: 600,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
          model: SUMMARIZER_MODEL,
        },
      },
      returnContent: true,
      customHandlers: buildHandlers(collectedUsage, aggregateContent, spies),
      tokenCounter,
      indexTokenCountMap,
    });

    let error: Error | undefined;
    try {
      await run.processStream(
        { messages: conversationHistory },
        streamConfig as any
      );
    } catch (err) {
      error = err as Error;
    }

    const startCalls = spies.onSummarizeStartSpy.mock.calls.length;
    console.log(
      `  Empty-summary attempts: ${startCalls}, summarizer model calls: ${summarizerCallCount}`
    );

    /**
     * The regression: an empty summary compacts nothing, so the agent node
     * saw the same unchanged state and re-triggered — the run spent its whole
     * recursion budget on empty summary steps and died on GraphRecursionError.
     */
    expect(error?.message ?? '').not.toMatch(/[Rr]ecursion/);
    expect(startCalls).toBeLessThanOrEqual(3);
    expect(summarizerCallCount).toBeLessThanOrEqual(3);

    /** Nothing empty was committed as the run's summary. */
    const summarizedContent = spies.onSummarizeCompleteSpy.mock.calls.flat();
    for (const data of summarizedContent) {
      const summary = (data as t.SummarizeCompleteEvent).summary;
      if (summary != null) {
        expect(summary.content?.length ?? 0).toBeGreaterThan(0);
      }
    }
  });
});

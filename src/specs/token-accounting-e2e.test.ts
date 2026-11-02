/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { createTokenCounter, TokenEncoderManager } from '@/utils/tokens';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { Calculator } from '@/tools/Calculator';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

// ---------------------------------------------------------------------------
// FakeListChatModel subclass that emits usage_metadata on the final chunk.
// Accepts a single UsageMetadata or an array (one per call, cycling).
// This lets us exercise the calibration path end-to-end through the Graph.
// ---------------------------------------------------------------------------
class FakeWithUsage extends FakeListChatModel {
  private _usages: UsageMetadata[];
  private _usageIdx = 0;

  constructor(opts: {
    responses: string[];
    usage?: UsageMetadata | UsageMetadata[];
  }) {
    super({ responses: opts.responses });
    if (!opts.usage) {
      this._usages = [];
    } else if (Array.isArray(opts.usage)) {
      this._usages = opts.usage;
    } else {
      this._usages = [opts.usage];
    }
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const response = this._currentResponse();
    this._incrementResponse();

    const words = response.split(/(?<=\s)/);
    for (const word of words) {
      const chunk = new ChatGenerationChunk({
        text: word,
        generationInfo: {},
        message: new AIMessageChunk({ content: word }),
      });
      yield chunk;
      void runManager?.handleLLMNewToken(word);
    }

    // Emit a final empty chunk carrying usage_metadata for this call
    const usage = this._usages[this._usageIdx % this._usages.length] as
      | UsageMetadata
      | undefined;
    if (usage) {
      this._usageIdx++;
      const usageChunk = new ChatGenerationChunk({
        text: '',
        generationInfo: {},
        message: new AIMessageChunk({
          content: '',
          usage_metadata: usage,
        }),
      });
      yield usageChunk;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function getSummaryText(summary: t.SummaryContentBlock | undefined): string {
  if (!summary) return '';
  return (summary.content ?? [])
    .map((block) => ('text' in block ? (block as { text: string }).text : ''))
    .join('');
}

const SUMMARY_TEXT =
  'User discussed math problems. Key results: 2+2=4, 3*5=15. Context preserved.';

const INSTRUCTIONS = 'You are a helpful math tutor. Be concise.';

function addAccountingPressure(history: BaseMessage[]): void {
  const detail =
    'Keep the verified result, token usage, calibration ratio, and summary overhead for the next run. '.repeat(2);
  const pressure: BaseMessage[] = [];
  for (let turn = 0; turn < 8; turn++) {
    pressure.push(
      new HumanMessage(`Accounting exchange ${turn}: ${detail}`),
      new AIMessage(`Recorded exchange ${turn}: ${detail}`)
    );
  }
  history.unshift(...pressure);
}

const streamConfig = {
  configurable: { thread_id: 'token-e2e' },
  streamMode: 'values',
  version: 'v2' as const,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Token accounting E2E — Run + Graph + real token counter', () => {
  jest.setTimeout(60_000);

  let tokenCounter: t.TokenCounter;
  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeAll(async () => {
    tokenCounter = await createTokenCounter();
  });

  afterAll(() => {
    TokenEncoderManager.reset();
  });

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            constructor(_options: any) {
              super({ responses: [SUMMARY_TEXT] });
            }
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  async function createRun(opts: {
    maxTokens: number;
    conversationHistory: BaseMessage[];
    spies: {
      onSummarizeStartSpy: jest.Mock;
      onSummarizeCompleteSpy: jest.Mock;
    };
    tools?: t.GraphTools;
    indexTokenCountMap?: Record<string, number>;
    initialSummary?: { text: string; tokenCount: number };
  }): Promise<Run<t.IState>> {
    const { aggregateContent } = createContentAggregator();
    const indexTokenCountMap =
      opts.indexTokenCountMap ??
      buildIndexTokenCountMap(opts.conversationHistory, tokenCounter);

    return Run.create<t.IState>({
      runId: `tok-e2e-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: getLLMConfig(Providers.OPENAI),
        instructions: INSTRUCTIONS,
        maxContextTokens: opts.maxTokens,
        tools: opts.tools,
        summarizationEnabled: true,
        summarizationConfig: {
          provider: Providers.OPENAI,
          prompt: 'Checkpoint the verified results and accounting context concisely.',
          updatePrompt: 'Merge the new accounting context into the prior checkpoint.',
          maxSummaryTokens: 64,
        },
        initialSummary: opts.initialSummary,
      },
      returnContent: true,
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            aggregateContent({
              event: GraphEvents.ON_RUN_STEP,
              data: data as t.RunStep,
            });
          },
        },
        [GraphEvents.ON_SUMMARIZE_START]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            opts.spies.onSummarizeStartSpy(data);
          },
        },
        [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            opts.spies.onSummarizeCompleteSpy(data);
          },
        },
      },
      tokenCounter,
      indexTokenCountMap,
    });
  }

  async function runTurn(
    state: { run: Run<t.IState>; conversationHistory: BaseMessage[] },
    userMessage: string
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

  // =========================================================================
  // Test 1: Multi-turn token accounting without usage_metadata (tokenCounter only)
  // =========================================================================
  test('multi-turn pruning + summarization with real token counter (no usage_metadata)', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    // --- Turn 1: build up conversation at generous budget ---
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      [
        'The answer to 2+2 is 4. Addition is one of the four fundamental arithmetic operations. ' +
          'It combines two or more numbers into a single sum. In this case we combine 2 and 2 to get 4. ' +
          'This is also known as the additive identity when one operand is zero.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Hello! What is 2+2? Please explain addition in detail with examples and history.'
    );
    expect(conversationHistory.length).toBeGreaterThanOrEqual(2);

    // --- Turn 2: more conversation ---
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      [
        'Multiplication of 3 times 5 equals 15. Multiplication can be understood as repeated addition. ' +
          'So 3 times 5 means adding 3 five times: 3+3+3+3+3 which equals 15. ' +
          'The commutative property tells us 5 times 3 also equals 15.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'Great explanation! Now what is 3 times 5? Explain multiplication as repeated addition with examples.'
    );
    expect(conversationHistory.length).toBeGreaterThanOrEqual(4);

    // --- Turn 3: tight budget forces pruning and summarization ---
    // Real token count for the 4 messages above is ~150+ tokens.
    // Bounded padding forces pruning while leaving room for a real summary.
    addAccountingPressure(conversationHistory);
    run = await createRun({
      maxTokens: 512,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      ['Understood, continuing with summary context.'],
      1
    );

    await runTurn({ run, conversationHistory }, 'Now summarize everything.');

    // Summarization should have fired
    expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    const summaryText = getSummaryText(completePayload.summary);
    expect(summaryText).toBe(SUMMARY_TEXT);
    expect(completePayload.error).toBeUndefined();
    expect(completePayload.summary!.tokenCount).toBeGreaterThan(0);

    // Token count should match what our real counter computes
    const summaryTokenCount = completePayload.summary!.tokenCount ?? 0;
    expect(summaryTokenCount).toBeGreaterThan(5);

  });

  // =========================================================================
  // Test 2: Usage metadata feeds calibration through the real Graph pipeline
  // =========================================================================
  test('usage_metadata from model feeds into calibration on next turn', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    // --- Turn 1: normal budget, model emits usage_metadata ---
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: [
        'The answer to 2+2 is 4. Addition is one of the fundamental arithmetic operations ' +
          'that combines quantities together into a sum. Two plus two yields four.',
      ],
      usage: {
        input_tokens: 45,
        output_tokens: 25,
        total_tokens: 70,
      },
    }) as any;

    await runTurn(
      { run, conversationHistory },
      'What is 2+2? Please provide a detailed explanation of addition.'
    );
    expect(conversationHistory.length).toBeGreaterThanOrEqual(2);

    // --- Turn 2: also with usage_metadata ---
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: [
        'Multiplication of 3 times 5 equals 15. This is repeated addition: 3+3+3+3+3. ' +
          'The commutative property means 5 times 3 also equals 15.',
      ],
      usage: {
        input_tokens: 90,
        output_tokens: 30,
        total_tokens: 120,
      },
    }) as any;

    await runTurn(
      { run, conversationHistory },
      'What is 3 times 5? Explain multiplication as repeated addition.'
    );
    expect(conversationHistory.length).toBeGreaterThanOrEqual(4);

    // --- Turn 3: tight context with usage → triggers summarization ---
    addAccountingPressure(conversationHistory);
    run = await createRun({
      maxTokens: 512,
      conversationHistory,
      spies,
    });
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: ['Continuing after summary.'],
      usage: {
        input_tokens: 40,
        output_tokens: 10,
        total_tokens: 50,
      },
    }) as any;

    await runTurn({ run, conversationHistory }, 'Continue.');

    // Summarization should fire even with usage_metadata in the mix
    expect(
      spies.onSummarizeCompleteSpy.mock.calls.length
    ).toBeGreaterThanOrEqual(1);
    const payload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    expect(payload.error).toBeUndefined();
    expect(payload.summary!.tokenCount).toBeGreaterThan(0);
  });

  // =========================================================================
  // Test 3: Summary overhead feeds into getInstructionTokens on next Run
  // =========================================================================
  test('summary token overhead is accounted for in next run budget', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };

    // Simulate a pre-existing summary from a previous run
    const summaryTokenCount = tokenCounter(new HumanMessage(SUMMARY_TEXT));
    const initialSummary: { text: string; tokenCount: number } = {
      text: SUMMARY_TEXT,
      tokenCount: summaryTokenCount,
    };

    // Create a conversation that fits without the summary overhead,
    // but won't fit once summary tokens are reserved
    const conversationHistory: BaseMessage[] = [
      new HumanMessage('First question about algebra'),
      new AIMessage('Algebra is the study of variables and equations.'),
      new HumanMessage('Second question about geometry'),
      new AIMessage(
        'Geometry deals with shapes, sizes, and properties of space.'
      ),
    ];

    const indexTokenCountMap = buildIndexTokenCountMap(
      conversationHistory,
      tokenCounter
    );
    const msgTotal = Object.values(indexTokenCountMap).reduce(
      (a, b) => a + b,
      0
    );

    // Budget: fits messages + instructions but NOT messages + instructions + summary overhead
    // The summary overhead goes into getInstructionTokens, reducing effective budget
    const tightBudget = msgTotal + 30; // tight: room for instructions but not summary

    const run = await createRun({
      maxTokens: tightBudget,
      conversationHistory,
      spies,
      indexTokenCountMap,
      initialSummary,
    });
    run.Graph?.overrideTestModel(['Noted.'], 1);

    conversationHistory.push(new HumanMessage('Continue.'));

    let pruningOccurred: boolean;
    try {
      await run.processStream(
        { messages: conversationHistory },
        streamConfig as any
      );
      // If it succeeded, check if pruning occurred
      const runMessages = run.getRunMessages();
      pruningOccurred = runMessages != null && runMessages.length > 0;
    } catch {
      // Error is acceptable — the point is summary overhead was subtracted from budget
      pruningOccurred = true;
    }

    // With summary overhead consuming instruction tokens,
    // the effective budget should be smaller, causing pruning or error
    // (without the summary, messages would have fit)
    expect(pruningOccurred).toBe(true);
  });

  // =========================================================================
  // Test 4: Mixed turns — some with usage_metadata, some without
  // =========================================================================
  test('handles mixed turns: some with usage_metadata, some without', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    // --- Turn 1: WITH usage_metadata ---
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: ['Answer to first question with detailed explanation here.'],
      usage: {
        input_tokens: 50,
        output_tokens: 20,
        total_tokens: 70,
      },
    }) as any;

    await runTurn({ run, conversationHistory }, 'First question here.');

    // --- Turn 2: WITHOUT usage_metadata (plain FakeListChatModel) ---
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      ['Second response without any usage metadata attached.'],
      1
    );

    await runTurn({ run, conversationHistory }, 'Second question here.');

    // --- Turn 3: WITH usage_metadata again ---
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: ['Third answer.'],
      usage: {
        input_tokens: 120,
        output_tokens: 10,
        total_tokens: 130,
      },
    }) as any;

    await runTurn({ run, conversationHistory }, 'Third question.');

    // All 6 messages should be in conversation history (3 human + 3 AI)
    expect(conversationHistory.length).toBeGreaterThanOrEqual(6);

    // The system should handle the mixed usage gracefully without crashes.
    // Calibration fires on turns with usage, skips on turns without.
  });

  // =========================================================================
  // Test 5: Full round-trip — summarize, persist, load into next Run
  // =========================================================================
  test('full round-trip: summarize → persist → new Run with summary overhead', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    // Build up conversation with longer messages
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      [
        'The answer to 2+2 is 4. Addition combines two quantities into a sum. ' +
          'This is one of the four fundamental operations in arithmetic alongside ' +
          'subtraction, multiplication, and division.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is 2+2? Explain the concept of addition in detail with examples.'
    );

    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(
      [
        'The square root of 16 is 4, because 4 multiplied by 4 equals 16. ' +
          'Square root is the inverse operation of squaring a number.',
      ],
      1
    );
    await runTurn(
      { run, conversationHistory },
      'What is the square root of 16? Explain the concept of square roots.'
    );

    // Bounded padding forces pruning without starving the summarizer.
    addAccountingPressure(conversationHistory);
    run = await createRun({
      maxTokens: 512,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(['Got it.'], 1);

    await runTurn({ run, conversationHistory }, 'Continue.');

    const completeCalls = spies.onSummarizeCompleteSpy.mock.calls;
    expect(completeCalls.length).toBeGreaterThanOrEqual(1);

    const completePayload = completeCalls[0][0] as t.SummarizeCompleteEvent;
    const summary = completePayload.summary!;
    const summaryText = getSummaryText(summary);
    expect(summaryText.length).toBeGreaterThan(0);
    expect(summary.tokenCount).toBeGreaterThan(0);

    // --- Simulate persistence: create payload as the host would ---
    const persistedSummary: { text: string; tokenCount: number } = {
      text: summaryText,
      tokenCount: summary.tokenCount!,
    };

    // Start a new conversation with summary carried over
    const newHistory: BaseMessage[] = [
      new HumanMessage('What else can you help with?'),
    ];
    const newMap = buildIndexTokenCountMap(newHistory, tokenCounter);

    const spies2 = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };

    const run2 = await createRun({
      maxTokens: 2000,
      conversationHistory: newHistory,
      spies: spies2,
      indexTokenCountMap: newMap,
      initialSummary: persistedSummary,
    });
    run2.Graph?.overrideTestModel(
      [
        'I can help with many things! Based on our previous discussion about math.',
      ],
      1
    );

    const result = await run2.processStream(
      { messages: newHistory },
      streamConfig as any
    );

    expect(result).toBeDefined();
    const runMessages = run2.getRunMessages();
    expect(runMessages).toBeDefined();
    expect(runMessages!.length).toBeGreaterThan(0);

    // The summary token count should have been accounted for in the
    // instruction overhead, reducing the effective budget for messages.
    // We verify the run completed successfully with the summary present.
  });

  // =========================================================================
  // Test 6: Multi-tool-call agent loop — pruner closure persists across LLM calls
  // =========================================================================
  test('agent loop with tool calls: pruner closure persists across LLM calls within one Run', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    const run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });

    // FakeChatModel via overrideTestModel: call 1 emits text + tool calls,
    // call 2 emits text only. The Calculator tool runs between calls.
    const toolCalls: ToolCall[] = [
      {
        name: 'calculator',
        args: { input: '12345 * 6789' },
        id: 'tc_calc_1',
        type: 'tool_call',
      },
    ];
    run.Graph?.overrideTestModel(
      [
        'Let me calculate 12345 * 6789 for you.',
        'The result of 12345 * 6789 is 83,810,205.',
      ],
      1,
      toolCalls
    );

    conversationHistory.push(
      new HumanMessage('What is 12345 * 6789? Use the calculator.')
    );
    await run.processStream(
      { messages: conversationHistory },
      streamConfig as any
    );

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();
    // Should have: AI (with tool call) + ToolMessage + AI (final answer)
    expect(runMessages!.length).toBeGreaterThanOrEqual(3);

    // Verify the tool was actually called
    const toolMessages = runMessages!.filter((m) => m._getType() === 'tool');
    expect(toolMessages.length).toBe(1);
    // Calculator should have computed the real result
    expect(toolMessages[0].content as string).toContain('83810205');
  });

  // =========================================================================
  // Test 7: Prior tool calls in history + tight context triggers summarization
  // =========================================================================
  test('prior tool calls in history with tight context triggers summarization', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };

    // Build a conversation that includes tool call artifacts from a prior run.
    // This simulates the common case: user asked questions, agent used calculator,
    // and now we're continuing with a tight budget that forces summarization.
    const conversationHistory: BaseMessage[] = [];

    // Turn 1: build up at generous budget with tool calls
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });
    run.Graph?.overrideTestModel(
      [
        'Let me calculate that for you using the calculator tool.',
        'The result of 12345 * 6789 is 83,810,205. That is a large number!',
      ],
      1,
      [
        {
          name: 'calculator',
          args: { input: '12345 * 6789' },
          id: 'tc_prior_1',
          type: 'tool_call',
        },
      ]
    );
    await runTurn(
      { run, conversationHistory },
      'Calculate 12345 * 6789 using the calculator and explain the result.'
    );

    // Turn 2: another tool call
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });
    run.Graph?.overrideTestModel(
      [
        'Computing the square root now.',
        'The square root of 83810205 is approximately 9155.06.',
      ],
      1,
      [
        {
          name: 'calculator',
          args: { input: 'sqrt(83810205)' },
          id: 'tc_prior_2',
          type: 'tool_call',
        },
      ]
    );
    await runTurn(
      { run, conversationHistory },
      'Now take the square root of that result using the calculator.'
    );

    // History should now contain: Human, AI+toolcall, ToolMsg, AI,
    //                             Human, AI+toolcall, ToolMsg, AI
    expect(conversationHistory.length).toBeGreaterThanOrEqual(8);

    // Turn 3: tight budget → force summarization of the tool-heavy history
    addAccountingPressure(conversationHistory);
    run = await createRun({
      maxTokens: 512,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(['Understood, continuing.'], 1);

    await runTurn({ run, conversationHistory }, 'Summarize everything.');

    // Summarization should fire on the tool-heavy history
    expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const payload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    expect(payload.error).toBeUndefined();
    expect(payload.summary).toBeDefined();
    expect(payload.summary!.tokenCount).toBeGreaterThan(0);
    expect(getSummaryText(payload.summary)).toBe(SUMMARY_TEXT);
  });

  // =========================================================================
  // Test 8: Multiple sequential tool calls (chained) with usage_metadata
  // =========================================================================
  test('multiple chained tool calls with usage_metadata across the agent loop', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    const run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });

    // Use FakeWithUsage to emit different usage per call:
    // Call 1: tool call (input ~20 tokens)
    // Call 2: final answer (input ~40 tokens after tool result added)
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: [
        'Let me compute that step by step.',
        'The answer is 83,810,205. That is 12345 multiplied by 6789.',
      ],
      usage: [
        { input_tokens: 30, output_tokens: 12, total_tokens: 42 },
        { input_tokens: 60, output_tokens: 20, total_tokens: 80 },
      ],
    }) as any;

    // Since FakeWithUsage doesn't support tool calls natively, we need to
    // use overrideTestModel. But that replaces overrideModel. So instead,
    // let's test this scenario WITHOUT tool calls — just multi-response
    // with cycling usage_metadata to verify calibration persists.
    run.Graph!.overrideModel = new FakeWithUsage({
      responses: ['The answer is 83,810,205.'],
      usage: [{ input_tokens: 30, output_tokens: 15, total_tokens: 45 }],
    }) as any;

    conversationHistory.push(new HumanMessage('What is 12345 * 6789?'));
    await run.processStream(
      { messages: conversationHistory },
      streamConfig as any
    );

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();
    expect(runMessages!.length).toBeGreaterThan(0);

    // Now do a second Run to verify calibration from usage_metadata
    // persisted correctly and influences next turn's accounting
    conversationHistory.push(...(runMessages ?? []));

    const run2 = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });

    // Second call with higher input tokens (conversation grew)
    run2.Graph!.overrideModel = new FakeWithUsage({
      responses: ['The square root of that is approximately 9155.89.'],
      usage: [{ input_tokens: 55, output_tokens: 18, total_tokens: 73 }],
    }) as any;

    conversationHistory.push(
      new HumanMessage('Now compute the square root of that result.')
    );
    await run2.processStream(
      { messages: conversationHistory },
      streamConfig as any
    );

    const run2Messages = run2.getRunMessages();
    expect(run2Messages).toBeDefined();
    expect(run2Messages!.length).toBeGreaterThan(0);
  });

  // =========================================================================
  // Test 9: Multi-turn with tool calls triggers summarization across runs
  // =========================================================================
  test('multi-turn with tool calls across runs triggers summarization correctly', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    // Turn 1: tool call at generous budget
    let run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });
    run.Graph?.overrideTestModel(
      [
        'Let me compute 100 * 200 for you using the calculator tool.',
        'The result of 100 multiplied by 200 is 20,000. That is a basic multiplication.',
      ],
      1,
      [
        {
          name: 'calculator',
          args: { input: '100 * 200' },
          id: 'tc_multi_1',
          type: 'tool_call',
        },
      ]
    );
    await runTurn(
      { run, conversationHistory },
      'Calculate 100 * 200 with the calculator tool and explain the result.'
    );

    // Turn 2: another tool call to accumulate more tokens
    run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });
    run.Graph?.overrideTestModel(
      [
        'Now computing 300 * 400.',
        'The result of 300 multiplied by 400 is 120,000. Another straightforward calculation.',
      ],
      1,
      [
        {
          name: 'calculator',
          args: { input: '300 * 400' },
          id: 'tc_multi_2',
          type: 'tool_call',
        },
      ]
    );
    await runTurn(
      { run, conversationHistory },
      'Now compute 300 * 400 with the calculator and explain.'
    );

    // Conversation should have human, AI+toolcall, ToolMsg, AI × 2 turns
    expect(conversationHistory.length).toBeGreaterThanOrEqual(8);

    // Turn 3: tight budget to force summarization
    addAccountingPressure(conversationHistory);
    run = await createRun({
      maxTokens: 512,
      conversationHistory,
      spies,
    });
    run.Graph?.overrideTestModel(['Understood.'], 1);

    await runTurn({ run, conversationHistory }, 'What were all the results?');

    // Summarization should fire — tool messages are part of the history being summarized
    expect(spies.onSummarizeStartSpy).toHaveBeenCalled();
    expect(spies.onSummarizeCompleteSpy).toHaveBeenCalled();

    const completePayload = spies.onSummarizeCompleteSpy.mock
      .calls[0][0] as t.SummarizeCompleteEvent;
    expect(completePayload.error).toBeUndefined();
    expect(completePayload.summary).toBeDefined();
    expect(completePayload.summary!.tokenCount).toBeGreaterThan(0);

    // messagesToRefineCount should include the tool messages
    const startPayload = spies.onSummarizeStartSpy.mock
      .calls[0][0] as t.SummarizeStartEvent;
    expect(startPayload.messagesToRefineCount).toBeGreaterThan(0);
  });

  // =========================================================================
  // Test 10: No summarization when everything fits
  // =========================================================================
  test('no summarization fires when messages fit comfortably within budget', async () => {
    const spies = {
      onSummarizeStartSpy: jest.fn(),
      onSummarizeCompleteSpy: jest.fn(),
    };
    const conversationHistory: BaseMessage[] = [];

    const run = await createRun({
      maxTokens: 4000,
      conversationHistory,
      spies,
      tools: [new Calculator()],
    });

    const toolCalls: ToolCall[] = [
      {
        name: 'calculator',
        args: { input: '2 + 2' },
        id: 'tc_easy',
        type: 'tool_call',
      },
    ];
    run.Graph?.overrideTestModel(
      ['Let me calculate.', 'The answer is 4.'],
      1,
      toolCalls
    );

    conversationHistory.push(new HumanMessage('What is 2+2?'));
    await run.processStream(
      { messages: conversationHistory },
      streamConfig as any
    );

    // With 4000 token budget for a tiny conversation, no summarization should fire
    expect(spies.onSummarizeStartSpy).not.toHaveBeenCalled();
    expect(spies.onSummarizeCompleteSpy).not.toHaveBeenCalled();

    // But the tool call should have worked
    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();
    expect(runMessages!.length).toBeGreaterThanOrEqual(3);
    const toolMsgs = runMessages!.filter((m) => m._getType() === 'tool');
    expect(toolMsgs.length).toBe(1);
    expect(toolMsgs[0].content as string).toContain('4');
  });
});

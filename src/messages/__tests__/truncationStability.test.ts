import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { AgentInputs, FadingTier } from '@/types/graph';
import type { TokenCounter } from '@/types/run';
import {
  FADING_MIN_BUDGET_TOKENS,
  fadingBudgetTokens,
  fadingRungForBudget,
  fadingRungForExchangeChars,
  fadingRungForResultChars,
  isFadingTier,
  isInformativeFadingTier,
  maxFadingRung,
  resolveFadingCaps,
  resolveFadingTier,
  seedFadingTier,
  createFadingTier,
} from '@/messages/fading';
import {
  maskConsumedToolResults,
  preFlightTruncateToolResults,
  createPruneMessages,
  projectToolCallInputs,
} from '@/messages/prune';
import {
  calculateMaxToolCallInputChars,
  calculateMaxToolResultChars,
} from '@/utils/truncation';
import { ContentTypes, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { hasNonEmptyTextContent } from '@/messages/core';

const tokenCounter: TokenCounter = (message) => {
  const content = message.content;
  const text = typeof content === 'string' ? content : JSON.stringify(content);
  return Math.ceil(text.length / 4);
};

function serialize(message: BaseMessage): string {
  const content = message.content;
  return typeof content === 'string' ? content : JSON.stringify(content);
}

function toolCall(...ids: string[]): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: ids.map((id) => ({
      id,
      name: 'fetch',
      args: {},
      type: 'tool_call' as const,
    })),
  });
}

function toolCallWithInput(id: string, chars: number): AIMessage {
  const query = 'q'.repeat(chars);
  const serialized = JSON.stringify({ query });
  return new AIMessage({
    content: [
      {
        type: 'tool_use',
        id,
        name: 'fetch',
        input: { query },
      },
    ],
    tool_calls: [
      {
        id,
        name: 'fetch',
        args: { query },
        type: 'tool_call',
      },
    ],
    additional_kwargs: {
      tool_calls: [
        {
          id,
          type: 'function',
          function: { name: 'fetch', arguments: serialized },
        },
      ],
    },
    response_metadata: {
      output: [
        {
          type: 'function_call',
          call_id: id,
          name: 'fetch',
          arguments: serialized,
        },
      ],
    },
  });
}

function legacyToolCallWithInput(chars: number): AIMessage {
  return new AIMessage({
    content: '',
    additional_kwargs: {
      function_call: {
        name: 'fetch',
        arguments: 'q'.repeat(chars),
      },
    },
  });
}

function serializeToolCallInputs(message: BaseMessage): string {
  const aiMessage = message as AIMessage;
  return JSON.stringify({
    content: aiMessage.content,
    toolCalls: aiMessage.tool_calls,
    additionalKwargs: aiMessage.additional_kwargs,
    responseMetadata: aiMessage.response_metadata,
  });
}

function toolResult(id: string, chars: number): ToolMessage {
  return new ToolMessage({
    content: `${id}:${'x'.repeat(chars)}`,
    tool_call_id: id,
    name: 'fetch',
  });
}

/** One user turn: question, tool call, tool result, answer. */
function round(index: number, resultChars: number): BaseMessage[] {
  const id = `tc-${index}`;
  return [
    new HumanMessage(`question ${index}`),
    toolCall(id),
    toolResult(id, resultChars),
    new AIMessage(`answer ${index}`),
  ];
}

function conversation(rounds: number[]): BaseMessage[] {
  return rounds.flatMap((chars, index) => round(index, chars));
}

function countMap(messages: BaseMessage[]): Record<string, number | undefined> {
  const map: Record<string, number | undefined> = {};
  for (let i = 0; i < messages.length; i++) {
    map[i] = tokenCounter(messages[i]);
  }
  return map;
}

describe('fading ladder', () => {
  it('halves the budget per rung down to the floor', () => {
    expect(fadingBudgetTokens(100_000, 0)).toBe(100_000);
    expect(fadingBudgetTokens(100_000, 1)).toBe(50_000);
    expect(fadingBudgetTokens(100_000, 3)).toBe(12_500);
    expect(fadingBudgetTokens(100_000, 20)).toBe(170);
    expect(fadingBudgetTokens(100, 5)).toBe(100);
    expect(maxFadingRung(100_000)).toBe(10);
    expect(maxFadingRung(100)).toBe(0);
  });

  it('picks the shallowest rung that fits a budget or a result cap', () => {
    expect(fadingRungForBudget(100_000, 100_000)).toBe(0);
    expect(fadingRungForBudget(100_000, 60_000)).toBe(0);
    expect(fadingRungForBudget(100_000, 9_400)).toBe(3);
    expect(fadingRungForBudget(100_000, 9_400, 1_000)).toBe(1);
    expect(fadingRungForBudget(100_000, 9_400, 0)).toBe(1);
    expect(fadingRungForBudget(100_000, 9_400, 1_000, 3)).toBe(3);
    expect(fadingRungForBudget(100_000, 0)).toBe(0);
    expect(
      calculateMaxToolResultChars(
        fadingBudgetTokens(100_000, fadingRungForBudget(100_000, 9_400))
      ) / 4
    ).toBeLessThanOrEqual(9_400 / 2);
    expect(fadingRungForResultChars(100_000, 200)).toBe(maxFadingRung(100_000));
    expect(
      calculateMaxToolResultChars(
        fadingBudgetTokens(100_000, fadingRungForResultChars(100_000, 5_000))
      )
    ).toBeLessThanOrEqual(5_000);
  });

  it('validates and seeds persisted tiers', () => {
    expect(isFadingTier({ v: 2, budgetTokens: 100, masked: true })).toBe(true);
    /** Version 1 tiers could latch on reconstructed history widths; they re-derive. */
    expect(isFadingTier({ v: 1, budgetTokens: 100, masked: true })).toBe(false);
    expect(
      seedFadingTier(100_000, { v: 1, budgetTokens: 6_250, masked: false })
    ).toEqual({
      v: 2,
      budgetTokens: 100_000,
      masked: false,
    });
    expect(isFadingTier({ v: 2, budgetTokens: 0, masked: true })).toBe(false);
    expect(isFadingTier({ v: 2, budgetTokens: 100, masked: 'yes' })).toBe(
      false
    );
    expect(isFadingTier(null)).toBe(false);
    expect(
      seedFadingTier(100_000, { v: 2, budgetTokens: 50_000, masked: true })
    ).toEqual({
      v: 2,
      budgetTokens: 50_000,
      masked: true,
      latched: true,
    });
    expect(
      seedFadingTier(100_000, { v: 2, budgetTokens: 150_000, masked: false })
    ).toEqual({
      v: 2,
      budgetTokens: 100_000,
      masked: false,
      latched: true,
    });
    expect(seedFadingTier(100_000, { rung: 3 })).toEqual(
      createFadingTier(100_000)
    );
  });

  it('only deepens and never oscillates across a band threshold', () => {
    const window = 100_000;
    const base = { effectiveRawTokens: 100_000, summarizationEnabled: false };
    let tier = createFadingTier(window);
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.5 });
    expect(tier).toEqual(createFadingTier(window));
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.86 });
    expect(tier).toEqual({
      v: 2,
      budgetTokens: 50_000,
      masked: true,
      latched: true,
    });
    const settled = tier;
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.84 });
    expect(tier).toBe(settled);
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.86 });
    expect(tier).toBe(settled);
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.91 });
    expect(tier.budgetTokens).toBe(25_000);
    tier = resolveFadingTier(tier, window, { ...base, contextPressure: 0.3 });
    expect(tier.budgetTokens).toBe(25_000);
    expect(tier.masked).toBe(true);
  });

  it('survives a mid-run budget correction and the return to the normal window', () => {
    const latched: FadingTier = {
      v: 2,
      budgetTokens: 100_000,
      masked: true,
      latched: true,
    };
    const corrected = seedFadingTier(180_000, latched);
    expect(corrected).toEqual(latched);
    expect(
      resolveFadingTier(corrected, 180_000, {
        contextPressure: 0.4,
        effectiveRawTokens: 170_000,
        summarizationEnabled: true,
      })
    ).toBe(corrected);
    expect(seedFadingTier(200_000, corrected)).toEqual(latched);
    expect(resolveFadingCaps(seedFadingTier(200_000, corrected))).toEqual(
      resolveFadingCaps(latched)
    );
  });

  it('keeps a restored tier informative after clamping to a smaller window', () => {
    const restored = seedFadingTier(30_000, {
      v: 2,
      budgetTokens: 50_000,
      masked: false,
      latched: true,
    });

    expect(restored).toEqual({
      v: 2,
      budgetTokens: 30_000,
      masked: false,
      latched: true,
    });
    expect(isInformativeFadingTier(restored, 30_000)).toBe(true);
    expect(seedFadingTier(100_000, restored)).toEqual(restored);
  });

  it('fits a complete tool exchange within the effective budget', () => {
    const tier = resolveFadingTier(createFadingTier(32_000), 32_000, {
      contextPressure: 0.3,
      effectiveRawTokens: 9_400,
      summarizationEnabled: true,
    });
    const caps = resolveFadingCaps(tier);
    expect((caps.resultChars + caps.inputChars) / 4).toBeLessThanOrEqual(9_400);
    expect(caps.resultChars).toBe(
      calculateMaxToolResultChars(caps.budgetTokens)
    );
    expect(caps.consumedChars).toBe(caps.resultChars);
  });

  it('does not over-deepen when a configured result cap already fits', () => {
    const tier = resolveFadingTier(
      createFadingTier(100_000),
      100_000,
      {
        contextPressure: 0.3,
        effectiveRawTokens: 9_400,
        summarizationEnabled: true,
      },
      1_000
    );

    expect(tier).toEqual({
      v: 2,
      budgetTokens: 50_000,
      masked: false,
      latched: true,
    });
    expect(resolveFadingCaps(tier, 1_000)).toMatchObject({
      resultChars: 1_000,
      inputChars: 30_000,
    });
  });

  it('fits every input and result in the widest parallel exchange', () => {
    const rawTokens = 9_400;
    const width = 3;
    const tier = resolveFadingTier(
      createFadingTier(100_000),
      100_000,
      {
        contextPressure: 0.3,
        effectiveRawTokens: rawTokens,
        summarizationEnabled: true,
        toolExchangeWidth: width,
      },
      1_000
    );
    const caps = resolveFadingCaps(tier, 1_000);

    expect(tier.budgetTokens).toBe(12_500);
    expect(width * (caps.inputChars + caps.resultChars)).toBeLessThanOrEqual(
      rawTokens * 4
    );
  });

  it('masks consumed results to a fraction of the fresh cap with a floor', () => {
    const masked: FadingTier = { v: 2, budgetTokens: 100_000, masked: true };
    const caps = resolveFadingCaps(masked);
    expect(caps.consumedChars).toBe(Math.floor(caps.resultChars * 0.1));
    expect(
      resolveFadingCaps({ ...masked, budgetTokens: 400 }).consumedChars
    ).toBe(300);
    expect(resolveFadingCaps(masked, 1_000).resultChars).toBe(1_000);
    expect(resolveFadingCaps(masked, 0).resultChars).toBe(0);
    expect(resolveFadingCaps(masked, 0).consumedChars).toBe(0);
  });
});

describe('preFlightTruncateToolResults stability', () => {
  it('truncates a historical result to the same bytes however many results follow it', () => {
    const snapshots = [0, 1, 4, 20].map((later) => {
      const messages = conversation([6_000, 6_000, ...Array(later).fill(50)]);
      const indexTokenCountMap = countMap(messages);
      preFlightTruncateToolResults({
        messages,
        maxContextTokens: 1_000,
        indexTokenCountMap,
        tokenCounter,
      });
      return [serialize(messages[2]), serialize(messages[6])];
    });

    expect(snapshots[0][0]).toContain('truncated');
    expect(snapshots[0][1]).toContain('truncated');
    for (const snapshot of snapshots) {
      expect(snapshot).toEqual(snapshots[0]);
    }
  });
});

describe('projectToolCallInputs proxy safety', () => {
  it('sanitizes a proxied nested tool call without invoking its traps', () => {
    const nestedToolCall = new Proxy(
      {},
      {
        ownKeys: () => {
          throw new Error('ownKeys trap must not run');
        },
        getOwnPropertyDescriptor: () => {
          throw new Error('descriptor trap must not run');
        },
      }
    );
    const message = new AIMessage({
      content: [
        {
          type: 'tool_call',
          tool_call: nestedToolCall,
        } as never,
      ],
    });

    const [projected] = projectToolCallInputs([message], 1_000);
    const block = projected.content[0] as unknown as {
      tool_call: { args: unknown };
    };

    expect(block.tool_call).toEqual({
      args: '[Property accessor omitted]',
    });
  });
});

describe('maskConsumedToolResults stability', () => {
  it('masks a consumed result to the same bytes however many consumed results exist', () => {
    const snapshots = [1, 3, 12].map((rounds) => {
      const messages = conversation(Array(rounds).fill(6_000));
      const indexTokenCountMap = countMap(messages);
      const masked = maskConsumedToolResults({
        messages,
        indexTokenCountMap,
        tokenCounter,
        maxChars: 2_000,
      });
      expect(masked).toBe(rounds);
      return serialize(messages[2]);
    });

    expect(snapshots[0].length).toBeLessThanOrEqual(2_000);
    expect(snapshots[0]).toContain('truncated');
    for (const snapshot of snapshots) {
      expect(snapshot).toBe(snapshots[0]);
    }
  });

  it('never masks below the placeholder floor', () => {
    const messages = conversation([6_000]);
    maskConsumedToolResults({
      messages,
      indexTokenCountMap: countMap(messages),
      tokenCounter,
      maxChars: 10,
    });
    expect(serialize(messages[2]).length).toBeLessThanOrEqual(300);
    expect(serialize(messages[2]).length).toBeGreaterThan(10);
  });
});

describe('pruner keeps historical tool results byte-stable across turns', () => {
  const maxTokens = 40_000;

  type TurnOptions = {
    summarizationEnabled: boolean;
    calibrationRatio: number;
    fadingTier?: FadingTier;
    instructionTokens?: number;
  };

  it('uses a same-length caller replacement as the new canonical result', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('fetch'),
      toolCall('replace-result'),
      new ToolMessage({
        content: `OLD:${'o'.repeat(100)}`,
        tool_call_id: 'replace-result',
      }),
      new AIMessage('done'),
    ];
    const pruneMessages = createPruneMessages({
      maxTokens: 10_000,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
    });

    pruneMessages({ messages });
    messages[2] = new ToolMessage({
      content: `NEW:${'n'.repeat(50_000)}`,
      tool_call_id: 'replace-result',
    });
    pruneMessages({ messages });

    expect(serialize(messages[2])).toContain('NEW:');
    expect(serialize(messages[2])).not.toContain('OLD:');
  });

  it('reapplies a latched tier after explicit canonical prefix replacement', () => {
    const canonicalMessages: BaseMessage[] = [
      new HumanMessage('fetch'),
      toolCall('replace-canonical-result'),
      new ToolMessage({
        content: `OLD:${'o'.repeat(50_000)}`,
        tool_call_id: 'replace-canonical-result',
      }),
      new AIMessage('done'),
    ];
    const messages = [...canonicalMessages];
    const pruneMessages = createPruneMessages({
      maxTokens: 10_000,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
      fadingTier: {
        v: 2,
        budgetTokens: 5000,
        masked: true,
        latched: true,
      },
    });

    pruneMessages({ messages, canonicalMessages });
    canonicalMessages[2] = new ToolMessage({
      content: `NEW:${'n'.repeat(50_000)}`,
      tool_call_id: 'replace-canonical-result',
    });
    messages[2] = canonicalMessages[2];
    pruneMessages({ messages, canonicalMessages });

    expect(serialize(messages[2])).toContain('NEW:');
    expect(serialize(messages[2])).toContain('truncated');
    expect(serialize(messages[2]).length).toBeLessThan(50_000);
    expect(serialize(canonicalMessages[2]).length).toBeGreaterThan(50_000);

    canonicalMessages[2] = new ToolMessage({
      content: `APPENDED:${'a'.repeat(50_000)}`,
      tool_call_id: 'replace-canonical-result',
    });
    messages[2] = canonicalMessages[2];
    const appended = new HumanMessage('next turn');
    canonicalMessages.push(appended);
    messages.push(appended);
    pruneMessages({ messages, canonicalMessages });

    expect(serialize(messages[2])).toContain('APPENDED:');
    expect(serialize(messages[2])).toContain('truncated');
    expect(serialize(messages[2]).length).toBeLessThan(50_000);
  });

  /** Mirrors a host that rebuilds full-content messages and a fresh pruner every run. */
  function runTurn(
    messages: BaseMessage[],
    options: TurnOptions
  ): {
    early: string;
    pressure: number;
    tier: FadingTier;
    contextLength: number;
  } {
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: options.summarizationEnabled,
      calibrationRatio: options.calibrationRatio,
      fadingTier: options.fadingTier,
      getInstructionTokens: () => options.instructionTokens ?? 0,
    });
    const result = pruneMessages({ messages });
    return {
      early: serialize(messages[2]),
      pressure: result.contextPressure ?? 0,
      tier: result.fadingTier,
      contextLength: result.context.length,
    };
  }

  it('fit-to-budget truncation (summarization on) ignores calibration drift and growth', () => {
    const first = runTurn(conversation([60_000]), {
      summarizationEnabled: true,
      calibrationRatio: 1,
    });
    expect(first.early).toContain('truncated');
    expect(first.early.length).toBeLessThanOrEqual(
      calculateMaxToolResultChars(maxTokens)
    );

    const ratios = [1.03, 1.12, 1.25, 1.4];
    ratios.forEach((calibrationRatio, index) => {
      const later = runTurn(
        conversation([60_000, ...Array(index + 1).fill(400)]),
        {
          summarizationEnabled: true,
          calibrationRatio,
        }
      );
      expect(later.pressure).toBeLessThan(0.8);
      expect(later.early).toBe(first.early);
    });
  });

  it('masking under pressure (summarization off) is stable within a tier', () => {
    const history = [60_000, 48_000, 16_000];
    const first = runTurn(conversation(history), {
      summarizationEnabled: false,
      calibrationRatio: 1,
    });
    expect(first.pressure).toBeGreaterThanOrEqual(0.8);
    expect(first.tier.masked).toBe(true);
    expect(first.early).toContain('truncated');
    expect(first.early.length).toBeLessThanOrEqual(
      resolveFadingCaps(first.tier).consumedChars
    );

    const ratios = [1.01, 1.02, 1.03];
    ratios.forEach((calibrationRatio, index) => {
      const later = runTurn(
        conversation([...history, ...Array(index + 1).fill(200)]),
        {
          summarizationEnabled: false,
          calibrationRatio,
        }
      );
      expect(later.tier).toEqual(first.tier);
      expect(later.early).toBe(first.early);
    });
  });

  it('a persisted tier reproduces the same bytes even when pressure falls below the threshold', () => {
    const first = runTurn(conversation([60_000, 48_000, 16_000]), {
      summarizationEnabled: false,
      calibrationRatio: 1,
    });
    expect(first.tier.masked).toBe(true);

    const seeded = runTurn(conversation([60_000, 48_000, 16_000]), {
      summarizationEnabled: false,
      calibrationRatio: 0.7,
      fadingTier: first.tier,
    });
    expect(seeded.pressure).toBeLessThan(0.8);
    expect(seeded.tier).toEqual(first.tier);
    expect(seeded.early).toBe(first.early);

    const unseeded = runTurn(conversation([60_000, 48_000, 16_000]), {
      summarizationEnabled: false,
      calibrationRatio: 0.7,
    });
    expect(unseeded.tier.masked).toBe(false);
    expect(unseeded.early).not.toBe(first.early);
  });

  it('keeps a first context non-empty when instructions dominate a small window', () => {
    const messages = conversation([38_400]);
    const pruneMessages = createPruneMessages({
      maxTokens: 32_000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
      calibrationRatio: 1,
      getInstructionTokens: () => 21_000,
    });
    const result = pruneMessages({ messages });
    expect(result.context.length).toBe(messages.length);
    expect(result.messagesToRefine).toEqual([]);
    expect(serialize(messages[2]).length).toBeLessThanOrEqual(
      resolveFadingCaps(result.fadingTier).resultChars
    );
  });

  it('holds bytes stable within one run while provider usage recalibrates', () => {
    const messages = conversation([60_000]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
    });
    const first = pruneMessages({ messages });
    const early = serialize(messages[2]);
    expect(early).toContain('truncated');

    let previousRatio = first.calibrationRatio ?? 1;
    for (let turn = 1; turn <= 3; turn++) {
      messages.push(...round(turn, 400));
      const result = pruneMessages({
        messages,
        usageMetadata: { input_tokens: 13_500 + 500 * turn, output_tokens: 10 },
        totalTokensFresh: true,
      });
      expect(result.calibrationRatio).not.toBe(previousRatio);
      previousRatio = result.calibrationRatio ?? previousRatio;
      expect(serialize(messages[2])).toBe(early);
      expect(result.fadingTier).toEqual(first.fadingTier);
    }
  });

  it('derives a deeper tier from canonical content instead of the prior tier output', () => {
    const messages = conversation([60_000]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
    });
    pruneMessages({ messages });
    const firstTierBytes = serialize(messages[2]);

    messages.push(...round(1, 60_000), ...round(2, 60_000));
    const escalated = pruneMessages({ messages });
    const escalatedBytes = serialize(messages[2]);
    expect(escalatedBytes).not.toBe(firstTierBytes);

    const rebuilt = conversation([60_000, 60_000, 60_000]);
    const fresh = createPruneMessages({
      maxTokens,
      startIndex: rebuilt.length,
      tokenCounter,
      indexTokenCountMap: countMap(rebuilt),
      summarizationEnabled: false,
      fadingTier: escalated.fadingTier,
    });
    fresh({ messages: rebuilt });

    expect(serialize(rebuilt[2])).toBe(escalatedBytes);
  });

  it('derives deeper tool-call input caps from canonical arguments', () => {
    const initial = [
      new HumanMessage('fetch with a large query'),
      toolCallWithInput('large-input', 60_000),
      toolResult('large-input', 100),
      new AIMessage('query complete'),
    ];
    const messages: BaseMessage[] = [...initial];
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
    });
    pruneMessages({ messages });
    const firstTierInputs = serializeToolCallInputs(messages[1]);

    messages.push(...round(1, 60_000), ...round(2, 60_000));
    const escalated = pruneMessages({ messages });
    const escalatedInputs = serializeToolCallInputs(messages[1]);
    expect(escalatedInputs).not.toBe(firstTierInputs);

    const rebuilt: BaseMessage[] = [
      new HumanMessage('fetch with a large query'),
      toolCallWithInput('large-input', 60_000),
      toolResult('large-input', 100),
      new AIMessage('query complete'),
      ...round(1, 60_000),
      ...round(2, 60_000),
    ];
    const fresh = createPruneMessages({
      maxTokens,
      startIndex: rebuilt.length,
      tokenCounter,
      indexTokenCountMap: countMap(rebuilt),
      summarizationEnabled: false,
      fadingTier: escalated.fadingTier,
    });
    const freshResult = fresh({ messages: rebuilt });

    expect(freshResult.fadingTier).toEqual(escalated.fadingTier);
    expect(serializeToolCallInputs(rebuilt[1])).toBe(escalatedInputs);
  });

  it('snapshots the true original for the summarizer when a capped result is masked later', () => {
    const originalChars = serialize(conversation([60_000])[2]).length;
    const messages = conversation([60_000]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
      calibrationRatio: 1,
      getInstructionTokens: () => 0,
    });
    const first = pruneMessages({ messages });
    expect(first.fadingTier.masked).toBe(false);
    expect(serialize(messages[2]).length).toBeLessThan(originalChars);

    messages.push(...round(1, 40_000), ...round(2, 40_000));
    const second = pruneMessages({ messages });
    expect(second.fadingTier.masked).toBe(true);
    expect(serialize(messages[2]).length).toBeLessThanOrEqual(
      resolveFadingCaps(second.fadingTier).consumedChars
    );
    expect(serialize(messages[2])).toContain(
      `truncated: ${originalChars} chars`
    );
    expect(second.newOriginalToolContent?.get(2)?.length).toBe(originalChars);
  });

  it('composes nearby tool-call input caps without retaining the source message', () => {
    const source = toolCallWithInput('composable-input', 60_000);
    const [firstProjection] = projectToolCallInputs([source], 30_000);
    const [composedProjection] = projectToolCallInputs(
      [firstProjection],
      29_990
    );
    const [directProjection] = projectToolCallInputs([source], 29_990);

    expect(serializeToolCallInputs(composedProjection)).toBe(
      serializeToolCallInputs(directProjection)
    );
  });

  it('restores legacy function calls identically across model windows', () => {
    const tier: FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: false,
      latched: true,
    };
    const projectAtWindow = (windowTokens: number): string => {
      const messages: BaseMessage[] = [
        new HumanMessage('fetch with legacy arguments'),
        legacyToolCallWithInput(180_000),
      ];
      const pruneMessages = createPruneMessages({
        maxTokens: windowTokens,
        startIndex: messages.length,
        tokenCounter,
        indexTokenCountMap: countMap(messages),
        summarizationEnabled: false,
        fadingTier: tier,
      });
      pruneMessages({ messages });
      return serializeToolCallInputs(messages[1]);
    };

    expect(projectAtWindow(100_000)).toBe(projectAtWindow(200_000));
  });

  it('recounts provider-bound inputs after fixed-cap canonicalization', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('fetch an oversized input'),
      toolCallWithInput('hard-cap-input', 300_000),
    ];
    const inputCounter: TokenCounter = (message) =>
      Math.ceil(serializeToolCallInputs(message).length / 4);
    const originalCount = inputCounter(messages[1]);
    const pruneMessages = createPruneMessages({
      maxTokens: 1_000_000,
      startIndex: messages.length,
      tokenCounter: inputCounter,
      indexTokenCountMap: {
        0: inputCounter(messages[0]),
        1: originalCount,
      },
      summarizationEnabled: false,
    });

    const result = pruneMessages({ messages });

    expect(result.indexTokenCountMap[1]).toBe(inputCounter(messages[1]));
    expect(result.indexTokenCountMap[1]).toBeLessThan(originalCount);
  });

  it('retains canonical result provenance through position pruning', () => {
    const contextPruningConfig = {
      enabled: true,
      keepLastAssistants: 1,
      softTrimRatio: 0,
      minPrunableToolChars: 1,
      softTrim: { maxChars: 4_000, headChars: 1_500, tailChars: 1_500 },
      hardClear: { enabled: false },
    };
    const messages = conversation([60_000]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
      contextPruningConfig,
    });
    pruneMessages({ messages });
    const positionPruned = serialize(messages[2]);

    messages.push(...round(1, 60_000), ...round(2, 60_000));
    const escalated = pruneMessages({ messages });
    const escalatedBytes = serialize(messages[2]);
    expect(escalatedBytes).not.toBe(positionPruned);

    const rebuilt = conversation([60_000, 60_000, 60_000]);
    const fresh = createPruneMessages({
      maxTokens,
      startIndex: rebuilt.length,
      tokenCounter,
      indexTokenCountMap: countMap(rebuilt),
      summarizationEnabled: false,
      contextPruningConfig,
      fadingTier: escalated.fadingTier,
    });
    fresh({ messages: rebuilt });

    expect(serialize(rebuilt[2])).toBe(escalatedBytes);
  });

  it('recreates the provider projection from canonical graph messages after overflow', () => {
    const canonicalMessages = conversation([60_000]);
    const messages = [...canonicalMessages];
    const originalResult = serialize(canonicalMessages[2]);
    const firstPruner = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
    });
    const first = firstPruner({ messages, canonicalMessages });
    expect(serialize(canonicalMessages[2])).toBe(originalResult);

    const correctedMessages = [...canonicalMessages];
    const correctedPruner = createPruneMessages({
      maxTokens: 20_000,
      startIndex: correctedMessages.length,
      tokenCounter,
      indexTokenCountMap: countMap(correctedMessages),
      summarizationEnabled: false,
      fadingTier: first.fadingTier,
    });
    const corrected = correctedPruner({
      messages: correctedMessages,
      canonicalMessages,
    });

    const rebuilt = conversation([60_000]);
    const freshPruner = createPruneMessages({
      maxTokens: 20_000,
      startIndex: rebuilt.length,
      tokenCounter,
      indexTokenCountMap: countMap(rebuilt),
      summarizationEnabled: false,
      fadingTier: corrected.fadingTier,
    });
    freshPruner({ messages: rebuilt });

    expect(serialize(correctedMessages[2])).toBe(serialize(rebuilt[2]));
    expect(serialize(canonicalMessages[2])).toBe(originalResult);
  });

  it('derives deeper tiers from canonical graph messages without mutating them', () => {
    const canonicalMessages = conversation([60_000]);
    const messages = [...canonicalMessages];
    const originalResult = serialize(canonicalMessages[2]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
    });

    pruneMessages({ messages, canonicalMessages });
    const firstProjection = serialize(messages[2]);
    canonicalMessages.push(...round(1, 60_000), ...round(2, 60_000));
    messages.push(...canonicalMessages.slice(messages.length));
    pruneMessages({ messages, canonicalMessages });

    expect(serialize(messages[2])).not.toBe(firstProjection);
    expect(serialize(canonicalMessages[2])).toBe(originalResult);
  });

  it('preserves OpenAI thinking normalization while capping canonical tool inputs', () => {
    const canonicalMessages: BaseMessage[] = [
      new HumanMessage('fetch'),
      new AIMessage({
        content: '',
        additional_kwargs: {
          reasoning_content: 'private reasoning',
          provider_specific_fields: {
            thinking_blocks: [
              {
                type: 'thinking',
                thinking: 'private reasoning',
                signature: 'signature',
              },
            ],
          },
        },
        tool_calls: [
          {
            id: 'call-1',
            name: 'fetch',
            args: { query: 'q'.repeat(60_000) },
            type: 'tool_call',
          },
        ],
      }),
    ];
    const messages = [...canonicalMessages];
    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: false,
      thinkingEnabled: true,
      fadingTier: {
        v: 2,
        budgetTokens: 10_000,
        masked: false,
        latched: true,
      },
    });

    pruneMessages({ messages, canonicalMessages });

    const projected = messages[1] as AIMessage;
    expect(projected.content).toEqual([
      expect.objectContaining({
        type: ContentTypes.THINKING,
        thinking: 'private reasoning',
      }),
    ]);
    expect(projected.additional_kwargs.reasoning_content).toBeUndefined();
    expect(JSON.stringify(projected.tool_calls?.[0].args)).toContain(
      '_inputPrefix'
    );
    expect(canonicalMessages[1].content).toBe('');
    expect(
      (canonicalMessages[1] as AIMessage).additional_kwargs.reasoning_content
    ).toBe('private reasoning');
  });

  it('latches a stable tier for a parallel exchange before emergency pruning', () => {
    const ids = ['p1', 'p2', 'p3', 'p4'];
    const canonicalMessages: BaseMessage[] = [
      ...conversation(Array(20).fill(200)),
      new HumanMessage('fetch everything at once'),
      toolCall(...ids),
      ...ids.map((id) => toolResult(id, 47_000)),
    ];
    const messages = [...canonicalMessages];
    const originalLengths = ids.map(
      (_, index) =>
        serialize(
          canonicalMessages[canonicalMessages.length - ids.length + index]
        ).length
    );
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
      calibrationRatio: 1,
    });

    const result = pruneMessages({ messages, canonicalMessages });

    expect(result.context.length).toBeGreaterThan(0);
    expect(result.fadingTier.budgetTokens).toBe(20_000);
    const caps = resolveFadingCaps(result.fadingTier);
    ids.forEach((_, index) => {
      expect(
        serialize(messages[messages.length - ids.length + index]).length
      ).toBeLessThanOrEqual(caps.resultChars);
      expect(
        serialize(
          canonicalMessages[canonicalMessages.length - ids.length + index]
        ).length
      ).toBe(originalLengths[index]);
    });
    const contextSet = new Set(result.context);
    expect(
      result.messagesToRefine?.some((message) => contextSet.has(message))
    ).toBe(false);

    const again = pruneMessages({ messages, canonicalMessages });
    expect(again.fadingTier).toEqual(result.fadingTier);
  });

  it('sizes parallel exchanges represented only by raw OpenAI tool calls', () => {
    const ids = ['raw-1', 'raw-2', 'raw-3', 'raw-4'];
    const rawToolCall = new AIMessage({
      content: '',
      tool_calls: [],
    });
    rawToolCall.additional_kwargs.tool_calls = ids.map((id) => ({
      id,
      type: 'function',
      function: { name: 'fetch', arguments: '{}' },
    }));
    const canonicalMessages: BaseMessage[] = [
      ...conversation(Array(20).fill(200)),
      new HumanMessage('fetch everything at once'),
      rawToolCall,
      ...ids.map((id) => toolResult(id, 47_000)),
    ];
    const messages = [...canonicalMessages];
    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
      calibrationRatio: 1,
    });

    const result = pruneMessages({ messages, canonicalMessages });

    expect(result.context.length).toBeGreaterThan(0);
    expect(result.fadingTier.budgetTokens).toBe(20_000);
    for (const id of ids) {
      expect(
        result.context.some(
          (message) =>
            message.getType() === 'tool' &&
            (message as ToolMessage).tool_call_id === id
        )
      ).toBe(true);
    }
  });
});

describe('isInformativeFadingTier', () => {
  it('recognizes OpenAI Responses output text as substantive', () => {
    expect(
      hasNonEmptyTextContent([
        { type: 'output_text', text: 'completed response' } as never,
      ])
    ).toBe(true);
  });

  it('ignores proxied and accessor-backed text blocks safely', () => {
    const accessorBlock = Object.defineProperty({}, 'type', {
      get() {
        throw new Error('type getter must not run');
      },
    });
    const proxiedBlock = new Proxy(
      { type: 'text', text: 'hidden' },
      {
        get() {
          throw new Error('proxy getter must not run');
        },
      }
    );

    expect(hasNonEmptyTextContent([accessorBlock, proxiedBlock] as never)).toBe(
      false
    );
    expect(hasNonEmptyTextContent(new Proxy([], {}) as never)).toBe(false);
  });

  it('reports only tiers a host should persist', () => {
    expect(isInformativeFadingTier(undefined, 100_000)).toBe(false);
    expect(isInformativeFadingTier(createFadingTier(100_000), 100_000)).toBe(
      false
    );
    expect(
      isInformativeFadingTier(
        { v: 2, budgetTokens: 100_000, masked: true },
        100_000
      )
    ).toBe(true);
    expect(
      isInformativeFadingTier(
        { v: 2, budgetTokens: 50_000, masked: false },
        100_000
      )
    ).toBe(true);
    expect(
      isInformativeFadingTier(
        { v: 2, budgetTokens: 50_000, masked: false },
        undefined
      )
    ).toBe(false);
  });
});

describe('per-agent fading tier persistence', () => {
  const graphAgent = (agentId: string): AgentInputs => ({
    agentId,
    provider: Providers.OPENAI,
    instructions: agentId,
    maxContextTokens: 100_000,
  });

  it('restores keyed tiers without applying the default tier to every agent', () => {
    const defaultTier: FadingTier = {
      v: 2,
      budgetTokens: 50_000,
      masked: false,
      latched: true,
    };
    const workerTier: FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const graph = new StandardGraph({
      agents: [
        graphAgent('default'),
        graphAgent('worker'),
        graphAgent('unseeded'),
      ],
      fadingTier: defaultTier,
      fadingTiers: { worker: workerTier },
    });

    expect(graph.agentContexts.get('default')?.fadingTier).toEqual(defaultTier);
    expect(graph.agentContexts.get('worker')?.fadingTier).toEqual(workerTier);
    expect(graph.agentContexts.get('unseeded')?.fadingTier).toBeUndefined();
    const snapshot = graph.getFadingTiers();
    expect(snapshot).toEqual({
      default: defaultTier,
      worker: workerTier,
    });
    snapshot.worker.budgetTokens = 1;
    expect(graph.getFadingTiers().worker).toEqual(workerTier);
    const singular = graph.getFadingTier();
    expect(singular).toEqual(defaultTier);
    if (singular != null) {
      singular.budgetTokens = 1;
    }
    expect(graph.getFadingTier()).toEqual(defaultTier);
  });

  it('ignores malformed restored tiers instead of reporting them back', () => {
    const graph = new StandardGraph({
      agents: [graphAgent('default'), graphAgent('worker')],
      fadingTier: {
        v: 2,
        budgetTokens: Number.NaN,
        masked: true,
      } as FadingTier,
      fadingTiers: {
        worker: {
          v: 2,
          budgetTokens: 'abc',
          masked: true,
        } as unknown as FadingTier,
      },
    });

    expect(graph.agentContexts.get('default')?.fadingTier).toBeUndefined();
    expect(graph.agentContexts.get('worker')?.fadingTier).toBeUndefined();
    expect(graph.getFadingTier()).toBeUndefined();
    expect(graph.getFadingTiers()).toEqual({});
  });

  it('restores a tier learned after a persistent initial summary', () => {
    const postSummaryTier: FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const graph = new StandardGraph({
      agents: [
        {
          ...graphAgent('default'),
          initialSummary: { text: 'Compacted history', tokenCount: 10 },
        },
      ],
      fadingTier: postSummaryTier,
      fadingTiers: { default: postSummaryTier },
    });

    expect(graph.agentContexts.get('default')?.fadingTier).toEqual(
      postSummaryTier
    );
  });

  it('round-trips prototype-sensitive agent IDs', () => {
    const tier: FadingTier = {
      v: 2,
      budgetTokens: 25_000,
      masked: true,
      latched: true,
    };
    const fadingTiers = Object.fromEntries([['__proto__', tier]]);
    const graph = new StandardGraph({
      agents: [graphAgent('__proto__')],
      fadingTiers,
    });

    expect(Object.hasOwn(graph.getFadingTiers(), '__proto__')).toBe(true);
    expect(graph.getFadingTiers()['__proto__']).toEqual(tier);
  });
});

describe('seedFadingTier provenance', () => {
  it('keeps a never-tightened seed fresh so a pruner rebuild cannot pin later runs', () => {
    const fresh = createFadingTier(200_000);
    const reseeded = seedFadingTier(200_000, fresh);
    expect(reseeded).toEqual(fresh);
    expect(reseeded.latched).toBeUndefined();
    expect(isInformativeFadingTier(reseeded, 200_000)).toBe(false);
  });

  it('latches a seed that was tightened, masked, already latched, or clamped', () => {
    expect(
      seedFadingTier(200_000, { v: 2, budgetTokens: 50_000, masked: false })
    ).toEqual({
      v: 2,
      budgetTokens: 50_000,
      masked: false,
      latched: true,
    });
    expect(
      seedFadingTier(200_000, { v: 2, budgetTokens: 200_000, masked: true })
    ).toEqual({
      v: 2,
      budgetTokens: 200_000,
      masked: true,
      latched: true,
    });
    expect(
      seedFadingTier(200_000, {
        v: 2,
        budgetTokens: 200_000,
        masked: false,
        latched: true,
      })
    ).toEqual({ v: 2, budgetTokens: 200_000, masked: false, latched: true });
    expect(
      seedFadingTier(8_000, { v: 2, budgetTokens: 200_000, masked: false })
    ).toEqual({
      v: 2,
      budgetTokens: 8_000,
      masked: false,
      latched: true,
    });
  });

  it('clamps a sub-floor budget up to the ladder floor so caps stay positive', () => {
    const seeded = seedFadingTier(200_000, {
      v: 2,
      budgetTokens: 1,
      masked: false,
    });
    expect(seeded.budgetTokens).toBe(FADING_MIN_BUDGET_TOKENS);
    expect(seeded.latched).toBe(true);
    expect(resolveFadingCaps(seeded).resultChars).toBeGreaterThan(0);
    expect(resolveFadingCaps(seeded).inputChars).toBeGreaterThan(0);
  });
});

describe('fadingRungForExchangeChars', () => {
  it('deepens until a result plus its input fit, even when the result cap is tiny', () => {
    const window = 200_000;
    const target = 8_000;
    const exchangeChars = (
      rung: number,
      maxToolResultChars?: number
    ): number => {
      const budget = fadingBudgetTokens(window, rung);
      const resultChars = calculateMaxToolResultChars(budget);
      return (
        (maxToolResultChars == null
          ? resultChars
          : Math.min(resultChars, maxToolResultChars)) +
        calculateMaxToolCallInputChars(budget)
      );
    };
    for (const cap of [undefined, 1_000, 8_000]) {
      const rung = fadingRungForExchangeChars(window, target, cap);
      expect(rung).toBeGreaterThan(0);
      expect(exchangeChars(rung, cap)).toBeLessThanOrEqual(target);
      expect(exchangeChars(rung - 1, cap)).toBeGreaterThan(target);
    }
    /** A result-only rung would stop at rung 0 for a tiny configured cap. */
    expect(fadingRungForResultChars(window, target, 1_000)).toBe(0);
    expect(fadingRungForExchangeChars(window, 10_000_000)).toBe(0);
    expect(fadingRungForExchangeChars(window, 1)).toBe(maxFadingRung(window));
  });
});

describe('hasNonEmptyTextContent hostile iteration', () => {
  it('never invokes iterators or accessors while scanning content blocks', () => {
    const hostileIterator: unknown[] = [{ type: 'text', text: 'hidden' }];
    Object.defineProperty(hostileIterator, Symbol.iterator, {
      get() {
        throw new Error('iterator invoked');
      },
    });
    expect(
      hasNonEmptyTextContent(hostileIterator as BaseMessage['content'])
    ).toBe(true);

    const accessorElement: unknown[] = [];
    Object.defineProperty(accessorElement, 0, {
      get() {
        throw new Error('accessor invoked');
      },
      enumerable: true,
    });
    accessorElement.length = 2;
    accessorElement[1] = { type: 'text', text: 'visible' };
    expect(
      hasNonEmptyTextContent(accessorElement as BaseMessage['content'])
    ).toBe(true);

    const accessorOnly: unknown[] = [];
    Object.defineProperty(accessorOnly, 0, {
      get() {
        throw new Error('accessor invoked');
      },
    });
    expect(hasNonEmptyTextContent(accessorOnly as BaseMessage['content'])).toBe(
      false
    );
    expect(
      hasNonEmptyTextContent({
        type: 'text',
        text: 'x',
      } as unknown as BaseMessage['content'])
    ).toBe(false);
  });
});

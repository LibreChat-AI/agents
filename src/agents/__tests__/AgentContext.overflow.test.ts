import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { messagesStateReducer } from '@/messages/reducer';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

/**
 * The overflow-recovery bookkeeping on AgentContext: the budget correction,
 * its restoration between runs, and the stall detector that stops a recovery
 * loop when a correction demonstrably changed nothing.
 */
describe('AgentContext overflow recovery state', () => {
  const createContext = (maxContextTokens?: number): AgentContext =>
    AgentContext.fromConfig({
      agentId: 'overflow-agent',
      provider: Providers.ANTHROPIC,
      instructions: 'Test instructions',
      maxContextTokens,
    } as Partial<t.AgentInputs> as t.AgentInputs);

  it('records the correction and counts the attempt', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 274_468);

    expect(context.maxContextTokens).toBe(190_000);
    expect(context.overflowRecoveryAttempts).toBe(1);
    /** Forces the pruner to be rebuilt against the corrected budget. */
    expect(context.pruneMessages).toBeUndefined();
  });

  it('tracks input-ceiling units after correction and restores configured units on reset', () => {
    const context = createContext(32_000);
    context.summarizationConfig = {contextBudgetMode: 'window'};
    expect(context.summarizationBudgetMode).toBe('window');
    context.applyContextBudgetCorrection(26_600);
    expect(context.maxContextTokens).toBe(26_600);
    expect(context.summarizationBudgetMode).toBe('input');
    expect(context.summarizationConfig.contextBudgetMode).toBe('window');
    context.reset();
    expect(context.maxContextTokens).toBe(32_000);
    expect(context.summarizationBudgetMode).toBe('window');
  });

  it('does not change budget units when an overflow supplies no numeric correction', () => {
    const context = createContext(32_000);
    context.summarizationConfig = {contextBudgetMode: 'window'};
    context.applyContextBudgetCorrection(undefined);
    expect(context.summarizationBudgetMode).toBe('window');
    context.reset();
    expect(context.summarizationBudgetMode).toBe('window');
  });

  it('keeps one provider projection per run and rebuilds it after overflow', () => {
    const context = AgentContext.fromConfig(
      {
        agentId: 'overflow-agent',
        provider: Providers.ANTHROPIC,
        instructions: 'Test instructions',
        maxContextTokens: 1_000_000,
      } as Partial<t.AgentInputs> as t.AgentInputs,
      () => 1
    );
    const canonical = [
      new HumanMessage('canonical'),
      new AIMessage({ content: [{ type: 'text', text: 'answer' }] }),
    ];

    const projection = context.getProviderProjectedMessages(canonical);
    projection[0] = new HumanMessage('projected');
    (projection[1].content as Array<{ text: string }>).unshift({
      text: 'provider-only',
    });
    canonical.push(new HumanMessage('next'));

    expect(context.getProviderProjectedMessages(canonical)).toBe(projection);
    expect(projection).toHaveLength(3);
    expect(canonical[0].content).toBe('canonical');
    expect(canonical[1].content).toEqual([{ type: 'text', text: 'answer' }]);

    context.applyContextBudgetCorrection(190_000, 274_468);
    const rebuilt = context.getProviderProjectedMessages(canonical);

    expect(rebuilt).not.toBe(projection);
    expect(rebuilt[0]).toBe(canonical[0]);
    expect(rebuilt[1].content).toEqual([{ type: 'text', text: 'answer' }]);

    canonical[canonical.length - 1] = new HumanMessage('rewritten tail');
    expect(context.getProviderProjectedMessages(canonical)).not.toBe(rebuilt);
  });

  it('invalidates a projection when a reducer replacement also appends', async () => {
    const context = AgentContext.fromConfig(
      {
        agentId: 'overflow-agent',
        provider: Providers.ANTHROPIC,
        instructions: 'Test instructions',
        maxContextTokens: 1_000_000,
      } as Partial<t.AgentInputs> as t.AgentInputs,
      () => 1,
      { 0: 100, 1: 200 }
    );
    await context.tokenCalculationPromise;
    const original = new HumanMessage({ id: 'human-1', content: 'original' });
    const existingReply = new AIMessage({ id: 'ai-1', content: 'reply' });
    const canonical = [original, existingReply];
    const projection = context.getProviderProjectedMessages(canonical);
    const replacement = new HumanMessage({
      id: 'human-1',
      content: 'replacement',
    });
    const appended = new HumanMessage({ id: 'human-2', content: 'next' });

    context.invalidateProviderProjectionForMessageUpdates(canonical);
    expect(context.getProviderProjectedMessages(canonical)).toBe(projection);

    const updates = [replacement, appended];
    context.pendingOriginalToolContent = new Map([[0, 'full original']]);
    context.invalidateProviderProjectionForMessageUpdates(updates);
    const nextCanonical = messagesStateReducer(canonical, updates);
    const rebuilt = context.getProviderProjectedMessages(nextCanonical);

    expect(rebuilt).not.toBe(projection);
    expect(rebuilt.map((message) => message.content)).toEqual([
      'replacement',
      'reply',
      'next',
    ]);
    expect(context.indexTokenCountMap).toEqual({ 0: 1, 1: 200, 2: 1 });
    expect(context.pendingOriginalToolContent).toBeUndefined();
  });

  it('accepts reducer message-like and empty updates during invalidation', () => {
    const context = AgentContext.fromConfig(
      {
        agentId: 'overflow-agent',
        provider: Providers.ANTHROPIC,
        instructions: 'Test instructions',
        maxContextTokens: 1_000_000,
      } as Partial<t.AgentInputs> as t.AgentInputs,
      () => 1
    );
    const canonical = [
      new HumanMessage({ id: 'human-1', content: 'original' }),
    ];
    const projection = context.getProviderProjectedMessages(canonical);
    const replacement = {
      id: 'human-1',
      role: 'user' as const,
      content: 'replacement',
    };

    expect(() =>
      context.invalidateProviderProjectionForMessageUpdates([
        undefined,
        replacement,
      ])
    ).not.toThrow();
    const nextCanonical = messagesStateReducer(canonical, replacement);

    expect(context.getProviderProjectedMessages(nextCanonical)).not.toBe(
      projection
    );
  });

  it('summarizes the first overflow when deterministic pruning is unavailable', () => {
    const context = createContext(1_000_000);
    context.summarizationEnabled = true;

    expect(context.shouldSummarizeOverflow()).toBe(true);
  });

  it('summarizes immediately when no pruning budget is configured', () => {
    const context = AgentContext.fromConfig(
      {
        agentId: 'overflow-agent',
        provider: Providers.ANTHROPIC,
        instructions: 'Test instructions',
        summarizationEnabled: true,
      } as Partial<t.AgentInputs> as t.AgentInputs,
      () => 1
    );

    expect(context.shouldSummarizeOverflow()).toBe(true);
  });

  it('stages deterministic pruning before summarization when a counter exists', () => {
    const context = AgentContext.fromConfig(
      {
        agentId: 'overflow-agent',
        provider: Providers.ANTHROPIC,
        instructions: 'Test instructions',
        maxContextTokens: 1_000_000,
        summarizationEnabled: true,
      } as Partial<t.AgentInputs> as t.AgentInputs,
      () => 1
    );

    expect(context.shouldSummarizeOverflow()).toBe(false);
    context.applyContextBudgetCorrection(190_000, 274_468);
    expect(context.shouldSummarizeOverflow()).toBe(true);
  });

  it('preserves the earliest full tool output when masking records collide', () => {
    const context = createContext(1_000_000);
    context.preserveOriginalToolContent(
      new Map([
        [2, 'full output'],
        [4, 'another output'],
      ])
    );
    context.preserveOriginalToolContent(
      new Map([
        [2, 'truncated placeholder'],
        [6, 'new output'],
      ])
    );

    expect(context.pendingOriginalToolContent).toEqual(
      new Map([
        [2, 'full output'],
        [4, 'another output'],
        [6, 'new output'],
      ])
    );
  });

  it('releases index-keyed tool output snapshots on reset', () => {
    const context = createContext(1_000_000);
    context.preserveOriginalToolContent(new Map([[2, 'full output']]));
    context.reset();

    expect(context.pendingOriginalToolContent).toBeUndefined();
  });

  it('preserves tool output snapshots when checkpointed messages survive reset', () => {
    const context = createContext(1_000_000);
    context.preserveOriginalToolContent(new Map([[2, 'full output']]));
    context.reset({ preserveOriginalToolContent: true });

    expect(context.pendingOriginalToolContent).toEqual(
      new Map([[2, 'full output']])
    );
  });

  it('restores the pre-correction budget on reset', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 274_468);
    context.applyContextBudgetCorrection(133_000, 180_000);
    context.reset();

    expect(context.maxContextTokens).toBe(1_000_000);
    expect(context.overflowRecoveryAttempts).toBe(0);
  });

  it('leaves an untouched budget alone on reset', () => {
    const context = createContext(1_000_000);
    context.maxContextTokens = 500_000;
    context.reset();

    expect(context.maxContextTokens).toBe(500_000);
  });

  it('keeps fallback calibration out of the primary agent context', () => {
    const context = createContext(1_000_000);
    context.calibrationRatio = 1.5;

    context.applyObservedOverflowCalibration(Providers.VERTEXAI, 2);
    expect(context.calibrationRatio).toBe(1.5);

    context.applyObservedOverflowCalibration(Providers.ANTHROPIC, 2);
    expect(context.calibrationRatio).toBe(2);
  });

  it('clamps provider-observed calibration to the shared safe range', () => {
    const context = createContext(1_000_000);

    context.applyObservedOverflowCalibration(Providers.ANTHROPIC, 10);
    expect(context.calibrationRatio).toBe(5);

    context.applyObservedOverflowCalibration(Providers.ANTHROPIC, 0.1);
    expect(context.calibrationRatio).toBe(0.5);
  });

  it('records a summary-only recovery without inventing a token budget', () => {
    const context = createContext();
    context.applyContextBudgetCorrection(undefined, undefined);

    expect(context.maxContextTokens).toBeUndefined();
    expect(context.overflowRecoveryAttempts).toBe(1);
  });

  it('reports a stall when the prompt did not shrink', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);

    expect(context.overflowRecoveryStalled(250_000)).toBe(true);
    expect(context.overflowRecoveryStalled(260_000)).toBe(true);
  });

  it('reports no stall while the prompt is still shrinking', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);

    expect(context.overflowRecoveryStalled(180_000)).toBe(false);
  });

  it('compares stall measurements in uncalibrated token units', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);
    context.calibrationRatio = 2;

    expect(context.overflowRecoveryStalled(360_000)).toBe(false);
    expect(context.overflowRecoveryStalled(500_000)).toBe(true);
  });

  it('leaves fixed instruction overhead out of calibration normalization', () => {
    const context = createContext(1_000_000);
    context.systemMessageTokens = 100_000;
    context.applyContextBudgetCorrection(190_000, 250_000);
    context.calibrationRatio = 0.5;

    expect(context.overflowRecoveryStalled(160_000)).toBe(false);
    expect(context.overflowRecoveryStalled(175_000)).toBe(true);
  });

  it('reports no stall before any correction, or without a measurement', () => {
    const context = createContext(1_000_000);
    expect(context.overflowRecoveryStalled(250_000)).toBe(false);

    context.applyContextBudgetCorrection(190_000, 250_000);
    expect(context.overflowRecoveryStalled(undefined)).toBe(false);
  });

  it('clears the stall measurement on reset', () => {
    const context = createContext(1_000_000);
    context.applyContextBudgetCorrection(190_000, 250_000);
    context.reset();

    expect(context.overflowRecoveryStalled(250_000)).toBe(false);
  });
});

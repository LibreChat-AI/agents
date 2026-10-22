import { HumanMessage } from '@langchain/core/messages';
import { CustomAnthropic } from './index';
import { scriptReply } from '@/specs/prompt-cache/scriptedReplies';
import { sseResponse } from '@/specs/prompt-cache/liveRun';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const BETA = 'thinking-binding-controls-2026-08-01';

const params = (fields: Record<string, unknown>): Record<string, unknown> =>
  new CustomAnthropic({
    model: 'claude-test',
    apiKey: 'test-key',
    ...fields,
  }).invocationParams({} as never) as unknown as Record<string, unknown>;

describe('thinking block binding request fields', () => {
  it('sends nothing unless replay is active', () => {
    const request = params({ thinking: { type: 'adaptive' } });
    expect(request.thinking).toEqual({ type: 'adaptive' });
    expect(request.betas).not.toContain(BETA);
  });

  it('sends the beta and drop_block together when replay is active', () => {
    const request = params({
      thinking: { type: 'adaptive' },
      dropMismatchedThinkingBlocks: true,
    });
    expect(request.thinking).toEqual({
      type: 'adaptive',
      block_binding: { prefix_mismatch_behavior: 'drop_block' },
    });
    expect(request.betas).toContain(BETA);
  });

  it('never injects thinking when none is configured', () => {
    const request = params({ dropMismatchedThinkingBlocks: true });
    expect(request.thinking).toBeUndefined();
    expect(request.betas).not.toContain(BETA);
  });

  it('leaves disabled thinking alone', () => {
    const request = params({
      thinking: { type: 'disabled' },
      dropMismatchedThinkingBlocks: true,
    });
    expect(request.thinking).toEqual({ type: 'disabled' });
    expect(request.betas).not.toContain(BETA);
  });

  it('is switched on by the agent option for Anthropic only', () => {
    const base = {
      agentId: 'a',
      clientOptions: { model: 'claude-test', apiKey: 'k' },
      providerReasoningReplay: true,
    };
    const anthropic = AgentContext.fromConfig({
      ...base,
      provider: Providers.ANTHROPIC,
    } as never);
    expect(
      (anthropic.clientOptions as { dropMismatchedThinkingBlocks?: boolean })
        .dropMismatchedThinkingBlocks
    ).toBe(true);
    const bedrock = AgentContext.fromConfig({
      ...base,
      provider: Providers.BEDROCK,
    } as never);
    expect(
      (bedrock.clientOptions as { dropMismatchedThinkingBlocks?: boolean })
        .dropMismatchedThinkingBlocks
    ).toBeUndefined();
    const off = AgentContext.fromConfig({
      ...base,
      providerReasoningReplay: false,
      provider: Providers.ANTHROPIC,
    } as never);
    expect(
      (off.clientOptions as { dropMismatchedThinkingBlocks?: boolean })
        .dropMismatchedThinkingBlocks
    ).toBeUndefined();
  });

  it('puts the beta on the wire only when replay is active', async () => {
    const betaHeaders: Array<string | null> = [];
    const spy = jest
      .spyOn(global, 'fetch')
      .mockImplementation(async (_input, init) => {
        betaHeaders.push(
          new Headers(init?.headers as HeadersInit).get('anthropic-beta')
        );
        return sseResponse(scriptReply('anthropic', { text: 'ok' }) as string[]);
      });
    try {
      for (const drop of [true, false]) {
        await new CustomAnthropic({
          model: 'claude-test',
          apiKey: 'test-key',
          maxRetries: 0,
          streaming: true,
          thinking: { type: 'adaptive' },
          dropMismatchedThinkingBlocks: drop,
        } as never).invoke([new HumanMessage('hi')]);
      }
    } finally {
      spy.mockRestore();
    }
    expect(betaHeaders[0]).toBe(BETA);
    expect(betaHeaders[1]).toBeNull();
  });
});

import { HumanMessage } from '@langchain/core/messages';
import { describe, it, expect, jest } from '@jest/globals';
import { HookRegistry } from '@/hooks/HookRegistry';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { StandardGraph } from '@/graphs/Graph';
import { attemptInvoke } from '@/llm/invoke';
import { HandlerRegistry } from '@/events';
import { FakeChatModel } from '@/llm/fake';

function createEmptyStreamModel(): FakeChatModel {
  return new FakeChatModel({
    responses: [''],
    splitStrategy: { type: 'fixed', value: 1 },
  });
}

describe.each([Providers.OPENAI, Providers.ANTHROPIC, Providers.BEDROCK])(
  'attemptInvoke empty stream for %s',
  (provider) => {
    it.each(['local', 'onChunk', 'registered'])(
      'rejects zero chunks in the %s streaming path',
      async (mode) => {
        const model = createEmptyStreamModel();
        const context = new StandardGraph({
          runId: 'empty-stream',
          agents: [{ agentId: 'agent', provider }],
        });
        if (mode === 'registered') {
          context.handlerRegistry = new HandlerRegistry();
          context.handlerRegistry.register(
            GraphEvents.CHAT_MODEL_STREAM,
            new ChatModelStreamHandler()
          );
        }
        const onChunk = jest.fn(() => {});

        await expect(
          attemptInvoke({
            model,
            messages: [new HumanMessage('hi')],
            provider,
            context,
            onChunk: mode === 'onChunk' ? onChunk : undefined,
          })
        ).rejects.toThrow('The model provider returned an empty response.');
        expect(onChunk).not.toHaveBeenCalled();
      }
    );

    it('accepts an emitted chunk with empty content', async () => {
      const model = new FakeChatModel({ responses: [''] });
      const onChunk = jest.fn(() => {});

      const result = await attemptInvoke({
        model,
        messages: [new HumanMessage('hi')],
        provider,
        onChunk,
      });

      expect(onChunk).toHaveBeenCalledTimes(1);
      expect(result.messages).toHaveLength(1);
      expect(result.messages?.[0].content).toBe('');
    });
  }
);

describe('attemptInvoke empty preemption', () => {
  it('preserves an intentional restart before the provider call', async () => {
    const model = createEmptyStreamModel();
    const stream = jest.spyOn(model, 'stream');
    const context = new StandardGraph({
      runId: 'empty-preemption',
      agents: [{ agentId: 'agent', provider: Providers.OPENAI }],
      preemption: {
        shouldPreempt: () => true,
        subscribe: () => () => {},
        restartGraceMs: 0,
      },
    });
    context.hookRegistry = new HookRegistry();
    context.hookRegistry.register('PreemptBoundary', {
      hooks: [async () => ({ additionalContext: 'steer' })],
    });

    const result = await attemptInvoke({
      model,
      messages: [new HumanMessage('hi')],
      provider: Providers.OPENAI,
      context,
      preemptAgentId: 'agent',
    });

    expect(result.messages).toEqual([]);
    expect(stream).not.toHaveBeenCalled();
    expect(context.getPreemptStats().restarts).toBe(1);
  });
});

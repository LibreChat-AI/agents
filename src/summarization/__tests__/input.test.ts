import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import {
  boundSummarizationInput,
  SummarizationInputTooLargeError,
} from '../input';
import { Providers } from '@/common';

const params = { provider: Providers.OPENAI, tokenCounter: () => 10 };

describe('boundSummarizationInput', () => {
  it('does not allocate when the window already fits', () => {
    const messages = [new HumanMessage('question'), new AIMessage('answer')];
    expect(boundSummarizationInput({ ...params, messages, budget: 20 })).toBe(
      messages
    );
  });

  it('never removes half of a parallel tool exchange', () => {
    const messages = [
      new HumanMessage('old'),
      new AIMessage('old answer'),
      new HumanMessage('new'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'a', name: 'tool', args: {} },
          { id: 'b', name: 'tool', args: {} },
        ],
      }),
      new ToolMessage({ content: 'a', tool_call_id: 'a' }),
      new ToolMessage({ content: 'b', tool_call_id: 'b' }),
    ];
    expect(
      boundSummarizationInput({ ...params, messages, budget: 40 })
    ).toEqual(messages.slice(2));
    expect(() =>
      boundSummarizationInput({ ...params, messages, budget: 20 })
    ).toThrow(SummarizationInputTooLargeError);
  });

  it('keeps the newest tool exchange even when a final answer follows it', () => {
    const messages = [
      new HumanMessage('old'),
      new AIMessage('old answer'),
      new HumanMessage('new'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'a', name: 'tool', args: {} }],
      }),
      new ToolMessage({ content: 'a', tool_call_id: 'a' }),
      new AIMessage('final answer'),
    ];
    expect(() =>
      boundSummarizationInput({ ...params, messages, budget: 10 })
    ).toThrow(SummarizationInputTooLargeError);
  });

  it.each([4, 1])(
    'allocates only residual input tokens to tool output (characters per token: %s)',
    (charactersPerToken) => {
      for (const fixedTokens of [200, 650, 900]) {
        const call = new AIMessage({
          content: '',
          tool_calls: [
            { id: 'latest', name: 'search', args: { query: 'large input' } },
          ],
        });
        const result = new ToolMessage({
          content: 'x'.repeat(10_000),
          tool_call_id: 'latest',
        });
        const messages = [call, result];
        const before = JSON.stringify(messages);
        const count = (message: BaseMessage): number =>
          message instanceof ToolMessage
            ? Math.ceil(String(message.content).length / charactersPerToken) +
              12
            : fixedTokens;
        const counter = jest.fn(count);
        const bounded = boundSummarizationInput({
          provider: Providers.OPENAI,
          messages,
          budget: 1_000,
          tokenCounter: counter,
        });
        expect(
          bounded.reduce((total, message) => total + count(message), 0)
        ).toBeLessThanOrEqual(1_000);
        expect(bounded[0]).toBe(call);
        expect((bounded[1] as ToolMessage).tool_call_id).toBe('latest');
        expect(JSON.stringify(messages)).toBe(before);
        expect(
          counter.mock.calls.filter(([message]) => message === call)
        ).toHaveLength(1);
      }
    }
  );

  it('refuses to drop the newest indivisible user payload', () => {
    const messages = [new HumanMessage('newest')];
    expect(() =>
      boundSummarizationInput({ ...params, messages, budget: 5 })
    ).toThrow(SummarizationInputTooLargeError);
  });
});

import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
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
    const messages = [new HumanMessage('old'), new AIMessage('old answer'),
      new HumanMessage('new'), new AIMessage({content: '', tool_calls: [{id: 'a', name: 'tool', args: {}}]}),
      new ToolMessage({content: 'a', tool_call_id: 'a'}), new AIMessage('final answer')];
    expect(() => boundSummarizationInput({...params, messages, budget: 10})).toThrow(SummarizationInputTooLargeError);
  });

  it('refuses to drop the newest indivisible user payload', () => {
    const messages = [new HumanMessage('newest')];
    expect(() =>
      boundSummarizationInput({ ...params, messages, budget: 5 })
    ).toThrow(SummarizationInputTooLargeError);
  });
});

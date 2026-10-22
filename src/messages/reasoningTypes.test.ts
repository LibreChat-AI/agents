import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { SentPrefix } from './reasoningTypes';
import {
  dropReasoningOfRewrittenPrefix,
  isReasoningContentBlock,
} from './reasoningTypes';

describe('isReasoningContentBlock', () => {
  it.each([
    'think',
    'thinking',
    'thinking_delta',
    'redacted_thinking',
    'reasoning',
    'reasoning-delta',
    'reasoning_content',
  ])('recognizes %s blocks', (type) => {
    expect(isReasoningContentBlock({ type })).toBe(true);
  });

  it.each(['text', 'thinker', 'reasoning_label', 'reasoning.text', undefined])(
    'rejects %s blocks',
    (type) => {
      expect(isReasoningContentBlock({ type })).toBe(false);
    }
  );
});

describe('dropReasoningOfRewrittenPrefix', () => {
  const toolTurn = (id: string, signature: string): AIMessage =>
    new AIMessage({
      content: [{ type: 'thinking', thinking: '', signature } as never],
      tool_calls: [{ id, name: 'search', args: {}, type: 'tool_call' }],
    });
  const result = (content: string, id: string): ToolMessage =>
    new ToolMessage({ content, tool_call_id: id, name: 'search' });
  const signatures = (messages: BaseMessage[]): string[] =>
    JSON.stringify(messages).match(/sig-\d/g) ?? [];

  it('keeps every block while nothing earlier was rewritten', () => {
    const sent: SentPrefix = { stale: new Set() };
    const first = [new HumanMessage('q'), toolTurn('c1', 'sig-1'), result('full', 'c1')];
    expect(dropReasoningOfRewrittenPrefix(first, sent).messages).toBe(first);
    const second = [...first, toolTurn('c2', 'sig-2'), result('more', 'c2')];
    const out = dropReasoningOfRewrittenPrefix(second, sent);
    expect(out.messages).toBe(second);
    expect(out.dropped).toEqual([]);
  });

  it('strips the blocks after a rewritten tool result and keeps the tool round', () => {
    const sent: SentPrefix = { stale: new Set() };
    const history = [new HumanMessage('q1'), toolTurn('c0', 'sig-0'), result('x'.repeat(50), 'c0')];
    const turn = toolTurn('c1', 'sig-1');
    dropReasoningOfRewrittenPrefix([...history, new HumanMessage('q2'), turn, result('r1', 'c1')], sent);
    const rewritten = [
      history[0],
      history[1],
      result('x', 'c0'),
      new HumanMessage('q2'),
      turn,
      result('r1', 'c1'),
    ];
    const out = dropReasoningOfRewrittenPrefix(rewritten, sent);
    expect(signatures(out.messages)).toEqual(['sig-0']);
    expect(out.dropped).toEqual([turn]);
    expect((out.messages[4] as AIMessage).tool_calls).toHaveLength(1);
    expect(out.messages[5].content).toBe('r1');
  });

  it('keeps a turn stripped on the requests that follow, and blocks produced after the rewrite', () => {
    const sent: SentPrefix = { stale: new Set() };
    const turn = toolTurn('c1', 'sig-1');
    const base = [new HumanMessage('q'), new AIMessage('a')];
    dropReasoningOfRewrittenPrefix([...base, turn, result('r1', 'c1')], sent);
    const rewritten = [new HumanMessage('q'), new AIMessage('changed'), turn, result('r1', 'c1')];
    dropReasoningOfRewrittenPrefix(rewritten, sent);
    const next = toolTurn('c2', 'sig-2');
    const out = dropReasoningOfRewrittenPrefix(
      [...rewritten, next, result('r2', 'c2')],
      sent
    );
    expect(signatures(out.messages)).toEqual(['sig-2']);
    expect(out.dropped).toEqual([]);
  });
});

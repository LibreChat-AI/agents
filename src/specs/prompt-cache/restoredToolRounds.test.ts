import type * as t from '@/types';
import { ContentTypes } from '@/common';
import { formatAgentMessages } from '@/messages';

/**
 * Sequential tool rounds of one turn must restore as separate assistant
 * messages: merged into one parallel batch, the next turn no longer starts
 * with the bytes of the previous request. The live-versus-restored byte
 * comparison per provider is in liveRestore.test.ts.
 */

const TOOL_NAMES = new Set(['lookup_order']);

describe('formatAgentMessages tool round grouping', () => {
  const call = (
    id: string,
    roundId?: string
  ): t.MessageContentComplex => ({
    type: ContentTypes.TOOL_CALL,
    tool_call: {
      id,
      name: 'lookup_order',
      args: '{"query":"1"}',
      output: 'ok',
      progress: 1,
      ...(roundId != null && { roundId }),
    },
  });
  const text = (value: string): t.MessageContentComplex => ({
    type: ContentTypes.TEXT,
    text: value,
  });
  const shape = (parts: t.MessageContentComplex[]): string[] =>
    formatAgentMessages(
      [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: parts },
      ],
      undefined,
      TOOL_NAMES
    ).messages.map((message) => {
      const toolCalls = (message as { tool_calls?: Array<{ id?: string }> })
        .tool_calls;
      const content =
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content);
      return `${message.getType()}:${content}:${(toolCalls ?? []).map((c) => c.id).join(',')}`;
    });

  it('keeps calls of one round together and splits at a new round', () => {
    expect(
      shape([call('a', 'r1'), call('b', 'r1'), call('c', 'r2'), text('Done.')])
    ).toEqual([
      'human:[{"type":"text","text":"hi"}]:',
      'ai::a,b',
      'tool:ok:',
      'tool:ok:',
      'ai::c',
      'tool:ok:',
      'ai:[{"type":"text","text":"Done."}]:',
    ]);
  });

  it('places text emitted before a call in that call’s message', () => {
    expect(
      shape([text('First.'), call('a', 'r1'), text('Second.'), call('b', 'r2')])
    ).toEqual([
      'human:[{"type":"text","text":"hi"}]:',
      'ai:First.:a',
      'tool:ok:',
      'ai:Second.:b',
      'tool:ok:',
    ]);
  });

  it('splits text-separated calls stored without round ids', () => {
    expect(shape([call('a'), text('Next.'), call('b')])).toEqual([
      'human:[{"type":"text","text":"hi"}]:',
      'ai::a',
      'tool:ok:',
      'ai:Next.:b',
      'tool:ok:',
    ]);
  });
});

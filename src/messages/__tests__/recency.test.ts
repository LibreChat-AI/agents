import {
  AIMessage,
  ChatMessage,
  FunctionMessage,
  HumanMessage,
  ToolMessage,
  SystemMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import {
  resolveIntraTurnRetainTokens,
  splitAtRecencyBoundary,
} from '@/messages/recency';

describe('resolveIntraTurnRetainTokens', () => {
  it('prefers an explicit budget and otherwise retains 16% of context', () => {
    expect(
      resolveIntraTurnRetainTokens({
        tokens: 4_096,
        maxContextTokens: 100_000,
      })
    ).toBe(4_096);
    expect(resolveIntraTurnRetainTokens({ maxContextTokens: 100_000 })).toBe(
      16_000
    );
  });

  it('disables the fallback without a usable context budget', () => {
    expect(resolveIntraTurnRetainTokens({})).toBeUndefined();
    expect(
      resolveIntraTurnRetainTokens({ maxContextTokens: 0 })
    ).toBeUndefined();
    expect(
      resolveIntraTurnRetainTokens({
        tokens: 0,
        maxContextTokens: 100_000,
      })
    ).toBeUndefined();
  });
});

describe('splitAtRecencyBoundary', () => {
  describe('default behavior (turns: 2)', () => {
    it('returns empty head and full tail for an empty array', () => {
      const result = splitAtRecencyBoundary([], { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
      expect(result.tailStartIndex).toBe(0);
    });

    it('always preserves the most recent turn even with one large message', () => {
      const messages = [new HumanMessage('huge first message'.repeat(1000))];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('keeps a complete user-assistant exchange in the tail', () => {
      const messages = [new HumanMessage('hi'), new AIMessage('hello')];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('places older turns in the head when there are more turns than the cap', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage('reply 1'),
        new HumanMessage('turn 2'),
        new AIMessage('reply 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual(messages.slice(0, 2));
      expect(result.tail).toEqual(messages.slice(2));
      expect(result.tailTurnCount).toBe(2);
      expect(result.tailStartIndex).toBe(2);
    });

    it('preserves tool_use / tool_result pairs across the boundary', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_a', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result A',
          tool_call_id: 'call_a',
          name: 'search',
        }),
        new AIMessage('done with turn 1'),
        new HumanMessage('turn 2'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_b', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result B',
          tool_call_id: 'call_b',
          name: 'search',
        }),
        new AIMessage('done with turn 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      // Head must contain turn 1's complete tool_use → tool_result pair.
      expect(result.head).toHaveLength(4);
      expect(result.head[0]).toBe(messages[0]);
      expect(result.head[3]).toBe(messages[3]);
      // Tail starts cleanly at turn 2's HumanMessage — never mid-pair.
      expect(result.tail[0]).toBe(messages[4]);
      expect(result.tail).toHaveLength(6);
    });
  });

  it('uses the serving provider to classify generic turn boundaries', () => {
    const genericAssistant = new ChatMessage({
      role: 'assistant',
      content: 'model turn',
    });
    const messages = [new HumanMessage('question'), genericAssistant];

    expect(
      splitAtRecencyBoundary(messages, { turns: 1, provider: 'openai' }).tail
    ).toEqual(messages);
    expect(
      splitAtRecencyBoundary(messages, { turns: 1, provider: 'bedrock' }).tail
    ).toEqual([genericAssistant]);
  });

  describe('disabled (turns: 0)', () => {
    it('puts everything in head when turns is 0', () => {
      const messages = [
        new HumanMessage('one'),
        new AIMessage('two'),
        new HumanMessage('three'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 0 });
      expect(result.head).toEqual(messages);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
    });

    it('treats negative turns as 0', () => {
      const messages = [new HumanMessage('a'), new AIMessage('b')];
      const result = splitAtRecencyBoundary(messages, { turns: -5 });
      expect(result.tail).toEqual([]);
      expect(result.head).toEqual(messages);
    });
  });

  describe('token cap', () => {
    it('honors the token cap when adding older turns', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage('reply 1'),
        new HumanMessage('turn 2'),
        new AIMessage('reply 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const tokenCounter = (): number => 100;
      const result = splitAtRecencyBoundary(messages, {
        turns: 5,
        tokens: 250,
        tokenCounter,
      });
      // Last turn is always preserved (200 tokens for 2 messages).
      // Adding turn 2 would push to 400, exceeding cap of 250 → stop.
      expect(result.tailTurnCount).toBe(1);
      expect(result.tail).toEqual(messages.slice(4));
    });

    it('always preserves the most recent turn even when it exceeds the cap', () => {
      const messages = [new HumanMessage('huge'), new AIMessage('also huge')];
      const tokenCounter = (): number => 1_000_000;
      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokens: 10,
        tokenCounter,
      });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('ignores the token cap when no tokenCounter is provided', () => {
      const messages = [
        new HumanMessage('a'),
        new AIMessage('b'),
        new HumanMessage('c'),
        new AIMessage('d'),
      ];
      const result = splitAtRecencyBoundary(messages, {
        turns: 3,
        tokens: 1, // would force tail to most-recent-only if applied
      });
      // No tokenCounter → fall back to turn-based selection only.
      expect(result.tailTurnCount).toBe(2);
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
    });
  });

  describe('pairing-balanced intra-turn fallback', () => {
    it('uses provider roles while finding an intra-turn boundary', () => {
      const genericAssistant = new ChatMessage({
        role: 'assistant',
        content: '',
        additional_kwargs: {
          tool_calls: [
            {
              id: 'generic',
              type: 'function',
              function: { name: 'search', arguments: '{}' },
            },
          ],
        },
      });
      const messages = [
        genericAssistant,
        new ToolMessage({
          content: 'result',
          tool_call_id: 'generic',
          name: 'search',
        }),
        new AIMessage('latest state'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        provider: 'bedrock',
        turns: 1,
        tokenCounter: () => 100,
        intraTurnTokens: 100,
      });

      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.usedIntraTurnFallback).toBe(false);
    });

    it('compacts older closed tool units from a single long turn', () => {
      const messages: BaseMessage[] = [new HumanMessage('inspect the repo')];
      for (let index = 0; index < 4; index++) {
        const id = `call_${index}`;
        messages.push(
          new AIMessage({
            content: '',
            tool_calls: [{ id, name: 'search', args: { query: index } }],
          }),
          new ToolMessage({
            content: `result ${index}`,
            tool_call_id: id,
            name: 'search',
          })
        );
      }
      messages.push(new AIMessage('preparing the change'));

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 300,
      });

      expect(result.head).toEqual(messages.slice(0, 7));
      expect(result.tail).toEqual(messages.slice(7));
      expect(result.tailStartIndex).toBe(7);
      expect(result.usedIntraTurnFallback).toBe(true);
      expect(result.tail[0]).toBeInstanceOf(AIMessage);
      expect((result.tail[0] as AIMessage).tool_calls?.[0]?.id).toBe('call_3');
      expect((result.tail[1] as ToolMessage).tool_call_id).toBe('call_3');
    });

    it('does not split parallel tool calls from their results', () => {
      const messages = [
        new HumanMessage('compare both'),
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'call_a', name: 'search', args: {} },
            { id: 'call_b', name: 'search', args: {} },
          ],
        }),
        new ToolMessage({
          content: 'result A',
          tool_call_id: 'call_a',
          name: 'search',
        }),
        new ToolMessage({
          content: 'result B',
          tool_call_id: 'call_b',
          name: 'search',
        }),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_c', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result C',
          tool_call_id: 'call_c',
          name: 'search',
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 300,
      });

      expect(result.head).toEqual(messages.slice(0, 4));
      expect(result.tail).toEqual(messages.slice(4));
    });

    it('falls back within the first turn after a leading preamble', () => {
      const messages = [
        new SystemMessage('programmatic preamble'),
        new HumanMessage('inspect'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_a', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result A',
          tool_call_id: 'call_a',
          name: 'search',
        }),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_b', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result B',
          tool_call_id: 'call_b',
          name: 'search',
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 300,
      });

      expect(result.head).toEqual(messages.slice(0, 4));
      expect(result.tail).toEqual(messages.slice(4));
    });

    it('does not split messages derived from one persisted source row', () => {
      const messages = [
        new HumanMessage('inspect'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_a', name: 'search', args: {} }],
          additional_kwargs: { sourceMessageId: 'expanded-row' },
        }),
        new ToolMessage({
          content: 'result A',
          tool_call_id: 'call_a',
          name: 'search',
          additional_kwargs: { sourceMessageId: 'expanded-row' },
        }),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_b', name: 'search', args: {} }],
          additional_kwargs: { sourceMessageId: 'expanded-row' },
        }),
        new ToolMessage({
          content: 'result B',
          tool_call_id: 'call_b',
          name: 'search',
          additional_kwargs: { sourceMessageId: 'expanded-row' },
        }),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_c', name: 'search', args: {} }],
          additional_kwargs: { sourceMessageId: 'retained-row' },
        }),
        new ToolMessage({
          content: 'result C',
          tool_call_id: 'call_c',
          name: 'search',
          additional_kwargs: { sourceMessageId: 'retained-row' },
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 300,
      });

      expect(result.head).toEqual(messages.slice(0, 5));
      expect(result.tail).toEqual(messages.slice(5));
    });

    it('recognizes raw and content-block provider tool pairs', () => {
      const rawCallMessage = new AIMessage('');
      rawCallMessage.additional_kwargs.tool_calls = [
        {
          id: 'raw_call',
          type: 'function',
          function: { name: 'search', arguments: '{}' },
        },
      ];
      const messages = [
        new HumanMessage('inspect'),
        rawCallMessage,
        new ToolMessage({
          content: 'raw result',
          tool_call_id: 'raw_call',
          name: 'search',
        }),
        new AIMessage({
          content: [
            {
              type: 'tool_use',
              id: 'content_call',
              name: 'search',
              input: {},
            },
          ],
        }),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'content_call',
              content: 'content result',
            },
          ],
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 100,
      });

      expect(result.head).toEqual(messages.slice(0, 5));
      expect(result.tail).toEqual(messages.slice(5));
    });

    it('does not treat provider-native result messages as user turns', () => {
      const messages = [
        new HumanMessage('inspect'),
        new AIMessage({
          content: [
            { type: 'tool_use', id: 'call_a', name: 'search', input: {} },
          ],
        }),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_a',
              content: 'result A',
            },
          ],
        }),
        new AIMessage({
          content: [
            { type: 'tool_use', id: 'call_b', name: 'search', input: {} },
          ],
        }),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_b',
              content: 'result B',
            },
          ],
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, { turns: 1 });

      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('preserves an unpaired provider-native result as a user turn', () => {
      const messages = [
        new HumanMessage('first turn'),
        new AIMessage('first reply'),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'missing_call',
              content: 'user-authored payload',
            },
          ],
        }),
      ];

      const result = splitAtRecencyBoundary(messages, { turns: 1 });

      expect(result.head).toEqual(messages.slice(0, 2));
      expect(result.tail).toEqual(messages.slice(2));
      expect(result.tailTurnCount).toBe(1);
    });

    it('preserves an empty HumanMessage as a user turn', () => {
      const messages = [
        new HumanMessage('first turn'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'closed', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'closed result',
          tool_call_id: 'closed',
          name: 'search',
        }),
        new HumanMessage({ content: [] }),
        new AIMessage('working'),
      ];

      const result = splitAtRecencyBoundary(messages, { turns: 1 });

      expect(result.head).toEqual(messages.slice(0, 3));
      expect(result.tail).toEqual(messages.slice(3));
      expect(result.tailTurnCount).toBe(1);
    });

    it('keeps incompatible provider results from closing a tool call', () => {
      const messages = [
        new HumanMessage('inspect'),
        new AIMessage({
          content: [
            { type: 'tool_use', id: 'shared', name: 'search', input: {} },
          ],
        }),
        new AIMessage({
          content: [
            {
              type: 'toolResponse',
              toolResponse: {
                id: 'shared',
                name: 'search',
                response: { output: 'wrong protocol' },
              },
            },
          ],
        }),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'shared',
              content: 'matched result',
            },
          ],
        }),
        new AIMessage({
          content: [
            { type: 'tool_use', id: 'recent', name: 'search', input: {} },
          ],
        }),
        new HumanMessage({
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'recent',
              content: 'recent result',
            },
          ],
        }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 300,
      });

      expect(result.head).toEqual(messages.slice(0, 4));
      expect(result.tail).toEqual(messages.slice(4));
    });

    it('keeps legacy OpenAI function calls with their FunctionMessage', () => {
      const legacyCall = new AIMessage('');
      legacyCall.additional_kwargs.function_call = {
        name: 'lookup',
        arguments: '{}',
      };
      const messages = [
        new HumanMessage('inspect'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'closed', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'closed result',
          tool_call_id: 'closed',
          name: 'search',
        }),
        legacyCall,
        new FunctionMessage({ name: 'lookup', content: 'legacy result' }),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 200,
      });

      expect(result.head).toEqual(messages.slice(0, 3));
      expect(result.tail).toEqual(messages.slice(3));
    });

    it('compacts paired Gemini code-execution units without identifiers', () => {
      const codeUnit = (value: number): AIMessage =>
        new AIMessage({
          content: [
            {
              type: 'executableCode',
              executableCode: {
                language: 'PYTHON',
                code: `print(${value})`,
              },
            },
            {
              type: 'codeExecutionResult',
              codeExecutionResult: {
                outcome: 'OUTCOME_OK',
                output: String(value),
              },
            },
          ],
        });
      const messages = [
        new HumanMessage('run the calculations'),
        codeUnit(1),
        codeUnit(2),
        codeUnit(3),
        new AIMessage('done'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 200,
      });

      expect(result.head).toEqual(messages.slice(0, 3));
      expect(result.tail).toEqual(messages.slice(3));
    });

    it('does not let an older tool pair license a cut in a later turn', () => {
      const messages = [
        new HumanMessage('first turn'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'closed', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'closed result',
          tool_call_id: 'closed',
          name: 'search',
        }),
        new HumanMessage('large second user request'),
        new AIMessage('working'),
        new AIMessage('still working'),
        new AIMessage('latest state'),
      ];

      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokenCounter: () => 100,
        intraTurnTokens: 200,
      });

      expect(result.head).toEqual(messages.slice(0, 3));
      expect(result.tail).toEqual(messages.slice(3));
      expect(result.usedIntraTurnFallback).toBe(true);
    });

    it('retains an open tool call and protects a lone user payload', () => {
      const openCall = [
        new HumanMessage('inspect'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'closed', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'closed result',
          tool_call_id: 'closed',
          name: 'search',
        }),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'open', name: 'search', args: {} }],
        }),
      ];
      const options = {
        turns: 2,
        tokenCounter: (): number => 100,
        intraTurnTokens: 100,
      };

      const openResult = splitAtRecencyBoundary(openCall, options);
      const loneUser = [new HumanMessage('payload'.repeat(10_000))];
      const loneResult = splitAtRecencyBoundary(loneUser, options);

      expect(openResult.head).toEqual(openCall.slice(0, 3));
      expect(openResult.tail).toEqual(openCall.slice(3));
      expect(loneResult.head).toEqual([]);
      expect(loneResult.tail).toEqual(loneUser);
    });

    it('counts every message at most once when falling back within a turn', () => {
      const messages: BaseMessage[] = [new HumanMessage('inspect')];
      for (let index = 0; index < 100; index++) {
        const id = `call_${index}`;
        messages.push(
          new AIMessage({
            content: '',
            tool_calls: [{ id, name: 'search', args: {} }],
          }),
          new ToolMessage({
            content: `result ${index}`,
            tool_call_id: id,
            name: 'search',
          })
        );
      }
      let calls = 0;

      splitAtRecencyBoundary(messages, {
        turns: 2,
        tokens: 30,
        tokenCounter: () => {
          calls += 1;
          return 1;
        },
        intraTurnTokens: 30,
      });

      expect(calls).toBe(messages.length);
    });
  });

  describe('linearity', () => {
    it('calls tokenCounter once per message in visited turns (no quadratic recount)', () => {
      // Build a long history: 200 turns × 10 messages = 2,000 messages.
      // If the boundary search were quadratic in the number of turns,
      // the call count would explode (e.g., 200 × 2,000 = 400,000).
      // The disjoint-slice invariant guarantees one call per visited
      // message, bounded by messages.length even with a generous turn
      // budget that visits every turn.
      const messages: BaseMessage[] = [];
      const turnCount = 200;
      const messagesPerTurn = 10;
      for (let t = 0; t < turnCount; t++) {
        messages.push(new HumanMessage(`turn ${t} query`));
        for (let m = 1; m < messagesPerTurn; m++) {
          messages.push(new AIMessage(`turn ${t} reply ${m}`));
        }
      }

      let calls = 0;
      const tokenCounter = (): number => {
        calls += 1;
        return 1;
      };

      // Generous tokens cap so the loop visits every turn.
      // turnsCap also generous so the limit isn't hit early.
      splitAtRecencyBoundary(messages, {
        turns: 1_000_000,
        tokens: 1_000_000,
        tokenCounter,
      });

      // Strictly bounded by messages.length.  No message is counted
      // twice, regardless of how many turns the splitter walks.
      expect(calls).toBeLessThanOrEqual(messages.length);
      expect(calls).toBe(messages.length);
    });

    it('stops counting once the tokens cap is exceeded (no scan past the boundary)', () => {
      const messages: BaseMessage[] = [];
      for (let t = 0; t < 50; t++) {
        messages.push(new HumanMessage(`turn ${t}`));
        messages.push(new AIMessage(`reply ${t}`));
      }

      let calls = 0;
      const tokenCounter = (): number => {
        calls += 1;
        return 1; // 1 token per message → 100 tokens total
      };

      // Cap of 10 tokens lets us include the last 5 turns (10 messages)
      // before the next turn's 2 tokens would overflow.
      const result = splitAtRecencyBoundary(messages, {
        turns: 1_000,
        tokens: 10,
        tokenCounter,
      });

      // Visited at most: 5 included turns × 2 messages + one over-budget
      // turn × 2 messages (counted then rejected) = 12 messages.  Far
      // less than the full 100.
      expect(calls).toBeLessThanOrEqual(12);
      expect(result.tailTurnCount).toBe(5);
    });
  });

  describe('degenerate inputs', () => {
    it('puts everything in the head when there is no HumanMessage', () => {
      const messages = [
        new SystemMessage('preamble'),
        new AIMessage('starter'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual(messages);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
    });

    it('handles a HumanMessage at index 0 with prior non-human messages absent', () => {
      const messages = [new HumanMessage('only')];
      const result = splitAtRecencyBoundary(messages, { turns: 3 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
    });

    it('handles tool messages as the very last messages', () => {
      const messages = [
        new HumanMessage('q1'),
        new AIMessage('a1'),
        new HumanMessage('q2'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'c1', name: 't', args: {} }],
        }),
        new ToolMessage({ content: 'r', tool_call_id: 'c1', name: 't' }),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 1 });
      // Most recent turn includes the trailing tool result.
      expect(result.tail).toEqual(messages.slice(2));
      expect(result.head).toEqual(messages.slice(0, 2));
    });
  });
});

describe('pruned windows that begin inside a tool-running turn', () => {
  const units = (native: boolean): BaseMessage[] => Array.from({length: 4}, (_, i) => native ? [
    new AIMessage({content: [{type: 'tool_use', id: `call-${i}`, name: 'search', input: {}}]}),
    new HumanMessage({content: [{type: 'tool_result', tool_use_id: `call-${i}`, content: `result ${i}`}]}),
  ] : [
    new AIMessage({content: '', tool_calls: [{id: `call-${i}`, name: 'search', args: {}}]}),
    new ToolMessage({content: `result ${i}`, tool_call_id: `call-${i}`}),
  ]).flat();

  it.each([false, true])('retains a bounded, paired tool tail without a user message (native=%s)', (native) => {
    const messages = units(native);
    const split = splitAtRecencyBoundary(messages, {turns: 2, tokens: 30, tokenCounter: () => 10});
    expect(split.head).toEqual(messages.slice(0, -2));
    expect(split.tail).toEqual(messages.slice(-2));
    expect(split.usedIntraTurnFallback).toBe(true);
  });

  it('retains an entire indivisible newest exchange when its budget is smaller', () => {
    const messages = units(false).slice(-2);
    const split = splitAtRecencyBoundary(messages, {tokens: 5, tokenCounter: () => 10});
    expect(split.tail).toEqual(messages);
  });

  it('honors an explicitly disabled recent tail', () => {
    const messages = units(false);
    const split = splitAtRecencyBoundary(messages, {turns: 0, tokens: 30, tokenCounter: () => 10});
    expect(split.head).toBe(messages);
    expect(split.tail).toEqual([]);
  });
});

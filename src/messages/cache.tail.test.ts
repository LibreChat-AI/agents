import { convertMessagesToCompletionsMessageParams } from '@langchain/openai';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { ContentBlock } from '@aws-sdk/client-bedrock-runtime';
import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicMessages } from '@/types/messages';
import {
  projectOpenAIChatToolMessageContent,
  projectOpenAIToolMessageContent,
  projectOpenAIResponsesToolMessageContent,
  projectOpenRouterToolMessageContent,
  projectComputerCallOutputsToText,
} from './core';
import {
  addTailCacheControl,
  addBedrockTailCacheControl,
  getBedrockMessageCachePointBudget,
} from './cache';
import { convertToConverseMessages } from '@/llm/bedrock/utils';
import { toLangChainContent } from './langchain';

type CacheControlBlock = MessageContentComplex & {
  cache_control?: { type: 'ephemeral'; ttl?: '1h' };
};

/** Count every block across all messages that carries a cache_control marker. */
function countCacheMarkers(
  messages: ReadonlyArray<{ content: unknown }>
): number {
  let count = 0;
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (block && typeof block === 'object' && 'cache_control' in block) {
        count++;
      }
    }
  }
  return count;
}

function blocksOf(message: { content: unknown }): CacheControlBlock[] {
  return message.content as CacheControlBlock[];
}

describe('addTailCacheControl (single tail breakpoint)', () => {
  test('places exactly one marker on the last message', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Hi there' }] },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Doing well' }] },
      { role: 'user', content: [{ type: 'text', text: 'Great!' }] },
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(
      (result[4].content[0] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
    expect(result[2].content[0]).not.toHaveProperty('cache_control');
  });

  test('anchors on a trailing tool_result block (tail is a tool turn)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Run the tool'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Calling it' },
          { type: 'tool_use', id: 't1', name: 'search', input: {} },
        ] as MessageContentComplex[]),
        tool_calls: [{ id: 't1', name: 'search', args: {} }],
      }),
      new ToolMessage({
        tool_call_id: 't1',
        content: toLangChainContent([
          {
            type: 'tool_result',
            tool_use_id: 't1',
            content: 'result body',
          },
        ] as MessageContentComplex[]),
      }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[2])[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('skips native computer outputs and removes stale screenshot markers', () => {
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            call_id: 'call_computer_cache',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_computer_cache',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const canonical = projectOpenAIToolMessageContent([
      new HumanMessage('Take a screenshot'),
      computerCall,
      computerOutput,
    ]);

    const cached = addTailCacheControl(canonical);

    expect(countCacheMarkers(cached)).toBe(1);
    expect(blocksOf(cached[0])[0].cache_control).toEqual({
      type: 'ephemeral',
    });
    expect(blocksOf(cached[2])[0]).not.toHaveProperty('cache_control');
    expect(() =>
      projectOpenAIResponsesToolMessageContent(cached)
    ).not.toThrow();

    const canonicalOutput = canonical[2] as ToolMessage;
    const staleOutput = new ToolMessage({
      content: toLangChainContent([
        {
          ...(canonicalOutput.content[0] as MessageContentComplex),
          cache_control: { type: 'ephemeral' },
        },
      ]),
      tool_call_id: canonicalOutput.tool_call_id,
      additional_kwargs: canonicalOutput.additional_kwargs,
    });
    const cleaned = addTailCacheControl([
      canonical[0],
      canonical[1],
      staleOutput,
    ]);

    expect(countCacheMarkers(cleaned)).toBe(1);
    expect(blocksOf(cleaned[2])[0]).not.toHaveProperty('cache_control');
    expect(blocksOf(staleOutput)[0]).toHaveProperty('cache_control');
    expect(() =>
      projectOpenAIResponsesToolMessageContent(cleaned)
    ).not.toThrow();
  });

  test('keeps a cached tool result structured through Chat projection', () => {
    const toolCall = new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_cached_tool', name: 'search', args: {} }],
    });
    const toolOutput = new ToolMessage({
      content: 'result body',
      tool_call_id: 'call_cached_tool',
    });
    const graphProjected = projectOpenAIToolMessageContent([
      new HumanMessage('Search'),
      toolCall,
      toolOutput,
    ]);
    const cached = addTailCacheControl(graphProjected);
    const attemptProjected = projectOpenRouterToolMessageContent(cached);
    const payload = convertMessagesToCompletionsMessageParams({
      messages: attemptProjected,
      model: 'gpt-4o',
    });

    expect(blocksOf(cached[2])[0]).toMatchObject({
      type: 'text',
      text: 'result body',
      cache_control: { type: 'ephemeral' },
    });
    expect((attemptProjected[2] as ToolMessage).content).toEqual(
      (cached[2] as ToolMessage).content
    );
    expect(payload[2]).toMatchObject({
      role: 'tool',
      tool_call_id: 'call_cached_tool',
      content: [
        {
          type: 'text',
          text: 'result body',
          cache_control: { type: 'ephemeral' },
        },
      ],
    });

    const fallbackProjected = projectOpenAIChatToolMessageContent(cached);
    const fallbackPayload = convertMessagesToCompletionsMessageParams({
      messages: fallbackProjected,
      model: 'gpt-4o',
    });

    expect((fallbackProjected[2] as ToolMessage).content).toBe('result body');
    expect(fallbackPayload[2]).toMatchObject({
      role: 'tool',
      tool_call_id: 'call_cached_tool',
      content: 'result body',
    });
    expect(JSON.stringify(fallbackPayload[2])).not.toContain('cache_control');

    expect(
      typeof (projectOpenAIToolMessageContent(cached)[2] as ToolMessage).content
    ).toBe('string');
  });

  test('does not stringify noncanonical cache metadata into tool output', () => {
    const messages = addTailCacheControl<BaseMessage>([
      new HumanMessage('Search'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_nested_cache', name: 'search', args: {} }],
      }),
      new ToolMessage({
        content: toLangChainContent([
          {
            type: 'tool_result',
            tool_use_id: 'call_nested_cache',
            content: 'result body',
          },
        ] as MessageContentComplex[]),
        tool_call_id: 'call_nested_cache',
      }),
    ]);

    const projected = projectOpenRouterToolMessageContent(messages);
    const content = (projected[2] as ToolMessage).content;

    expect(typeof content).toBe('string');
    expect(content).toContain('result body');
    expect(content).not.toContain('cache_control');
    expect(JSON.stringify(messages[2].content)).toContain('cache_control');
  });

  test('strips ALL stale markers and re-anchors a single one at the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'old marker',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'another old marker',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new AIMessage({ content: 'reply' }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[2])[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocksOf(result[0])[0]).not.toHaveProperty('cache_control');
    expect(blocksOf(result[1])[0]).not.toHaveProperty('cache_control');
  });

  test('does not anchor on thinking blocks', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hi'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'thought through it' },
          { type: 'thinking', thinking: 'secret reasoning' },
        ] as MessageContentComplex[]),
      }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[1])[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocksOf(result[1])[1]).not.toHaveProperty('cache_control');
  });

  test.each(['reasoning_content', 'reasoning', 'think'])(
    'does not anchor on a trailing foreign reasoning block (%s)',
    (reasoningType) => {
      // Foreign reasoning (Bedrock/Google/LibreChat) is dropped by the
      // Anthropic converter on assistant turns; anchoring the only breakpoint
      // there would silently lose tail caching on a cross-provider handoff.
      const messages: BaseMessage[] = [
        new HumanMessage('Hi'),
        new AIMessage({
          content: toLangChainContent([
            { type: 'text', text: 'Here is my answer.' },
            { type: reasoningType, text: 'foreign reasoning' },
          ] as MessageContentComplex[]),
        }),
      ];

      const result = addTailCacheControl(messages);

      expect(countCacheMarkers(result)).toBe(1);
      // Marker must land on the surviving text block, not the reasoning block.
      expect(blocksOf(result[1])[0].cache_control).toEqual({
        type: 'ephemeral',
      });
      expect(blocksOf(result[1])[1]).not.toHaveProperty('cache_control');
    }
  );

  test('skips synthetic meta tail and anchors on the previous real message', () => {
    const realTail = new AIMessage({ content: 'real answer' });
    const metaTail = new HumanMessage({ content: 'reinjected skill body' });
    (
      metaTail as unknown as { additional_kwargs: Record<string, unknown> }
    ).additional_kwargs = { isMeta: true };

    const result = addTailCacheControl([
      new HumanMessage({ content: 'question' }),
      realTail,
      metaTail,
    ]);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[1])[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('handles string content on the tail', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Final' },
    ];

    const result = addTailCacheControl(messages);

    expect(result[0].content).toBe('Hello');
    expect(result[1].content[0]).toEqual({
      type: 'text',
      text: 'Final',
      cache_control: { type: 'ephemeral' },
    });
  });

  test('does not mutate the original messages', () => {
    const original: AnthropicMessages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'World' }] },
    ];

    addTailCacheControl(original);

    expect(original[1].content[0]).not.toHaveProperty('cache_control');
  });

  test('returns input unchanged for empty array', () => {
    const messages: AnthropicMessages = [];
    expect(addTailCacheControl(messages)).toEqual([]);
  });
});

/** Count every Bedrock cachePoint block across all messages. */
function countCachePoints(
  messages: ReadonlyArray<{ content: unknown }>
): number {
  let count = 0;
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (block && typeof block === 'object' && 'cachePoint' in block) {
        count++;
      }
    }
  }
  return count;
}

describe('addBedrockTailCacheControl (single tail cachePoint)', () => {
  test('inserts exactly one cachePoint after the last text block of the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('First question'),
      new AIMessage('First answer'),
      new HumanMessage('Second question'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(countCachePoints(result)).toBe(1);
    const tail = blocksOf(result[2]);
    expect(tail[tail.length - 1]).toEqual({ cachePoint: { type: 'default' } });
  });

  test('strips stale cachePoints and re-anchors a single one at the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage({
        content: toLangChainContent([
          { type: 'text', text: 'old' },
          { cachePoint: { type: 'default' } },
        ] as MessageContentComplex[]),
      }),
      new AIMessage('reply'),
      new HumanMessage('newest'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(countCachePoints(result)).toBe(1);
    const tail = blocksOf(result[2]);
    expect(tail[tail.length - 1]).toEqual({ cachePoint: { type: 'default' } });
    expect(blocksOf(result[0]).some((b) => 'cachePoint' in b)).toBe(false);
  });

  test('strips Anthropic cache_control from a system message but never anchors it', () => {
    const messages: BaseMessage[] = [
      new SystemMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'system rules',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new HumanMessage('hi'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(blocksOf(result[0])[0]).not.toHaveProperty('cache_control');
    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[1])[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  test('skips synthetic meta tail and anchors on the previous real message', () => {
    const metaTail = new HumanMessage({ content: 'reinjected skill body' });
    (
      metaTail as unknown as { additional_kwargs: Record<string, unknown> }
    ).additional_kwargs = { source: 'skill' };

    const result = addBedrockTailCacheControl([
      new HumanMessage('question'),
      new AIMessage('real answer'),
      metaTail,
    ]);

    expect(countCachePoints(result)).toBe(1);
    const realTail = blocksOf(result[1]);
    expect(realTail[realTail.length - 1]).toEqual({
      cachePoint: { type: 'default' },
    });
  });

  test('handles string content on the tail', () => {
    const result = addBedrockTailCacheControl([
      new HumanMessage('Hello'),
      new AIMessage('Final'),
    ]);

    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[1])).toEqual([
      { type: 'text', text: 'Final' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  test('anchors on a trailing string tool result (agent-loop tail)', () => {
    const result = addBedrockTailCacheControl([
      new HumanMessage('Run the tool'),
      new AIMessage({
        content: 'Calling it',
        tool_calls: [{ id: 't1', name: 'search', args: {} }],
      }),
      new ToolMessage({ tool_call_id: 't1', content: 'result body' }),
    ]);

    // The single cachePoint must land on the trailing tool result so the
    // tool output is part of the cached prefix; the converter later hoists it
    // out of toolResult.content (see toolResultCachePoint.test.ts).
    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[2])).toEqual([
      { type: 'text', text: 'result body' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  test('keeps the cachePoint before an omitted computer screenshot', () => {
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_bedrock_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const cached = addBedrockTailCacheControl([
      new HumanMessage('Take a screenshot'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_bedrock_computer',
            name: 'computer',
            args: { action: 'screenshot' },
          },
        ],
      }),
      computerOutput,
    ]);

    expect(countCachePoints(cached)).toBe(1);
    expect(blocksOf(cached[0])[1]).toEqual({
      cachePoint: { type: 'default' },
    });
    expect(cached[2].content).toBe(computerOutput.content);

    const projected = projectComputerCallOutputsToText(cached);
    const converse = convertToConverseMessages(projected);
    expect((projected[2] as ToolMessage).content).toBe(
      '[Computer screenshot omitted for this provider]'
    );
    expect(JSON.stringify(converse).match(/"cachePoint"/g)).toHaveLength(1);
    expect(computerOutput.content).toBe('data:image/png;base64,AA==');
  });
});

function toolRound(id: string): BaseMessage[] {
  return [
    new AIMessage({
      content: `Calling ${id}`,
      tool_calls: [{ id, name: 'search', args: { q: id } }],
    }),
    new ToolMessage({ tool_call_id: id, content: `result ${id}` }),
  ];
}

function toolRounds(prefix: string, count: number): BaseMessage[] {
  return Array.from({ length: count }, (_, i) =>
    toolRound(`${prefix}${i}`)
  ).flat();
}

function cachePointIndexes(message: { content: unknown }): number[] {
  if (!Array.isArray(message.content)) {
    return [];
  }
  return (message.content as CacheControlBlock[]).flatMap((block, index) =>
    'cachePoint' in block ? [index] : []
  );
}

describe('addBedrockTailCacheControl with a second message cachePoint', () => {
  test('anchors the tail and the latest human message before it', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Turn one'),
      ...toolRounds('a', 3),
      new AIMessage('Turn one answer'),
      new HumanMessage('Turn two'),
    ];

    const result = addBedrockTailCacheControl(messages, undefined, 2);

    expect(countCachePoints(result)).toBe(2);
    expect(blocksOf(result[0])).toEqual([
      { type: 'text', text: 'Turn one' },
      { cachePoint: { type: 'default' } },
    ]);
    expect(blocksOf(result[result.length - 1])).toEqual([
      { type: 'text', text: 'Turn two' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  test('anchors the opening human message during a tool loop', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Run the tools'),
      ...toolRounds('a', 2),
    ];

    const result = addBedrockTailCacheControl(messages, undefined, 2);

    expect(countCachePoints(result)).toBe(2);
    expect(cachePointIndexes(result[0])).toEqual([1]);
    expect(cachePointIndexes(result[result.length - 1])).toEqual([1]);
  });

  test('keeps a single tail cachePoint by default', () => {
    const result = addBedrockTailCacheControl([
      new HumanMessage('Turn one'),
      new AIMessage('answer'),
      new HumanMessage('Turn two'),
    ]);

    expect(countCachePoints(result)).toBe(1);
    expect(cachePointIndexes(result[0])).toEqual([]);
    expect(cachePointIndexes(result[2])).toEqual([1]);
  });

  test('places only the tail when no earlier human message exists', () => {
    const result = addBedrockTailCacheControl(
      [new HumanMessage('Only question')],
      undefined,
      2
    );

    expect(countCachePoints(result)).toBe(1);
  });

  test('never anchors the second cachePoint on tool results, assistant turns or synthetic messages', () => {
    const meta = new HumanMessage({ content: 'reinjected skill body' });
    (
      meta as unknown as { additional_kwargs: Record<string, unknown> }
    ).additional_kwargs = { source: 'skill' };
    const messages: BaseMessage[] = [
      new HumanMessage('Real question'),
      meta,
      ...toolRounds('a', 2),
    ];

    const result = addBedrockTailCacheControl(messages, undefined, 2);

    expect(countCachePoints(result)).toBe(2);
    expect(cachePointIndexes(result[0])).toEqual([1]);
    expect(cachePointIndexes(result[1])).toEqual([]);
    expect(cachePointIndexes(result[2])).toEqual([]);
    expect(cachePointIndexes(result[3])).toEqual([]);
    expect(cachePointIndexes(result[4])).toEqual([]);
    expect(cachePointIndexes(result[5])).toEqual([1]);
  });

  test('strips stale cachePoints on older messages while placing two', () => {
    const stale = new HumanMessage({
      content: toLangChainContent([
        { type: 'text', text: 'oldest' },
        { cachePoint: { type: 'default' } },
      ] as MessageContentComplex[]),
    });

    const result = addBedrockTailCacheControl(
      [
        stale,
        new AIMessage('reply'),
        new HumanMessage('middle'),
        new AIMessage('reply'),
        new HumanMessage('newest'),
      ],
      undefined,
      2
    );

    expect(countCachePoints(result)).toBe(2);
    expect(cachePointIndexes(result[0])).toEqual([]);
    expect(cachePointIndexes(result[2])).toEqual([1]);
    expect(cachePointIndexes(result[4])).toEqual([1]);
  });

  test('applies the configured TTL to both cachePoints', () => {
    const result = addBedrockTailCacheControl(
      [new HumanMessage('one'), new AIMessage('a'), new HumanMessage('two')],
      '1h',
      2
    );

    expect(blocksOf(result[0])[1]).toEqual({
      cachePoint: { type: 'default', ttl: '1h' },
    });
    expect(blocksOf(result[2])[1]).toEqual({
      cachePoint: { type: 'default', ttl: '1h' },
    });
  });
});

describe('getBedrockMessageCachePointBudget', () => {
  const systemWith = (cachePoints: number): SystemMessage =>
    new SystemMessage({
      content: toLangChainContent([
        { type: 'text', text: 'stable' },
        ...Array.from({ length: cachePoints }, () => ({
          cachePoint: { type: 'default' },
        })),
        { type: 'text', text: 'dynamic' },
      ] as MessageContentComplex[]),
    });

  test.each([
    [0, 2],
    [1, 2],
    [2, 1],
    [3, 1],
  ])(
    'with %i system cachePoints allows %i message cachePoints',
    (system, expected) => {
      expect(
        getBedrockMessageCachePointBudget([
          systemWith(system),
          new HumanMessage('q'),
        ])
      ).toBe(expected);
    }
  );

  test('allows two without a system message or with a string system message', () => {
    expect(getBedrockMessageCachePointBudget([new HumanMessage('q')])).toBe(2);
    expect(
      getBedrockMessageCachePointBudget([
        new SystemMessage('rules'),
        new HumanMessage('q'),
      ])
    ).toBe(2);
    expect(getBedrockMessageCachePointBudget([])).toBe(2);
  });

  test('never exceeds four cachePoints with the system and a tool checkpoint', () => {
    for (const systemPoints of [0, 1, 2]) {
      const messages: BaseMessage[] = [
        systemWith(systemPoints),
        new HumanMessage('one'),
        ...toolRounds('a', 2),
        new HumanMessage('two'),
      ];
      const cached = addBedrockTailCacheControl(
        messages,
        undefined,
        getBedrockMessageCachePointBudget(messages)
      );
      const toolCheckpoint = 1;
      expect(countCachePoints(cached) + toolCheckpoint).toBeLessThanOrEqual(4);
    }
  });
});

describe('Bedrock lookback from the message cachePoints (Converse payload)', () => {
  /** Bedrock finds an earlier entry at most this many added blocks back (21 hits, 22 misses). */
  const MAX_BLOCKS_ADDED = 21;

  type Flat = { blocks: ContentBlock[]; points: number[] };

  function flattenConverse(messages: BaseMessage[]): Flat {
    const { converseMessages } = convertToConverseMessages(messages);
    const blocks: ContentBlock[] = [];
    const points: number[] = [];
    for (const message of converseMessages) {
      for (const block of (message.content ?? []) as ContentBlock[]) {
        if ('cachePoint' in block) {
          points.push(blocks.length);
          continue;
        }
        blocks.push(block);
      }
    }
    return { blocks, points };
  }

  /** Assistant step of a tool round; `live` adds a block that a rebuilt history drops (e.g. unreplayed reasoning). */
  function assistantStep(id: string, live: boolean): AIMessage {
    const text = { type: 'text', text: `Calling ${id}` };
    const extra = { type: 'text', text: `notes for ${id}` };
    return new AIMessage({
      content: toLangChainContent(
        (live ? [extra, text] : [text]) as MessageContentComplex[]
      ),
      tool_calls: [{ id, name: 'search', args: { q: id } }],
    });
  }

  function turnOne(rounds: number, live: boolean): BaseMessage[] {
    const conv: BaseMessage[] = [new HumanMessage('Turn one question')];
    for (let i = 0; i < rounds; i++) {
      conv.push(assistantStep(`a${i}`, live));
      conv.push(
        new ToolMessage({ tool_call_id: `a${i}`, content: `result a${i}` })
      );
    }
    return conv;
  }

  /** Prefixes written by every call of turn one (each call caches up to its cachePoints). */
  function cachedEntries(
    rounds: number,
    maxCachePoints: number
  ): ContentBlock[][] {
    const live = turnOne(rounds, true);
    const entries: ContentBlock[][] = [];
    for (let end = 1; end <= live.length; end += 2) {
      const flat = flattenConverse(
        addBedrockTailCacheControl(
          live.slice(0, end),
          undefined,
          maxCachePoints
        )
      );
      for (const point of flat.points) {
        entries.push(flat.blocks.slice(0, point));
      }
    }
    return entries;
  }

  /** Whether any cachePoint in `request` reaches an identical earlier entry within the lookback. */
  function readsCachedPrefix(
    request: Flat,
    entries: ContentBlock[][]
  ): boolean {
    return request.points.some((point) =>
      entries.some(
        (entry) =>
          entry.length <= point &&
          point - entry.length <= MAX_BLOCKS_ADDED &&
          JSON.stringify(entry) ===
            JSON.stringify(request.blocks.slice(0, entry.length))
      )
    );
  }

  function turnTwo(rounds: number, live: boolean): BaseMessage[] {
    return [
      ...turnOne(rounds, live),
      new AIMessage('Turn one answer'),
      new HumanMessage('Turn two question'),
    ];
  }

  test('a rebuilt history reaches the previous turn only through the second cachePoint', () => {
    const rounds = 12;
    const rebuilt = turnTwo(rounds, false);

    const tailOnly = flattenConverse(addBedrockTailCacheControl(rebuilt));
    const withSecond = flattenConverse(
      addBedrockTailCacheControl(rebuilt, undefined, 2)
    );

    expect(readsCachedPrefix(tailOnly, cachedEntries(rounds, 1))).toBe(false);
    expect(readsCachedPrefix(withSecond, cachedEntries(rounds, 2))).toBe(true);
  });

  test('with an unchanged history the tail alone already reaches the previous turn', () => {
    const rounds = 12;
    const unchanged = flattenConverse(
      addBedrockTailCacheControl(turnTwo(rounds, true))
    );

    expect(readsCachedPrefix(unchanged, cachedEntries(rounds, 1))).toBe(true);
  });

  test('a short rebuilt turn stays within reach of the tail', () => {
    const rounds = 3;
    const tailOnly = flattenConverse(
      addBedrockTailCacheControl(turnTwo(rounds, false))
    );

    expect(readsCachedPrefix(tailOnly, cachedEntries(rounds, 1))).toBe(true);
  });
});

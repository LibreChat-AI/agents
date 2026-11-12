import {
  OPENAI_RESPONSES_ORIGIN_KEY,
  gateResponsesProviderOutput,
} from './responsesToolRounds';
import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { formatAgentMessages } from './format';
import { Constants, ContentTypes } from '@/common';

describe('formatAgentMessages with tools parameter', () => {
  it('should process messages normally when tools is not provided', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Hello' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: '{"query":"weather"}',
              output: 'The weather is sunny.',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('123');
  });

  it('restores persisted structured tool output for harness accounting', () => {
    const output = [
      {
        type: ContentTypes.TEXT,
        text: JSON.stringify([
          { id: 1, value: 'first' },
          { id: 2, value: 'second' },
        ]),
      },
    ];
    const payload: TPayload = [
      { role: 'user', content: 'Query the table' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'query-1',
              name: 'run_select_query',
              args: '{}',
              output,
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, { 0: 5, 1: 0 });

    const toolMessage = result.messages[2] as ToolMessage;
    expect(toolMessage).toBeInstanceOf(ToolMessage);
    expect(toolMessage.content).toEqual(output);
    expect(toolMessage.tool_call_id).toBe('query-1');
  });

  it('normalizes opaque persisted tool output before provider dispatch', () => {
    const output = [{ type: 'json', rows: [{ id: 1, value: 'first' }] }];
    const payload: TPayload = [
      { role: 'user', content: 'Query the table' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'query-opaque',
              name: 'run_select_query',
              args: '{}',
              output,
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);
    const toolMessage = result.messages[2] as ToolMessage;

    expect(typeof toolMessage.content).toBe('string');
    expect(toolMessage.content).toBe(JSON.stringify(output));
  });

  it('hard-caps persisted structured output without invoking toJSON', () => {
    let toJSONCalls = 0;
    const output = {
      rows: [{ value: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS + 1_000) }],
      toJSON() {
        toJSONCalls++;
        return 'y'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2);
      },
    };
    const payload: TPayload = [
      { role: 'user', content: 'Query the table' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'query-bounded',
              name: 'run_select_query',
              args: '{}',
              output,
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);
    const content = (result.messages[2] as ToolMessage).content;

    expect(toJSONCalls).toBe(0);
    expect(typeof content).toBe('string');
    expect((content as string).length).toBeLessThanOrEqual(
      HARD_MAX_TOOL_RESULT_CHARS
    );
    expect(content).toContain('truncated');
    expect(content).not.toContain('y'.repeat(1_000));
  });

  it('should filter out all tool calls when tools set is empty', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    // Provide an empty set of allowed tools
    const allowedTools = new Set<string>();

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should filter out the tool call, keeping only text content
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool_calls (they were filtered out)
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(0);
  });

  it('should filter out tool calls not in the allowed set', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should filter out the invalid tool call, keeping text content
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool_calls (check_weather was filtered out)
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(0);
  });

  it('should not convert tool messages when tool is in the allowed set', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    // Provide a set of allowed tools that includes 'check_weather'
    const allowedTools = new Set(['check_weather', 'search']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should keep the original structure
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);
  });

  it('should handle multiple tool calls with mixed allowed/disallowed tools', () => {
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Tell me about the weather and calculate something',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather first.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now let me calculate something for you.',
            tool_call_ids: ['calc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
        ],
      },
    ];

    // Allow calculator but not check_weather
    const allowedTools = new Set(['calculator', 'search']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should keep the valid tool and omit the invalid tool without rewriting its output as text
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have the calculator tool_call
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );

    expect(result.messages[1].content).not.toContain('check_weather');
    expect(result.messages[1].content).not.toContain('Sunny, 75°F');

    // The ToolMessage should be for calculator
    expect((result.messages[2] as ToolMessage).name).toBe('calculator');
    expect(result.messages[2].content).toBe('2');
  });

  it('should update indexTokenCountMap correctly when converting tool messages', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 10, // 10 tokens for user message
      1: 40, // 40 tokens for assistant message with tool call
    };

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(
      payload,
      indexTokenCountMap,
      allowedTools
    );

    // Should have 2 messages and 2 entries in the token count map
    expect(result.messages).toHaveLength(2);
    expect(Object.keys(result.indexTokenCountMap || {}).length).toBe(2);

    // User message token count should be unchanged
    expect(result.indexTokenCountMap?.[0]).toBe(10);

    // All assistant message tokens should be assigned to the single AIMessage
    expect(result.indexTokenCountMap?.[1]).toBe(40);
  });

  it('omits an invalid tool and its output when no other content exists', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    expect(result.messages).toHaveLength(0);
  });

  it.each([
    ['subagent', Constants.SUBAGENT],
    ['handoff', `${Constants.LC_TRANSFER_TO_}researcher`],
    ['conditional handoff', 'conditional_transfer'],
  ])('preserves an SDK-managed %s call as structured history', (_, toolName) => {
    const payload: TPayload = [
      { role: 'user', content: 'Delegate this task' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'managed_1',
              name: toolName,
              args: '{}',
              output: 'completed',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, new Set());

    expect(result.messages).toHaveLength(3);
    expect((result.messages[1] as AIMessage).tool_calls).toEqual([
      expect.objectContaining({ id: 'managed_1', name: toolName }),
    ]);
    expect(result.messages[1].content).not.toContain(`Tool: ${toolName}`);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);
    expect((result.messages[2] as ToolMessage).name).toBe(toolName);
    expect(result.messages[2].content).toBe('completed');
  });

  it('should handle complex sequences with multiple tool calls', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Help me with a complex task' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for information first.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'search',
              args: '{"query":"complex task"}',
              output: 'Found information about complex tasks.',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now I\'ll check the weather.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Finally, I\'ll calculate something.',
            tool_call_ids: ['calc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: 'Here\'s your answer based on all that information.',
      },
    ];

    // Allow search and calculator but not check_weather
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // With selective filtering, valid tools are kept and invalid tools are omitted
    // 1. HumanMessage
    // 2. AIMessage (search tool_call)
    // 3. ToolMessage (search result)
    // 4. AIMessage (text only; invalid weather tool omitted)
    // 5. AIMessage (calculator tool_call)
    // 6. ToolMessage (calculator result)
    // 7. AIMessage (final text)
    expect(result.messages).toHaveLength(7);

    // Check the types of messages
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage); // Search message
    expect(result.messages[2]).toBeInstanceOf(ToolMessage); // Search tool response
    expect(result.messages[3]).toBeInstanceOf(AIMessage); // Weather text
    expect(result.messages[4]).toBeInstanceOf(AIMessage); // Calculator message
    expect(result.messages[5]).toBeInstanceOf(ToolMessage); // Calculator tool response
    expect(result.messages[6]).toBeInstanceOf(AIMessage); // Final message

    // Check that search tool was kept
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );

    // Check that weather output is not rewritten as assistant-authored text
    expect((result.messages[3] as AIMessage).tool_calls).toHaveLength(0);
    const weatherContent = result.messages[3].content;
    const weatherContentStr =
      typeof weatherContent === 'string'
        ? weatherContent
        : JSON.stringify(weatherContent);
    expect(weatherContentStr).not.toContain('check_weather');
    expect(weatherContentStr).not.toContain('Sunny');

    // Check that calculator tool was kept
    expect((result.messages[4] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[4] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );
  });

  describe('restored tool rounds match the live request', () => {
    const toolCall = (id: string, extra: Record<string, unknown> = {}) => ({
      type: ContentTypes.TOOL_CALL,
      tool_call: { id, name: 'lookup', args: '{}', output: 'ok', ...extra },
    });

    it('keeps sequential rounds as separate assistant messages', () => {
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', { roundId: 'r1' }),
            toolCall('b', { roundId: 'r1' }),
            toolCall('c', { roundId: 'r2' }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const rounds = messages.filter(
        (m) => m instanceof AIMessage
      ) as AIMessage[];

      expect(rounds.map((m) => m.tool_calls?.map((c) => c.id))).toEqual([
        ['a', 'b'],
        ['c'],
      ]);
    });

    it('puts stored Responses items back on the assistant message', () => {
      const items = [
        { id: 'rs_1', type: 'reasoning', summary: [] },
        {
          id: 'fc_1',
          type: 'function_call',
          status: 'completed',
          arguments: '{}',
          call_id: 'a',
          name: 'lookup',
        },
      ];
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', {
              providerItems: { origin: 'o', created_at: 1, items },
            }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const round = messages.find((m) => m instanceof AIMessage) as AIMessage;

      expect(round.response_metadata.output).toEqual(items);
      expect(round.tool_calls).toHaveLength(1);
    });

    it('replays a stored call in the form its part carries', () => {
      const call = (name: string, args: string) => ({
        id: 'fc_1',
        type: 'function_call',
        status: 'completed',
        arguments: args,
        call_id: 'a',
        name,
      });
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', {
              name: 'wrapper',
              args: '{"name":"lookup"}',
              providerItems: {
                origin: 'o',
                created_at: 1,
                items: [call('lookup', '{}')],
              },
            }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const round = messages.find((m) => m instanceof AIMessage) as AIMessage;

      expect(round.response_metadata.output).toEqual([
        call('wrapper', '{"name":"lookup"}'),
      ]);
    });

    it('keeps the stored call when the part args did not survive storage', () => {
      const item = {
        id: 'fc_1',
        type: 'function_call',
        status: 'completed',
        arguments: '{}',
        call_id: 'a',
        name: 'lookup',
      };
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', {
              name: 'wrapper',
              args: '{"name":"look',
              providerItems: { origin: 'o', created_at: 1, items: [item] },
            }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const round = messages.find((m) => m instanceof AIMessage) as AIMessage;

      expect(round.response_metadata.output).toEqual([item]);
    });

    it('restores no items when one call carries more than the size cap', () => {
      const item = (callId: string, reasoning: string) => [
        {
          id: `rs_${callId}`,
          type: 'reasoning',
          summary: [],
          encrypted_content: reasoning,
        },
        {
          id: `fc_${callId}`,
          type: 'function_call',
          status: 'completed',
          arguments: '{}',
          call_id: callId,
          name: 'lookup',
        },
      ];
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', {
              roundId: 'r1',
              providerItems: { origin: 'o', items: item('a', 'ENC') },
            }),
            toolCall('b', {
              roundId: 'r1',
              providerItems: {
                origin: 'o',
                items: item('b', 'x'.repeat(512_000)),
              },
            }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const round = messages.find((m) => m instanceof AIMessage) as AIMessage;

      expect(round.tool_calls).toHaveLength(2);
      expect(round.response_metadata.output).toBeUndefined();
    });

    it('parses stored content blocks back into blocks', () => {
      const blocks = [{ type: 'text', text: 'one' }];
      const payload: TPayload = [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: [
            toolCall('a', {
              output: JSON.stringify(blocks),
              outputFormat: 'blocks',
            }),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload);
      const result = messages.find((m) => m instanceof ToolMessage);

      expect(result?.content).toEqual(blocks);
    });
  });

  describe('Responses replay gate', () => {
    const now = 1_000 + 31 * 24 * 60 * 60;
    const output = [
      { id: 'rs_ref', type: 'reasoning', summary: [] },
      { id: 'rs_enc', type: 'reasoning', summary: [], encrypted_content: 'E' },
      {
        id: 'fc_1',
        type: 'function_call',
        status: 'completed',
        arguments: '{}',
        call_id: 'call_1',
        name: 'lookup',
      },
    ];
    const toolRound = (createdAt: number): AIMessage =>
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_1', name: 'lookup', args: {} }],
        response_metadata: {
          output,
          created_at: createdAt,
          [OPENAI_RESPONSES_ORIGIN_KEY]: 'origin-a',
        },
      });
    const gate = (
      message: AIMessage,
      options: { origin?: string; zdr?: boolean; at?: number } = {}
    ): AIMessage =>
      gateResponsesProviderOutput(
        [message],
        options.origin ?? 'origin-a',
        options.zdr ?? false,
        options.at ?? now
      )[0] as AIMessage;
    const outputIds = (message: AIMessage): unknown =>
      (message.response_metadata.output as Array<{ id: string }> | undefined)?.map(
        (item) => item.id
      );

    it('keeps the output for its origin and drops it for another origin or under ZDR', () => {
      const own = toolRound(now);
      expect(gate(own)).toBe(own);

      for (const gated of [
        gate(toolRound(now), { origin: 'origin-b' }),
        gate(toolRound(now), { zdr: true }),
      ]) {
        expect(gated.response_metadata.output).toBeUndefined();
        expect(
          gated.response_metadata[OPENAI_RESPONSES_ORIGIN_KEY]
        ).toBeUndefined();
        expect(gated.tool_calls).toHaveLength(1);
        expect(gated.content).toEqual([]);
      }
    });

    it('drops reasoning references older than 29 days and keeps encrypted ones', () => {
      const limit = 29 * 24 * 60 * 60;

      expect(outputIds(gate(toolRound(1_000), { at: 1_000 + limit }))).toEqual(
        ['rs_ref', 'rs_enc', 'fc_1']
      );
      expect(
        outputIds(gate(toolRound(1_000), { at: 1_000 + limit + 1 }))
      ).toEqual(['rs_enc', 'fc_1']);
    });
  });
});

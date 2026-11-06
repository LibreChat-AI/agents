import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { annotateMessagesForLLM, ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { formatAgentMessages } from './format';
import { createContentAggregator } from '@/stream';
import { Constants, ContentTypes, GraphEvents } from '@/common';

describe('formatAgentMessages tool output reference labels', () => {
  it('restores the stored key and unresolved refs, labelled as live', () => {
    const payload: TPayload = [
      { role: 'user', content: 'go' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'c1',
              name: 'echo',
              args: '{}',
              output: 'bad',
              ref: 'tool1turn3',
              unresolved_refs: ['tool0turn0'],
            },
          },
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);
    const restored = messages.find((m) => m instanceof ToolMessage)!;
    const [wire] = annotateMessagesForLLM(
      [restored],
      new ToolOutputReferenceRegistry()
    );

    expect(restored.additional_kwargs).toMatchObject({
      _refKey: 'tool1turn3',
      _unresolvedRefs: ['tool0turn0'],
    });
    expect(wire.content).toBe(
      '[ref: tool1turn3]\nbad\n[unresolved refs: tool0turn0]'
    );
    const call = (messages.find((m) => m instanceof AIMessage) as AIMessage)
      .tool_calls?.[0];
    expect(call).not.toHaveProperty('ref');
  });

  it('keeps the labels the content aggregator stored with the result', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_1',
        stepIndex: 0,
        index: 0,
        type: 'tool_calls',
        stepDetails: {
          type: 'tool_calls',
          tool_calls: [{ id: 'c1', name: 'echo', args: '{}' }],
        },
        usage: null,
      } as never,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step_1',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'c1',
            name: 'echo',
            args: '{}',
            output: 'out',
            progress: 1,
            ref: 'tool0turn2',
            unresolved_refs: ['tool0turn9'],
          },
        },
      } as never,
    });

    expect(contentParts[0]).toMatchObject({
      tool_call: { ref: 'tool0turn2', unresolved_refs: ['tool0turn9'] },
    });
  });
});

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
});

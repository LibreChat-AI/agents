// src/messages/__tests__/tools.test.ts
import {
  AIMessageChunk,
  ToolMessage,
  HumanMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { extractToolDiscoveries, hasToolSearchInCurrentTurn } from '../tools';
import { Constants } from '@/common';

describe('Tool Discovery Functions', () => {
  /**
   * Helper to create an AIMessageChunk with tool calls
   */
  const createAIMessage = (
    content: string,
    toolCalls: Array<{
      id: string;
      name: string;
      args: Record<string, unknown>;
    }>
  ): AIMessageChunk => {
    return new AIMessageChunk({
      content,
      tool_calls: toolCalls.map((tc) => ({
        id: tc.id,
        name: tc.name,
        args: tc.args,
        type: 'tool_call' as const,
      })),
    });
  };

  /**
   * Helper to create a ToolMessage (tool search result)
   */
  const createToolSearchResult = (
    toolCallId: string,
    discoveredTools: string[]
  ): ToolMessage => {
    return new ToolMessage({
      content: `Found ${discoveredTools.length} tools`,
      tool_call_id: toolCallId,
      name: Constants.TOOL_SEARCH,
      artifact: {
        tool_references: discoveredTools.map((name) => ({
          tool_name: name,
          match_score: 0.9,
          matched_field: 'description',
          snippet: 'Test snippet',
        })),
        metadata: {
          total_searched: 10,
          pattern: 'test',
        },
      },
    });
  };

  /**
   * Helper to create a regular ToolMessage (non-search)
   */
  const createRegularToolMessage = (
    toolCallId: string,
    name: string,
    content: string
  ): ToolMessage => {
    return new ToolMessage({
      content,
      tool_call_id: toolCallId,
      name,
    });
  };

  describe('extractToolDiscoveries', () => {
    it('extracts tool names from a skill artifact in the current turn', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Load the proposal skill'),
        createAIMessage('Loading skill...', [
          {
            id: 'skill_call',
            name: Constants.SKILL_TOOL,
            args: { skill: 'dolibarr-proposal' },
          },
        ]),
        new ToolMessage({
          content: 'Loaded skill',
          tool_call_id: 'skill_call',
          name: Constants.SKILL_TOOL,
          artifact: {
            tool_references: [
              { tool_name: 'create_proposal_mcp_dolibarr' },
              { tool_name: 'add_proposal_lines_mcp_dolibarr' },
            ],
          },
        }),
      ];

      expect(extractToolDiscoveries(messages)).toEqual([
        'create_proposal_mcp_dolibarr',
        'add_proposal_lines_mcp_dolibarr',
      ]);
      expect(hasToolSearchInCurrentTurn(messages)).toBe(true);
    });

    it('extracts tool names from a single tool search result', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search for database tools'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'database' },
          },
        ]),
        createToolSearchResult('call_1', ['database_query', 'database_insert']),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual(['database_query', 'database_insert']);
    });

    it('extracts tool names from multiple tool search results in same turn', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search for tools'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'database' },
          },
          {
            id: 'call_2',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'file' },
          },
        ]),
        createToolSearchResult('call_1', ['database_query']),
        createToolSearchResult('call_2', ['file_read', 'file_write']),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual(['database_query', 'file_read', 'file_write']);
    });

    it('returns empty array when no messages', () => {
      const result = extractToolDiscoveries([]);
      expect(result).toEqual([]);
    });

    it('returns empty array when last message is not a ToolMessage', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Hello'),
        createAIMessage('Hi there!', []),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual([]);
    });

    it('returns empty array when no AI message with tool calls found', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Hello'),
        new ToolMessage({
          content: 'Some result',
          tool_call_id: 'orphan_call',
          name: Constants.TOOL_SEARCH,
        }),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual([]);
    });

    it('ignores tool search results from previous turns', () => {
      const messages: BaseMessage[] = [
        // Turn 1: Previous search
        new HumanMessage('Search for old tools'),
        createAIMessage('Searching...', [
          {
            id: 'old_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'old' },
          },
        ]),
        createToolSearchResult('old_call', ['old_tool_1', 'old_tool_2']),
        // Turn 2: Current turn
        new HumanMessage('Search for new tools'),
        createAIMessage('Searching again...', [
          {
            id: 'new_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'new' },
          },
        ]),
        createToolSearchResult('new_call', ['new_tool']),
      ];

      const result = extractToolDiscoveries(messages);

      // Should only return tools from current turn
      expect(result).toEqual(['new_tool']);
    });

    it('ignores non-search tool results', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Do some work'),
        createAIMessage('Working...', [
          {
            id: 'search_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
          { id: 'other_call', name: 'get_weather', args: { city: 'NYC' } },
        ]),
        createToolSearchResult('search_call', ['found_tool']),
        createRegularToolMessage('other_call', 'get_weather', '{"temp": 72}'),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual(['found_tool']);
    });

    it('handles empty tool_references in artifact', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'xyz' },
          },
        ]),
        new ToolMessage({
          content: 'No tools found',
          tool_call_id: 'call_1',
          name: Constants.TOOL_SEARCH,
          artifact: {
            tool_references: [],
            metadata: { total_searched: 10, pattern: 'xyz' },
          },
        }),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual([]);
    });

    it('handles missing artifact', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
        ]),
        new ToolMessage({
          content: 'Error occurred',
          tool_call_id: 'call_1',
          name: Constants.TOOL_SEARCH,
          // No artifact
        }),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual([]);
    });

    it('ignores tool messages with wrong tool_call_id', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
        ]),
        // This has a different tool_call_id that doesn't match the AI message
        createToolSearchResult('wrong_id', ['should_not_appear']),
        createToolSearchResult('call_1', ['correct_tool']),
      ];

      const result = extractToolDiscoveries(messages);

      expect(result).toEqual(['correct_tool']);
    });

    it('only looks at messages after the latest AI parent', () => {
      const messages: BaseMessage[] = [
        // First AI message with search
        createAIMessage('First search', [
          {
            id: 'first_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'first' },
          },
        ]),
        createToolSearchResult('first_call', ['first_tool']),
        // Second AI message - this is the "latest AI parent" for the last tool message
        createAIMessage('Second search', [
          {
            id: 'second_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'second' },
          },
        ]),
        createToolSearchResult('second_call', ['second_tool']),
      ];

      const result = extractToolDiscoveries(messages);

      // Should only find second_tool (from the turn of the latest AI parent)
      expect(result).toEqual(['second_tool']);
    });
  });

  describe('hasToolSearchInCurrentTurn', () => {
    it('returns true when current turn has tool search results', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
        ]),
        createToolSearchResult('call_1', ['found_tool']),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(true);
    });

    it('returns false when no messages', () => {
      const result = hasToolSearchInCurrentTurn([]);
      expect(result).toBe(false);
    });

    it('returns false when last message is not a ToolMessage', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Hello'),
        createAIMessage('Hi!', []),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(false);
    });

    it('returns false when no AI parent found', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Hello'),
        new ToolMessage({
          content: 'Result',
          tool_call_id: 'orphan',
          name: Constants.TOOL_SEARCH,
        }),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(false);
    });

    it('returns false when current turn has no tool search (only other tools)', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Get weather'),
        createAIMessage('Getting weather...', [
          { id: 'call_1', name: 'get_weather', args: { city: 'NYC' } },
        ]),
        createRegularToolMessage('call_1', 'get_weather', '{"temp": 72}'),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(false);
    });

    it('returns true even with mixed tool types in current turn', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search and get weather'),
        createAIMessage('Working...', [
          {
            id: 'search_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
          { id: 'weather_call', name: 'get_weather', args: { city: 'NYC' } },
        ]),
        createRegularToolMessage('weather_call', 'get_weather', '{"temp": 72}'),
        createToolSearchResult('search_call', ['found_tool']),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(true);
    });

    it('returns false for tool search from previous turn', () => {
      const messages: BaseMessage[] = [
        // Previous turn with search
        createAIMessage('Searching...', [
          {
            id: 'old_call',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'old' },
          },
        ]),
        createToolSearchResult('old_call', ['old_tool']),
        // Current turn without search
        new HumanMessage('Get weather now'),
        createAIMessage('Getting weather...', [
          { id: 'weather_call', name: 'get_weather', args: { city: 'NYC' } },
        ]),
        createRegularToolMessage('weather_call', 'get_weather', '{"temp": 72}'),
      ];

      const result = hasToolSearchInCurrentTurn(messages);

      expect(result).toBe(false);
    });
  });

  describe('Integration: extractToolDiscoveries + hasToolSearchInCurrentTurn', () => {
    it('hasToolSearchInCurrentTurn is true when extractToolDiscoveries returns results', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
        ]),
        createToolSearchResult('call_1', ['tool_a', 'tool_b']),
      ];

      const hasSearch = hasToolSearchInCurrentTurn(messages);
      const discoveries = extractToolDiscoveries(messages);

      expect(hasSearch).toBe(true);
      expect(discoveries.length).toBeGreaterThan(0);
    });

    it('both return empty/false for non-search turns', () => {
      const messages: BaseMessage[] = [
        new HumanMessage('Get weather'),
        createAIMessage('Getting...', [
          { id: 'call_1', name: 'get_weather', args: { city: 'NYC' } },
        ]),
        createRegularToolMessage('call_1', 'get_weather', '{"temp": 72}'),
      ];

      const hasSearch = hasToolSearchInCurrentTurn(messages);
      const discoveries = extractToolDiscoveries(messages);

      expect(hasSearch).toBe(false);
      expect(discoveries).toEqual([]);
    });

    it('hasToolSearchInCurrentTurn can be used as quick check before extraction', () => {
      const messagesWithSearch: BaseMessage[] = [
        new HumanMessage('Search'),
        createAIMessage('Searching...', [
          {
            id: 'call_1',
            name: Constants.TOOL_SEARCH,
            args: { pattern: 'test' },
          },
        ]),
        createToolSearchResult('call_1', ['tool_a']),
      ];

      const messagesWithoutSearch: BaseMessage[] = [
        new HumanMessage('Hello'),
        createAIMessage('Hi!', []),
      ];

      // Pattern: quick check first, then extract only if needed
      if (hasToolSearchInCurrentTurn(messagesWithSearch)) {
        const discoveries = extractToolDiscoveries(messagesWithSearch);
        expect(discoveries).toEqual(['tool_a']);
      }

      if (hasToolSearchInCurrentTurn(messagesWithoutSearch)) {
        // This should not execute
        expect(true).toBe(false);
      }
    });
  });
});

/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from './message_inputs';
import { _makeMessageChunkFromAnthropicEvent } from './message_outputs';

type AnthropicTestBlock = {
  id?: unknown;
  type?: unknown;
};

const isServerToolId = (id: unknown): id is string =>
  typeof id === 'string' && id.startsWith('srvtoolu_');

describe('_convertMessagesToAnthropicPayload — server tool use (web search) multi-turn', () => {
  it('corrects tool_use blocks with srvtoolu_ IDs to server_tool_use', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X and Y'),
      new AIMessage({
        content: [
          { type: 'text', text: 'I will search for that.' },
          {
            type: 'tool_use',
            id: 'srvtoolu_1',
            name: 'web_search',
            input: { query: 'X' },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_1',
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com',
                title: 'Result',
                encrypted_content: 'abc',
                page_age: '1d',
              },
            ],
          },
          {
            type: 'tool_use',
            id: 'srvtoolu_2',
            name: 'web_search',
            input: { query: 'Y' },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_2',
            content: [
              {
                type: 'web_search_result',
                url: 'https://example2.com',
                title: 'Result 2',
                encrypted_content: 'def',
                page_age: '2d',
              },
            ],
          },
          { type: 'text', text: 'Here are the results.' },
        ],
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
          {
            id: 'srvtoolu_2',
            name: 'web_search',
            args: { query: 'Y' },
            type: 'tool_call',
          },
        ],
      }),
      new HumanMessage('follow up question'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    expect(messages).toHaveLength(3);

    const assistantContent = messages[1].content as any[];
    const serverToolBlocks = assistantContent.filter(
      (b: any) => b.type === 'server_tool_use'
    );
    const searchResultBlocks = assistantContent.filter(
      (b: any) => b.type === 'web_search_tool_result'
    );
    const regularToolUseBlocks = assistantContent.filter((block: unknown) => {
      const b = block as AnthropicTestBlock;
      return b.type === 'tool_use' && isServerToolId(b.id);
    });

    expect(serverToolBlocks).toHaveLength(2);
    expect(searchResultBlocks).toHaveLength(2);
    expect(regularToolUseBlocks).toHaveLength(0);

    // Verify blocks are clean (no extra streaming properties)
    for (const b of serverToolBlocks) {
      expect(Object.keys(b).sort()).toEqual(
        ['id', 'input', 'name', 'type'].sort()
      );
    }
    for (const b of searchResultBlocks) {
      expect(Object.keys(b).sort()).toEqual(
        ['content', 'tool_use_id', 'type'].sort()
      );
    }
  });

  it('corrects text-typed server tool blocks back to correct types', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X'),
      new AIMessage({
        content: [
          {
            type: 'text',
            id: 'srvtoolu_1',
            name: 'web_search',
            input: '{"query":"X"}',
          },
          {
            type: 'text',
            tool_use_id: 'srvtoolu_1',
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com',
                title: 'Result',
                encrypted_content: 'abc',
                page_age: '1d',
              },
            ],
          },
          { type: 'text', text: 'Found results.' },
        ],
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
        ],
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];

    expect(assistantContent[0]).toEqual({
      type: 'server_tool_use',
      id: 'srvtoolu_1',
      name: 'web_search',
      input: { query: 'X' },
    });
    expect(assistantContent[1].type).toBe('web_search_tool_result');
    expect(assistantContent[1].tool_use_id).toBe('srvtoolu_1');
    expect(assistantContent[2]).toEqual({
      type: 'text',
      text: 'Found results.',
    });
  });

  it('filters server tool calls when content is a string', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X'),
      new AIMessage({
        content: 'I searched and found results.',
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
          {
            id: 'toolu_regular',
            name: 'calculator',
            args: { expr: '2+2' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: '4',
        tool_call_id: 'toolu_regular',
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];

    const toolUseBlocks = assistantContent.filter(
      (b: any) => b.type === 'tool_use'
    );
    expect(toolUseBlocks).toHaveLength(1);
    expect(toolUseBlocks[0].id).toBe('toolu_regular');

    const serverToolBlocks = assistantContent.filter((block: unknown) => {
      const b = block as AnthropicTestBlock;
      return isServerToolId(b.id);
    });
    expect(serverToolBlocks).toHaveLength(0);
  });

  it('handles empty string content with only server tool calls', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
        ],
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];
    expect(assistantContent).toHaveLength(1);
    expect(assistantContent[0].type).toBe('text');
    expect(assistantContent[0].text).toBe('_');
    expect(assistantContent[0].text.trim()).toBe('_');
  });

  it('uses non-whitespace fallback content after filtering empty array text', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage({
        content: [
          { type: 'text', text: ' ' },
          { type: 'text', text: '\n' },
        ],
      }),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    expect(messages[0].content).toEqual([{ type: 'text', text: '_' }]);
  });

  it('uses non-whitespace fallback content for empty server tool result artifacts', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: '',
        tool_call_id: 'srvtoolu_1',
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);

    expect(messages[1].content).toEqual([{ type: 'text', text: '_' }]);
    expect(messages[2].content).toEqual([{ type: 'text', text: '_' }]);
  });

  it('does not throw when a non-string ToolMessage has undefined content', () => {
    const undefinedContentToolMessage = {
      _getType: (): 'tool' => 'tool',
      content: undefined,
      tool_call_id: 'toolu_calc',
    } as unknown as ToolMessage;
    const messageHistory: BaseMessage[] = [
      new HumanMessage('call the calculator'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'toolu_calc',
            name: 'calculator',
            args: { expr: '2+2' },
            type: 'tool_call',
          },
        ],
      }),
      undefinedContentToolMessage,
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    expect(messages[2].content).toEqual([
      {
        type: 'tool_result',
        tool_use_id: 'toolu_calc',
      },
    ]);
  });

  it('preserves regular tool_use blocks alongside corrected server tool blocks', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X and calculate 2+2'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me help.' },
          {
            type: 'tool_use',
            id: 'srvtoolu_1',
            name: 'web_search',
            input: { query: 'X' },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_1',
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com',
                title: 'Result',
                encrypted_content: 'abc',
                page_age: '1d',
              },
            ],
          },
          {
            type: 'tool_use',
            id: 'toolu_calc',
            name: 'calculator',
            input: { expr: '2+2' },
          },
        ],
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
          {
            id: 'toolu_calc',
            name: 'calculator',
            args: { expr: '2+2' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: '4',
        tool_call_id: 'toolu_calc',
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];

    const serverToolUse = assistantContent.filter(
      (b: any) => b.type === 'server_tool_use'
    );
    const webSearchResult = assistantContent.filter(
      (b: any) => b.type === 'web_search_tool_result'
    );
    const regularToolUse = assistantContent.filter((block: unknown) => {
      const b = block as AnthropicTestBlock;
      return b.type === 'tool_use' && !isServerToolId(b.id);
    });

    expect(serverToolUse).toHaveLength(1);
    expect(serverToolUse[0].id).toBe('srvtoolu_1');
    expect(webSearchResult).toHaveLength(1);
    expect(regularToolUse).toHaveLength(1);
    expect(regularToolUse[0].id).toBe('toolu_calc');
  });

  it('filters out empty text blocks from array content', () => {
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search for X'),
      new AIMessage({
        content: [
          { type: 'text', text: '' },
          {
            type: 'server_tool_use',
            id: 'srvtoolu_1',
            name: 'web_search',
            input: { query: 'X' },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_1',
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com',
                title: 'Result',
                encrypted_content: 'abc',
                page_age: '1d',
              },
            ],
          },
          { type: 'text', text: '' },
          { type: 'text', text: 'Here are the results.' },
        ],
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'X' },
            type: 'tool_call',
          },
        ],
      }),
      new HumanMessage('follow up'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];

    const emptyTextBlocks = assistantContent.filter(
      (b: any) => b.type === 'text' && b.text === ''
    );
    expect(emptyTextBlocks).toHaveLength(0);

    const textBlocks = assistantContent.filter((b: any) => b.type === 'text');
    expect(textBlocks).toHaveLength(1);
    expect(textBlocks[0].text).toBe('Here are the results.');
  });

  /**
   * Regression for LibreChat discussion #12806.
   *
   * Anthropic web_search responses can include text blocks whose text is
   * whitespace-only (e.g. ' ', '\n', '\t') alongside server_tool_use and
   * web_search_tool_result blocks. On follow-up turns the API rejects these
   * with: "messages: text content blocks must contain non-whitespace text".
   *
   * The empty-string check alone is insufficient — the filter must drop any
   * text block whose trimmed content is empty.
   */
  it.each([
    ['single space', ' '],
    ['newline', '\n'],
    ['tab', '\t'],
    ['multiple spaces', '   '],
    ['mixed whitespace', ' \n\t '],
  ])(
    'filters whitespace-only text blocks from array content (%s)',
    (_label, whitespace) => {
      const messageHistory: BaseMessage[] = [
        new HumanMessage('search for X'),
        new AIMessage({
          content: [
            { type: 'text', text: whitespace },
            {
              type: 'server_tool_use',
              id: 'srvtoolu_1',
              name: 'web_search',
              input: { query: 'X' },
            },
            {
              type: 'web_search_tool_result',
              tool_use_id: 'srvtoolu_1',
              content: [
                {
                  type: 'web_search_result',
                  url: 'https://example.com',
                  title: 'Result',
                  encrypted_content: 'abc',
                  page_age: '1d',
                },
              ],
            },
            { type: 'text', text: 'Here are the results.' },
          ],
          tool_calls: [
            {
              id: 'srvtoolu_1',
              name: 'web_search',
              args: { query: 'X' },
              type: 'tool_call',
            },
          ],
        }),
        new HumanMessage('follow up'),
      ];

      const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
      const assistantContent = messages[1].content as any[];

      const whitespaceTextBlocks = assistantContent.filter(
        (b: any) =>
          b.type === 'text' &&
          typeof b.text === 'string' &&
          b.text.trim() === ''
      );
      expect(whitespaceTextBlocks).toHaveLength(0);

      const textBlocks = assistantContent.filter((b: any) => b.type === 'text');
      expect(textBlocks).toHaveLength(1);
      expect(textBlocks[0].text).toBe('Here are the results.');
    }
  );

  it('preserves code_execution_tool_result in stream chunks and replays caller/code_execution_tool_result for web_search_20260209', () => {
    const codeExecResultEvent = {
      type: 'content_block_start',
      index: 3,
      content_block: {
        type: 'code_execution_tool_result',
        tool_use_id: 'srvtoolu_code_1',
        content: {
          type: 'code_execution_result',
          stdout: 'filtered results',
          stderr: '',
          return_code: 0,
        },
      },
    } as any;

    const streamed = _makeMessageChunkFromAnthropicEvent(codeExecResultEvent, {
      streamUsage: false,
      coerceContentToString: false,
    });
    expect(streamed).not.toBeNull();
    const streamedBlocks = streamed!.chunk.content as any[];
    expect(streamedBlocks).toHaveLength(1);
    expect(streamedBlocks[0]).toMatchObject({
      type: 'code_execution_tool_result',
      tool_use_id: 'srvtoolu_code_1',
      content: {
        type: 'code_execution_result',
        stdout: 'filtered results',
        stderr: '',
        return_code: 0,
      },
    });

    const caller = {
      type: 'code_execution_20260120',
      tool_id: 'srvtoolu_code_1',
    };
    const messageHistory: BaseMessage[] = [
      new HumanMessage('search with dynamic filtering'),
      new AIMessage({
        content: [
          {
            type: 'server_tool_use',
            id: 'srvtoolu_code_1',
            name: 'code_execution',
            input: { code: 'print(1)' },
            index: 0,
          },
          {
            type: 'server_tool_use',
            id: 'srvtoolu_search_1',
            name: 'web_search',
            input: { query: 'test' },
            caller,
            index: 1,
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_search_1',
            caller,
            index: 2,
            input: undefined,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com',
                title: 'Example',
                encrypted_content: 'enc',
                page_age: '1d',
              },
            ],
          },
          {
            type: 'code_execution_tool_result',
            tool_use_id: 'srvtoolu_code_1',
            index: 3,
            input: undefined,
            content: {
              type: 'code_execution_result',
              stdout: 'filtered results',
              stderr: '',
              return_code: 0,
            },
          },
          { type: 'text', text: 'Done.' },
        ],
      }),
      new HumanMessage('next turn'),
    ];

    const { messages } = _convertMessagesToAnthropicPayload(messageHistory);
    const assistantContent = messages[1].content as any[];

    expect(assistantContent).toHaveLength(5);
    expect(assistantContent[0]).toEqual({
      type: 'server_tool_use',
      id: 'srvtoolu_code_1',
      name: 'code_execution',
      input: { code: 'print(1)' },
    });
    expect(assistantContent[1]).toEqual({
      type: 'server_tool_use',
      id: 'srvtoolu_search_1',
      name: 'web_search',
      input: { query: 'test' },
      caller,
    });
    expect(assistantContent[2]).toEqual({
      type: 'web_search_tool_result',
      tool_use_id: 'srvtoolu_search_1',
      caller,
      content: [
        {
          type: 'web_search_result',
          url: 'https://example.com',
          title: 'Example',
          encrypted_content: 'enc',
          page_age: '1d',
        },
      ],
    });
    expect(assistantContent[3]).toEqual({
      type: 'code_execution_tool_result',
      tool_use_id: 'srvtoolu_code_1',
      content: {
        type: 'code_execution_result',
        stdout: 'filtered results',
        stderr: '',
        return_code: 0,
      },
    });
    expect(assistantContent[4]).toEqual({ type: 'text', text: 'Done.' });
  });
});

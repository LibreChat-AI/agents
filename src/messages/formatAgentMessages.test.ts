import {
  HumanMessage,
  AIMessage,
  AIMessageChunk,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import {
  convertMessagesToCompletionsMessageParams,
  convertResponsesDeltaToChatGenerationChunk,
  convertMessagesToResponsesInput,
  convertResponsesMessageToAIMessage,
} from '@langchain/openai';
import type { BaseMessage } from '@langchain/core/messages';
import type { MessageContentComplex, TPayload } from '@/types';
import {
  OPENAI_RESPONSES_REPLAY_POSITIONS_KEY,
  convertMessagesToContent,
  formatAnthropicArtifactContent,
  formatArtifactPayload,
  projectAnthropicArtifactContent,
  projectArtifactPayload,
  projectComputerCallOutputsToText,
  projectOpenAIToolMessageContent,
  projectOpenAIResponsesToolMessageContent,
  projectToolStreamContentForProvider,
} from './core';
import {
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
} from '@/llm/openai/utils';
import {
  getProviderMessageProvenance,
  setProviderMessageProvenance,
} from './provenance';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { Constants, ContentTypes, Providers } from '@/common';
import { serializeToolContent } from '@/utils/toolContent';
import { formatAgentMessages } from './format';

type AnthropicPayloadBlock = {
  content?: unknown;
  id?: string;
  input?: unknown;
  name?: string;
  text?: string;
  tool_use_id?: string;
  type: string;
};

const CODE_INTERPRETER_REPLAY_EXTRAS = {
  librechatServerToolResult: { toolName: 'code_interpreter' },
} as const;
const IMAGE_GENERATION_REPLAY_EXTRAS = {
  librechatServerToolResult: { toolName: 'image_generation' },
} as const;

const getAnthropicPayloadBlocks = (
  content: unknown
): AnthropicPayloadBlock[] => {
  expect(Array.isArray(content)).toBe(true);
  return content as AnthropicPayloadBlock[];
};

describe('formatAgentMessages', () => {
  describe('empty user content stand-in', () => {
    it('substitutes a stand-in for a user message with empty string content', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi!' },
        { role: 'user', content: '' },
      ];
      const result = formatAgentMessages(payload);
      const emptyTurn = result.messages[result.messages.length - 1];
      expect(emptyTurn).toBeInstanceOf(HumanMessage);
      expect(JSON.stringify(emptyTurn.content)).toContain('(no text)');
    });

    it('substitutes a stand-in when empty text parts are all a user turn has', () => {
      const payload: TPayload = [
        {
          role: 'user',
          content: [
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '' },
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '   ' },
          ],
        },
      ];
      const result = formatAgentMessages(payload);
      expect(result.messages).toHaveLength(1);
      expect(JSON.stringify(result.messages[0].content)).toContain('(no text)');
    });

    it('leaves a vision turn whose only non-empty part is an image alone', () => {
      const content: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: '' },
        {
          type: 'image_url',
          image_url: { url: 'https://example.com/cat.png' },
        },
      ];
      const result = formatAgentMessages([{ role: 'user', content }]);
      expect(result.messages).toHaveLength(1);
      expect(result.messages[0].content).toEqual(content);
    });

    it('does not touch assistant messages with empty content', () => {
      const result = formatAgentMessages([
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: '' },
      ]);
      expect(
        JSON.stringify(result.messages.map((message) => message.role))
      ).not.toContain('(no text)');
    });
  });

  it('should format simple user and AI messages', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages.map((message) => message.role)).toEqual([
      'user',
      'assistant',
    ]);
    expect(Object.keys(result.messages[0])).not.toContain('role');
    expect(Object.keys(result.messages[1])).not.toContain('role');
  });

  it('preserves source messageId correlation with unique formatted ids', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'msg_assistant_1',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
        ],
      },
      { role: 'user', messageId: 'msg_user_1', content: 'thanks' },
    ];

    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(HumanMessage);
    expect(result.messages.map((message) => message.role)).toEqual([
      'assistant',
      'tool',
      'user',
    ]);
    expect(result.messages[0].id).toBe('msg_assistant_1');
    expect(result.messages[1].id).toBeUndefined();
    expect(result.messages[2].id).toBe('msg_user_1');
    expect(
      result.messages.map(
        (message) => message.additional_kwargs.sourceMessageId
      )
    ).toEqual(['msg_assistant_1', 'msg_assistant_1', 'msg_user_1']);
  });

  it('should handle system messages', () => {
    const payload = [
      { role: 'system', content: 'You are a helpful assistant.' },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(SystemMessage);
    expect(result.messages[0].role).toBe('system');
    expect(Object.keys(result.messages[0])).not.toContain('role');
  });

  it('should prepend the latest summary and trim context before its boundary', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Old user message' },
      { role: 'assistant', content: 'Old assistant message' },
      {
        messageId: 'summary-boundary-row',
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Covered by summary' },
          {
            type: ContentTypes.SUMMARY,
            text: 'Conversation summary',
            tokenCount: 12,
          },
          { type: ContentTypes.TEXT, text: 'Preserved tail' },
        ],
      },
      { role: 'user', content: 'Latest user message' },
    ];

    const result = formatAgentMessages(payload, {
      0: 5,
      1: 6,
      2: 18,
      3: 4,
    });

    expect(result.messages).toHaveLength(2);
    expect(result.summary).toBeDefined();
    expect(result.summary!.text).toBe('Conversation summary');
    expect(result.summary!.tokenCount).toBe(12);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(HumanMessage);
    expect(
      (result.messages[0].content as MessageContentComplex[])[0]
    ).toMatchObject({
      type: ContentTypes.TEXT,
      text: 'Preserved tail',
    });
    expect(result.messages[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'summary-boundary-row',
          sourceContentPartIndices: [2],
        },
      ],
    });
    expect(result.indexTokenCountMap?.[0]).toBeLessThan(18);
    expect(result.indexTokenCountMap?.[0]).toBeGreaterThan(0);
    expect(result.indexTokenCountMap?.[1]).toBe(4);
  });

  it('should apply last-summary-wins when multiple summary blocks exist', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.SUMMARY, text: 'Old summary', tokenCount: 3 },
          { type: ContentTypes.TEXT, text: 'Old tail' },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Drop this part' },
          { type: ContentTypes.SUMMARY, text: 'Newest summary', tokenCount: 9 },
          { type: ContentTypes.TEXT, text: 'Keep this part' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.summary).toBeDefined();
    expect(result.summary!.text).toBe('Newest summary');
    expect(result.summary!.tokenCount).toBe(9);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(
      (result.messages[0].content as MessageContentComplex[])[0]
    ).toMatchObject({
      type: ContentTypes.TEXT,
      text: 'Keep this part',
    });
  });

  it('should format messages with content arrays', () => {
    const payload = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello' }],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
  });

  it('should handle tool calls and create ToolMessages', () => {
    const payload = [
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
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('123');
  });

  it('skips persisted Anthropic server tool calls from web search turns', () => {
    const payload: TPayload = [
      {
        role: 'user',
        content:
          'who is the lowest seed survived in 2026 nba playoffs, only the team name, nothing else',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}web_search`,
              name: 'web_search',
              args: '{"query":"2026 NBA playoffs lowest seed survived"}',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Philadelphia 76ers',
          },
        ],
      },
      {
        role: 'user',
        content: 'who are 76ers\' opponents in current series?',
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );

    expect(result.messages).toHaveLength(3);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(
      result.messages.some((message) => message instanceof ToolMessage)
    ).toBe(false);
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(0);
    expect(result.messages[1].content).toEqual([
      {
        type: ContentTypes.TEXT,
        [ContentTypes.TEXT]: 'Philadelphia 76ers',
      },
    ]);
  });

  it('preserves paused Anthropic server tool calls without creating ToolMessages', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`,
              name: 'web_search',
              args: '{"query":"latest Anthropic server tools"}',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(
      result.messages
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(
      result.messages.some((message) => message instanceof ToolMessage)
    ).toBe(false);
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(0);
    expect(result.messages[0].content).toEqual([
      {
        type: 'server_tool_use',
        id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`,
        name: 'web_search',
        input: { query: 'latest Anthropic server tools' },
      },
    ]);
    expect(anthropicPayload.messages[0].content).toEqual([
      {
        type: 'server_tool_use',
        id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`,
        name: 'web_search',
        input: { query: 'latest Anthropic server tools' },
      },
    ]);
  });

  it('keeps srvtoolu tool calls portable for non-Anthropic providers', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`,
              name: 'web_search',
              args: '{"query":"latest Anthropic server tools"}',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.OPENAI }
    );

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[0].content).toBe('');
    expect((result.messages[0] as AIMessage).tool_calls).toEqual([
      {
        id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`,
        name: 'web_search',
        args: { query: 'latest Anthropic server tools' },
      },
    ]);
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe(
      `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}paused`
    );
  });

  it('does not emit empty Anthropic payload content for persisted web search turns', () => {
    const payload: TPayload = [
      {
        role: 'user',
        content:
          'who is the lowest seed survived in 2026 nba playoffs, only the team name, nothing else',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}web_search`,
              name: 'web_search',
              args: '{"query":"2026 NBA playoffs lowest seed survived"}',
            },
          },
          {
            type: ContentTypes.TEXT,
            text: 'Philadelphia 76ers',
          },
        ],
      },
      {
        role: 'user',
        content: 'who are 76ers\' opponents in current series?',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);

    expect(anthropicPayload.messages).toHaveLength(3);
    for (const message of anthropicPayload.messages) {
      expect(Array.isArray(message.content)).toBe(true);
      const content = message.content as Array<{
        text?: unknown;
        type: string;
      }>;
      expect(content.length).toBeGreaterThan(0);
      for (const block of content) {
        if (block.type === ContentTypes.TEXT) {
          expect(typeof block.text).toBe('string');
          expect((block.text as string).trim().length).toBeGreaterThan(0);
        }
      }
    }
  });

  it('preserves Anthropic Vertex web search pair when mixed with regular tool calls', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}vrtx_search_1`;
    const calculatorToolId = 'toolu_calc_1';
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search current Claude Vertex docs and calculate 6 * 7.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Claude Vertex web search"}',
            },
          },
          {
            type: ContentTypes.TEXT,
            text: 'I will calculate the number too.',
            tool_call_ids: [calculatorToolId],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '{"expression":"6*7"}',
              output: '42',
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: serverToolId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/claude-vertex',
                title: 'Claude on Vertex',
                encrypted_content: 'opaque',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: ' \n\t ',
          },
          {
            type: ContentTypes.TEXT,
            text: 'The calculation result is 42.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Follow up on that.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search', 'calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );
    const serverToolMessageBlocks = anthropicPayload.messages
      .map((message) =>
        typeof message.content === 'string'
          ? []
          : getAnthropicPayloadBlocks(message.content)
      )
      .find((blocks) => blocks.some((block) => block.id === serverToolId));
    const serverToolUseBlocks = allBlocks.filter(
      (block) => block.type === 'server_tool_use'
    );
    const webSearchResultBlocks = allBlocks.filter(
      (block) => block.type === 'web_search_tool_result'
    );
    const regularToolUseBlocks = allBlocks.filter(
      (block) => block.type === 'tool_use' && block.id === calculatorToolId
    );
    const whitespaceTextBlocks = allBlocks.filter(
      (block) =>
        block.type === ContentTypes.TEXT &&
        typeof block.text === 'string' &&
        block.text.trim().length === 0
    );

    expect(messages.some((message) => message instanceof ToolMessage)).toBe(
      true
    );
    expect(
      messages.some(
        (message) =>
          message instanceof ToolMessage &&
          message.tool_call_id === serverToolId
      )
    ).toBe(false);
    expect(serverToolUseBlocks).toEqual([
      {
        type: 'server_tool_use',
        id: serverToolId,
        name: 'web_search',
        input: { query: 'Claude Vertex web search' },
      },
    ]);
    expect(webSearchResultBlocks).toHaveLength(1);
    expect(webSearchResultBlocks[0].tool_use_id).toBe(serverToolId);
    expect(serverToolMessageBlocks).toBeDefined();
    expect(
      serverToolMessageBlocks?.findIndex((block) => block.id === serverToolId)
    ).toBeLessThan(
      serverToolMessageBlocks?.findIndex(
        (block) => block.tool_use_id === serverToolId
      ) ?? -1
    );
    expect(regularToolUseBlocks).toHaveLength(1);
    expect(whitespaceTextBlocks).toHaveLength(0);
  });

  it('preserves multiple Anthropic server tool pairs from one persisted turn', () => {
    const firstSearchId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}search_1`;
    const secondSearchId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}search_2`;
    const calculatorToolId = 'toolu_calc_duplicate';
    const payload: TPayload = [
      {
        role: 'user',
        content:
          'Use native web search twice for current docs and calculate 19 * 23.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'I will check the docs and calculate.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: firstSearchId,
              name: 'web_search',
              args: '{"query":"Anthropic web search docs"}',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '{"input":"19 * 23"}',
              output: '437',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '',
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: firstSearchId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/anthropic-web-search',
                title: 'Anthropic web search',
                encrypted_content: 'opaque-1',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'I found the first result and will check one more.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: secondSearchId,
              name: 'web_search',
              args: '{"query":"Anthropic web_search_20260209 docs"}',
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: secondSearchId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/anthropic-web-search-20260209',
                title: 'Anthropic web_search_20260209',
                encrypted_content: 'opaque-2',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'The calculation result is 437.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Summarize the prior results.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search', 'calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );
    const serverToolUseBlocks = allBlocks.filter(
      (block) => block.type === 'server_tool_use'
    );
    const webSearchResultBlocks = allBlocks.filter(
      (block) => block.type === 'web_search_tool_result'
    );
    const calculatorUseBlocks = allBlocks.filter(
      (block) => block.type === 'tool_use' && block.id === calculatorToolId
    );
    const serverBlocksByMessage = anthropicPayload.messages
      .map((message) =>
        typeof message.content === 'string'
          ? []
          : getAnthropicPayloadBlocks(message.content)
      )
      .filter((blocks) =>
        blocks.some(
          (block) =>
            block.id === firstSearchId ||
            block.id === secondSearchId ||
            block.tool_use_id === firstSearchId ||
            block.tool_use_id === secondSearchId
        )
      );

    expect(serverToolUseBlocks.map((block) => block.id)).toEqual([
      firstSearchId,
      secondSearchId,
    ]);
    expect(webSearchResultBlocks.map((block) => block.tool_use_id)).toEqual([
      firstSearchId,
      secondSearchId,
    ]);
    expect(calculatorUseBlocks).toHaveLength(1);
    for (const blocks of serverBlocksByMessage) {
      const serverUseIndexes = new Map<string, number>();
      blocks.forEach((block, index) => {
        if (block.type === 'server_tool_use' && typeof block.id === 'string') {
          serverUseIndexes.set(block.id, index);
        }
      });
      for (const block of blocks) {
        if (
          block.type !== 'web_search_tool_result' ||
          typeof block.tool_use_id !== 'string'
        ) {
          continue;
        }
        expect(serverUseIndexes.get(block.tool_use_id)).toBeLessThan(
          blocks.indexOf(block)
        );
      }
    }
  });

  it('preserves Anthropic server tool pairs before regular tool boundaries', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}search_before_calc`;
    const calculatorToolId = 'toolu_calc_after_search';
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search first, then calculate 19 * 23.',
      },
      {
        messageId: 'server-tool-row',
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'I will search first.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Anthropic web search docs"}',
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: serverToolId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/anthropic-web-search',
                title: 'Anthropic web search',
                encrypted_content: 'opaque',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'Now I will calculate.',
            tool_call_ids: [calculatorToolId],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '{"input":"19 * 23"}',
              output: '437',
            },
          },
          {
            type: ContentTypes.TEXT,
            text: 'The calculation result is 437.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Summarize the prior results.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search', 'calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const assistantBlocks = anthropicPayload.messages
      .filter((message) => message.role === 'assistant')
      .flatMap((message) =>
        typeof message.content === 'string'
          ? []
          : getAnthropicPayloadBlocks(message.content)
      );
    const serverToolUseIndex = assistantBlocks.findIndex(
      (block) => block.type === 'server_tool_use' && block.id === serverToolId
    );
    const serverResultIndex = assistantBlocks.findIndex(
      (block) =>
        block.type === 'web_search_tool_result' &&
        block.tool_use_id === serverToolId
    );
    const calculatorToolUseIndex = assistantBlocks.findIndex(
      (block) => block.type === 'tool_use' && block.id === calculatorToolId
    );

    expect(serverToolUseIndex).toBeGreaterThanOrEqual(0);
    expect(serverResultIndex).toBeGreaterThan(serverToolUseIndex);
    expect(calculatorToolUseIndex).toBeGreaterThan(serverResultIndex);
    const serverResultMessage = messages.find(
      (message) =>
        message instanceof AIMessage &&
        Array.isArray(message.content) &&
        (message.content as MessageContentComplex[]).some(
          (block) => block.type === 'web_search_tool_result'
        )
    );
    expect(serverResultMessage?.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'server-tool-row',
          sourceContentPartIndices: [0, 1],
        },
        {
          attribution: 'tool',
          sourceMessageId: 'server-tool-row',
          sourceContentPartIndices: [2],
        },
        {
          attribution: 'model',
          sourceMessageId: 'server-tool-row',
          sourceContentPartIndices: [3, 4],
        },
      ],
    });
    expect(
      messages.some(
        (message) =>
          message instanceof ToolMessage &&
          message.tool_call_id === calculatorToolId
      )
    ).toBe(true);
  });

  it('keeps non-Anthropic array-content tool calls on AIMessage.tool_calls', () => {
    const toolId = 'toolu_openai_after_image';
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.IMAGE_URL,
            image_url: { url: 'https://example.com/chart.png' },
          },
          {
            type: ContentTypes.TEXT,
            text: 'I will inspect the image.',
            tool_call_ids: [toolId],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: toolId,
              name: 'describe_image',
              args: '{"focus":"colors"}',
              output: 'Blue and gray.',
            },
          },
        ],
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['describe_image']),
      undefined,
      { provider: Providers.OPENAI }
    );
    const aiMessage = messages.find(
      (message) => message instanceof AIMessage
    ) as AIMessage;

    expect(Array.isArray(aiMessage.content)).toBe(true);
    expect(aiMessage.tool_calls).toEqual([
      {
        id: toolId,
        name: 'describe_image',
        args: { focus: 'colors' },
      },
    ]);
    expect(
      (aiMessage.content as MessageContentComplex[]).some(
        (block) => block.type === 'tool_use'
      )
    ).toBe(false);
    expect(
      messages.some(
        (message) =>
          message instanceof ToolMessage && message.tool_call_id === toolId
      )
    ).toBe(true);
  });

  it('normalizes Anthropic inlined tool use ids before tool results', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}search_before_invalid_calc`;
    const rawCalculatorToolId = 'toolu|responses|calculator|invalid';
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search first, then calculate 21 * 2.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Anthropic web search docs"}',
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: serverToolId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/anthropic-web-search',
                title: 'Anthropic web search',
                encrypted_content: 'opaque',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'Now I will calculate.',
            tool_call_ids: [rawCalculatorToolId],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: rawCalculatorToolId,
              name: 'calculator',
              args: '{"input":"21 * 2"}',
              output: '42',
            },
          },
        ],
      },
      {
        role: 'user',
        content: 'Summarize the result.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search', 'calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );
    const toolUseBlock = allBlocks.find(
      (block) => block.type === 'tool_use' && block.name === 'calculator'
    );
    const toolResultBlock = allBlocks.find(
      (block) => block.type === 'tool_result'
    );

    expect(toolUseBlock?.id).not.toBe(rawCalculatorToolId);
    expect(toolUseBlock?.id).toMatch(/^[a-zA-Z0-9_-]+$/);
    expect(toolUseBlock?.id).toBe(toolResultBlock?.tool_use_id);
  });

  it('preserves repairable Anthropic server search results with drifted types', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}repairable_result`;
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search current Anthropic web search docs.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Anthropic web search docs"}',
            },
          },
          {
            type: ContentTypes.TEXT,
            tool_use_id: serverToolId,
            content: [
              {
                type: 'web_search_result',
                url: 'https://example.com/anthropic-web-search',
                title: 'Anthropic web search',
                encrypted_content: 'opaque',
                page_age: '1d',
              },
            ],
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'I found the docs.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Follow up.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const assistantBlocks = anthropicPayload.messages
      .filter((message) => message.role === 'assistant')
      .flatMap((message) =>
        typeof message.content === 'string'
          ? []
          : getAnthropicPayloadBlocks(message.content)
      );
    const serverToolUseIndex = assistantBlocks.findIndex(
      (block) => block.type === 'server_tool_use' && block.id === serverToolId
    );
    const serverResultIndex = assistantBlocks.findIndex(
      (block) =>
        block.type === 'web_search_tool_result' &&
        block.tool_use_id === serverToolId
    );

    expect(serverToolUseIndex).toBeGreaterThanOrEqual(0);
    expect(serverResultIndex).toBeGreaterThan(serverToolUseIndex);
    expect(
      messages.flatMap(
        (message) =>
          getProviderMessageProvenance(message)?.parts.filter(
            (part) => part.sourceContentPartIndices?.includes(1) === true
          ) ?? []
      )
    ).toEqual([
      {
        attribution: 'model',
        sourceContentPartIndices: [0, 1, 2],
      },
    ]);
    expect(
      messages.some(
        (message) =>
          message instanceof ToolMessage &&
          message.tool_call_id === serverToolId
      )
    ).toBe(false);
  });

  it('does not pair malformed Anthropic server tool result blocks', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}malformed_result`;
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search current docs.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Anthropic web search docs"}',
            },
          },
          {
            type: ContentTypes.TEXT,
            text: 'Malformed result should not pair this server tool.',
            tool_use_id: serverToolId,
          } as MessageContentComplex,
          {
            type: ContentTypes.TEXT,
            text: 'Here is the final answer.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Follow up.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );

    expect(allBlocks.some((block) => block.id === serverToolId)).toBe(false);
    expect(allBlocks.some((block) => block.tool_use_id === serverToolId)).toBe(
      false
    );
  });

  it('drops unpaired historical Anthropic Vertex web search calls from mixed turns', () => {
    const serverToolId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}vrtx_missing_result`;
    const calculatorToolId = 'toolu_calc_1';
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Search for current Claude info and calculate 6 * 7.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: serverToolId,
              name: 'web_search',
              args: '{"query":"Claude Vertex web search"}',
              output: 'Search results were formatted for the assistant.',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '{"expression":"6*7"}',
              output: '42',
            },
          },
        ],
      },
      {
        role: 'user',
        content: 'Follow up on that.',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search', 'calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );

    expect(
      messages.some(
        (message) =>
          message instanceof ToolMessage &&
          message.tool_call_id === serverToolId
      )
    ).toBe(false);
    expect(allBlocks.some((block) => block.id === serverToolId)).toBe(false);
    expect(allBlocks.some((block) => block.tool_use_id === serverToolId)).toBe(
      false
    );
    expect(
      allBlocks.some(
        (block) => block.type === 'tool_use' && block.id === calculatorToolId
      )
    ).toBe(true);
  });

  it('deduplicates repeated Anthropic Vertex client tool calls in replay payloads', () => {
    const calculatorToolId = 'toolu_vrtx_calc_1';
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Calculate 19 * 23.',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '{"input":"19 * 23"}',
              output: '437',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: calculatorToolId,
              name: 'calculator',
              args: '',
            },
          },
          {
            type: ContentTypes.TEXT,
            text: '437437',
          },
        ],
      },
      {
        role: 'user',
        content: 'What was the result?',
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['calculator']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );
    const anthropicPayload = _convertMessagesToAnthropicPayload(messages);
    const allBlocks = anthropicPayload.messages.flatMap((message) =>
      typeof message.content === 'string'
        ? []
        : getAnthropicPayloadBlocks(message.content)
    );
    const toolUseBlocks = allBlocks.filter(
      (block) => block.type === 'tool_use' && block.id === calculatorToolId
    );
    const toolResultBlocks = allBlocks.filter(
      (block) =>
        block.type === 'tool_result' && block.tool_use_id === calculatorToolId
    );

    expect(toolUseBlocks).toEqual([
      {
        type: 'tool_use',
        id: calculatorToolId,
        name: 'calculator',
        input: { input: '19 * 23' },
      },
    ]);
    expect(toolResultBlocks).toHaveLength(1);
    expect(toolResultBlocks[0].content).toBe('437');
  });

  it('should handle malformed tool call entries with missing tool_call property', () => {
    const tools = new Set(['search']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that.',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            // Missing tool_call property - should not crash
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Result',
            },
          },
        ],
      },
    ];
    // Should not throw error
    const result = formatAgentMessages(payload, undefined, tools);
    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it('should handle malformed tool call entries with missing name', () => {
    const tools = new Set(['search']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Checking...',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '456',
              // Missing name property
              args: '{}',
            },
          },
        ],
      },
    ];
    // Should not throw error
    const result = formatAgentMessages(payload, undefined, tools);
    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it('should handle multiple content parts in assistant messages', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Part 1' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Part 2' },
        ],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toHaveLength(2);
  });

  it('should heal invalid tool call structure by creating a preceding AIMessage', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
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

    // Should have 2 messages: an AIMessage and a ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have an empty content and the tool_call
    expect(result.messages[0].content).toBe('');
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: '123',
      name: 'search',
      args: { query: 'weather' },
    });

    // The ToolMessage should have the correct properties
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('123');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('The weather is sunny.');
  });

  it('should handle tool calls with non-JSON args', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Checking...',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: 'non-json-string',
              output: 'Result',
            },
          },
        ],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(2);
    expect(
      (result.messages[0] as AIMessage).tool_calls?.[0].args
    ).toStrictEqual({ input: 'non-json-string' });
  });

  it('should handle complex tool calls with multiple steps', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for that information.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'search',
              args: '{"query":"weather in New York"}',
              output:
                'The weather in New York is currently sunny with a temperature of 75°F.',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now, I\'ll convert the temperature.',
            tool_call_ids: ['convert_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'convert_1',
              name: 'convert_temperature',
              args: '{"temperature": 75, "from": "F", "to": "C"}',
              output: '23.89°C',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here\'s your answer.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(5);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);
    expect(result.messages[3]).toBeInstanceOf(ToolMessage);
    expect(result.messages[4]).toBeInstanceOf(AIMessage);

    // Check first AIMessage
    expect(result.messages[0].content).toBe(
      'I\'ll search for that information.'
    );
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'search_1',
      name: 'search',
      args: { query: 'weather in New York' },
    });

    // Check first ToolMessage
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('search_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe(
      'The weather in New York is currently sunny with a temperature of 75°F.'
    );

    // Check second AIMessage
    expect(result.messages[2].content).toBe(
      'Now, I\'ll convert the temperature.'
    );
    expect((result.messages[2] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[2] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'convert_1',
      name: 'convert_temperature',
      args: { temperature: 75, from: 'F', to: 'C' },
    });

    // Check second ToolMessage
    expect((result.messages[3] as ToolMessage).tool_call_id).toBe('convert_1');
    expect(result.messages[3].name).toBe('convert_temperature');
    expect(result.messages[3].content).toBe('23.89°C');

    // Check final AIMessage
    expect(result.messages[4].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Here\'s your answer.', type: ContentTypes.TEXT },
    ]);
  });

  it('preserves tool-only assistant turn boundaries when converting messages to content', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: { step: 1 },
            type: 'tool_call' as const,
          },
        ],
      }),
      new ToolMessage({
        content: 'first result',
        tool_call_id: 'call_1',
        name: 'lookup',
      }),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_2',
            name: 'lookup',
            args: { step: 2 },
            type: 'tool_call' as const,
          },
        ],
      }),
      new ToolMessage({
        content: 'second result',
        tool_call_id: 'call_2',
        name: 'lookup',
      }),
    ];

    const content = convertMessagesToContent(messages);
    expect(content).toHaveLength(4);
    expect(content[0]).toMatchObject({
      type: ContentTypes.TEXT,
      text: '',
      tool_call_ids: ['call_1'],
    });
    expect(content[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_1',
        name: 'lookup',
        output: 'first result',
      },
    });
    expect(content[2]).toMatchObject({
      type: ContentTypes.TEXT,
      text: '',
      tool_call_ids: ['call_2'],
    });
    expect(content[3]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_2',
        name: 'lookup',
        output: 'second result',
      },
    });

    const result = formatAgentMessages([{ role: 'assistant', content }]);
    expect(result.messages).toHaveLength(4);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);
    expect(result.messages[3]).toBeInstanceOf(ToolMessage);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].id).toBe('call_1');
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('call_1');
    expect((result.messages[2] as AIMessage).tool_calls?.[0].id).toBe('call_2');
    expect((result.messages[3] as ToolMessage).tool_call_id).toBe('call_2');
  });

  /**
   * Empty-string content rather than a forced `undefined`: `@langchain/core`
   * 1.2.6 made `instanceof` structural, so a message whose `content` was
   * overwritten with `undefined` stops registering as a `ToolMessage` and
   * never reaches the merge. The constructor normalizes `undefined` to `[]`,
   * so that state is no longer representable; `''` covers the same non-array
   * branch through `stringifyToolMessageContent`.
   */
  it('keeps absent tool content empty when merging Anthropic artifacts', () => {
    const toolMessage = new ToolMessage({
      content: '',
      tool_call_id: 'call_artifact',
      name: 'render',
      artifact: {
        content: [{ type: ContentTypes.TEXT, text: 'artifact text' }],
      },
    });

    const originalMessages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_artifact',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      toolMessage,
    ];
    const formattedMessages = projectAnthropicArtifactContent(originalMessages);
    const formattedToolMessage = formattedMessages[1] as ToolMessage;

    expect(formattedToolMessage.content).toEqual([
      { type: ContentTypes.TEXT, text: '' },
      { type: ContentTypes.TEXT, text: 'artifact text' },
    ]);
    expect(toolMessage.content).toBe('');
    expect(formattedMessages).not.toBe(originalMessages);
  });

  it('adds unindexed tool lineage only when Anthropic artifact bytes survive projection', () => {
    const toolMessage = new ToolMessage({
      content: [{ type: ContentTypes.TEXT, text: 'base' }],
      tool_call_id: 'call_artifact_lineage',
      name: 'render',
      additional_kwargs: {
        provenance: {
          version: 1,
          parts: [
            {
              attribution: 'tool',
              sourceMessageId: 'stored-tool-row',
              sourceContentPartIndices: [3],
            },
          ],
        },
        sourceMessageId: 'stored-tool-row',
        sourceMessageIds: ['stored-tool-row'],
      },
      artifact: {
        content: [{ type: ContentTypes.TEXT, text: 'artifact' }],
      },
    });
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_artifact_lineage',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      toolMessage,
    ];

    const projected = projectAnthropicArtifactContent(messages, 100);
    const projectedTool = projected[1] as ToolMessage;

    expect(getProviderMessageProvenance(projectedTool)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'stored-tool-row',
        sourceContentPartIndices: [3],
      },
      { attribution: 'tool', sourceMessageId: 'stored-tool-row' },
    ]);
    expect(projectedTool.lc_kwargs.additional_kwargs).toBe(
      projectedTool.additional_kwargs
    );
    expect(getProviderMessageProvenance(toolMessage)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'stored-tool-row',
        sourceContentPartIndices: [3],
      },
    ]);

    const capped = projectAnthropicArtifactContent(messages, 4)[1];
    expect(getProviderMessageProvenance(capped)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'stored-tool-row',
        sourceContentPartIndices: [3],
      },
    ]);

    formatAnthropicArtifactContent(messages);
    expect(getProviderMessageProvenance(toolMessage)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'stored-tool-row',
        sourceContentPartIndices: [3],
      },
      { attribution: 'tool', sourceMessageId: 'stored-tool-row' },
    ]);
    expect(toolMessage.lc_kwargs.additional_kwargs).toBe(
      toolMessage.additional_kwargs
    );
  });

  it('caps Anthropic artifact expansion after pruning', () => {
    const toolMessage = new ToolMessage({
      content: 'short result',
      tool_call_id: 'call_artifact_capped',
      artifact: {
        content: 'artifact'.repeat(1_000),
      },
    });
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_artifact_capped',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      toolMessage,
    ];

    const formatted = projectAnthropicArtifactContent(messages, 200);
    const formattedTool = formatted[1] as ToolMessage;

    expect(
      serializeToolContent(formattedTool.content).length
    ).toBeLessThanOrEqual(200);
    expect(serializeToolContent(formattedTool.content)).toContain('truncated');
    expect(toolMessage.content).toBe('short result');
    expect(toolMessage.artifact.content).toContain('artifact');
    expect(projectAnthropicArtifactContent(formatted, 200)).toBe(formatted);
  });

  it('caps the aggregate OpenAI/Google artifact payload', () => {
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_artifact_payload',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      new ToolMessage({
        content: 'short result',
        tool_call_id: 'call_artifact_payload',
        additional_kwargs: {
          sourceMessageId: 'stored-tool-row',
          sourceMessageIds: ['stored-tool-row'],
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId: 'stored-tool-row',
                sourceContentPartIndices: [3],
              },
            ],
          },
        },
        artifact: {
          content: [
            { type: ContentTypes.TEXT, text: 'artifact'.repeat(1_000) },
          ],
        },
      }),
    ];

    const formatted = projectArtifactPayload(messages, 200);

    const payload = formatted[formatted.length - 1] as HumanMessage;
    expect(payload).toBeInstanceOf(HumanMessage);
    expect(serializeToolContent(payload.content).length).toBeLessThanOrEqual(
      200
    );
    expect(messages).toHaveLength(2);
    expect((messages[1] as ToolMessage).content).toBe('short result');
    expect(formatted[1].additional_kwargs).not.toBe(
      messages[1].additional_kwargs
    );
    expect(messages[1].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'stored-tool-row',
          sourceContentPartIndices: [3],
        },
      ],
    });
    expect(payload.additional_kwargs.sourceMessageIds).toEqual([
      'stored-tool-row',
    ]);
    expect(payload.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'stored-tool-row',
          sourceContentPartIndices: [3],
        },
        {
          attribution: 'tool',
          sourceMessageId: 'stored-tool-row',
        },
      ],
    });
    expect(projectArtifactPayload(formatted, 200)).toBe(formatted);
  });

  it('stamps only artifact source parts that survive the aggregate cap', () => {
    const createToolMessage = (
      toolCallId: string,
      sourceMessageId: string,
      content: ToolMessage['content'],
      artifact: string,
      sourceContentPartIndices: number[]
    ): ToolMessage =>
      new ToolMessage({
        content,
        tool_call_id: toolCallId,
        artifact: { content: artifact },
        additional_kwargs: {
          sourceMessageId,
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId,
                sourceContentPartIndices,
              },
            ],
          },
        },
      });
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'first', name: 'render', args: {}, type: 'tool_call' },
          { id: 'second', name: 'render', args: {}, type: 'tool_call' },
        ],
      }),
      createToolMessage(
        'first',
        'first-row',
        [
          { type: ContentTypes.TEXT, text: 'A'.repeat(1_000) },
          { type: ContentTypes.TEXT, text: 'OMITTED-FIRST-PART' },
        ],
        'omitted first artifact',
        [0, 1]
      ),
      createToolMessage(
        'second',
        'second-row',
        'OMITTED-SECOND-RESULT',
        'omitted second artifact',
        [0]
      ),
    ];

    const formatted = projectArtifactPayload(messages, 80);
    const payload = formatted[formatted.length - 1] as HumanMessage;
    const payloadText = serializeToolContent(payload.content);

    expect(payloadText).not.toContain('OMITTED-FIRST-PART');
    expect(payloadText).not.toContain('OMITTED-SECOND-RESULT');
    expect(payload.additional_kwargs.sourceMessageIds).toEqual(['first-row']);
    expect(payload.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'first-row',
          sourceContentPartIndices: [0],
        },
      ],
    });
  });

  it.each([
    ['duplicate', [[0], [0]]],
    ['gapped/out-of-range', [[0], [2]]],
    ['reordered', [[1], [0]]],
  ])(
    'handles a %s artifact mapping without ordinal lineage loss',
    (mappingKind, sourceContentPartIndices) => {
      const messages = [
        new AIMessageChunk({
          content: '',
          tool_calls: [
            { id: 'ambiguous', name: 'render', args: {}, type: 'tool_call' },
          ],
        }),
        new ToolMessage({
          content: [
            { type: ContentTypes.TEXT, text: 'A'.repeat(1_000) },
            { type: ContentTypes.TEXT, text: 'OMITTED-SECOND' },
          ],
          tool_call_id: 'ambiguous',
          artifact: { content: 'omitted artifact' },
          additional_kwargs: {
            sourceMessageIds: ['first-row', 'second-row'],
            provenance: {
              version: 1,
              parts: [
                {
                  attribution: 'model',
                  sourceMessageId: 'first-row',
                  sourceContentPartIndices: sourceContentPartIndices[0],
                },
                {
                  attribution: 'user',
                  sourceMessageId: 'second-row',
                  sourceContentPartIndices: sourceContentPartIndices[1],
                },
              ],
            },
          },
        }),
      ];

      const formatted = projectArtifactPayload(messages, 80);
      const payload = formatted[formatted.length - 1] as HumanMessage;

      expect(serializeToolContent(payload.content)).not.toContain(
        'OMITTED-SECOND'
      );
      expect(payload.additional_kwargs.sourceMessageIds).toEqual(
        mappingKind === 'reordered' ? ['second-row'] : undefined
      );
      expect(payload.additional_kwargs.provenance).toEqual({
        version: 1,
        parts:
          mappingKind === 'reordered'
            ? [
              {
                attribution: 'tool',
                sourceMessageId: 'second-row',
                sourceContentPartIndices: [0],
              },
            ]
            : [{ attribution: 'tool' }],
      });
    }
  );

  it('skips empty artifact blocks without dropping later visible blocks', () => {
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'empty-first', name: 'render', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: [
          { type: ContentTypes.TEXT, text: '' },
          { type: ContentTypes.TEXT, text: 'SECOND-BLOCK' },
        ],
        tool_call_id: 'empty-first',
        artifact: { content: 'ARTIFACT' },
        additional_kwargs: {
          sourceMessageId: 'tool-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId: 'tool-row',
                sourceContentPartIndices: [0, 1],
              },
            ],
          },
        },
      }),
    ];

    const formatted = projectArtifactPayload(messages, 1_000);
    const payload = formatted[formatted.length - 1] as HumanMessage;

    expect(serializeToolContent(payload.content)).toContain('SECOND-BLOCK');
    expect(serializeToolContent(payload.content)).toContain('ARTIFACT');
    expect(payload.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'tool-row',
          sourceContentPartIndices: [1],
        },
        { attribution: 'tool', sourceMessageId: 'tool-row' },
      ],
    });
  });

  it('drops ambiguous legacy source ids after partial artifact compaction', () => {
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'legacy', name: 'render', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: [
          { type: ContentTypes.TEXT, text: 'A'.repeat(1_000) },
          { type: ContentTypes.TEXT, text: 'OMITTED-SECOND' },
        ],
        tool_call_id: 'legacy',
        artifact: { content: 'omitted artifact' },
        additional_kwargs: { sourceMessageIds: ['first-row', 'second-row'] },
      }),
    ];

    const formatted = projectArtifactPayload(messages, 80);
    const payload = formatted[formatted.length - 1] as HumanMessage;

    expect(serializeToolContent(payload.content)).not.toContain(
      'OMITTED-SECOND'
    );
    expect(payload.additional_kwargs.sourceMessageIds).toBeUndefined();
    expect(payload.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [{ attribution: 'tool' }],
    });
  });

  it('drops ambiguous source ids after partial scalar artifact projection', () => {
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'scalar', name: 'render', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'A'.repeat(1_000),
        tool_call_id: 'scalar',
        artifact: { content: 'omitted artifact' },
        additional_kwargs: {
          sourceMessageIds: ['first-row', 'second-row'],
        },
      }),
    ];

    const formatted = projectArtifactPayload(messages, 80);
    const payload = formatted[formatted.length - 1] as HumanMessage;

    expect(payload.additional_kwargs.sourceMessageIds).toBeUndefined();
    expect(payload.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [{ attribution: 'tool' }],
    });
  });

  it('bounds artifact block arrays without spreading them into call arguments', () => {
    const repeatedBlock = {
      type: ContentTypes.TEXT,
      text: 'x',
    } as MessageContentComplex;
    const artifactContent = new Array<MessageContentComplex>(150_000).fill(
      repeatedBlock
    );
    const toolMessage = new ToolMessage({
      content: 'short result',
      tool_call_id: 'call_artifact_many_blocks',
      artifact: { content: artifactContent },
    });
    const messages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_artifact_many_blocks',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      toolMessage,
    ];

    const formatted = projectArtifactPayload(messages, 200);
    const payload = formatted[formatted.length - 1] as HumanMessage;

    expect(payload).toBeInstanceOf(HumanMessage);
    expect(serializeToolContent(payload.content).length).toBeLessThanOrEqual(
      200
    );
    expect(toolMessage.content).toBe('short result');
    expect(toolMessage.artifact.content).toBe(artifactContent);
  });

  it('projects structured tool content to the bounded string both OpenAI APIs send', () => {
    let toJSONCalls = 0;
    const toolMessage = new ToolMessage({
      id: 'tool-message-id',
      name: 'render',
      status: 'success',
      tool_call_id: 'call_openai_structured',
      content: [
        { type: ContentTypes.TEXT, text: 'rendered chart' },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/png;base64,${'A'.repeat(2_000)}`,
          },
          toJSON() {
            toJSONCalls++;
            return { expanded: 'B'.repeat(20_000) };
          },
        },
      ],
      artifact: { retained: true },
    });

    const original = [toolMessage];
    const projected = projectOpenAIToolMessageContent(original, 200);
    const projectedTool = projected[0] as ToolMessage;
    const chatPayload = _convertMessagesToOpenAIParams(projected);
    const responsesPayload = _convertMessagesToOpenAIResponsesParams(projected);

    expect(projected).not.toBe(original);
    expect(typeof projectedTool.content).toBe('string');
    expect((projectedTool.content as string).length).toBeLessThanOrEqual(200);
    expect(projectedTool.tool_call_id).toBe(toolMessage.tool_call_id);
    expect(projectedTool.status).toBe(toolMessage.status);
    expect(projectedTool.artifact).toBe(toolMessage.artifact);
    expect(Array.isArray(toolMessage.content)).toBe(true);
    expect(toJSONCalls).toBe(0);
    expect(chatPayload[0]).toMatchObject({
      role: 'tool',
      content: projectedTool.content,
      tool_call_id: toolMessage.tool_call_id,
    });
    expect(responsesPayload[0]).toMatchObject({
      type: 'function_call_output',
      output: projectedTool.content,
      call_id: toolMessage.tool_call_id,
    });
  });

  it('neutralizes nonterminal Responses references on preempted provider input', () => {
    const message = new AIMessage({
      id: 'msg_interrupted',
      content: 'Partial answer.',
      additional_kwargs: {
        reasoning: {
          id: 'rs_interrupted',
          type: 'reasoning',
          summary: [],
        },
        tool_outputs: [
          {
            id: 'ci_interrupted',
            type: 'code_interpreter_call',
            status: 'completed',
          },
        ],
        __openai_function_call_ids__: { call_1: 'fc_interrupted' },
        __openai_custom_tool_call_ids__: { call_2: 'ctc_interrupted' },
        retained: 'application metadata',
      },
      response_metadata: {
        id: 'resp_interrupted',
        model_provider: 'openai',
        output: [],
        tool_outputs: [{ id: 'ci_interrupted' }],
        model_name: 'gpt-5.6',
        preempted: true,
      },
    });

    const messages = [message];
    const unsafeProviderInput = convertMessagesToResponsesInput({
      messages,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });
    const projected = projectOpenAIResponsesToolMessageContent(messages);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projected).not.toBe(messages);
    expect(projectedMessage).not.toBe(message);
    expect(Object.getPrototypeOf(projectedMessage)).toBe(
      Object.getPrototypeOf(message)
    );
    expect(projectedMessage.id).toBeUndefined();
    expect(projectedMessage.content).toBe('Partial answer.');
    expect(projectedMessage.additional_kwargs).toEqual({
      retained: 'application metadata',
    });
    expect(projectedMessage.response_metadata).toEqual({
      model_provider: 'openai',
      model_name: 'gpt-5.6',
      preempted: true,
    });
    const unsafeSerializedInput = JSON.stringify(unsafeProviderInput);
    const serializedInput = JSON.stringify(providerInput);
    expect(unsafeSerializedInput).toContain('rs_interrupted');
    expect(unsafeSerializedInput).toContain('ci_interrupted');
    expect(serializedInput).toContain('Partial answer.');
    expect(serializedInput).not.toContain('interrupted');
    expect(message.id).toBe('msg_interrupted');
    expect(message.additional_kwargs.reasoning).toBeDefined();
    expect(message.response_metadata).toHaveProperty('output');

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it('retains self-contained encrypted reasoning on preempted Responses input', () => {
    const reasoning = {
      id: 'rs_encrypted',
      type: 'reasoning',
      status: 'completed',
      summary: [],
      encrypted_content: 'opaque-reasoning',
    };
    const message = new AIMessage({
      id: 'msg_interrupted',
      content: 'Partial answer.',
      additional_kwargs: { reasoning },
      response_metadata: { model_provider: 'openai', preempted: true },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: [projected],
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage.additional_kwargs.reasoning).toBe(reasoning);
    expect(providerInput).toContainEqual(
      expect.objectContaining({
        id: 'rs_encrypted',
        type: 'reasoning',
        encrypted_content: 'opaque-reasoning',
      })
    );
  });

  it.each([
    [
      'Responses projector',
      (messages: BaseMessage[]) =>
        projectOpenAIResponsesToolMessageContent(messages),
    ],
    [
      'common native projector',
      (messages: BaseMessage[]) =>
        projectToolStreamContentForProvider(messages, 'native'),
    ],
  ])(
    'does not classify provider-neutral v1 Chat content as Responses through the %s',
    (_name, project) => {
      const applicationBlock = {
        type: 'non_standard' as const,
        value: { source: 'application' },
      };
      const message = new AIMessage({
        content: [{ type: 'text', text: 'Chat answer.' }, applicationBlock],
        response_metadata: {
          model_provider: 'openai',
          output_version: 'v1',
          preempted: true,
        },
      });
      const messages = [message];

      const projected = project(messages);

      expect(projected).toBe(messages);
      expect(projected[0]).toBe(message);
      expect(message.content).toEqual([
        { type: 'text', text: 'Chat answer.' },
        applicationBlock,
      ]);
    }
  );

  it('normalizes Responses text metadata before OpenAI Chat fallback replay', () => {
    const annotatedText = {
      type: 'text' as const,
      text: 'Cited answer.',
      annotations: [
        {
          type: 'citation' as const,
          url: 'https://example.com/source',
          title: 'Source',
        },
      ],
      extras: { phase: 'final_answer' },
      index: 0,
    };
    const message = new AIMessage({
      content: [annotatedText],
      response_metadata: {
        model_provider: 'openai',
        output: [],
        output_version: 'v1',
        preempted: true,
      },
    });

    const native = projectToolStreamContentForProvider([message], 'native');
    const fallback = projectToolStreamContentForProvider([message], 'fallback');
    const fallbackMessage = fallback[0] as AIMessage;

    expect((native[0] as AIMessage).content).toEqual([annotatedText]);
    expect(fallbackMessage.content).toEqual([
      { type: 'text', text: 'Cited answer.' },
    ]);
    expect(_convertMessagesToOpenAIParams(fallback)).toEqual([
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'Cited answer.' }],
      },
    ]);
    expect(message.content).toEqual([annotatedText]);

    const fallbackAgain = projectToolStreamContentForProvider(
      fallback,
      'fallback'
    );
    expect(fallbackAgain).toBe(fallback);
    expect(fallbackAgain[0]).toBe(fallbackMessage);
  });

  it.each([
    [
      'Responses projector',
      (messages: BaseMessage[]) =>
        projectOpenAIResponsesToolMessageContent(messages),
    ],
    [
      'common native projector',
      (messages: BaseMessage[]) =>
        projectToolStreamContentForProvider(messages, 'native'),
    ],
  ])('sanitizes native v0 fragments through the %s', (_name, project) => {
    const emptyText = { type: 'text' as const, text: '' };
    const partialText = { type: 'text' as const, text: 'Partial answer.' };
    const bareReasoning = {
      type: 'reasoning' as const,
      reasoning: 'summary',
    };
    const message = new AIMessage({
      id: 'resp_fragmented',
      content: [emptyText, partialText, bareReasoning],
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });
    const messages = [message];

    const projected = project(messages);
    const projectedMessage = projected[0] as AIMessage;

    expect(projected).not.toBe(messages);
    expect(projectedMessage.content).toEqual([partialText]);
    expect(projectedMessage.response_metadata.output_version).toBeUndefined();
    expect(message.content).toEqual([emptyText, partialText, bareReasoning]);

    const projectedAgain = project(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it.each([
    [
      'Responses projector',
      (messages: BaseMessage[]) =>
        projectOpenAIResponsesToolMessageContent(messages),
    ],
    [
      'common native projector',
      (messages: BaseMessage[]) =>
        projectToolStreamContentForProvider(messages, 'native'),
    ],
  ])(
    'sanitizes unpromoted translated v0 content through the %s',
    (_name, project) => {
      const emptyText = { type: 'text' as const, text: '' };
      const partialText = { type: 'text' as const, text: 'Partial answer.' };
      const applicationImage = {
        type: 'image' as const,
        mimeType: 'image/png',
        data: 'AQ==',
      };
      const bareReasoning = {
        type: 'reasoning' as const,
        reasoning: 'unfinished summary',
      };
      const message = new AIMessage({
        id: 'msg_unpromoted',
        content: [emptyText, partialText, applicationImage, bareReasoning],
        additional_kwargs: {
          tool_outputs: [
            {
              id: 'ig_unusable',
              type: 'image_generation_call',
              status: 'in_progress',
              result: 'AA==',
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const messages = [message];

      const projected = project(messages);
      const projectedMessage = projected[0] as AIMessage;

      expect(projected).not.toBe(messages);
      expect(projectedMessage.id).toBeUndefined();
      expect(projectedMessage.content).toEqual([partialText, applicationImage]);
      expect(projectedMessage.additional_kwargs).not.toHaveProperty(
        'tool_outputs'
      );
      expect(projectedMessage.response_metadata.output_version).toBeUndefined();
      expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
        'ig_unusable'
      );
      expect(message.content).toEqual([
        emptyText,
        partialText,
        applicationImage,
        bareReasoning,
      ]);

      const projectedAgain = project(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it('recognizes response_metadata.tool_outputs as a Responses marker', () => {
    const message = new AIMessage({
      content: 'Partial answer.',
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
        tool_outputs: [{ id: 'ci_metadata_only' }],
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;

    expect(projectedMessage).not.toBe(message);
    expect(projectedMessage.content).toBe('Partial answer.');
    expect(projectedMessage.response_metadata).not.toHaveProperty(
      'tool_outputs'
    );
    expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
      'ci_metadata_only'
    );
    expect(message.response_metadata).toHaveProperty('tool_outputs');
  });

  it('neutralizes terminal Responses ids when request retention is unknown', () => {
    const message = new AIMessage({
      id: 'msg_completed',
      content: 'Complete answer.',
      additional_kwargs: {
        reasoning: {
          id: 'rs_completed',
          type: 'reasoning',
          summary: [],
        },
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
        status: 'completed',
        output: [
          {
            id: 'msg_completed',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'Complete answer.',
                annotations: [],
              },
            ],
          },
        ],
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage).not.toBe(message);
    expect(projectedMessage.id).toBeUndefined();
    expect(projectedMessage.additional_kwargs.reasoning).toBeUndefined();
    expect(projectedMessage.response_metadata).toEqual({
      model_provider: 'openai',
      preempted: true,
      status: 'completed',
    });
    expect(JSON.stringify(providerInput)).toContain('Complete answer.');
    expect(JSON.stringify(providerInput)).not.toContain('msg_completed');
    expect(JSON.stringify(providerInput)).not.toContain('rs_completed');
    expect(message.response_metadata).toHaveProperty('output');
  });

  it.each(['additional_kwargs', 'response_metadata'] as const)(
    'preserves completed v0 generated images from %s without provider ids',
    (imageSource) => {
      const events: Parameters<
        typeof convertResponsesDeltaToChatGenerationChunk
      >[0][] = [
        {
          type: 'response.output_item.done',
          sequence_number: 0,
          output_index: 0,
          item: {
            id: 'ig_interrupted',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        },
        {
          type: 'response.output_item.added',
          sequence_number: 1,
          output_index: 1,
          item: {
            id: 'msg_interrupted',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
        },
        {
          type: 'response.output_text.delta',
          sequence_number: 2,
          output_index: 1,
          content_index: 0,
          item_id: 'msg_interrupted',
          delta: 'Partial answer.',
          logprobs: [],
        },
      ];
      let streamedMessage: AIMessageChunk | undefined;
      for (const event of events) {
        const generation = convertResponsesDeltaToChatGenerationChunk(event);
        if (generation == null) {
          continue;
        }
        streamedMessage =
          streamedMessage == null
            ? (generation.message as AIMessageChunk)
            : (streamedMessage.concat(
                generation.message as AIMessageChunk
            ) as AIMessageChunk);
      }
      expect(streamedMessage).toBeDefined();
      const toolOutputs = streamedMessage!.additional_kwargs.tool_outputs;
      const message =
        imageSource === 'additional_kwargs'
          ? streamedMessage!
          : new AIMessage({
            content: 'Partial answer.',
            response_metadata: {
              model_provider: 'openai',
              output: toolOutputs,
            },
          });
      message.response_metadata.preempted = true;

      const unsafeProviderInput = convertMessagesToResponsesInput({
        messages: [message],
        zdrEnabled: false,
        model: 'gpt-5.6',
      });
      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;

      expect(JSON.stringify(unsafeProviderInput)).toContain('ig_interrupted');
      expect(projectedMessage.response_metadata).toMatchObject({
        model_provider: 'openai',
        output_version: 'v1',
        preempted: true,
      });
      expect(projectedMessage.content).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ type: 'text', text: 'Partial answer.' }),
          {
            type: 'image',
            mimeType: 'image/png',
            data: 'AA==',
            extras: IMAGE_GENERATION_REPLAY_EXTRAS,
          },
        ])
      );
      expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
        'ig_interrupted'
      );
      expect(JSON.stringify(message.toJSON())).toContain('ig_interrupted');

      for (const zdrEnabled of [false, true]) {
        const providerInput = convertMessagesToResponsesInput({
          messages: projected,
          zdrEnabled,
          model: 'gpt-5.6',
        });
        expect(providerInput).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              type: 'message',
              role: 'assistant',
              content: expect.arrayContaining([
                expect.objectContaining({
                  type: 'output_text',
                  text: 'Partial answer.',
                }),
                {
                  type: 'input_image',
                  detail: 'auto',
                  image_url: 'data:image/png;base64,AA==',
                },
              ]),
            }),
          ])
        );
        expect(JSON.stringify(providerInput)).not.toContain(
          'image_generation_call'
        );
        expect(JSON.stringify(providerInput)).not.toContain('ig_interrupted');
      }

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it('keeps completed v1 generated images in their Responses output positions', () => {
    const firstGeneratedImage = {
      id: 'ig_before_text',
      type: 'image_generation_call',
      status: 'completed',
      result: 'AA==',
    };
    const secondGeneratedImage = {
      id: 'ig_between_text',
      type: 'image_generation_call',
      status: 'completed',
      result: 'AQ==',
    };
    const message = new AIMessage({
      content: [
        {
          type: 'image',
          mimeType: 'image/png',
          data: firstGeneratedImage.result,
          id: firstGeneratedImage.id,
          metadata: { status: 'completed' },
        },
        { type: 'text', text: 'The image above is the first result.' },
        {
          type: 'image',
          mimeType: 'image/png',
          data: secondGeneratedImage.result,
          id: secondGeneratedImage.id,
          metadata: { status: 'completed' },
        },
        { type: 'text', text: 'The second image is directly above.' },
      ],
      response_metadata: {
        model_provider: 'openai',
        output_version: 'v1',
        preempted: true,
        output: [
          firstGeneratedImage,
          {
            id: 'msg_first_narration',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'The image above is the first result.',
                annotations: [],
              },
            ],
          },
          secondGeneratedImage,
          {
            id: 'msg_second_narration',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [
              {
                type: 'output_text',
                text: 'The second image is directly above.',
                annotations: [],
              },
            ],
          },
        ],
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage.content).toEqual([
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
      { type: 'text', text: 'The image above is the first result.' },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AQ==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
      { type: 'text', text: 'The second image is directly above.' },
    ]);
    expect(providerInput).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,AA==',
          },
          {
            type: 'output_text',
            text: 'The image above is the first result.',
            annotations: [],
          },
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,AQ==',
          },
          {
            type: 'output_text',
            text: 'The second image is directly above.',
            annotations: [],
          },
        ],
      },
    ]);
    expect(JSON.stringify(projectedMessage.toJSON())).not.toMatch(
      /ig_(before|between)_text/
    );
    expect(message.response_metadata).toHaveProperty('output');

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it.each([
    ['v0', 'jpeg', '/9j/4AAQSkZJRg==', 'image/jpeg'],
    ['v1', 'jpeg', '/9j/4AAQSkZJRg==', 'image/jpeg'],
    ['v0', 'webp', 'UklGRgAAAABXRUJQ', 'image/webp'],
    ['v1', 'webp', 'UklGRgAAAABXRUJQ', 'image/webp'],
  ] as const)(
    'preserves %s generated %s media type during replay',
    (outputVersion, _format, data, mimeType) => {
      const generatedImage = {
        id: `ig_${outputVersion}_${_format}`,
        type: 'image_generation_call',
        status: 'completed',
        result: data,
      };
      const v0Message = new AIMessage({
        content: [{ type: 'text', text: 'Generated image.' }],
        additional_kwargs: { tool_outputs: [generatedImage] },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const message =
        outputVersion === 'v0'
          ? v0Message
          : new AIMessage({
            contentBlocks: v0Message.contentBlocks,
            additional_kwargs: v0Message.additional_kwargs,
            response_metadata: {
              model_provider: 'openai',
              output_version: 'v1',
              preempted: true,
            },
          });
      const originalSerialized = JSON.stringify(message.toJSON());

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });

      expect(projectedMessage.content).toEqual(
        expect.arrayContaining([
          {
            type: 'image',
            mimeType,
            data,
            extras: IMAGE_GENERATION_REPLAY_EXTRAS,
          },
        ])
      );
      expect(JSON.stringify(providerInput)).toContain(
        `data:${mimeType};base64,${data}`
      );
      expect(JSON.stringify(providerInput)).not.toContain(generatedImage.id);
      expect(JSON.stringify(providerInput)).not.toContain(
        'librechatServerToolResult'
      );
      expect(JSON.stringify(providerInput)).not.toContain('extras');

      const fallback = projectToolStreamContentForProvider(
        [message],
        'fallback'
      );
      expect((fallback[0] as AIMessage).content).toEqual([
        { type: 'text', text: 'Generated image.' },
      ]);
      expect(_convertMessagesToOpenAIParams(fallback)).toEqual([
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'Generated image.' }],
        },
      ]);
      expect(JSON.stringify(fallback[0].toJSON())).not.toContain(
        'librechatServerToolResult'
      );
      expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);
      expect(originalSerialized).toContain(generatedImage.id);
      expect(originalSerialized).not.toContain('librechatServerToolResult');

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it.each([
    ['in_progress', 'AA=='],
    ['completed', ''],
  ] as const)(
    'does not promote %s v0 generated image data %j',
    (status, result) => {
      const message = new AIMessage({
        content: 'Partial answer.',
        additional_kwargs: {
          tool_outputs: [
            {
              id: 'ig_unusable',
              type: 'image_generation_call',
              status,
              result,
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });

      const [projected] = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected as AIMessage;
      const providerInput = convertMessagesToResponsesInput({
        messages: [projected],
        zdrEnabled: false,
        model: 'gpt-5.6',
      });

      expect(projectedMessage.response_metadata.output_version).toBeUndefined();
      expect(JSON.stringify(providerInput)).toContain('Partial answer.');
      expect(JSON.stringify(providerInput)).not.toContain('input_image');
      expect(JSON.stringify(providerInput)).not.toContain('ig_unusable');
    }
  );

  it('uses response output over stale v0 generated-image tool outputs', () => {
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_stale',
            type: 'image_generation_call',
            status: 'in_progress',
            result: 'AQ==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        output: [
          {
            id: 'ig_authoritative',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        ],
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: [projected],
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
    expect(JSON.stringify(providerInput)).toContain(
      'data:image/png;base64,AA=='
    );
    expect(JSON.stringify(providerInput)).not.toContain('AQ==');
    expect(JSON.stringify(providerInput)).not.toContain('ig_stale');
    expect(JSON.stringify(providerInput)).not.toContain('ig_authoritative');
  });

  it('preserves unrelated self-contained image blocks during v0 promotion', () => {
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AQ==',
      metadata: { status: 'ready' },
    };
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }, applicationImage],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_interrupted',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected as AIMessage;

    expect(projectedMessage.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      applicationImage,
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
    expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
      'ig_interrupted'
    );
  });

  it.each([
    [
      'data URL',
      { url: 'data:image/png;base64,AQ==' },
      'data:image/png;base64,AQ==',
    ],
    [
      'HTTPS URL',
      { url: 'https://example.com/application-image.png' },
      'https://example.com/application-image.png',
    ],
    [
      'file ID',
      { fileId: 'file_application_image' },
      '"file_id":"file_application_image"',
    ],
  ])(
    'preserves an unrelated v0 application image backed by a %s',
    (_label, source, providerFragment) => {
      const applicationImage = {
        type: 'image' as const,
        ...source,
        metadata: { source: 'application' },
      };
      const providerImage = {
        type: 'image' as const,
        id: 'ig_url_provider',
        url: 'data:image/png;base64,AA==',
        metadata: { status: 'completed' },
      };
      const message = new AIMessage({
        content: [
          applicationImage,
          { type: 'text', text: 'Partial answer.' },
          providerImage,
        ],
        additional_kwargs: {
          tool_outputs: [
            {
              id: providerImage.id,
              type: 'image_generation_call',
              status: 'completed',
              result: 'AA==',
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });

      const [projected] = projectOpenAIResponsesToolMessageContent([message]);
      const providerInput = convertMessagesToResponsesInput({
        messages: [projected],
        zdrEnabled: false,
        model: 'gpt-5.6',
      });

      expect(projected.content).toEqual([
        applicationImage,
        { type: 'text', text: 'Partial answer.' },
        {
          type: 'image',
          mimeType: 'image/png',
          data: 'AA==',
          extras: IMAGE_GENERATION_REPLAY_EXTRAS,
        },
      ]);
      expect(JSON.stringify(projected.toJSON())).not.toContain(
        providerImage.id
      );
      expect(JSON.stringify(providerInput)).toContain(providerFragment);
    }
  );

  it('ignores dropped v0 text placeholders when restoring image positions', () => {
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AQ==',
    };
    const message = new AIMessage({
      content: [
        { type: 'text', text: '' },
        applicationImage,
        { type: 'text', text: 'Later narration.' },
      ],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_after_placeholder',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual([
      applicationImage,
      { type: 'text', text: 'Later narration.' },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
  });

  it.each([
    ['an earlier replay item before a leading image', true],
    ['a trailing image before a later replay item', false],
  ] as const)('keeps %s during v0 promotion', (_label, imageFirst) => {
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AQ==',
    };
    const narration = { type: 'text' as const, text: 'Narration.' };
    const generatedImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'Ag==',
      extras: IMAGE_GENERATION_REPLAY_EXTRAS,
    };
    const generatedImageId = `ig_${imageFirst ? 'before' : 'after'}_image`;
    const message = new AIMessage({
      content: imageFirst
        ? [applicationImage, narration]
        : [narration, applicationImage],
      additional_kwargs: {
        tool_outputs: [
          {
            id: generatedImageId,
            type: 'image_generation_call',
            status: 'completed',
            result: generatedImage.data,
          },
        ],
        [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: imageFirst
          ? [
            {
              itemId: generatedImageId,
              kind: 'output',
              outputIndex: 0,
            },
            {
              itemId: 'msg_after_image',
              kind: 'text',
              outputIndex: 1,
              contentIndex: 0,
            },
          ]
          : [
            {
              itemId: 'msg_before_image',
              kind: 'text',
              outputIndex: 0,
              contentIndex: 0,
            },
            {
              itemId: generatedImageId,
              kind: 'output',
              outputIndex: 1,
            },
          ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual(
      imageFirst
        ? [generatedImage, applicationImage, narration]
        : [narration, applicationImage, generatedImage]
    );
  });

  it('restores v0 image positions before appending a generated image', () => {
    const imageA = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AQ==',
    };
    const imageB = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'Ag==',
    };
    const imageC = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'Aw==',
    };
    const message = new AIMessage({
      content: [
        imageA,
        { type: 'text', text: 'First.' },
        imageB,
        { type: 'text', text: 'Second.' },
        imageC,
      ],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_ordered',
            type: 'image_generation_call',
            status: 'completed',
            result: 'BA==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage.content).toEqual([
      imageA,
      { type: 'text', text: 'First.' },
      imageB,
      { type: 'text', text: 'Second.' },
      imageC,
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'BA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
    expect(providerInput).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,AQ==',
          },
          { type: 'output_text', text: 'First.', annotations: [] },
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,Ag==',
          },
          { type: 'output_text', text: 'Second.', annotations: [] },
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,Aw==',
          },
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'data:image/png;base64,BA==',
          },
        ],
      },
    ]);
    expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
      'ig_ordered'
    );
    expect(message.content).toEqual([
      imageA,
      { type: 'text', text: 'First.' },
      imageB,
      { type: 'text', text: 'Second.' },
      imageC,
    ]);

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it('keeps encrypted reasoning before image-only v0 content', () => {
    const reasoning = {
      id: 'rs_image_only',
      type: 'reasoning',
      status: 'completed',
      summary: [],
      encrypted_content: 'opaque-image-reasoning',
    };
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AQ==',
    };
    const message = new AIMessage({
      content: [applicationImage],
      additional_kwargs: {
        reasoning,
        [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
          {
            itemId: reasoning.id,
            kind: 'reasoning',
            outputIndex: 0,
          },
          {
            itemId: 'msg_image_only',
            kind: 'message',
            outputIndex: 1,
          },
          {
            itemId: 'ig_image_only',
            kind: 'output',
            outputIndex: 2,
          },
        ],
        tool_outputs: [
          {
            id: 'ig_image_only',
            type: 'image_generation_call',
            status: 'completed',
            result: 'Ag==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual([
      { type: 'non_standard', value: reasoning },
      applicationImage,
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'Ag==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
  });

  it('does not collapse identical application images during v0 promotion', () => {
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AA==',
    };
    const providerImage = {
      ...applicationImage,
      id: 'ig_duplicate',
      metadata: { status: 'completed' },
    };
    const message = new AIMessage({
      content: [applicationImage, applicationImage, providerImage],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_duplicate',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual([
      applicationImage,
      applicationImage,
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
  });

  it('keeps an id-less completed application image with generated bytes', () => {
    const applicationImage = {
      type: 'image' as const,
      mimeType: 'image/png',
      data: 'AA==',
      metadata: { source: 'application', status: 'completed' },
    };
    const message = new AIMessage({
      content: [applicationImage],
      additional_kwargs: {
        tool_outputs: [
          {
            id: 'ig_same_bytes',
            type: 'image_generation_call',
            status: 'completed',
            result: 'AA==',
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual([
      applicationImage,
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
  });

  it('deduplicates repeated authoritative generated-image items by id', () => {
    const generatedImage = {
      id: 'ig_duplicate_output',
      type: 'image_generation_call',
      status: 'completed',
      result: 'AA==',
    };
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: {
        tool_outputs: [generatedImage, { ...generatedImage }],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);

    expect(projected.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
  });

  it.each(['v0', 'v1'] as const)(
    'marks native %s code-interpreter data-url image replay without changing provider payload',
    (outputVersion) => {
      const dataUrl = 'data:image/png;base64,AA==';
      const toolOutput = {
        id: `ci_inline_${outputVersion}`,
        type: 'code_interpreter_call',
        status: 'completed',
        code: 'display(chart)',
        outputs: [{ type: 'image', url: dataUrl }],
      };
      const v0Message = new AIMessage({
        content: [{ type: 'text', text: 'Partial answer.' }],
        additional_kwargs: { tool_outputs: [toolOutput] },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const message =
        outputVersion === 'v0'
          ? v0Message
          : new AIMessage({
            contentBlocks: v0Message.contentBlocks,
            additional_kwargs: v0Message.additional_kwargs,
            response_metadata: {
              model_provider: 'openai',
              output_version: 'v1',
              preempted: true,
            },
          });
      const originalSerialized = JSON.stringify(message.toJSON());

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });

      expect(projectedMessage.content).toEqual([
        { type: 'text', text: 'Partial answer.' },
        {
          type: 'image',
          url: dataUrl,
          extras: CODE_INTERPRETER_REPLAY_EXTRAS,
        },
      ]);
      expect(providerInput).toEqual([
        {
          type: 'message',
          role: 'assistant',
          content: [
            {
              type: 'output_text',
              text: 'Partial answer.',
              annotations: [],
            },
            {
              type: 'input_image',
              detail: 'auto',
              image_url: dataUrl,
            },
          ],
        },
      ]);
      expect(JSON.stringify(providerInput)).not.toContain(
        'librechatServerToolResult'
      );
      expect(JSON.stringify(providerInput)).not.toContain('extras');

      const fallback = projectToolStreamContentForProvider(
        [message],
        'fallback'
      );
      const fallbackMessage = fallback[0] as AIMessage;
      expect(fallbackMessage.content).toEqual([
        { type: 'text', text: 'Partial answer.' },
        {
          type: 'text',
          text: JSON.stringify({
            serverToolResult: {
              librechatResponsesReplay: true,
              toolName: 'code_interpreter',
              status: 'success',
              output: {
                type: 'code_interpreter_image',
                url: dataUrl,
              },
            },
          }),
        },
      ]);
      expect(_convertMessagesToOpenAIParams(fallback)).toEqual([
        {
          role: 'assistant',
          content: fallbackMessage.content,
        },
      ]);
      expect(JSON.stringify(fallbackMessage.toJSON())).not.toContain('extras');
      expect(JSON.stringify(fallbackMessage.toJSON())).not.toContain(
        'librechatServerToolResult'
      );
      expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);
      expect(originalSerialized).toContain(toolOutput.id);
      expect(originalSerialized).not.toContain('librechatServerToolResult');

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it.each(['v0', 'v1'] as const)(
    'preserves completed %s server-tool results without replay ids',
    (outputVersion) => {
      const toolOutput = {
        id: 'ci_completed',
        type: 'code_interpreter_call',
        status: 'completed',
        code: 'print("computed")',
        outputs: [
          { type: 'logs', logs: `computed${'x'.repeat(1_000)}` },
          {
            type: 'image',
            url: 'https://example.com/ephemeral-chart.png',
          },
        ],
      };
      const v0Message = new AIMessage({
        content: [{ type: 'text', text: 'Partial answer.' }],
        additional_kwargs: { tool_outputs: [toolOutput] },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const message =
        outputVersion === 'v0'
          ? v0Message
          : new AIMessage({
            contentBlocks: v0Message.contentBlocks,
            additional_kwargs: v0Message.additional_kwargs,
            response_metadata: {
              model_provider: 'openai',
              output_version: 'v1',
              preempted: true,
            },
          });

      const projected = projectOpenAIResponsesToolMessageContent(
        [message],
        400
      );
      const projectedMessage = projected[0] as AIMessage;
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });

      expect(projectedMessage.response_metadata.output_version).toBe('v1');
      expect(projectedMessage.content).toEqual([
        { type: 'text', text: 'Partial answer.' },
        {
          type: 'text',
          text: expect.stringContaining('computed'),
          extras: {
            librechatServerToolResult: { toolName: 'code_interpreter' },
          },
        },
        {
          type: 'text',
          text: expect.stringContaining(
            'https://example.com/ephemeral-chart.png'
          ),
          extras: {
            librechatServerToolResult: { toolName: 'code_interpreter' },
          },
        },
      ]);
      expect(
        (projectedMessage.content[1] as { text: string }).text.length
      ).toBeLessThanOrEqual(400);
      const serializedProviderInput = JSON.stringify(providerInput);
      expect(serializedProviderInput).toContain('computed');
      expect(serializedProviderInput).toContain('ephemeral-chart.png');
      expect(serializedProviderInput).not.toContain('ci_completed');
      expect(serializedProviderInput).not.toContain('function_call_output');
      expect(serializedProviderInput).not.toContain('server_tool_call');
      expect(JSON.stringify(message.toJSON())).toContain('ci_completed');

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it('uses captured output positions for translated v0 server-tool results', () => {
    const toolOutputs = [
      {
        id: 'ci_position_item',
        type: 'code_interpreter_call',
        status: 'completed',
        code: 'print("CODE_BEFORE_TEXT")',
        outputs: [{ type: 'logs', logs: 'CODE_BEFORE_TEXT' }],
      },
      {
        id: 'fs_position_item',
        type: 'file_search_call',
        status: 'completed',
        queries: ['positioned file result'],
        results: [
          {
            file_id: 'file_without_provider_replay',
            filename: 'position.txt',
            score: 1,
            text: 'FILE_BETWEEN_TEXT',
          },
        ],
      },
      {
        id: 'ws_position_item',
        type: 'web_search_call',
        status: 'completed',
        action: { type: 'search', query: 'positioned web result' },
        results: [{ title: 'WEB_BETWEEN_TEXT' }],
      },
      {
        id: 'ts_position_item',
        type: 'tool_search_output',
        status: 'completed',
        tools: [
          {
            type: 'function',
            name: 'TOOL_BETWEEN_TEXT',
            description: 'positioned tool-search result',
            parameters: { type: 'object', properties: {} },
          },
        ],
      },
    ];
    const message = new AIMessage({
      content: [
        { type: 'text', text: 'First narration.' },
        { type: 'text', text: 'Second narration.' },
      ],
      additional_kwargs: {
        tool_outputs: toolOutputs,
        [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
          { itemId: 'ci_position_item', kind: 'output', outputIndex: 0 },
          {
            itemId: 'msg_first_position',
            kind: 'text',
            outputIndex: 1,
            contentIndex: 0,
          },
          { itemId: 'fs_position_item', kind: 'output', outputIndex: 2 },
          { itemId: 'ws_position_item', kind: 'output', outputIndex: 3 },
          { itemId: 'ts_position_item', kind: 'output', outputIndex: 4 },
          {
            itemId: 'msg_second_position',
            kind: 'text',
            outputIndex: 5,
            contentIndex: 0,
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });
    const originalSerialized = JSON.stringify(message.toJSON());

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const markers = [
      'CODE_BEFORE_TEXT',
      'FILE_BETWEEN_TEXT',
      'WEB_BETWEEN_TEXT',
      'TOOL_BETWEEN_TEXT',
    ];
    const orderedContent = (
      projectedMessage.content as Array<{ text?: string; type: string }>
    ).map((block) => {
      if (
        block.text === 'First narration.' ||
        block.text === 'Second narration.'
      ) {
        return block.text;
      }
      return markers.find((marker) => block.text?.includes(marker) === true);
    });
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });
    const serializedProviderInput = JSON.stringify(providerInput);

    expect(orderedContent).toEqual([
      'CODE_BEFORE_TEXT',
      'First narration.',
      'FILE_BETWEEN_TEXT',
      'WEB_BETWEEN_TEXT',
      'TOOL_BETWEEN_TEXT',
      'Second narration.',
    ]);
    expect(serializedProviderInput).toContain('WEB_BETWEEN_TEXT');
    for (const toolOutput of toolOutputs) {
      expect(serializedProviderInput).not.toContain(toolOutput.id);
    }
    expect(projectedMessage.additional_kwargs).not.toHaveProperty(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );
    expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it.each(['v0', 'v1'] as const)(
    'preserves terminal incomplete %s file-search and code-interpreter outputs',
    (outputVersion) => {
      const dataUrl = 'data:image/png;base64,AA==';
      const toolOutputs = [
        {
          id: `fs_incomplete_${outputVersion}`,
          type: 'file_search_call',
          status: 'incomplete',
          queries: ['partial result'],
          results: [
            {
              file_id: 'partial_file',
              filename: 'partial.txt',
              score: 0.5,
              text: 'PARTIAL_FILE_RESULT',
            },
          ],
        },
        {
          id: `ci_incomplete_${outputVersion}`,
          type: 'code_interpreter_call',
          status: 'incomplete',
          code: 'display(partial_chart)',
          outputs: [
            { type: 'logs', logs: 'PARTIAL_CODE_LOG' },
            { type: 'image', url: dataUrl },
          ],
        },
      ];
      const message = new AIMessage({
        content: [{ type: 'text', text: 'Partial narration.' }],
        additional_kwargs: {
          tool_outputs: toolOutputs,
          [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
            {
              itemId: toolOutputs[0].id,
              kind: 'output',
              outputIndex: 0,
            },
            {
              itemId: toolOutputs[1].id,
              kind: 'output',
              outputIndex: 1,
            },
            {
              itemId: 'msg_incomplete',
              kind: 'text',
              outputIndex: 2,
              contentIndex: 0,
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          ...(outputVersion === 'v1' ? { output_version: 'v1' } : {}),
          preempted: true,
        },
      });
      const originalSerialized = JSON.stringify(message.toJSON());

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });
      const serialized = JSON.stringify(projectedMessage.content);
      const serializedProviderInput = JSON.stringify(providerInput);
      const serverResults = (
        projectedMessage.content as Array<{ text?: string; type: string }>
      )
        .filter(
          (block) =>
            block.type === 'text' &&
            block.text?.startsWith('{"serverToolResult":') === true
        )
        .map(
          (block) =>
            (
              JSON.parse(block.text!) as {
                serverToolResult: { output: unknown; status: string };
              }
            ).serverToolResult
        );

      expect(serverResults).toEqual([
        expect.objectContaining({ status: 'error' }),
        expect.objectContaining({ status: 'error' }),
      ]);
      expect(serialized).toContain('PARTIAL_FILE_RESULT');
      expect(serialized).toContain('PARTIAL_CODE_LOG');
      expect(projectedMessage.content).toEqual(
        expect.arrayContaining([
          {
            type: 'image',
            url: dataUrl,
            extras: CODE_INTERPRETER_REPLAY_EXTRAS,
          },
        ])
      );
      expect(serializedProviderInput).toContain(dataUrl);
      expect(serializedProviderInput).not.toContain(toolOutputs[0].id);
      expect(serializedProviderInput).not.toContain(toolOutputs[1].id);
      expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it.each([
    ['logs then image', ['logs', 'image']],
    ['image then logs', ['image', 'logs']],
  ] as const)(
    'preserves code-interpreter %s order at its cross-item position',
    (_label, outputOrder) => {
      const dataUrl = 'data:image/png;base64,AQ==';
      const outputs = outputOrder.map((type) =>
        type === 'logs'
          ? { type: 'logs', logs: 'ORDERED_CODE_LOG' }
          : { type: 'image', url: dataUrl }
      );
      const message = new AIMessage({
        content: [
          { type: 'text', text: 'First narration.' },
          { type: 'text', text: 'Second narration.' },
        ],
        additional_kwargs: {
          tool_outputs: [
            {
              id: 'ci_ordered_outputs',
              type: 'code_interpreter_call',
              status: 'completed',
              code: 'display(result)',
              outputs,
            },
          ],
          [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
            {
              itemId: 'msg_order_first',
              kind: 'text',
              outputIndex: 0,
              contentIndex: 0,
            },
            {
              itemId: 'ci_ordered_outputs',
              kind: 'output',
              outputIndex: 1,
            },
            {
              itemId: 'msg_order_second',
              kind: 'text',
              outputIndex: 2,
              contentIndex: 0,
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const orderedContent = (
        projectedMessage.content as Array<{
          text?: string;
          type: string;
          url?: string;
        }>
      ).map((block) => {
        if (
          block.text === 'First narration.' ||
          block.text === 'Second narration.'
        ) {
          return block.text;
        }
        if (block.type === 'image' && block.url === dataUrl) {
          return 'image';
        }
        return block.text?.includes('ORDERED_CODE_LOG') === true
          ? 'logs'
          : undefined;
      });

      expect(orderedContent).toEqual([
        'First narration.',
        ...outputOrder,
        'Second narration.',
      ]);
      expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
        'ci_ordered_outputs'
      );
    }
  );

  it('uses response output over stale v0 server-tool results', () => {
    const makeCodeOutput = (id: string, logs: string) => ({
      id,
      type: 'code_interpreter_call',
      status: 'completed',
      code: `print(${JSON.stringify(logs)})`,
      outputs: [{ type: 'logs', logs }],
    });
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: {
        tool_outputs: [makeCodeOutput('ci_stale', 'stale result')],
      },
      response_metadata: {
        model_provider: 'openai',
        output: [makeCodeOutput('ci_authoritative', 'current result')],
        preempted: true,
      },
    });

    const [projected] = projectOpenAIResponsesToolMessageContent([message]);
    const serialized = JSON.stringify(projected.content);

    expect(serialized).toContain('current result');
    expect(serialized).not.toContain('stale result');
    expect(serialized).not.toContain('ci_authoritative');
    expect(serialized).not.toContain('ci_stale');
  });

  it.each(['v0', 'v1'] as const)(
    'preserves embedded %s MCP output without its provider id',
    (outputVersion) => {
      const mcpOutput = {
        id: 'mcp_provider_item',
        type: 'mcp_call',
        name: 'lookup',
        server_label: 'documents',
        arguments: '{"query":"answer"}',
        status: 'completed',
        output: 'MCP answer',
      };
      const v0Message = new AIMessage({
        content: [{ type: 'text', text: 'Partial answer.' }],
        additional_kwargs: { tool_outputs: [mcpOutput] },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const message =
        outputVersion === 'v0'
          ? v0Message
          : new AIMessage({
            contentBlocks: v0Message.contentBlocks,
            additional_kwargs: v0Message.additional_kwargs,
            response_metadata: {
              model_provider: 'openai',
              output_version: 'v1',
              preempted: true,
            },
          });

      const [projected] = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected as AIMessage;
      const serialized = JSON.stringify(projected.content);

      expect(projectedMessage.response_metadata.output_version).toBe('v1');
      expect(serialized).toContain('MCP answer');
      expect(serialized).toContain('documents');
      expect(serialized).not.toContain('mcp_provider_item');
      expect(JSON.stringify(message.toJSON())).toContain('mcp_provider_item');
    }
  );

  it.each(['v0', 'v1'] as const)(
    'preserves captured %s Responses results that LangChain does not convert',
    (outputVersion) => {
      const toolOutputs = [
        {
          id: 'local_output_item',
          type: 'local_shell_call_output',
          status: 'incomplete',
          output: 'local shell partial output',
        },
        {
          id: 'shell_output_item',
          call_id: 'shell_call_id',
          type: 'shell_call_output',
          status: 'incomplete',
          output: [
            {
              stdout: 'shell partial output',
              stderr: '',
              outcome: { type: 'exit', exit_code: 0 },
            },
          ],
        },
        {
          id: 'patch_output_item',
          call_id: 'patch_call_id',
          type: 'apply_patch_call_output',
          status: 'failed',
          output: 'patch failed output',
        },
        {
          id: 'program_output_item',
          call_id: 'program_call_id',
          type: 'program_output',
          status: 'incomplete',
          result: 'program partial output',
        },
        {
          id: 'mcp_list_item',
          type: 'mcp_list_tools',
          server_label: 'documents',
          tools: [
            {
              name: 'lookup',
              description: 'Look up a document',
              input_schema: { type: 'object' },
            },
          ],
          error: 'listing was incomplete',
        },
      ];
      const v0Message = new AIMessage({
        content: [{ type: 'text', text: 'Partial answer.' }],
        additional_kwargs: { tool_outputs: toolOutputs },
        response_metadata: {
          model_provider: 'openai',
          preempted: true,
        },
      });
      const message =
        outputVersion === 'v0'
          ? v0Message
          : new AIMessage({
            contentBlocks: v0Message.contentBlocks,
            additional_kwargs: v0Message.additional_kwargs,
            response_metadata: {
              model_provider: 'openai',
              output_version: 'v1',
              preempted: true,
            },
          });

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const serialized = JSON.stringify(projectedMessage.content);
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });
      const serializedProviderInput = JSON.stringify(providerInput);
      const replayStatuses = (
        projectedMessage.content as Array<{ text?: string; type: string }>
      )
        .filter(
          (block) =>
            block.type === 'text' &&
            block.text?.startsWith('{"serverToolResult":') === true
        )
        .map(
          (block) =>
            (
              JSON.parse(block.text!) as {
                serverToolResult: { status: string };
              }
            ).serverToolResult.status
        );

      expect(projectedMessage.response_metadata.output_version).toBe('v1');
      expect(serialized).toContain('local shell partial output');
      expect(serialized).toContain('shell partial output');
      expect(serialized).toContain('patch failed output');
      expect(serialized).toContain('program partial output');
      expect(serialized).toContain('listing was incomplete');
      expect(serialized).toContain('lookup');
      expect(replayStatuses).toEqual(Array(5).fill('error'));
      expect(serializedProviderInput).not.toContain('local_output_item');
      expect(serializedProviderInput).not.toContain('shell_call_id');
      expect(serializedProviderInput).not.toContain('patch_call_id');
      expect(serializedProviderInput).not.toContain('program_call_id');
      expect(serializedProviderInput).not.toContain('mcp_list_item');
      expect(serializedProviderInput).not.toContain('shell_call_output');
      expect(serializedProviderInput).not.toContain('apply_patch_call_output');
      expect(serializedProviderInput).not.toContain('program_output');
    }
  );

  it('replays captured server-tool results at their output positions', () => {
    const makeResult = (id: string, output: string) => ({
      id,
      type: 'local_shell_call_output',
      status: 'completed',
      output,
    });
    const results = [
      makeResult('result_after', 'after result'),
      {
        call_id: 'patch_without_item_id',
        type: 'apply_patch_call_output',
        status: 'completed',
        output: 'no-id patch result',
      },
      makeResult('result_before', 'before result'),
      makeResult('result_middle', 'middle result'),
    ];
    const message = new AIMessage({
      content: [
        { type: 'text', text: 'First narration.' },
        { type: 'text', text: 'Second narration.' },
      ],
      additional_kwargs: {
        tool_outputs: results,
        [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
          {
            itemId: 'result_after',
            kind: 'output',
            outputIndex: 5,
          },
          {
            itemId: 'msg_first',
            kind: 'text',
            outputIndex: 1,
            contentIndex: 0,
          },
          {
            itemId: 'result_before',
            kind: 'output',
            outputIndex: 0,
          },
          {
            itemId: 'msg_first',
            kind: 'text',
            outputIndex: 1,
            contentIndex: 0,
          },
          {
            itemId: 'result_middle',
            kind: 'output',
            outputIndex: 3,
          },
          {
            itemId: 'patch_without_item_id',
            kind: 'output',
            outputIndex: 2,
          },
          {
            itemId: 'msg_second',
            kind: 'text',
            outputIndex: 4,
            contentIndex: 0,
          },
        ],
      },
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const orderedText = (
      projectedMessage.content as Array<{
        extras?: { librechatServerToolResult?: unknown };
        text?: string;
        type: string;
      }>
    ).map((block) => {
      if (block.extras?.librechatServerToolResult != null) {
        return (
          JSON.parse(block.text!) as {
            serverToolResult: { output: string };
          }
        ).serverToolResult.output;
      }
      return block.text;
    });

    expect(orderedText).toEqual([
      'before result',
      'First narration.',
      'no-id patch result',
      'middle result',
      'Second narration.',
      'after result',
    ]);
    expect(projectedMessage.additional_kwargs).not.toHaveProperty(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );
    expect(JSON.stringify(projectedMessage.toJSON())).not.toMatch(/result_/);
    expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
      'patch_without_item_id'
    );
    expect(JSON.stringify(projectedMessage.toJSON())).not.toContain(
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    );

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it('derives id-less server-result positions from authoritative output', () => {
    const output = [
      {
        type: 'local_shell_call_output',
        status: 'completed',
        output: 'IDLESS_BEFORE_TEXT',
      },
      {
        id: 'msg_authoritative_first',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: 'First narration.',
            annotations: [],
          },
        ],
      },
      {
        id: 'msg_authoritative_empty',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [
          {
            type: 'output_text',
            text: '',
            annotations: [],
          },
        ],
      },
      {
        type: 'program_output',
        status: 'completed',
        result: 'IDLESS_BETWEEN_TEXT',
      },
      {
        id: 'msg_authoritative_second',
        type: 'message',
        role: 'assistant',
        status: 'in_progress',
        content: [
          {
            type: 'output_text',
            text: 'Second narration.',
            annotations: [],
          },
        ],
      },
    ];
    const message = new AIMessage({
      content: [
        { type: 'text', text: 'First narration.' },
        { type: 'text', text: 'Second narration.' },
      ],
      response_metadata: {
        model_provider: 'openai',
        output,
        output_version: 'v1',
        preempted: true,
      },
    });
    const originalSerialized = JSON.stringify(message.toJSON());

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const orderedContent = (
      projectedMessage.content as Array<{ text?: string; type: string }>
    ).map((block) => {
      if (
        block.text === 'First narration.' ||
        block.text === 'Second narration.'
      ) {
        return block.text;
      }
      if (block.text?.includes('IDLESS_BEFORE_TEXT') === true) {
        return 'IDLESS_BEFORE_TEXT';
      }
      return block.text?.includes('IDLESS_BETWEEN_TEXT') === true
        ? 'IDLESS_BETWEEN_TEXT'
        : undefined;
    });
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(orderedContent).toEqual([
      'IDLESS_BEFORE_TEXT',
      'First narration.',
      'IDLESS_BETWEEN_TEXT',
      'Second narration.',
    ]);
    expect(JSON.stringify(providerInput)).not.toContain(
      'local_shell_call_output'
    );
    expect(JSON.stringify(providerInput)).not.toContain('program_output');
    expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it.each(['v0', 'v1'] as const)(
    'replays encrypted %s reasoning at its captured output position',
    (outputVersion) => {
      const reasoning = {
        id: `rs_positioned_${outputVersion}`,
        type: 'reasoning',
        status: 'completed',
        summary: [],
        encrypted_content: `opaque-positioned-${outputVersion}`,
      };
      const serverResult = {
        id: `local_before_reasoning_${outputVersion}`,
        type: 'local_shell_call_output',
        status: 'completed',
        output: 'RESULT_BEFORE_REASONING',
      };
      const message = new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'unfinished summary' },
          { type: 'text', text: 'Narration after reasoning.' },
        ],
        additional_kwargs: {
          reasoning,
          tool_outputs: [serverResult],
          [OPENAI_RESPONSES_REPLAY_POSITIONS_KEY]: [
            {
              itemId: serverResult.id,
              kind: 'output',
              outputIndex: 0,
            },
            {
              itemId: reasoning.id,
              kind: 'reasoning',
              outputIndex: 1,
            },
            {
              itemId: 'msg_after_reasoning',
              kind: 'text',
              outputIndex: 2,
              contentIndex: 0,
            },
          ],
        },
        response_metadata: {
          model_provider: 'openai',
          ...(outputVersion === 'v1' ? { output_version: 'v1' } : {}),
          preempted: true,
        },
      });
      const originalSerialized = JSON.stringify(message.toJSON());

      const projected = projectOpenAIResponsesToolMessageContent([message]);
      const projectedMessage = projected[0] as AIMessage;
      const replayOrder = (
        projectedMessage.content as Array<{
          text?: string;
          type: string;
          value?: unknown;
        }>
      ).map((block) => {
        if (block.type === 'non_standard' && block.value === reasoning) {
          return 'reasoning';
        }
        if (block.text === 'Narration after reasoning.') {
          return 'narration';
        }
        return block.text?.includes('RESULT_BEFORE_REASONING') === true
          ? 'result'
          : undefined;
      });
      const providerInput = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled: false,
        model: 'gpt-5.6',
      });
      const serializedProviderInput = JSON.stringify(providerInput);
      const resultIndex = serializedProviderInput.indexOf(
        'RESULT_BEFORE_REASONING'
      );
      const reasoningIndex = serializedProviderInput.indexOf(reasoning.id);
      const narrationIndex = serializedProviderInput.indexOf(
        'Narration after reasoning.'
      );

      expect(replayOrder).toEqual(['result', 'reasoning', 'narration']);
      expect(resultIndex).toBeGreaterThanOrEqual(0);
      expect(resultIndex).toBeLessThan(reasoningIndex);
      expect(reasoningIndex).toBeLessThan(narrationIndex);
      expect(serializedProviderInput).not.toContain(serverResult.id);
      expect(projectedMessage.additional_kwargs.reasoning).toBe(reasoning);
      expect(projectedMessage.additional_kwargs).not.toHaveProperty(
        OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
      );
      expect(JSON.stringify(message.toJSON())).toBe(originalSerialized);

      const projectedAgain =
        projectOpenAIResponsesToolMessageContent(projected);
      expect(projectedAgain).toBe(projected);
      expect(projectedAgain[0]).toBe(projectedMessage);
    }
  );

  it('neutralizes provider references duplicated into v1 content blocks', () => {
    const reasoning = {
      id: 'rs_interrupted',
      type: 'reasoning',
      status: 'in_progress',
      summary: [],
    };
    const toolOutputs = [
      {
        id: 'ci_interrupted',
        type: 'code_interpreter_call',
        status: 'completed',
        code: 'print(1)',
        outputs: [{ type: 'logs', logs: '1' }],
      },
      {
        type: 'image_generation_call',
        id: 'ig_interrupted',
        status: 'completed',
        result: 'AA==',
      },
    ];
    const providerMessage = new AIMessage({
      content: [{ type: 'text', text: 'Partial answer.' }],
      additional_kwargs: { reasoning, tool_outputs: toolOutputs },
      response_metadata: { model_provider: 'openai' },
    });
    const message = new AIMessage({
      contentBlocks: providerMessage.contentBlocks,
      additional_kwargs: providerMessage.additional_kwargs,
      response_metadata: {
        id: 'resp_interrupted',
        model_provider: 'openai',
        output_version: 'v1',
        preempted: true,
      },
    });

    expect(message.content).toEqual([
      { type: 'reasoning', reasoning: '' },
      { type: 'text', text: 'Partial answer.' },
      {
        type: 'server_tool_call',
        id: 'ci_interrupted',
        name: 'code_interpreter',
        args: { code: 'print(1)' },
      },
      {
        type: 'server_tool_call_result',
        toolCallId: 'ci_interrupted',
        status: 'success',
        output: {
          type: 'code_interpreter_output',
          returnCode: 0,
          stderr: undefined,
          stdout: '1',
        },
      },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        id: 'ig_interrupted',
        metadata: { status: 'completed' },
      },
      {
        type: 'non_standard',
        value: toolOutputs[1],
      },
    ]);
    expect(message.additional_kwargs.tool_outputs).toBe(toolOutputs);

    const unsafeProviderInput = convertMessagesToResponsesInput({
      messages: [message],
      zdrEnabled: false,
      model: 'gpt-5.6',
    });
    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(unsafeProviderInput).toContainEqual(
      expect.objectContaining({ type: 'reasoning' })
    );
    expect(JSON.stringify(unsafeProviderInput)).toContain('ci_interrupted');
    expect(JSON.stringify(unsafeProviderInput)).toContain(
      'image_generation_call'
    );
    expect(JSON.stringify(providerInput)).toContain('Partial answer.');
    expect(JSON.stringify(providerInput)).toContain(
      'data:image/png;base64,AA=='
    );
    expect(JSON.stringify(providerInput)).not.toContain('rs_interrupted');
    expect(JSON.stringify(providerInput)).not.toContain('ci_interrupted');
    expect(JSON.stringify(providerInput)).not.toContain(
      'image_generation_call'
    );
    expect(projectedMessage.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      {
        type: 'text',
        text: JSON.stringify({
          serverToolResult: {
            librechatResponsesReplay: true,
            toolName: 'code_interpreter',
            status: 'success',
            output: {
              type: 'code_interpreter_output',
              returnCode: 0,
              stdout: '1',
            },
          },
        }),
        extras: {
          librechatServerToolResult: { toolName: 'code_interpreter' },
        },
      },
      {
        type: 'image',
        mimeType: 'image/png',
        data: 'AA==',
        extras: IMAGE_GENERATION_REPLAY_EXTRAS,
      },
    ]);
    expect(projectedMessage.additional_kwargs).toEqual({});
    const serialized = JSON.stringify(projectedMessage.toJSON());
    expect(serialized).toContain('Partial answer.');
    expect(serialized).toContain('AA==');
    expect(serialized).not.toContain('rs_interrupted');
    expect(serialized).not.toContain('ci_interrupted');
    expect(serialized).not.toContain('image_generation_call');
    expect(JSON.stringify(message.toJSON())).toContain('rs_interrupted');
    expect(JSON.stringify(message.toJSON())).toContain('ci_interrupted');

    const fallback = projectToolStreamContentForProvider([message], 'fallback');
    const fallbackMessage = fallback[0] as AIMessage;
    expect(fallbackMessage.content).toEqual([
      { type: 'text', text: 'Partial answer.' },
      {
        type: 'text',
        text: JSON.stringify({
          serverToolResult: {
            librechatResponsesReplay: true,
            toolName: 'code_interpreter',
            status: 'success',
            output: {
              type: 'code_interpreter_output',
              returnCode: 0,
              stdout: '1',
            },
          },
        }),
      },
    ]);
    expect(_convertMessagesToOpenAIParams(fallback)).toEqual([
      {
        role: 'assistant',
        content: fallbackMessage.content,
      },
    ]);
    expect(JSON.stringify(fallbackMessage.toJSON())).not.toContain('extras');
    expect(JSON.stringify(fallbackMessage.toJSON())).not.toContain(
      'librechatServerToolResult'
    );

    const projectedAgain = projectOpenAIResponsesToolMessageContent(projected);
    expect(projectedAgain).toBe(projected);
    expect(projectedAgain[0]).toBe(projectedMessage);
  });

  it('retains exactly one self-contained encrypted reasoning item in v1', () => {
    const reasoning = {
      id: 'rs_encrypted',
      type: 'reasoning',
      status: 'completed',
      summary: [],
      encrypted_content: 'opaque-reasoning',
    };
    const message = new AIMessage({
      content: [
        {
          type: 'reasoning',
          reasoning: 'summary',
        },
        { type: 'text', text: 'Partial answer.' },
      ],
      additional_kwargs: { reasoning },
      response_metadata: {
        model_provider: 'openai',
        output_version: 'v1',
        preempted: true,
      },
    });

    const projected = projectOpenAIResponsesToolMessageContent([message]);
    const projectedMessage = projected[0] as AIMessage;
    const providerInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(projectedMessage.content).toEqual([
      { type: 'non_standard', value: reasoning },
      { type: 'text', text: 'Partial answer.' },
    ]);
    expect(
      providerInput.filter(
        (item) => item.type === 'reasoning' && item.id === 'rs_encrypted'
      )
    ).toEqual([
      expect.objectContaining({
        id: 'rs_encrypted',
        type: 'reasoning',
        encrypted_content: 'opaque-reasoning',
      }),
    ]);
    expect(JSON.stringify(providerInput)).toContain('Partial answer.');
  });

  it('preserves Responses computer outputs handled as provider-native media', () => {
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            id: 'computer-item',
            call_id: 'call_computer',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: [
        {
          type: 'computer_screenshot',
          image_url: 'data:image/png;base64,AA==',
        },
      ],
      tool_call_id: 'call_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const messages = [computerCall, computerOutput];
    const projected = projectOpenAIToolMessageContent(messages, 10);
    const responsesInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'computer-use-preview',
    });

    expect(projected).not.toBe(messages);
    expect(
      responsesInput.find((item) => item.type === 'computer_call_output')
    ).toMatchObject({
      type: 'computer_call_output',
      call_id: 'call_computer',
      output: {
        type: 'computer_screenshot',
        image_url: 'data:image/png;base64,AA==',
      },
    });
  });

  it('deduplicates parsed and raw Responses computer calls in ZDR mode', () => {
    const response = {
      id: 'resp_computer',
      object: 'response',
      created_at: 0,
      model: 'computer-use-preview',
      output: [
        {
          type: 'computer_call',
          id: 'computer-item',
          call_id: 'call_computer',
          action: { type: 'screenshot' },
          status: 'completed',
        },
      ],
      status: 'completed',
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        total_tokens: 2,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 0 },
      },
      error: null,
      incomplete_details: null,
      metadata: {},
      user: null,
      service_tier: 'default',
    } as unknown as Parameters<typeof convertResponsesMessageToAIMessage>[0];
    const computerCall = convertResponsesMessageToAIMessage(response);
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });

    const projected = projectOpenAIResponsesToolMessageContent([
      computerCall,
      computerOutput,
    ]);
    const zdrInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: true,
      model: 'computer-use-preview',
    });
    const retainedInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'computer-use-preview',
    });

    expect(computerCall.tool_calls).toHaveLength(1);
    expect((projected[0] as AIMessage).tool_calls).toHaveLength(0);
    for (const input of [zdrInput, retainedInput]) {
      expect(
        input.filter(
          (item) =>
            item.type === 'computer_call' && item.call_id === 'call_computer'
        )
      ).toHaveLength(1);
      expect(
        input.filter(
          (item) =>
            item.type === 'computer_call_output' &&
            item.call_id === 'call_computer'
        )
      ).toHaveLength(1);
    }
  });

  it('deduplicates streaming Responses computer calls before replay', () => {
    const event = {
      type: 'response.output_item.done',
      sequence_number: 1,
      output_index: 0,
      item: {
        type: 'computer_call',
        id: 'computer-stream-item',
        call_id: 'call_computer_stream',
        action: { type: 'screenshot' },
        status: 'completed',
      },
    } as unknown as Parameters<
      typeof convertResponsesDeltaToChatGenerationChunk
    >[0];
    const generation = convertResponsesDeltaToChatGenerationChunk(event);
    const computerCall = generation?.message;
    expect(computerCall).toBeDefined();

    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_computer_stream',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const projected = projectOpenAIResponsesToolMessageContent([
      computerCall!,
      computerOutput,
    ]);

    expect((computerCall as AIMessage).tool_calls).toHaveLength(1);
    expect((computerCall as AIMessageChunk).tool_call_chunks).toHaveLength(1);
    expect((projected[0] as AIMessage).tool_calls).toHaveLength(0);
    expect((projected[0] as AIMessageChunk).tool_call_chunks).toHaveLength(0);
    expect(Object.getPrototypeOf(projected[0])).toBe(
      Object.getPrototypeOf(computerCall)
    );
    expect(
      new AIMessageChunk(projected[0] as AIMessageChunk).tool_calls
    ).toHaveLength(0);
    for (const zdrEnabled of [true, false]) {
      const input = convertMessagesToResponsesInput({
        messages: projected,
        zdrEnabled,
        model: 'computer-use-preview',
      });
      expect(
        input.filter(
          (item) =>
            item.type === 'computer_call' &&
            item.call_id === 'call_computer_stream'
        )
      ).toHaveLength(1);
      expect(
        input.filter(
          (item) =>
            (item.type === 'function_call' || item.type === 'computer_call') &&
            item.call_id === 'call_computer_stream'
        )
      ).toHaveLength(1);
    }

    const chatProjected = projectComputerCallOutputsToText(
      projectOpenAIToolMessageContent([computerCall!, computerOutput])
    );
    const chatInput = convertMessagesToCompletionsMessageParams({
      messages: chatProjected,
      model: 'gpt-4o',
    });

    expect((chatProjected[0] as AIMessage).tool_calls).toHaveLength(1);
    expect(chatInput).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          expect.objectContaining({
            id: 'call_computer_stream',
            type: 'function',
          }),
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        tool_call_id: 'call_computer_stream',
        content: '[Computer screenshot omitted for this provider]',
      }),
    ]);
  });

  it('marks a provider-only computer screenshot omission as synthetic', () => {
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_computer_omitted',
      additional_kwargs: { type: 'computer_call_output' },
    });
    setProviderMessageProvenance(computerOutput, [
      {
        attribution: 'tool',
        sourceMessageId: 'computer-output-row',
        sourceContentPartIndices: [0],
      },
    ]);

    const messages = [computerOutput];
    const projected = projectComputerCallOutputsToText(messages);

    expect(projected).not.toBe(messages);
    expect(projected[0].content).toBe(
      '[Computer screenshot omitted for this provider]'
    );
    expect(getProviderMessageProvenance(projected[0])?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
    expect(getProviderMessageProvenance(computerOutput)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'computer-output-row',
        sourceContentPartIndices: [0],
      },
    ]);
  });

  it('canonicalizes real OpenAI file-backed computer screenshots on the production converter path', () => {
    const computerCall = new AIMessage({
      content: '',
      additional_kwargs: {
        tool_outputs: [
          {
            type: 'computer_call',
            call_id: 'call_computer_file',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: [
        {
          type: 'input_image',
          file_id: 'file-abc123',
          detail: 'low',
        },
      ],
      tool_call_id: 'call_computer_file',
      additional_kwargs: { type: 'computer_call_output' },
    });

    const projected = projectOpenAIToolMessageContent(
      [computerCall, computerOutput],
      100
    );
    const responsesInput = convertMessagesToResponsesInput({
      messages: projected,
      zdrEnabled: false,
      model: 'computer-use-preview',
    });

    expect(responsesInput).toContainEqual({
      type: 'computer_call_output',
      call_id: 'call_computer_file',
      output: {
        type: 'input_image',
        file_id: 'file-abc123',
        detail: 'low',
      },
    });
  });

  it('rejects extra invalid screenshot fields instead of shipping the original block', () => {
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            call_id: 'call_computer_extra',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const malformed = new ToolMessage({
      content: [
        {
          type: 'input_image',
          image_url: 'data:image/png;base64,AA==',
          file_id: 'not-a-file-id',
        },
      ],
      tool_call_id: 'call_computer_extra',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(() =>
      projectOpenAIToolMessageContent([computerCall, malformed], 100)
    ).toThrow('Invalid computer call output screenshot');
  });

  it('rejects computer outputs that precede their call', () => {
    const computerCall = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            call_id: 'call_computer_order',
            action: { type: 'screenshot' },
          },
        ],
      },
    });
    const computerOutput = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_computer_order',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(() =>
      projectOpenAIToolMessageContent([computerOutput, computerCall], 100)
    ).toThrow('Invalid computer call output pairing');
  });

  it('rejects a malformed marked computer output before provider conversion', () => {
    const malformed = new ToolMessage({
      content: [{ type: 'text', text: 'A'.repeat(2_000) }],
      tool_call_id: 'call_computer_malformed',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(() => projectOpenAIToolMessageContent([malformed], 100)).toThrow(
      'Invalid computer call output screenshot'
    );
  });

  it('rejects a marked screenshot paired with a normal function call', () => {
    const functionCall = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_function',
          name: 'render',
          args: {},
          type: 'tool_call',
        },
      ],
    });
    const malformedPair = new ToolMessage({
      content: 'data:image/png;base64,AA==',
      tool_call_id: 'call_function',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(() =>
      projectOpenAIToolMessageContent([functionCall, malformedPair], 100)
    ).toThrow('Invalid computer call output pairing');
  });

  it('keeps the mutating artifact formatter API backward compatible', () => {
    const anthropicToolMessage = new ToolMessage({
      content: 'result',
      tool_call_id: 'anthropic_legacy',
      artifact: {
        content: [{ type: ContentTypes.TEXT, text: 'anthropic artifact' }],
      },
    });
    const anthropicMessages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'anthropic_legacy',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      anthropicToolMessage,
    ];

    expect(formatAnthropicArtifactContent(anthropicMessages)).toBeUndefined();
    expect(anthropicToolMessage.content).toEqual([
      { type: ContentTypes.TEXT, text: 'result' },
      { type: ContentTypes.TEXT, text: 'anthropic artifact' },
    ]);
    expect(anthropicToolMessage.artifact.content).toHaveLength(1);

    const payloadToolMessage = new ToolMessage({
      content: 'result',
      tool_call_id: 'payload_legacy',
      artifact: {
        content: [{ type: ContentTypes.TEXT, text: 'payload artifact' }],
      },
    });
    const payloadMessages = [
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'payload_legacy',
            name: 'render',
            args: {},
            type: 'tool_call' as const,
          },
        ],
      }),
      payloadToolMessage,
    ];

    expect(formatArtifactPayload(payloadMessages)).toBeUndefined();
    expect(payloadToolMessage.content).toContain(
      'Tool response is included in the next message'
    );
    expect(payloadMessages[payloadMessages.length - 1]).toBeInstanceOf(
      HumanMessage
    );
    expect(payloadToolMessage.artifact.content).toHaveLength(1);
  });

  it('should dynamically discover tools from tool_search output and keep their tool calls', () => {
    const tools = new Set(['tool_search', 'calculator']);
    const payload = [
      {
        role: 'user',
        content: 'Search for commits and list them',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for tools first.',
            tool_call_ids: ['ts_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'ts_1',
              name: 'tool_search',
              args: '{"query":"commits"}',
              output: '{"found": 1, "tools": [{"name": "list_commits"}]}',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now listing commits.',
            tool_call_ids: ['lc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_1',
              name: 'list_commits',
              args: '{"repo":"test"}',
              output: '[{"sha":"abc123"}]',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here are the results.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Thanks!',
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /**
     * Since tool_search discovered list_commits, both should be kept.
     * The dynamic discovery adds list_commits to the valid tools set.
     */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(2);

    const toolNames = toolMessages.map((m) => m.name).sort();
    expect(toolNames).toEqual(['list_commits', 'tool_search']);
  });

  it('should filter out tool calls not in set and not discovered by tool_search', () => {
    const tools = new Set(['tool_search', 'calculator']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll call an unknown tool.',
            tool_call_ids: ['uk_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'uk_1',
              name: 'unknown_tool',
              args: '{}',
              output: 'result',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Done.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** unknown_tool should be filtered out since it's not in tools set and not discovered */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(0);
  });

  it('should keep all tool calls when all are in the tools set', () => {
    const tools = new Set(['search', 'calculator']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me help.',
            tool_call_ids: ['s1', 'c1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 's1',
              name: 'search',
              args: '{"q":"test"}',
              output: 'Search results',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'c1',
              name: 'calculator',
              args: '{"expr":"2+2"}',
              output: '4',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(2);
    expect(toolMessages.map((m) => m.name).sort()).toEqual([
      'calculator',
      'search',
    ]);
  });

  it('should preserve discovered tools across multiple assistant messages', () => {
    /**
     * This test verifies that once tool_search discovers a tool, it remains valid
     * for all subsequent messages in the conversation, not just the current message.
     */
    const tools = new Set(['tool_search']);
    const payload = [
      {
        role: 'user',
        content: 'Find me a tool to list commits and use it',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me search for that tool.',
            tool_call_ids: ['ts_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'ts_1',
              name: 'tool_search',
              args: '{"query":"commits"}',
              output:
                '{"found": 1, "tools": [{"name": "list_commits_mcp_github"}]}',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now using the discovered tool.',
            tool_call_ids: ['lc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_1',
              name: 'list_commits_mcp_github',
              args: '{"repo":"test"}',
              output: '[{"sha":"abc123","message":"Initial commit"}]',
            },
          },
        ],
      },
      {
        role: 'user',
        content: 'Show me more commits',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Fetching more commits.',
            tool_call_ids: ['lc_2'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_2',
              name: 'list_commits_mcp_github',
              args: '{"repo":"test","page":2}',
              output: '[{"sha":"def456","message":"Second commit"}]',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** All three tool calls should be preserved as ToolMessages */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];

    expect(toolMessages.length).toBe(3);
    expect(toolMessages[0].name).toBe('tool_search');
    expect(toolMessages[1].name).toBe('list_commits_mcp_github');
    expect(toolMessages[2].name).toBe('list_commits_mcp_github');
  });

  it('omits invalid tools while keeping valid tools as ToolMessages', () => {
    /**
     * This test documents the hybrid behavior:
     * - Valid tools remain as proper AIMessage + ToolMessage structures
     * - Invalid tools and their outputs are omitted from model context
     */
    const tools = new Set(['calculator']);
    const payload = [
      {
        messageId: 'filtered-tool-row',
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will use two tools.',
            tool_call_ids: ['calc_1', 'unknown_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expr":"2+2"}',
              output: '4',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'unknown_1',
              name: 'some_unknown_tool',
              args: '{"query":"test"}',
              output: 'This is the result from unknown tool',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** Should have AIMessage + ToolMessage for calculator */
    expect(result.messages.length).toBe(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    /** The valid tool should be kept */
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );
    expect((result.messages[1] as ToolMessage).name).toBe('calculator');

    /** The invalid tool output must not become assistant-authored content */
    const aiContent = result.messages[0].content;
    const aiContentStr =
      typeof aiContent === 'string' ? aiContent : JSON.stringify(aiContent);
    expect(aiContentStr).not.toContain('some_unknown_tool');
    expect(aiContentStr).not.toContain('This is the result from unknown tool');
    expect(result.messages[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'filtered-tool-row',
          sourceContentPartIndices: [0, 1],
        },
      ],
    });
    expect(result.messages[1].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'filtered-tool-row',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('omits an output-less unavailable tool call', () => {
    const { messages } = formatAgentMessages(
      [
        {
          role: 'assistant',
          messageId: 'output-less-filtered-tool-row',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'unavailable-tool',
                name: 'unavailable_tool',
                args: '{}',
                output: '',
              },
            },
          ],
        },
      ],
      undefined,
      new Set(['available_tool'])
    );

    expect(messages).toHaveLength(0);
  });

  it('should simulate realistic deferred tools flow with tool_search', () => {
    /**
     * This test simulates the real-world use case:
     * 1. Agent only has tool_search initially (deferred tools not in set)
     * 2. User asks to do something that requires a deferred tool
     * 3. Agent uses tool_search to discover the tool
     * 4. Agent then uses the discovered tool
     * 5. On subsequent conversation turns, both tool calls should be valid
     */
    const tools = new Set(['tool_search', 'execute_code']);
    const payload = [
      { role: 'user', content: 'List the recent commits from the repo' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]:
              'I need to find a tool for listing commits. Let me search.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'tool_search',
              args: '{"query":"git commits list"}',
              output:
                '{\n  "found": 1,\n  "tools": [\n    {\n      "name": "list_commits_mcp_github",\n      "score": 0.95,\n      "matched_in": "name",\n      "snippet": "Lists commits from a GitHub repository"\n    }\n  ],\n  "total_searched": 50,\n  "query": "git commits list"\n}',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Found the tool! Now I will list the commits.',
            tool_call_ids: ['commits_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'commits_1',
              name: 'list_commits_mcp_github',
              args: '{"owner":"librechat","repo":"librechat"}',
              output:
                '[{"sha":"abc123","message":"feat: add deferred tools"},{"sha":"def456","message":"fix: tool loading"}]',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** Both tool_search and list_commits_mcp_github should be preserved */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];

    expect(toolMessages.length).toBe(2);
    expect(toolMessages[0].name).toBe('tool_search');
    expect(toolMessages[1].name).toBe('list_commits_mcp_github');

    /** The AI messages should have proper tool_calls */
    const aiMessages = result.messages.filter(
      (m) => m._getType() === 'ai'
    ) as AIMessage[];

    const toolCallNames = aiMessages.flatMap(
      (m) => m.tool_calls?.map((tc) => tc.name) ?? []
    );
    expect(toolCallNames).toContain('tool_search');
    expect(toolCallNames).toContain('list_commits_mcp_github');
  });

  it.skip('should not produce two consecutive assistant messages and format content correctly', () => {
    const payload = [
      { role: 'user', content: 'Hello' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hi there!' },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'How can I help you?',
          },
        ],
      },
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
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
            [ContentTypes.TEXT]: 'Here\'s the weather information.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Check correct message count and types
    expect(result.messages).toHaveLength(6);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(HumanMessage);
    expect(result.messages[3]).toBeInstanceOf(AIMessage);
    expect(result.messages[4]).toBeInstanceOf(ToolMessage);
    expect(result.messages[5]).toBeInstanceOf(AIMessage);

    // Check content of messages
    expect(result.messages[0].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Hello', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[1].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Hi there!', type: ContentTypes.TEXT },
      { [ContentTypes.TEXT]: 'How can I help you?', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[2].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'What\'s the weather?', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[3].content).toBe('Let me check that for you.');
    expect(result.messages[4].content).toBe('Sunny, 75°F');
    expect(result.messages[5].content).toStrictEqual([
      {
        [ContentTypes.TEXT]: 'Here\'s the weather information.',
        type: ContentTypes.TEXT,
      },
    ]);

    // Check that there are no consecutive AIMessages
    const messageTypes = result.messages.map((message) => message.constructor);
    for (let i = 0; i < messageTypes.length - 1; i++) {
      expect(
        messageTypes[i] === AIMessage && messageTypes[i + 1] === AIMessage
      ).toBe(false);
    }

    // Additional check to ensure the consecutive assistant messages were combined
    expect(result.messages[1].content).toHaveLength(2);
  });

  it('should strip THINK content and join TEXT parts as string', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Initial response' },
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Reasoning about the problem...',
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toEqual(
      'Initial response\nFinal answer'
    );
  });

  it('should join TEXT content as string when THINK content type is present', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Analyzing the problem...',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'First part of response',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Second part of response',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Final part of response',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(typeof result.messages[0].content).toBe('string');
    expect(result.messages[0].content).toBe(
      'First part of response\nSecond part of response\nFinal part of response'
    );
    expect(result.messages[0].content).not.toContain(
      'Analyzing the problem...'
    );
  });

  it('should strip reasoning_content blocks and join TEXT parts as string', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '\n\n' },
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: { text: 'Thinking deeply...', signature: 'sig123' },
            index: 0,
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'The answer is 42.' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toBe('The answer is 42.');
    expect(JSON.stringify(result.messages[0].content)).not.toContain(
      'reasoning_content'
    );
  });

  it('should preserve hidden reasoning_content for DeepSeek assistant messages', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Need calculator.',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Using calculator.',
            tool_call_ids: ['call_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_1',
              name: 'calculator',
              args: '{"input":"127 * 453"}',
              output: '57531',
            },
          },
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Calculator returned 57531.',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: '127 * 453 = 57531.',
          },
        ],
      },
    ];

    const defaultResult = formatAgentMessages(payload);
    expect(
      (defaultResult.messages[0] as AIMessage).additional_kwargs
        .reasoning_content
    ).toBeUndefined();

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { provider: Providers.DEEPSEEK }
    );

    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);

    const toolCallMessage = result.messages[0] as AIMessage;
    const finalMessage = result.messages[2] as AIMessage;

    expect(toolCallMessage.content).toBe('Using calculator.');
    expect(toolCallMessage.tool_calls).toHaveLength(1);
    expect(toolCallMessage.additional_kwargs.reasoning_content).toBe(
      'Need calculator.'
    );
    expect(finalMessage.content).toBe('127 * 453 = 57531.');
    expect(finalMessage.additional_kwargs.reasoning_content).toBe(
      'Calculator returned 57531.'
    );
  });

  it('should preserve hidden reasoning_content via explicit preserveReasoningContent without a provider', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Need calculator.',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Using calculator.',
            tool_call_ids: ['call_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_1',
              name: 'calculator',
              args: '{"input":"127 * 453"}',
              output: '57531',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
      }
    );

    const toolCallMessage = result.messages[0] as AIMessage;
    expect(toolCallMessage.tool_calls).toHaveLength(1);
    expect(toolCallMessage.additional_kwargs.reasoning_content).toBe(
      'Need calculator.'
    );
  });

  it('preserves reasoning text from persisted stream delta blocks', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking_delta', thinking: 'First thought. ' },
          { type: 'reasoning-delta', reasoning: 'Second thought.' },
          { type: ContentTypes.TEXT, text: 'Answer.' },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { preserveReasoningContent: true }
    );

    expect(result.messages[0].additional_kwargs.reasoning_content).toBe(
      'First thought. Second thought.'
    );
  });

  it('should not reconstruct reasoning_content when preserveReasoningContent is explicitly false for DeepSeek', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Need calculator.',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Using calculator.',
            tool_call_ids: ['call_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_1',
              name: 'calculator',
              args: '{"input":"127 * 453"}',
              output: '57531',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      {
        provider: Providers.DEEPSEEK,
        preserveReasoningContent: false,
      }
    );

    const toolCallMessage = result.messages[0] as AIMessage;
    expect(toolCallMessage.additional_kwargs.reasoning_content).toBeUndefined();
  });

  it('should preserve DeepSeek reasoning from supported hidden content blocks', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Think. ',
          },
          {
            type: ContentTypes.THINKING,
            thinking: 'Thinking. ',
          },
          {
            type: ContentTypes.REASONING,
            reasoning: 'Reasoning. ',
          },
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: { text: 'Reasoning content.' },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Done.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { provider: Providers.DEEPSEEK }
    );

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toBe('Done.');
    expect(
      (result.messages[0] as AIMessage).additional_kwargs.reasoning_content
    ).toBe('Think. Thinking. Reasoning. Reasoning content.');
  });

  it('should attach later DeepSeek reasoning to an existing tool-call assistant message', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Need calculator. ',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Using calculator.',
            tool_call_ids: ['call_1'],
          },
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Preparing tool call.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_1',
              name: 'calculator',
              args: '{"input":"127 * 453"}',
              output: '57531',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { provider: Providers.DEEPSEEK }
    );

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    const toolCallMessage = result.messages[0] as AIMessage;

    expect(toolCallMessage.content).toBe('Using calculator.');
    expect(toolCallMessage.tool_calls).toHaveLength(1);
    expect(toolCallMessage.additional_kwargs.reasoning_content).toBe(
      'Need calculator. Preparing tool call.'
    );
  });

  it('should strip thinking blocks and join TEXT parts as string', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINKING,
            thinking: 'Internal reasoning...',
            signature: 'sig456',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here is my answer.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toBe('Here is my answer.');
    expect(JSON.stringify(result.messages[0].content)).not.toContain(
      'thinking'
    );
  });

  it('should strip redacted_thinking blocks and join TEXT parts as string', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: 'redacted_thinking', data: 'REDACTED_SIGNATURE' },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here is my answer.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toBe('Here is my answer.');
    expect(JSON.stringify(result.messages[0].content)).not.toContain(
      'redacted_thinking'
    );
  });

  it('should produce no AIMessage when only reasoning_content and whitespace text are present', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '\n\n' },
          {
            type: ContentTypes.REASONING_CONTENT,
            reasoningText: { text: 'Silent reasoning', signature: 'sig' },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(0);
  });

  it('should drop whitespace-only text parts from non-reasoning messages', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '\n\n' },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Actual content here.',
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '   ' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    const content = result.messages[0].content;
    expect(Array.isArray(content)).toBe(true);
    expect(
      (content as { type: string; text?: string }[]).every(
        (p) => (p.text ?? '').trim() !== ''
      )
    ).toBe(true);
  });

  it('should preserve whitespace-only text that has tool_call_ids (common Bedrock pattern)', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: '\n\n',
            tool_call_ids: ['tc-1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tc-1',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Results here',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tc-1');
  });

  it('should handle whitespace-only text without tool_call_ids before a tool call', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: '\n\n' },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tc-2',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Results here',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
  });

  it('should exclude ERROR type content parts', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello there' },
          {
            type: ContentTypes.ERROR,
            [ContentTypes.ERROR]:
              'An error occurred while processing the request: Something went wrong',
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toEqual([
      { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello there' },
      { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
    ]);

    const hasErrorContent =
      Array.isArray(result.messages[0].content) &&
      result.messages[0].content.some(
        (item) =>
          item.type === ContentTypes.ERROR ||
          JSON.stringify(item).includes('An error occurred')
      );
    expect(hasErrorContent).toBe(false);
  });
  it('should handle indexTokenCountMap and return updated map', () => {
    const payload = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ];

    const indexTokenCountMap = {
      0: 5, // 5 tokens for "Hello"
      1: 10, // 10 tokens for "Hi there!"
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    expect(result.messages).toHaveLength(2);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(result.indexTokenCountMap?.[0]).toBe(5);
    expect(result.indexTokenCountMap?.[1]).toBe(10);
  });

  it('should handle complex message transformations with indexTokenCountMap', () => {
    const payload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
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
      0: 10, // 10 tokens for "What's the weather?"
      1: 50, // 50 tokens for the assistant message with tool call
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // The original message at index 1 should be split into two messages
    expect(result.messages).toHaveLength(3);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(result.indexTokenCountMap?.[0]).toBe(10); // User message stays the same

    // The assistant message tokens should be distributed across the resulting messages
    const totalAssistantTokens =
      Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, count) => sum + count,
        0
      ) - 10; // Subtract user message tokens

    expect(totalAssistantTokens).toBe(50); // Should match the original token count
  });

  it('should handle one-to-many message expansion with tool calls', () => {
    // One message with multiple tool calls expands to multiple messages
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'First tool call:',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Search result',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Second tool call:',
            tool_call_ids: ['tool_2'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_2',
              name: 'calculate',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Final response',
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 100, // 100 tokens for the complex assistant message
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // One message expands to 5 messages (2 tool calls + text before, between, and after)
    expect(result.messages).toHaveLength(5);
    expect(result.indexTokenCountMap).toBeDefined();

    // The sum of all token counts should equal the original
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(100);

    // Check that each resulting message has a token count
    for (let i = 0; i < result.messages.length; i++) {
      expect(result.indexTokenCountMap?.[i]).toBeDefined();
    }
  });

  it('should handle content filtering that reduces message count', () => {
    // Message with THINK and ERROR parts that get filtered out
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.THINK, [ContentTypes.THINK]: 'Thinking...' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Visible response' },
          { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'Error occurred' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 60, // 60 tokens for the message with filtered content
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // Only one message should remain after filtering
    expect(result.messages).toHaveLength(1);
    expect(result.indexTokenCountMap).toBeDefined();

    // All tokens should be assigned to the remaining message
    expect(result.indexTokenCountMap?.[0]).toBe(60);
  });

  it('should handle empty result after content filtering', () => {
    // Message with only THINK and ERROR parts that all get filtered out
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.THINK, [ContentTypes.THINK]: 'Thinking...' },
          { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'Error occurred' },
          { type: ContentTypes.AGENT_UPDATE, update: 'Processing...' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 40, // 40 tokens for the message with filtered content
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // No messages should remain after filtering
    expect(result.messages).toHaveLength(0);
    expect(result.indexTokenCountMap).toBeDefined();

    // The token count map should be empty since there are no messages
    expect(Object.keys(result.indexTokenCountMap || {})).toHaveLength(0);
  });

  it('should demonstrate how 2 input messages can become more than 2 output messages', () => {
    // Two input messages where one contains tool calls
    const payload = [
      { role: 'user', content: 'Can you help me with something?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll help you with that.',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"help topics"}',
              output: 'Found several help topics.',
            },
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 15, // 15 tokens for the user message
      1: 45, // 45 tokens for the assistant message with tool call
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // 2 input messages become 3 output messages (user + assistant + tool)
    expect(payload).toHaveLength(2);
    expect(result.messages).toHaveLength(3);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(Object.keys(result.indexTokenCountMap ?? {}).length).toBe(3);

    // Check message types
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);

    // The sum of all token counts should equal the original total
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(60); // 15 + 45
  });

  it('should handle an AI message with 5 tool calls in a single message', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll perform multiple operations for you.',
            tool_call_ids: ['tool_1', 'tool_2', 'tool_3', 'tool_4', 'tool_5'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"latest news"}',
              output: 'Found several news articles.',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_2',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_3',
              name: 'calculate',
              args: '{"expression":"356 * 24"}',
              output: '8544',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_4',
              name: 'translate',
              args: '{"text":"Hello world","source":"en","target":"fr"}',
              output: 'Bonjour le monde',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_5',
              name: 'fetch_data',
              args: '{"endpoint":"/api/users","params":{"limit":5}}',
              output:
                '{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"},{"id":3,"name":"Charlie"},{"id":4,"name":"David"},{"id":5,"name":"Eve"}]}',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 6 messages: 1 AIMessage and 5 ToolMessages
    expect(result.messages).toHaveLength(6);

    // Check message types in the correct sequence
    expect(result.messages[0]).toBeInstanceOf(AIMessage); // Initial message with all tool calls
    expect(result.messages[1]).toBeInstanceOf(ToolMessage); // Tool 1 response
    expect(result.messages[2]).toBeInstanceOf(ToolMessage); // Tool 2 response
    expect(result.messages[3]).toBeInstanceOf(ToolMessage); // Tool 3 response
    expect(result.messages[4]).toBeInstanceOf(ToolMessage); // Tool 4 response
    expect(result.messages[5]).toBeInstanceOf(ToolMessage); // Tool 5 response

    // Check AIMessage has all 5 tool calls
    expect(result.messages[0].content).toBe(
      'I\'ll perform multiple operations for you.'
    );
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(5);

    // Verify each tool call in the AIMessage
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'tool_1',
      name: 'search',
      args: { query: 'latest news' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[1]).toEqual({
      id: 'tool_2',
      name: 'check_weather',
      args: { location: 'New York' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[2]).toEqual({
      id: 'tool_3',
      name: 'calculate',
      args: { expression: '356 * 24' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[3]).toEqual({
      id: 'tool_4',
      name: 'translate',
      args: { text: 'Hello world', source: 'en', target: 'fr' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[4]).toEqual({
      id: 'tool_5',
      name: 'fetch_data',
      args: { endpoint: '/api/users', params: { limit: 5 } },
    });

    // Check each ToolMessage
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('Found several news articles.');

    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('tool_2');
    expect(result.messages[2].name).toBe('check_weather');
    expect(result.messages[2].content).toBe('Sunny, 75°F');

    expect((result.messages[3] as ToolMessage).tool_call_id).toBe('tool_3');
    expect(result.messages[3].name).toBe('calculate');
    expect(result.messages[3].content).toBe('8544');

    expect((result.messages[4] as ToolMessage).tool_call_id).toBe('tool_4');
    expect(result.messages[4].name).toBe('translate');
    expect(result.messages[4].content).toBe('Bonjour le monde');

    expect((result.messages[5] as ToolMessage).tool_call_id).toBe('tool_5');
    expect(result.messages[5].name).toBe('fetch_data');
    expect(result.messages[5].content).toBe(
      '{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"},{"id":3,"name":"Charlie"},{"id":4,"name":"David"},{"id":5,"name":"Eve"}]}'
    );
  });

  it('should heal tool call structure with thinking content', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]:
              'I\'ll add this agreement as an observation to our existing troubleshooting task in the project memory system.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tooluse_Zz-mw_wHTrWTvDHaCbfaZg',
              name: 'add_observations_mcp_project-memory',
              args: '{"observations":[{"entityName":"MCP_Tool_Error_Troubleshooting","contents":["Agreement established: Document all future tests in the project memory system to maintain a comprehensive troubleshooting log","This will provide a structured record of the entire troubleshooting process and help identify patterns in the error behavior"]}]}',
              type: 'tool_call',
              progress: 1,
              output:
                '[\n  {\n    "entityName": "MCP_Tool_Error_Troubleshooting",\n    "addedObservations": [\n      {\n        "content": "Agreement established: Document all future tests in the project memory system to maintain a comprehensive troubleshooting log",\n        "timestamp": "2025-03-26T00:46:42.154Z"\n      },\n      {\n        "content": "This will provide a structured record of the entire troubleshooting process and help identify patterns in the error behavior",\n        "timestamp": "2025-03-26T00:46:42.154Z"\n      }\n    ]\n  }\n]',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]:
              '\n\nI\'ve successfully added our agreement to the project memory system. The observation has been recorded in the "MCP_Tool_Error_Troubleshooting" entity with the current timestamp.\n\nGoing forward, I will:\n\n1. Document each test we perform\n2. Record the methodology and results\n3. Update the project memory with our findings\n4. Establish appropriate relationships between tests and related components\n5. Provide a summary of what we\'ve learned from each test\n\nThis structured approach will help us build a comprehensive knowledge base of the error behavior and our troubleshooting process, which may prove valuable for resolving similar issues in the future or for other developers facing similar challenges.\n\nWhat test would you like to perform next in our troubleshooting process?',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 3 messages: an AIMessage with empty content, a ToolMessage, and a final AIMessage with the text
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);

    // The first AIMessage should have an empty content and the tool_call
    expect(result.messages[0].content).toBe('');
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'add_observations_mcp_project-memory'
    );

    // The ToolMessage should have the correct properties
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe(
      'tooluse_Zz-mw_wHTrWTvDHaCbfaZg'
    );
    expect(result.messages[1].name).toBe('add_observations_mcp_project-memory');
    expect(result.messages[1].content).toContain(
      'MCP_Tool_Error_Troubleshooting'
    );

    // The final AIMessage should contain the text response
    expect(typeof result.messages[2].content).toBe('string');
    expect((result.messages[2].content as string).trim()).toContain(
      'I\'ve successfully added our agreement to the project memory system'
    );
  });

  it('should demonstrate how messages can be filtered out, reducing count', () => {
    // Two input messages where one gets completely filtered out
    const payload = [
      { role: 'user', content: 'Hello there' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Thinking about response...',
          },
          {
            type: ContentTypes.ERROR,
            [ContentTypes.ERROR]: 'Error in processing',
          },
          { type: ContentTypes.AGENT_UPDATE, update: 'Working on it...' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 10, // 10 tokens for the user message
      1: 30, // 30 tokens for the assistant message that will be filtered out
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // 2 input messages become 1 output message (only the user message remains)
    expect(payload).toHaveLength(2);
    expect(result.messages).toHaveLength(1);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(Object.keys(result.indexTokenCountMap ?? {}).length).toBe(1);

    // Check message type
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);

    // Only the user message tokens should remain
    expect(result.indexTokenCountMap?.[0]).toBe(10);

    // The total tokens should be just the user message tokens
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(10);
  });

  it('should skip invalid tool calls with no name AND no output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me help you with that.',
            tool_call_ids: ['valid_tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'invalid_tool_1',
              name: '',
              args: '{"query":"test"}',
              output: '',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'valid_tool_1',
              name: 'search',
              args: '{"query":"weather"}',
              output: 'The weather is sunny.',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage (invalid tool call is skipped)
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should only have 1 tool call (the valid one)
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );
    expect((result.messages[0] as AIMessage).tool_calls?.[0].id).toBe(
      'valid_tool_1'
    );

    // The ToolMessage should be for the valid tool call
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe(
      'valid_tool_1'
    );
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('The weather is sunny.');
  });

  it('should skip tool calls with no name AND null output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'invalid_tool_1',
              name: '',
              args: '{"query":"test"}',
              output: null,
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here is the information.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 1 message: AIMessage (invalid tool call is skipped)
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool calls or an empty array
    const toolCalls = (result.messages[0] as AIMessage).tool_calls;
    expect(toolCalls === undefined || toolCalls.length === 0).toBe(true);
    expect(result.messages[0].content).toStrictEqual([
      {
        type: ContentTypes.TEXT,
        [ContentTypes.TEXT]: 'Here is the information.',
      },
    ]);
  });

  it('should NOT skip tool calls with no name but valid output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: '',
              args: '{"query":"test"}',
              output: 'Valid output despite missing name',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have 1 tool call
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);

    // The ToolMessage should have the output
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].content).toBe(
      'Valid output despite missing name'
    );
  });

  it('should NOT skip tool calls with valid name but no output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"test"}',
              output: '',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have 1 tool call
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );

    // The ToolMessage should have empty content
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('');
  });

  describe('proportional token distribution', () => {
    it('should distribute tokens proportionally based on content length', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Short text',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'search',
                args: '{"query":"test"}',
                output:
                  'A much longer tool result that contains significantly more content than the original message text',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(100);
      expect(toolTokens).toBeGreaterThan(aiTokens);
    });

    it('should give the vast majority of tokens to a large tool result vs tiny AI message', () => {
      const bigOutput = 'x'.repeat(10000);
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'ok',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'snapshot',
                args: '{}',
                output: bigOutput,
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 5000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(5000);
      expect(toolTokens).toBeGreaterThan(4900);
      expect(aiTokens).toBeLessThan(100);
    });

    it('should fall back to even distribution when all content lengths are zero', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: '',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'noop',
                args: '{}',
                output: '',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 20 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(20);
      expect(aiTokens).toBeGreaterThanOrEqual(0);
      expect(toolTokens).toBeGreaterThanOrEqual(0);
    });

    it('should handle odd token counts without losing remainder', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'abc',
              tool_call_ids: ['tool_1', 'tool_2', 'tool_3'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'a',
                args: '{}',
                output: 'abc',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_2',
                name: 'b',
                args: '{}',
                output: 'abc',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_3',
                name: 'c',
                args: '{}',
                output: 'abc',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 7 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(4);
      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(7);
      for (let i = 0; i < result.messages.length; i++) {
        expect(result.indexTokenCountMap?.[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it('should never produce negative token counts', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'a',
              tool_call_ids: ['t1', 't2', 't3', 't4', 't5'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'x', args: '{}', output: 'b' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't2', name: 'x', args: '{}', output: 'c' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't3', name: 'x', args: '{}', output: 'd' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't4', name: 'x', args: '{}', output: 'e' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't5', name: 'x', args: '{}', output: 'f' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 3 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(3);
      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle single token budget distributed across many messages', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'hello',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'a',
                args: '{}',
                output: 'result one',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'b',
                args: '{}',
                output: 'result two',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 1 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(1);
      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle zero token budget', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'hello',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'world' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 0 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(0);
    });

    it('should distribute tokens proportionally with 5 tool calls of varying sizes', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'I will perform multiple operations.',
              tool_call_ids: ['t1', 't2', 't3', 't4', 't5'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'navigate',
                args: '{"url":"https://example.com"}',
                output: 'Navigated successfully.',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'snapshot',
                args: '{}',
                output: 'x'.repeat(5000),
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't3',
                name: 'click',
                args: '{"selector":"#btn"}',
                output: 'Clicked.',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't4',
                name: 'snapshot',
                args: '{}',
                output: 'y'.repeat(8000),
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't5',
                name: 'extract',
                args: '{"selector":"h1"}',
                output: 'Page Title',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 3000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(6);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(3000);

      const snapshotIdx1 = 2;
      const snapshotIdx2 = 4;
      const bigSnapshotTokens =
        (result.indexTokenCountMap?.[snapshotIdx1] ?? 0) +
        (result.indexTokenCountMap?.[snapshotIdx2] ?? 0);
      expect(bigSnapshotTokens).toBeGreaterThan(2500);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle HN-like payload: AI with 18 tool calls and large snapshot results', () => {
      const smallOutput = 'Successfully navigated to page.';
      const hugeSnapshot = 'uid=8_0 RootWebArea ' + 'x'.repeat(20000);

      const toolCalls: Array<{
        type: string;
        tool_call: { id: string; name: string; args: string; output: string };
      }> = [];
      const toolCallIds: string[] = [];

      for (let i = 0; i < 18; i++) {
        const id = `tool_${i}`;
        toolCallIds.push(id);
        const isSnapshot = i % 3 === 1;
        toolCalls.push({
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id,
            name: isSnapshot ? 'take_snapshot' : 'navigate_page',
            args: isSnapshot ? '{}' : `{"url":"https://example.com/${i}"}`,
            output: isSnapshot ? hugeSnapshot : smallOutput,
          },
        });
      }

      const payload = [
        {
          role: 'user',
          content: 'Look up top 5 posts on HN',
        },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: '',
              tool_call_ids: toolCallIds,
            },
            ...toolCalls,
          ],
        },
      ];

      const indexTokenCountMap = { 0: 20, 1: 10000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages.length).toBeGreaterThan(2);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(10020);

      expect(result.indexTokenCountMap?.[0]).toBe(20);

      let snapshotTokenTotal = 0;
      let navTokenTotal = 0;
      for (let i = 1; i < result.messages.length; i++) {
        const tokens = result.indexTokenCountMap?.[i] ?? 0;
        expect(tokens).toBeGreaterThanOrEqual(0);

        if (result.messages[i] instanceof ToolMessage) {
          const content = result.messages[i].content;
          if (typeof content === 'string' && content.length > 1000) {
            snapshotTokenTotal += tokens;
          } else {
            navTokenTotal += tokens;
          }
        }
      }

      expect(snapshotTokenTotal).toBeGreaterThan(navTokenTotal);
    });

    it('should complete proportional distribution within reasonable time for large payloads', () => {
      const toolCalls: Array<{
        type: string;
        tool_call: { id: string; name: string; args: string; output: string };
      }> = [];
      const toolCallIds: string[] = [];

      for (let i = 0; i < 50; i++) {
        const id = `tool_${i}`;
        toolCallIds.push(id);
        toolCalls.push({
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id,
            name: `tool_${i}`,
            args: JSON.stringify({ data: 'x'.repeat(100) }),
            output: 'y'.repeat(Math.floor(Math.random() * 10000)),
          },
        });
      }

      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Processing...',
              tool_call_ids: toolCallIds,
            },
            ...toolCalls,
          ],
        },
      ];

      const indexTokenCountMap = { 0: 50000 };

      const start = performance.now();
      const result = formatAgentMessages(payload, indexTokenCountMap);
      const elapsed = performance.now() - start;

      expect(elapsed).toBeLessThan(500);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(50000);
    });

    it('should always preserve total token count across multiple original messages', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Let me search.',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{"q":"test"}',
                output:
                  'Found 10 results with detailed descriptions: ' +
                  'z'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: 'Thanks' },
        { role: 'assistant', content: 'You are welcome!' },
      ];

      const indexTokenCountMap = { 0: 5, 1: 200, 2: 3, 3: 8 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(216);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(Number.isInteger(val)).toBe(true);
      }
    });

    it('should produce integer token counts (no floating point)', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'abc',
              tool_call_ids: ['t1', 't2', 't3'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'defgh' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't2', name: 'b', args: '{}', output: 'ij' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't3',
                name: 'c',
                args: '{}',
                output: 'klmnopqrst',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 97 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(Number.isInteger(val)).toBe(true);
      }

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(97);
    });

    it('should account for tool call args in content length calculation', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'x',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'tiny_tool',
                args: '{}',
                output: 'small',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'big_args_tool',
                args: JSON.stringify({ data: 'a'.repeat(5000) }),
                output: 'small',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 1000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(3);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(1000);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should not throw when indexTokenCountMap has undefined values for some indices', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'response',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: 'result',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap: Record<number, number | undefined> = {
        0: undefined,
        1: 50,
      };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(50);
      }).not.toThrow();
    });

    it('should not throw when indexTokenCountMap is sparse (missing indices)', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'World' },
        { role: 'user', content: 'Bye' },
      ];

      const indexTokenCountMap = { 0: 5, 2: 3 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        expect(result.indexTokenCountMap?.[0]).toBe(5);
        expect(result.indexTokenCountMap?.[2]).toBe(3);
      }).not.toThrow();
    });

    it('should not throw when indexTokenCountMap has extra indices beyond payload', () => {
      const payload = [{ role: 'user', content: 'Hello' }];

      const indexTokenCountMap = { 0: 5, 1: 10, 2: 15, 99: 999 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap?.[0]).toBe(5);
      }).not.toThrow();
    });

    it('should not throw with empty payload and non-empty indexTokenCountMap', () => {
      const payload: Array<{ role: string; content: string }> = [];
      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.messages).toHaveLength(0);
      }).not.toThrow();
    });

    it('should not throw when assistant message content is empty array', () => {
      const payload = [
        {
          role: 'assistant',
          content: [] as Array<{ type: string; text?: string }>,
        },
      ];

      const indexTokenCountMap = { 0: 50 };

      expect(() => {
        formatAgentMessages(payload, indexTokenCountMap);
      }).not.toThrow();
    });

    it('should not throw when tool call output is null or undefined', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'calling tools',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: null as unknown as string,
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'fetch',
                args: '{}',
                output: undefined as unknown as string,
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 30 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(30);
      }).not.toThrow();
    });

    it('should not throw when tool call args are deeply nested objects', () => {
      const deepArgs = { a: { b: { c: { d: { e: { f: 'deep' } } } } } };
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'deep call',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'deep_tool',
                args: JSON.stringify(deepArgs),
                output: 'done',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(100);
      }).not.toThrow();
    });

    it('should not throw when tool call args are not valid JSON strings', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'bad args',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'tool',
                args: '{not valid json!!!',
                output: 'output',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 40 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(40);
      }).not.toThrow();
    });

    it('should not throw when content array has mixed types including unexpected values', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'hello' },
            null as unknown as { type: string },
            undefined as unknown as { type: string },
            { type: 'unknown_type', something: 'weird' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 25 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap?.[0]).toBe(25);
      }).not.toThrow();
    });

    it('should not throw when tool call has empty name and empty args', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'test',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: '',
                args: '',
                output: 'some output',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 50 };

      expect(() => {
        formatAgentMessages(payload, indexTokenCountMap);
      }).not.toThrow();
    });

    it('should not throw when all content parts are filtered out (THINK + ERROR only)', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINK, [ContentTypes.THINK]: 'thinking...' },
            { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'error...' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(Object.keys(result.indexTokenCountMap || {}).length).toBe(0);
      }).not.toThrow();
    });

    it('should not throw with very large token count values', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'big tokens',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'b' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: Number.MAX_SAFE_INTEGER };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(Number.MAX_SAFE_INTEGER);
      }).not.toThrow();
    });

    it('should not throw when multiple payload messages expand and some have undefined token counts', () => {
      const payload = [
        { role: 'user', content: 'msg1' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'response with tool',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: 'found',
              },
            },
          ],
        },
        { role: 'user', content: 'msg2' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'another response',
              tool_call_ids: ['t2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'fetch',
                args: '{}',
                output: 'data',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap: Record<number, number | undefined> = {
        0: 5,
        1: undefined,
        2: 3,
        3: 80,
      };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        expect(result.indexTokenCountMap?.[0]).toBe(5);
      }).not.toThrow();
    });
  });

  describe('summary boundary token count adjustment', () => {
    /** Atomic media costs a fixed provider price the character heuristic cannot
     *  see, so scaling by the measurable siblings alone erases it. Both shapes
     *  collapsed a four-figure count to 1 before this guard. */
    it.each([
      [
        'text before the summary',
        { type: ContentTypes.TEXT, text: 'hello there' },
      ],
      [
        'a tool call before the summary',
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tc1',
            name: 'search',
            args: '{"q":"x"}',
            output: 'result text',
          },
        },
      ],
    ])(
      'skips the positional discount when retained media is unmeasurable, with %s',
      (_label, leading) => {
        const payload: TPayload = [
          {
            role: 'assistant',
            content: [
              leading as MessageContentComplex,
              {
                type: ContentTypes.SUMMARY,
                text: 'S'.repeat(400),
                tokenCount: 100,
              },
              {
                type: 'image_url',
                image_url: { url: 'data:image/png;base64,x' },
              },
            ],
          },
        ];

        const result = formatAgentMessages(payload, { 0: 1200 });

        expect(result.indexTokenCountMap?.[0]).toBe(1200);
        expect(result.boundaryTokenAdjustment).toBeUndefined();
      }
    );

    /** The media sits a level down, inside `tool_call.output`, where serializing
     *  gives it a nonzero length while the token counter charges its fixed media
     *  cost. Eligibility is decided by part type, so nesting depth is irrelevant. */
    it('skips the positional discount when retained tool output carries media', () => {
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'a'.repeat(4000) },
            {
              type: ContentTypes.SUMMARY,
              text: 'S'.repeat(400),
              tokenCount: 100,
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tc1',
                name: 'render',
                args: '{}',
                output: [
                  {
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,y' },
                  },
                ],
              },
            },
          ],
        },
      ];

      const result = formatAgentMessages(payload, { 0: 4000 });

      const emitted = Object.values(result.indexTokenCountMap ?? {}).reduce(
        (sum, value) => sum + value,
        0
      );
      expect(emitted).toBe(4000);
      expect(result.boundaryTokenAdjustment).toBeUndefined();
    });

    it('still proportions when every retained part is measurable', () => {
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'a'.repeat(400) },
            {
              type: ContentTypes.SUMMARY,
              text: 'S'.repeat(100),
              tokenCount: 20,
            },
            { type: ContentTypes.TEXT, text: 'b'.repeat(100) },
          ],
        },
      ];

      const result = formatAgentMessages(payload, { 0: 600 });

      expect(result.boundaryTokenAdjustment?.original).toBe(600);
      expect(result.indexTokenCountMap?.[0]).toBeLessThan(600);
      expect(result.indexTokenCountMap?.[0]).toBeGreaterThan(0);
    });

    it('should proportion token count when thinking block is sliced off by boundary', () => {
      const thinkingText = 'x'.repeat(1000);
      const payload: TPayload = [
        { role: 'user', content: 'Old question' },
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINKING, thinking: thinkingText },
            {
              type: ContentTypes.SUMMARY,
              text: 'Summary of conversation',
              tokenCount: 15,
            },
            { type: ContentTypes.TEXT, text: 'Brief response after summary' },
          ],
        },
        { role: 'user', content: 'Follow-up question' },
      ];

      const indexTokenCountMap = { 0: 5, 1: 1590, 2: 8 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();
      expect(result.summary!.text).toBe('Summary of conversation');

      const boundaryMsgTokens = result.indexTokenCountMap?.[0];
      expect(boundaryMsgTokens).toBeDefined();
      expect(boundaryMsgTokens!).toBeLessThan(200);
      expect(boundaryMsgTokens!).toBeGreaterThan(0);

      expect(result.indexTokenCountMap?.[1]).toBe(8);
    });

    /** Reframed: a tool call anywhere in the entry now cancels the ratio, since
     *  telling a text-bearing tool payload from a media-bearing one requires
     *  recursing into arbitrary nested output. The entry keeps its count. */
    it('should not proportion when a tool_use part is present', () => {
      const thinkingText = 'a'.repeat(800);
      const toolInput = JSON.stringify({ data: 'b'.repeat(400) });
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINKING, thinking: thinkingText },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tc1',
                name: 'search',
                args: toolInput,
                output: 'result',
              },
            },
            {
              type: ContentTypes.SUMMARY,
              text: 'Conversation summary after tool use',
              tokenCount: 20,
            },
            { type: ContentTypes.TEXT, text: 'Short tail' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 2000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();

      const totalOutputTokens = Object.values(
        result.indexTokenCountMap || {}
      ).reduce((sum, v) => sum + v, 0);

      expect(totalOutputTokens).toBe(2000);
      expect(result.boundaryTokenAdjustment).toBeUndefined();
    });

    it('should roughly halve token count when content is evenly split around boundary', () => {
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'a'.repeat(100) },
            {
              type: ContentTypes.SUMMARY,
              text: 'Mid-conversation summary',
              tokenCount: 10,
            },
            { type: ContentTypes.TEXT, text: 'b'.repeat(100) },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 500 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();

      const adjustedTokens = result.indexTokenCountMap?.[0] ?? 0;
      expect(adjustedTokens).toBeGreaterThan(150);
      expect(adjustedTokens).toBeLessThan(350);
    });

    it('should still adjust when summary is the first content part (its own text is sliced off)', () => {
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'Summary at start',
              tokenCount: 10,
            },
            { type: ContentTypes.TEXT, text: 'Everything after the summary' },
          ],
        },
        { role: 'user', content: 'Next question' },
      ];

      const indexTokenCountMap = { 0: 300, 1: 10 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();

      const adjustedTokens = result.indexTokenCountMap?.[0] ?? 0;
      expect(adjustedTokens).toBeLessThan(300);
      expect(adjustedTokens).toBeGreaterThan(100);
      expect(result.indexTokenCountMap?.[1]).toBe(10);
    });

    /** Previously the removed `tool_use` input was counted into the denominator.
     *  A base64 payload there serializes to a huge length while the counter
     *  charges a fixed estimate, so the ratio dragged retained text below its
     *  real cost. The discount is cancelled instead. */
    it('should not use tool_use input size in the char-length ratio', () => {
      const hugeInput = JSON.stringify({ payload: 'z'.repeat(5000) });
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use' as ContentTypes,
              input: hugeInput,
            } as unknown as MessageContentComplex,
            {
              type: ContentTypes.SUMMARY,
              text: 'After heavy tool use',
              tokenCount: 12,
            },
            { type: ContentTypes.TEXT, text: 'Tiny tail' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 3000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();

      const adjustedTokens = result.indexTokenCountMap?.[0] ?? 0;
      expect(adjustedTokens).toBe(3000);
      expect(result.boundaryTokenAdjustment).toBeUndefined();
    });

    it('should handle multiple content parts after the boundary', () => {
      const thinkingText = 'x'.repeat(2000);
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINKING, thinking: thinkingText },
            {
              type: ContentTypes.SUMMARY,
              text: 'Conversation checkpoint',
              tokenCount: 14,
            },
            { type: ContentTypes.TEXT, text: 'Part A of the tail' },
            {
              type: ContentTypes.TEXT,
              text: 'Part B of the tail with more text',
            },
          ],
        },
        { role: 'user', content: 'Next message' },
      ];

      const indexTokenCountMap = { 0: 4000, 1: 6 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();

      const adjustedTokens = result.indexTokenCountMap?.[0] ?? 0;
      expect(adjustedTokens).toBeLessThan(200);
      expect(adjustedTokens).toBeGreaterThan(0);

      expect(result.indexTokenCountMap?.[1]).toBe(6);
    });

    it('should produce integer token counts after proportional adjustment', () => {
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINKING, thinking: 'x'.repeat(333) },
            {
              type: ContentTypes.SUMMARY,
              text: 'Summary',
              tokenCount: 5,
            },
            { type: ContentTypes.TEXT, text: 'y'.repeat(77) },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 997 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const adjustedTokens = result.indexTokenCountMap?.[0];
      expect(adjustedTokens).toBeDefined();
      expect(Number.isInteger(adjustedTokens)).toBe(true);
    });
  });

  describe('summary coverage boundary', () => {
    const buildSummaryPart = (coverage?: {
      retainedFromMessageId: string;
    }) => ({
      type: ContentTypes.SUMMARY,
      content: [
        { type: ContentTypes.TEXT, text: 'Summary of the earliest turns' },
      ],
      tokenCount: 12,
      ...(coverage != null ? { coverage } : {}),
    });

    /** Mirrors a compaction with `retainRecent.turns: 1`: m1/m2 were refined
     *  into the summary, m3/m4 are the retained tail, and the block itself is
     *  persisted on the assistant message that came after all of them. */
    const compactedPayload = (coverage?: {
      retainedFromMessageId: string;
    }): TPayload => [
      { messageId: 'm1', role: 'user', content: 'Covered question' },
      { messageId: 'm2', role: 'assistant', content: 'Covered answer' },
      { messageId: 'm3', role: 'user', content: 'Retained question' },
      { messageId: 'm4', role: 'assistant', content: 'Retained answer' },
      {
        messageId: 'm5',
        role: 'assistant',
        content: [
          buildSummaryPart(coverage),
          { type: ContentTypes.TEXT, text: 'Post-compaction reply' },
        ],
      },
    ];

    const textOf = (message: BaseMessage): string => {
      const { content } = message;
      if (typeof content === 'string') {
        return content;
      }
      return (content as MessageContentComplex[])
        .map((part) => ('text' in part ? (part as { text: string }).text : ''))
        .join('');
    };

    it('preserves the retained tail that the summary never covered', () => {
      const result = formatAgentMessages(
        compactedPayload({ retainedFromMessageId: 'm3' })
      );

      expect(result.messages.map(textOf)).toEqual([
        'Retained question',
        'Retained answer',
        'Post-compaction reply',
      ]);
      expect(result.summary!.text).toBe('Summary of the earliest turns');
      expect(result.summary!.tokenCount).toBe(12);
    });

    it('retains the anchor message itself, dropping only what precedes it', () => {
      const result = formatAgentMessages(
        compactedPayload({ retainedFromMessageId: 'm4' })
      );

      expect(result.messages.map(textOf)).toEqual([
        'Retained answer',
        'Post-compaction reply',
      ]);
    });

    /** Coverage mode leaves the block's entry at its full count on purpose. The
     *  summary's cost in the reader's token units is not obtainable here — no
     *  tokenizer reaches this function, and a figure recorded at write time is
     *  in the writing run's units. Over-counting prunes early; under-counting
     *  would risk an over-context request. */
    it('does not discount the entry carrying the summary block', () => {
      const payload: TPayload = [
        { messageId: 'm1', role: 'user', content: 'Covered question' },
        { messageId: 'm2', role: 'user', content: 'Retained question' },
        {
          messageId: 'm3',
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              content: [{ type: ContentTypes.TEXT, text: 'S'.repeat(500) }],
              tokenCount: 120,
              coverage: { retainedFromMessageId: 'm2' },
            },
            { type: ContentTypes.TEXT, text: 'Reply' },
          ],
        },
      ];

      const result = formatAgentMessages(payload, { 0: 5, 1: 6, 2: 1000 });

      expect(result.indexTokenCountMap?.[1]).toBe(1000);
      expect(result.boundaryTokenAdjustment).toBeUndefined();
    });

    it('keeps token counts for the retained tail and drops covered entries', () => {
      const result = formatAgentMessages(
        compactedPayload({ retainedFromMessageId: 'm3' }),
        { 0: 5, 1: 6, 2: 7, 3: 8, 4: 40 }
      );

      expect(result.indexTokenCountMap?.[0]).toBe(7);
      expect(result.indexTokenCountMap?.[1]).toBe(8);
      expect(Object.keys(result.indexTokenCountMap ?? {})).toHaveLength(3);
    });

    it('leaves entries without summary parts untouched', () => {
      const result = formatAgentMessages(
        compactedPayload({ retainedFromMessageId: 'm3' }),
        { 0: 5, 1: 6, 2: 7, 3: 8, 4: 40 }
      );

      expect(result.indexTokenCountMap?.[0]).toBe(7);
      expect(result.indexTokenCountMap?.[1]).toBe(8);
    });

    it('falls back to positional trimming for legacy blocks without coverage', () => {
      const result = formatAgentMessages(compactedPayload());

      expect(result.messages.map(textOf)).toEqual(['Post-compaction reply']);
      expect(result.summary!.text).toBe('Summary of the earliest turns');
    });

    it('falls back to positional trimming when coverage cannot be resolved', () => {
      const result = formatAgentMessages(
        compactedPayload({ retainedFromMessageId: 'pruned-from-payload' })
      );

      expect(result.messages.map(textOf)).toEqual(['Post-compaction reply']);
    });

    it('ignores an anchor pointing past its own block', () => {
      const result = formatAgentMessages([
        ...compactedPayload({ retainedFromMessageId: 'm6' }),
        { messageId: 'm6', role: 'user', content: 'Later question' },
      ]);

      expect(result.messages.map(textOf)).toEqual([
        'Post-compaction reply',
        'Later question',
      ]);
    });

    it('applies last-summary-wins across mixed coverage and legacy blocks', () => {
      const payload: TPayload = [
        { messageId: 'm1', role: 'user', content: 'Covered question' },
        {
          messageId: 'm2',
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'Older summary',
              tokenCount: 3,
            },
            { type: ContentTypes.TEXT, text: 'Older tail' },
          ],
        },
        { messageId: 'm3', role: 'user', content: 'Retained question' },
        {
          messageId: 'm4',
          role: 'assistant',
          content: [
            buildSummaryPart({ retainedFromMessageId: 'm3' }),
            { type: ContentTypes.TEXT, text: 'Newest reply' },
          ],
        },
      ];

      const result = formatAgentMessages(payload);

      expect(result.messages.map(textOf)).toEqual([
        'Retained question',
        'Newest reply',
      ]);
      expect(result.summary!.text).toBe('Summary of the earliest turns');
    });
  });

  describe('cross-run summary token accounting', () => {
    it('should conserve tokens: summary boundary excludes pre-boundary messages from the map', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Old question' },
        { role: 'assistant', content: 'Old answer' },
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'Text before summary' },
            {
              type: ContentTypes.SUMMARY,
              text: 'This is a conversation summary capturing prior context.',
              tokenCount: 25,
            },
            { type: ContentTypes.TEXT, text: 'Text after summary' },
          ],
        },
        { role: 'user', content: 'New question after summary' },
        { role: 'assistant', content: 'New answer after summary' },
      ];

      const indexTokenCountMap = {
        0: 8,
        1: 12,
        2: 60,
        3: 10,
        4: 15,
      };

      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();
      expect(result.summary!.text).toBe(
        'This is a conversation summary capturing prior context.'
      );
      expect(result.summary!.tokenCount).toBe(25);

      const outputKeys = Object.keys(result.indexTokenCountMap || {}).map(
        Number
      );
      expect(outputKeys).toHaveLength(3);

      const boundaryMsgTokens = result.indexTokenCountMap?.[0] ?? 0;
      expect(boundaryMsgTokens).toBeLessThan(60);
      expect(boundaryMsgTokens).toBeGreaterThan(0);
      expect(result.indexTokenCountMap?.[1]).toBe(10);
      expect(result.indexTokenCountMap?.[2]).toBe(15);
    });

    it('should preserve summary token at index 0 when tool calls expand post-boundary messages', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Summarized away' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'Summary of the conversation so far.',
              tokenCount: 20,
            },
          ],
        },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Let me compute that.',
              tool_call_ids: ['calc_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'calc_1',
                name: 'calculator',
                args: '{"expr":"2+2"}',
                output: '4',
              },
            },
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'The answer is 4.',
            },
          ],
        },
        { role: 'user', content: 'Thanks!' },
      ];

      const indexTokenCountMap = {
        0: 5,
        1: 30,
        2: 80,
        3: 6,
      };

      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();
      expect(result.summary!.text).toBe('Summary of the conversation so far.');
      expect(result.summary!.tokenCount).toBe(20);

      const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, count) => sum + count,
        0
      );
      expect(totalTokens).toBe(80 + 6);
    });

    it('should produce correct maps across a simulated multi-run lifecycle', () => {
      const run1Payload: TPayload = [
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: 'The answer is 4.' },
      ];
      const run1Map = { 0: 10, 1: 12 };

      const run1Result = formatAgentMessages(run1Payload, run1Map);
      expect(run1Result.messages).toHaveLength(2);
      expect(run1Result.indexTokenCountMap?.[0]).toBe(10);
      expect(run1Result.indexTokenCountMap?.[1]).toBe(12);

      const run2Payload: TPayload = [
        ...run1Payload,
        { role: 'user', content: 'Now multiply 4 by 10.' },
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'Sure, the answer is 40.' },
            {
              type: ContentTypes.SUMMARY,
              text: 'User asked basic arithmetic: 2+2=4, then 4*10=40.',
              tokenCount: 18,
            },
          ],
        },
      ];
      const run2Map = { 0: 10, 1: 12, 2: 14, 3: 50 };

      const run2Result = formatAgentMessages(run2Payload, run2Map);
      expect(run2Result.summary).toBeDefined();
      expect(run2Result.summary!.text).toBe(
        'User asked basic arithmetic: 2+2=4, then 4*10=40.'
      );
      expect(run2Result.summary!.tokenCount).toBe(18);

      const run2TotalPostBoundary = Object.values(
        run2Result.indexTokenCountMap || {}
      ).reduce((sum, v) => sum + v, 0);
      expect(run2TotalPostBoundary).toBe(0);

      const run3Payload: TPayload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'User asked basic arithmetic: 2+2=4, then 4*10=40.',
              tokenCount: 18,
            },
          ],
        },
        { role: 'user', content: 'What is the square root of 40?' },
        {
          role: 'assistant',
          content: 'The square root of 40 is approximately 6.32.',
        },
      ];
      const run3Map = { 0: 18, 1: 15, 2: 20 };

      const run3Result = formatAgentMessages(run3Payload, run3Map);
      expect(run3Result.summary).toBeDefined();
      expect(run3Result.summary!.text).toBe(
        'User asked basic arithmetic: 2+2=4, then 4*10=40.'
      );
      expect(run3Result.summary!.tokenCount).toBe(18);

      const run3Total = Object.values(
        run3Result.indexTokenCountMap || {}
      ).reduce((sum, count) => sum + count, 0);
      expect(run3Total).toBe(15 + 20);
    });
  });
});

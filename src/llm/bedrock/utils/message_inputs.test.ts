import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { convertToConverseMessages } from './message_inputs';
import { toLangChainContent } from '@/messages/langchain';

/**
 * Native-Bedrock reasoning serialization. A `reasoning_content` block whose
 * `reasoningText.text` is null/empty (e.g. a signature-only block that never
 * merged with its text) is invalid for Bedrock Converse — it rejects with
 * `...reasoningContent.reasoningText.text ... Member must not be null`. Such a
 * block must be dropped on replay rather than sent; a block carrying real text
 * is still converted.
 */
type ConverseResult = ReturnType<typeof convertToConverseMessages>;

/** Minimal view of a converted Bedrock Converse content block the assertions read. */
interface ConverseBlock {
  text?: string;
  reasoningContent?: {
    reasoningText?: { text?: string; signature?: string };
    redactedContent?: Uint8Array;
  };
  toolUse?: {
    toolUseId?: string;
    name?: string;
    input?: Record<string, string>;
  };
}

const assistantContent = (result: ConverseResult): ConverseBlock[] => {
  const msg = result.converseMessages.find((m) => m.role === 'assistant');
  return (msg?.content ?? []) as ConverseBlock[];
};

describe('convertToConverseMessages — Anthropic tool replay', () => {
  it('deduplicates raw tool_use blocks and unwraps tool_result content', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Search'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Searching.' },
          {
            type: 'tool_use',
            id: 'call_search',
            name: 'search',
            input: '',
          },
        ]),
        tool_calls: [
          {
            id: 'call_search',
            name: 'search',
            args: { query: 'test' },
          },
        ],
      }),
      new ToolMessage({
        content: toLangChainContent([
          {
            type: 'tool_result',
            tool_use_id: 'call_search',
            is_error: true,
            content: 'result body',
          },
        ]),
        tool_call_id: 'call_search',
      }),
    ];

    const converted = convertToConverseMessages(messages);
    const serialized = JSON.stringify(converted);

    expect(serialized.match(/"toolUseId":"call_search"/g)).toHaveLength(2);
    expect(serialized.match(/"toolUse":/g)).toHaveLength(1);
    expect(serialized.match(/"toolResult":/g)).toHaveLength(1);
    expect(serialized).toContain('"text":"result body"');
    expect(serialized).toContain('"status":"error"');
    expect(serialized).not.toContain('"type":"tool_result"');
  });
});

/**
 * Bedrock Converse requires `toolUse.input` to be a JSON object. History can
 * carry non-object values: pre-3.x context-pressure truncation persisted
 * `null` onto both a block's inline input and its `tool_calls` args (see the
 * Anthropic-replay fix in PR #369), and Anthropic-shaped inline blocks keep
 * the raw streamed JSON string. These must coerce — never ship as-is, never
 * throw the whole request away.
 */
describe('convertToConverseMessages — non-object toolUse input coercion', () => {
  const toolUseBlocks = (result: ConverseResult): ConverseBlock[] =>
    assistantContent(result).filter((b) => b.toolUse != null);

  it('coerces null args to {} when materializing from tool_calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('What is 9 * 9?'),
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'tool_use',
            id: 'call_null',
            name: 'calculator',
            input: null,
          },
        ]),
        tool_calls: [
          {
            id: 'call_null',
            name: 'calculator',
            args: null as never,
          },
        ],
      }),
    ];

    const [block] = toolUseBlocks(convertToConverseMessages(messages));
    expect(block.toolUse?.input).toEqual({});
  });

  it('parses a complete raw-string input on an unmirrored inline block instead of throwing', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Search'),
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'tool_use',
            id: 'call_str',
            name: 'search',
            input: '{"query": "test"}',
          },
        ]),
        tool_calls: [],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const [block] = toolUseBlocks(convertToConverseMessages(messages));
    expect(block.toolUse?.input).toEqual({ query: 'test' });
  });

  it('degrades non-object inline inputs (partial string, number-string, null) to {}', () => {
    for (const input of ['{"query": "tru', '123', null]) {
      const messages: BaseMessage[] = [
        new HumanMessage('Search'),
        new AIMessage({
          content: toLangChainContent([
            { type: 'tool_use', id: 'call_bad', name: 'search', input },
          ]),
          tool_calls: [],
        }),
      ];

      const [block] = toolUseBlocks(convertToConverseMessages(messages));
      expect(block.toolUse?.input).toEqual({});
    }
  });

  it('still rejects a tool_use block missing its id or name', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Search'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'tool_use', name: 'search', input: { query: 'x' } },
        ]),
        tool_calls: [],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).toThrow(
      'Invalid Anthropic tool_use content block'
    );
  });

  it('coerces null args on the v1 tool_call and tool_calls fallback paths', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Run both'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'tool_call', id: 'v1_block', name: 'search', args: null },
        ]),
        tool_calls: [
          { id: 'v1_fallback', name: 'lookup', args: null as never },
        ],
        response_metadata: { output_version: 'v1' },
      }),
    ];

    const blocks = toolUseBlocks(convertToConverseMessages(messages));
    expect(blocks).toHaveLength(2);
    for (const block of blocks) {
      expect(block.toolUse?.input).toEqual({});
    }
  });
});

describe('convertToConverseMessages — native Bedrock reasoning serialization', () => {
  it('drops a signature-only reasoning block, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('what data do you have?'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { signature: 'sig-abc' },
          },
          { type: 'text', text: 'Let me check your databases.' },
        ],
        tool_calls: [
          {
            id: 'tooluse_list',
            name: 'list_databases',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));

    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(JSON.stringify(content)).not.toContain('sig-abc');
    expect(content.some((b) => b.text === 'Let me check your databases.')).toBe(
      true
    );
    const toolUse = content.find((b) => b.toolUse != null);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'tooluse_list',
      name: 'list_databases',
    });
  });

  it('drops a reasoning block whose text is empty', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: '', signature: 'sig' },
          },
          { type: 'text', text: 'answer' },
        ],
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('emits a placeholder (not empty content) when the only block is a signature-only reasoning block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning_content', reasoningText: { signature: 'sig' } },
        ],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.length).toBeGreaterThan(0);
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.every((b) => typeof b.text === 'string')).toBe(true);
  });

  it('still converts a reasoning block that carries text (not dropped)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: {
              text: 'native bedrock reasoning',
              signature: 'sig',
            },
          },
          { type: 'text', text: 'answer' },
        ],
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoning = content.find((b) => b.reasoningContent != null);
    expect(reasoning).toBeDefined();
    expect(reasoning?.reasoningContent?.reasoningText?.text).toBe(
      'native bedrock reasoning'
    );
  });
});

/**
 * Same failure class, v1 converter path. Assistant messages carrying
 * `response_metadata.output_version === 'v1'` are converted by
 * `convertFromV1ToChatBedrockConverseMessage`, which serialized `reasoning` /
 * `reasoning_content` blocks without the null/empty-text guard applied to the
 * non-v1 path — a `reasoning` block whose `reasoning` is null/empty (e.g. a
 * model responding with `thinking.display: "omitted"`, the Opus 4.7+ /
 * Sonnet 5 default) reached Bedrock as `reasoningText: { text: null }` and the
 * whole request was rejected with `Member must not be null`.
 */
describe('convertToConverseMessages — v1 reasoning serialization', () => {
  const v1Metadata = {
    output_version: 'v1',
    model_provider: 'anthropic',
  } as const;

  it('drops a v1 reasoning block whose reasoning text is missing, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('what data do you have?'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: undefined } as never,
          { type: 'text', text: 'Let me check your databases.' },
        ],
        tool_calls: [
          {
            id: 'tooluse_list',
            name: 'list_databases',
            args: {},
            type: 'tool_call',
          },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));

    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'Let me check your databases.')).toBe(
      true
    );
    const toolUse = content.find((b) => b.toolUse != null);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'tooluse_list',
      name: 'list_databases',
    });
  });

  it('drops a v1 reasoning block whose reasoning text is empty', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: '' },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('drops a v1 signature-only reasoning_content block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { signature: 'sig-abc' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(JSON.stringify(content)).not.toContain('sig-abc');
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('emits a placeholder (not empty content) when dropping empties a v1 turn', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [{ type: 'reasoning', reasoning: undefined } as never],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.length).toBeGreaterThan(0);
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.every((b) => typeof b.text === 'string')).toBe(true);
  });

  it.each(['', ' \n '])(
    'emits a placeholder for invalid v1 text %j',
    (text) => {
      const messages: BaseMessage[] = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [{ type: 'text', text }],
          response_metadata: v1Metadata,
        }),
      ];

      const content = assistantContent(convertToConverseMessages(messages));
      expect(content).toEqual([{ text: '_' }]);
    }
  );

  it('merges whitespace-only v1 text into the preceding text block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'text', text: 'answer' },
          { type: 'text', text: ' \n ' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content).toEqual([{ text: 'answer \n ' }]);
  });

  it('merges split v1 reasoning_content text and signature blocks before serialization', () => {
    const splitContent = [
      {
        type: 'reasoning_content',
        reasoningText: { text: 'first ' },
      },
      {
        type: 'reasoning_content',
        reasoningText: { text: 'second' },
      },
      {
        type: 'reasoning_content',
        reasoningText: { signature: 'sig-abc' },
      },
      { type: 'text', text: 'answer' },
    ];
    const originalContent = structuredClone(splitContent);
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: splitContent,
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoning = content.filter((b) => b.reasoningContent != null);
    expect(reasoning).toHaveLength(1);
    expect(reasoning[0].reasoningContent?.reasoningText).toEqual({
      text: 'first second',
      signature: 'sig-abc',
    });
    expect(splitContent).toEqual(originalContent);
  });

  it('keeps independently signed v1 reasoning blocks separate', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: 'first', signature: 'sig-first' },
          },
          {
            type: 'reasoning_content',
            reasoningText: { text: 'second', signature: 'sig-second' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(
      content
        .filter((block) => block.reasoningContent != null)
        .map((block) => block.reasoningContent?.reasoningText)
    ).toEqual([
      { text: 'first', signature: 'sig-first' },
      { text: 'second', signature: 'sig-second' },
    ]);
  });

  it('keeps adjacent redacted v1 reasoning payloads separate', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning_content', redactedContent: 'YQ==' },
          { type: 'reasoning_content', redactedContent: 'Yg==' },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const redacted = content
      .map((block) => block.reasoningContent?.redactedContent)
      .filter((value): value is Uint8Array => value != null)
      .map((value) => Buffer.from(value).toString('utf8'));
    expect(redacted).toEqual(['a', 'b']);
  });

  it('throws instead of returning empty assistant content for an unhandled v1 block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'image',
            source_type: 'base64',
            data: 'aGVsbG8=',
            mime_type: 'image/png',
          } as never,
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).toThrow(
      'Unsupported v1 content block type: image'
    );
  });

  it('still converts v1 reasoning and reasoning_content blocks that carry text', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'v1 standard reasoning' },
          {
            type: 'reasoning_content',
            reasoningText: { text: 'native reasoning', signature: 'sig' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoningTexts = content
      .filter((b) => b.reasoningContent != null)
      .map((b) => b.reasoningContent?.reasoningText?.text);
    expect(reasoningTexts).toEqual([
      'v1 standard reasoning',
      'native reasoning',
    ]);
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });
});

/**
 * Converse rejects consecutive same-role messages categorically, and both
 * `ToolMessage` and `HumanMessage` convert to `role: 'user'` — so a hook
 * injection landing a text turn directly after tool results (`PostToolBatch`
 * / `PreemptBoundary` drains) must merge into one user message here, with
 * block order preserved so the toolUse/toolResult pairing stays intact.
 */
describe('convertToConverseMessages — user-role run merging', () => {
  it('merges a tool result followed by an injected text turn into one user message', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('run the search'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call_1', name: 'search', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'search output',
        tool_call_id: 'call_1',
        name: 'search',
      }),
      new HumanMessage({
        content: 'Actually, focus on the second result.',
        additional_kwargs: { source: 'steer' },
      }),
    ];

    const { converseMessages } = convertToConverseMessages(messages);

    expect(converseMessages.map((m) => m.role)).toEqual([
      'user',
      'assistant',
      'user',
    ]);
    const merged = converseMessages[2];
    const blockKinds = (merged.content ?? []).map((block) =>
      'toolResult' in block ? 'toolResult' : 'text'
    );
    expect(blockKinds).toEqual(['toolResult', 'text']);
    expect((merged.content ?? []).find((block) => 'text' in block)?.text).toBe(
      'Actually, focus on the second result.'
    );
  });

  it('still merges adjacent tool-result-only turns', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call_1', name: 'a', args: {}, type: 'tool_call' },
          { id: 'call_2', name: 'b', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({ content: 'one', tool_call_id: 'call_1', name: 'a' }),
      new ToolMessage({ content: 'two', tool_call_id: 'call_2', name: 'b' }),
    ];

    const { converseMessages } = convertToConverseMessages(messages);

    expect(converseMessages.map((m) => m.role)).toEqual(['assistant', 'user']);
    const toolResultIds = (converseMessages[1].content ?? [])
      .map((block) =>
        'toolResult' in block ? block.toolResult?.toolUseId : undefined
      )
      .filter(Boolean);
    expect(toolResultIds).toEqual(['call_1', 'call_2']);
  });

  it('does not merge across an assistant turn', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('first'),
      new AIMessage('answer'),
      new HumanMessage('second'),
    ];
    const { converseMessages } = convertToConverseMessages(messages);
    expect(converseMessages.map((m) => m.role)).toEqual([
      'user',
      'assistant',
      'user',
    ]);
  });
});

describe('convertToConverseMessages — blank user content', () => {
  it('replaces an empty-string user message with the placeholder instead of a blank text block', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage(''),
    ]);
    expect(converseMessages[0]).toEqual({
      role: 'user',
      content: [{ text: '_' }],
    });
  });

  it('replaces whitespace-only user text with the placeholder', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('   '),
    ]);
    expect(converseMessages[0]).toEqual({
      role: 'user',
      content: [{ text: '_' }],
    });
  });

  it('drops a blank text block from an image-only user message and keeps the image', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage({
        content: [
          { type: 'text', text: '' },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
          },
        ],
      }),
    ]);
    const content = converseMessages[0].content ?? [];
    expect(content).toHaveLength(1);
    expect(content[0]).toHaveProperty('image');
    expect(content.some((block) => 'text' in block)).toBe(false);
  });

  it('folds whitespace-only text into the preceding block instead of dropping it', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage({
        content: [
          { type: 'text', text: 'hello' },
          { type: 'text', text: ' ' },
          { type: 'text', text: 'world' },
        ],
      }),
    ]);
    expect(converseMessages[0]).toEqual({
      role: 'user',
      content: [{ text: 'hello ' }, { text: 'world' }],
    });
  });

  it('does not inject the placeholder into a merged user group', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('hello'),
      new HumanMessage(''),
    ]);
    expect(converseMessages).toEqual([
      { role: 'user', content: [{ text: 'hello' }] },
    ]);
  });

  it('applies the placeholder when an entire user group is empty', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage(''),
      new HumanMessage('   '),
    ]);
    expect(converseMessages).toEqual([
      { role: 'user', content: [{ text: '_' }] },
    ]);
  });

  it('adds a text block to a promptless document upload, which Converse requires', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage({
        content: [
          { type: 'text', text: '' },
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: 'JVBERi0xLjQK',
          },
        ],
      }),
    ]);
    const content = converseMessages[0].content ?? [];
    expect(content).toHaveLength(2);
    expect(content[0]).toHaveProperty('document');
    expect(content[1]).toEqual({ text: '_' });
  });

  it('keeps whitespace that precedes the first text fragment', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage({
        content: [
          { type: 'text', text: ' ' },
          { type: 'text', text: 'hello' },
        ],
      }),
    ]);
    expect(converseMessages[0]).toEqual({
      role: 'user',
      content: [{ text: ' hello' }],
    });
  });

  it('keeps whitespace between media and the text that follows it', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage({
        content: [
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
          },
          { type: 'text', text: ' ' },
          { type: 'text', text: 'hello' },
        ],
      }),
    ]);
    const content = converseMessages[0].content ?? [];
    expect(content).toHaveLength(2);
    expect(content[0]).toHaveProperty('image');
    expect(content[1]).toEqual({ text: ' hello' });
  });

  it('adds the document text block to a merged group only when no member had text', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('read this'),
      new HumanMessage({
        content: [
          {
            type: 'file',
            source_type: 'base64',
            mime_type: 'application/pdf',
            data: 'JVBERi0xLjQK',
          },
        ],
      }),
    ]);
    const content = converseMessages[0].content ?? [];
    expect(content).toHaveLength(2);
    expect(content[0]).toEqual({ text: 'read this' });
    expect(content[1]).toHaveProperty('document');
  });

  it('keeps non-blank user text unchanged', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('hello'),
    ]);
    expect(converseMessages[0]).toEqual({
      role: 'user',
      content: [{ text: 'hello' }],
    });
  });
});

describe('convertToConverseMessages — standard base64 documents', () => {
  const documentOf = (
    result: ConverseResult
  ): { name?: string; format?: string; source?: { bytes?: Uint8Array } } => {
    const block = (result.converseMessages[0].content ?? []).find(
      (item) => 'document' in item
    );
    return (block as { document: ReturnType<typeof documentOf> }).document;
  };

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([
    [
      'text/csv',
      'csv',
      Buffer.from('id,name\n1,Zoë ☕\n'.repeat(300)).subarray(0, 4099),
    ],
    [
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'docx',
      Buffer.from(Array.from({ length: 4096 }, (_, i) => (i * 151 + 17) & 255)),
    ],
  ])(
    'sends a %s file as a %s document with its exact bytes',
    (mime, format, file) => {
      const atob = jest.spyOn(globalThis, 'atob');
      const result = convertToConverseMessages([
        new HumanMessage({
          content: [
            { type: 'text', text: 'Summarize' },
            {
              type: 'file',
              source_type: 'base64',
              mime_type: mime,
              data: file.toString('base64'),
              metadata: { name: 'ledger' },
            },
          ],
        }),
      ]);

      const document = documentOf(result);
      expect(document.name).toBe('ledger');
      expect(document.format).toBe(format);
      expect(document.source?.bytes).toBeInstanceOf(Uint8Array);
      expect(Buffer.from(document.source?.bytes ?? []).equals(file)).toBe(true);
      expect(atob).not.toHaveBeenCalled();
    }
  );
});

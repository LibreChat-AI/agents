import { ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import {
  compactToolContent,
  getResponsesToolOutputParts,
  isAtomicToolContentBlock,
} from '@/utils/toolContent';
import { _convertMessagesToOpenAIResponsesParams } from './index';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { formatAgentMessages } from '@/messages/format';
import { ContentTypes } from '@/common';

const image = {
  type: 'image_url',
  image_url: { url: 'https://example.com/chart.png', detail: 'high' },
};
const file = {
  type: 'file',
  file: {
    filename: 'report.pdf',
    file_data: 'data:application/pdf;base64,YQ==',
  },
};
const mixed = [{ type: 'text', text: 'Report' }, image, file];
const expected = [
  { type: 'input_text', text: 'Report' },
  {
    type: 'input_image',
    image_url: 'https://example.com/chart.png',
    detail: 'high',
  },
  {
    type: 'input_file',
    filename: 'report.pdf',
    file_data: 'data:application/pdf;base64,YQ==',
  },
];

describe('Responses function output content arrays', () => {
  it('preserves an empty output array from formatter through provider request', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Run empty tool' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'empty',
              name: 'empty_tool',
              args: '{}',
              output: [],
            },
          },
        ],
      },
    ];
    const { messages } = formatAgentMessages(payload);
    const tools = messages.filter(
      (message) => message.getType() === 'tool'
    ) as ToolMessage[];
    expect(tools).toHaveLength(1);
    expect(tools[0].content).toEqual([]);
    expect(compactToolContent([], 0).content).toEqual([]);
    expect(getResponsesToolOutputParts([])).toEqual([]);
    const outputs = _convertMessagesToOpenAIResponsesParams(messages).filter(
      (item) => item.type === 'function_call_output'
    );
    expect(outputs).toEqual([
      expect.objectContaining({ call_id: 'empty', output: [] }),
    ]);
  });

  it.each([undefined, null, ''])(
    'accepts a file without a nonempty filename (%s)',
    (filename) => {
      const block = {
        type: 'file',
        file: {
          file_url: 'https://example.com/report.pdf',
          ...(filename !== undefined ? { filename } : {}),
        },
      };
      const payload: TPayload = [
        { role: 'user', content: 'Get file' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'file',
                name: 'file_tool',
                args: '{}',
                output: [block],
              },
            },
          ],
        },
      ];
      const { messages } = formatAgentMessages(payload);
      const tools = messages.filter(
        (message) => message.getType() === 'tool'
      ) as ToolMessage[];
      expect(tools).toHaveLength(1);
      expect(tools[0].content).toEqual([block]);
      const outputs = _convertMessagesToOpenAIResponsesParams(messages).filter(
        (item) => item.type === 'function_call_output'
      );
      expect(outputs).toEqual([
        expect.objectContaining({
          call_id: 'file',
          output: [
            {
              type: 'input_file',
              file_url: 'https://example.com/report.pdf',
              ...(filename === '' ? { filename: '' } : {}),
            },
          ],
        }),
      ]);
    }
  );
  it('preserves mixed parts through the formatter and emits one output per parallel call', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Get reports' },
      {
        role: 'assistant',
        content: ['a', 'b'].map((id) => ({
          type: ContentTypes.TOOL_CALL,
          tool_call: { id, name: 'report', args: '{}', output: mixed },
        })),
      },
    ];
    const { messages } = formatAgentMessages(payload);
    const tools = messages.filter(
      (message) => message.getType() === 'tool'
    ) as ToolMessage[];
    expect(tools).toHaveLength(2);
    expect(tools.map((message) => message.content)).toEqual([mixed, mixed]);
    expect(
      messages.filter((message) => message.getType() === 'human')
    ).toHaveLength(1);
    const request = _convertMessagesToOpenAIResponsesParams(messages);
    const outputs = request.filter(
      (item) => item.type === 'function_call_output'
    );
    expect(outputs).toHaveLength(2);
    for (const id of ['a', 'b']) {
      expect(outputs.filter((item) => item.call_id === id)).toEqual([
        expect.objectContaining({ call_id: id, output: expected }),
      ]);
    }
  });

  it.each([
    [
      [{ type: 'text', text: 'hello' }],
      [{ type: 'input_text', text: 'hello' }],
    ],
    [[image], [expected[1]]],
    [[file], [expected[2]]],
    [
      [
        {
          type: 'file',
          file: {
            filename: 'report.pdf',
            file_url: 'https://example.com/report.pdf',
          },
        },
      ],
      [
        {
          type: 'input_file',
          filename: 'report.pdf',
          file_url: 'https://example.com/report.pdf',
        },
      ],
    ],
    [
      [
        {
          type: 'file',
          source_type: 'base64',
          data: 'YQ==',
          mime_type: 'application/pdf',
          metadata: { filename: 'report.pdf' },
        },
      ],
      [expected[2]],
    ],
    [
      [
        {
          type: 'file',
          source_type: 'url',
          url: 'https://example.com/report.pdf',
        },
      ],
      [{ type: 'input_file', file_url: 'https://example.com/report.pdf' }],
    ],
    [
      [{ type: 'file', source_type: 'id', id: 'file-123' }],
      [{ type: 'input_file', file_id: 'file-123' }],
    ],
    [expected, expected],
  ])('maps supported content without splitting outputs', (content, output) => {
    const items = _convertMessagesToOpenAIResponsesParams([
      new ToolMessage({ tool_call_id: 'one', content }),
    ]);
    expect(items).toEqual([
      expect.objectContaining({
        type: 'function_call_output',
        call_id: 'one',
        output,
      }),
    ]);
  });

  it('keeps generic JSON arrays as compatible string results', () => {
    const content = [{ type: 'json', rows: [1, 2] }];
    const items = _convertMessagesToOpenAIResponsesParams([
      new ToolMessage({ tool_call_id: 'json', content }),
    ]);
    expect(items[0]).toMatchObject({ output: JSON.stringify(content) });
  });

  it('recognizes raw input_file as atomic and omits oversized media whole', () => {
    const small = { type: 'input_file', file_id: 'file-123' };
    expect(isAtomicToolContentBlock(small)).toBe(true);
    expect(compactToolContent([small], 500).content).toEqual([small]);
    const large = {
      type: 'input_file',
      filename: 'large.pdf',
      file_data: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS + 1000),
    };
    const items = _convertMessagesToOpenAIResponsesParams([
      new ToolMessage({ tool_call_id: 'large', content: [large, image] }),
    ]);
    const wire = JSON.stringify(items[0]);
    expect(wire).not.toContain(large.file_data);
    expect(wire).toContain('omitted');
    // Exhausted character-traversal budgets may conservatively omit later media.
    expect(wire.length).toBeLessThan(HARD_MAX_TOOL_RESULT_CHARS + 200);
    const compacted = compactToolContent(
      [{ type: 'input_file', file_data: 'x'.repeat(2_000) }, image],
      1_000
    ).content;
    expect(compacted).toEqual(expect.arrayContaining([image]));
    expect(JSON.stringify(compacted)).not.toContain('x'.repeat(2_000));
  });

  it('does not execute getters while identifying native parts', () => {
    let reads = 0;
    const content = [
      {
        type: 'file',
        get file() {
          reads++;
          return { file_id: 'file-123' };
        },
      },
    ];
    expect(getResponsesToolOutputParts(content)).toBeUndefined();
    expect(reads).toBe(0);
  });
});

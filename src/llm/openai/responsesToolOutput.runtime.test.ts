import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { ChatOpenAI } from './index';

class InspectableChatOpenAI extends ChatOpenAI {
  get responseDelegate() {
    return this.responses;
  }
}

const image = {
  type: 'image_url',
  image_url: { url: 'https://example.com/chart.png' },
};
const file = {
  type: 'file',
  file: { filename: '', file_url: 'https://example.com/report.pdf' },
};
const content = [{ type: 'text', text: 'Report' }, image, file];

describe('real ChatOpenAI Responses request construction', () => {
  it.each([
    { type: 'file', file: { file_data: '' } },
    { type: 'file', file: { file_url: '' } },
    { type: 'image_url', image_url: 'not-a-url' },
    { type: 'audio', data: 'YQ==' },
  ])(
    'rejects invalid or unsupported media rather than stringifying it: %j',
    async (block) => {
      const model = new InspectableChatOpenAI({
        apiKey: 'test',
        model: 'gpt-4.1',
        useResponsesApi: true,
      });
      const capture = jest
        .spyOn(model.responseDelegate, 'completionWithRetry')
        .mockRejectedValue(new Error('unexpected request construction'));
      await expect(
        model.invoke([
          new ToolMessage({ tool_call_id: 'bad', content: [block] }),
        ])
      ).rejects.toThrow(/Responses tool output/);
      expect(capture).not.toHaveBeenCalled();
    }
  );
  it.each(['invoke', 'chunks', 'events'])(
    'sends one native mixed output per call via %s',
    async (mode) => {
      const model = new InspectableChatOpenAI({
        apiKey: 'test',
        model: 'gpt-4.1',
        useResponsesApi: true,
      });
      const delegate = model.responseDelegate;
      const stop = new Error('captured request without API call');
      const capture = jest
        .spyOn(delegate, 'completionWithRetry')
        .mockRejectedValue(stop);
      const messages = [
        new AIMessage({
          content: '',
          tool_calls: ['a', 'b'].map((id) => ({
            id,
            name: 'report',
            args: {},
            type: 'tool_call',
          })),
        }),
        ...['a', 'b'].map(
          (id) => new ToolMessage({ tool_call_id: id, content })
        ),
      ];
      if (mode === 'invoke') {
        await expect(model.invoke(messages)).rejects.toThrow(stop.message);
      } else {
        const stream =
          mode === 'chunks'
            ? delegate._streamResponseChunks(messages, {})
            : delegate._streamChatModelEvents(messages, {});
        await expect(stream.next()).rejects.toThrow(stop.message);
      }
      expect(capture).toHaveBeenCalledTimes(1);
      const request = capture.mock.calls[0][0];
      const input = request.input as Array<{
        type: string;
        call_id?: string;
        output?: unknown;
      }>;
      const outputs = input.filter(
        (item) => item.type === 'function_call_output'
      );
      expect(outputs).toHaveLength(2);
      for (const id of ['a', 'b']) {
        expect(outputs.filter((item) => item.call_id === id)).toEqual([
          expect.objectContaining({
            output: [
              { type: 'input_text', text: 'Report' },
              {
                type: 'input_image',
                image_url: 'https://example.com/chart.png',
              },
              {
                type: 'input_file',
                filename: '',
                file_url: 'https://example.com/report.pdf',
              },
            ],
          }),
        ]);
      }
      expect(messages[1].content).toEqual(content);
    }
  );

  it('keeps a 17k image URL native and preserves empty output arrays', async () => {
    const model = new InspectableChatOpenAI({
      apiKey: 'test',
      model: 'gpt-4.1',
      useResponsesApi: true,
    });
    const capture = jest
      .spyOn(model.responseDelegate, 'completionWithRetry')
      .mockRejectedValue(new Error('capture'));
    const url = `https://example.com/${'a'.repeat(17_000)}`;
    await expect(
      model.invoke([
        new ToolMessage({
          tool_call_id: 'long',
          content: [
            { type: 'text', text: 'Image' },
            { type: 'image_url', image_url: url },
          ],
        }),
        new ToolMessage({ tool_call_id: 'empty', content: [] }),
      ])
    ).rejects.toThrow('capture');
    expect(capture.mock.calls[0][0].input).toEqual([
      expect.objectContaining({
        call_id: 'long',
        output: [
          { type: 'input_text', text: 'Image' },
          { type: 'input_image', image_url: url },
        ],
      }),
      expect.objectContaining({ call_id: 'empty', output: [] }),
    ]);
  });
});

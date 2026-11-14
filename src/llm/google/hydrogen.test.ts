import { GenerativeModel } from '@google/generative-ai';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { afterEach, describe, expect, jest, test } from '@jest/globals';
import type { EnhancedGenerateContentResponse } from '@google/generative-ai';
import { toLangChainContent } from '@/messages/langchain';
import { CustomChatGoogleGenerativeAI } from './index';

const response: EnhancedGenerateContentResponse = {
  text: () => 'Answer.',
  functionCall: () => undefined,
  functionCalls: () => [],
  candidates: [
    {
      content: {
        role: 'model',
        parts: [{ text: 'Answer.' }],
      },
      index: 0,
    },
  ],
};

function createModel(): CustomChatGoogleGenerativeAI {
  return new CustomChatGoogleGenerativeAI({
    model: 'gemini-4-hydrogen',
    apiKey: 'test-key',
    thinkingConfig: { thinkingLevel: 'XHIGH' },
    _lc_stream_delay: 0,
  });
}

describe('Google Hydrogen request configuration', () => {
  test.each(['application/pdf', 'image/png', 'audio/wav', 'video/mp4'])(
    'serializes supported media %s @scenario:google-hydrogen-media',
    async (mimeType) => {
      const model = createModel();
      const generate = jest
        .spyOn(GenerativeModel.prototype, 'generateContent')
        .mockResolvedValue({ response });
      const message = new HumanMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Describe.' },
          { type: 'media', mimeType, data: 'AA==' },
        ]),
      });
      await model._generate([message], {});
      expect(generate.mock.calls[0][0]).toMatchObject({
        contents: [
          {
            role: 'user',
            parts: [
              { text: 'Describe.' },
              { inlineData: { mimeType, data: 'AA==' } },
            ],
          },
        ],
      });
    }
  );

  afterEach(() => {
    jest.restoreAllMocks();
  });
  test('forwards XHIGH and strips trailing model prefill on invoke @scenario:google-hydrogen-invoke', async () => {
    const model = createModel();
    const generate = jest
      .spyOn(GenerativeModel.prototype, 'generateContent')
      .mockResolvedValue({ response });
    await model._generate(
      [new HumanMessage('Question.'), new AIMessage('Draft.')],
      {}
    );
    expect(generate.mock.calls[0][0]).toMatchObject({
      contents: [{ role: 'user', parts: [{ text: 'Question.' }] }],
    });
    expect(generate.mock.contexts[0]).toHaveProperty(
      'generationConfig.thinkingConfig.thinkingLevel',
      'XHIGH'
    );
  });

  test('strips trailing model prefill on stream @scenario:google-hydrogen-stream', async () => {
    const model = createModel();
    async function* responses(): AsyncGenerator<EnhancedGenerateContentResponse> {
      yield response;
    }
    const generate = jest
      .spyOn(GenerativeModel.prototype, 'generateContentStream')
      .mockResolvedValue({
        stream: responses(),
        response: Promise.resolve(response),
      });
    const chunks = [];
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage('Question.'), new AIMessage('Draft.')],
      {}
    ))
      chunks.push(chunk);
    expect(chunks).toHaveLength(1);
    expect(generate.mock.calls[0][0]).toMatchObject({
      contents: [{ role: 'user', parts: [{ text: 'Question.' }] }],
    });
  });
});

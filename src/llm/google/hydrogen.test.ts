import { afterEach, describe, expect, jest, test } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { EnhancedGenerateContentResponse } from '@google/generative-ai';
import type { BaseMessage } from '@langchain/core/messages';
import { toLangChainContent } from '@/messages/langchain';
import { CustomChatGoogleGenerativeAI } from './index';
import { GoogleTransport } from './transport';

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
  test.each([false, true])(
    'counts thinking as output and cached input on stream=%s @scenario:google-thinking-usage',
    async (stream) => {
      const usageMetadata = {
        promptTokenCount: 3,
        candidatesTokenCount: 2,
        totalTokenCount: 9,
        thoughtsTokenCount: 4,
        cachedContentTokenCount: 1,
        promptTokensDetails: [],
      };
      const responseWithUsage = { ...response, usageMetadata };
      const model = createModel();
      let message: BaseMessage;
      if (stream) {
        async function* responses(): AsyncGenerator<EnhancedGenerateContentResponse> {
          yield responseWithUsage;
        }
        jest
          .spyOn(GoogleTransport.prototype, 'generateContentStream')
          .mockResolvedValue({
            stream: responses(),
          });
        let combined = new AIMessageChunk('');
        for await (const chunk of model._streamResponseChunks(
          [new HumanMessage('Question.')],
          {}
        )) {
          if (!(chunk.message instanceof AIMessageChunk))
            throw new Error('Expected AI chunk');
          combined = combined.concat(chunk.message);
        }
        message = combined;
      } else {
        jest
          .spyOn(GoogleTransport.prototype, 'generateContent')
          .mockResolvedValue({ response: responseWithUsage });
        message = (await model._generate([new HumanMessage('Question.')], {}))
          .generations[0].message;
      }
      expect(message).toMatchObject({
        usage_metadata: {
          input_tokens: 3,
          output_tokens: 6,
          total_tokens: 9,
          input_token_details: { cache_read: 1 },
        },
      });
    }
  );

  test.each(['application/pdf', 'image/png', 'audio/wav', 'video/mp4'])(
    'serializes supported media %s @scenario:google-hydrogen-media',
    async (mimeType) => {
      const model = createModel();
      const generate = jest
        .spyOn(GoogleTransport.prototype, 'generateContent')
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
      .spyOn(GoogleTransport.prototype, 'generateContent')
      .mockResolvedValue({ response });
    await model._generate(
      [new HumanMessage('Question.'), new AIMessage('Draft.')],
      {}
    );
    expect(generate.mock.calls[0][0]).toMatchObject({
      contents: [{ role: 'user', parts: [{ text: 'Question.' }] }],
    });
    expect(generate.mock.calls[0][0]).toMatchObject({
      generationConfig: { thinkingConfig: { thinkingLevel: 'XHIGH' } },
    });
  });

  test('strips trailing model prefill on stream @scenario:google-hydrogen-stream', async () => {
    const model = createModel();
    async function* responses(): AsyncGenerator<EnhancedGenerateContentResponse> {
      yield response;
    }
    const generate = jest
      .spyOn(GoogleTransport.prototype, 'generateContentStream')
      .mockResolvedValue({
        stream: responses(),
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

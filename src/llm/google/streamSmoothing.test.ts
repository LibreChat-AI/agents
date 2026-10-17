import { expect, test, describe, jest } from '@jest/globals';
import { HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import { CustomChatGoogleGenerativeAI } from './index';

describe('Google stream smoothing', () => {
  function textResponse(text: string): Record<string, unknown> {
    return {
      candidates: [
        {
          content: { role: 'model', parts: [{ text }] },
          index: 0,
        },
      ],
    };
  }

  async function runStream(
    responses: Record<string, unknown>[],
    modelFields: Record<string, unknown> = {}
  ): Promise<{
    yielded: ChatGenerationChunk[];
    dispatchedTokens: string[];
  }> {
    const model = new CustomChatGoogleGenerativeAI({
      model: 'gemini-2.5-flash',
      apiKey: 'test-key',
      ...modelFields,
    });

    (
      model as unknown as {
        client: { generateContentStream: unknown };
      }
    ).client.generateContentStream = jest.fn(async () => ({
      response: Promise.resolve(responses[responses.length - 1]),
      stream: (async function* () {
        yield* responses;
      })(),
    }));

    const dispatchedTokens: string[] = [];
    const runManager = {
      handleLLMNewToken: jest.fn(async (token: string) => {
        dispatchedTokens.push(token);
      }),
    } as unknown as CallbackManagerForLLMRun;

    const yielded: ChatGenerationChunk[] = [];
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage('hi')],
      {} as Parameters<
        CustomChatGoogleGenerativeAI['_streamResponseChunks']
      >[1],
      runManager
    )) {
      yielded.push(chunk);
    }
    return { yielded, dispatchedTokens };
  }

  test('splits large text responses at stream boundaries with pacing', async () => {
    const { yielded, dispatchedTokens } = await runStream(
      [textResponse('alpha beta gamma')],
      { _lc_stream_delay: 1 }
    );

    const texts = yielded.map((chunk) => chunk.text).filter(Boolean);
    expect(texts).toEqual(['alpha ', 'beta ', 'gamma']);
    expect(dispatchedTokens.filter(Boolean)).toEqual([
      'alpha ',
      'beta ',
      'gamma',
    ]);
  });

  test('passes chunks through unsplit when smoothing is disabled', async () => {
    const { yielded } = await runStream([textResponse('alpha beta gamma')], {
      _lc_stream_delay: 0,
    });

    expect(yielded.map((chunk) => chunk.text).filter(Boolean)).toEqual([
      'alpha beta gamma',
    ]);
  });

  test('emits the final usage chunk without pacing delay', async () => {
    const start = Date.now();
    const { yielded } = await runStream(
      [
        {
          ...textResponse('short text here'),
          usageMetadata: {
            promptTokenCount: 3,
            candidatesTokenCount: 4,
            totalTokenCount: 7,
          },
        },
      ],
      { _lc_stream_delay: 30 }
    );
    const elapsed = Date.now() - start;

    const usageChunk = yielded.find(
      (chunk) =>
        chunk.text === '' &&
        (chunk.message as AIMessageChunk).usage_metadata != null
    );
    expect(usageChunk).toBeDefined();
    expect(elapsed).toBeLessThan(1000);
  });

  test('defaults to 25ms adaptive smoothing with 0 disabling', () => {
    const base = { model: 'gemini-2.5-flash', apiKey: 'test-key' };
    expect(new CustomChatGoogleGenerativeAI(base)._lc_stream_delay).toBe(25);
    expect(
      new CustomChatGoogleGenerativeAI({ ...base, _lc_stream_delay: 0 })
        ._lc_stream_delay
    ).toBe(0);
  });
});

import { describe, expect, test } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  GoogleLLMResponse,
  GeminiRequest,
} from '@langchain/google-common';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import {
  GEMINI_PARTS,
  restoreGeminiReplayParts,
} from '@/llm/google/utils/signatures';
import { toLangChainContent } from '@/messages/langchain';
import { ChatVertexAI } from './index';

class InspectableVertex extends ChatVertexAI {
  format(messages: BaseMessage[]): Promise<GeminiRequest> {
    return this.connection.formatData(
      messages,
      this.invocationParams({})
    ) as Promise<GeminiRequest>;
  }
  decodeGeneration(response: GoogleLLMResponse): ChatGenerationChunk | null {
    return this.connection.api.responseToChatGeneration(response);
  }
  decodeBlocked(response: GoogleLLMResponse, stream: boolean): void {
    if (stream) this.connection.api.responseToChatGeneration(response);
    else this.connection.api.responseToChatResult(response);
  }
  decode(response: GoogleLLMResponse, stream: boolean): BaseMessage {
    if (stream)
      return this.connection.api.responseToChatGeneration(response)!.message;
    return this.connection.api.responseToChatResult(response).generations[0]
      .message;
  }
}

const nativeParts = [
  { text: 'Checking.', thoughtSignature: 'signed-text' },
  {
    functionCall: {
      id: 'native-first',
      name: 'weather',
      args: { city: 'Rome' },
    },
    thoughtSignature: 'signed-call',
  },
  { text: 'And.' },
  {
    functionCall: {
      id: 'native-second',
      name: 'time',
      args: { city: 'Paris' },
    },
  },
];
const response: GoogleLLMResponse = {
  data: {
    candidates: [
      {
        content: { role: 'model', parts: nativeParts },
        finishReason: 'STOP',
        index: 0,
        safetyRatings: [],
        logprobsResult: { topCandidates: [], chosenCandidates: [] },
      },
    ],
    promptFeedback: { safetyRatings: [] },
    usageMetadata: {
      promptTokenCount: 3,
      candidatesTokenCount: 2,
      totalTokenCount: 9,
      thoughtsTokenCount: 4,
      toolUsePromptTokenCount: 0,
      cachedContentTokenCount: 1,
      promptTokensDetails: [],
      toolUsePromptTokensDetails: [],
      cacheTokensDetails: [],
      candidatesTokensDetails: [],
    },
  },
};

const createModel = () =>
  new InspectableVertex({
    model: 'gemini-4-hydrogen',
    apiKey: 'test-key',
    location: 'global',
    thinkingConfig: { thinkingLevel: 'XHIGH' },
  });

describe('Vertex Hydrogen native history', () => {
  test('never restores stale native signatures through legacy fallback @scenario:vertex-stale-history', async () => {
    const model = createModel();
    const message = model.decode(response, false);
    message.content = 'Changed.';
    const formatted = await model.format([
      new HumanMessage('Question.'),
      message,
      new HumanMessage('Next.'),
    ]);
    expect(
      formatted.contents?.[1].parts.every(
        (part) => part.thoughtSignature == null
      )
    ).toBe(true);
  });

  test.each(['application/pdf', 'image/png', 'audio/wav', 'video/mp4'])(
    'serializes supported media %s @scenario:vertex-hydrogen-media',
    async (mimeType) => {
      const message = new HumanMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Describe.' },
          { type: 'media', mimeType, data: 'AA==' },
        ]),
      });
      const formatted = await createModel().format([message]);
      expect(formatted.contents).toEqual([
        {
          role: 'user',
          parts: [
            { text: 'Describe.' },
            { inlineData: { mimeType, data: 'AA==' } },
          ],
        },
      ]);
    }
  );

  test('counts response usage once for multiple native parts @scenario:vertex-stream-usage', () => {
    const generation = createModel().decodeGeneration(response)!;
    expect(generation.generationInfo?.usage_metadata).toMatchObject({
      input_tokens: 3,
      output_tokens: 6,
      total_tokens: 9,
      input_token_details: { cache_read: 1 },
      output_token_details: { reasoning: 4 },
    });
    if (!(generation.message instanceof AIMessageChunk))
      throw new Error('Expected AI chunk');
    expect(generation.message.usage_metadata).toBeUndefined();
  });

  test.each([false, true])(
    'preserves the safety error for contentless candidates in stream=%s @scenario:vertex-safety-failure',
    (stream) => {
      const model = createModel();
      const blocked = structuredClone(response);
      if (!('candidates' in blocked.data))
        throw new Error('Invalid response fixture');
      blocked.data.candidates[0].finishReason = 'SAFETY';
      Reflect.deleteProperty(blocked.data.candidates[0], 'content');
      expect(() => model.decodeBlocked(blocked, stream)).toThrow(
        'Finish reason: SAFETY'
      );
    }
  );

  test.each([false, true])(
    'retains signed text and unsigned parallel calls from stream=%s through request formatting @scenario:vertex-native-history',
    async (stream) => {
      const model = createModel();
      const message = model.decode(response, stream);
      expect(restoreGeminiReplayParts(message)).toEqual(nativeParts);
      expect(message.additional_kwargs[GEMINI_PARTS]).toHaveLength(4);
      const formatted = await model.format([
        new HumanMessage('Weather?'),
        message,
        new ToolMessage({
          content: 'Sunny.',
          name: 'weather',
          tool_call_id: 'native-first',
        }),
        new ToolMessage({
          content: 'Noon.',
          name: 'time',
          tool_call_id: 'native-second',
        }),
      ]);
      expect(formatted.contents?.[1].parts).toEqual(nativeParts);
      expect(formatted.contents?.[2].role).toBe('user');
      expect(
        formatted.contents?.flatMap((content) =>
          content.parts.flatMap((part) =>
            'functionResponse' in part ? [part.functionResponse.name] : []
          )
        )
      ).toEqual(['weather', 'time']);
      expect(formatted.generationConfig?.thinkingConfig).toMatchObject({
        thinkingLevel: 'XHIGH',
      });
    }
  );

  test('removes Hydrogen prefill while retaining previous signed turns @scenario:vertex-hydrogen-prefill', async () => {
    const model = createModel();
    const message = model.decode(response, false);
    const formatted = await model.format([
      new HumanMessage('Weather?'),
      message,
      new HumanMessage('Next.'),
      new AIMessage('Draft.'),
    ]);
    expect(formatted.contents).toHaveLength(3);
    expect(formatted.contents?.[1].parts).toEqual(nativeParts);
    expect(formatted.contents?.[2].role).toBe('user');
  });

  test('keeps legacy model prefill behavior', async () => {
    const model = new InspectableVertex({
      model: 'gemini-2.5-pro',
      apiKey: 'test-key',
    });
    const formatted = await model.format([
      new HumanMessage('Hi.'),
      new AIMessage('Draft.'),
    ]);
    expect(formatted.contents?.at(-1)?.role).toBe('model');
  });
});

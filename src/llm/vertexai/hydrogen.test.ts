import { describe, expect, test } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type {
  GoogleLLMResponse,
  GeminiRequest,
} from '@langchain/google-common';
import type { BaseMessage } from '@langchain/core/messages';
import {
  GEMINI_PARTS,
  restoreGeminiReplayParts,
} from '@/llm/google/utils/signatures';
import { ChatVertexAI } from './index';

class InspectableVertex extends ChatVertexAI {
  format(messages: BaseMessage[]): Promise<GeminiRequest> {
    return this.connection.formatData(
      messages,
      this.invocationParams({})
    ) as Promise<GeminiRequest>;
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
      name: 'weather',
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
      totalTokenCount: 5,
      thoughtsTokenCount: 0,
      toolUsePromptTokenCount: 0,
      cachedContentTokenCount: 0,
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
          content: 'Rainy.',
          name: 'weather',
          tool_call_id: 'native-second',
        }),
      ]);
      expect(formatted.contents?.[1].parts).toEqual(nativeParts);
      expect(formatted.contents?.[2].role).toBe('user');
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

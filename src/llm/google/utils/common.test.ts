import { expect, test, describe } from '@jest/globals';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type {
  Content,
  EnhancedGenerateContentResponse,
} from '@google/generative-ai';
import {
  convertResponseContentToChatGenerationChunk,
  convertMessageContentToParts,
  dropUnsupportedModelTurnPrefill,
  rejectsModelTurnPrefill,
  isGemini3PlusModel,
} from './common';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { toLangChainContent } from '@/messages/langchain';

describe('convertMessageContentToParts cross-provider tool calls', () => {
  test.each([
    ['gemini-3.5-flash', true],
    ['gemini-4-argon', true],
    ['models/gemini-4-argon-cyber-permissive', true],
    ['gemini-2.5-pro', false],
  ])(
    'adds a placeholder thought signature to unsigned tool calls on %s: %s',
    (model, expected) => {
      const message = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_weather', name: 'weather', args: {} }],
      });

      const [part] = convertMessageContentToParts(message, false, [], model);

      expect('thoughtSignature' in part).toBe(expected);
    }
  );

  test('deduplicates Anthropic tool_use content mirrored in parsed tool_calls', () => {
    const message = new AIMessage({
      content: toLangChainContent([
        { type: 'text', text: 'Checking.' },
        {
          type: 'tool_use',
          id: 'call_weather',
          name: 'weather',
          input: '',
        },
        { type: 'text', index: 0, input: '{"city":"NYC"}' },
      ]),
      tool_calls: [
        {
          id: 'call_weather',
          name: 'weather',
          args: { city: 'NYC' },
        },
      ],
    });

    const parts = convertMessageContentToParts(
      message,
      false,
      [],
      'gemini-3.5-flash'
    );
    const functionCalls = parts.filter((part) => 'functionCall' in part);

    expect(functionCalls).toHaveLength(1);
    expect(parts).not.toContainEqual({});
    expect(functionCalls[0]).toMatchObject({
      functionCall: {
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC' },
      },
    });
  });
});

function buildResponse(
  parts: Array<Record<string, unknown>>
): EnhancedGenerateContentResponse {
  return {
    candidates: [
      {
        content: { role: 'model', parts },
        index: 0,
      },
    ],
  } as unknown as EnhancedGenerateContentResponse;
}

function asAIMessageChunk(message: unknown): AIMessageChunk {
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

describe('convertResponseContentToChatGenerationChunk seal metadata', () => {
  test('stamps an on-arrival seal on function call chunks', () => {
    const chunk = convertResponseContentToChatGenerationChunk(
      buildResponse([
        {
          functionCall: { name: 'weather', args: { city: 'NYC' } },
        },
      ]),
      { usageMetadata: undefined, index: 0 }
    );

    const message = asAIMessageChunk(chunk?.message);
    expect(message.response_metadata).toMatchObject({
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
      [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
    });
    expect(message.tool_call_chunks).toHaveLength(1);
    expect(message.tool_calls?.[0]).toMatchObject({
      name: 'weather',
      args: { city: 'NYC' },
    });
  });

  test('does not stamp seal metadata on text-only chunks', () => {
    const chunk = convertResponseContentToChatGenerationChunk(
      buildResponse([{ text: 'hello' }]),
      { usageMetadata: undefined, index: 0 }
    );

    const metadata = asAIMessageChunk(chunk?.message)
      .response_metadata as Record<string, unknown>;
    expect(metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]).toBeUndefined();
    expect(metadata[STREAMED_TOOL_CALL_SEAL_METADATA_KEY]).toBeUndefined();
  });
});

describe('rejectsModelTurnPrefill', () => {
  test('is true for models that reject a trailing model turn', () => {
    expect(rejectsModelTurnPrefill('gemini-3.7-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-3.6-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-3.5-flash-lite')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-3.6-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-3.7-flash-latest')).toBe(
      true
    );
    expect(rejectsModelTurnPrefill('google/gemini-3.5-flash-lite-latest')).toBe(
      true
    );
  });

  test('is false for models that still accept prefill and for empty input', () => {
    expect(rejectsModelTurnPrefill('gemini-3.5-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-2.5-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3-pro-preview')).toBe(false);
    expect(rejectsModelTurnPrefill(undefined)).toBe(false);
    expect(rejectsModelTurnPrefill('')).toBe(false);
  });

  test('covers Flash releases newer than the cutoff without an entry', () => {
    expect(rejectsModelTurnPrefill('gemini-3.8-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-3.9-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-3.10-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-4.0-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-3.8-flash-latest')).toBe(
      true
    );
    expect(rejectsModelTurnPrefill('google/gemini-3.8-flash-lite')).toBe(true);
  });

  test('covers Gemini 4 Argon and its variants', () => {
    expect(rejectsModelTurnPrefill('gemini-4-argon')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-4-argon')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-4-argon-cyber-permissive')).toBe(
      true
    );
  });

  test('splits the 3.5 generation, where only Flash-Lite rejects prefill', () => {
    expect(rejectsModelTurnPrefill('gemini-3.5-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3.5-flash-latest')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3.5-flash-lite')).toBe(true);
  });

  test('reads a major-only Flash id as `.0`', () => {
    expect(rejectsModelTurnPrefill('gemini-3-flash-preview')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-4-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-4-flash-preview')).toBe(true);
  });

  test('does not widen past Flash to other model lines', () => {
    expect(rejectsModelTurnPrefill('gemini-3.8-pro')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-4.0-pro-preview')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3.7-flashy')).toBe(false);
    expect(rejectsModelTurnPrefill('not-gemini-3.8-flash')).toBe(false);
  });
});

describe('dropUnsupportedModelTurnPrefill', () => {
  const userTurn: Content = { role: 'user', parts: [{ text: 'Hi' }] };
  const modelTurn: Content = {
    role: 'model',
    parts: [{ text: 'Hello, I am' }],
  };

  test('drops a trailing model turn for no-prefill models', () => {
    const contents: Content[] = [userTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(
      contents,
      'gemini-3.6-flash'
    );
    expect(result).toEqual([userTurn]);
  });

  test('drops a trailing model turn for Gemini 3.7 Flash', () => {
    const contents: Content[] = [userTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(
      contents,
      'gemini-3.7-flash'
    );
    expect(result).toEqual([userTurn]);
  });

  test('drops multiple consecutive trailing model turns but keeps one turn', () => {
    const contents: Content[] = [userTurn, modelTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(
      contents,
      'gemini-3.5-flash-lite'
    );
    expect(result).toEqual([userTurn]);
  });

  test('leaves a trailing model turn for models that accept prefill', () => {
    const contents: Content[] = [userTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(
      contents,
      'gemini-3.5-flash'
    );
    expect(result).toBe(contents);
  });

  test('is a no-op when the request already ends with a user turn', () => {
    const contents: Content[] = [modelTurn, userTurn];
    const result = dropUnsupportedModelTurnPrefill(
      contents,
      'gemini-3.6-flash'
    );
    expect(result).toBe(contents);
  });

  test('is a no-op for empty or undefined contents', () => {
    expect(dropUnsupportedModelTurnPrefill([], 'gemini-3.6-flash')).toEqual([]);
    expect(
      dropUnsupportedModelTurnPrefill(undefined, 'gemini-3.6-flash')
    ).toBeUndefined();
  });
});

describe('isGemini3PlusModel', () => {
  test('matches Gemini 3 and later, with or without a prefix', () => {
    expect(isGemini3PlusModel('gemini-3-pro-preview')).toBe(true);
    expect(isGemini3PlusModel('gemini-4-argon')).toBe(true);
    expect(isGemini3PlusModel('models/gemini-4-argon')).toBe(true);
    expect(isGemini3PlusModel('google/gemini-10-pro')).toBe(true);
  });

  test('rejects earlier generations and missing ids', () => {
    expect(isGemini3PlusModel('gemini-2.5-flash')).toBe(false);
    expect(isGemini3PlusModel('gemini-pro')).toBe(false);
    expect(isGemini3PlusModel('gemma-3-27b-it')).toBe(false);
    expect(isGemini3PlusModel(undefined)).toBe(false);
  });
});

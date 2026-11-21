import { expect, test, describe } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  Content,
  EnhancedGenerateContentResponse,
} from '@google/generative-ai';
import {
  convertResponseContentToChatGenerationChunk,
  _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
  convertMessageContentToParts,
  convertBaseMessagesToContent,
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

const serverToolCall = {
  type: 'toolCall',
  toolCall: {
    toolType: 'GOOGLE_SEARCH_WEB',
    args: { queries: ['go'] },
    id: 'srv',
  },
  thoughtSignature: 'sig-tool-call',
};
const serverToolResponse = {
  type: 'toolResponse',
  toolResponse: { toolType: 'GOOGLE_SEARCH_WEB', response: {}, id: 'srv' },
  thoughtSignature: 'sig-tool-response',
};

describe('convertMessageContentToParts cross-provider tool calls', () => {
  test.each([
    ['gemini-3.5-flash', true],
    ['gemini-4-argon', true],
    ['models/gemini-4-argon-cyber-permissive', true],
    ['gemini-2.5-pro', false],
  ])(
    'adds a placeholder thought signature to unsigned tool calls on %s: %s @scenario:gemini-foreign-history',
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

  test.each([
    'gemini-4-hydrogen',
    'models/gemini-4-hydrogen',
    'gemini-4-hydrogen-preview',
  ])('rejects Hydrogen prefill for %s', (model) => {
    expect(rejectsModelTurnPrefill(model)).toBe(true);
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

describe('convertMessageContentToParts function call order', () => {
  const model = 'gemini-4-hydrogen';
  const readCall = { id: 'call_read', name: 'read_file', args: { path: 'a' } };
  const runCall = { id: 'call_run', name: 'run', args: { cmd: 'ls' } };

  test('keeps function calls between server-side tool parts @scenario:gemini-server-tool-order', () => {
    const message = new AIMessage({
      content: toLangChainContent([
        serverToolCall,
        { functionCall: { name: 'read_file', args: { path: 'a' } } },
        { functionCall: { name: 'run', args: { cmd: 'ls' } } },
        serverToolResponse,
      ]),
      tool_calls: [readCall, runCall],
      additional_kwargs: {
        [_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY]: { call_read: 'sig-read' },
      },
    });

    const contents = convertBaseMessagesToContent(
      [message],
      true,
      false,
      model
    ) as Content[];

    expect(contents).toHaveLength(1);
    expect(contents[0].parts).toEqual([
      {
        toolCall: serverToolCall.toolCall,
        thoughtSignature: 'sig-tool-call',
      },
      { functionCall: readCall, thoughtSignature: 'sig-read' },
      { functionCall: runCall },
      {
        toolResponse: serverToolResponse.toolResponse,
        thoughtSignature: 'sig-tool-response',
      },
    ]);
  });

  test('replaces function call blocks by id when they carry one', () => {
    const message = new AIMessage({
      content: toLangChainContent([
        { functionCall: { id: 'call_run', name: 'run', args: {} } },
        { functionCall: { id: 'call_read', name: 'read_file', args: {} } },
      ]),
      tool_calls: [readCall, runCall],
    });

    expect(convertMessageContentToParts(message, true, [], model)).toEqual([
      expect.objectContaining({ functionCall: runCall }),
      expect.objectContaining({ functionCall: readCall }),
    ]);
  });

  test('appends calls after the content when it has no function call blocks', () => {
    const message = new AIMessage({
      content: [{ type: 'text', text: 'Looking.' }],
      tool_calls: [readCall, runCall],
    });

    expect(convertMessageContentToParts(message, true, [], model)).toEqual([
      { text: 'Looking.' },
      expect.objectContaining({ functionCall: readCall }),
      expect.objectContaining({ functionCall: runCall }),
    ]);
  });

  test('matches same-name function calls without ids in order', () => {
    const message = new AIMessage({
      content: toLangChainContent([
        { functionCall: { name: 'read_file', args: { path: 'a' } } },
        { text: 'between', type: 'text' },
        { functionCall: { name: 'read_file', args: { path: 'b' } } },
      ]),
      tool_calls: [
        { id: 'first', name: 'read_file', args: { path: 'a' } },
        { id: 'second', name: 'read_file', args: { path: 'b' } },
      ],
    });

    expect(convertMessageContentToParts(message, true, [], model)).toEqual([
      expect.objectContaining({
        functionCall: { id: 'first', name: 'read_file', args: { path: 'a' } },
      }),
      { text: 'between' },
      expect.objectContaining({
        functionCall: { id: 'second', name: 'read_file', args: { path: 'b' } },
      }),
    ]);
  });

  test('drops a function call block that matches no tool call', () => {
    const message = new AIMessage({
      content: toLangChainContent([
        { type: 'text', text: 'Hi.' },
        { functionCall: { name: 'ghost', args: {} } },
      ]),
      tool_calls: [readCall],
    });

    const parts = convertMessageContentToParts(message, true, [], model);

    expect(parts).toEqual([
      { text: 'Hi.' },
      expect.objectContaining({ functionCall: readCall }),
    ]);
  });

  test('drops function call blocks from tool message content', () => {
    const ai = new AIMessage({ content: '', tool_calls: [readCall] });
    const tool = new ToolMessage({
      content: toLangChainContent([
        { type: 'text', text: 'ok' },
        { functionCall: { name: 'read_file', args: {} } },
      ]),
      name: 'read_file',
      tool_call_id: 'call_read',
    });

    expect(convertMessageContentToParts(tool, true, [ai], model)).toEqual([
      {
        functionResponse: {
          name: 'read_file',
          response: { result: [{ text: 'ok' }] },
          id: 'call_read',
        },
      },
    ]);
  });

  test('keeps plain tool_calls-only messages unchanged', () => {
    const message = new AIMessage({ content: '', tool_calls: [readCall] });

    expect(convertMessageContentToParts(message, true, [], model)).toEqual([
      expect.objectContaining({ functionCall: readCall }),
    ]);
  });
});

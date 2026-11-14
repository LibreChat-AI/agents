import { describe, expect, test } from '@jest/globals';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { EnhancedGenerateContentResponse } from '@google/generative-ai';
import type { GeminiPart } from '@langchain/google-common';
import {
  convertMessageContentToParts,
  convertResponseContentToChatGenerationChunk,
  mapGenerateContentResultToChatResult,
} from './common';
import {
  GEMINI_PARTS,
  isGeminiReplayParts,
  restoreGeminiReplayParts,
  captureGeminiReplayParts,
} from './signatures';

const parts = [
  { text: 'Checking ', thoughtSignature: 'text-signature' },
  {
    functionCall: { id: 'first', name: 'weather', args: { city: 'Rome' } },
    thoughtSignature: 'call-signature',
  },
  { text: 'and ' },
  { functionCall: { id: 'second', name: 'weather', args: { city: 'Paris' } } },
  { text: 'finishing.', thoughtSignature: 'last-signature' },
];

function response(nativeParts: GeminiPart[]): EnhancedGenerateContentResponse {
  return {
    candidates: [{ content: { role: 'model', parts: nativeParts }, index: 0 }],
  } as EnhancedGenerateContentResponse;
}

function plain(message: AIMessage | AIMessageChunk): AIMessage {
  return new AIMessage(
    JSON.parse(
      JSON.stringify({
        content: message.content,
        tool_calls: message.tool_calls,
        additional_kwargs: message.additional_kwargs,
      })
    )
  );
}

describe('Gemini native part replay', () => {
  test.each([
    null,
    [null],
    new Array(1),
    [{ type: 'text', length: -1, hash: 'invalid' }],
    [{ type: 'text', length: 0, hash: 'a'.repeat(64), thoughtSignature: null }],
  ])('rejects malformed descriptors %p @scenario:gemini-malformed-history', (value) => {
    expect(isGeminiReplayParts(value)).toBe(false);
  });
  test('round-trips signed text and unsigned parallel calls through nonstream history @scenario:gemini-native-history', () => {
    const message = mapGenerateContentResultToChatResult(response(parts))
      .generations[0].message as AIMessage;
    const restored = plain(message);
    expect(
      convertMessageContentToParts(restored, true, [], 'gemini-4-hydrogen')
    ).toEqual(parts);
    expect(restored.additional_kwargs[GEMINI_PARTS]).toHaveLength(parts.length);
    expect(
      JSON.stringify(restored.additional_kwargs[GEMINI_PARTS])
    ).not.toContain('Checking');
  });

  test('preserves boundaries and signatures when streaming chunks concatenate @scenario:gemini-stream-history', () => {
    let message = new AIMessageChunk('');
    for (const part of parts) {
      const chunk = convertResponseContentToChatGenerationChunk(
        response([part]),
        { index: 0 }
      );
      message = message.concat(chunk!.message as AIMessageChunk);
    }
    expect(restoreGeminiReplayParts(plain(message))).toEqual(parts);
  });

  test('retains signed reasoning text separately from visible text @scenario:gemini-thought-history', () => {
    const nativeParts = [
      {
        text: 'Reasoning.',
        thought: true,
        thoughtSignature: 'reasoning-signature',
      },
      { text: 'Answer.', thoughtSignature: 'answer-signature' },
    ];
    const message = mapGenerateContentResultToChatResult(response(nativeParts))
      .generations[0].message as AIMessage;
    expect(message.additional_kwargs.reasoning).toBe('Reasoning.');
    expect(restoreGeminiReplayParts(plain(message))).toEqual(nativeParts);
  });

  test('rejects stale replay after a same-length edit or missing call @scenario:gemini-stale-history', () => {
    const message = mapGenerateContentResultToChatResult(response(parts))
      .generations[0].message as AIMessage;
    const edited = plain(message);
    edited.content = 'Changing and finishing.';
    expect(restoreGeminiReplayParts(edited)).toBeUndefined();
    edited.content = message.content;
    edited.tool_calls![0].args = { city: 'Milan' };
    expect(restoreGeminiReplayParts(edited)).toBeUndefined();
    edited.tool_calls = [];
    expect(restoreGeminiReplayParts(edited)).toBeUndefined();
  });

  test('keeps signatures isolated over multiple model turns', () => {
    const first = mapGenerateContentResultToChatResult(response(parts))
      .generations[0].message as AIMessage;
    const secondParts = [
      { text: 'Second answer.', thoughtSignature: 'second-text' },
    ];
    const second = mapGenerateContentResultToChatResult(response(secondParts))
      .generations[0].message as AIMessage;
    expect(restoreGeminiReplayParts(plain(first))).toEqual(parts);
    expect(restoreGeminiReplayParts(plain(second))).toEqual(secondParts);
  });

  test('never fills an unsigned native sibling with a fabricated signature', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'a', name: 'a', args: {} },
        { id: 'b', name: 'b', args: {} },
      ],
      additional_kwargs: {
        __gemini_function_call_thought_signatures__: { a: 'original' },
      },
    });
    expect(
      convertMessageContentToParts(message, true, [], 'gemini-4-hydrogen')
    ).toEqual([
      {
        functionCall: { id: 'a', name: 'a', args: {} },
        thoughtSignature: 'original',
      },
      { functionCall: { id: 'b', name: 'b', args: {} } },
    ]);
  });

  test('supports restored Vertex thinking content without reasoning kwargs', () => {
    const message = new AIMessage({
      content: [
        { type: 'thinking', thinking: 'One.' },
        { type: 'thinking', thinking: 'Two.' },
        { type: 'text', text: 'Answer.' },
      ],
    });
    const nativeParts = [
      { text: 'One.', thought: true, thoughtSignature: 'one' },
      { text: 'Two.', thought: true },
      { text: 'Answer.' },
    ];
    captureGeminiReplayParts(message, nativeParts);
    delete message.additional_kwargs.reasoning;
    expect(restoreGeminiReplayParts(message)).toEqual(nativeParts);
  });
});

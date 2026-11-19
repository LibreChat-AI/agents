import { Outcome } from '@google/generative-ai';
import { describe, expect, test } from '@jest/globals';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type {
  EnhancedGenerateContentResponse,
  Part,
} from '@google/generative-ai';
import type { GeminiPart } from '@langchain/google-common';
import type { TPayload } from '@/types';
import {
  GEMINI_PARTS,
  GEMINI_REPLAY_UNSUPPORTED,
  GEMINI_SIGNATURES,
  isGeminiReplayParts,
  restoreGeminiReplayParts,
  captureGeminiReplayParts,
} from './signatures';
import {
  convertMessageContentToParts,
  convertResponseContentToChatGenerationChunk,
  mapGenerateContentResultToChatResult,
} from './common';
import { formatContentStrings } from '@/messages/content';
import { formatAgentMessages } from '@/messages/format';
import { ContentTypes } from '@/common';

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

function response(
  nativeParts: Array<GeminiPart | Part>
): EnhancedGenerateContentResponse {
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
  test.each([false, true])(
    'retains call signatures with unsupported output in split=%s @scenario:gemini-mixed-output-history',
    (split) => {
      const unsupported = {
        codeExecutionResult: { outcome: Outcome.OUTCOME_OK, output: '1' },
      };
      const mixed = [
        parts[0],
        unsupported,
        ...parts.slice(1, 4),
        unsupported,
        parts[4],
      ];
      const chunks = split ? mixed.map((part) => [part]) : [mixed];
      let message = new AIMessageChunk('');
      for (const chunk of chunks)
        message = message.concat(
          convertResponseContentToChatGenerationChunk(response(chunk), {
            index: 0,
          })!.message as AIMessageChunk
        );
      expect(message.additional_kwargs[GEMINI_REPLAY_UNSUPPORTED]).toBe(true);
      const fresh = new AIMessage({
        content: message.content,
        tool_calls: message.tool_calls,
      });
      captureGeminiReplayParts(fresh, mixed, 'gemini-4-hydrogen');
      expect(fresh.additional_kwargs[GEMINI_SIGNATURES]).toEqual({
        first: 'call-signature',
      });
      expect(fresh.additional_kwargs[GEMINI_REPLAY_UNSUPPORTED]).toBe(true);
      expect(
        convertMessageContentToParts(
          message,
          true,
          [],
          'gemini-4-hydrogen'
        ).filter((part) => 'functionCall' in part)
      ).toEqual(parts.filter((part) => 'functionCall' in part));
    }
  );

  test('preserves fresh native hashes through graph legacy projection @scenario:gemini-graph-history', () => {
    const message = mapGenerateContentResultToChatResult(response(parts))
      .generations[0].message;
    message.content = parts.flatMap((part) =>
      typeof part.text === 'string'
        ? [{ type: 'text' as const, text: part.text }]
        : []
    );
    const projected = formatContentStrings([message])[0];
    expect(restoreGeminiReplayParts(projected)).toEqual(parts);
  });

  test('retains older-model stream signatures without partially capturing text @scenario:gemini-older-stream-history', () => {
    let message = new AIMessageChunk('');
    const nativeParts = [
      { text: 'Checking ' },
      {
        functionCall: { id: 'legacy', name: 'weather', args: {} },
        thoughtSignature: 'original',
      },
    ];
    for (const part of nativeParts) {
      message = message.concat(
        convertResponseContentToChatGenerationChunk(response([part]), {
          index: 0,
          model: 'gemini-2.5-pro',
        })!.message as AIMessageChunk
      );
    }
    expect(message.additional_kwargs[GEMINI_PARTS]).toBeUndefined();
    expect(
      convertMessageContentToParts(message, true, [], 'gemini-4-hydrogen')
    ).toContainEqual({
      functionCall: { id: 'legacy', name: 'weather', args: {} },
      thoughtSignature: 'original',
    });
  });

  test('preserves signed text bytes through subsequent legacy flattening @scenario:gemini-formatted-text', () => {
    const nativeParts = [
      { text: ' Leading', thoughtSignature: 'one' },
      { text: ' ' },
      { text: 'and trailing. ', thoughtSignature: 'two' },
    ];
    const native = mapGenerateContentResultToChatResult(response(nativeParts))
      .generations[0].message;
    const payload: TPayload = [
      {
        role: 'assistant',
        content: nativeParts.map((part) => ({
          type: ContentTypes.TEXT,
          text: part.text,
        })),
      },
    ];
    const formatted = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { preserveTextContent: true }
    ).messages;
    const projected = formatContentStrings(formatted);
    expect(projected[0].content).toBe(' Leading and trailing. ');
    expect(
      restoreGeminiReplayParts(
        projected[0],
        native.additional_kwargs[GEMINI_PARTS]
      )
    ).toEqual(nativeParts);
  });

  test.each([false, true])(
    'restores actual formatter reasoning and exact text with legacyContent=%s @scenario:gemini-formatted-history',
    (legacyContent) => {
      const nativeParts = [
        {
          text: 'First thought. ',
          thought: true,
          thoughtSignature: 'thought-one',
        },
        { text: 'Second thought.', thought: true },
        { text: ' Answer', thoughtSignature: 'text-one' },
        { text: ' ' },
        { text: 'continues. ', thoughtSignature: 'text-two' },
      ];
      const native = mapGenerateContentResultToChatResult(response(nativeParts))
        .generations[0].message;
      const payload: TPayload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.THINK,
              think: 'First thought. Second thought.',
            },
            { type: ContentTypes.TEXT, text: ' Answer' },
            { type: ContentTypes.TEXT, text: ' ' },
            { type: ContentTypes.TEXT, text: 'continues. ' },
          ],
        },
      ];
      const formatted = formatAgentMessages(
        payload,
        undefined,
        undefined,
        undefined,
        {
          preserveTextContent: true,
          preserveReasoningContent: true,
          legacyContent,
        }
      ).messages[0];
      expect(formatted.content).toBe(' Answer continues. ');
      expect(formatted.additional_kwargs.reasoning_content).toBe(
        'First thought. Second thought.'
      );
      expect(
        restoreGeminiReplayParts(
          formatted,
          native.additional_kwargs[GEMINI_PARTS]
        )
      ).toEqual(nativeParts);
    }
  );

  test('restores exact text and reasoning across the actual formatter tool boundary @scenario:gemini-formatted-tools', () => {
    const nativeParts = [
      { text: 'Planning.', thought: true, thoughtSignature: 'thought' },
      { text: 'Checking ', thoughtSignature: 'text' },
      { text: 'now.' },
      {
        functionCall: { id: 'a', name: 'weather', args: { city: 'Rome' } },
        thoughtSignature: 'call',
      },
      { functionCall: { id: 'b', name: 'weather', args: { city: 'Paris' } } },
    ];
    const native = mapGenerateContentResultToChatResult(response(nativeParts))
      .generations[0].message;
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.THINK, think: 'Planning.' },
          { type: ContentTypes.TEXT, text: 'Checking ' },
          { type: ContentTypes.TEXT, text: 'now.', tool_call_ids: ['a', 'b'] },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'a',
              name: 'weather',
              args: '{"city":"Rome"}',
              output: 'Sunny',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'b',
              name: 'weather',
              args: '{"city":"Paris"}',
              output: 'Rainy',
            },
          },
        ],
      },
    ];
    const formatted = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { preserveTextContent: true, preserveReasoningContent: true }
    ).messages[0];
    expect(formatted.content).toBe('Checking now.');
    expect(
      restoreGeminiReplayParts(
        formatted,
        native.additional_kwargs[GEMINI_PARTS]
      )
    ).toEqual(nativeParts);
  });

  test('preserves reasoning-only history through the formatter @scenario:gemini-reasoning-only-history', () => {
    const nativeParts = [
      { text: 'Still thinking.', thought: true, thoughtSignature: 'thought' },
    ];
    const native = mapGenerateContentResultToChatResult(response(nativeParts))
      .generations[0].message;
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [{ type: ContentTypes.THINK, think: 'Still thinking.' }],
      },
    ];
    const formatted = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { preserveTextContent: true, preserveReasoningContent: true }
    ).messages[0];
    expect(
      restoreGeminiReplayParts(
        formatted,
        native.additional_kwargs[GEMINI_PARTS]
      )
    ).toEqual(nativeParts);
  });

  test('keeps older unsigned model history eligible for cross-model fallback @scenario:gemini-cross-model-history', () => {
    const nativeParts = [
      { functionCall: { id: 'old-call', name: 'weather', args: {} } },
    ];
    const message = mapGenerateContentResultToChatResult(
      response(nativeParts),
      { usageMetadata: undefined, model: 'gemini-2.0-flash' }
    ).generations[0].message;
    expect(message.additional_kwargs[GEMINI_PARTS]).toBeUndefined();
    expect(
      convertMessageContentToParts(message, true, [], 'gemini-4-hydrogen')[0]
    ).toHaveProperty('thoughtSignature');
  });

  test.each([
    null,
    [null],
    new Array(1),
    [{ type: 'text', length: -1, hash: 'invalid' }],
    [{ type: 'text', length: 0, hash: 'a'.repeat(64), thoughtSignature: null }],
  ])(
    'rejects malformed descriptors %p @scenario:gemini-malformed-history',
    (value) => {
      expect(isGeminiReplayParts(value)).toBe(false);
    }
  );
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
    const wire = convertMessageContentToParts(
      edited,
      true,
      [],
      'gemini-4-hydrogen'
    );
    expect(wire.every((part) => !('thoughtSignature' in part))).toBe(true);
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

  test.each(['thinking', 'reasoning'] as const)(
    'validates %s blocks and rejects same-length edits even with captured kwargs',
    (type) => {
      const message = new AIMessage({
        content: [
          type === 'thinking'
            ? { type, thinking: 'Plan.' }
            : { type, reasoning: 'Plan.' },
        ],
        tool_calls: [{ id: 'a', name: 'calculator', args: { value: 1 } }],
      });
      const nativeParts = [
        { text: 'Plan.', thought: true, thoughtSignature: 'thought-signature' },
        {
          functionCall: { id: 'a', name: 'calculator', args: { value: 1 } },
          thoughtSignature: 'call-signature',
        },
      ];
      captureGeminiReplayParts(message, nativeParts, 'gemini-3.8-flash');
      expect(restoreGeminiReplayParts(message)).toEqual(nativeParts);
      message.additional_kwargs.reasoning = 'Plan.';
      message.content = [
        type === 'thinking'
          ? { type, thinking: 'Edit.' }
          : { type, reasoning: 'Edit.' },
      ];
      expect(restoreGeminiReplayParts(message)).toBeUndefined();
    }
  );

  test('restores split Vertex reasoning blocks without duplicate kwargs', () => {
    const message = new AIMessage({
      content: [
        { type: 'reasoning', reasoning: 'One.' },
        { type: 'reasoning', reasoning: '' },
        { type: 'reasoning', reasoning: 'Two.' },
        { type: 'text', text: 'Answer.' },
      ],
    });
    const nativeParts = [
      { text: 'One.', thought: true, thoughtSignature: 'one' },
      { text: '', thought: true, thoughtSignature: 'empty-thought' },
      { text: 'Two.', thought: true },
      { text: 'Answer.' },
    ];
    captureGeminiReplayParts(message, nativeParts, 'gemini-4-hydrogen');
    delete message.additional_kwargs.reasoning;
    expect(restoreGeminiReplayParts(message)).toEqual(nativeParts);
  });
});

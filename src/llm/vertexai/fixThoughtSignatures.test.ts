import { expect, test, describe } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { GeminiContent } from '@langchain/google-common';
import { fixThoughtSignatures } from './index';

const SIG_A = 'AY89a1/sigA==';
const SIG_B = 'AY89a1/sigB==';

const buildContents = (
  blocks: Array<['user' | 'model' | 'function', GeminiContent['parts']]>
): GeminiContent[] =>
  blocks.map(([role, parts]) => ({ role, parts }) as GeminiContent);

describe('fixThoughtSignatures', () => {
  test('restores reliable legacy call maps without signing unsigned siblings @scenario:vertex-legacy-call-map', () => {
    const ai = new AIMessage({
      content: 'Checking.',
      tool_calls: [
        { id: 'a', name: 'a', args: {} },
        { id: 'b', name: 'b', args: {} },
      ],
      additional_kwargs: {
        __gemini_function_call_thought_signatures__: { a: SIG_A },
      },
    });
    const contents = buildContents([
      [
        'model',
        [
          { text: 'Checking.' },
          { functionCall: { name: 'a', args: {} } },
          { functionCall: { name: 'b', args: {} }, thoughtSignature: SIG_B },
        ],
      ],
    ]);
    fixThoughtSignatures(contents, [ai]);
    expect(contents[0].parts[1].thoughtSignature).toBe(SIG_A);
    expect(contents[0].parts[2].thoughtSignature).toBeUndefined();
  });

  test('declines legacy text-call-text ambiguity @scenario:vertex-ambiguous-legacy-history', () => {
    const ai = new AIMessage({
      content: 'Merged text.',
      tool_calls: [{ id: 'a', name: 'a', args: {} }],
      additional_kwargs: { signatures: ['', SIG_A, SIG_B] },
    });
    const contents = buildContents([
      [
        'model',
        [{ text: 'Merged text.' }, { functionCall: { name: 'a', args: {} } }],
      ],
    ]);
    fixThoughtSignatures(contents, [ai]);
    expect(contents[0].parts[1].thoughtSignature).toBeUndefined();
  });

  test('declines ambiguous call offsets after signed text chunks merged @scenario:vertex-signed-legacy-history', () => {
    const ai = new AIMessage({
      content: 'Merged text.',
      tool_calls: [
        { id: 'a', name: 'a', args: {} },
        { id: 'b', name: 'b', args: {} },
      ],
      additional_kwargs: { signatures: [SIG_A, '', SIG_B, ''] },
    });
    const contents = buildContents([
      [
        'model',
        [
          { text: 'Merged text.' },
          { functionCall: { name: 'a', args: {} } },
          { functionCall: { name: 'b', args: {} } },
        ],
      ],
    ]);
    fixThoughtSignatures(contents, [ai]);
    expect(contents[0].parts[1].thoughtSignature).toBeUndefined();
    expect(contents[0].parts[2].thoughtSignature).toBeUndefined();
  });

  test('declines merged legacy text without original boundaries @scenario:vertex-merged-legacy-history', () => {
    const ai = new AIMessage({
      content: 'Merged text.',
      tool_calls: [
        { id: 'a', name: 'a', args: {} },
        { id: 'b', name: 'b', args: {} },
      ],
      additional_kwargs: { signatures: ['', '', SIG_A, ''] },
    });
    const contents = buildContents([
      [
        'model',
        [
          { text: 'Merged text.' },
          { functionCall: { name: 'a', args: {} } },
          { functionCall: { name: 'b', args: {} } },
        ],
      ],
    ]);
    fixThoughtSignatures(contents, [ai]);
    expect(contents[0].parts[1].thoughtSignature).toBeUndefined();
    expect(contents[0].parts[2].thoughtSignature).toBeUndefined();
  });

  test('attaches signature to functionCall part when prior turn is a plain-text AI message (issue LibreChat#13006-followup)', () => {
    // Reproduces the live failure from the issue: a Gemini 3 conversation
    // where turn 1 was plain text ("Hello!") and turn 2 emitted a tool call
    // with a thought signature. The plain-text AI message has no signatures,
    // so the old position-by-filter code matched the toolcall AIMessage with
    // the WRONG model content.
    const helloAi = new AIMessage('Hello! How can I help you today?');
    const toolcallAi = new AIMessage({
      content: '',
      tool_calls: [
        { name: 'bash_tool', args: { command: 'echo hi' }, id: 'tc1' },
      ],
      additional_kwargs: { signatures: [SIG_A, ''] },
    });
    const input = [
      new HumanMessage('hi there'),
      helloAi,
      new HumanMessage('run something'),
      toolcallAi,
      new ToolMessage({ content: 'ok', tool_call_id: 'tc1' }),
    ];

    const contents = buildContents([
      ['user', [{ text: 'hi there' }]],
      ['model', [{ text: 'Hello! How can I help you today?' }]],
      ['user', [{ text: 'run something' }]],
      [
        'model',
        [{ functionCall: { name: 'bash_tool', args: { command: 'echo hi' } } }],
      ],
      [
        'user',
        [
          {
            functionResponse: {
              name: 'bash_tool',
              response: { content: 'ok' },
            },
          },
        ],
      ],
    ]);

    fixThoughtSignatures(contents, input);

    expect(contents[1].parts[0].thoughtSignature).toBeUndefined();
    expect(contents[3].parts[0]).toMatchObject({
      functionCall: { name: 'bash_tool' },
      thoughtSignature: SIG_A,
    });
  });

  test('attaches signatures across multiple tool-call turns by position', () => {
    const turn1 = new AIMessage({
      content: '',
      tool_calls: [{ name: 'a', args: {}, id: 't1' }],
      additional_kwargs: { signatures: [SIG_A, ''] },
    });
    const turn2 = new AIMessage({
      content: '',
      tool_calls: [{ name: 'b', args: {}, id: 't2' }],
      additional_kwargs: { signatures: [SIG_B, ''] },
    });

    const input = [
      new HumanMessage('q1'),
      turn1,
      new ToolMessage({ content: '1', tool_call_id: 't1' }),
      new HumanMessage('q2'),
      turn2,
      new ToolMessage({ content: '2', tool_call_id: 't2' }),
    ];
    const contents = buildContents([
      ['user', [{ text: 'q1' }]],
      ['model', [{ functionCall: { name: 'a', args: {} } }]],
      ['user', [{ functionResponse: { name: 'a', response: {} } }]],
      ['user', [{ text: 'q2' }]],
      ['model', [{ functionCall: { name: 'b', args: {} } }]],
      ['user', [{ functionResponse: { name: 'b', response: {} } }]],
    ]);

    fixThoughtSignatures(contents, input);

    expect(contents[1].parts[0].thoughtSignature).toBe(SIG_A);
    expect(contents[4].parts[0].thoughtSignature).toBe(SIG_B);
  });

  test('does not overwrite signatures already attached by the library', () => {
    const ai = new AIMessage({
      content: '',
      tool_calls: [{ name: 'a', args: {}, id: 't1' }],
      additional_kwargs: { signatures: [SIG_A] },
    });
    const input = [new HumanMessage('q'), ai];
    const contents = buildContents([
      ['user', [{ text: 'q' }]],
      [
        'model',
        [{ functionCall: { name: 'a', args: {} }, thoughtSignature: SIG_B }],
      ],
    ]);

    fixThoughtSignatures(contents, input);

    expect(contents[1].parts[0].thoughtSignature).toBe(SIG_B);
  });

  test('preserves unsigned legacy slots instead of shifting a text signature onto a call @scenario:vertex-legacy-history', () => {
    const ai = new AIMessage({
      content: 'Checking.',
      tool_calls: [
        { name: 'a', args: {}, id: 'a' },
        { name: 'b', args: {}, id: 'b' },
      ],
      additional_kwargs: { signatures: [SIG_A, '', SIG_B] },
    });
    const contents = buildContents([
      [
        'model',
        [
          { text: 'Checking.' },
          { functionCall: { name: 'a', args: {} } },
          { functionCall: { name: 'b', args: {} } },
        ],
      ],
    ]);
    fixThoughtSignatures(contents, [ai]);
    expect(contents[0].parts[0].thoughtSignature).toBe(SIG_A);
    expect(contents[0].parts[1].thoughtSignature).toBeUndefined();
    expect(contents[0].parts[2].thoughtSignature).toBe(SIG_B);
  });

  test('no-op when AI message has no signatures', () => {
    const ai = new AIMessage({
      content: '',
      tool_calls: [{ name: 'a', args: {}, id: 't1' }],
    });
    const input = [new HumanMessage('q'), ai];
    const contents = buildContents([
      ['user', [{ text: 'q' }]],
      ['model', [{ functionCall: { name: 'a', args: {} } }]],
    ]);

    fixThoughtSignatures(contents, input);

    expect(contents[1].parts[0].thoughtSignature).toBeUndefined();
  });

  test('skips empty-string signatures', () => {
    const ai = new AIMessage({
      content: '',
      tool_calls: [{ name: 'a', args: {}, id: 't1' }],
      additional_kwargs: { signatures: ['', '', ''] },
    });
    const input = [new HumanMessage('q'), ai];
    const contents = buildContents([
      ['user', [{ text: 'q' }]],
      ['model', [{ functionCall: { name: 'a', args: {} } }]],
    ]);

    fixThoughtSignatures(contents, input);

    expect(contents[1].parts[0].thoughtSignature).toBeUndefined();
  });
});

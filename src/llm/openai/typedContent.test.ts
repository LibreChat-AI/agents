import type { OpenAIClient } from '@langchain/openai';
import {
  splitTypedDeltaContent,
  flattenTypedMessageContent,
} from './typedContent';

type Chunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;
type Completion = OpenAIClient.Chat.Completions.ChatCompletion;

const think = (text: string): object => ({
  type: 'thinking',
  thinking: [{ type: 'text', text }],
});
const text = (value: string): object => ({ type: 'text', text: value });

function chunk(
  delta: Record<string, unknown>,
  finishReason: string | null = null
): Chunk {
  return {
    id: 'chunk',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'mistral-large-4',
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  } as unknown as Chunk;
}

function completion(content: unknown): Completion {
  return {
    id: 'completion',
    object: 'chat.completion',
    created: 0,
    model: 'mistral-large-4',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content, refusal: null },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
  } as unknown as Completion;
}

const deltas = (chunks: Chunk[] | undefined): unknown[] =>
  (chunks ?? []).map((piece) => piece.choices[0].delta);

describe('splitTypedDeltaContent', () => {
  it('leaves string, empty, deltaless and choiceless chunks alone', () => {
    const filtered = chunk({});
    delete (filtered.choices[0] as { delta?: unknown }).delta;

    expect(splitTypedDeltaContent(chunk({ content: 'Hi' }))).toBeUndefined();
    expect(splitTypedDeltaContent(chunk({ content: null }))).toBeUndefined();
    expect(splitTypedDeltaContent(chunk({}))).toBeUndefined();
    expect(splitTypedDeltaContent(filtered)).toBeUndefined();
    expect(
      splitTypedDeltaContent({ ...chunk({}), choices: [] })
    ).toBeUndefined();
  });

  it('joins text parts into string content', () => {
    const pieces = splitTypedDeltaContent(
      chunk({ content: [text('A free'), text('lancer')] })
    );

    expect(deltas(pieces)).toEqual([{ content: 'A freelancer' }]);
  });

  it('keeps a reasoning-only chunk whole, with its finish reason', () => {
    const pieces = splitTypedDeltaContent(
      chunk({ role: 'assistant', content: [think('Plan.')] }, 'stop')
    );

    expect(deltas(pieces)).toEqual([
      { role: 'assistant', content: '', reasoning_content: 'Plan.' },
    ]);
    expect(pieces?.[0].choices[0].finish_reason).toBe('stop');
  });

  it('streams the closing reasoning before the answer it opens', () => {
    const pieces = splitTypedDeltaContent(
      chunk(
        { role: 'assistant', content: [think(' done.'), text('A freel')] },
        'stop'
      )
    );

    expect(deltas(pieces)).toEqual([
      { role: 'assistant', content: '', reasoning_content: ' done.' },
      { role: 'assistant', content: 'A freel' },
    ]);
    expect(pieces?.map((piece) => piece.choices[0].finish_reason)).toEqual([
      null,
      'stop',
    ]);
  });

  it('keeps logprobs on the answer half of a split chunk only', () => {
    const logprobs = { content: [], refusal: null };
    const typed = chunk({ content: [think('.'), text('A')] });
    typed.choices[0].logprobs = logprobs;

    const pieces = splitTypedDeltaContent(typed);

    expect(pieces?.map((piece) => piece.choices[0].logprobs)).toEqual([
      null,
      logprobs,
    ]);
  });

  it('streams the closing reasoning before the tool call it opens', () => {
    const toolCall = {
      index: 0,
      id: 'call_1',
      type: 'function',
      function: { name: 'read_page', arguments: '{' },
    };
    const pieces = splitTypedDeltaContent(
      chunk({ content: [think('.')], tool_calls: [toolCall] })
    );

    expect(deltas(pieces)).toEqual([
      { content: '', reasoning_content: '.' },
      { content: '', tool_calls: [toolCall] },
    ]);
  });

  it('ignores parts that are neither text nor thinking', () => {
    const pieces = splitTypedDeltaContent(
      chunk({ content: [{ type: 'reference', reference_ids: [1] }, text('A')] })
    );

    expect(deltas(pieces)).toEqual([{ content: 'A' }]);
  });
});

describe('flattenTypedMessageContent', () => {
  it('returns a string-content or choiceless completion as is', () => {
    const response = completion('A freelancer.');
    const failed = { error: { message: 'busy' } } as unknown as Completion;

    expect(flattenTypedMessageContent(response)).toBe(response);
    expect(flattenTypedMessageContent(failed)).toBe(failed);
  });

  it('splits typed content into the answer and its reasoning', () => {
    const response = flattenTypedMessageContent(
      completion([think('Plan.'), text('A freelancer.')])
    );

    expect(response.choices[0].message).toEqual({
      role: 'assistant',
      content: 'A freelancer.',
      refusal: null,
      reasoning_content: 'Plan.',
    });
  });
});

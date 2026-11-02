import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  toGenerationSmoothItem,
  hasReasoningKwargs,
  getReasoningKwargsText,
  cloneGenerationChunkPiece,
} from './chunkAdapters';

type MessageChunkFields = {
  usage_metadata?: AIMessageChunk['usage_metadata'];
  additional_kwargs?: AIMessageChunk['additional_kwargs'];
  response_metadata?: AIMessageChunk['response_metadata'];
};

function textChunk(
  text: string,
  extra: MessageChunkFields = {}
): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text,
    message: new AIMessageChunk({ content: text, ...extra }),
  });
}

describe('hasReasoningKwargs', () => {
  it.each([
    { reasoning: '' },
    { reasoning: ' \n\t' },
    { reasoning_content: '' },
    { reasoning_content: ' \n\t' },
    { reasoning_details: [] },
    { reasoning: {} },
    { reasoning: { id: 'rs_1', status: 'completed' } },
    { reasoning: { summary: [] } },
    { reasoning: { summary: [{ type: 'summary_text', text: ' \n' }] } },
    { reasoning: { summary: [null, {}, { type: 'summary_text' }] } },
    { reasoning: { content: [{ type: 'reasoning_text', text: '' }] } },
    { reasoning: { encrypted_content: '' } },
    { reasoning: { encrypted_content: ' ' } },
    { reasoning: { effort: 'high', metadata: { text: 'not a payload' } } },
    { reasoning: false },
    { reasoning: 5 },
    { reasoning_details: [null, {}] },
    { reasoning_details: [{ type: 'reasoning.text', text: ' ' }] },
    { reasoning_details: [{ type: 'reasoning.summary', summary: '' }] },
    { reasoning_details: [{ type: 'reasoning.encrypted', id: 'rs_1' }] },
    { reasoning_details: [{ type: 'reasoning.encrypted', data: '' }] },
    { reasoning_details: [{ type: 'reasoning.encrypted', data: ' ' }] },
  ])('rejects empty reasoning evidence: %j', (additional_kwargs) => {
    expect(
      hasReasoningKwargs(new AIMessageChunk({ content: '', additional_kwargs }))
    ).toBe(false);
    expect(
      toGenerationSmoothItem(textChunk('visible output', { additional_kwargs }))
        .atomic
    ).not.toBe(true);
  });

  it.each([
    { reasoning: 'thought' },
    { reasoning_content: 'thought' },
    { reasoning: { summary: [{ text: 'thought' }] } },
    { reasoning: { summary: [{ text: ' ' }, { text: 'later thought' }] } },
    { reasoning: { content: [{ type: 'reasoning_text', text: 'thought' }] } },
    { reasoning: { encrypted_content: 'opaque' } },
    { reasoning: { summary: [], encrypted_content: 'opaque' } },
    { reasoning_details: [{ type: 'reasoning.text', text: 'thought' }] },
    { reasoning_details: [{ type: 'reasoning.encrypted', data: 'opaque' }] },
    {
      reasoning_details: [{}, { type: 'reasoning.encrypted', data: 'opaque' }],
    },
  ])('retains genuine reasoning evidence: %j', (additional_kwargs) => {
    expect(
      hasReasoningKwargs(new AIMessageChunk({ content: '', additional_kwargs }))
    ).toBe(true);
    expect(
      toGenerationSmoothItem(textChunk('visible output', { additional_kwargs }))
        .atomic
    ).toBe(true);
  });
});

describe('toGenerationSmoothItem classification', () => {
  it('splits plain string-content text chunks', () => {
    const item = toGenerationSmoothItem(textChunk('alpha beta gamma'));
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBeUndefined();
    expect(item.text).toBe('alpha beta gamma');
  });

  it('keeps chunks carrying reasoning_content kwargs atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: { reasoning_content: 'hidden thought' },
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBe(true);
    expect(item.emit({ text: item.text, isFirst: true, isLast: true })).toBe(
      chunk
    );
  });

  it('keeps chunks carrying a reasoning summary object atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: { reasoning: { summary: [{ text: 'thought' }] } },
    });
    expect(toGenerationSmoothItem(chunk).atomic).toBe(true);
  });

  it('keeps chunks carrying OpenRouter reasoning_details atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: {
        reasoning_details: [{ type: 'reasoning.text', text: 'thought' }],
      },
    });
    expect(toGenerationSmoothItem(chunk).atomic).toBe(true);
  });

  it('keeps chunks carrying camelCase finishReason atomic', () => {
    const chunk = new ChatGenerationChunk({
      text: 'final text with several words here',
      generationInfo: { finishReason: 'STOP' },
      message: new AIMessageChunk({
        content: 'final text with several words here',
      }),
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.atomic).toBe(true);
    expect(item.emit({ text: item.text, isFirst: true, isLast: true })).toBe(
      chunk
    );
  });

  it('keeps mixed text/tool-call chunks atomic', () => {
    const chunk = new ChatGenerationChunk({
      text: 'calling the weather tool for you now',
      message: new AIMessageChunk({
        content: 'calling the weather tool for you now',
        tool_call_chunks: [
          { name: 'weather', args: '{"city":', id: 'call_1', index: 0 },
        ],
      }),
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.atomic).toBe(true);
    expect(item.emit({ text: item.text, isFirst: true, isLast: true })).toBe(
      chunk
    );
  });

  it('paces reasoning-only chunks atomically via the kwargs extractor', () => {
    const thoughtOnly = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        additional_kwargs: { reasoning: 'a hidden gemini thought' },
      }),
    });
    const item = toGenerationSmoothItem(thoughtOnly, getReasoningKwargsText);
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBe(true);
    expect(item.text).toBe('a hidden gemini thought');
  });

  it('paces reasoning_details-only chunks atomically via the kwargs extractor', () => {
    const detailsOnly = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        additional_kwargs: {
          reasoning_details: [
            { type: 'reasoning.text', text: 'first thought ' },
            { type: 'reasoning.text', text: 'second thought' },
          ],
        },
      }),
    });
    const item = toGenerationSmoothItem(detailsOnly, getReasoningKwargsText);
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBe(true);
    expect(item.text).toBe('first thought second thought');
  });

  it('classifies usage-only chunks as passthrough', () => {
    const chunk = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
      }),
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.smooth).toBe(false);
    expect(item.text).toBe('');
  });
});

describe('cloneGenerationChunkPiece metadata scoping', () => {
  const chunk = textChunk('alpha beta gamma', {
    usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
    additional_kwargs: { annotation: 'once' },
    response_metadata: { model_name: 'test-model' },
  });

  it('keeps kwargs, response metadata and usage on the first piece only', () => {
    const first = cloneGenerationChunkPiece(chunk, {
      text: 'alpha ',
      isFirst: true,
      isLast: false,
    });
    const later = cloneGenerationChunkPiece(chunk, {
      text: 'beta ',
      isFirst: false,
      isLast: false,
    });

    const firstMessage = first.message as AIMessageChunk;
    const laterMessage = later.message as AIMessageChunk;
    expect(firstMessage.additional_kwargs).toEqual({ annotation: 'once' });
    expect(firstMessage.response_metadata).toEqual({
      model_name: 'test-model',
    });
    expect(firstMessage.usage_metadata).toBeDefined();
    expect(laterMessage.additional_kwargs).toEqual({});
    expect(laterMessage.response_metadata).toEqual({});
    expect(laterMessage.usage_metadata).toBeUndefined();
  });

  it('keeps generationInfo on the first piece only', () => {
    const infoChunk = new ChatGenerationChunk({
      text: 'alpha beta gamma',
      generationInfo: {
        usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
      },
      message: new AIMessageChunk({ content: 'alpha beta gamma' }),
    });
    const first = cloneGenerationChunkPiece(infoChunk, {
      text: 'alpha ',
      isFirst: true,
      isLast: false,
    });
    const later = cloneGenerationChunkPiece(infoChunk, {
      text: 'beta ',
      isFirst: false,
      isLast: false,
    });
    expect(first.generationInfo).toBeDefined();
    expect(later.generationInfo).toBeUndefined();
  });

  it('returns the original chunk for unsplit pieces', () => {
    expect(
      cloneGenerationChunkPiece(chunk, {
        text: 'alpha beta gamma',
        isFirst: true,
        isLast: true,
      })
    ).toBe(chunk);
  });
});

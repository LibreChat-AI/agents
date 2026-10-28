import type { ChatOpenAIReasoningSummary } from '@langchain/openai';
import type { AIMessageChunk } from '@langchain/core/messages';
import { getChunkContent } from './stream';
import { Providers } from '@/common';

describe('getChunkContent', () => {
  it('should handle reasoning content for OpenAI/Azure providers', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: 'Regular content',
      additional_kwargs: {
        reasoning: {
          summary: [{ text: 'Reasoning summary text' }],
        } as Partial<ChatOpenAIReasoningSummary>,
      },
    };

    const result = getChunkContent({
      chunk,
      provider: Providers.OPENAI,
      reasoningKey: 'reasoning',
    });

    expect(result).toBe('Reasoning summary text');
  });

  it('should fallback to reasoningKey when no visible content is present', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: '',
      additional_kwargs: {
        reasoning_content: 'Reasoning from key',
      },
    };

    const result = getChunkContent({
      chunk,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe('Reasoning from key');
  });

  it('should use OpenRouter reasoning_content when no visible content is present', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: '',
      additional_kwargs: {
        reasoning_content: 'OpenRouter reasoning delta',
      },
    };

    const result = getChunkContent({
      chunk,
      provider: Providers.OPENROUTER,
      reasoningKey: 'reasoning',
    });

    expect(result).toBe('OpenRouter reasoning delta');
  });

  it('should prefer visible content when hidden reasoning is present on the same chunk', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: 'Regular content',
      additional_kwargs: {
        reasoning_content: 'Reasoning from key',
      },
    };

    const result = getChunkContent({
      chunk,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe('Regular content');
  });

  it('preserves structured Bedrock reasoning when additional kwargs duplicate its text', () => {
    const content = [
      {
        type: 'reasoning_content',
        reasoningText: { text: 'Reasoning summary text' },
      },
    ];
    const chunk: Partial<AIMessageChunk> = {
      content,
      additional_kwargs: {
        reasoning_content: 'Reasoning summary text',
      },
    };

    const result = getChunkContent({
      chunk,
      provider: Providers.BEDROCK,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe(content);
  });

  it('uses Bedrock reasoning kwargs when structured content is empty', () => {
    const result = getChunkContent({
      chunk: {
        content: [],
        additional_kwargs: {
          reasoning_content: 'Reasoning summary text',
        },
      },
      provider: Providers.BEDROCK,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe('Reasoning summary text');
  });

  it('should prefer visible OpenRouter content over reasoning_content on the same chunk', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: 'Regular content',
      additional_kwargs: {
        reasoning_content: 'OpenRouter reasoning delta',
      },
    };

    const result = getChunkContent({
      chunk,
      provider: Providers.OPENROUTER,
      reasoningKey: 'reasoning',
    });

    expect(result).toBe('Regular content');
  });

  it('should fallback to chunk.content when reasoningKey value is null or undefined', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: 'Fallback content',
      additional_kwargs: {
        reasoning_content: null,
      },
    };

    const result = getChunkContent({
      chunk,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe('Fallback content');
  });

  it('should fallback to chunk.content when reasoningKey value is empty string', () => {
    const chunk: Partial<AIMessageChunk> = {
      content: ' can',
      additional_kwargs: {
        reasoning_content: '',
      },
    };

    const result = getChunkContent({
      chunk,
      reasoningKey: 'reasoning_content',
    });

    expect(result).toBe(' can');
  });

  it('should return undefined when no content is available', () => {
    const chunk: Partial<AIMessageChunk> = {
      additional_kwargs: {},
    };

    const result = getChunkContent({
      chunk,
      reasoningKey: 'reasoning',
    });

    expect(result).toBeUndefined();
  });

  it('should handle missing chunk gracefully', () => {
    const result = getChunkContent({
      reasoningKey: 'reasoning',
    });

    expect(result).toBeUndefined();
  });
});

import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';

import {
  getCacheWriteTokens,
  attachCacheWriteUsage,
  attachCacheWriteMetadata,
} from './index';

/**
 * Regression coverage for a crash reported against OpenAI-compatible
 * third-party servers (e.g. mlx_vlm.server) whose `/v1/responses` usage
 * payload omits `input_tokens_details` entirely — a shape the OpenAI API
 * itself always populates, but which `ResponsesUsageWithCacheWrite`
 * declares optional. Reading `.cache_write_tokens` off that missing object
 * without a second `?.` threw "Cannot read properties of undefined
 * (reading 'cache_write_tokens')" on every completion from such a server.
 */
describe('cache write token extraction (Responses API)', () => {
  describe('getCacheWriteTokens', () => {
    it('returns undefined without throwing when usage has no input_tokens_details', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            // input_tokens_details intentionally omitted, mirroring a
            // minimal OpenAI-compatible server's usage payload.
          },
        },
      });

      expect(() => getCacheWriteTokens(message)).not.toThrow();
      expect(getCacheWriteTokens(message)).toBeUndefined();
    });

    it('returns undefined without throwing when there is no usage at all', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {},
      });

      expect(() => getCacheWriteTokens(message)).not.toThrow();
      expect(getCacheWriteTokens(message)).toBeUndefined();
    });

    it('still reports cache_write_tokens when the field is present', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            input_tokens_details: { cache_write_tokens: 5 },
          },
        },
      });

      expect(getCacheWriteTokens(message)).toBe(5);
    });

    it('falls back to the serialized metadata key when usage is absent', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          metadata: { __librechat_cache_write_tokens: '7' },
        },
      });

      expect(getCacheWriteTokens(message)).toBe(7);
    });
  });

  it.each([0, 5])(
    'prefers raw aggregate %p over serialized metadata',
    (count) => {
      const message = new AIMessage({
        content: '',
        response_metadata: {
          usage: { input_tokens_details: { cache_write_tokens: count } },
          metadata: { __librechat_cache_write_tokens: '99' },
        },
      });
      expect(getCacheWriteTokens(message)).toBe(count);
    }
  );

  describe('attachCacheWriteUsage', () => {
    it('leaves usage_metadata untouched without throwing when input_tokens_details is missing', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
        },
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
        },
      });

      expect(() => attachCacheWriteUsage(message)).not.toThrow();
      expect(message.usage_metadata?.input_token_details).toBeUndefined();
    });
  });

  describe('attachCacheWriteMetadata', () => {
    it('returns the response unmodified without throwing when input_tokens_details is missing', () => {
      const response = {
        usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
      } as OpenAIClient.Responses.Response;

      expect(() => attachCacheWriteMetadata(response)).not.toThrow();
      expect(attachCacheWriteMetadata(response)).toBe(response);
    });

    it('returns the response unmodified without throwing when usage is missing entirely', () => {
      const response = {} as OpenAIClient.Responses.Response;

      expect(() => attachCacheWriteMetadata(response)).not.toThrow();
    });
  });
});

describe('Responses cache write TTL details', () => {
  const rawUsage = {
    input_tokens: 100,
    output_tokens: 10,
    total_tokens: 110,
    output_tokens_details: { reasoning_tokens: 2 },
    input_tokens_details: {
      cached_tokens: 20,
      cache_write_tokens: 60,
      cache_write_tokens_details: { ttl_30m_tokens: 15, ttl_12h_tokens: 45 },
    },
  };
  const expectedDetails = {
    cache_read: 20,
    cache_creation: 60,
    cache_creation_30m: 15,
    cache_creation_12h: 45,
  };

  it.each([AIMessage, AIMessageChunk])(
    'normalizes TTL subsets on %p without adding them to aggregate usage',
    (Message) => {
      const message = new Message({
        content: 'hi',
        response_metadata: { usage: rawUsage },
        usage_metadata: {
          input_tokens: 100,
          output_tokens: 10,
          total_tokens: 110,
          input_token_details: { cache_read: 20 },
        },
      });
      attachCacheWriteUsage(message);
      attachCacheWriteUsage(message);
      expect(message.usage_metadata).toEqual({
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        input_token_details: expectedDetails,
      });
      expect(message.response_metadata.usage).toBe(rawUsage);
    }
  );

  it('bridges only counts and removes only internal keys without mutating user metadata', () => {
    const metadata = { user: 'kept', nested: { user: 'also kept' } };
    const response = Object.assign({} as OpenAIClient.Responses.Response, {
      usage: rawUsage,
      metadata,
      private: 'not transported',
    });
    const bridged = attachCacheWriteMetadata(response);
    expect(bridged.metadata).toEqual({
      ...metadata,
      __librechat_cache_write_tokens: '60',
      __librechat_cache_write_tokens_30m: '15',
      __librechat_cache_write_tokens_12h: '45',
    });
    expect(response.metadata).toBe(metadata);
    const message = new AIMessage({
      content: 'hi',
      response_metadata: { metadata: bridged.metadata, unrelated: 'kept' },
      usage_metadata: {
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        input_token_details: { cache_read: 20 },
      },
    });
    attachCacheWriteUsage(message);
    expect(message.usage_metadata?.input_token_details).toEqual(
      expectedDetails
    );
    expect(message.response_metadata).toEqual({ metadata, unrelated: 'kept' });
    expect(bridged.metadata).toHaveProperty(
      '__librechat_cache_write_tokens_12h'
    );
    expect(JSON.stringify(message.response_metadata)).not.toContain(
      'not transported'
    );
  });

  it.each([undefined, 0])(
    'preserves a TTL count with aggregate %p',
    (aggregate) => {
      const message = new AIMessage({
        content: '',
        response_metadata: {
          usage: {
            input_tokens_details: {
              cache_write_tokens: aggregate,
              cache_write_tokens_details: { ttl_12h_tokens: 0 },
            },
          },
        },
        usage_metadata: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
      });
      attachCacheWriteUsage(message);
      expect(message.usage_metadata?.input_token_details).toEqual({
        ...(aggregate == null ? {} : { cache_creation: aggregate }),
        cache_creation_12h: 0,
      });
    }
  );

  it.each([-1, Infinity, NaN, '45'])(
    'ignores invalid raw TTL count %p',
    (count) => {
      const message = new AIMessage({
        content: '',
        response_metadata: {
          usage: {
            input_tokens_details: {
              cache_write_tokens_details: { ttl_12h_tokens: count },
            },
          },
        },
        usage_metadata: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
      });
      attachCacheWriteUsage(message);
      expect(message.usage_metadata?.input_token_details).toBeUndefined();
    }
  );

  it('keeps aggregate-only cache creation unchanged when TTL details are absent', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: {
        usage: { input_tokens_details: { cache_write_tokens: 7 } },
      },
      usage_metadata: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
    });
    attachCacheWriteUsage(message);
    expect(message.usage_metadata).toEqual({
      input_tokens: 10,
      output_tokens: 2,
      total_tokens: 12,
      input_token_details: { cache_creation: 7 },
    });
  });

  it('scrubs invalid internal counts even without normalized usage', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: {
        metadata: {
          __librechat_cache_write_tokens_12h: 'invalid',
          user: 'kept',
        },
      },
    });
    attachCacheWriteUsage(message);
    expect(message.usage_metadata).toBeUndefined();
    expect(message.response_metadata.metadata).toEqual({ user: 'kept' });
  });
});

import { FakeChatModel } from '@langchain/core/utils/testing';
import type * as t from '@/types';
import { getConfiguredCompletionTokens } from '@/llm/request';
import { registerProvider } from '@/llm/providers';
import { Providers } from '@/common';

const options = (fields: Record<string, unknown>): t.ClientOptions =>
  fields as t.ClientOptions;

describe('getConfiguredCompletionTokens', () => {
  it('returns an explicit allowance under either key', () => {
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({ model: 'claude-sonnet-5-5', maxTokens: 2048 })
      )
    ).toBe(2048);
    expect(
      getConfiguredCompletionTokens(
        Providers.GOOGLE,
        options({ model: 'gemini-3-pro', maxOutputTokens: 8192 })
      )
    ).toBe(8192);
  });

  it.each(['claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-opus-4-8'])(
    'reserves the Anthropic default for %s when no limit is set',
    (model) => {
      expect(
        getConfiguredCompletionTokens(Providers.ANTHROPIC, options({ model }))
      ).toBe(16384);
    }
  );

  it('reserves the max_tokens sent through invocationKwargs', () => {
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({
          model: 'claude-sonnet-5-5',
          invocationKwargs: { max_tokens: 2048 },
        })
      )
    ).toBe(2048);
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({
          model: 'claude-sonnet-5-5',
          maxTokens: 8192,
          invocationKwargs: { max_tokens: 2048 },
        })
      )
    ).toBe(2048);
  });

  it('reads the model through the modelName alias', () => {
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({ modelName: 'claude-sonnet-5-5' })
      )
    ).toBe(16384);
  });

  it('reserves nothing it cannot attribute to a default', () => {
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({ model: 'claude-3-haiku-20240307' })
      )
    ).toBeUndefined();
    expect(
      getConfiguredCompletionTokens(
        Providers.OPENAI,
        options({ model: 'claude-sonnet-5-5' })
      )
    ).toBeUndefined();
    expect(
      getConfiguredCompletionTokens(Providers.ANTHROPIC, undefined)
    ).toBeUndefined();
  });

  it('reserves nothing for a host provider in the Anthropic family', () => {
    const provider = 'anthropic-family-completion-test';
    const unregister = registerProvider({
      provider,
      model: FakeChatModel,
      family: 'anthropic',
    });
    try {
      expect(
        getConfiguredCompletionTokens(
          provider,
          options({ model: 'claude-sonnet-5-5' })
        )
      ).toBeUndefined();
      expect(
        getConfiguredCompletionTokens(
          provider,
          options({ model: 'claude-sonnet-5-5', maxTokens: 4096 })
        )
      ).toBe(4096);
    } finally {
      unregister();
    }
  });

  it('does not substitute the default for an explicit non-positive value', () => {
    expect(
      getConfiguredCompletionTokens(
        Providers.ANTHROPIC,
        options({ model: 'claude-sonnet-5-5', maxTokens: -1 })
      )
    ).toBeUndefined();
  });
});

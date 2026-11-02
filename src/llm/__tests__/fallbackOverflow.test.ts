import { AIMessageChunk } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import {
  tryFallbackProviders,
  getFallbackErrorContext,
  getFallbackOverflowCandidates,
} from '@/llm/invoke';
import { OVERFLOW_SIGNATURES } from '@/utils/__tests__/fixtures/contextOverflowSignatures';
import { prepareProviderRequest } from '@/llm/prepareProviderRequest';
import { Providers } from '@/common';
import * as init from '@/llm/init';
import { RequestBodyBudgetError, RequestBodyLimitError } from '@/llm/requestBodyBudget';

function signatureError(model: string): Error {
  const signature = OVERFLOW_SIGNATURES.find((s) => s.model === model);
  if (signature == null) {
    throw new Error(`missing fixture for ${model}`);
  }
  const error = new Error(String(signature.error.message));
  return Object.assign(error, signature.error);
}

/**
 * `tryFallbackProviders` builds its own clients through `initializeModel`, so
 * the only seam for driving per-fallback outcomes is that factory.
 */
function stubModels(
  outcomes: Array<Error | 'ok'>
): jest.SpiedFunction<typeof init.initializeModel> {
  let call = 0;
  return jest.spyOn(init, 'initializeModel').mockImplementation(() => {
    const outcome = outcomes[call++];
    return {
      invoke: async (): Promise<AIMessageChunk> => {
        if (outcome instanceof Error) {
          throw outcome;
        }
        return new AIMessageChunk({ content: 'ok' });
      },
    } as unknown as ReturnType<typeof init.initializeModel>;
  });
}

const messages: BaseMessage[] = [];
const fallbacks: Array<{ provider: Providers }> = [
  { provider: Providers.ANTHROPIC },
  { provider: Providers.OPENAI },
];

describe('tryFallbackProviders surfacing', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('does not dispatch fallbacks for a primary local body rejection', async () => {
    const initialize = stubModels(['ok']);
    const error = new RequestBodyLimitError(140_577, 131_072);
    await expect(tryFallbackProviders({ fallbacks, messages, primaryError: error })).rejects.toBe(error);
    expect(initialize).not.toHaveBeenCalled();
  });

  it.each([
    new RequestBodyLimitError(140_577, 131_072),
    new RequestBodyBudgetError('unsupported body'),
  ])('stops the fallback chain on a local body failure', async (error) => {
    const initialize = stubModels([error, 'ok']);
    await expect(tryFallbackProviders({ fallbacks, messages, primaryError: new Error('503 primary') })).rejects.toBe(error);
    expect(initialize).toHaveBeenCalledTimes(1);
  });

  it('sends the exact prepared fallback request without projecting again', async () => {
    const invocations: BaseMessage[][] = [];
    const fallbackModel = {
      invoke: async (preparedMessages: BaseMessage[]) => {
        invocations.push(preparedMessages);
        return new AIMessageChunk({ content: 'ok' });
      },
    } as unknown as ReturnType<typeof init.initializeModel>;
    jest.spyOn(init, 'initializeModel').mockReturnValue(fallbackModel);
    let preparedMessages: BaseMessage[] | undefined;

    await tryFallbackProviders({
      fallbacks: [{ provider: Providers.MISTRAL }],
      messages,
      primaryError: new Error('primary boom'),
      prepareProviderRequest: (input) => {
        const request = prepareProviderRequest(input);
        preparedMessages = request.messages;
        return request;
      },
    });

    expect(invocations).toHaveLength(1);
    expect(invocations[0]).toBe(preparedMessages);
  });

  it('throws the overflow even when a later fallback failed for another reason', async () => {
    const overflow = signatureError('claude-haiku-4-5-20251001');
    const unrelated = new Error('503 upstream unavailable');
    stubModels([overflow, unrelated]);

    await expect(
      tryFallbackProviders({
        fallbacks,
        messages,
        primaryError: new Error('primary boom'),
      })
    ).rejects.toThrow(/prompt is too long/i);
  });

  it('keeps last-error-wins when no fallback overflowed', async () => {
    stubModels([
      new Error('first failed'),
      new Error('503 upstream unavailable'),
    ]);

    await expect(
      tryFallbackProviders({
        fallbacks,
        messages,
        primaryError: new Error('primary boom'),
      })
    ).rejects.toThrow(/upstream unavailable/);
  });

  it('surfaces a primary overflow when every fallback also fails', async () => {
    stubModels([new Error('first failed'), new Error('second failed')]);

    await expect(
      tryFallbackProviders({
        fallbacks,
        messages,
        primaryError: signatureError('us.amazon.nova-lite-v1:0'),
      })
    ).rejects.toThrow(/Input Tokens Exceeded/i);
  });

  it('recognises a reasonless Vertex overflow when given corroboration', async () => {
    const vertex = signatureError('gemini-2.5-flash-lite');
    stubModels([vertex, new Error('503 upstream unavailable')]);

    await expect(
      tryFallbackProviders({
        fallbacks: [
          { provider: Providers.VERTEXAI, maxContextTokens: 200_000 },
          { provider: Providers.OPENAI },
        ],
        messages,
        primaryError: new Error('primary boom'),
        overflowContext: {
          provider: Providers.VERTEXAI,
          estimatedPromptTokens: 190_000,
        },
      })
    ).rejects.toThrow(/Google request failed/);
  });

  it('uses the fallback context window to corroborate a Vertex overflow', async () => {
    const vertex = signatureError('gemini-2.5-flash-lite');
    stubModels([vertex, new Error('503 upstream unavailable')]);

    const error = await tryFallbackProviders({
      fallbacks: [
        { provider: Providers.VERTEXAI, maxContextTokens: 32_000 },
        { provider: Providers.OPENAI },
      ],
      messages,
      primaryError: new Error('primary boom'),
      overflowContext: {
        provider: Providers.VERTEXAI,
        estimatedPromptTokens: 50_000,
        maxContextTokens: 200_000,
      },
    }).catch((fallbackError: Error) => fallbackError);

    expect(error).toBe(vertex);
    expect(getFallbackErrorContext(error)).toEqual({
      provider: Providers.VERTEXAI,
      clientOptions: undefined,
      maxContextTokens: 32_000,
    });
  });

  it('does not corroborate Vertex against the primary context window', async () => {
    const vertex = signatureError('gemini-2.5-flash-lite');
    stubModels([vertex, new Error('503 upstream unavailable')]);

    await expect(
      tryFallbackProviders({
        fallbacks: [
          { provider: Providers.VERTEXAI },
          { provider: Providers.OPENAI },
        ],
        messages,
        primaryError: new Error('primary boom'),
        overflowContext: {
          provider: Providers.OPENAI,
          estimatedPromptTokens: 190_000,
          maxContextTokens: 200_000,
        },
      })
    ).rejects.toThrow(/upstream unavailable/);
  });

  it('prefers a fallback overflow over the primary one', async () => {
    /**
     * Reaching this function with an overflowing primary means the caller
     * already failed to recover from it, so the fallback's — which sits
     * against a different window — is the more useful of the two.
     */
    const primary = signatureError('gpt-5-nano');
    const fallbackOverflow = signatureError('claude-haiku-4-5-20251001');
    stubModels([fallbackOverflow, new Error('503 upstream unavailable')]);

    await expect(
      tryFallbackProviders({ fallbacks, messages, primaryError: primary })
    ).rejects.toThrow(/prompt is too long/i);
  });

  it('attributes a fallback overflow to the client that produced it', async () => {
    const fallbackOverflow = signatureError('claude-haiku-4-5-20251001');
    stubModels([fallbackOverflow, new Error('503 upstream unavailable')]);

    const thrown = await tryFallbackProviders({
      fallbacks,
      messages,
      primaryError: new Error('primary boom'),
    }).catch((error: unknown) => error);

    const attribution = getFallbackErrorContext(thrown);
    expect(attribution?.provider).toBe(Providers.ANTHROPIC);
  });

  it('retains every fallback overflow candidate', async () => {
    const first = new Error(
      'This model\'s maximum context length is 4096 tokens. However, your messages resulted in 6000 tokens.'
    );
    const second = signatureError('claude-haiku-4-5-20251001');
    stubModels([first, second]);

    const thrown = await tryFallbackProviders({
      fallbacks,
      messages,
      primaryError: new Error('primary boom'),
    }).catch((error: unknown) => error);

    expect(getFallbackOverflowCandidates(thrown)).toEqual([
      {
        error: first,
        context: {
          provider: Providers.ANTHROPIC,
          clientOptions: undefined,
          maxContextTokens: undefined,
        },
      },
      {
        error: second,
        context: {
          provider: Providers.OPENAI,
          clientOptions: undefined,
          maxContextTokens: undefined,
        },
      },
    ]);
  });

  it('declines ambiguous pressure from another provider token space', async () => {
    const vertex = signatureError('gemini-2.5-flash-lite');
    stubModels([vertex, new Error('503 upstream unavailable')]);

    await expect(
      tryFallbackProviders({
        fallbacks: [
          { provider: Providers.VERTEXAI, maxContextTokens: 32_000 },
          { provider: Providers.OPENAI },
        ],
        messages,
        primaryError: new Error('primary boom'),
        overflowContext: {
          provider: Providers.OPENAI,
          estimatedPromptTokens: 50_000,
          maxContextTokens: 200_000,
        },
      })
    ).rejects.toThrow(/upstream unavailable/);
  });

  it('continues past a frozen fallback error', async () => {
    const fallbackOverflow = Object.freeze(
      signatureError('claude-haiku-4-5-20251001')
    );
    stubModels([fallbackOverflow, 'ok']);

    const result = await tryFallbackProviders({
      fallbacks,
      messages,
      primaryError: new Error('primary boom'),
    });

    expect((result?.messages?.[0] as AIMessageChunk | undefined)?.content).toBe(
      'ok'
    );
  });

  it('leaves a primary overflow unattributed', async () => {
    stubModels([new Error('first failed'), new Error('second failed')]);

    const thrown = await tryFallbackProviders({
      fallbacks,
      messages,
      primaryError: signatureError('us.amazon.nova-lite-v1:0'),
    }).catch((error: unknown) => error);

    expect(getFallbackErrorContext(thrown)).toBeUndefined();
  });

  it('returns the first success without consulting later fallbacks', async () => {
    stubModels([signatureError('claude-haiku-4-5-20251001'), 'ok']);

    const result = await tryFallbackProviders({
      fallbacks,
      messages,
      primaryError: new Error('primary boom'),
    });

    expect((result?.messages?.[0] as AIMessageChunk | undefined)?.content).toBe(
      'ok'
    );
  });
});

import { AIMessage } from '@langchain/core/messages';
import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { SummarizationAttemptUsage } from '@/types/summarize';
import {
  convertConverseMessageToLangChainMessage,
  handleConverseStreamMetadata,
} from '@/llm/bedrock/utils/message_outputs';
import {
  buildSummarizationStepUsage,
  readSummarizationUsage,
} from '@/summarization/usage';

const PRIVATE_CONTENT = 'PRIVATE_REASONING_AND_PROVIDER_DATA';

describe('readSummarizationUsage', () => {
  it('preserves reported cache and reasoning details without message or raw metadata', () => {
    const usage: UsageMetadata = {
      input_tokens: 120,
      output_tokens: 30,
      total_tokens: 150,
      input_token_details: { cache_read: 80, cache_creation: 20 },
      output_token_details: { reasoning: 25, text: 5 },
    };
    const response = new AIMessage({
      content: PRIVATE_CONTENT,
      additional_kwargs: { reasoning_content: PRIVATE_CONTENT },
      usage_metadata: usage,
      response_metadata: {
        private: PRIVATE_CONTENT,
        metadata: { usage: { inputTokens: 999, outputTokens: 999 } },
      },
    });

    const result = readSummarizationUsage(response);

    expect(result).toEqual(usage);
    expect(JSON.stringify(result)).not.toContain(PRIVATE_CONTENT);
  });

  it('reads Bedrock fallback counts and cache details without leaking raw fields', () => {
    const response = new AIMessage({
      content: PRIVATE_CONTENT,
      response_metadata: {
        metadata: {
          usage: {
            inputTokens: 10,
            outputTokens: 5,
            totalTokens: 35,
            cacheReadInputTokens: 12,
            cacheWriteInputTokens: 8,
            private: PRIVATE_CONTENT,
          },
        },
      },
    });

    expect(readSummarizationUsage(response)).toEqual({
      input_tokens: 30,
      output_tokens: 5,
      total_tokens: 35,
      input_token_details: { cache_read: 12, cache_creation: 8 },
    });
  });

  it('reads native Bedrock usage directly from response_metadata', () => {
    const response = new AIMessage({
      content: PRIVATE_CONTENT,
      response_metadata: {
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          totalTokens: 35,
          cacheReadInputTokens: 12,
          cacheWriteInputTokens: 8,
          private: PRIVATE_CONTENT,
        },
      },
    });

    expect(readSummarizationUsage(response)).toEqual({
      input_tokens: 30,
      output_tokens: 5,
      total_tokens: 35,
      input_token_details: { cache_read: 12, cache_creation: 8 },
    });
  });

  it('handles responses without response_metadata', () => {
    const response = { content: 'summary' } as BaseMessage;

    expect(readSummarizationUsage(response)).toBeUndefined();
  });

  it('preserves reported zero without inventing missing counts or cache fields', () => {
    const response = new AIMessage({
      content: '',
      response_metadata: {
        metadata: { usage: { inputTokens: 0, cacheReadInputTokens: 0 } },
      },
    });

    expect(readSummarizationUsage(response)).toEqual({
      input_tokens: 0,
      input_token_details: { cache_read: 0 },
    });
  });

  it.each([
    undefined,
    new AIMessage(PRIVATE_CONTENT),
    new AIMessage({
      content: '',
      response_metadata: { metadata: { usage: {} } },
    }),
    new AIMessage({
      content: '',
      response_metadata: {
        metadata: {
          usage: {
            inputTokens: NaN,
            outputTokens: Infinity,
            totalTokens: -1,
            cacheReadInputTokens: '12',
          },
        },
      },
    }),
  ])('does not fabricate an entry when no usable usage is reported: %#', (response) => {
    const usage = readSummarizationUsage(response);
    const attempts = usage == null ? [] : [{ provider: 'openAI', usage }];

    expect(usage).toBeUndefined();
    expect(buildSummarizationStepUsage(attempts)).toBeUndefined();
  });
});

describe('buildSummarizationStepUsage', () => {
  it('keeps the exact legacy single-attempt shape and prefers reported total_tokens', () => {
    expect(buildSummarizationStepUsage([{
      provider: 'bedrock',
      model: 'summary-model',
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 35 },
    }])).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 35,
    });
  });

  it('falls back to input plus output only when total_tokens is missing', () => {
    expect(buildSummarizationStepUsage([{
      provider: 'openAI',
      usage: { input_tokens: 10, output_tokens: 5 },
    }])).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
  });

  it('honors explicitly reported zero total_tokens', () => {
    expect(buildSummarizationStepUsage([{
      provider: 'openAI',
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 0 },
    }])).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 0,
    });
  });

  it('keeps missing per-attempt counts unknown while defaulting aggregate fields to zero', () => {
    const attempts: readonly SummarizationAttemptUsage[] = [
      { provider: 'openAI', usage: { output_tokens: 5 } },
      { provider: 'bedrock', usage: { total_tokens: 12 } },
    ];

    expect(buildSummarizationStepUsage(attempts)).toEqual({
      prompt_tokens: 0,
      completion_tokens: 5,
      total_tokens: 17,
      attemptUsage: attempts,
    });
  });

  it('ignores non-finite and negative counts in aggregate fields', () => {
    expect(buildSummarizationStepUsage([{
      provider: 'openAI',
      usage: { input_tokens: NaN, output_tokens: 5, total_tokens: Infinity },
    }, {
      provider: 'bedrock',
      usage: { input_tokens: -1, output_tokens: 2, total_tokens: -1 },
    }])).toMatchObject({
      prompt_tokens: 0,
      completion_tokens: 7,
      total_tokens: 7,
    });
  });

  it('retains distinct calls for the same model across heterogeneous providers', () => {
    const attempts: readonly SummarizationAttemptUsage[] = [
      {
        provider: 'openAI',
        model: 'shared-model',
        usage: {
          input_tokens: 100,
          output_tokens: 20,
          total_tokens: 120,
          output_token_details: { reasoning: 20 },
        },
      },
      {
        provider: 'bedrock',
        model: 'shared-model',
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 35,
          input_token_details: { cache_read: 12, cache_creation: 8 },
        },
      },
      {
        provider: 'anthropic',
        model: 'successful-model',
        usage: { input_tokens: 30, output_tokens: 7 },
      },
    ];

    expect(buildSummarizationStepUsage(attempts)).toEqual({
      prompt_tokens: 140,
      completion_tokens: 32,
      total_tokens: 192,
      attemptUsage: attempts,
    });
  });

  it('snapshots attempts and token details without mutating or retaining the input ledger', () => {
    const inputDetails = { cache_read: 12, cache_creation: 8 };
    const outputDetails = { reasoning: 4 };
    const usage: UsageMetadata = {
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 35,
      input_token_details: inputDetails,
      output_token_details: outputDetails,
    };
    const attempts: SummarizationAttemptUsage[] = [
      { provider: 'bedrock', model: 'first-model', usage },
      { provider: 'anthropic', usage: { output_tokens: 7 } },
    ];

    const result = buildSummarizationStepUsage(attempts);

    expect(attempts).toHaveLength(2);
    expect(usage.input_token_details).toBe(inputDetails);
    usage.input_tokens = 99;
    inputDetails.cache_read = 99;
    outputDetails.reasoning = 99;
    attempts[0].model = 'changed-model';
    attempts.push({ provider: 'openAI', usage: { output_tokens: 99 } });
    expect(result).toEqual({
      prompt_tokens: 10,
      completion_tokens: 12,
      total_tokens: 42,
      attemptUsage: [
        {
          provider: 'bedrock',
          model: 'first-model',
          usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 35,
            input_token_details: { cache_read: 12, cache_creation: 8 },
            output_token_details: { reasoning: 4 },
          },
        },
        { provider: 'anthropic', usage: { output_tokens: 7 } },
      ],
    });
  });
});

describe('Bedrock summarization cache accounting', () => {
  const raw = {
    inputTokens: 10,
    outputTokens: 5,
    totalTokens: 35,
    cacheReadInputTokens: 12,
    cacheWriteInputTokens: 8,
  };
  const reported: UsageMetadata = {
    input_tokens: 10,
    output_tokens: 5,
    total_tokens: 35,
    input_token_details: { cache_read: 12, cache_creation: 8 },
    output_token_details: { reasoning: 4, text: 1 },
  };
  const normalized = {
    ...reported,
    input_tokens: 30,
  };

  const responses = [
    ['native invoke', convertConverseMessageToLangChainMessage(
      { role: 'assistant', content: [{ text: PRIVATE_CONTENT }] },
      { usage: raw, stopReason: 'end_turn', metrics: { latencyMs: 1 } }
    )],
    ['native stream', handleConverseStreamMetadata(
      { usage: raw, metrics: { latencyMs: 1 } }, { streamUsage: true }
    ).message],
    ['direct metadata', new AIMessage({
      content: PRIVATE_CONTENT,
      response_metadata: { usage: raw },
    })],
    ['nested metadata', new AIMessage({
      content: PRIVATE_CONTENT,
      response_metadata: { metadata: { usage: raw } },
    })],
  ] as const;

  it.each(responses)('includes cache input from %s in a mixed-provider ledger', (_name, response) => {
    const usage = readSummarizationUsage(response);
    expect(usage).toEqual({
      input_tokens: 30,
      output_tokens: 5,
      total_tokens: 35,
      input_token_details: { cache_read: 12, cache_creation: 8 },
    });
    const attempts: SummarizationAttemptUsage[] = [
      { provider: 'bedrock', model: 'cached-model', usage: usage! },
      { provider: 'openAI', model: 'shared-model', usage: normalized },
      { provider: 'anthropic', model: 'shared-model', usage: normalized },
    ];
    expect(buildSummarizationStepUsage(attempts)).toEqual({
      prompt_tokens: 90,
      completion_tokens: 15,
      total_tokens: 105,
      attemptUsage: attempts,
    });
    expect(JSON.stringify(usage)).not.toContain(PRIVATE_CONTENT);
    if ('usage_metadata' in response && response.usage_metadata != null) {
      expect((response as AIMessage).usage_metadata?.input_tokens).toBe(10);
    }
  });

  it.each(['direct', 'nested'] as const)('does not double-count already-normalized input with %s metadata', (location) => {
    const response = new AIMessage({
      content: '',
      usage_metadata: normalized,
      response_metadata: location === 'direct'
        ? { usage: raw }
        : { metadata: { usage: raw } },
    });
    const usage = readSummarizationUsage(response);
    expect(usage).toBe(response.usage_metadata);
    expect(usage).toEqual(normalized);
    expect(buildSummarizationStepUsage([{ provider: 'bedrock', usage: usage! }])).toEqual({
      prompt_tokens: 30, completion_tokens: 5, total_tokens: 35,
    });
  });

  it.each(['direct', 'nested'] as const)('retains reported totals and details with %s metadata without mutating the response', (location) => {
    const response = new AIMessage({
      content: '',
      usage_metadata: reported,
      response_metadata: location === 'direct'
        ? { usage: raw }
        : { metadata: { usage: raw } },
    });
    const usage = readSummarizationUsage(response);
    expect(usage).toEqual(normalized);
    expect(response.usage_metadata).toEqual(reported);
    expect(response.response_metadata).toEqual(location === 'direct'
      ? { usage: raw }
      : { metadata: { usage: raw } });
  });

  it.each(['bedrock', 'openAI', 'anthropic'] as const)('does not infer normalization for %s without raw metadata', (provider) => {
    const response = new AIMessage({ content: '', usage_metadata: normalized });
    const usage = readSummarizationUsage(response);
    expect(buildSummarizationStepUsage([{ provider, usage: usage! }])).toEqual({
      prompt_tokens: 30, completion_tokens: 5, total_tokens: 35,
    });
  });

  it.each([undefined, 0, 15, 34, NaN])('uses raw metadata, not the reported total (%s), to corroborate normalization', (total) => {
    const usage = readSummarizationUsage(new AIMessage({
      content: '',
      usage_metadata: { ...reported, total_tokens: total } as UsageMetadata,
      response_metadata: { metadata: { usage: raw } },
    }));
    expect(usage).toEqual({ ...normalized, total_tokens: total });
    expect(buildSummarizationStepUsage([{ provider: 'bedrock', usage: usage! }])).toEqual({
      prompt_tokens: 30,
      completion_tokens: 5,
      total_tokens: total == null || Number.isNaN(total) ? 35 : total,
    });
  });

  it.each(['direct', 'nested'] as const)('includes known cache input when raw %s metadata omits totals', (location) => {
    const withoutTotal = { ...raw, totalTokens: undefined };
    const usage = readSummarizationUsage(new AIMessage({
      content: '',
      response_metadata: location === 'direct'
        ? { usage: withoutTotal }
        : { metadata: { usage: withoutTotal } },
    }));
    expect(usage?.total_tokens).toBeUndefined();
    expect(buildSummarizationStepUsage([{ provider: 'bedrock', usage: usage! }])).toEqual({
      prompt_tokens: 30, completion_tokens: 5, total_tokens: 35,
    });
  });

  it.each([undefined, 15, 35])('does not guess cache exclusivity without raw metadata from total %s', (total) => {
    const usage = readSummarizationUsage(new AIMessage({
      content: '', usage_metadata: { ...reported, total_tokens: total } as UsageMetadata,
    }));
    expect(usage?.input_tokens).toBe(10);
  });

  it.each([
    { inputTokens: 11 },
    { inputTokens: NaN },
    { inputTokens: -1 },
    { inputTokens: undefined },
    { inputTokens: 10, cacheReadInputTokens: NaN, cacheWriteInputTokens: -1 },
  ])('does not normalize uncorroborated or invalid raw counts: %#', (overrides) => {
    const response = new AIMessage({
      content: '',
      usage_metadata: reported,
      response_metadata: { usage: { ...raw, ...overrides } },
    });
    expect(readSummarizationUsage(response)).toBe(response.usage_metadata);
  });
});

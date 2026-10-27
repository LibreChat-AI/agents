import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { SummarizationAttemptUsage } from '@/types/summarize';

interface BedrockSummarizationUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
}

function tokenCount(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

export function readSummarizationUsage(
  response: BaseMessage | undefined
): Partial<UsageMetadata> | undefined {
  if (response == null) return undefined;
  if ('usage_metadata' in response && response.usage_metadata != null) {
    return response.usage_metadata as Partial<UsageMetadata>;
  }

  const metadata = response.response_metadata as {
    usage?: BedrockSummarizationUsage;
    metadata?: { usage?: BedrockSummarizationUsage };
  } | undefined;
  const raw = metadata?.metadata?.usage ?? metadata?.usage;
  if (raw == null) return undefined;

  const input_tokens = tokenCount(raw.inputTokens);
  const output_tokens = tokenCount(raw.outputTokens);
  const total_tokens = tokenCount(raw.totalTokens);
  const cache_read = tokenCount(raw.cacheReadInputTokens);
  const cache_creation = tokenCount(raw.cacheWriteInputTokens);
  if (
    input_tokens == null &&
    output_tokens == null &&
    total_tokens == null &&
    cache_read == null &&
    cache_creation == null
  ) return undefined;

  return {
    ...(input_tokens == null ? {} : { input_tokens }),
    ...(output_tokens == null ? {} : { output_tokens }),
    ...(total_tokens == null ? {} : { total_tokens }),
    ...(cache_read == null && cache_creation == null
      ? {}
      : {
        input_token_details: {
          ...(cache_read == null ? {} : { cache_read }),
          ...(cache_creation == null ? {} : { cache_creation }),
        },
      }),
  };
}

export function buildSummarizationStepUsage(
  attempts: readonly SummarizationAttemptUsage[]
): {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  attemptUsage?: readonly SummarizationAttemptUsage[];
} | undefined {
  if (attempts.length === 0) return undefined;

  let prompt_tokens = 0;
  let completion_tokens = 0;
  let total_tokens = 0;
  const attemptUsage: SummarizationAttemptUsage[] | undefined =
    attempts.length > 1 ? [] : undefined;
  for (const attempt of attempts) {
    const { usage } = attempt;
    const input = tokenCount(usage.input_tokens) ?? 0;
    const output = tokenCount(usage.output_tokens) ?? 0;
    prompt_tokens += input;
    completion_tokens += output;
    total_tokens += tokenCount(usage.total_tokens) ?? input + output;
    attemptUsage?.push({
      provider: attempt.provider,
      ...(attempt.model == null ? {} : { model: attempt.model }),
      usage: {
        ...usage,
        ...(usage.input_token_details == null
          ? {}
          : { input_token_details: { ...usage.input_token_details } }),
        ...(usage.output_token_details == null
          ? {}
          : { output_token_details: { ...usage.output_token_details } }),
      },
    });
  }

  return {
    prompt_tokens,
    completion_tokens,
    total_tokens,
    ...(attemptUsage == null ? {} : { attemptUsage }),
  };
}

import { SystemMessage } from '@langchain/core/messages';
import { convertToOpenAIFunction } from '@langchain/core/utils/function_calling';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type {
  GraphTools,
  ProviderName,
  ClientOptions,
  TokenCounter,
} from '@/types';
import {
  ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
  DEFAULT_TOOL_TOKEN_MULTIPLIER,
} from '@/common/constants';
import { getProviderFamily } from '@/llm/providerRegistry';
import { resolveClientOptionsModel } from '@/llm/request';
import { toJsonSchema } from '@/utils/schema';
import { Providers } from '@/common/enum';

function toolPayload(tool: GraphTools[number], anthropic: boolean): object {
  if (!('schema' in tool)) {
    return tool;
  }
  const candidate = tool as Partial<
    Pick<StructuredToolInterface, 'name' | 'description' | 'schema' | 'extras'>
  >;
  const extras = candidate.extras;
  const providerToolDefinition = extras?.providerToolDefinition;
  if (
    anthropic &&
    providerToolDefinition != null &&
    typeof providerToolDefinition === 'object'
  ) {
    return providerToolDefinition;
  }
  const schema = candidate.schema;
  if (schema == null || typeof schema !== 'object') {
    return tool;
  }
  const normalized = {
    name: candidate.name ?? '',
    description: candidate.description ?? '',
    schema: toJsonSchema(schema),
  };
  if (anthropic) {
    return {
      name: normalized.name,
      description: normalized.description,
      input_schema: normalized.schema,
      ...extras,
    };
  }
  return { type: 'function', function: convertToOpenAIFunction(normalized) };
}

/** Counts only the resolved bound inventory, using the serving model's counter. */
export function countSummarizationToolTokens({
  tools,
  provider,
  clientOptions,
  count,
}: {
  tools?: GraphTools;
  provider: ProviderName;
  clientOptions?: ClientOptions;
  count: TokenCounter;
}): number {
  if (tools == null || tools.length === 0) {
    return 0;
  }
  const family = getProviderFamily(provider);
  const anthropic = provider === Providers.ANTHROPIC || family === 'anthropic';
  const anthropicMultiplier =
    provider !== Providers.BEDROCK &&
    family !== 'bedrock' &&
    (anthropic ||
      /anthropic|claude/i.test(resolveClientOptionsModel(clientOptions) ?? ''));
  const multiplier = anthropicMultiplier
    ? ANTHROPIC_TOOL_TOKEN_MULTIPLIER
    : DEFAULT_TOOL_TOKEN_MULTIPLIER;
  let tokens = 0;
  for (const tool of tools) {
    tokens += count(
      new SystemMessage(JSON.stringify(toolPayload(tool, anthropic)))
    );
  }
  return Math.ceil(tokens * multiplier);
}

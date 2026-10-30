import type * as t from '@/types';
import { getCurrentClaudeDefaultMaxTokens } from '@/llm/anthropic/defaults';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

/**
 * Returns true when the provider + clientOptions indicate extended thinking
 * is enabled.  Works across Anthropic (direct), Bedrock (additionalModelRequestFields),
 * and OpenAI-compat (modelKwargs.thinking).
 */
export function isThinkingEnabled(
  provider: t.ProviderName,
  clientOptions?: t.ClientOptions
): boolean {
  if (!clientOptions) return false;
  const family = getProviderFamily(provider);

  if (
    (provider === Providers.ANTHROPIC || family === 'anthropic') &&
    (clientOptions as t.AnthropicClientOptions).thinking != null
  ) {
    return true;
  }

  if (
    (provider === Providers.BEDROCK || family === 'bedrock') &&
    (clientOptions as t.BedrockAnthropicInput).additionalModelRequestFields?.[
      'thinking'
    ] != null
  ) {
    return true;
  }

  if (
    (provider === Providers.OPENAI || family === 'openai') &&
    (
      (clientOptions as t.OpenAIClientOptions).modelKwargs
        ?.thinking as t.AnthropicClientOptions['thinking']
    )?.type === 'enabled'
  ) {
    return true;
  }

  return false;
}

/**
 * Model configured on client options, under whichever of the two keys carries
 * it. LangChain accepts `modelName` as an alias for `model` and this package
 * writes both (see `buildSummarizationClientConfig`), while hosts configure
 * agents through either. Reading only one key therefore does not fail loudly:
 * it reports an unconfigured model, and every caller here treats that as a cue
 * to fall back to a default, so a Claude agent configured through the alias is
 * silently handled as something else.
 */
export function resolveClientOptionsModel(
  clientOptions: t.ClientOptions | undefined
): string | undefined {
  const options = clientOptions as
    | { model?: unknown; modelName?: unknown }
    | undefined;
  if (typeof options?.model === 'string' && options.model !== '') {
    return options.model;
  }
  if (typeof options?.modelName === 'string' && options.modelName !== '') {
    return options.modelName;
  }
  return undefined;
}

/**
 * Returns the correct key for setting max output tokens on the model
 * constructor options.  Google/Vertex use `maxOutputTokens`, all others
 * use `maxTokens`.
 */
export function getMaxOutputTokensKey(
  provider: t.ProviderName
): 'maxOutputTokens' | 'maxTokens' {
  return provider === Providers.GOOGLE ||
    provider === Providers.VERTEXAI ||
    getProviderFamily(provider) === 'google'
    ? 'maxOutputTokens'
    : 'maxTokens';
}

/**
 * The completion allowance the provider will reserve, under whichever key its
 * client uses. Providers count it against the same ceiling as the prompt, so
 * overflow recovery has to reserve it when the error did not itemize the
 * total. Without an explicit value, Anthropic clients still send their model
 * default, so that default is reserved instead.
 */
export function getConfiguredCompletionTokens(
  provider: t.ProviderName,
  clientOptions: t.ClientOptions | undefined
): number | undefined {
  const options = clientOptions as
    | { maxTokens?: unknown; maxOutputTokens?: unknown }
    | undefined;
  for (const value of [options?.maxTokens, options?.maxOutputTokens]) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      return value;
    }
  }
  if (
    options?.maxTokens != null ||
    options?.maxOutputTokens != null ||
    (provider !== Providers.ANTHROPIC &&
      getProviderFamily(provider) !== 'anthropic')
  ) {
    return undefined;
  }
  return getCurrentClaudeDefaultMaxTokens(
    resolveClientOptionsModel(clientOptions)
  );
}

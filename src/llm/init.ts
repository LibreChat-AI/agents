import type { Runnable } from '@langchain/core/runnables';
import type * as t from '@/types';
import {
  isOpenAILike,
  isLibreChatOpenAIModel,
  constructorChainHasLcName,
} from '@/utils';
import { getProviderFamily } from '@/llm/providerRegistry';
import { getChatModelClass } from '@/llm/providers';
import { Providers } from '@/common';

type InitializeModelParams<P extends t.ProviderName> = {
  provider: P;
  tools?: t.GraphTools;
  disableToolCalling?: boolean;
} & (
  | {
      override: t.ChatModelInstance;
      clientOptions?: t.ProviderOptionsFor<P>;
    }
  | ([P] extends [keyof t.ProviderOptionsMap]
      ? {
          override?: t.ChatModelInstance;
          clientOptions?: t.ProviderOptionsFor<P>;
        }
      : object extends t.ProviderOptionsFor<P>
        ? {
            override?: t.ChatModelInstance;
            clientOptions?: t.ProviderOptionsFor<P>;
          }
        : {
            override?: undefined;
            clientOptions: t.ProviderOptionsFor<P>;
          })
);

const VERTEX_LC_NAMES: ReadonlySet<string> = new Set(['ChatVertexAI']);

/** Structural stand-in for `instanceof ChatVertexAI`: matches both builds of
 *  `@langchain/google-vertexai`, while this package's own vertex class reports
 *  `LibreChatVertexAI` and keeps failing this guard exactly as it did under
 *  `instanceof`. */
function isLangchainVertexModel(
  model: unknown
): model is import('@langchain/google-vertexai').ChatVertexAI {
  return constructorChainHasLcName(model, VERTEX_LC_NAMES);
}

/**
 * Creates a chat model instance for a given built-in or host-registered
 * provider, applies provider-specific field assignments, and optionally binds
 * tools.
 */
export function initializeModel<P extends t.ProviderName>({
  provider,
  clientOptions,
  tools,
  disableToolCalling,
  override,
}: InitializeModelParams<P>): Runnable {
  const model =
    override ??
    new (getChatModelClass(provider))(
      (clientOptions ?? {}) as t.ProviderOptionsFor<P>
    );

  if (isOpenAILike(provider) && isLibreChatOpenAIModel(model)) {
    const opts = clientOptions as t.OpenAIClientOptions | undefined;
    if (opts) {
      model.temperature = opts.temperature as number;
      model.topP = opts.topP as number;
      model.frequencyPenalty = opts.frequencyPenalty as number;
      model.presencePenalty = opts.presencePenalty as number;
      model.n = opts.n as number;
    }
  } else if (provider === Providers.VERTEXAI && isLangchainVertexModel(model)) {
    const opts = clientOptions as t.VertexAIClientOptions | undefined;
    if (opts) {
      model.temperature = opts.temperature as number;
      model.topP = opts.topP as number;
      model.topK = opts.topK as number;
      model.topLogprobs = opts.topLogprobs as number;
      model.frequencyPenalty = opts.frequencyPenalty as number;
      model.presencePenalty = opts.presencePenalty as number;
      model.maxOutputTokens = opts.maxOutputTokens as number;
    }
  }

  if (!tools || tools.length === 0) {
    return model;
  }

  if (!('bindTools' in model) || typeof model.bindTools !== 'function') {
    throw new TypeError(
      `LLM provider does not support tool binding: ${provider}`
    );
  }

  const family = getProviderFamily(provider);
  const canDisableTools =
    family === 'openai' ||
    family === 'google' ||
    provider === Providers.MOONSHOT;
  if (disableToolCalling !== true || !canDisableTools)
    return model.bindTools(tools);
  return model.bindTools(tools, { tool_choice: 'none' });
}

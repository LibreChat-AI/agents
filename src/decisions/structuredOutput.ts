import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

export type DecisionOutputMethod = 'jsonSchema' | 'functionCalling';

/** Binds only verified enforcement paths; a model profile alone is not a strict-output guarantee. */
export function withDecisionStructuredOutput(
  model: BaseChatModel,
  schema: Record<string, unknown>,
  method: DecisionOutputMethod
): ReturnType<BaseChatModel['withStructuredOutput']> {
  const options = {
    name: 'DecideQuestions',
    method,
    includeRaw: true as const,
  };
  const adapter = model._llmType();
  if (
    adapter === 'openai' ||
    adapter === 'azure_openai' ||
    (adapter === 'anthropic' && method === 'functionCalling')
  ) {
    return model.withStructuredOutput(schema, { ...options, strict: true });
  }
  throw new Error('unverified strict decision model adapter');
}

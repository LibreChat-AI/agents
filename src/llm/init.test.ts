import { RunnableBinding } from '@langchain/core/runnables';
import type {ChatOpenAICallOptions} from '@langchain/openai';
import type * as t from '@/types';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { CustomAnthropic } from '@/llm/anthropic';
import { initializeModel } from '@/llm/init';
import { ChatOpenAI } from '@/llm/openai';
import { Providers } from '@/common';

const tools: t.GraphTools = [
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Calculate a result',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string' } },
      },
    },
  },
];

it('disables tool calls on a thinking summary while preserving the normal Anthropic tool prefix', () => {
  const model = new CustomAnthropic({
    apiKey: 'offline-test',
    model: 'claude-haiku-4-5',
    thinking: { type: 'enabled', budget_tokens: 1024 },
    maxTokens: 2048,
  });
  const normal = initializeModel({
    provider: Providers.ANTHROPIC,
    override: model,
    tools,
  });
  const summary = initializeModel({
    provider: Providers.ANTHROPIC,
    override: model,
    tools,
    disableToolCalling: true,
  });
  if (
    !(normal instanceof RunnableBinding) ||
    !(summary instanceof RunnableBinding)
  )
    throw new Error('Expected tool bindings');
  const normalParams = model.invocationParams(normal.config);
  const summaryParams = model.invocationParams(summary.config);
  expect(summaryParams.tool_choice).toEqual({ type: 'none' });
  expect(normalParams.tool_choice).toBeUndefined();
  expect(summaryParams.tools).toEqual(normalParams.tools);
  expect(summaryParams.thinking).toEqual(normalParams.thinking);
  expect(summaryParams.max_tokens).toBe(2048);
});

it('preserves OpenAI tool definitions when disabling summary tool calls', () => {
  const model = new ChatOpenAI({
    apiKey: 'offline-test',
    model: 'gpt-4.1-mini',
  });
  const normal = initializeModel({
    provider: Providers.OPENAI,
    override: model,
    tools,
  });
  const summary = initializeModel({
    provider: Providers.OPENAI,
    override: model,
    tools,
    disableToolCalling: true,
  });
  const summaryOptions = Reflect.get(summary, 'defaultOptions') as ChatOpenAICallOptions;
  const normalOptions = Reflect.get(normal, 'defaultOptions') as ChatOpenAICallOptions;
  expect(summaryOptions).toMatchObject({ tool_choice: 'none' });
  expect(summaryOptions.tools).toEqual(normalOptions.tools);
  expect(normalOptions).not.toHaveProperty('tool_choice');
});

it('does not send an unsupported none choice to Bedrock Converse', () => {
  const model = new CustomChatBedrockConverse({
    region: 'us-east-1',
    model: 'anthropic.claude-haiku-4-5-20251001-v1:0',
    credentials: {
      accessKeyId: 'offline-test',
      secretAccessKey: 'offline-test',
    },
  });
  const summary = initializeModel({
    provider: Providers.BEDROCK,
    override: model,
    tools,
    disableToolCalling: true,
  });
  if (!(summary instanceof RunnableBinding))
    throw new Error('Expected a tool binding');
  expect(summary.config).not.toHaveProperty('tool_choice');
  expect(model.invocationParams(summary.config).toolConfig?.tools).toHaveLength(1);
});

it('leaves a tool-less model untouched', () => {
  const model = new CustomAnthropic({
    apiKey: 'offline-test',
    model: 'claude-haiku-4-5',
  });
  expect(
    initializeModel({
      provider: Providers.ANTHROPIC,
      override: model,
      disableToolCalling: true,
    })
  ).toBe(model);
});

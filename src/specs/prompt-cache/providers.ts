import type * as t from '@/types';
import type { CapturedRequest, Json } from './harness';
import { CAPTURED } from './harness';
import { Providers } from '@/common';

type JsonObject = { [key: string]: Json };
type Restore = () => void;

/**
 * One provider configuration the harness drives. `intercept` installs a
 * capture point that records the serialized request and aborts it before it
 * leaves the process; `reduce` maps that request to cache order.
 */
export interface ProviderCase {
  name: string;
  provider: Providers;
  clientOptions: () => t.ClientOptions;
  intercept: (agent: t.AgentInputs, record: (body: Json) => void) => Restore;
  reduce: (body: Json) => CapturedRequest;
  /** Explicit breakpoint family and the provider's limit per request. */
  breakpoints?: { pattern: RegExp; max: number };
  /** Prompt caching moves `additional_instructions` ahead of the newest human message. */
  movesDynamicPart?: boolean;
  /** Provider rejects consecutive messages of the same role. */
  strictAlternation?: boolean;
}

const API_KEY = 'test-key';
const ANTHROPIC_BREAKPOINTS = { pattern: /"cache_control":/g, max: 4 };
const BEDROCK_BREAKPOINTS = { pattern: /"cachePoint":/g, max: 4 };

function asObject(value: Json | undefined): JsonObject {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value
    : {};
}

function asArray(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}

function interceptFetch(record: (body: Json) => void): Restore {
  const spy = jest
    .spyOn(global, 'fetch')
    .mockImplementation(async (input, init) => {
      const raw =
        input instanceof Request
          ? await input.clone().text()
          : String(init?.body ?? '{}');
      record(JSON.parse(raw) as Json);
      throw new Error(CAPTURED);
    });
  return () => spy.mockRestore();
}

function reduceChatCompletions(body: Json): CapturedRequest {
  const request = asObject(body);
  const messages = asArray(request.messages);
  const isSystem = (message: Json): boolean => {
    const role = asObject(message).role;
    return role === 'system' || role === 'developer';
  };
  return {
    tools: asArray(request.tools),
    system: messages.filter(isSystem),
    messages: messages.filter((message) => !isSystem(message)),
    raw: body,
  };
}

function reduceResponses(body: Json): CapturedRequest {
  const request = asObject(body);
  const input = asArray(request.input);
  const isSystem = (item: Json): boolean => {
    const role = asObject(item).role;
    return role === 'system' || role === 'developer';
  };
  return {
    tools: asArray(request.tools),
    system: [request.instructions ?? null, ...input.filter(isSystem)],
    messages: input.filter((item) => !isSystem(item)),
    raw: body,
  };
}

function reduceAnthropic(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(request.tools),
    system: request.system ?? null,
    messages: asArray(request.messages),
    raw: body,
  };
}

function reduceGoogle(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(request.tools).flatMap((entry) => {
      const declarations = asObject(entry).functionDeclarations;
      return Array.isArray(declarations) ? declarations : [entry];
    }),
    system: request.systemInstruction ?? null,
    messages: asArray(request.contents),
    raw: body,
  };
}

function reduceBedrock(body: Json): CapturedRequest {
  const request = asObject(body);
  return {
    tools: asArray(asObject(request.toolConfig).tools),
    system: request.system ?? null,
    messages: asArray(request.messages),
    raw: body,
  };
}

function fetchCase(fields: Omit<ProviderCase, 'intercept'>): ProviderCase {
  return { ...fields, intercept: (_agent, record) => interceptFetch(record) };
}

function bedrockCase(name: string, promptCache: boolean): ProviderCase {
  return {
    name,
    provider: Providers.BEDROCK,
    clientOptions: () =>
      ({
        model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
        region: 'us-east-1',
        credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
        maxRetries: 0,
        promptCache,
      }) as t.ClientOptions,
    intercept: (agent, record) => {
      const options = agent.clientOptions as { client?: object };
      options.client = {
        send: async (command: { input: object }): Promise<never> => {
          record(JSON.parse(JSON.stringify(command.input)) as Json);
          throw new Error(CAPTURED);
        },
        config: {},
        destroy: (): void => undefined,
        middlewareStack: {
          add: (): void => undefined,
          use: (): void => undefined,
        },
      };
      return () => undefined;
    },
    reduce: reduceBedrock,
    breakpoints: promptCache ? BEDROCK_BREAKPOINTS : undefined,
    movesDynamicPart: promptCache,
    strictAlternation: true,
  };
}

export const PROVIDER_CASES: ProviderCase[] = [
  fetchCase({
    name: 'openAI',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({ model: 'gpt-4.1', apiKey: API_KEY, maxRetries: 0 }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'openAI responses',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({
        model: 'gpt-5',
        apiKey: API_KEY,
        maxRetries: 0,
        useResponsesApi: true,
      }) as t.ClientOptions,
    reduce: reduceResponses,
  }),
  fetchCase({
    name: 'openAI-compatible endpoint',
    provider: Providers.OPENAI,
    clientOptions: () =>
      ({
        model: 'local-model',
        apiKey: API_KEY,
        maxRetries: 0,
        configuration: { baseURL: 'https://openai-compatible.invalid/v1' },
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'azureOpenAI',
    provider: Providers.AZURE,
    clientOptions: () =>
      ({
        model: 'gpt-4.1',
        azureOpenAIApiKey: API_KEY,
        azureOpenAIApiInstanceName: 'harness',
        azureOpenAIApiDeploymentName: 'gpt-4-1',
        azureOpenAIApiVersion: '2024-10-21',
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'deepseek',
    provider: Providers.DEEPSEEK,
    clientOptions: () =>
      ({
        model: 'deepseek-chat',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'xai',
    provider: Providers.XAI,
    clientOptions: () =>
      ({ model: 'grok-4', apiKey: API_KEY, maxRetries: 0 }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'moonshot',
    provider: Providers.MOONSHOT,
    clientOptions: () =>
      ({
        model: 'kimi-k2.5',
        apiKey: API_KEY,
        maxRetries: 0,
        configuration: {
          apiKey: API_KEY,
          baseURL: 'https://api.moonshot.ai/v1',
        },
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'anthropic',
    provider: Providers.ANTHROPIC,
    clientOptions: () =>
      ({
        model: 'claude-sonnet-4-5',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceAnthropic,
  }),
  fetchCase({
    name: 'anthropic promptCache',
    provider: Providers.ANTHROPIC,
    clientOptions: () =>
      ({
        model: 'claude-sonnet-4-5',
        apiKey: API_KEY,
        maxRetries: 0,
        promptCache: true,
      }) as t.ClientOptions,
    reduce: reduceAnthropic,
    breakpoints: ANTHROPIC_BREAKPOINTS,
    movesDynamicPart: true,
  }),
  fetchCase({
    name: 'openrouter',
    provider: Providers.OPENROUTER,
    clientOptions: () =>
      ({
        model: 'anthropic/claude-sonnet-4.5',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
  }),
  fetchCase({
    name: 'openrouter promptCache',
    provider: Providers.OPENROUTER,
    clientOptions: () =>
      ({
        model: 'anthropic/claude-sonnet-4.5',
        apiKey: API_KEY,
        maxRetries: 0,
        promptCache: true,
      }) as t.ClientOptions,
    reduce: reduceChatCompletions,
    breakpoints: ANTHROPIC_BREAKPOINTS,
    movesDynamicPart: true,
  }),
  bedrockCase('bedrock', false),
  bedrockCase('bedrock promptCache', true),
  fetchCase({
    name: 'google',
    provider: Providers.GOOGLE,
    clientOptions: () =>
      ({
        model: 'gemini-2.5-flash',
        apiKey: API_KEY,
        maxRetries: 0,
      }) as t.ClientOptions,
    reduce: reduceGoogle,
  }),
];

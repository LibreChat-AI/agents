import { AIMessageChunk } from '@langchain/core/messages';
import { expect, test, describe, beforeEach, afterAll } from '@jest/globals';
import type { BaseMessageChunk } from '@langchain/core/messages';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { ChatOpenAI, AzureChatOpenAI } from './index';

type ExternalExecution = { mode: 'external'; provider?: string };

type ToolCallRecord = {
  index?: number;
  execution?: ExternalExecution;
};

type ToolCallMessage = AIMessageChunk & {
  tool_calls?: ToolCallRecord[];
  tool_call_chunks?: ToolCallRecord[];
};

type DeltaConverter = {
  _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: Record<string, unknown>
  ): BaseMessageChunk;
};

const rawResponse = {
  id: 'chatcmpl-1',
  object: 'chat.completion.chunk',
  created: 1,
  model: 'gpt-5.5',
  choices: [],
};

const toolCallDelta = {
  role: 'assistant',
  tool_calls: [
    {
      index: 0,
      id: 'call_1',
      type: 'function',
      function: { name: 'weather', arguments: '{"ci' },
    },
  ],
};

function convertDelta(
  model: unknown,
  delta: Record<string, unknown>
): AIMessageChunk {
  const converter = (model as { completions: DeltaConverter }).completions;
  const message = converter._convertCompletionsDeltaToBaseMessageChunk(
    delta,
    rawResponse
  );
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

function adapterOf(message: AIMessageChunk): unknown {
  return (message.response_metadata as Record<string, unknown>)[
    STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY
  ];
}

function executionAt(message: AIMessageChunk, index: number): ExternalExecution | undefined {
  return (message as ToolCallMessage).tool_calls?.[index]?.execution;
}

describe('Chat Completions sequential tool-call seal stamping', () => {
  // Both the implementation (OPENAI_BASE_URL fallback) and the Azure
  // constructor (AZURE_OPENAI_BASE_PATH fallback) read the environment, so
  // isolate these vars to keep the suite deterministic across shells.
  const ISOLATED_ENV_VARS = ['OPENAI_BASE_URL', 'AZURE_OPENAI_BASE_PATH'];
  const originalEnv = new Map(
    ISOLATED_ENV_VARS.map((name) => [name, process.env[name]])
  );

  beforeEach(() => {
    for (const name of ISOLATED_ENV_VARS) {
      delete process.env[name];
    }
  });

  afterAll(() => {
    for (const [name, value] of originalEnv) {
      if (value == null) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  test('stamps tool-call deltas when no baseURL is configured (official)', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('stamps tool-call deltas for an explicit api.openai.com baseURL', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5.5',
      apiKey: 'test',
      configuration: { baseURL: 'https://api.openai.com/v1' },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test.each([
    ['the default port written explicitly', 'https://api.openai.com:443/v1'],
    ['a mixed-case host', 'https://API.OpenAI.com/v1'],
  ])('stamps for a first-party URL spelled with %s', (_label, baseURL) => {
    const model = new ChatOpenAI({
      model: 'gpt-5.5',
      apiKey: 'test',
      configuration: { baseURL },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test.each([
    ['a non-default port', 'https://api.openai.com:8443/v1'],
    ['a lookalike host', 'https://api.openai.com.example.net/v1'],
    ['a plaintext scheme', 'http://api.openai.com/v1'],
    ['an unparseable value', 'not a url'],
  ])('does not stamp for %s', (_label, baseURL) => {
    const model = new ChatOpenAI({
      model: 'gpt-5.5',
      apiKey: 'test',
      configuration: { baseURL },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('does not stamp tool-call deltas for OpenAI-compatible endpoints', () => {
    const model = new ChatOpenAI({
      model: 'kimi-k2',
      apiKey: 'test',
      configuration: { baseURL: 'https://api.moonshot.ai/v1' },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('propagates approved external execution ownership into parsed tool calls', () => {
    const model = new ChatOpenAI({
      model: 'custom-model',
      apiKey: 'test',
      configuration: { baseURL: 'https://example.invalid/v1' },
      externalToolExecution: { enabled: true, acceptedProviders: ['remote-runtime'] },
    });
    const message = convertDelta(model, {
      ...toolCallDelta,
      tool_calls: [
        {
          ...toolCallDelta.tool_calls[0],
          execution: { mode: 'external', provider: 'remote-runtime' },
        },
      ],
    });

    expect(executionAt(message, 0)).toEqual({ mode: 'external', provider: 'remote-runtime' });
  });

  test('ignores external execution ownership from an unapproved provider', () => {
    const model = new ChatOpenAI({
      model: 'custom-model',
      apiKey: 'test',
      configuration: { baseURL: 'https://example.invalid/v1' },
      externalToolExecution: { enabled: true, acceptedProviders: ['remote-runtime'] },
    });
    const message = convertDelta(model, {
      ...toolCallDelta,
      tool_calls: [
        {
          ...toolCallDelta.tool_calls[0],
          execution: { mode: 'external', provider: 'unapproved-runtime' },
        },
      ],
    });

    expect(executionAt(message, 0)).toBeUndefined();
  });

  test('does not stamp text-only deltas', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, {
      role: 'assistant',
      content: 'hello',
    });
    expect(adapterOf(message)).toBeUndefined();
  });

  test('does not stamp when OPENAI_BASE_URL routes to a compatible endpoint', () => {
    process.env.OPENAI_BASE_URL = 'https://api.moonshot.ai/v1';
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('stamps when OPENAI_BASE_URL points at api.openai.com', () => {
    process.env.OPENAI_BASE_URL = 'https://api.openai.com/v1';
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('stamps Azure OpenAI tool-call deltas (first-party endpoint)', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiInstanceName: 'test-instance',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('stamps Azure deltas for an *.openai.azure.com base path', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      azureOpenAIBasePath:
        'https://test-resource.openai.azure.com/openai/deployments',
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('stamps Azure deltas for a regional cognitive services base path', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      azureOpenAIBasePath:
        'https://westeurope.api.cognitive.microsoft.com/openai/deployments',
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test.each([
    [
      'a mixed-case Azure host',
      'https://RESOURCE.OPENAI.AZURE.COM/openai/deployments',
    ],
    [
      'a mixed-case cognitive services host',
      'https://Resource.CognitiveServices.Azure.com/openai/deployments',
    ],
  ])('stamps Azure deltas for %s', (_label, azureOpenAIBasePath) => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      azureOpenAIBasePath,
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test.each([
    ['a plaintext scheme', 'http://resource.openai.azure.com/openai'],
    ['an unparseable value', 'not a url'],
  ])('does not stamp Azure deltas for %s', (_label, azureOpenAIBasePath) => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      azureOpenAIBasePath,
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('does not stamp Azure deltas routed through a proxy base path', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      azureOpenAIBasePath: 'https://proxy.example.com/openai/deployments',
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('does not stamp Azure deltas with a custom client baseURL', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiInstanceName: 'test-instance',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
      configuration: { baseURL: 'https://gateway.example.com/azure' },
    } as unknown as ConstructorParameters<typeof AzureChatOpenAI>[0]);
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });
});

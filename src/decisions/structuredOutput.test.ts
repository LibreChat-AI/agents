import { ChatAnthropic } from '@langchain/anthropic';
import { ChatOpenAI, AzureChatOpenAI } from '@langchain/openai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { StructuredChatDecisionModelOptions } from './index';
import {
  ChatOpenAI as LibreChatOpenAI,
  AzureChatOpenAI as LibreChatAzure,
} from '@/llm/openai';
import {
  createStructuredChatDecisionModel,
  booleanQuestion,
  choiceQuestion,
} from './index';
import { CustomAnthropic } from '@/llm/anthropic';

type Method = StructuredChatDecisionModelOptions['method'];
interface ProviderCall {
  tools?: object[];
  tool_choice?: object;
  output_config?: { format?: { type?: string; schema?: object } };
  response_format?: { json_schema?: { strict?: boolean; schema?: object } };
}

const modelId = 'claude-haiku-4-5-20251001';
const answers = { q: { decision: true }, role: { choice: 'hook' } };
const request = {
  state: 'Evidence to classify',
  questions: {
    q: booleanQuestion('Is this a test?'),
    role: choiceQuestion('Which role?', { hook: 'A hook', util: 'A utility' }),
  },
};
const anthropicAdapters = [
  {
    name: 'upstream Anthropic',
    create: (fetch: typeof globalThis.fetch): BaseChatModel =>
      new ChatAnthropic({
        model: modelId,
        apiKey: 'test',
        maxRetries: 0,
        clientOptions: { fetch },
      }),
  },
  {
    name: 'LibreChat Anthropic',
    create: (fetch: typeof globalThis.fetch): BaseChatModel =>
      new CustomAnthropic({
        model: modelId,
        apiKey: 'test',
        maxRetries: 0,
        clientOptions: { fetch },
      }),
  },
];

function anthropicFetch(
  parsed: object,
  method: Method,
  calls: ProviderCall[]
): typeof globalThis.fetch {
  const toolUse = {
    type: 'tool_use',
    id: 'toolu_test',
    name: 'DecideQuestions',
    input: parsed,
  };
  const content =
    method === 'jsonSchema'
      ? [{ type: 'text', text: JSON.stringify(parsed) }]
      : [toolUse];
  return async (_url, init) => {
    calls.push(JSON.parse(String(init?.body)) as ProviderCall);
    return new Response(
      JSON.stringify({
        id: 'msg_test',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content,
        stop_reason: method === 'jsonSchema' ? 'end_turn' : 'tool_use',
        stop_sequence: null,
        usage: { input_tokens: 8, output_tokens: 2 },
      }),
      { headers: { 'content-type': 'application/json' } }
    );
  };
}

afterEach(() => jest.restoreAllMocks());

describe.each(anthropicAdapters)(
  '$name decision output capabilities',
  ({ create }) => {
    it('retains strict tools and the allowed-choice enum through the real adapter', async () => {
      const calls: ProviderCall[] = [];
      const model = create(
        anthropicFetch({ answers }, 'functionCalling', calls)
      );
      const binding = jest.spyOn(model, 'withStructuredOutput');
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId,
        method: 'functionCalling',
      });
      const result = await decisionModel.decide(request);
      expect(calls).toHaveLength(1);
      expect(binding).toHaveBeenCalledTimes(1);
      expect(binding.mock.calls[0][1]).toMatchObject({
        method: 'functionCalling',
        strict: true,
        includeRaw: true,
        name: 'DecideQuestions',
      });
      expect(calls[0]).toHaveProperty('tools.0.strict', true);
      expect(calls[0]).toHaveProperty('tool_choice', {
        type: 'tool',
        name: 'DecideQuestions',
      });
      expect(calls[0]).toHaveProperty(
        'tools.0.input_schema.properties.answers.properties.q.properties.decision.type',
        'boolean'
      );
      expect(calls[0]).toHaveProperty(
        'tools.0.input_schema.properties.answers.properties.role.properties.choice.enum',
        ['hook', 'util']
      );
      expect(result.answers.q).toEqual({
        type: 'boolean',
        decision: true,
        probability: null,
      });
      expect(result.answers.role).toEqual({
        type: 'choice',
        choice: 'hook',
        confidence: null,
        probabilities: null,
      });
      expect(result.usage).toEqual({ inputTokens: 8, outputTokens: 2 });
    });

    it('rejects native schema before binding instead of silently weakening or switching modes', async () => {
      const calls: ProviderCall[] = [];
      const model = create(anthropicFetch({ answers }, 'jsonSchema', calls));
      const binding = jest.spyOn(model, 'withStructuredOutput');
      const onAnswered = jest.fn();
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId,
        method: 'jsonSchema',
        onAnswered,
      });
      await expect(decisionModel.decide(request)).rejects.toMatchObject({
        failure: 'unsupported_mode',
      });
      expect(binding).not.toHaveBeenCalled();
      expect(calls).toHaveLength(0);
      expect(onAnswered).not.toHaveBeenCalled();
    });

    it('pins the locked native transform that moves choice enums into descriptive text', async () => {
      const calls: ProviderCall[] = [];
      const model = create(
        anthropicFetch({ answer: 'hook' }, 'jsonSchema', calls)
      );
      const structured = model.withStructuredOutput(
        {
          type: 'object',
          properties: { answer: { type: 'string', enum: ['hook', 'util'] } },
          required: ['answer'],
          additionalProperties: false,
        },
        { method: 'jsonSchema', includeRaw: true }
      );
      await structured.invoke('Pick a role');
      expect(calls).toHaveLength(1);
      expect(calls[0]).toHaveProperty(
        'output_config.format.type',
        'json_schema'
      );
      expect(calls[0]).not.toHaveProperty(
        'output_config.format.schema.properties.answer.enum'
      );
      expect(calls[0]).toHaveProperty(
        'output_config.format.schema.properties.answer.description',
        '{enum: ["hook","util"]}'
      );
    });
  }
);

const openAIFields = { model: 'gpt-4o-mini', apiKey: 'test', maxRetries: 0 };
const azureFields = {
  model: 'gpt-4o-mini',
  maxRetries: 0,
  azureOpenAIApiKey: 'test',
  azureOpenAIApiVersion: '2024-10-21',
  azureOpenAIApiInstanceName: 'test-instance',
  azureOpenAIApiDeploymentName: 'test-deployment',
};
const openAIAdapters = [
  {
    name: 'upstream OpenAI',
    create: (fetch: typeof globalThis.fetch): BaseChatModel =>
      new ChatOpenAI({ ...openAIFields, configuration: { fetch } }),
  },
  {
    name: 'LibreChat OpenAI',
    create: (fetch: typeof globalThis.fetch): BaseChatModel =>
      new LibreChatOpenAI({ ...openAIFields, configuration: { fetch } }),
  },
  {
    name: 'upstream Azure',
    create: (fetch: typeof globalThis.fetch): BaseChatModel =>
      new AzureChatOpenAI({ ...azureFields, configuration: { fetch } }),
  },
  {
    name: 'LibreChat Azure',
    create: (fetch: typeof globalThis.fetch): BaseChatModel => {
      jest.spyOn(globalThis, 'fetch').mockImplementation(fetch);
      return new LibreChatAzure(azureFields);
    },
  },
];

describe.each(openAIAdapters)(
  '$name decision output capabilities',
  ({ create }) => {
    it.each(['jsonSchema', 'functionCalling'] as const)(
      'retains strict enforcement in %s',
      async (method) => {
        const calls: ProviderCall[] = [];
        const toolCall = {
          id: 'call_test',
          type: 'function',
          function: {
            name: 'DecideQuestions',
            arguments: JSON.stringify({ answers }),
          },
        };
        const fetch: typeof globalThis.fetch = async (_url, init) => {
          calls.push(JSON.parse(String(init?.body)) as ProviderCall);
          return new Response(
            JSON.stringify({
              id: 'chatcmpl_test',
              object: 'chat.completion',
              created: 0,
              model: 'gpt-4o-mini',
              choices: [
                {
                  index: 0,
                  finish_reason:
                    method === 'jsonSchema' ? 'stop' : 'tool_calls',
                  logprobs: null,
                  message: {
                    role: 'assistant',
                    content:
                      method === 'jsonSchema'
                        ? JSON.stringify({ answers })
                        : null,
                    tool_calls:
                      method === 'functionCalling' ? [toolCall] : undefined,
                  },
                },
              ],
              usage: {
                prompt_tokens: 12,
                completion_tokens: 4,
                total_tokens: 16,
              },
            }),
            { headers: { 'content-type': 'application/json' } }
          );
        };
        const model = create(fetch);
        const binding = jest.spyOn(model, 'withStructuredOutput');
        const decisionModel = createStructuredChatDecisionModel({
          model,
          modelId: 'gpt-4o-mini',
          method,
        });
        const result = await decisionModel.decide(request);
        expect(calls).toHaveLength(1);
        expect(binding.mock.calls[0][1]).toMatchObject({
          method,
          strict: true,
          includeRaw: true,
        });
        const strictPath =
          method === 'jsonSchema'
            ? 'response_format.json_schema.strict'
            : 'tools.0.function.strict';
        const schemaPath =
          method === 'jsonSchema'
            ? 'response_format.json_schema.schema'
            : 'tools.0.function.parameters';
        expect(calls[0]).toHaveProperty(strictPath, true);
        expect(calls[0]).toHaveProperty(
          `${schemaPath}.properties.answers.properties.role.properties.choice.enum`,
          ['hook', 'util']
        );
        expect(result.answers.q).toEqual({
          type: 'boolean',
          decision: true,
          probability: null,
        });
      }
    );

    it('returns a sanitized native-schema rejection without switching to tool calling', async () => {
      const calls: ProviderCall[] = [];
      const fetch: typeof globalThis.fetch = async (_url, init) => {
        calls.push(JSON.parse(String(init?.body)) as ProviderCall);
        return new Response(
          JSON.stringify({ error: { message: 'private endpoint response' } }),
          { status: 400 }
        );
      };
      const model = create(fetch);
      const binding = jest.spyOn(model, 'withStructuredOutput');
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      try {
        await decisionModel.decide(request);
        throw new Error('expected typed endpoint rejection');
      } catch (error) {
        expect(error).toMatchObject({ failure: 'bad_request', status: 400 });
        expect(String(error)).not.toContain('private');
      }
      expect(binding).toHaveBeenCalledTimes(1);
      expect(calls).toHaveLength(1);
      expect(calls[0]).toHaveProperty(
        'response_format.json_schema.strict',
        true
      );
      expect(calls[0]).not.toHaveProperty('tools');
    });

    it('sanitizes binding failures without invoking the provider', async () => {
      const fetch = jest.fn(async (): Promise<Response> => {
        throw new Error('unexpected network request');
      });
      const model = create(fetch);
      jest.spyOn(model, 'withStructuredOutput').mockImplementationOnce(() => {
        throw new Error('private adapter setup');
      });
      const decisionModel = createStructuredChatDecisionModel({
        model,
        modelId: 'gpt-4o-mini',
        method: 'jsonSchema',
      });
      await expect(decisionModel.decide(request)).rejects.toMatchObject({
        failure: 'unsupported_mode',
      });
      expect(fetch).not.toHaveBeenCalled();
    });
  }
);

it('does not turn broad profile metadata or a provider label into strict adapter support', async () => {
  const model = new ChatGoogleGenerativeAI({
    model: 'gemini-2.5-flash',
    apiKey: 'test',
  });
  jest
    .spyOn(model, 'profile', 'get')
    .mockReturnValue({ structuredOutput: true });
  const structured = jest.spyOn(model, 'withStructuredOutput');
  const generate = jest.spyOn(model, '_generate');
  const decisionModel = createStructuredChatDecisionModel({
    model,
    modelId: 'gpt-4o-mini',
    providerId: 'openai',
    method: 'jsonSchema',
  });
  await expect(decisionModel.decide(request)).rejects.toMatchObject({
    failure: 'unsupported_mode',
  });
  expect(structured).not.toHaveBeenCalled();
  expect(generate).not.toHaveBeenCalled();
});
